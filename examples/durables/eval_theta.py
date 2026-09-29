"""Evaluate the SMM objective at given parameter vectors, without the optimiser.

The cross-entropy estimator reports the objective only at the optimum it
found, so a change in a numerical setting (grid ceiling, grid size, income
points) changes two things at once: the objective surface and the basin the
sampler lands in. Evaluating fixed θ vectors under several settings separates
the two — at a fixed θ the objective differs only through the settings.

Each evaluation is one solve and one simulation, as in ``estimate.py``'s
``trial`` (same ``N_sim``, same ``simulation_seed`` from the spec), so a θ
evaluated under the settings of the run that produced it reproduces that
run's objective to every printed digit; that is the check that the plan is
set up correctly. Evaluations run in parallel processes.

Plan file (JSON): a list of evaluations, each
    {"label": "base theta, a_max 15", "settings": {"a_max": 15, "w_max": 20},
     "theta": {"alpha": 0.628, "beta": 0.99, "gamma_c": 1.253, "gamma_h": 1.21, "tau": 0.25}}
The evaluation's ``settings`` are laid over the command-line settings
overrides (grid sizes). Output: one JSON file with, per evaluation, the loss,
θ, settings and, for every moment, data, simulated value and weighted
contribution (contributions sum to the loss; the weights are kikku's
defaults, relative squared deviations for |data| >= 1 and absolute ones
otherwise), plus a CSV with the same rows.

Usage:
    python3 -m examples.durables.eval_theta --mod syntax/separable \
        --spec baseline_xlarge_egm.yaml --plan plan.json --out eval_theta \
        --settings-override n_a=600 n_h=600 n_w=600 --params-override t0=20 \
        --N-sim 10000 --jobs 8
    (add ``--methods-override adjuster_cons.upper_env.upper_envelope=NEGM`` for NEGM)
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from kikku.run.estimate import load_estimation_spec, make_criterion
from kikku.run.moments import make_moment_fn

from .estimate import _parse_key_value_list, _solver_method_from_cli
from .solve import solve
from .solvers.simulate import simulate_lifecycle

# Moment keys whose values are levels in model units and are rescaled to AUD
# before comparison with the data (same list as estimate.py).
_LEVEL_PREFIXES = (
    'mean_', 'sd_', 'av_', 'cond_discrete_0_mean_',
    'cond_discrete_1_mean_', 'cond_discrete_0_sd_',
    'cond_discrete_1_sd_',
)


def _target_data_moments(spec):
    """The data moments the model is asked to match (precomputed CSV, filtered
    to the spec's targets exactly as estimate.py does)."""
    moment_spec = spec['moment_spec']
    if moment_spec.get('data_source', 'precomputed') != 'precomputed':
        raise SystemExit("eval_theta handles precomputed data moments only")
    data_moments = spec['data_moments']
    targets = moment_spec.get('targets') or []
    ident = moment_spec.get('identification') or {}
    prefixes = {t['key'] for t in targets}
    for var in ident.get('mean', []) or []:
        prefixes.add(f'mean_{var}')
    for var in ident.get('sd', []) or ident.get('sds', []) or []:
        prefixes.add(f'sd_{var}')
    for pair in ident.get('corrs', []) or []:
        prefixes.add(f'corr_{pair[0]}_{pair[1]}')
    for var in ident.get('autocorrs', []) or []:
        prefixes.add(f'autocorr_{var}')
    return {k: v for k, v in data_moments.items()
            if any(k.startswith(p + '__') or k == p for p in prefixes)}


def _weights(data_moments):
    """kikku's default weights, keyed by moment, over the non-NaN data moments."""
    out = {}
    for k in sorted(data_moments):
        d = float(data_moments[k])
        if np.isnan(d):
            continue
        out[k] = 1.0 / max(d * d if abs(d) >= 1.0 else 1.0, 1e-20)
    return out


def evaluate_one(job):
    """Solve, simulate and score one (θ, settings) pair. Runs in a worker process."""
    mod_dir, spec_path, spec_factory, methods_override, calib_overrides, \
        base_settings, N_sim, entry = job
    settings = {**base_settings, **{k: (int(v) if isinstance(v, bool) else v)
                                    for k, v in (entry.get('settings') or {}).items()}}
    theta = {k: float(v) for k, v in entry['theta'].items()}
    spec = load_estimation_spec(str(spec_path))
    moment_spec = spec['moment_spec']
    simulation_seed = int(spec['method_options'].get('simulation_seed', 99))
    raw_moment_fn = make_moment_fn(moment_spec)
    from dolo.compiler.stage_factory import load_syntax
    _cal, _sett, *_ = load_syntax(mod_dir, calib_overrides, settings)
    denorm = 1.0 / float(_sett.get('normalisation', 1.0))

    def moment_fn(panels):
        raw = raw_moment_fn(panels)
        out = {}
        for k, v in raw.items():
            base = k.rsplit('__age', 1)[0] if '__age' in k else k
            out[k] = v * denorm if any(base.startswith(p) for p in _LEVEL_PREFIXES) else v
        return out

    class _Args:  # what _solver_method_from_cli reads
        pass
    _a = _Args()
    _a.methods_override = methods_override
    solver_method = _solver_method_from_cli(_a)

    def trial(th):
        nest, grids = solve(
            str(mod_dir), spec_factory_name=spec_factory, method_switch=solver_method,
            draw={"calibration": {**calib_overrides, **th}, "settings": settings},
            verbose=False, strip_solved=True,
        )
        return simulate_lifecycle(nest, grids, N=N_sim, seed=simulation_seed)

    data_moments = _target_data_moments(spec)
    criterion = make_criterion(trial, moment_fn, data_moments)
    t0 = time.time()
    loss = criterion(theta)
    sim = criterion.last_sim_moments or {}
    w = _weights(data_moments)
    rows = []
    for k, wk in w.items():
        d = float(data_moments[k])
        s = sim.get(k)
        contrib = (wk * (s - d) ** 2) if s is not None else None
        rows.append({"moment": k, "data": d, "simulated": s, "contribution": contrib})
    return {"label": entry.get('label', ''), "settings": settings, "theta": theta,
            "loss": loss, "seconds": round(time.time() - t0, 1), "moments": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mod', required=True, help='mod directory, e.g. syntax/separable')
    ap.add_argument('--spec', default='baseline.yaml', help='estimation spec, relative to mod/estimation/')
    ap.add_argument('--spec-factory', dest='spec_factory', default='spec_factory.yaml')
    ap.add_argument('--plan', required=True, help='JSON list of {label, settings, theta}')
    ap.add_argument('--out', required=True, help='output path without extension (.json and .csv are written)')
    ap.add_argument('--settings-override', nargs='*', default=[], dest='settings_override')
    ap.add_argument('--params-override', nargs='*', default=[], dest='params_override')
    ap.add_argument('--methods-override', nargs='*', default=[], dest='methods_override')
    ap.add_argument('--N-sim', type=int, default=10000, dest='N_sim')
    ap.add_argument('--jobs', type=int, default=1, help='evaluations run in parallel')
    args = ap.parse_args()

    example_root = Path(__file__).parent
    mod_dir = (example_root / args.mod).resolve()
    spec_path = mod_dir / 'estimation' / args.spec
    if not spec_path.exists():
        raise FileNotFoundError(spec_path)
    plan = json.loads(Path(args.plan).read_text())
    base_settings = _parse_key_value_list(args.settings_override)
    calib_overrides = _parse_key_value_list(args.params_override)
    jobs = [(mod_dir, spec_path, args.spec_factory, args.methods_override, calib_overrides,
             base_settings, args.N_sim, e) for e in plan]
    print(f"eval_theta: {len(jobs)} evaluations, {args.jobs} in parallel, spec {spec_path.name}, "
          f"settings {base_settings}, calibration {calib_overrides}", flush=True)
    results = []
    # Workers are spawned, not forked: the parent has already loaded MKL/OpenMP
    # through numpy and numba, and GNU OpenMP aborts a forked child
    # ("fork() called from a process already using GNU OpenMP", Gadi, 29 Sep 2026).
    import multiprocessing
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context('spawn')) as ex:
        for r in ex.map(evaluate_one, jobs):
            print(f"  {r['label']:<40s} loss {r['loss']:.6f}  ({r['seconds']} s)", flush=True)
            results.append(r)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(
        {"spec": str(spec_path), "settings_override": base_settings, "params_override": calib_overrides,
         "methods_override": args.methods_override, "N_sim": args.N_sim, "evaluations": results}, indent=2))
    with out.with_suffix('.csv').open('w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(["label", "loss", "moment", "data", "simulated", "contribution"])
        for r in results:
            for m in r["moments"]:
                wr.writerow([r["label"], r["loss"], m["moment"], m["data"], m["simulated"], m["contribution"]])
    print(f"written {out.with_suffix('.json')} and {out.with_suffix('.csv')}", flush=True)


if __name__ == '__main__':
    main()
