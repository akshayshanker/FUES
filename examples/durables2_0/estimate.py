"""SMM estimation entry point for durables2_0.

Constructs the trial function from the solver, loads the estimation spec
from the mod directory, and runs the cross-entropy loop.

Usage:
    # Serial
    python3 -m examples.durables2_0.estimate --mod mod/separable --spec baseline.yaml

    # MPI
    mpirun -np 48 python3 -u -m mpi4py -m examples.durables2_0.estimate \
        --mod mod/separable --spec baseline.yaml

    # With overrides
    python3 -m examples.durables2_0.estimate \
        --mod mod/cobb_douglas \
        --spec baseline.yaml \
        --scratch /scratch/tp66/$USER/est \
        --results results/durables2_0 \
        --setting-override n_a=300 n_h=300 n_w=300
"""

import argparse
import gc
import os
from pathlib import Path

import yaml

from kikku.run.estimate import (
    load_estimation_spec, make_criterion, estimate, diagnostics,
)
from kikku.run.moments import make_moment_fn, moment_names as get_moment_names
from kikku.run.mpi import get_comm, is_root

from .solve import solve
from .horses.simulate import simulate_lifecycle


def _parse_key_value_list(items):
    """Parse ['n_a=300', 'n_h=300'] into {'n_a': 300, 'n_h': 300}."""
    if not items:
        return {}
    out = {}
    for item in items:
        k, v = item.split('=', 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        out[k.strip()] = v
    return out


def main():
    parser = argparse.ArgumentParser(
        description="SMM estimation for durables2_0")

    parser.add_argument(
        '--mod', type=str, required=True,
        help='Path to mod directory (e.g. mod/separable, mod/cobb_douglas)')
    parser.add_argument(
        '--spec', type=str, default='baseline.yaml',
        help='Estimation spec YAML, relative to mod/estimation/ (default: baseline.yaml)')
    parser.add_argument(
        '--scratch', type=str, default=None,
        help='Scratch dir for intermediate outputs (overrides YAML)')
    parser.add_argument(
        '--results', type=str, default=None,
        help='Results dir for final outputs (overrides YAML)')
    parser.add_argument(
        '--setting-override', nargs='*', default=[],
        help='Grid/solver overrides (e.g. n_a=300 n_h=300)')
    parser.add_argument(
        '--calib-override', nargs='*', default=[],
        help='Calibration overrides (e.g. t0=20)')
    parser.add_argument(
        '--N-sim', type=int, default=10000,
        help='Number of simulation agents (default: 10000)')
    parser.add_argument(
        '--method', type=str, default=None,
        help='Solver method override for adjuster (e.g. NEGM). Default: YAML methods.')
    parser.add_argument(
        '--local-results', type=str, default=None,
        help='Local results dir (default: mod/<syntax>/estimation/results/). '
             'Set to "none" to disable local copy.')

    args = parser.parse_args()
    comm = get_comm()

    # --- Resolve paths ---
    # mod_dir is relative to the example root
    example_root = Path(__file__).parent
    mod_dir = (example_root / args.mod).resolve()
    spec_path = mod_dir / 'estimation' / args.spec

    if not mod_dir.exists():
        raise FileNotFoundError(f"Mod directory not found: {mod_dir}")
    if not spec_path.exists():
        raise FileNotFoundError(f"Estimation spec not found: {spec_path}")

    # --- Load estimation spec ---
    spec = load_estimation_spec(str(spec_path))
    moment_spec = spec['moment_spec']
    param_spec = spec['free']
    method_options = spec['method_options']

    # Load scratch/results dirs from the raw YAML (not in spec dict)
    with spec_path.open() as f:
        raw_yaml = yaml.safe_load(f)
    est_yaml = raw_yaml.get('estimation', {})

    # CLI overrides for paths (CLI > YAML > defaults)
    scratch_dir = args.scratch or est_yaml.get('scratch_dir', 'scratch/est')
    results_dir = args.results or est_yaml.get('results_dir', 'results/est')

    # Expand {user}
    scratch_dir = scratch_dir.replace('{user}', os.environ.get('USER', 'unknown'))
    results_dir = results_dir.replace('{user}', os.environ.get('USER', 'unknown'))

    setting_overrides = _parse_key_value_list(args.setting_override)
    calib_overrides = _parse_key_value_list(args.calib_override)

    simulation_seed = int(method_options.get('simulation_seed', 99))
    N_sim = args.N_sim

    if is_root(comm):
        print(f"Estimation: {spec_path.name}")
        print(f"  Mod: {mod_dir}")
        print(f"  Free params: {list(param_spec.keys())}")
        print(f"  Method: {spec['method']}")
        print(f"  n_samples={method_options.get('n_samples')}, "
              f"n_elite={method_options.get('n_elite')}, "
              f"max_iter={method_options.get('max_iter')}")
        print(f"  N_sim={N_sim}, simulation_seed={simulation_seed}")
        print(f"  Scratch: {scratch_dir}")
        print(f"  Results: {results_dir}")
        if setting_overrides:
            print(f"  Setting overrides: {setting_overrides}")
        if calib_overrides:
            print(f"  Calib overrides: {calib_overrides}")

    # --- Build moment function ---
    moment_fn = make_moment_fn(moment_spec)

    # --- Build data moments ---
    data_source = moment_spec.get('data_source', 'precomputed')
    if data_source == 'selfgen':
        # Generate data on rank 0 only, then broadcast.
        # Use calib_overrides (e.g. t0=20) so data covers the same age range
        # as the trial function. Without this, selfgen uses YAML defaults
        # (t0=40) and produces NaN for age groups outside that range.
        if is_root(comm):
            print("  Data source: selfgen (solving at default calibration + overrides)...")
            nest_data, grids_data = solve(
                str(mod_dir),
                calib_overrides=calib_overrides,
                setting_overrides=setting_overrides,
                verbose=False,
            )
            data_panels = simulate_lifecycle(
                nest_data, grids_data, N=N_sim, seed=simulation_seed)
            data_moments = moment_fn(data_panels)
            del nest_data, grids_data, data_panels
        else:
            data_moments = None
        from kikku.run.mpi import bcast_item as _bcast
        data_moments = _bcast(data_moments, comm, root=0)
    else:
        data_moments = spec['data_moments']

    if is_root(comm):
        print(f"  Data moments: {len(data_moments)} keys")

    # --- Check for conflicts: calib overrides vs free params ---
    conflicts = set(calib_overrides.keys()) & set(param_spec.keys())
    if conflicts and is_root(comm):
        print(f"  WARNING: --calib-override sets {conflicts} which are free params. "
              f"CE draws will override these values at each evaluation.")

    # --- Build trial function ---
    # Precedence: theta (CE draw) > calib_overrides > YAML calibration defaults.
    # Free params always come from the CE, never from overrides.
    solver_method = args.method  # None = YAML default, 'NEGM' = override adjuster

    def _strip_nest_for_sim(nest):
        """Drop arrays the forward simulator doesn't need (V, marginals, arvl).

        build_period_pushforwards only reads:
          tenure/dcsn/adj, keeper_cons/dcsn/c,
          adjuster_cons/dcsn/{c, h_choice}, callables.
        Everything else is dead weight during estimation.
        """
        for sol in nest["solutions"]:
            # keeper: keep only c
            kd = sol["keeper_cons"]["dcsn"]
            for k in list(kd):
                if k != "c":
                    del kd[k]
            # adjuster: keep only c, h_choice
            ad = sol["adjuster_cons"]["dcsn"]
            for k in list(ad):
                if k not in ("c", "h_choice"):
                    del ad[k]
            # tenure dcsn: keep only adj
            td = sol["tenure"]["dcsn"]
            for k in list(td):
                if k != "adj":
                    del td[k]
            # tenure arvl: not needed for simulation
            sol["tenure"].pop("arvl", None)

    def trial(theta):
        merged_calib = {**calib_overrides, **theta}  # theta wins on overlap
        nest, grids = solve(
            str(mod_dir),
            method=solver_method,
            calib_overrides=merged_calib,
            setting_overrides=setting_overrides,
            verbose=False,
        )
        _strip_nest_for_sim(nest)
        gc.collect()  # reclaim stripped arrays before simulation allocates
        panels = simulate_lifecycle(nest, grids, N=N_sim, seed=simulation_seed)
        del nest, grids
        gc.collect()
        return panels

    # --- Compose criterion ---
    criterion = make_criterion(trial, moment_fn, data_moments)

    # --- Create output directories ---
    import json
    from datetime import datetime

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Organise by spec name: estimation/baseline/est_20260327_143000/
    spec_name = Path(args.spec).stem  # e.g. 'baseline', 'baseline_large_egm'
    scratch_run = os.path.join(scratch_dir, spec_name, f'est_{run_id}')
    results_run = os.path.join(results_dir, spec_name, f'est_{run_id}')

    if is_root(comm):
        os.makedirs(scratch_run, exist_ok=True)
        os.makedirs(results_run, exist_ok=True)
        # Save manifest
        manifest = {
            'mod': str(mod_dir),
            'spec': str(spec_path),
            'run_id': run_id,
            'n_samples': method_options.get('n_samples'),
            'n_elite': method_options.get('n_elite'),
            'max_iter': method_options.get('max_iter'),
            'grid': setting_overrides,
            'N_sim': N_sim,
            'simulation_seed': simulation_seed,
            'sampling_seed': method_options.get('sampling_seed'),
            'free_params': list(param_spec.keys()),
            'timestamp': run_id,
        }
        with open(os.path.join(scratch_run, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"  Scratch: {scratch_run}")
        print(f"  Results: {results_run}")

    # --- Pass checkpoint_dir so CE saves state per iteration ---
    method_options['checkpoint_dir'] = scratch_run

    # --- Estimate ---
    result = estimate(
        criterion, param_spec,
        method=spec['method'],
        method_options=method_options,
        comm=comm,
        verbose=is_root(comm),
    )

    # --- Save results ---
    if is_root(comm):
        diag = diagnostics(result, data_moments,
                           moment_names=get_moment_names(moment_spec))

        # Extract elite mean and SE from the last iteration's history
        theta_mean = result.theta  # fallback
        theta_se = {n: float('nan') for n in param_spec}
        if result.history:
            last = result.history[-1]
            if 'means' in last:
                theta_mean = last['means']
            if 'cov' in last:
                import numpy as _np
                cov = _np.asarray(last['cov'])
                names_sorted = sorted(param_spec.keys())
                se_vec = _np.sqrt(_np.diag(cov))
                theta_se = {names_sorted[i]: float(se_vec[i]) for i in range(len(names_sorted))}

        # theta_best.json
        with open(os.path.join(results_run, 'theta_best.json'), 'w') as f:
            json.dump(result.theta, f, indent=2)

        # theta_mean.json (elite mean at convergence)
        with open(os.path.join(results_run, 'theta_mean.json'), 'w') as f:
            json.dump(theta_mean, f, indent=2)

        # theta_se.json (SE from elite covariance)
        with open(os.path.join(results_run, 'theta_se.json'), 'w') as f:
            json.dump(theta_se, f, indent=2)

        # summary.json
        summary = {
            'theta_best': result.theta,
            'theta_mean': theta_mean,
            'theta_se': theta_se,
            'objective': result.objective,
            'converged': result.converged,
            'n_iter': result.n_iter,
        }
        with open(os.path.join(results_run, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        # fit_table.csv
        import csv
        with open(os.path.join(results_run, 'fit_table.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['moment', 'data', 'simulated', 'residual', 'contribution', 'contribution_pct'])
            writer.writeheader()
            for row in diag['fit_table']:
                writer.writerow(row)

        # convergence.csv
        with open(os.path.join(results_run, 'convergence.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['iter', 'best_loss', 'elite_mean_loss'])
            writer.writeheader()
            for i, h in enumerate(result.history):
                writer.writerow({
                    'iter': i,
                    'best_loss': h.get('best_loss'),
                    'elite_mean_loss': h.get('elite_mean_loss'),
                })

        # Print summary
        print(f"\n{'='*60}")
        print(f"Loss:       {result.objective:.6f}")
        print(f"Converged:  {result.converged} ({result.n_iter} iters)")
        print(f"\n{'param':12s} {'best':>10s} {'mean':>10s} {'SE':>10s}")
        print('-' * 44)
        for n in sorted(param_spec.keys()):
            print(f"{n:12s} {result.theta[n]:10.4f} {theta_mean[n]:10.4f} {theta_se[n]:10.6f}")
        print(f"\nFit table (at theta_best):")
        for row in diag['worst_moments']:
            print(f"  {row['moment']:40s} data={row['data']:10.4f} "
                  f"sim={row['simulated']:10.4f} contrib={row['contribution']:.4f}")
        # Local copy: --local-results (CLI) > default (mod/estimation/results/)
        # Set --local-results none to disable.
        if args.local_results and args.local_results.lower() == 'none':
            local_root = None
        elif args.local_results:
            local_root = Path(args.local_results)
        else:
            local_root = mod_dir / 'estimation' / 'results'

        if local_root is not None:
            local_results = local_root / spec_name / f'est_{run_id}'
            try:
                os.makedirs(local_results, exist_ok=True)
                import shutil
                for fname in ('theta_best.json', 'theta_mean.json', 'theta_se.json',
                              'summary.json', 'fit_table.csv', 'convergence.csv'):
                    src = os.path.join(results_run, fname)
                    if os.path.exists(src):
                        shutil.copy2(src, str(local_results / fname))
                print(f"\nLocal copy:  {local_results}")
            except OSError as e:
                print(f"\nWARNING: could not save local copy: {e}")

        print(f"Results saved to: {results_run}")
        print(f"Checkpoint: {scratch_run}/state.pkl")


if __name__ == '__main__':
    main()
