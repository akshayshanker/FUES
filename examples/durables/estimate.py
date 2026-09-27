"""SMM estimation entry point for durables.

Constructs the trial function from the solver, loads the estimation spec
from the mod directory, and runs the cross-entropy loop.

Usage:
    # Serial
    python3 -m examples.durables.estimate --mod syntax/separable --spec baseline.yaml

    # MPI
    mpirun -np 48 python3 -u -m mpi4py -m examples.durables.estimate \
        --mod syntax/separable --spec baseline.yaml

    # MPI with permanent types (spec block `estimation.types`, n members per
    # candidate): the rank count must be a multiple of n; each candidate is
    # evaluated by a group of n ranks, one member per rank (type_groups.py).
    mpirun -np 4160 python3 -u -m mpi4py -m examples.durables.estimate \
        --mod syntax/separable --spec baseline_large_egm_types.yaml

    # With overrides
    python3 -m examples.durables.estimate \
        --mod syntax/cobb_douglas \
        --spec baseline.yaml \
        --scratch /scratch/tp66/$USER/est \
        --results results/durables \
        --settings-override n_a=300 n_h=300 n_w=300
"""

import argparse
import csv
import gc
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime
from itertools import product as cart_product
from pathlib import Path

import numpy as np
import yaml

from dolo.compiler.spec_factory import load as load_spec, make as make_spec
from kikku.run.estimate import (
    BIG_LOSS, load_estimation_spec, make_criterion, estimate, diagnostics,
)
from kikku.run.moments import make_moment_fn, moment_names as get_moment_names
from kikku.run.mpi import get_comm, is_root, bcast_item

from .solve import solve
from .solvers.beta_types import draw_types, expand_types, pool_by_type
from .solvers.simulate import simulate_lifecycle, simulate_type_subset
from .type_groups import (
    STOP, check_divisibility, make_group_trial, split_type_groups, worker_loop,
)

# Moment keys whose values are levels in model units (normalised by
# `normalisation` in settings); they are multiplied by 1/normalisation so the
# loss compares AUD with AUD. Correlations and autocorrelations are left alone.
_LEVEL_PREFIXES = (
    'mean_', 'sd_', 'av_', 'cond_discrete_0_mean_',
    'cond_discrete_1_mean_', 'cond_discrete_0_sd_',
    'cond_discrete_1_sd_',
)


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


def _build_sweep_grid(sweep_spec):
    """Build list of dicts from sweep spec (cartesian product).

    Example: {'sigma_w': [0.1, 0.2], 'phi_w': [0.8, 0.9]}
    -> [{'sigma_w': 0.1, 'phi_w': 0.8}, {'sigma_w': 0.1, 'phi_w': 0.9},
        {'sigma_w': 0.2, 'phi_w': 0.8}, {'sigma_w': 0.2, 'phi_w': 0.9}]
    """
    keys = sorted(sweep_spec.keys())
    vals = [sweep_spec[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in cart_product(*vals)]


def _sweep_point_label(point):
    """e.g. {'sigma_w': 0.10} -> 'sigma_w=0.10'"""
    return '_'.join(f'{k}={v}' for k, v in sorted(point.items()))


def _solver_method_from_cli(args):
    """Return upper-envelope tag from ``--methods-override`` (kikku path=value), or None."""
    raw = getattr(args, "methods_override", None) or []
    for item in raw:
        s = str(item).strip()
        if "=" not in s:
            continue
        path, val = s.split("=", 1)
        if "upper_envelope" in path:
            return val.strip()
    return None


def _write_manifest(manifest, folders):
    """Write ``manifest.json`` into every folder of ``folders``."""
    for folder in folders:
        with open(os.path.join(folder, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)


def _git_commit(repo_root):
    """Commit hash of the checkout at ``repo_root`` (``git rev-parse HEAD``).

    Manifest metadata only: when git is absent, the directory is not a
    checkout, or the command fails, the manifest records ``None`` and the
    run is not affected.
    """
    try:
        proc = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=str(repo_root),
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _denormalised_moment_fn(mod_dir, moment_spec, calib_overrides, setting_overrides):
    """Moment function that reports levels in the data's units (AUD).

    The model simulates in normalised units (``normalisation`` in settings,
    1e-5 in both registries). Data moments, from the CSV or self-generated,
    are in natural units: AUD for means and standard deviations, dimensionless
    for correlations. Level moments are therefore multiplied by
    ``denorm = 1/normalisation`` so that the relative-deviation weighting
    (1/data^2 for |data| >= 1) treats all means and SDs consistently (all far
    above 1 in AUD) and correlations as absolute deviations (|data| < 1).

    Returns ``(moment_fn, denorm)``.
    """
    raw_moment_fn = make_moment_fn(moment_spec)
    from dolo.compiler.stage_factory import load_syntax
    _cal, _sett, *_ = load_syntax(mod_dir, calib_overrides, setting_overrides)
    denorm = 1.0 / float(_sett.get('normalisation', 1.0))

    def moment_fn(panels):
        """Compute moments and denormalise levels to AUD."""
        raw = raw_moment_fn(panels)
        out = {}
        for k, v in raw.items():
            base = k.rsplit('__age', 1)[0] if '__age' in k else k
            if any(base.startswith(p) for p in _LEVEL_PREFIXES):
                out[k] = v * denorm
            else:
                out[k] = v
        return out

    return moment_fn, denorm


def _release_solver_memory():
    """Collect garbage and return freed heap to the system after a solve.

    ``malloc_trim`` exists in glibc only; on other platforms the call is
    skipped (the ``OSError``/``AttributeError`` is the platform's absence of
    the symbol, not a failure of the run).
    """
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL(None).malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _make_member_solver(
    mod_dir, spec_factory, solver_method, calib_overrides, setting_overrides,
    N_sim, simulation_seed, types_spec,
):
    """The per-member solve and the pooling step of a trial with types.

    Returns ``(solve_member, pool)``:

    * ``solve_member(theta, k) -> (agent_idx, panels_k)`` expands the candidate
      into its K member calibrations (``expand_types``), solves member k with
      ``{**calib_overrides, **calibration_k}``, and simulates only the agents
      whose birth draw is k (``simulate_type_subset``);
    * ``pool(theta, parts) -> panels`` scatters the K parts back into full
      panels in the agents' original order (``pool_by_type``), adding
      ``type_idx`` and one ``(N,)`` array per heterogeneous parameter.

    The birth draw ``draw_types(N_sim, shares, simulation_seed)`` depends on
    its arguments only, so every rank of a group computes the same assignment
    and the root's pooling agrees with the members' subsets. The location and
    spread parameters never reach the solver: ``expand_types`` removes them.
    """
    K = int(types_spec['n'])
    names = list(types_spec['parameters'])

    def members(theta):
        records = expand_types(theta, types_spec)
        if len(records) != K:
            raise RuntimeError(
                f"expand_types returned {len(records)} members for n={K}")
        shares = np.array([share for share, _ in records], dtype=np.float64)
        return records, draw_types(N_sim, shares, simulation_seed)

    def solve_member(theta, k):
        records, type_idx = members(theta)
        _, calibration_k = records[k]
        agent_idx = np.flatnonzero(type_idx == k)
        nest, grids = solve(
            str(mod_dir),
            spec_factory_name=spec_factory,
            method_switch=solver_method,
            draw={
                "calibration": {**calib_overrides, **calibration_k},
                "settings": setting_overrides,
            },
            verbose=False,
            strip_solved=True,
        )
        panels_k = simulate_type_subset(
            nest, grids, agent_idx, N_sim, simulation_seed)
        del nest, grids
        _release_solver_memory()
        return agent_idx, panels_k

    def pool(theta, parts):
        records, type_idx = members(theta)
        member_values = {
            name: [calibration_k[name] for _, calibration_k in records]
            for name in names
        }
        return pool_by_type(list(parts), type_idx, N_sim, member_values)

    return solve_member, pool


def types_summary(types_spec, theta_best=None):
    """The ``beta_types`` block of ``summary.json`` and ``manifest.json``.

    Always: ``n``, ``shares`` and, per heterogeneous parameter, its
    ``location``, ``spread`` and ``transform``. With ``theta_best``: the
    ``nodes`` (member values) and the ``implied_mean`` (shares-weighted mean)
    of every parameter at that candidate, and, when ``beta`` is among the
    parameters, the same two as top-level ``nodes`` and ``implied_mean`` (the
    names the spec gives them, Section 5.6).
    """
    K = int(types_spec['n'])
    parameters = {
        name: {
            'location': block['location'],
            'spread': block['spread'],
            'transform': block.get('transform', 'logit'),
        }
        for name, block in types_spec['parameters'].items()
    }
    out = {'n': K, 'shares': [1.0 / K] * K, 'parameters': parameters}
    if theta_best is None:
        return out
    records = expand_types(theta_best, types_spec)
    shares = [float(share) for share, _ in records]
    out['shares'] = shares
    for name in parameters:
        nodes = [float(calibration_k[name]) for _, calibration_k in records]
        parameters[name]['nodes'] = nodes
        parameters[name]['implied_mean'] = float(np.dot(shares, nodes))
    if 'beta' in parameters:
        out['nodes'] = parameters['beta']['nodes']
        out['implied_mean'] = parameters['beta']['implied_mean']
    return out


def build_criterion(
    mod_dir, spec, spec_factory, solver_method, calib_overrides,
    setting_overrides, N_sim, simulation_seed, comm, *,
    types_spec=None, group=None,
):
    """Compose the SMM criterion the driver minimises.

    Returns ``(criterion, moment_fn, data_moments, denorm)``.

    ``criterion(theta) -> loss`` is kikku's closure over the trial
    ``solve -> simulate_lifecycle -> panels``; ``moment_fn`` maps panels to
    moments in the data's units; ``data_moments`` are the targets the loss
    is computed against (for ``data_source: precomputed``, ``spec['data_moments']``
    filtered to the spec's targets; for ``selfgen``, ``spec['data_moments']``
    as the caller generated them); ``denorm`` is the level factor
    1/normalisation.

    Attributes set on the returned ``criterion``:

    * ``trial``: the trial function, wrapped so that any exception is
      printed (MPI rank, theta, message) and counted before it is re-raised.
      kikku's closure then scores the candidate at ``BIG_LOSS``; the count
      and the printed line are what keep such a failure from being silent.
    * ``n_failures``: number of trial exceptions so far on this rank.
    * ``last_nest``: ``[nest, grids]`` from the most recent trial, kept for
      ``best.nst``; every earlier nest is cleared when a new one arrives.
      With types it stays ``[None, None]``: a candidate has K nests, and
      ``best.nst`` is not written for a types run.

    With ``types_spec`` (the ``estimation.types`` block) the trial is
    ``expand_types -> K x (solve, simulate_type_subset) -> pool_by_type``:

    * ``group=None`` (serial): the K members are solved in sequence inside
      the trial on this process;
    * ``group=(group_comm, K)`` (MPI): this process is a group root; the
      trial is ``type_groups.make_group_trial``, which has every rank of the
      group solve one member and pools the gathered parts.
    """
    moment_spec = spec['moment_spec']
    moment_fn, denorm = _denormalised_moment_fn(
        mod_dir, moment_spec, calib_overrides, setting_overrides)

    if is_root(comm):
        print(f"  Denorm factor: {denorm:.0f} (model units -> AUD)")

    data_source = moment_spec.get('data_source', 'precomputed')
    data_moments = spec['data_moments']

    # Filter precomputed data moments to only keys the model can produce.
    # The CSV has 130+ columns but the model only targets ~10.
    # Unmatched keys would get NAN_PENALTY, making the loss ~1e9.
    # Selfgen data is already in model keys by construction — no filtering needed.
    if data_source == 'precomputed':
        targets = moment_spec.get('targets') or []
        ident = moment_spec.get('identification') or {}
        target_prefixes = set()
        for t in targets:
            target_prefixes.add(t['key'])
        for var in ident.get('mean', []) or []:
            target_prefixes.add(f'mean_{var}')
        for var in ident.get('sd', []) or ident.get('sds', []) or []:
            target_prefixes.add(f'sd_{var}')
        for pair in ident.get('corrs', []) or []:
            target_prefixes.add(f'corr_{pair[0]}_{pair[1]}')
        for var in ident.get('autocorrs', []) or []:
            target_prefixes.add(f'autocorr_{var}')
        data_moments = {
            k: v for k, v in data_moments.items()
            if any(k.startswith(p + '__') or k == p for p in target_prefixes)
        }

    if is_root(comm):
        print(f"  Data moments: {len(data_moments)} keys")
    if not data_moments:
        raise RuntimeError("No data moments to match — check data_source and moment spec.")

    # --- Build trial function ---
    # _last_nest stores the nest+grids from the most recent trial call.
    # After estimation, if is_final, the driver saves it as best.nst.
    _last_nest = [None, None]  # [nest, grids] — mutable container for closure

    def single_trial(theta):
        merged_calib = {**calib_overrides, **theta}
        nest, grids = solve(
            str(mod_dir),
            spec_factory_name=spec_factory,
            method_switch=solver_method,
            draw={
                "calibration": merged_calib,
                "settings": setting_overrides,
            },
            verbose=False,
            strip_solved=True,
        )
        panels = simulate_lifecycle(nest, grids, N=N_sim, seed=simulation_seed)
        # Keep the nest/grids from this call (overwrite previous).
        # Only the last evaluation's nest survives — all prior ones are freed.
        if _last_nest[0] is not None:
            _last_nest[0].clear()
        if _last_nest[1] is not None:
            _last_nest[1].clear()
        _last_nest[0] = nest
        _last_nest[1] = grids
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL(None).malloc_trim(0)
        except (OSError, AttributeError):
            pass
        return panels

    if types_spec is None:
        trial = single_trial
    else:
        K = int(types_spec['n'])
        solve_member, pool = _make_member_solver(
            mod_dir, spec_factory, solver_method, calib_overrides,
            setting_overrides, N_sim, simulation_seed, types_spec)

        def log(message):
            print(message, flush=True)

        if group is None:
            def serial_types_trial(theta):
                """K member solves in sequence on this process, then pooling."""
                parts, seconds = [], []
                for k in range(K):
                    start = time.perf_counter()
                    parts.append(solve_member(theta, k))
                    seconds.append(time.perf_counter() - start)
                log("[types] serial member solve times (s): "
                    + " ".join(f"k={k} {s:.2f}" for k, s in enumerate(seconds)))
                return pool(theta, parts)

            trial = serial_types_trial
        else:
            group_comm, K_group = group
            if int(K_group) != K:
                raise ValueError(
                    f"group has K={K_group} members but the types block has n={K}")
            trial = make_group_trial(group_comm, K, solve_member, pool, log)

    def logged_trial(theta):
        """Run the trial; print and count any exception before re-raising it.

        kikku's criterion closure turns the exception into the penalty loss
        and prints only the first three per process, so without this line a
        run in which every evaluation fails would look like a converged run.
        """
        try:
            return trial(theta)
        except Exception as exc:
            rank = comm.Get_rank() if comm is not None else None
            where = f" rank={rank}" if rank is not None else ""
            print(f"[trial failure]{where} {type(exc).__name__}: {exc}  "
                  f"theta={theta}", flush=True)
            criterion.n_failures += 1
            raise

    criterion = make_criterion(logged_trial, moment_fn, data_moments)
    criterion.n_failures = 0
    criterion.trial = logged_trial
    criterion.last_nest = _last_nest
    return criterion, moment_fn, data_moments, denorm


def resolved_calibration_keys(mod_dir, spec_factory_name):
    """Names of the calibration the solver reads for this registry and spec factory.

    Resolves the spec factory's calibration chain exactly as ``solve()`` does
    (``load_spec`` then ``make_spec`` with no ``$draw``) and returns the union
    of the calibration keys over the recipe's stages at horizon 0. The male
    overlay therefore counts, and the top-level ``calibration.yaml`` (which
    matches the chain only by coincidence today) is not consulted.
    """
    registry_dir = Path(mod_dir)
    recipe = load_spec(str(registry_dir / spec_factory_name))
    resolved = make_spec(recipe, registry_dir=str(registry_dir))
    keys = set()
    for stage_name in recipe.stages:
        keys |= set(resolved[stage_name][0]["calibration"].keys())
    return keys


def check_parameter_names(resolved_calibration_keys, free_names, calib_overrides, types_spec):
    """Stop the driver when a parameter name is not one the solver reads.

    ``solve()`` accepts unknown calibration keys silently: a free parameter or
    an override with a stale name is ignored and the estimation proceeds as if
    the parameter existed (this is how the old Cobb-Douglas spec estimated
    ``gamma_c``, which that utility does not have). Every offending name is
    listed in one ``ValueError``.

    Rules (spec Sections 5.2 and 5.6):

    * every name under ``free`` and every key of ``calib_overrides`` (from
      ``--params-override`` and the sweep block) must be a calibration key,
      except the names a ``types`` block maps through ``location`` and
      ``spread`` (of every entry under ``types.parameters``), which must
      *not* be calibration keys (they would shadow a model parameter) and
      may not be passed as overrides (they never reach the solver);
    * each ``types`` target (a key of ``types.parameters``) must be a
      calibration key and must not itself be listed under ``free``.

    Only the keys of ``calib_overrides`` are examined.
    """
    keys = set(resolved_calibration_keys)
    free = list(free_names)
    mapped = set()
    offending = []
    for target, block in _types_parameters(types_spec).items():
        if target not in keys:
            offending.append(f"{target!r} (types target) is not a calibration key")
        if target in free:
            offending.append(
                f"{target!r} (types target) is also listed under free; "
                f"the types block replaces it with its location and spread")
        for role in ('location', 'spread'):
            name = block[role]
            mapped.add(name)
            if name in keys:
                offending.append(
                    f"{name!r} ({role} of types target {target!r}) is a "
                    f"calibration key and would shadow a model parameter")
    for name in free:
        if name in mapped or name in keys:
            continue
        offending.append(f"{name!r} (free parameter) is not a calibration key")
    for name in calib_overrides:
        if name in mapped:
            offending.append(
                f"{name!r} (calibration override) is mapped by the types block "
                f"and may not be passed to the solver")
        elif name not in keys:
            offending.append(f"{name!r} (calibration override) is not a calibration key")
    if offending:
        raise ValueError(
            "parameter names do not match the calibration the solver will use:\n  "
            + "\n  ".join(offending)
            + f"\n  calibration keys: {sorted(keys)}"
        )


def _types_parameters(types_spec):
    """The ``parameters`` mapping of a types block (``{}`` when there is none)."""
    return dict((types_spec or {}).get('parameters') or {})


def _types_mapped_names(types_spec):
    """Every location and spread name a types block maps."""
    return {block[role] for block in _types_parameters(types_spec).values()
            for role in ('location', 'spread')}


def load_types_spec(est_yaml):
    """Return the ``estimation.types`` block, or None when the spec has none.

    The block has the shape

        types:
          n: 4                          # members per candidate
          parameters:
            beta: {location: beta_bar, spread: sigma_beta, transform: logit}

    ``n`` and a non-empty ``parameters`` mapping are required; every entry
    needs ``location`` and ``spread`` (``transform`` defaults to ``logit`` in
    ``expand_types``). Member k takes the k-th equiprobable quantile of every
    listed parameter (spec 5.6).
    """
    block = (est_yaml or {}).get('types')
    if not block:
        return None
    if not isinstance(block, dict):
        raise ValueError("estimation.types must be a mapping with 'n' and 'parameters'")
    if 'n' not in block:
        raise ValueError("estimation.types lacks 'n', the number of type members")
    try:
        n = int(block['n'])
    except (TypeError, ValueError):
        raise ValueError(f"estimation.types.n must be an integer, got {block['n']!r}")
    if n < 1:
        raise ValueError(f"estimation.types.n must be at least 1, got {n}")
    parameters = block.get('parameters')
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(
            "estimation.types lacks a non-empty 'parameters' mapping "
            "(e.g. parameters: {beta: {location: beta_bar, spread: sigma_beta}})")
    for target, entry in parameters.items():
        if not isinstance(entry, dict):
            raise ValueError(f"estimation.types.parameters.{target} must be a mapping")
        for role in ('location', 'spread'):
            if role not in entry:
                raise ValueError(
                    f"estimation.types.parameters.{target} lacks '{role}'")
    out = dict(block)
    out['n'] = n
    out['parameters'] = {target: dict(entry) for target, entry in parameters.items()}
    return out


def check_types_data_source(types_spec, moment_spec):
    """Refuse a ``types`` block together with self-generated data (spec 5.6).

    The true values of the location and spread parameters would have no
    defined source: they are not calibration keys, and a value passed as a
    calibration override is ignored by ``solve()``. Self-generated runs with
    types are a later spec.
    """
    if types_spec is None:
        return
    data_source = str((moment_spec or {}).get('data_source') or 'precomputed').lower()
    if data_source == 'selfgen':
        raise ValueError(
            "estimation.types cannot be combined with moments.data_source: "
            "selfgen. The true values of the mapped parameters "
            f"({sorted(_types_mapped_names(types_spec))}) "
            "have no defined source in a self-generated run. Use precomputed "
            "data moments, or drop the types block."
        )


def _run_single_estimation(
    mod_dir, spec_path, spec, est_yaml, args,
    calib_overrides, setting_overrides, comm,
    results_dir, scratch_dir, run_id, sub_label=None,
):
    """Run one CE-SMM estimation. Used by both single and sweep modes."""

    moment_spec = spec['moment_spec']
    param_spec = spec['free']
    method_options = dict(spec['method_options'])

    # In sweep mode, n_samples = sub_comm size (not the YAML total)
    if comm is not None:
        sub_size = comm.Get_size()
        method_options['n_samples'] = sub_size
    # --n-samples applies to serial runs only; under an MPI launcher the
    # candidate count is the communicator size (set just above). A run is
    # serial when there is no communicator (mpi4py not importable) or when
    # the communicator has a single rank: with mpi4py installed, a plain
    # `python -m` run still receives COMM_WORLD, of size 1.
    serial = comm is None or comm.Get_size() == 1
    if serial and getattr(args, 'n_samples', None) is not None:
        method_options['n_samples'] = int(args.n_samples)
    if getattr(args, 'n_elite', None) is not None:
        method_options['n_elite'] = int(args.n_elite)

    simulation_seed = int(method_options.get('simulation_seed', 99))
    N_sim = args.N_sim
    solver_method = _solver_method_from_cli(args)
    types_spec = load_types_spec(est_yaml)
    # Types under MPI: the communicator is split into groups of K ranks
    # below, after the last collective on `comm` that precedes estimate().
    # In serial mode the K members are solved in sequence inside the trial.
    types_mpi = types_spec is not None and not serial
    spec_name = Path(args.spec).stem
    mod_name = Path(args.mod).name  # e.g. 'separable', 'separable_males'

    # Build path: results_dir/<mod_name>/<spec_name>/<sub_label>/est_<run_id>/
    path_parts = [results_dir, mod_name, spec_name]
    scratch_parts = [scratch_dir, mod_name, spec_name]
    if sub_label:
        path_parts.append(sub_label)
        scratch_parts.append(sub_label)
    path_parts.append(f'est_{run_id}')
    scratch_parts.append(f'est_{run_id}')
    results_run = os.path.join(*path_parts)
    scratch_run = os.path.join(*scratch_parts)

    if is_root(comm):
        print(f"\n{'='*60}")
        if sub_label:
            print(f"Sweep point: {sub_label}")
        print(f"  Free params: {list(param_spec.keys())}")
        print(f"  n_samples={method_options.get('n_samples')}, "
              f"n_elite={method_options.get('n_elite')}")
        print(f"  Calib overrides: {calib_overrides}")

    # --- Build data moments ---
    # Precomputed data (the CSV, already in AUD) is read by build_criterion
    # from spec['data_moments']. Self-generated data is produced here, at the
    # default calibration, with the same denormalised moment function the
    # criterion uses, and handed to build_criterion in the spec's place.
    data_source = moment_spec.get('data_source', 'precomputed')
    if data_source == 'selfgen':
        moment_fn, denorm = _denormalised_moment_fn(
            mod_dir, moment_spec, calib_overrides, setting_overrides)
        # Selfgen: generate data at default calibration, denormalise via moment_fn
        if is_root(comm):
            print("  Data source: selfgen...")
            nest_data, grids_data = solve(
                str(mod_dir),
                spec_factory_name=args.spec_factory,
                draw={
                    "calibration": calib_overrides,
                    "settings": setting_overrides,
                },
                verbose=False,
                strip_solved=False,  # keep full nest for .nst save
            )
            data_panels = simulate_lifecycle(
                nest_data, grids_data, N=N_sim, seed=simulation_seed)
            data_moments = moment_fn(data_panels)  # already denormalised
            # Save the selfgen (true) nest
            try:
                from kikku.run.nest_io import save_nest
                save_nest(nest_data, os.path.join(scratch_run, 'true.nst'),
                          solutions=True,
                          metadata={'calib_overrides': calib_overrides,
                                    'data_source': 'selfgen'})
                save_nest(nest_data, os.path.join(results_run, 'true.nst'),
                          solutions=True,
                          metadata={'calib_overrides': calib_overrides,
                                    'data_source': 'selfgen'})
                print(f"  Saved true.nst ({len(nest_data['solutions'])} periods)")
            except Exception as e:
                print(f"  WARNING: could not save true.nst: {e}")
            del nest_data, grids_data, data_panels
        else:
            data_moments = None
        data_moments = bcast_item(data_moments, comm, root=0)
        spec_for_criterion = {**spec, 'data_moments': data_moments}
    else:
        # Precomputed: CSV is already in AUD — no normalisation needed.
        spec_for_criterion = spec

    # --- Build the criterion: trial (solve -> simulate -> panels), moments, loss ---
    # With types under MPI the criterion is built on the group roots only,
    # after the split below; workers never hold one.
    if not types_mpi:
        criterion, moment_fn, data_moments, denorm = build_criterion(
            mod_dir, spec_for_criterion, args.spec_factory, solver_method,
            calib_overrides, setting_overrides, N_sim, simulation_seed, comm,
            types_spec=types_spec,
        )
        # [nest, grids] of the most recent trial; saved as best.nst when is_final.
        _last_nest = criterion.last_nest

    # --- Create output directories ---
    if is_root(comm):
        os.makedirs(scratch_run, exist_ok=True)
        os.makedirs(results_run, exist_ok=True)
        print(f"  Scratch: {scratch_run}")
        print(f"  Results: {results_run}")

    method_options['checkpoint_dir'] = scratch_run

    # --- Resume from checkpoint (root loads, then broadcasts) ---
    resume_state = None
    if args.resume and is_root(comm):
        state_path = os.path.join(scratch_run, 'state.pkl')
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                resume_state = pickle.load(f)
            print(f"  Resuming from checkpoint: iter {resume_state['it'] + 1}")
        else:
            print(f"  No checkpoint found — starting fresh.")
    if args.resume:
        resume_state = bcast_item(resume_state if is_root(comm) else None, comm, root=0)

    if args.max_iter_this_run is not None:
        method_options['max_iter_this_run'] = args.max_iter_this_run
    if getattr(args, 'max_iter', None) is not None:
        method_options['max_iter'] = args.max_iter

    # --- Types under MPI: the two layers (spec 5.6, type_groups.py) ---
    # The resume broadcast above was the last collective on `comm` before
    # estimate(); from here the workers talk to their group root only, until
    # they return to the post-estimate collectives on `comm` below.
    group_comm = None
    est_comm = comm
    if types_mpi:
        K = int(types_spec['n'])
        group_comm, roots_comm = split_type_groups(comm, K)
        if roots_comm is None:
            # Worker: serve the group root until STOP, then take part in the
            # two collectives on `comm` that follow estimate() on the roots
            # (allreduce of the failure count, broadcast of all_failed) and
            # in everything after (the is_final broadcasts, barriers, gather).
            solve_member, _pool = _make_member_solver(
                mod_dir, args.spec_factory, solver_method, calib_overrides,
                setting_overrides, N_sim, simulation_seed, types_spec)
            worker_loop(group_comm, solve_member)
            comm.allreduce(0)
            all_failed = bcast_item(None, comm, root=0)
            return None, None, all_failed
        est_comm = roots_comm
        method_options['n_samples'] = roots_comm.Get_size()
        if is_root(comm):
            print(f"[types] K={K} members per candidate: {roots_comm.Get_size()} "
                  f"groups of {K} ranks on {comm.Get_size()} ranks; "
                  f"n_samples={roots_comm.Get_size()} candidates per iteration",
                  flush=True)
        criterion, moment_fn, data_moments, denorm = build_criterion(
            mod_dir, spec_for_criterion, args.spec_factory, solver_method,
            calib_overrides, setting_overrides, N_sim, simulation_seed, comm,
            types_spec=types_spec, group=(group_comm, K),
        )
        _last_nest = criterion.last_nest

    # --- Manifest (root): written once the run's options are final ---
    # After the --max-iter merge and the split, so that max_iter and
    # n_samples are the values estimate() runs with.
    if is_root(comm):
        manifest = {
            'mod': str(mod_dir),
            'spec': str(spec_path),
            'spec_factory': args.spec_factory,
            'solver_method': solver_method,
            'run_id': run_id,
            'sweep_point': sub_label,
            'calib_overrides': calib_overrides,
            'n_samples': method_options.get('n_samples'),
            'n_elite': method_options.get('n_elite'),
            'max_iter': method_options.get('max_iter'),
            'grid': setting_overrides,
            'N_sim': N_sim,
            'simulation_seed': simulation_seed,
            'sampling_seed': method_options.get('sampling_seed'),
            'free_params': list(param_spec.keys()),
            'timestamp': run_id,
            'git_commit': _git_commit(Path(__file__).resolve().parent.parent.parent),
            'beta_types': (types_summary(types_spec)
                           if types_spec is not None else None),
        }
        # Scratch is purged on Gadi; the results copy keeps the grid, N_sim,
        # seeds, spec factory and method of a finished run with its estimates.
        _write_manifest(manifest, (scratch_run, results_run))

    # --- Estimate ---
    # The roots of a types run broadcast STOP to their workers whatever
    # happens inside estimate(), so an exception there cannot leave the
    # workers waiting for the next EVAL.
    try:
        result = estimate(
            criterion, param_spec,
            method=spec['method'],
            method_options=method_options,
            comm=est_comm,
            verbose=is_root(comm),
            resume_state=resume_state,
        )
    finally:
        if group_comm is not None:
            group_comm.bcast((STOP,), root=0)

    # Determine if this is a final segment (write results) or intermediate (checkpoint only)
    global_max = int(method_options.get('max_iter', 200))
    is_final = (
        result.converged
        or (args.max_iter_this_run is None)
        or (result.n_iter >= global_max)
    )

    # --- Failures are never silent (spec 5.2 item 5) ---
    # Each rank counted the trial exceptions kikku scored at BIG_LOSS; the
    # total over the communicator is printed on the root. When the best loss
    # is itself the penalty, no candidate was ever evaluated, although kikku
    # reports convergence: the run is declared failed on a final segment and
    # main() exits with code 3 on every rank. Restart segments (exit 42) are
    # left to the existing logic.
    n_failures = int(criterion.n_failures)
    if comm is not None:
        n_failures = comm.allreduce(n_failures)  # default op: sum over ranks
    all_failed = False
    if is_root(comm):
        print(f"  Trial failures: {n_failures} evaluation(s) raised and were "
              f"scored at the penalty loss {BIG_LOSS:g}", flush=True)
        if is_final and result.objective >= BIG_LOSS:
            print(f"  ERROR: every evaluation failed (best loss {result.objective:g} "
                  f"is the penalty); no candidate was evaluated. "
                  f"Exiting with code 3.", flush=True)
            all_failed = True
    all_failed = bcast_item(all_failed if is_root(comm) else None, comm, root=0)

    # --- Save results ---
    summary_row = None
    if is_root(comm) and is_final:
        diag = diagnostics(result, data_moments,
                           moment_names=get_moment_names(moment_spec))

        theta_mean = result.theta
        theta_se = {n: float('nan') for n in param_spec}
        if result.history:
            last = result.history[-1]
            if 'means' in last:
                theta_mean = last['means']
            if 'cov' in last:
                cov = np.asarray(last['cov'])
                names_sorted = sorted(param_spec.keys())
                se_vec = np.sqrt(np.diag(cov))
                theta_se = {names_sorted[i]: float(se_vec[i])
                            for i in range(len(names_sorted))}

        summary = {
            'theta_best': result.theta,
            'theta_mean': theta_mean,
            'theta_se': theta_se,
            'objective': result.objective,
            'converged': result.converged,
            'n_iter': result.n_iter,
            'sweep_point': sub_label,
            'calib_overrides': calib_overrides,
        }
        if types_spec is not None:
            # The type nodes, shares and implied mean at theta_best; the
            # manifest is rewritten so that it carries the same block.
            beta_types = types_summary(types_spec, result.theta)
            summary['beta_types'] = beta_types
            manifest['beta_types'] = beta_types
            _write_manifest(manifest, (scratch_run, results_run))

        for fname, obj in [
            ('theta_best.json', result.theta),
            ('theta_mean.json', theta_mean),
            ('theta_se.json', theta_se),
            ('summary.json', summary),
        ]:
            with open(os.path.join(results_run, fname), 'w') as f:
                json.dump(obj, f, indent=2)

        with open(os.path.join(results_run, 'fit_table.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'moment', 'data', 'simulated', 'residual',
                'contribution', 'contribution_pct'])
            writer.writeheader()
            for row in diag['fit_table']:
                writer.writerow(row)

        with open(os.path.join(results_run, 'convergence.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['iter', 'best_loss', 'elite_mean_loss'])
            writer.writeheader()
            for i, h in enumerate(result.history):
                writer.writerow({
                    'iter': i,
                    'best_loss': h.get('best_loss'),
                    'elite_mean_loss': h.get('elite_mean_loss'),
                })

        # Save best.nst (stripped nest from last trial evaluation)
        if _last_nest[0] is not None:
            try:
                from kikku.run.nest_io import save_nest
                meta = {'theta_best': result.theta,
                        'objective': result.objective,
                        'n_iter': result.n_iter}
                save_nest(_last_nest[0], os.path.join(results_run, 'best.nst'),
                          solutions=True, metadata=meta)
                save_nest(_last_nest[0], os.path.join(scratch_run, 'best.nst'),
                          solutions=True, metadata=meta)
                print(f"  Saved best.nst (stripped)")
            except Exception as e:
                print(f"  WARNING: could not save best.nst: {e}")
        _last_nest[0] = None
        _last_nest[1] = None

        # Print summary
        print(f"\nLoss:       {result.objective:.6f}")
        print(f"Converged:  {result.converged} ({result.n_iter} iters)")
        print(f"\n{'param':12s} {'best':>10s} {'mean':>10s} {'SE':>10s}")
        print('-' * 44)
        for n in sorted(param_spec.keys()):
            print(f"{n:12s} {result.theta[n]:10.4f} {theta_mean[n]:10.4f} {theta_se[n]:10.6f}")

        # Local copy — default: benchmarks/durables/estimation/results/
        if args.local_results and args.local_results.lower() == 'none':
            local_root = None
        elif args.local_results:
            local_root = Path(args.local_results)
        else:
            repo_root = Path(__file__).parent.parent.parent
            local_root = repo_root / 'benchmarks' / 'durables' / 'estimation' / 'results'

        if local_root is not None:
            lp = [str(local_root), mod_name, spec_name]
            if sub_label:
                lp.append(sub_label)
            lp.append(f'est_{run_id}')
            local_results = Path(os.path.join(*lp))
            try:
                os.makedirs(local_results, exist_ok=True)
                for fname in ('theta_best.json', 'theta_mean.json', 'theta_se.json',
                              'summary.json', 'fit_table.csv', 'convergence.csv'):
                    src = os.path.join(results_run, fname)
                    if os.path.exists(src):
                        shutil.copy2(src, str(local_results / fname))
                print(f"Local copy:  {local_results}")
            except OSError as e:
                print(f"WARNING: could not save local copy: {e}")

        print(f"Results: {results_run}")

        # Build summary row for sweep aggregation.
        # Prefix sweep-point calib overrides with 'true_' so they aren't
        # overwritten by the estimated theta (e.g. true_gamma_c vs gamma_c).
        sweep_true = {f'true_{k}': v for k, v in calib_overrides.items()}
        summary_row = {**sweep_true, **result.theta,
                       'objective': result.objective,
                       'converged': result.converged,
                       'n_iter': result.n_iter}

    elif is_root(comm):
        print(f"\n  Checkpoint saved at iter {result.n_iter}. Will resume on next restart.")
        print(f"  best_loss={result.objective:.6f}")

    return summary_row, is_final, all_failed


def main():
    parser = argparse.ArgumentParser(
        description="SMM estimation for durables")

    parser.add_argument(
        '--mod', type=str, required=True,
        help='Path to mod directory (e.g. syntax/separable, syntax/cobb_douglas)')
    parser.add_argument(
        '--spec', type=str, default='baseline.yaml',
        help='Estimation spec YAML, relative to mod/estimation/ (default: baseline.yaml)')
    parser.add_argument(
        '--spec-factory', type=str, default='spec_factory.yaml',
        dest='spec_factory',
        help='spec_factory YAML in the mod dir that selects the calibration chain '
             '(default: spec_factory.yaml = female baseline; pass '
             'spec_factory_males.yaml for the male calibration overlay).')
    parser.add_argument(
        '--scratch', type=str, default=None,
        help='Scratch dir for intermediate outputs (overrides YAML)')
    parser.add_argument(
        '--results', type=str, default=None,
        help='Results dir for final outputs (overrides YAML)')
    parser.add_argument(
        '--settings-override', nargs='*', default=[],
        dest='settings_override',
        help='Grid/solver overrides (e.g. n_a=300 n_h=300)')
    parser.add_argument(
        '--params-override', nargs='*', default=[],
        dest='params_override',
        help='Calibration / parameter overrides (e.g. t0=20)')
    parser.add_argument(
        '--N-sim', type=int, default=10000,
        help='Number of simulation agents (default: 10000)')
    parser.add_argument(
        '--methods-override', nargs='*', default=[],
        dest='methods_override',
        help=(
            'Adjuster upper-envelope method, e.g. '
            'adjuster_cons.upper_env.upper_envelope=NEGM '
            '(the kernel/policy node; the tag after the last `=` is what is '
            'read).'
        ),
    )
    parser.add_argument(
        '--local-results', type=str, default=None,
        help='Local results dir (default: benchmarks/durables/estimation/results/). '
             'Set to "none" to disable local copy.')
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from latest checkpoint in scratch dir')
    parser.add_argument(
        '--max-iter-this-run', type=int, default=None,
        dest='max_iter_this_run',
        help='Max CE iterations for this restart segment. '
             'Exits with code 42 when exhausted (not converged).')
    parser.add_argument(
        '--max-iter', type=int, default=None,
        dest='max_iter',
        help='Override the spec global max CE iterations. Set high (e.g. 100000) '
             'so the PBS time-bounded restart loop lets walltime bind.')
    parser.add_argument(
        '--run-id', type=str, default=None,
        help='Explicit run ID (timestamp). Used by PBS restart loop to '
             'ensure all segments use the same results directory.')
    parser.add_argument(
        '--n-samples', type=int, default=None, dest='n_samples',
        help='Candidates per CE iteration, overriding the spec. Serial runs '
             'only: under MPI the candidate count is the communicator size.')
    parser.add_argument(
        '--n-elite', type=int, default=None, dest='n_elite',
        help='Elite candidates per CE iteration, overriding the spec '
             '(serial and MPI).')

    args = parser.parse_args()
    world_comm = get_comm()

    # --- Resolve paths ---
    example_root = Path(__file__).parent
    mod_dir = (example_root / args.mod).resolve()
    spec_path = mod_dir / 'estimation' / args.spec

    if not mod_dir.exists():
        raise FileNotFoundError(f"Mod directory not found: {mod_dir}")
    if not spec_path.exists():
        raise FileNotFoundError(f"Estimation spec not found: {spec_path}")

    # --- Load estimation spec ---
    spec = load_estimation_spec(str(spec_path))

    with spec_path.open() as f:
        raw_yaml = yaml.safe_load(f)
    est_yaml = raw_yaml.get('estimation', {})

    scratch_dir = args.scratch or est_yaml.get('scratch_dir', 'scratch/est')
    results_dir = args.results or est_yaml.get('results_dir', 'results/est')
    scratch_dir = scratch_dir.replace('{user}', os.environ.get('USER', 'unknown'))
    results_dir = results_dir.replace('{user}', os.environ.get('USER', 'unknown'))

    setting_overrides = _parse_key_value_list(args.settings_override)
    calib_overrides = _parse_key_value_list(args.params_override)

    # --- Check for sweep ---
    sweep_spec = est_yaml.get('sweep')

    # --- Start-up guards, before any solve (spec 5.2 item 3, 5.6) ---
    # Every rank runs the same deterministic checks on the same files, so a
    # refusal raises on all ranks together. The calibration is resolved through
    # the spec factory in use, so the male overlay counts.
    types_spec = load_types_spec(est_yaml)
    check_types_data_source(types_spec, spec['moment_spec'])
    check_parameter_names(
        resolved_calibration_keys(mod_dir, args.spec_factory),
        list(spec['free'].keys()),
        {**calib_overrides, **(sweep_spec or {})},
        types_spec,
    )
    # Types under MPI need K ranks per candidate for every sweep point. The
    # check runs on every rank, before any Split (the sweep split below and
    # the type-group split inside _run_single_estimation), so that all ranks
    # raise together and no output folder is created. A single rank solves
    # the K members in sequence and needs no check.
    if types_spec is not None and world_comm is not None and world_comm.Get_size() > 1:
        n_points = len(_build_sweep_grid(sweep_spec)) if sweep_spec is not None else 1
        check_divisibility(world_comm, int(types_spec['n']), n_points)

    # run_id: use --run-id if provided (restart loop), otherwise generate fresh
    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    spec_name = Path(args.spec).stem
    mod_name = Path(args.mod).name  # e.g. 'separable', 'separable_males'

    # Broadcast run_id so all ranks agree
    run_id = bcast_item(run_id if is_root(world_comm) else None, world_comm, root=0)

    if sweep_spec is None:
        # ── Single estimation (no sweep) ──
        if is_root(world_comm):
            print(f"Estimation: {spec_path.name}")
            print(f"  Mod: {mod_dir}")
            print(f"  Method: {spec['method']}")
        _, is_final, all_failed = _run_single_estimation(
            mod_dir, spec_path, spec, est_yaml, args,
            calib_overrides, setting_overrides, world_comm,
            results_dir, scratch_dir, run_id,
        )

        # Exit codes for restart loop — ALL ranks must agree and exit together
        is_final = bcast_item(is_final if is_root(world_comm) else None, world_comm, root=0)
        if args.max_iter_this_run is not None and not is_final:
            if world_comm is not None:
                world_comm.Barrier()  # sync all ranks before exit
            sys.exit(42)
        # Every evaluation failed on a final segment (decision already
        # broadcast on world_comm inside _run_single_estimation).
        if all_failed:
            if world_comm is not None:
                world_comm.Barrier()
            sys.exit(3)
    else:
        # ── Sweep mode: split communicator ──
        sweep_grid = _build_sweep_grid(sweep_spec)
        n_points = len(sweep_grid)

        world_rank = world_comm.Get_rank() if world_comm else 0
        world_size = world_comm.Get_size() if world_comm else 1

        if is_root(world_comm):
            print(f"Estimation SWEEP: {spec_path.name}")
            print(f"  Mod: {mod_dir}")
            print(f"  Sweep: {n_points} points across {world_size} ranks")
            for i, pt in enumerate(sweep_grid):
                print(f"    [{i}] {pt}")

        # Split communicator: color = which sweep point this rank belongs to
        color = world_rank * n_points // world_size
        sub_comm = world_comm.Split(color, world_rank) if world_comm else None
        my_point = sweep_grid[color]
        sub_label = _sweep_point_label(my_point)

        # Merge sweep point into calib overrides
        sweep_calib = {**calib_overrides, **my_point}

        if is_root(sub_comm):
            sub_size = sub_comm.Get_size() if sub_comm else 1
            print(f"\n  Sweep [{color}] {sub_label}: {sub_size} ranks")

        summary_row, is_final, all_failed = _run_single_estimation(
            mod_dir, spec_path, spec, est_yaml, args,
            sweep_calib, setting_overrides, sub_comm,
            results_dir, scratch_dir, run_id, sub_label=sub_label,
        )

        # All ranks agree on is_final for their sub_comm
        is_final = bcast_item(is_final if is_root(sub_comm) else None, sub_comm, root=0)
        # World-level: final only if ALL ranks are final (min across all)
        my_final = 1 if is_final else 0
        from mpi4py import MPI as _MPI
        all_final_sum = world_comm.allreduce(my_final, op=_MPI.MIN) if world_comm else my_final
        all_final = bool(all_final_sum)
        # World-level: failed if ANY sweep point had every evaluation fail
        # (max across all); the sweep summary is still written before exit.
        my_failed = 1 if all_failed else 0
        any_failed = bool(
            world_comm.allreduce(my_failed, op=_MPI.MAX) if world_comm else my_failed)

        if not all_final and args.max_iter_this_run is not None:
            # Restart needed — skip gather, all ranks exit together
            if is_root(world_comm):
                print(f"\n  Sweep restart: not all points converged. Checkpoint saved.")
            if world_comm is not None:
                world_comm.Barrier()
            sys.exit(42)

        # --- Gather sweep summary on world root ---
        # Each sub_comm root has a summary_row; other ranks have None
        all_rows = world_comm.gather(summary_row, root=0) if world_comm else [summary_row]

        if is_root(world_comm):
            rows = [r for r in all_rows if r is not None]
            if rows:
                sweep_results_dir = os.path.join(results_dir, mod_name, spec_name)
                os.makedirs(sweep_results_dir, exist_ok=True)
                summary_path = os.path.join(sweep_results_dir, f'sweep_summary_{run_id}.csv')
                fieldnames = list(rows[0].keys())
                with open(summary_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
                print(f"\n{'='*60}")
                print(f"Sweep summary ({len(rows)} points): {summary_path}")

                # Also save to local results
                if args.local_results and args.local_results.lower() == 'none':
                    local_root = None
                elif args.local_results:
                    local_root = Path(args.local_results)
                else:
                    repo_root = Path(__file__).parent.parent.parent
                    local_root = repo_root / 'benchmarks' / 'durables' / 'estimation' / 'results'
                if local_root is not None:
                    local_sweep = local_root / mod_name / spec_name
                    try:
                        os.makedirs(local_sweep, exist_ok=True)
                        shutil.copy2(summary_path,
                                     str(local_sweep / f'sweep_summary_{run_id}.csv'))
                    except OSError:
                        pass

        if any_failed:
            if is_root(world_comm):
                print("\n  ERROR: at least one sweep point had every evaluation "
                      "fail. Exiting with code 3.", flush=True)
            if world_comm is not None:
                world_comm.Barrier()
            sys.exit(3)


if __name__ == '__main__':
    main()
