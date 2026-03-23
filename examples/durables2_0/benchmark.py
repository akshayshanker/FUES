"""Benchmarking for durables2_0: DDSL pipeline vs original, timing, accuracy.

Runs ``solve`` (YAML-driven) and ``solveEGM`` (monolithic)
and verifies array-level equivalence.  Also provides timing benchmarks
using the generic sweep infrastructure.
"""

import numpy as np
import time
import sys

# Patch UCGrid before any durables import
import interpolation.splines as _splines
_orig = _splines.UCGrid
def _patch(*args):
    return _orig(*[(float(a), float(b), int(n))
                   for a, b, n in args])
_splines.UCGrid = _patch

from examples.durables.durables_plot import solveEGM

from .solve import solve
from .outputs import get_timing, consumption_deviation

from kikku.run.sweep import sweep, param_grid
from kikku.run.metrics import format_table, write_table


def run_comparison(syntax_dir, calib_overrides=None,
                   config_overrides=None, verbose=True):
    """Run both pipelines and compare all arrays.

    Parameters
    ----------
    syntax_dir : str
        Path to durables2_0 syntax directory.
    calib_overrides, config_overrides : dict, optional
    verbose : bool

    Returns
    -------
    dict
        ``{'max_diff', 'all_match', ...}``
    """
    print("=" * 60)
    print("Durables2.0 DDSL pipeline vs Original")
    print("=" * 60)

    # --- New pipeline (DDSL) ---
    print("\n--- New (solve_nest from YAML) ---")
    t0 = time.time()
    nest, cp, grids, callables, _settings = solve(
        syntax_dir,
        calib_overrides=calib_overrides,
        config_overrides=config_overrides,
        verbose=False,
    )
    solutions = nest["solutions"]
    print(f"  Time: {time.time() - t0:.2f}s")
    print(f"  Periods solved: {len(solutions)}")

    timing = get_timing(nest)
    print(f"  Mean timing — solve: {timing['solve_time']*1000:.1f}ms, "
          f"keeper: {timing['keeper_ms']:.1f}ms, "
          f"adj: {timing['adj_ms']:.1f}ms, "
          f"discrete: {timing['discrete_ms']:.1f}ms")

    # --- Original pipeline ---
    print("\n--- Original (solveEGM) ---")
    t0 = time.time()
    old = solveEGM(cp, verbose=False)
    print(f"  Time: {time.time() - t0:.2f}s")

    # --- Compare ---
    print("\n--- Comparison ---")
    max_diffs = {}

    for sol in solutions:
        t = sol['t']
        if t not in old:
            continue
        o = old[t]
        kp = sol['keeper_cons']['dcsn']
        aj = sol['adjuster_cons']['dcsn']

        for name, new_arr, old_key in [
            ('c_keep', kp['c'], 'Ckeeper'),
            ('c_adj', aj['c'], 'Cadj'),
            ('h_adj', aj['h_choice'], 'Hadj'),
        ]:
            if old_key in o:
                d = np.max(np.abs(new_arr - o[old_key]))
                max_diffs[f'{name}[t={t}]'] = d

    overall = max(max_diffs.values()) if max_diffs else 0.0
    ok = overall < 1e-12

    sorted_d = sorted(max_diffs.items(), key=lambda x: -x[1])
    print(f"\n  Max difference: {overall:.2e}")
    print(f"  Match (< 1e-12): {ok}")
    if not ok and verbose:
        print("\n  Worst 10:")
        for name, d in sorted_d[:10]:
            print(f"    {name}: {d:.2e}")

    return {'max_diff': overall, 'all_match': ok}


def run_timing(syntax_dir, n_runs=3, calib_overrides=None,
               config_overrides=None, verbose=True):
    """Best-of-n timing benchmark using kikku.run.sweep.

    Parameters
    ----------
    syntax_dir : str
    n_runs : int
    calib_overrides, config_overrides : dict, optional
    verbose : bool

    Returns
    -------
    dict
        ``{'best_timing': dict, 'nest': dict, 'cp', 'grids', 'callables'}``
    """
    base_calib = dict(calib_overrides or {})
    base_config = dict(config_overrides or {})

    # Single-point grid (sweep infrastructure handles best-of-n)
    grid = [{}]

    last_result = {}

    def solve_fn(ov):
        nest, cp, grids, callables, _settings = solve(
            syntax_dir,
            calib_overrides=base_calib,
            config_overrides=base_config,
            verbose=False,
        )
        last_result.update({
            'nest': nest, 'cp': cp,
            'grids': grids, 'callables': callables,
        })
        return {'nest': nest}

    metric_fns = {
        'solve_ms': lambda r: get_timing(r['nest'])['solve_time'] * 1000,
        'keeper_ms': lambda r: get_timing(r['nest'])['keeper_ms'],
        'adj_ms': lambda r: get_timing(r['nest'])['adj_ms'],
        'discrete_ms': lambda r: get_timing(r['nest'])['discrete_ms'],
    }

    results = sweep(
        solve_fn, grid, metric_fns,
        n_reps=n_runs, warmup=True,
        best='min', verbose=verbose)

    best_timing = {
        'solve_time': results[0]['solve_ms'] / 1000,
        'keeper_ms': results[0]['keeper_ms'],
        'adj_ms': results[0]['adj_ms'],
        'discrete_ms': results[0]['discrete_ms'],
    }

    if verbose:
        print(f"Best of {n_runs} runs:")
        print(f"  solve: {best_timing['solve_time']*1000:.1f}ms, "
              f"keeper: {best_timing['keeper_ms']:.1f}ms, "
              f"adj: {best_timing['adj_ms']:.1f}ms, "
              f"discrete: {best_timing['discrete_ms']:.1f}ms")

    return {
        'best_timing': best_timing,
        **last_result,
    }


if __name__ == '__main__':
    syntax_dir = 'examples/durables2_0/syntax'

    if '--timing' in sys.argv:
        result = run_timing(syntax_dir, n_runs=3)
    else:
        result = run_comparison(syntax_dir)
        sys.exit(0 if result['all_match'] else 1)
