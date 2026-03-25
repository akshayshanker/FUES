"""Benchmarking for durables2_0: DDSL pipeline timing.

Array-level comparison against the legacy monolithic ``solveEGM`` path
lived here historically; this example uses the YAML-driven DDSL stack only.
Use ``examples/old/durables`` directly for legacy parity checks.

Also provides timing benchmarks using the generic sweep infrastructure.
"""

import numpy as np
import time
import sys

from .solve import solve
from .outputs import get_timing, consumption_deviation

from kikku.run.sweep import sweep, param_grid
from kikku.run.metrics import format_table, write_table


def run_comparison(syntax_dir, calib_overrides=None,
                   setting_overrides=None, verbose=True):
    """Run DDSL solve and print timing (legacy array comparison removed).

    Parameters
    ----------
    syntax_dir : str
        Path to durables2_0 syntax directory.
    calib_overrides, setting_overrides : dict, optional
    verbose : bool

    Returns
    -------
    dict
        ``{'legacy_skipped': True, 'max_diff': None, 'all_match': None}``
    """
    print("=" * 60)
    print("Durables2.0 DDSL pipeline vs Original")
    print("=" * 60)

    # --- New pipeline (DDSL) ---
    print("\n--- New (solve_nest from YAML) ---")
    t0 = time.time()
    nest, grids = solve(
        syntax_dir,
        calib_overrides=calib_overrides,
        setting_overrides=setting_overrides,
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

    print("\n--- Legacy solveEGM comparison omitted (DDSL-only example) ---")

    return {'max_diff': None, 'all_match': None, 'legacy_skipped': True}


def run_timing(syntax_dir, n_runs=3, calib_overrides=None,
               setting_overrides=None, verbose=True):
    """Best-of-n timing benchmark using kikku.run.sweep.

    Parameters
    ----------
    syntax_dir : str
    n_runs : int
    calib_overrides, setting_overrides : dict, optional
    verbose : bool

    Returns
    -------
    dict
        ``{'best_timing': dict, 'nest', 'grids'}``
    """
    base_calib = dict(calib_overrides or {})
    base_settings = dict(setting_overrides or {})

    # Single-point grid (sweep infrastructure handles best-of-n)
    grid = [{}]

    last_result = {}

    def solve_fn(ov):
        nest, grids = solve(
            syntax_dir,
            calib_overrides=base_calib,
            setting_overrides=base_settings,
            verbose=False,
        )
        last_result.update({
            'nest': nest,
            'grids': grids,
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
        sys.exit(0)
