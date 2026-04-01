"""Benchmarking functions for retirement model - timing sweeps and comparisons.

Compares FUES vs DC-EGM vs RFC vs CONSAV across grid sizes and delta values.
Uses the canonical pipeline (solve_nest) for all runs.

Author: Akshay Shanker, University of New South Wales, akshay.shanker@me.com
"""

import numpy as np
import os
import sys
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pathlib import Path
from .solve import solve_nest
from .outputs import (
    generate_timing_table_combined, generate_accuracy_table,
    get_policy, get_timing,
    euler, consumption_deviation,
)
from kikku.run.sweep import sweep, param_grid

SYNTAX_DIR = Path(__file__).resolve().parent / "syntax"

_cal_path = SYNTAX_DIR / "calibration.yaml"
_set_path = SYNTAX_DIR / "settings.yaml"

METHODS = ('RFC', 'FUES', 'DCEGM', 'CONSAV')


def _load_baseline():
    """Load baseline calibration and settings from syntax dir."""
    with open(_cal_path) as f:
        cal = yaml.safe_load(f)['calibration']
    with open(_set_path) as f:
        settings = yaml.safe_load(f)['settings']
    return cal, settings


def test_Timings(grid_sizes, delta_values, n=3, results_dir="results",
                 true_grid_size=20000, true_method='DCEGM',
                 calib_overrides=None, config_overrides=None,
                 latex_grids=None):
    """Run timing benchmarks across grid sizes and delta values.

    Uses ``kikku.run.sweep`` to iterate over the
    ``(grid_size, delta)`` Cartesian product.  For each point
    the solve_fn loops over all four UE methods internally,
    preserving the per-row table structure.

    Parameters
    ----------
    grid_sizes : list
        List of grid sizes to test.
    delta_values : list
        List of delta values to test.
    n : int
        Number of runs per configuration (best of n). Default is 3.
    results_dir : str
        Directory to save results. Default is "results".
    true_grid_size : int
        Grid size for computing "true" reference solution. Default is 20000.
    true_method : str
        Method used for "true" reference solution. Default is 'DCEGM'.
    calib_overrides : dict, optional
        Extra calibration overrides.
        ``delta`` is always overridden per sweep row.
    config_overrides : dict, optional
        Extra config overrides.
        ``grid_size`` and ``padding_mbar`` are always overridden per sweep row.
    latex_grids : list of int, optional
        Subset of grid_sizes to include in LaTeX tables.
        Markdown tables always include all grid sizes.
    """
    extra_calib = dict(calib_overrides or {})
    extra_config = dict(config_overrides or {})
    base_cal, base_settings = _load_baseline()
    base_cal.update(extra_calib)
    base_settings.update(extra_config)

    benchmark_params = {**base_cal, **base_settings,
                        'true_grid_size': true_grid_size,
                        'true_method': true_method}

    # MPI: detect communicator for distributing sweep points
    comm = None
    try:
        from mpi4py import MPI
        _comm = MPI.COMM_WORLD
        if _comm.Get_size() > 1:
            comm = _comm
    except ImportError:
        pass

    is_root = comm is None or comm.Get_rank() == 0

    # ── Pre-compute "true" solutions per delta (all ranks need these) ──
    true_solutions = {}
    for delta in delta_values:
        print(f"\nComputing true solution for delta={delta} "
              f"with {true_grid_size} grid points using {true_method}...")

        cal_ov = {**extra_calib, 'delta': delta}
        cfg_ov = {**extra_config, 'grid_size': true_grid_size,
                  'padding_mbar': -0.011}
        # Warmup
        solve_nest(SYNTAX_DIR, method=true_method,
                   calib_overrides=cal_ov, config_overrides=cfg_ov)
        # Actual run
        nest_true, model_true, _, _ = solve_nest(
            SYNTAX_DIR, method=true_method,
            calib_overrides=cal_ov, config_overrides=cfg_ov)
        true_solutions[delta] = {
            'c_true': get_policy(nest_true, 'c'),
            'a_grid': model_true.asset_grid_A,
        }
        print(f"  True solution computed.")

    # ── Build sweep grid: method × grid_size × delta ──
    grid = param_grid(
        method=list(METHODS),
        grid_size=grid_sizes,
        delta=delta_values,
    )
    # e.g. 4 methods × 15 grids × 4 deltas = 240 points

    # ── solve_fn: ONE method per point ──
    def solve_fn(ov):
        gs = ov['grid_size']
        d = ov['delta']
        method = ov['method']
        c_true = true_solutions[d]['c_true']
        a_grid_true = true_solutions[d]['a_grid']

        nest, model, _, _ = solve_nest(
            SYNTAX_DIR, method=method,
            calib_overrides={**extra_calib, 'delta': d},
            config_overrides={**extra_config, 'grid_size': gs,
                              'padding_mbar': -0.011},
        )
        c_refined = get_policy(nest, 'c')
        timing = get_timing(nest)
        return {
            'ue_time': timing[0],
            'total_time': timing[1],
            'error': euler(model, c_refined),
            'cdev': consumption_deviation(
                model, c_refined, c_true, a_grid_true),
        }

    # ── Metric extractors ──
    metric_fns = {
        'ue_time': lambda r: r['ue_time'],
        'total_time': lambda r: r['total_time'],
        'error': lambda r: r['error'],
        'cdev': lambda r: r['cdev'],
    }

    results = sweep(solve_fn, grid, metric_fns,
                    n_reps=n, warmup=True, best='min', comm=comm)

    # Non-root ranks get [] from gather — nothing to do
    if not results:
        return

    # ── Reshape: flat rows → per-(grid, delta) rows with method columns ──
    # Table generators expect: [grid_size, delta, RFC_val, FUES_val, DCEGM_val, CONSAV_val]
    by_key = {}
    for r in results:
        by_key[(r['grid_size'], r['delta'], r['method'])] = r

    latex_errors_data = []
    latex_timings_data = []
    latex_total_timing_data = []
    latex_cdev_data = []

    for gs in sorted(set(r['grid_size'] for r in results)):
        for d in sorted(set(r['delta'] for r in results)):
            if not all((gs, d, m) in by_key for m in METHODS):
                continue
            latex_errors_data.append([
                gs, d,
                *[by_key[(gs, d, m)]['error'] for m in METHODS],
            ])
            latex_timings_data.append([
                gs, d,
                *[by_key[(gs, d, m)]['ue_time'] * 1000 for m in METHODS],
            ])
            latex_total_timing_data.append([
                gs, d,
                *[by_key[(gs, d, m)]['total_time'] * 1000 for m in METHODS],
            ])
            latex_cdev_data.append([
                gs, d,
                *[by_key[(gs, d, m)]['cdev'] for m in METHODS],
            ])

            print(
                f'\nGrid={gs}, delta={d}:\n'
                f'  Euler errors: '
                + ', '.join(f'{m}: {by_key[(gs, d, m)]["error"]:.6f}' for m in METHODS)
                + f'\n  Cons. dev (log10): '
                + ', '.join(f'{m}: {by_key[(gs, d, m)]["cdev"]:.6f}' for m in METHODS)
                + f'\n  Timings (s): '
                + ', '.join(f'{m}: {by_key[(gs, d, m)]["ue_time"]:.6f}' for m in METHODS)
            )

    # ── Generate tables ──
    generate_timing_table_combined(
        latex_timings_data, latex_total_timing_data,
        "timing", "Retirement model", results_dir,
        params=benchmark_params, latex_grids=latex_grids,
    )
    generate_accuracy_table(
        latex_errors_data, latex_cdev_data,
        "accuracy", "Retirement model", results_dir,
        params=benchmark_params, latex_grids=latex_grids,
    )


if __name__ == "__main__":
    from examples.retirement.run import main
    sys.argv = [
        sys.argv[0], '--run-timings',
        '--sweep-grids', '500',
        '--sweep-deltas', '0.5',
        '--sweep-runs', '1',
        '--output-dir', 'results/retirement',
    ]
    main()
