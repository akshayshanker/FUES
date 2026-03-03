"""Benchmarking functions for retirement model - timing sweeps and comparisons.

Compares FUES vs DC-EGM vs RFC vs CONSAV across grid sizes and delta values.
Uses the canonical pipeline (solve_canonical) for all runs.

Author: Akshay Shanker, University of New South Wales, akshay.shanker@me.com
"""

import numpy as np
import os
import sys
import yaml

# Ensure `dcsmm` is importable when running from a repo checkout.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pathlib import Path
from .solve import solve_canonical
from .outputs import (
    generate_timing_table_combined, generate_accuracy_table,
    plot_egrids, plot_cons_pol, plot_dcegm_cf,
    get_policy, get_timing, get_solution_at_age,
    euler, consumption_deviation,
)

SYNTAX_DIR = Path(__file__).resolve().parent / "syntax"

# Load baseline calibration + settings from syntax dir (cached).
_cal_path = SYNTAX_DIR / "calibration.yaml"
_set_path = SYNTAX_DIR / "settings.yaml"


def _load_baseline():
    """Load baseline calibration and settings from syntax dir."""
    with open(_cal_path) as f:
        cal = yaml.safe_load(f)['calibration']
    with open(_set_path) as f:
        settings = yaml.safe_load(f)['settings']
    return cal, settings


def test_Timings(grid_sizes, delta_values, n=3, results_dir="results",
                 true_grid_size=20000, true_method='DCEGM'):
    """Run timing benchmarks across grid sizes and delta values.

    All runs go through the canonical pipeline (solve_canonical).

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
    """
    base_cal, base_settings = _load_baseline()

    # Build a benchmark params dict for table metadata
    benchmark_params = {**base_cal, **base_settings,
                        'true_grid_size': true_grid_size,
                        'true_method': true_method}

    latex_errors_data = []
    latex_timings_data = []
    latex_total_timing_data = []
    latex_cdev_data = []

    # Pre-compute "true" solutions for each delta value
    true_solutions = {}
    for delta in delta_values:
        print(f"\nComputing true solution for delta={delta} "
              f"with {true_grid_size} grid points using {true_method}...")

        # Warmup
        solve_canonical(
            SYNTAX_DIR, method=true_method,
            calib_overrides={'delta': delta},
            config_overrides={
                'grid_size': true_grid_size,
                'padding_mbar': -0.011,
            },
        )
        # Actual run
        nest_true, model_true, _ = solve_canonical(
            SYNTAX_DIR, method=true_method,
            calib_overrides={'delta': delta},
            config_overrides={
                'grid_size': true_grid_size,
                'padding_mbar': -0.011,
            },
        )
        c_true = get_policy(nest_true, 'c')
        true_solutions[delta] = {
            'c_true': c_true,
            'a_grid': model_true.asset_grid_A,
        }
        print(f"  True solution computed.")

    for g_size in grid_sizes:
        for delta in delta_values:
            print(f"\nTesting with grid size: {g_size} and delta: {delta}")

            c_true = true_solutions[delta]['c_true']
            a_grid_true = true_solutions[delta]['a_grid']

            best = {m: {'time': float('inf'), 'total': float('inf'),
                        'error': float('inf'), 'cdev': float('inf')}
                    for m in ('RFC', 'FUES', 'DCEGM', 'CONSAV')}

            for _ in range(n):
                for method in ('RFC', 'FUES', 'DCEGM', 'CONSAV'):
                    nest, model, _ = solve_canonical(
                        SYNTAX_DIR, method=method,
                        calib_overrides={'delta': delta},
                        config_overrides={
                            'grid_size': g_size,
                            'padding_mbar': -0.011,
                        },
                    )
                    c_refined = get_policy(nest, 'c')
                    timing = get_timing(nest)
                    err = euler(model, c_refined)
                    cdev = consumption_deviation(
                        model, c_refined, c_true, a_grid_true,
                    )

                    best[method]['time'] = min(
                        best[method]['time'], timing[0])
                    best[method]['total'] = min(
                        best[method]['total'], timing[1])
                    best[method]['error'] = min(
                        best[method]['error'], err)
                    best[method]['cdev'] = min(
                        best[method]['cdev'], cdev)

            methods = ('RFC', 'FUES', 'DCEGM', 'CONSAV')
            latex_errors_data.append([
                g_size, delta,
                *[best[m]['error'] for m in methods],
            ])
            latex_timings_data.append([
                g_size, delta,
                *[best[m]['time'] * 1000 for m in methods],
            ])
            latex_total_timing_data.append([
                g_size, delta,
                *[best[m]['total'] * 1000 for m in methods],
            ])
            latex_cdev_data.append([
                g_size, delta,
                *[best[m]['cdev'] for m in methods],
            ])

            print(
                f'Euler errors: '
                + ', '.join(f'{m}: {best[m]["error"]:.6f}'
                            for m in methods)
            )
            print(
                f'Cons. dev (log10): '
                + ', '.join(f'{m}: {best[m]["cdev"]:.6f}'
                            for m in methods)
            )
            print(
                f'Timings (s): '
                + ', '.join(f'{m}: {best[m]["time"]:.6f}'
                            for m in methods)
            )

    # Generate tables
    generate_timing_table_combined(
        latex_timings_data, latex_total_timing_data,
        "timing", "Retirement model", results_dir,
        params=benchmark_params,
    )
    generate_accuracy_table(
        latex_errors_data, latex_cdev_data,
        "accuracy", "Retirement model", results_dir,
        params=benchmark_params,
    )


if __name__ == "__main__":
    grid_sizes = [500, 1000, 2000, 3000, 10000]
    delta_values = [0.25, 0.5, 1, 2]
    egrid_plot_age = 17

    save_path = os.path.join('results', 'plots', 'retirement')
    os.makedirs(save_path, exist_ok=True)

    test_Timings(grid_sizes, delta_values)

    # Generate baseline solution and plots via canonical pipeline
    nest, model, _ = solve_canonical(SYNTAX_DIR, method='RFC')

    results = {}
    for method in ['RFC', 'FUES', 'DCEGM', 'CONSAV']:
        nest, model, _ = solve_canonical(SYNTAX_DIR, method=method)
        results[method] = {
            'nest': nest,
            'c': get_policy(nest, 'c'),
            'timing': get_timing(nest),
            'euler': euler(model, get_policy(nest, 'c')),
        }

    print()
    print("| Method | Euler Error    | Avg UE time(ms) | Total time(ms) |")
    print("|--------|----------------|-----------------|----------------|")
    for method in ['RFC', 'FUES', 'DCEGM', 'CONSAV']:
        r = results[method]
        print(
            f"| {method:<6} | {r['euler']:<14.6f} "
            f"| {r['timing'][0]*1000:<15.3f} | {r['timing'][1]*1000:<14.3f} |"
        )
    print()

    # Generate plots
    nest_rfc = results['RFC']['nest']
    e_grid = get_policy(nest_rfc, 'egrid', stage='work_cons')
    vf_work = get_policy(nest_rfc, 'q_hat', stage='work_cons')
    c_worker = get_policy(nest_rfc, 'c_hat', stage='work_cons')
    dela = get_policy(nest_rfc, 'da_pre_ue', stage='work_cons')

    plot_egrids(
        egrid_plot_age, e_grid, vf_work, c_worker, dela,
        3000, model, save_path, tag='sigma0',
    )
    plot_cons_pol(results['FUES']['c'], model, save_path)
    plot_dcegm_cf(
        egrid_plot_age, 3000, e_grid, vf_work, c_worker,
        dela, model.asset_grid_A, model, save_path,
        tag='sigma0', plot=True,
    )
