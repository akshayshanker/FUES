"""Benchmarking functions for retirement model - timing sweeps and comparisons.

Compares FUES vs DC-EGM vs RFC vs CONSAV across grid sizes and delta values.

Author: Akshay Shanker, University of New South Wales, akshay.shanker@me.com
"""

import numpy as np
import os
import sys

# Ensure `dcsmm` is importable when running from a repo checkout.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pathlib import Path
from .retirement import Operator_Factory, RetirementModel
from .solve_block import backward_induction
from .helpers import (
    generate_timing_table_combined, generate_accuracy_table,
    plot_egrids, plot_cons_pol, plot_dcegm_cf,
    get_policy, get_timing, get_solution_at_age,
    euler, consumption_deviation,
)

SYNTAX_DIR = Path(__file__).resolve().parent.parent / "syntax" / "syntax"


def test_Timings(grid_sizes, delta_values, n=3, results_dir="results", m_bar=1.2,
                 true_grid_size=20000, true_method='DCEGM'):
    """Run timing benchmarks across grid sizes and delta values.

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
    m_bar : float
        FUES jump detection threshold. Default is 1.2.
    true_grid_size : int
        Grid size for computing "true" reference solution. Default is 20000.
    true_method : str
        Method used for computing "true" reference solution. Default is 'DCEGM'.
        Options: 'RFC', 'FUES', 'DCEGM', 'CONSAV'.
    """
    syntax_dir = SYNTAX_DIR

    # Fixed parameters for the benchmark sweep
    benchmark_params = {
        'r': 0.02,
        'beta': 0.96,
        'T': 50,
        'y': 20,
        'b': 1e-100,
        'grid_max_A': 500,
        'm_bar': m_bar,
        'smooth_sigma': 0,
        'true_grid_size': true_grid_size,
        'true_method': true_method,
    }

    latex_errors_data = []
    latex_timings_data = []
    latex_total_timing_data = []
    latex_cdev_data = []

    # Pre-compute "true" solutions for each delta value
    true_solutions = {}
    for delta in delta_values:
        print(f"\nComputing true solution for delta={delta} with {true_grid_size} grid points using {true_method}...")
        cp_true = RetirementModel(
            r=benchmark_params['r'],
            beta=benchmark_params['beta'],
            delta=delta,
            y=benchmark_params['y'],
            b=benchmark_params['b'],
            grid_max_A=benchmark_params['grid_max_A'],
            grid_size=true_grid_size,
            T=benchmark_params['T'],
            smooth_sigma=benchmark_params['smooth_sigma'],
            m_bar=benchmark_params['m_bar'],
            padding_mbar=-0.011,
        )
        movers_true = Operator_Factory(cp_true)
        # Warm-up run
        backward_induction(cp_true, movers_true, syntax_dir, method=true_method)
        # Actual run
        nest_true = backward_induction(cp_true, movers_true, syntax_dir, method=true_method)
        c_true = get_policy(nest_true, 'c')
        true_solutions[delta] = {
            'c_true': c_true,
            'a_grid': cp_true.asset_grid_A
        }
        print(f"  True solution computed.")

    for g_size_baseline in grid_sizes:
        for delta in delta_values:
            print(f"\nTesting with grid size: {g_size_baseline} and delta: {delta}")

            cp = RetirementModel(
                r=benchmark_params['r'],
                beta=benchmark_params['beta'],
                delta=delta,
                y=benchmark_params['y'],
                b=benchmark_params['b'],
                grid_max_A=benchmark_params['grid_max_A'],
                grid_size=g_size_baseline,
                T=benchmark_params['T'],
                smooth_sigma=benchmark_params['smooth_sigma'],
                m_bar=benchmark_params['m_bar'],
                padding_mbar=-0.011,
            )

            movers = Operator_Factory(cp)

            best_time_RFC = float('inf')
            best_time_FUES = float('inf')
            best_time_DCEGM = float('inf')
            best_time_CONSAV = float('inf')
            best_total_time_RFC = float('inf')
            best_total_time_FUES = float('inf')
            best_total_time_DCEGM = float('inf')
            best_total_time_CONSAV = float('inf')
            best_error_RFC = float('inf')
            best_error_FUES = float('inf')
            best_error_DCEGM = float('inf')
            best_error_CONSAV = float('inf')
            best_cdev_RFC = float('inf')
            best_cdev_FUES = float('inf')
            best_cdev_DCEGM = float('inf')
            best_cdev_CONSAV = float('inf')

            # Get true solution for this delta
            c_true = true_solutions[delta]['c_true']
            a_grid_true = true_solutions[delta]['a_grid']

            for _ in range(n):
                # Test RFC
                nest_RFC = backward_induction(cp, movers, syntax_dir, method='RFC')
                c_refined_RFC = get_policy(nest_RFC, 'c')
                timing_RFC = get_timing(nest_RFC)
                time_end_RFC = timing_RFC[0]
                total_time_RFC = timing_RFC[1]
                Euler_error_RFC = euler(cp, c_refined_RFC)
                cons_dev_RFC = consumption_deviation(cp, c_refined_RFC, c_true, a_grid_true)

                # Test FUES
                nest_FUES = backward_induction(cp, movers, syntax_dir, method='FUES')
                c_refined_FUES = get_policy(nest_FUES, 'c')
                timing_FUES = get_timing(nest_FUES)
                time_end_FUES = timing_FUES[0]
                total_time_FUES = timing_FUES[1]
                Euler_error_FUES = euler(cp, c_refined_FUES)
                cons_dev_FUES = consumption_deviation(cp, c_refined_FUES, c_true, a_grid_true)

                # Test DCEGM
                nest_DCEGM = backward_induction(cp, movers, syntax_dir, method='DCEGM')
                c_refined_DCEGM = get_policy(nest_DCEGM, 'c')
                timing_DCEGM = get_timing(nest_DCEGM)
                time_end_DCEGM = timing_DCEGM[0]
                total_time_DCEGM = timing_DCEGM[1]
                Euler_error_DCEGM = euler(cp, c_refined_DCEGM)
                cons_dev_DCEGM = consumption_deviation(cp, c_refined_DCEGM, c_true, a_grid_true)

                # Test CONSAV
                nest_CONSAV = backward_induction(cp, movers, syntax_dir, method='CONSAV')
                c_refined_CONSAV = get_policy(nest_CONSAV, 'c')
                timing_CONSAV = get_timing(nest_CONSAV)
                time_end_CONSAV = timing_CONSAV[0]
                total_time_CONSAV = timing_CONSAV[1]
                Euler_error_CONSAV = euler(cp, c_refined_CONSAV)
                cons_dev_CONSAV = consumption_deviation(cp, c_refined_CONSAV, c_true, a_grid_true)

                # Take the best of n runs
                best_time_RFC = min(best_time_RFC, time_end_RFC)
                best_time_FUES = min(best_time_FUES, time_end_FUES)
                best_time_DCEGM = min(best_time_DCEGM, time_end_DCEGM)
                best_time_CONSAV = min(best_time_CONSAV, time_end_CONSAV)

                best_total_time_RFC = min(best_total_time_RFC, total_time_RFC)
                best_total_time_FUES = min(best_total_time_FUES, total_time_FUES)
                best_total_time_DCEGM = min(best_total_time_DCEGM, total_time_DCEGM)
                best_total_time_CONSAV = min(best_total_time_CONSAV, total_time_CONSAV)

                best_error_RFC = min(best_error_RFC, Euler_error_RFC)
                best_error_FUES = min(best_error_FUES, Euler_error_FUES)
                best_error_DCEGM = min(best_error_DCEGM, Euler_error_DCEGM)
                best_error_CONSAV = min(best_error_CONSAV, Euler_error_CONSAV)

                best_cdev_RFC = min(best_cdev_RFC, cons_dev_RFC)
                best_cdev_FUES = min(best_cdev_FUES, cons_dev_FUES)
                best_cdev_DCEGM = min(best_cdev_DCEGM, cons_dev_DCEGM)
                best_cdev_CONSAV = min(best_cdev_CONSAV, cons_dev_CONSAV)

            latex_errors_data.append([
                g_size_baseline, delta, best_error_RFC,
                best_error_FUES, best_error_DCEGM, best_error_CONSAV
            ])
            latex_timings_data.append([
                g_size_baseline, delta, best_time_RFC * 1000,
                best_time_FUES * 1000, best_time_DCEGM * 1000, best_time_CONSAV * 1000
            ])
            latex_total_timing_data.append([
                g_size_baseline, delta, best_total_time_RFC * 1000,
                best_total_time_FUES * 1000, best_total_time_DCEGM * 1000,
                best_total_time_CONSAV * 1000
            ])
            latex_cdev_data.append([
                g_size_baseline, delta, best_cdev_RFC,
                best_cdev_FUES, best_cdev_DCEGM, best_cdev_CONSAV
            ])

            print(
                f'Euler errors: RFC: {best_error_RFC:.6f}, FUES: {best_error_FUES:.6f}, '
                f'DCEGM: {best_error_DCEGM:.6f}, CONSAV: {best_error_CONSAV:.6f}'
            )
            print(
                f'Cons. dev (log10): RFC: {best_cdev_RFC:.6f}, FUES: {best_cdev_FUES:.6f}, '
                f'DCEGM: {best_cdev_DCEGM:.6f}, CONSAV: {best_cdev_CONSAV:.6f}'
            )
            print(
                f'Timings (s): RFC: {best_time_RFC:.6f}, FUES: {best_time_FUES:.6f}, '
                f'DCEGM: {best_time_DCEGM:.6f}, CONSAV: {best_time_CONSAV:.6f}'
            )

    # Generate tables with sub-columns
    generate_timing_table_combined(latex_timings_data, latex_total_timing_data,
                                   "timing", "Retirement model", results_dir,
                                   params=benchmark_params)
    generate_accuracy_table(latex_errors_data, latex_cdev_data,
                            "accuracy", "Retirement model", results_dir,
                            params=benchmark_params)


if __name__ == "__main__":
    grid_sizes = [500, 1000, 2000, 3000, 10000]
    delta_values = [0.25, 0.5, 1, 2]
    egrid_plot_age = 17
    run_performance_tests = False

    save_path = os.path.join('results', 'plots', 'retirement')
    os.makedirs(save_path, exist_ok=True)

    if run_performance_tests:
        test_Timings(grid_sizes, delta_values)

    # Generate baseline solution and plots
    g_size_baseline = 3000

    cp = RetirementModel(
        r=0.02, beta=0.98, delta=1, y=20, b=1E-10, grid_max_A=500,
        grid_size=3000, T=20, smooth_sigma=0
    )

    syntax_dir = SYNTAX_DIR
    stage_ops = Operator_Factory(cp)

    # Precompile and run
    backward_induction(cp, stage_ops, syntax_dir, method='RFC')

    results = {}
    for method in ['RFC', 'FUES', 'DCEGM', 'CONSAV']:
        nest = backward_induction(cp, stage_ops, syntax_dir, method=method)
        results[method] = {
            'nest': nest,
            'c': get_policy(nest, 'c'),
            'timing': get_timing(nest),
            'euler': euler(cp, get_policy(nest, 'c')),
        }

    print()
    print("| Method | Euler Error    | Avg UE time(ms) | Total time(ms) |")
    print("|--------|----------------|-----------------|----------------|")
    for method in ['RFC', 'FUES', 'DCEGM', 'CONSAV']:
        r = results[method]
        print(f"| {method:<6} | {r['euler']:<14.6f} | {r['timing'][0]*1000:<15.3f} | {r['timing'][1]*1000:<14.3f} |")
    print()

    # Generate plots — read arrays from nest solutions
    nest_rfc = results['RFC']['nest']
    sol_age = get_solution_at_age(nest_rfc, egrid_plot_age)
    e_grid = get_policy(nest_rfc, 'egrid', stage='work_cons')
    vf_work = get_policy(nest_rfc, 'q_hat', stage='work_cons')
    c_worker = get_policy(nest_rfc, 'c_hat', stage='work_cons')
    dela = get_policy(nest_rfc, 'da_pre_ue', stage='work_cons')

    plot_egrids(egrid_plot_age, e_grid, vf_work, c_worker,
                dela, g_size_baseline, cp, save_path, tag='sigma0')

    plot_cons_pol(results['FUES']['c'], cp, save_path)

    plot_dcegm_cf(egrid_plot_age, g_size_baseline, e_grid, vf_work,
                  c_worker, dela, cp.asset_grid_A, cp, save_path,
                  tag='sigma0', plot=True)
