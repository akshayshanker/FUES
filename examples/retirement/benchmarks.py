"""Benchmarking functions for retirement model - timing sweeps and comparisons.

Compares FUES vs DC-EGM vs RFC vs CONSAV across grid sizes and delta values.

Author: Akshay Shanker, University of New South Wales, akshay.shanker@me.com
"""

import numpy as np
import os

from dc_smm.models.retirement.retirement import Operator_Factory, RetirementModel, euler, consumption_deviation
from .tables import generate_timing_table_combined, generate_accuracy_table
from .plots import plot_egrids, plot_cons_pol, plot_dcegm_cf


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
    latex_cons_dev_data = []

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
        _, _, iter_bell_true = Operator_Factory(cp_true)
        # Warm-up run
        _ = iter_bell_true(cp_true, method=true_method)
        # Actual run
        _, _, _, _, c_true, _, _ = iter_bell_true(cp_true, method=true_method)
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

            Ts_ret, Ts_work, iter_bell = Operator_Factory(cp)

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
            best_cons_dev_RFC = float('inf')
            best_cons_dev_FUES = float('inf')
            best_cons_dev_DCEGM = float('inf')
            best_cons_dev_CONSAV = float('inf')

            # Get true solution for this delta
            c_true = true_solutions[delta]['c_true']
            a_grid_true = true_solutions[delta]['a_grid']

            for _ in range(n):
                # Test RFC
                _, _, _, _, c_refined_RFC, _, iter_time_age = iter_bell(cp, method='RFC')
                time_end_RFC = np.mean(iter_time_age[0])
                total_time_RFC = iter_time_age[1]
                Euler_error_RFC = euler(cp, c_refined_RFC)
                cons_dev_RFC = consumption_deviation(cp, c_refined_RFC, c_true, a_grid_true)

                # Test FUES
                _, _, _, _, c_refined_FUES, _, iter_time_age = iter_bell(cp, method='FUES')
                time_end_FUES = np.mean(iter_time_age[0])
                total_time_FUES = iter_time_age[1]
                Euler_error_FUES = euler(cp, c_refined_FUES)
                cons_dev_FUES = consumption_deviation(cp, c_refined_FUES, c_true, a_grid_true)

                # Test DCEGM
                _, _, _, _, c_refined_DCEGM, _, iter_time_age = iter_bell(cp, method='DCEGM')
                time_end_DCEGM = np.mean(iter_time_age[0])
                total_time_DCEGM = iter_time_age[1]
                Euler_error_DCEGM = euler(cp, c_refined_DCEGM)
                cons_dev_DCEGM = consumption_deviation(cp, c_refined_DCEGM, c_true, a_grid_true)

                # Test CONSAV
                _, _, _, _, c_refined_CONSAV, _, iter_time_age = iter_bell(cp, method='CONSAV')
                time_end_CONSAV = np.mean(iter_time_age[0])
                total_time_CONSAV = iter_time_age[1]
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

                best_cons_dev_RFC = min(best_cons_dev_RFC, cons_dev_RFC)
                best_cons_dev_FUES = min(best_cons_dev_FUES, cons_dev_FUES)
                best_cons_dev_DCEGM = min(best_cons_dev_DCEGM, cons_dev_DCEGM)
                best_cons_dev_CONSAV = min(best_cons_dev_CONSAV, cons_dev_CONSAV)

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
            latex_cons_dev_data.append([
                g_size_baseline, delta, best_cons_dev_RFC,
                best_cons_dev_FUES, best_cons_dev_DCEGM, best_cons_dev_CONSAV
            ])

            print(
                f'Euler errors: RFC: {best_error_RFC:.6f}, FUES: {best_error_FUES:.6f}, '
                f'DCEGM: {best_error_DCEGM:.6f}, CONSAV: {best_error_CONSAV:.6f}'
            )
            print(
                f'Cons. dev (log10): RFC: {best_cons_dev_RFC:.6f}, FUES: {best_cons_dev_FUES:.6f}, '
                f'DCEGM: {best_cons_dev_DCEGM:.6f}, CONSAV: {best_cons_dev_CONSAV:.6f}'
            )
            print(
                f'Timings (s): RFC: {best_time_RFC:.6f}, FUES: {best_time_FUES:.6f}, '
                f'DCEGM: {best_time_DCEGM:.6f}, CONSAV: {best_time_CONSAV:.6f}'
            )

    # Generate tables with sub-columns
    generate_timing_table_combined(latex_timings_data, latex_total_timing_data,
                                   "timing", "Retirement model", results_dir,
                                   params=benchmark_params)
    generate_accuracy_table(latex_errors_data, latex_cons_dev_data,
                            "accuracy", "Retirement model", results_dir,
                            params=benchmark_params)


if __name__ == "__main__":
    grid_sizes = [500, 1000, 2000, 3000, 10000]
    delta_values = [0.25, 0.5, 1, 2]
    egrid_plot_age = 17
    run_performance_tests = True

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

    Ts_ret, Ts_work, iter_bell = Operator_Factory(cp)

    # Precompile and run
    _ = iter_bell(cp, method='RFC')
    e_grid_worker_unref, vf_work_unref, vf_refined, c_worker_unref, \
        c_refined_RFC, dela_unrefined, time_end_RFC = iter_bell(cp, method='RFC')

    _ = iter_bell(cp, method='FUES')
    _, _, _, _, c_refined_FUES, _, time_end_FUES = iter_bell(cp, method='FUES')

    _ = iter_bell(cp, method='DCEGM')
    _, _, _, _, c_refined_DCEGM, _, time_end_DCEGM = iter_bell(cp, method='DCEGM')

    _ = iter_bell(cp, method='CONSAV')
    _, _, _, _, c_refined_CONSAV, _, time_end_CONSAV = iter_bell(cp, method='CONSAV')

    Euler_error_RFC = euler(cp, c_refined_RFC)
    Euler_error_FUES = euler(cp, c_refined_FUES)
    Euler_error_DCEGM = euler(cp, c_refined_DCEGM)
    Euler_error_CONSAV = euler(cp, c_refined_CONSAV)

    print()
    print("| Method | Euler Error    | Avg UE time(ms) | Total time(ms) |")
    print("|--------|----------------|-----------------|----------------|")
    print(f"| RFC    | {Euler_error_RFC:<14.6f} | {time_end_RFC[0]*1000:<15.3f} | {time_end_RFC[1]*1000:<14.3f} |")
    print(f"| FUES   | {Euler_error_FUES:<14.6f} | {time_end_FUES[0]*1000:<15.3f} | {time_end_FUES[1]*1000:<14.3f} |")
    print(f"| DCEGM  | {Euler_error_DCEGM:<14.6f} | {time_end_DCEGM[0]*1000:<15.3f} | {time_end_DCEGM[1]*1000:<14.3f} |")
    print(f"| CONSAV | {Euler_error_CONSAV:<14.6f} | {time_end_CONSAV[0]*1000:<15.3f} | {time_end_CONSAV[1]*1000:<14.3f} |")
    print()

    # Generate plots
    plot_egrids(egrid_plot_age, e_grid_worker_unref, vf_work_unref, c_worker_unref,
                dela_unrefined, g_size_baseline, cp, save_path, tag='sigma0')

    plot_cons_pol(c_refined_FUES, cp, save_path)

    plot_dcegm_cf(egrid_plot_age, g_size_baseline, e_grid_worker_unref, vf_work_unref,
                  c_worker_unref, dela_unrefined, cp.asset_grid_A, cp, save_path,
                  tag='sigma0', plot=True)
