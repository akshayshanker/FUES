"""
Author: Akshay Shanker, University of New South Wales, 
a.shanker@unsw.edu.au

Script to plot NEGM and RFC solutions for Application 3 in Dobrescu and
Shanker (2022) (FUES). Model with a durable and non-durable continuous asset. 

Todo
----
- Fix the way results are stored for the infinite horizon case. 
- Make the Euler error "truly" infinite horizon.
"""

import numpy as np
import time
import yaml
from interpolation.splines import UCGrid, eval_linear
from interpolation.splines import extrap_options as xto
from mpi4py import MPI
import os
import sys
import dill as pickle
import matplotlib.pyplot as plt

# Import local modules
cwd = os.getcwd()
sys.path.append('..')
os.chdir(cwd)
from examples.durables.durables import ConsumerProblem, Operator_Factory
from examples.durables.plot import plot_pols, plot_grids, plot_wage_profile
from examples.durables.simulate import euler_errors, compute_euler_stats, print_euler_stats
from dcsmm.fues.helpers.math_funcs import f, interp_as

# Plot styling (matching durable-cons/figs.py)
plt.rcParams.update({"axes.grid": True, "grid.color": "black", "grid.alpha": "0.25", "grid.linestyle": "--"})
plt.rcParams.update({'font.size': 12})


def plot_lifecycle_compare(sim_egm, sim_negm, cp, output_dir=None):
    """
    Plot lifecycle comparison of mean paths for EGM/FUES vs NEGM.

    Parameters
    ----------
    sim_egm : dict
        Simulation data from EGM/FUES method.
    sim_negm : dict
        Simulation data from NEGM method.
    cp : ConsumerProblem
        The consumer problem instance.
    output_dir : str, optional
        Directory to save plots. If None, uses FUES_OUTPUT_DIR or current dir.
    """
    if output_dir is None:
        output_dir = os.environ.get('FUES_OUTPUT_DIR', '.')

    t0 = cp.t0
    T = cp.T

    # Variables to plot: (key, label, ylabel)
    varlist = [
        ('y', 'Income', '$y_t$'),
        ('c', 'Consumption', '$c_t$'),
        ('h', 'Housing', '$h_t$'),
        ('a', 'Liquid Assets', '$a_t$'),
    ]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    age = np.arange(t0, T)

    for i, (var, title, ylabel) in enumerate(varlist):
        ax = axes[i]

        # Get data for valid time range
        data_egm = sim_egm[var][t0:T, :]
        data_negm = sim_negm[var][t0:T, :]

        # Compute means (ignore zeros from uninitialized periods)
        # Use nanmean to handle any nan values
        mean_egm = np.nanmean(data_egm, axis=1)
        mean_negm = np.nanmean(data_negm, axis=1)

        # Plot means
        ax.plot(age, mean_egm, lw=2, label='EGM/FUES', color='C0')
        ax.plot(age, mean_negm, lw=2, label='NEGM', color='C1', linestyle='--')

        # Add percentile bands (25th-75th)
        p25_egm = np.nanpercentile(data_egm, 25, axis=1)
        p75_egm = np.nanpercentile(data_egm, 75, axis=1)
        ax.fill_between(age, p25_egm, p75_egm, alpha=0.2, color='C0')

        p25_negm = np.nanpercentile(data_negm, 25, axis=1)
        p75_negm = np.nanpercentile(data_negm, 75, axis=1)
        ax.fill_between(age, p25_negm, p75_negm, alpha=0.2, color='C1')

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        ax.grid(True)

        # Set x-axis ticks
        if T - t0 > 10:
            ax.set_xticks(age[::5])
        else:
            ax.set_xticks(age)

    # Add x-axis label to bottom plots
    axes[2].set_xlabel('Age')
    axes[3].set_xlabel('Age')

    # Add adjuster share subplot (replace housing with adjuster share comparison)
    # Actually, let's add a 5th plot for adjuster share
    fig.tight_layout()

    # Save figure
    plot_path = os.path.join(output_dir, 'plots', 'lifecycle_comparison.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Lifecycle comparison plot saved to: {plot_path}")

    # Also create adjuster share plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    discrete_egm = sim_egm['discrete'][t0:T, :]
    discrete_negm = sim_negm['discrete'][t0:T, :]

    adj_rate_egm = np.mean(discrete_egm, axis=1)
    adj_rate_negm = np.mean(discrete_negm, axis=1)

    ax2.plot(age, adj_rate_egm, lw=2, label='EGM/FUES', color='C0')
    ax2.plot(age, adj_rate_negm, lw=2, label='NEGM', color='C1', linestyle='--')

    ax2.set_title('Adjustment Rate')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Fraction Adjusting')
    ax2.legend(loc='best')
    ax2.grid(True)

    if T - t0 > 10:
        ax2.set_xticks(age[::5])
    else:
        ax2.set_xticks(age)

    plot_path2 = os.path.join(output_dir, 'plots', 'lifecycle_adjuster_rate.png')
    fig2.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Adjuster rate plot saved to: {plot_path2}")


def _get_avg_time(res) -> float:
    """Robustly extract avg_time from solver result dicts.

    Some solvers store timing at top-level (e.g. results['avg_time']),
    while others store it under results[0]['avg_time'] (EGM/NEGM here).
    """
    if res is None:
        return float("nan")
    if isinstance(res, dict):
        if "avg_time" in res:
            return res["avg_time"]
        if 0 in res and isinstance(res[0], dict) and "avg_time" in res[0]:
            return res[0]["avg_time"]
    return float("nan")


def _get_avg_keeper_ms(res) -> float:
    """Extract average keeper time in ms from solver result dicts."""
    if res is None:
        return float("nan")
    if isinstance(res, dict):
        if "avg_keeper_ms" in res:
            return res["avg_keeper_ms"]
        if 0 in res and isinstance(res[0], dict) and "avg_keeper_ms" in res[0]:
            return res[0]["avg_keeper_ms"]
    return float("nan")


def _get_avg_adj_ms(res) -> float:
    """Extract average adjuster time in ms from solver result dicts."""
    if res is None:
        return float("nan")
    if isinstance(res, dict):
        if "avg_adj_ms" in res:
            return res["avg_adj_ms"]
        if 0 in res and isinstance(res[0], dict) and "avg_adj_ms" in res[0]:
            return res[0]["avg_adj_ms"]
    return float("nan")

def euler_housing(results, cp):
    """
    Compute Euler errors for the housing model.

    Parameters
    ----------
    results : dict
        Results from the solver, containing policy functions.
    cp : ConsumerProblem
        The consumer problem instance.

    Returns
    -------
    euler : np.array
        Euler errors across the state space.
    """
    ug_grid_all = UCGrid((cp.b, cp.grid_max_A, cp.grid_size_A),
                         (cp.b, cp.grid_max_H, len(cp.asset_grid_H)))

    euler = np.zeros((cp.T, len(cp.z_vals), len(cp.asset_grid_A),
                      len(cp.asset_grid_H)))
    euler.fill(np.nan)
    cpdiff = np.zeros(euler.shape)
    cpdiff.fill(np.nan)
    a_grid = cp.asset_grid_A
    h_grid = cp.asset_grid_H
    asset_grid_WE = cp.asset_grid_WE
    y_func = cp.y_func
    R = cp.R
    R_H = cp.R_H
    delta = cp.delta

    if cp.stat:
        t0  = min(int(k) for k in results.keys() if int(k) > 0)
        T = cp.T - 1
    else:
        t0 = cp.t0
        T = cp.T-1

    # Loop over grid
    for t in range(t0, T):
        for i_h in range(len(h_grid)):
            for i_a in range(len(a_grid)):
                for i_z in range(len(cp.z_vals)):
                    h = h_grid[i_h]
                    a = a_grid[i_a]
                    z = cp.z_vals[i_z]
                    D_adj = results[t]['D'][i_z, i_a, i_h]

                    D_adj = 1 # solve only adjuster error. 

                    if D_adj == 1:
                        wealth = R * a + R_H * h * (1 - delta) + y_func(t, z)
                        a_prime = interp_as(asset_grid_WE,
                                            results[t]['Aadj'][i_z, :],
                                            np.array([wealth]))[0]
                        hnxt = interp_as(asset_grid_WE,
                                         results[t]['Hadj'][i_z, :],
                                         np.array([wealth]))[0]
                        c = interp_as(asset_grid_WE,
                                      results[t]['Cadj'][i_z, :],
                                      np.array([wealth]))[0]
                    if D_adj == 0:
                        wealth = R * a + y_func(t, z)
                        hnxt = h
                        a_prime = interp_as(a_grid,
                                            results[t]['Akeeper'][i_z, :, i_h],
                                            np.array([wealth]))[0]
                        c = interp_as(a_grid,
                                      results[t]['Ckeeper'][i_z, :, i_h],
                                      np.array([wealth]))[0]

                    if a_prime <= 0.1 or c <= 0.1:
                        continue

                    rhs = 0
                    lambda_h_plus = 0
                    for i_eta in range(len(cp.z_vals)):
                        c_plus = eval_linear(ug_grid_all,
                                             results[t + 1]['C'][i_z],
                                             np.array([a_prime, hnxt]),
                                             xto.LINEAR)
                        ##a_plus = eval_linear(ug_grid_all,
                        #                     results[t + 1]['A'][i_z],
                        #                     np.array([a_prime, hnxt]),
                       #                      xto.LINEAR)
                        
                        D_plus = eval_linear(ug_grid_all,
                                             results[t + 1]['D'][i_z],
                                             np.array([a_prime, hnxt]),
                                             xto.LINEAR)
                        D_plus = int(min(max(D_plus, 0), 1))

                        if D_plus == 1:

                            #h_plus = eval_linear(ug_grid_all,
                            #                     results[t + 1]['H'][i_eta],
                            #                     np.array([a_prime, hnxt]),
                            #                     xto.LINEAR)
                            wealth_plus = R * a_prime + R_H * hnxt * (1 - delta) + y_func(t + 1, cp.z_vals[i_eta])
                            c_plus_adj = interp_as(asset_grid_WE,
                                                   results[t + 1]['Cadj'][i_z, :],
                                                   np.array([wealth_plus]))[0]
                            lambda_h_plus = lambda_h_plus + cp.Pi[i_z, i_eta] * cp.beta * (1 - cp.delta) * cp.du_c(c_plus_adj)
                            
                            
                        else:
                            wealth_plus = R * a_prime + y_func(t + 1, cp.z_vals[i_eta])
                            a_plus = interp_as(a_grid,
                                               results[t + 1]['Akeeper'][i_z, :, i_h],
                                               np.array([wealth_plus]))[0]

                            lambda_h_plus = lambda_h_plus + cp.Pi[i_z, i_eta] * cp.beta * (1 - cp.delta) * (eval_linear(ug_grid_all,
                                                                         results[t + 2]['ELambdaHnxt'][i_eta],
                                                                         np.array([a_plus, hnxt]),
                                                                         xto.LINEAR)
                                                            + cp.du_h(hnxt))
                        
                        
                        rhs += cp.Pi[i_z, i_eta] * cp.beta * cp.R * \
                            cp.du_c(c_plus)

                    lhs = (cp.du_h(hnxt) + lambda_h_plus) / (1 + cp.tau)
                    euler_raw2 = c - cp.du_c_inv(lhs)
                    euler[t, i_z, i_a, i_h] = np.log10(np.abs(
                        euler_raw2 / c) + 1e-16)
    return euler

def timing(solver, cp, rep=4, do_print=False, N_sim=None):
    """
    Run the solver and track the best time, average Euler error,
    and number of iterations to convergence.

    Parameters
    ----------
    solver : callable
        The solver function to use (e.g., solveNEGM, solveEGM).
    cp : ConsumerProblem
        The consumer problem instance.
    rep : int, optional
        Number of repetitions to average results over, default is 4.
    do_print : bool, optional
        Whether to print intermediate results, default is False.
    N_sim : int, optional
        Number of simulated individuals for Euler errors.
        If None, uses cp.N_sim from settings.

    Returns
    -------
    dict
        The solution with the best time, Euler errors, and iteration count.
    """
    # Use cp.N_sim if N_sim not specified
    if N_sim is None:
        N_sim = getattr(cp, 'N_sim', 10000)

    time_best = np.inf
    iter_best = None

    for i in range(rep):
        solution = solver(cp)

        # Compute Euler errors using simulation
        euler, euler_hd, sim_data = euler_errors(solution, cp, N=N_sim, seed=42+i)
        stats = compute_euler_stats(euler, discrete=sim_data['discrete'])
        stats_hd = compute_euler_stats(euler_hd, discrete=sim_data['discrete'])

        tot_time = solution[0]['avg_time']
        iterations = solution[0].get('iterations', None)

        if do_print:
            print(f'\n--- Run {i+1}/{rep} ---')
            print(f'Time: {tot_time:.2f} secs, Iterations: {iterations}')
            print(f'Euler errors (log10):')
            print(f'  Combined:  mean={stats["combined"]["mean"]:.4f}, '
                  f'median={stats["combined"]["median"]:.4f}')
            print(f'  Adjuster:  mean={stats["adjuster"]["mean"]:.4f}, '
                  f'median={stats["adjuster"]["median"]:.4f} '
                  f'(n={stats["adjuster"]["n_obs"]})')
            print(f'  Keeper:    mean={stats["keeper"]["mean"]:.4f}, '
                  f'median={stats["keeper"]["median"]:.4f} '
                  f'(n={stats["keeper"]["n_obs"]})')
            print(f'  Adj. rate: {stats["pct_adjuster"]:.2f}%')

        if tot_time < time_best:
            time_best = tot_time
            model_best = solution
            iter_best = iterations
            euler_best = euler
            euler_hd_best = euler_hd
            sim_data_best = sim_data
            stats_best = stats
            stats_hd_best = stats_hd

    # Store results in the best model
    model_best['euler'] = euler_best
    model_best['euler_hd'] = euler_hd_best
    model_best['euler_stats'] = stats_best
    model_best['euler_stats_hd'] = stats_hd_best
    model_best['sim_data'] = sim_data_best
    model_best['iterations'] = iter_best

    return model_best
        

def initVal(cp):
    shape = (len(cp.z_vals), len(cp.asset_grid_A), len(cp.asset_grid_H))
    EVnxt = np.empty(shape)
    ELambdaAnxt = np.empty(shape)
    ELambdaHnxt = np.empty(shape)

    for state in range(len(cp.X_all)):
        a = cp.asset_grid_A[cp.X_all[state][1]]
        h = cp.asset_grid_H[cp.X_all[state][2]]
        i_a = cp.X_all[state][1]
        i_h = cp.X_all[state][2]
        i_z = cp.X_all[state][0]
        z = cp.z_vals[i_z]

        EVnxt[i_z, i_a, i_h] = cp.term_u(
            cp.R_H * (1 - cp.delta) * h + cp.R * a)
        ELambdaAnxt[i_z, i_a, i_h] = cp.beta * cp.R * cp.term_du(
            cp.R_H * (1 - cp.delta) * h + cp.R * a)
        ELambdaHnxt[i_z, i_a, i_h] = cp.beta * cp.R_H * cp.term_du(
            cp.R_H * (1 - cp.delta) * h + cp.R * a) * (1 - cp.delta)

    return EVnxt, ELambdaAnxt, ELambdaHnxt


def solveVFI(cp, verbose=False):
    iterVFI, _, condition_V, _, _ = Operator_Factory(cp)

    # Initial VF
    EVnxt, _, _ = initVal(cp)

    # Start Bellman iteration
    t = cp.T
    results = {}
    times = []
    error = None  # Initialize error

    while t >= cp.t0:
        results[t] = {}

        start = time.time()
        bellman_sol = iterVFI(t, EVnxt)

        Vcurr, Anxt, Hnxt, Cnxt, Aadj, Hadj = bellman_sol

        # Store results
        results[t]["VF"] = Vcurr
        results[t]["C"] = Cnxt
        results[t]["A"] = Anxt
        results[t]["H"] = Hnxt
        results[t]["Hadj"] = Hadj
        results[t]["Aadj"] = Aadj

        # Condition the value function
        EVnxt, _, _ = condition_V(Vcurr, Vcurr, Vcurr)

        # Compute error BEFORE printing (only for non-terminal periods)
        if t < cp.T:
            times.append(time.time() - start)
            error = np.max(np.abs(Vcurr - results[t + 1]["VF"]))

        if verbose:
            print(f"Bellman iteration no. {t}, time is {time.time() - start}")
            if t < cp.T and error is not None:
                print(f"Error at age {t} is {error}")

        t = t - 1

    results['avg_time'] = np.mean(times) if times else 0.0

    return results

def solveEGM(cp, LS=True, verbose=True, plot_age=58):
    _, iterEGM, condition_V, condition_V_HD, _ = Operator_Factory(cp)

    # Initial values
    EVnxt, ELambdaAnxt, ELambdaHnxt = initVal(cp)

    # HD Lambda initialization (None for terminal period, will be handled by iterEGM)
    ELambdaH_HD_nxt = None

    t = cp.T
    times = []
    keeper_times = []
    adj_times = []
    results = {}
    results[0] = {}

    error = 1

    while t >= cp.t0 and error > cp.tol_bel:
        start = time.time()

        (Vcurr, Hnxt, Cnxt, Dnxt, LambdaAcurr, LambdaHcurr, LambdaH_HD_curr, AdjPol,
         KeeperPol, EGMGrids, timing) = iterEGM(t, EVnxt, ELambdaAnxt,
                                        ELambdaHnxt, ELambdaH_HD_nxt, m_bar=cp.m_bar, plot_age=plot_age,
                                        verbose_timing=verbose)

        EVnxt, ELambdaAnxt, ELambdaHnxt = condition_V(
                                            Vcurr, LambdaAcurr, LambdaHcurr
        )
        # Condition HD Lambda for next iteration and store conditioned version
        if cp.N_HD_LAMBDA > 1:
            ELambdaH_HD_nxt = condition_V_HD(LambdaH_HD_curr)
            # Store conditioned Lambda: E[Lambda_H(t)|z] for simulation
            ELambdaH_HD_store = ELambdaH_HD_nxt
        else:
            ELambdaH_HD_store = LambdaH_HD_curr  # Dummy when not using HD

        if t < cp.T:
            times.append(time.time() - start)
            keeper_times.append(timing['keeper_ms'])
            adj_times.append(timing['adj_ms'])

        results[t] = {}
        results[t]["D"] = Dnxt
        results[t]["C"] = Cnxt
        results[t]["H"] = Hnxt
        results[t]["VF"] = Vcurr
        results[t]["Hadj"] = AdjPol['H']
        results[t]["Aadj"] = AdjPol['A']
        results[t]["Cadj"] = AdjPol['C']
        results[t]["Vadj"] = KeeperPol['V']
        results[t]["Ckeeper"] = KeeperPol['C']
        results[t]["Akeeper"] = KeeperPol['A']
        results[t]["EGMGrids"] = EGMGrids
        results[t]["ELambdaHnxt"] = ELambdaHnxt
        results[t]["ELambdaH_HD"] = ELambdaH_HD_store  # Conditioned HD Lambda for simulation

        if t < cp.T:
            error = np.max(np.abs(Vcurr - results[t + 1]["VF"]))

        if verbose:
            print(f"EGM age {t}, time is {time.time() - start}")
            if t< cp.T: print(f"Error at age {t} is {error}")

        t = t - 1

    results[0]['avg_time'] = np.mean(times)
    results[0]['avg_keeper_ms'] = np.mean(keeper_times) if keeper_times else 0.0
    results[0]['avg_adj_ms'] = np.mean(adj_times) if adj_times else 0.0
    results[0]['iterations'] = cp.T1 -t

    return results

def solveNEGM(cp, LS=True, verbose=True):
    iterVFI, iterEGM, condition_V, condition_V_HD, iterNEGM = Operator_Factory(cp)

    # Initial values
    EVnxt, ELambdaAnxt, ELambdaHnxt = initVal(cp)

    # HD Lambda initialization (None for terminal period, will be handled by iterNEGM)
    ELambdaH_HD_nxt = None

    # Results dictionaries
    results = {}
    results[0] = {}
    times = []
    keeper_times = []
    adj_times = []
    t = cp.T
    error = 1
    while t >= cp.t0 and error > cp.tol_bel:
        results[t] = {}
        start = time.time()

        (Vcurr, Hnxt, Cnxt, Dnxt, LambdaAcurr, LambdaHcurr, LambdaH_HD_curr, AdjPol, KeeperPol, timing) = iterNEGM(
            EVnxt, ELambdaAnxt, ELambdaHnxt, t, ELambdaH_HD_nxt, verbose_timing=verbose)

        EVnxt, ELambdaAnxt, ELambdaHnxt = condition_V(
            Vcurr, LambdaAcurr, LambdaHcurr)
        # Condition HD Lambda for next iteration and store conditioned version
        if cp.N_HD_LAMBDA > 1:
            ELambdaH_HD_nxt = condition_V_HD(LambdaH_HD_curr)
            # Store conditioned Lambda: E[Lambda_H(t)|z] for simulation
            ELambdaH_HD_store = ELambdaH_HD_nxt
        else:
            ELambdaH_HD_store = LambdaH_HD_curr  # Dummy when not using HD

        if t < cp.T:
            times.append(time.time() - start)
            keeper_times.append(timing['keeper_ms'])
            adj_times.append(timing['adj_ms'])

        results[t] = {}
        results[t]["D"] = Dnxt
        results[t]["C"] = Cnxt
        results[t]["H"] = Hnxt
        results[t]["VF"] = Vcurr
        results[t]["Hadj"] = AdjPol['H']
        results[t]["Aadj"] = AdjPol['A']
        results[t]["Cadj"] = AdjPol['C']
        results[t]["Vadj"] = KeeperPol['V']
        results[t]["Ckeeper"] = KeeperPol['C']
        results[t]["Akeeper"] = KeeperPol['A']
        results[t]["ELambdaHnxt"] = ELambdaHnxt
        results[t]["ELambdaH_HD"] = ELambdaH_HD_store  # Conditioned HD Lambda for simulation

        if t < cp.T:
            error = np.max(np.abs(Vcurr - results[t + 1]["VF"]))

        if verbose:
            print(f"NEGM age {t}, time {time.time() - start}")
            if t< cp.T: print(f"Error at age {t} is {error}")

        t = t - 1

    results[0]['avg_time'] = np.mean(times)
    results[0]['avg_keeper_ms'] = np.mean(keeper_times) if keeper_times else 0.0
    results[0]['avg_adj_ms'] = np.mean(adj_times) if adj_times else 0.0
    results[0]['iterations'] = cp.T1 -t

    return results

def compare_grids_and_tau(cp_settings, tau_values, grid_sizes, max_iter=200, tol=1e-03, rep=1, methods=['NEGM', 'FUES']):
    """
    Compare the performance of FUES and NEGM over different grid sizes and tau values using MPI.

    Parameters
    ----------
    cp_settings : dict
        Settings for the consumer problem.
    tau_values : list
        List of `tau` values to compare.
    grid_sizes : list
        List of grid sizes (same for assets and housing) to compare.
    max_iter : int, optional
        Maximum number of iterations for the solver, default is 200.
    tol : float, optional
        Tolerance for convergence, default is 1e-03.
    rep : int, optional
        Number of repetitions to average results over, default is 4.
    methods : list
        Methods to compare, default is ['DCEGM', 'FUES'].

    Returns
    -------
    results_summary : list
        A list containing performance metrics for each grid size, tau, and method.
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get the rank (ID) of the current process
    size = comm.Get_size()  # Get the total number of processes
    print(f"[Rank {rank}] Inside compare_grids_and_tau, size={size}", flush=True)

    # Distribute work based on rank
    total_combinations = len(tau_values) * len(grid_sizes) * len(methods)
    combinations_per_process = total_combinations // size  # Split work equally
    leftover_combinations = total_combinations % size  # Handle leftover combinations
    print(f"[Rank {rank}] total_combinations={total_combinations}, per_process={combinations_per_process}", flush=True)

    # Each process works on a subset of combinations
    results_summary = []

    # Flatten the combinations of tau, grid_size, and method for easy distribution
    combinations = [(tau, grid_size, method) for tau in tau_values for grid_size in grid_sizes for method in methods]

    # Get the range of combinations this process will handle
    start_index = rank * combinations_per_process
    end_index = start_index + combinations_per_process
    if rank < leftover_combinations:
        start_index += rank
        end_index += rank + 1
    else:
        start_index += leftover_combinations
        end_index += leftover_combinations

    my_combinations = combinations[start_index:end_index]
    print(f"[Rank {rank}] Handling {len(my_combinations)} combinations: {my_combinations}", flush=True)

    # Each process works on its subset of combinations
    for tau, grid_size, method in my_combinations:
        # Update the tau and grid size in the model parameters
        # Use AR(1) wage process via Tauchen method (Pi and z_vals generated internally)
        # Note: float()/int() conversions needed because YAML may read scientific notation as strings
        cp = ConsumerProblem(cp_settings,
                             r=float(cp_settings['r']),
                             sigma=float(cp_settings['sigma']),
                             r_H=float(cp_settings['r_H']),
                             beta=float(cp_settings['beta']),
                             alpha=float(cp_settings['alpha']),
                             kappa=float(cp_settings.get('kappa', 1.0)),
                             delta=float(cp_settings['delta']),
                             # AR(1) wage process parameters (Tauchen method)
                             phi_w=float(cp_settings.get('phi_w', 0.9170411)),
                             sigma_w=float(cp_settings.get('sigma_w', 0.0817012)),
                             N_wage=int(cp_settings.get('N_wage', 3)),
                             b=float(cp_settings['b']),
                             grid_max_A=float(cp_settings['grid_max_A']),
                             grid_max_WE=float(cp_settings['grid_max_WE']),
                             grid_size_W=int(grid_size),
                             grid_max_H=float(cp_settings['grid_max_H']),
                             grid_size_A=int(grid_size),
                             grid_size_H=int(grid_size),
                             gamma_c=float(cp_settings['gamma_c']),
                             gamma_h=float(cp_settings['gamma_h']),
                             chi=float(cp_settings['chi']),
                             tau=float(tau),
                             K=float(cp_settings['K']),
                             tol_bel=float(cp_settings['tol_bel']),
                             m_bar=float(cp_settings['M_bar']),
                             T=int(cp_settings['max_iter']),
                             theta=float(cp_settings['theta']),
                             t0=int(cp_settings.get('t0', 0)),
                             root_eps=float(cp_settings['root_eps']),
                             stat=cp_settings.get('stat', False)
                             )

        print(f"[Rank {rank}] Solving tau={tau}, grid={grid_size}, method={method}...", flush=True)

        # Timing and Euler error calculations
        best_time = np.inf
        best_iterations = None
        total_runtime = 0
        best_stats = None
        best_stats_hd = None
        best_npv = None
        best_keeper_ms = 0.0
        best_adj_ms = 0.0

        for _ in range(rep):  # Run each method `rep` times
            if method == 'NEGM':
                solution = solveNEGM(cp, verbose=False)
            else:
                solution = solveEGM(cp, verbose=False)

            # Debug: print timing from this run
            print(f"[Rank {rank}] {method} solved: keeper={solution[0].get('avg_keeper_ms', 0.0):.1f}ms, "
                  f"adj={solution[0].get('avg_adj_ms', 0.0):.1f}ms", flush=True)
            # Use simulation-based Euler errors with adjuster/keeper breakdown
            euler, euler_hd, sim_data = euler_errors(solution, cp, N=cp.N_sim, seed=42)
            stats = compute_euler_stats(euler, discrete=sim_data['discrete'])
            stats_hd = compute_euler_stats(euler_hd, discrete=sim_data['discrete'])
            npv = sim_data['npv_utility']
            npv_adj = sim_data['npv_utility_adj']
            n_adj = sim_data['n_adj_periods']

            runtime = solution[0]['avg_time']
            iterations = solution[0]['iterations']
            total_runtime += runtime

            if runtime < best_time:
                best_time = runtime
                best_iterations = iterations
                best_stats = stats
                best_stats_hd = stats_hd
                best_npv = npv
                best_npv_adj = npv_adj
                best_n_adj = n_adj
                # Track keeper/adjuster timing from best run
                best_keeper_ms = solution[0].get('avg_keeper_ms', 0.0)
                best_adj_ms = solution[0].get('avg_adj_ms', 0.0)

        avg_time = total_runtime / rep

        # Compute adjuster-conditional NPV (only agents with at least one adjustment)
        adj_mask = best_n_adj > 0
        npv_adj_mean = np.mean(best_npv_adj[adj_mask]) if np.sum(adj_mask) > 0 else 0.0

        results_summary.append({
            'Tau': tau,
            'Grid_Size': grid_size,
            'Method': method,
            'Avg_Time': avg_time,
            'Avg_Keeper_ms': best_keeper_ms,
            'Avg_Adj_ms': best_adj_ms,
            'Euler_Combined': best_stats['combined']['mean'],
            'Euler_Combined_HD': best_stats_hd['combined']['mean'],
            'Euler_Adjuster': best_stats['adjuster']['mean'],
            'Euler_Adjuster_HD': best_stats_hd['adjuster']['mean'],
            'Euler_Keeper': best_stats['keeper']['mean'],
            'Euler_Keeper_HD': best_stats_hd['keeper']['mean'],
            'NPV_Mean': np.mean(best_npv),
            'NPV_Median': np.median(best_npv),
            'NPV_Adj_Mean': npv_adj_mean,
            'Adj_Rate': best_stats['pct_adjuster'],
            'Iterations': best_iterations,
            # Model parameters for table notes
            'beta': cp.beta,
            'gamma_c': cp.gamma_c,
            'gamma_h': cp.gamma_h,
            'alpha': cp.alpha,
            'delta': cp.delta,
            'r': cp.r,
            'r_H': cp.r_H,
            'phi_w': cp.phi_w,
            'sigma_w': cp.sigma_w,
            'N_sim': cp.N_sim
        })

    # Gather results from all processes
    all_results = comm.gather(results_summary, root=0)

    # If root process, combine and process results
    if rank == 0:
        combined_results = [item for sublist in all_results for item in sublist]

        # Get versioned paths from environment (or use defaults)
        version = os.environ.get('FUES_VERSION', 'default')
        results_dir = os.environ.get('FUES_RESULTS_DIR', '../../results/durables')
        scratch_dir = os.environ.get('FUES_OUTPUT_DIR', '/scratch/tp66/as3442/FUES/durables')

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(scratch_dir, exist_ok=True)

        # Save pkl to scratch (large file)
        pkl_path = os.path.join(scratch_dir, 'durable_timings.pkl')
        with open(pkl_path, 'wb') as file:
            pickle.dump(combined_results, file)
        print(f"Results saved to: {pkl_path}")

        # Generate and save LaTeX table to home results directory
        latex_table = create_latex_table(combined_results)
        latex_path = os.path.join(results_dir, 'durables_table.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {latex_path}")

        return combined_results, latex_table
    else:
        return None, None

def create_latex_table(results_summary):
    """
    Generate a LaTeX table comparing timing and Euler errors for different
    grid sizes and tau values across FUES and NEGM.

    Format matches housing_timing.tex: methods as column groups.
    Shows separate keeper and adjuster timing columns.

    Parameters
    ----------
    results_summary : list
        A list containing performance metrics for each grid size, tau, and method.

    Returns
    -------
    latex_table : str
        A LaTeX-formatted table.
    """
    # Sort results by Grid_Size and Tau
    results_summary = sorted(results_summary, key=lambda x: (x['Grid_Size'], x['Tau']))

    # Get unique grid sizes and tau values
    grid_sizes = sorted(set(r['Grid_Size'] for r in results_summary))
    tau_values = sorted(set(r['Tau'] for r in results_summary))

    # Helper to get result for specific combination
    def get_result(grid_size, tau, method):
        try:
            return next(r for r in results_summary
                       if r['Grid_Size'] == grid_size and r['Tau'] == tau and r['Method'] == method)
        except StopIteration:
            return None

    # Build table with hierarchical headers:
    # Method -> Time (NA, Adj) | Euler (Comb, NA, Adj)
    # NA = Non-Adjuster
    table = "% Requires: \\usepackage{booktabs}\n"
    table += "\\begin{table}[htbp]\n\\centering\n"
    table += "\\begingroup\n\\small\n"
    table += "\\setlength{\\tabcolsep}{3pt}\n"
    table += "\\renewcommand{\\arraystretch}{1.05}\n"
    table += "\\caption{Durables Model: Per-Period Timing and Accuracy}\n"
    table += "\\label{tab:durables_timing}\n"
    table += "\\begin{tabular}{@{}rr rrrrr rrrrr@{}}\n"
    table += "\\toprule\n"

    # Header row 1: method groups
    table += (
        "& & "
        "\\multicolumn{5}{c}{FUES} & "
        "\\multicolumn{5}{c}{NEGM} \\\\\n"
    )
    table += (
        "\\cmidrule(lr){3-7} \\cmidrule(lr){8-12}\n"
    )

    # Header row 2: Time and Euler subgroups
    table += (
        "& & "
        "\\multicolumn{2}{c}{Time} & \\multicolumn{3}{c}{Euler} & "
        "\\multicolumn{2}{c}{Time} & \\multicolumn{3}{c}{Euler} \\\\\n"
    )
    table += (
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-12}\n"
    )

    # Header row 3: individual column labels
    # Use "NA" for Non-Adjuster to keep columns compact
    table += (
        "$N_A$ & $\\tau$ & "
        "NA & Adj & Comb & NA & Adj & "
        "NA & Adj & Comb & NA & Adj \\\\\n"
    )
    table += "\\midrule\n"

    current_grid = None
    for grid_size in grid_sizes:
        taus_for_grid = [t for t in tau_values if get_result(grid_size, t, 'FUES') is not None]

        for i, tau in enumerate(taus_for_grid):
            fues = get_result(grid_size, tau, 'FUES')
            negm = get_result(grid_size, tau, 'NEGM')

            if fues is None or negm is None:
                continue

            # Grid size column (only show for first tau in group)
            if i == 0:
                if current_grid is not None:
                    table += "\\midrule\n"
                grid_str = f"{grid_size:,}"
                current_grid = grid_size
            else:
                grid_str = ""

            # Use HD errors (most accurate measure)
            fues_comb = fues.get('Euler_Combined_HD', fues['Euler_Combined'])
            negm_comb = negm.get('Euler_Combined_HD', negm['Euler_Combined'])
            fues_adj = fues.get('Euler_Adjuster_HD', fues['Euler_Adjuster'])
            negm_adj = negm.get('Euler_Adjuster_HD', negm['Euler_Adjuster'])
            fues_keep = fues.get('Euler_Keeper_HD', fues['Euler_Keeper'])
            negm_keep = negm.get('Euler_Keeper_HD', negm['Euler_Keeper'])

            # Get timing in ms
            fues_keeper_ms = fues.get('Avg_Keeper_ms', 0.0)
            fues_adj_ms = fues.get('Avg_Adj_ms', 0.0)
            negm_keeper_ms = negm.get('Avg_Keeper_ms', 0.0)
            negm_adj_ms = negm.get('Avg_Adj_ms', 0.0)

            # Data row: Grid | FUES (Time: NA, Adj | Euler: Comb, NA, Adj) | NEGM (...)
            table += (
                f"{grid_str} & {tau} & "
                f"{fues_keeper_ms:.0f} & {fues_adj_ms:.0f} & {fues_comb:.2f} & {fues_keep:.2f} & {fues_adj:.2f} & "
                f"{negm_keeper_ms:.0f} & {negm_adj_ms:.0f} & {negm_comb:.2f} & {negm_keep:.2f} & {negm_adj:.2f} "
                "\\\\\n"
            )

    table += "\\bottomrule\n\\end{tabular}\n"
    table += "\\vspace{0.3em}\n"

    # Extract parameters from first result (they're the same across all runs)
    if results_summary:
        p = results_summary[0]
        param_str = (
            f"$\\beta={p.get('beta', 0.93):.2f}$, "
            f"$\\gamma_c={p.get('gamma_c', 3):.1f}$, "
            f"$\\gamma_h={p.get('gamma_h', 3):.1f}$, "
            f"$\\alpha={p.get('alpha', 0.7):.2f}$, "
            f"$\\delta={p.get('delta', 0.0):.2f}$, "
            f"$r={p.get('r', 0.01):.2f}$, "
            f"$r_H={p.get('r_H', 0.0):.2f}$, "
            f"$\\rho_w={p.get('phi_w', 0.9):.2f}$, "
            f"$\\sigma_w={p.get('sigma_w', 0.08):.2f}$, "
            f"$N_{{sim}}={p.get('N_sim', 10000):,}$"
        )
    else:
        param_str = ""

    table += (
        "\\par\\small \\textit{Notes:} "
        "Time in ms; Euler errors in $\\log_{10}$ scale. "
        "\\textbf{Comb}: combined mean. "
        "\\textbf{NA}: non-adjusters. "
        "\\textbf{Adj}: adjusters. "
        f"Parameters: {param_str}.\n"
    )
    table += "\\endgroup\n\\end{table}\n"

    return table

if __name__ == "__main__":
    # Get MPI info early for debug output
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    print(f"[Rank {rank}/{size}] Starting...", flush=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['negm', 'egm', 'both'], default='both',
                        help='Which method to run: negm, egm, or both (default)')
    parser.add_argument('--plot-start', type=int, default=None,
                        help='Start period for plotting (default: T-2). Will be clamped to valid range.')
    parser.add_argument('--run-tests', action='store_true',
                        help='Run grid/tau comparison tests (18 combinations, requires 18 MPI ranks)')
    args = parser.parse_args()

    run_tests = args.run_tests
    print(f"[Rank {rank}] Parsed args, run_tests={run_tests}", flush=True)

    # Read settings
    with open("settings.yml", "r") as stream:
        settings = yaml.safe_load(stream)

    # 1. Error and timing comparisons across grid sizes and tau values
    # Tau values and grid sizes to compare
    cp_settings = settings['durables']

    tau_values = [0.05, 0.07, 0.12]
    grid_sizes_A = [250, 500, 750, 1000]

    # Run comparison across grid sizes and tau values
    if run_tests:
        print(f"[Rank {rank}] Entering compare_grids_and_tau...", flush=True)
        results, latex_table = compare_grids_and_tau(cp_settings, tau_values, grid_sizes_A)
        if MPI.COMM_WORLD.Get_rank() == 0 and latex_table:
            print("\n" + "="*70)
            print("LaTeX Table:")
            print("="*70)
            print(latex_table)
        sys.exit(0)  # Exit after tests complete

    # 2. Solve the baseline model with MPI parallelization across methods
    
    # All ranks need to participate in MPI operations
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # All ranks create the consumer problem
    # Use AR(1) wage process via Tauchen method (Pi and z_vals generated internally)
    # Parameters taken from settings.yml unless overwritten below
    # Note: float()/int() conversions needed because YAML may read scientific notation as strings
    cp = ConsumerProblem(cp_settings,
                     r=float(cp_settings.get('r', 0.01)),
                     sigma=float(cp_settings.get('sigma', 0.001)),
                     r_H=float(cp_settings.get('r_H', 0)),
                     beta=float(cp_settings.get('beta', 0.93)),
                     alpha=float(cp_settings.get('alpha', 0.7)),
                     kappa=float(cp_settings.get('kappa', 1.0)),
                     delta=float(cp_settings.get('delta', 0)),
                     b=float(cp_settings.get('b', 1e-08)),
                     grid_max_A=float(cp_settings.get('grid_max_A', 20.0)),
                     grid_max_WE=float(cp_settings.get('grid_max_WE', 70.0)),
                     grid_size_W=int(cp_settings.get('grid_size_W', 300)),
                     grid_max_H=float(cp_settings.get('grid_max_H', 50.0)),
                     grid_size_A=int(cp_settings.get('grid_size_A', 300)),
                     grid_size_H=int(cp_settings.get('grid_size_H', 300)),
                     gamma_c=float(cp_settings.get('gamma_c', 3)),
                     gamma_h=float(cp_settings.get('gamma_h', 3)),
                     chi=float(cp_settings.get('chi', 0)),
                     tau=float(cp_settings.get('tau', 0.12)),
                     K=float(cp_settings.get('K', 1.3)),
                     tol_bel=float(cp_settings.get('tol_bel', 1e-03)),
                     m_bar=float(cp_settings.get('M_bar', 1.4)),
                     theta=float(cp_settings.get('theta', 1.3498)),
                     stat=cp_settings.get('stat', False),
                     T=int(cp_settings.get('max_iter', 60)),
                     t0=int(cp_settings.get('t0', 20)),
                     N_wage=int(cp_settings.get('N_wage', 4)),
                     phi_w=float(cp_settings.get('phi_w', 0.8170411)),
                     sigma_w=float(cp_settings.get('sigma_w', 0.1117012)))
            
    # Plot wage profile diagnostics before solving (rank 0 only)
    if rank == 0:
        print("\n" + "="*60)
        print("Plotting wage profile diagnostics...")
        print("="*60)
        plot_wage_profile(cp, age_min=20, age_max=60)

    # 0. Solve with Bellman (VFI option)
    #iterVFI, iterEGM, condition_V, NEGM = Operator_Factory(cp)
    #pickle.dump(bell_results, open("bell_results_300.p", "wb"))
    #bell_results = pickle.load(open("bell_results_300.p", "rb"))

    # 1. Solve using NEGM and EGM/FUES (parallel across MPI ranks)
    
    # Define method assignments for parallel execution
    methods = [
        ('NEGM', solveNEGM),
        ('FUES', solveEGM)
    ]

    # Use scratch for large pickle files, home for plots/tables
    scratch_dir = os.path.expandvars("/scratch/tp66/$USER/FUES/durables")
    results_dir = "../../results/durables"
    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Select methods based on --method argument or MPI rank
    if args.method == 'negm':
        methods_to_run = [('NEGM', solveNEGM)]
    elif args.method == 'egm':
        methods_to_run = [('FUES', solveEGM)]
    else:
        # Default: use MPI rank to select method
        methods_to_run = [(label, solver) for i, (label, solver) in enumerate(methods) if i % size == rank]

    for label, solver in methods_to_run:
        print(f"Solving {label}...")
        result = timing(solver, cp, rep=1)
        result['label'] = label
        result_file = os.path.join(scratch_dir, f"{label}_result.pkl")
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
        print(f"Saved {label} to {result_file}")

    # Barrier - wait for all ranks to finish solving (only if using MPI)
    if args.method == 'both':
        comm.Barrier()
    
    # Rank 0 loads all results and does plotting
    if rank == 0:
        print("[Rank 0] Loading results from all methods...")
        NEGMRes = pickle.load(open(os.path.join(scratch_dir, "NEGM_result.pkl"), 'rb'))
        EGMRes_fues = pickle.load(open(os.path.join(scratch_dir, "FUES_result.pkl"), 'rb'))

        # Get output directory from environment (scratch drive)
        output_dir = os.environ.get('FUES_OUTPUT_DIR', scratch_dir)
        print(f"[Rank 0] Saving plots to: {output_dir}/plots/")

        # Determine plot start period with validation
        # First valid period depends on solution structure (typically 0 or 1)
        first_valid_period = 1  # Adjust if needed based on solution indexing
        last_valid_period = cp.T - 1

        if args.plot_start is not None:
            plot_start = max(first_valid_period, min(args.plot_start, last_valid_period))
            if plot_start != args.plot_start:
                print(f"  Warning: plot_start clamped from {args.plot_start} to {plot_start} (valid range: {first_valid_period}-{last_valid_period})")
        else:
            plot_start = max(first_valid_period, cp.T - 2)

        # 2a. Policy function plots for multiple periods
        print(f"  Plotting periods {plot_start} to {cp.T - 1}...")
        for t in range(plot_start, cp.T):
            print(f"    Plotting age {t}...")
            plot_pols(cp, EGMRes_fues, NEGMRes, t, [100, 150, 200], output_dir=output_dir)

        # 2.b. plot of adjuster endog. grids for FUES
        print("  Plotting FUES grids...")
        plot_grids(EGMRes_fues, cp, term_t=plot_start, output_dir=output_dir)

        print(f"[Rank 0] All plots saved to: {output_dir}/plots/")

        # save results option 
        #pickle.dump(EGMRes_fues, open("EGMRes_fues.p", "wb"))
        #pickle.dump(NEGMRes, open("NEGMRes.p", "wb"))
        
        # 3. Euler errors (simulation-based with adjuster/keeper breakdown)
        print("\n" + "="*70)
        print(f"Computing Euler errors via simulation (N={cp.N_sim})...")
        print("="*70)

        # Compute simulation-based Euler errors with timing
        # Use same seed so both methods simulate identical agents for proper welfare comparison
        t_euler_start = time.time()
        euler_egm, euler_egm_hd, sim_egm = euler_errors(EGMRes_fues, cp, N=cp.N_sim, seed=42)
        euler_negm, euler_negm_hd, sim_negm = euler_errors(NEGMRes, cp, N=cp.N_sim, seed=42)
        t_euler = time.time() - t_euler_start
        print(f"  Euler error computation time: {t_euler:.2f}s")

        # Get detailed stats with adjuster/keeper breakdown (normal and HD versions)
        stats_egm = compute_euler_stats(euler_egm, discrete=sim_egm['discrete'])
        stats_negm = compute_euler_stats(euler_negm, discrete=sim_negm['discrete'])
        stats_egm_hd = compute_euler_stats(euler_egm_hd, discrete=sim_egm['discrete'])
        stats_negm_hd = compute_euler_stats(euler_negm_hd, discrete=sim_negm['discrete'])

        # Print detailed Euler error report
        print("\n" + "-"*70)
        print("EGM/FUES Euler Errors (log10 scale):")
        print("-"*70)
        print(f"  Combined:   mean={stats_egm['combined']['mean']:.4f}, "
              f"median={stats_egm['combined']['median']:.4f}, "
              f"p5={stats_egm['combined']['p5']:.4f}, "
              f"p95={stats_egm['combined']['p95']:.4f}")
        print(f"  Adjuster:   mean={stats_egm['adjuster']['mean']:.4f}, "
              f"median={stats_egm['adjuster']['median']:.4f} "
              f"(n={stats_egm['adjuster']['n_obs']})")
        print(f"  Keeper:     mean={stats_egm['keeper']['mean']:.4f}, "
              f"median={stats_egm['keeper']['median']:.4f} "
              f"(n={stats_egm['keeper']['n_obs']})")
        print(f"  Adj. rate:  {stats_egm['pct_adjuster']:.2f}%")

        print("\n" + "-"*70)
        print("NEGM Euler Errors (log10 scale):")
        print("-"*70)
        print(f"  Combined:   mean={stats_negm['combined']['mean']:.4f}, "
              f"median={stats_negm['combined']['median']:.4f}, "
              f"p5={stats_negm['combined']['p5']:.4f}, "
              f"p95={stats_negm['combined']['p95']:.4f}")
        print(f"  Adjuster:   mean={stats_negm['adjuster']['mean']:.4f}, "
              f"median={stats_negm['adjuster']['median']:.4f} "
              f"(n={stats_negm['adjuster']['n_obs']})")
        print(f"  Keeper:     mean={stats_negm['keeper']['mean']:.4f}, "
              f"median={stats_negm['keeper']['median']:.4f} "
              f"(n={stats_negm['keeper']['n_obs']})")
        print(f"  Adj. rate:  {stats_negm['pct_adjuster']:.2f}%")

        # Print HD Euler errors (using finer Lambda_H grid)
        print("\n" + "-"*70)
        print("HD Euler Errors (log10 scale, using finer Lambda_H grid):")
        print("-"*70)
        print(f"  EGM/FUES Adjuster (HD):  mean={stats_egm_hd['adjuster']['mean']:.4f}, "
              f"median={stats_egm_hd['adjuster']['median']:.4f}")
        print(f"  NEGM Adjuster (HD):      mean={stats_negm_hd['adjuster']['mean']:.4f}, "
              f"median={stats_negm_hd['adjuster']['median']:.4f}")

        # Report NPV utility
        npv_egm = sim_egm['npv_utility']
        npv_negm = sim_negm['npv_utility']

        print("\n" + "-"*70)
        print("Net Present Discounted Utility (per agent):")
        print("-"*70)
        print(f"  EGM/FUES:   mean={np.mean(npv_egm):.4f}, "
              f"median={np.median(npv_egm):.4f}, "
              f"std={np.std(npv_egm):.4f}")
        print(f"  NEGM:       mean={np.mean(npv_negm):.4f}, "
              f"median={np.median(npv_negm):.4f}, "
              f"std={np.std(npv_negm):.4f}")

        # Consumption equivalent welfare comparison
        # For CRRA utility, CE = 1 - (V_NEGM / V_FUES)^(1/(1-γ))
        # This approximation works best for the non-shifted CRRA form
        gamma_c = cp.gamma_c
        v_fues = np.mean(npv_egm)
        v_negm = np.mean(npv_negm)

        # Utility difference as percentage
        util_diff_pct = 100 * (v_fues - v_negm) / abs(v_fues) if v_fues != 0 else 0.0

        # CE calculation using CRRA formula
        if v_fues != 0 and v_negm != 0:
            ratio = v_negm / v_fues
            exponent = 1 / (1 - gamma_c)
            ce_raw = 100 * (1 - ratio ** exponent)
        else:
            ce_raw = 0.0

        print("\n" + "-"*70)
        print("Consumption Equivalent Welfare Comparison:")
        print("-"*70)
        print(f"  gamma_c = {gamma_c}")
        print(f"  V_FUES = {v_fues:.6f}, V_NEGM = {v_negm:.6f}")
        print(f"  Utility difference: {util_diff_pct:.4f}%")
        print(f"  V_NEGM/V_FUES = {v_negm/v_fues:.6f}")

        if v_fues > v_negm:
            # FUES is better - report CE gain from using FUES (= loss from using NEGM)
            ce_pct = abs(ce_raw)
            print(f"  FUES delivers higher welfare than NEGM")
            print(f"  CE loss from using NEGM: {ce_pct:.4f}% of consumption")
        elif v_negm > v_fues:
            # NEGM is better - report CE loss from using FUES
            ce_pct = -abs(ce_raw)  # Make it negative to indicate NEGM is better
            print(f"  NEGM delivers higher welfare than FUES")
            print(f"  CE loss from using FUES: {abs(ce_raw):.4f}% of consumption")
        else:
            ce_pct = 0.0
            print(f"  Both methods deliver equivalent welfare")

        # Adjuster-conditional welfare comparison
        # This computes CE based only on utility in adjustment periods
        npv_adj_egm = sim_egm['npv_utility_adj']
        npv_adj_negm = sim_negm['npv_utility_adj']
        n_adj_egm = sim_egm['n_adj_periods']
        n_adj_negm = sim_negm['n_adj_periods']

        # Only include agents who had at least one adjustment period
        adj_mask_egm = n_adj_egm > 0
        adj_mask_negm = n_adj_negm > 0
        # Use union mask to ensure we're comparing same agents
        adj_mask = adj_mask_egm & adj_mask_negm

        v_fues_adj = np.mean(npv_adj_egm[adj_mask]) if np.sum(adj_mask) > 0 else 0.0
        v_negm_adj = np.mean(npv_adj_negm[adj_mask]) if np.sum(adj_mask) > 0 else 0.0

        print("\n" + "-"*70)
        print("Adjuster-Conditional Welfare Comparison:")
        print("-"*70)
        print(f"  (Utility accumulated only in periods when agent adjusts)")
        print(f"  N agents with adjustment periods: {np.sum(adj_mask)}")
        print(f"  Mean adj periods per agent: {np.mean(n_adj_egm[adj_mask]):.2f}")
        print(f"  V_FUES (adj only) = {v_fues_adj:.6f}")
        print(f"  V_NEGM (adj only) = {v_negm_adj:.6f}")

        # Utility difference for adjusters
        util_diff_adj_pct = 100 * (v_fues_adj - v_negm_adj) / abs(v_fues_adj) if v_fues_adj != 0 else 0.0
        print(f"  Utility difference (adj only): {util_diff_adj_pct:.4f}%")

        # CE for adjusters
        if v_fues_adj != 0 and v_negm_adj != 0:
            ratio_adj = v_negm_adj / v_fues_adj
            ce_adj_raw = 100 * (1 - ratio_adj ** exponent)
        else:
            ce_adj_raw = 0.0

        if v_fues_adj > v_negm_adj:
            ce_adj_pct = abs(ce_adj_raw)
            print(f"  FUES delivers higher adjuster welfare than NEGM")
            print(f"  CE loss from using NEGM (adj only): {ce_adj_pct:.4f}% of consumption")
        elif v_negm_adj > v_fues_adj:
            ce_adj_pct = -abs(ce_adj_raw)
            print(f"  NEGM delivers higher adjuster welfare than FUES")
            print(f"  CE loss from using FUES (adj only): {abs(ce_adj_raw):.4f}% of consumption")
        else:
            ce_adj_pct = 0.0
            print(f"  Both methods deliver equivalent adjuster welfare")

        # 4. Tabulate timing and errors
        negm_time = _get_avg_time(NEGMRes)
        egm_time = _get_avg_time(EGMRes_fues)
        egm_keeper_ms = _get_avg_keeper_ms(EGMRes_fues)
        egm_adj_ms = _get_avg_adj_ms(EGMRes_fues)
        negm_keeper_ms = _get_avg_keeper_ms(NEGMRes)
        negm_adj_ms = _get_avg_adj_ms(NEGMRes)

        print("\n" + "="*70)
        print("Summary Table")
        print("="*70)
        print(f"{'Metric':<30} {'EGM/FUES':>15} {'NEGM':>15}")
        print("-"*70)
        print(f"{'Combined mean':<30} {stats_egm['combined']['mean']:>15.4f} {stats_negm['combined']['mean']:>15.4f}")
        print(f"{'Combined median':<30} {stats_egm['combined']['median']:>15.4f} {stats_negm['combined']['median']:>15.4f}")
        print(f"{'Adjuster mean':<30} {stats_egm['adjuster']['mean']:>15.4f} {stats_negm['adjuster']['mean']:>15.4f}")
        print(f"{'Adjuster mean (HD)':<30} {stats_egm_hd['adjuster']['mean']:>15.4f} {stats_negm_hd['adjuster']['mean']:>15.4f}")
        print(f"{'Keeper mean':<30} {stats_egm['keeper']['mean']:>15.4f} {stats_negm['keeper']['mean']:>15.4f}")
        print(f"{'5th percentile':<30} {stats_egm['combined']['p5']:>15.4f} {stats_negm['combined']['p5']:>15.4f}")
        print(f"{'95th percentile':<30} {stats_egm['combined']['p95']:>15.4f} {stats_negm['combined']['p95']:>15.4f}")
        print(f"{'Adjustment rate (%)':<30} {stats_egm['pct_adjuster']:>15.2f} {stats_negm['pct_adjuster']:>15.2f}")
        print(f"{'NPV Utility (mean)':<30} {v_fues:>15.4f} {v_negm:>15.4f}")
        print(f"{'CE loss from NEGM (%)':<30} {ce_pct:>15.4f} {'-':>15}")
        print(f"{'NPV Utility adj (mean)':<30} {v_fues_adj:>15.4f} {v_negm_adj:>15.4f}")
        print(f"{'CE loss from NEGM adj (%)':<30} {ce_adj_pct:>15.4f} {'-':>15}")
        print(f"{'Avg. time/iter (sec)':<30} {egm_time:>15.2f} {negm_time:>15.2f}")
        print(f"{'Avg. keeper time (ms)':<30} {egm_keeper_ms:>15.1f} {negm_keeper_ms:>15.1f}")
        print(f"{'Avg. adjuster time (ms)':<30} {egm_adj_ms:>15.1f} {negm_adj_ms:>15.1f}")
        print("="*70)

        # Build markdown table for file output
        lines = []
        lines.append('| Metric | EGM/FUES | NEGM |\n')
        lines.append('|--------|----------|------|\n')
        lines.append(f"| Combined mean | {stats_egm['combined']['mean']:.4f} | {stats_negm['combined']['mean']:.4f} |\n")
        lines.append(f"| Combined median | {stats_egm['combined']['median']:.4f} | {stats_negm['combined']['median']:.4f} |\n")
        lines.append(f"| Adjuster mean | {stats_egm['adjuster']['mean']:.4f} | {stats_negm['adjuster']['mean']:.4f} |\n")
        lines.append(f"| Adjuster mean (HD) | {stats_egm_hd['adjuster']['mean']:.4f} | {stats_negm_hd['adjuster']['mean']:.4f} |\n")
        lines.append(f"| Keeper mean | {stats_egm['keeper']['mean']:.4f} | {stats_negm['keeper']['mean']:.4f} |\n")
        lines.append(f"| 5th percentile | {stats_egm['combined']['p5']:.4f} | {stats_negm['combined']['p5']:.4f} |\n")
        lines.append(f"| 95th percentile | {stats_egm['combined']['p95']:.4f} | {stats_negm['combined']['p95']:.4f} |\n")
        lines.append(f"| Adjustment rate (%) | {stats_egm['pct_adjuster']:.2f} | {stats_negm['pct_adjuster']:.2f} |\n")
        lines.append(f"| NPV Utility (mean) | {v_fues:.4f} | {v_negm:.4f} |\n")
        lines.append(f"| CE loss from NEGM (%) | {ce_pct:.4f} | - |\n")
        lines.append(f"| NPV Utility adj (mean) | {v_fues_adj:.4f} | {v_negm_adj:.4f} |\n")
        lines.append(f"| CE loss from NEGM adj (%) | {ce_adj_pct:.4f} | - |\n")
        lines.append(f"| Avg. time/iter (sec) | {egm_time:.2f} | {negm_time:.2f} |\n")
        lines.append(f"| Avg. keeper time (ms) | {egm_keeper_ms:.1f} | {negm_keeper_ms:.1f} |\n")
        lines.append(f"| Avg. adjuster time (ms) | {egm_adj_ms:.1f} | {negm_adj_ms:.1f} |\n")

        with open("../../results/durables/durables_single_result.tex", "w") as f:
            f.writelines(lines)

        # 5. Lifecycle comparison plots
        print("\n" + "="*70)
        print("Generating lifecycle comparison plots...")
        print("="*70)
        plot_lifecycle_compare(sim_egm, sim_negm, cp, output_dir=output_dir)