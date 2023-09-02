""" Script to plot solution of Ishkakov et al (2017) retirement choice
model using FUES-EGM by Dobrescu and Shanker (2022).

Author: Akshay Shanker, University of Sydney, akshay.shanker@me.com.

See examples/retirement_choice for model. 

Todo
----
- Improve integration with DC-EGM and implement
    timing comparison with DC-EGM with jit compiled
    version of DC-EGM


"""

import numpy as np
from numba import jit
import time
import dill as pickle
from numba import njit, prange
from sklearn.utils.extmath import cartesian 

from FUES.FUES import FUES

from FUES.math_funcs import interp_as, upper_envelope

from HARK.interpolation import LinearInterp
from HARK.dcegm import calc_nondecreasing_segments, upper_envelope, calc_linear_crossing

from interpolation import interp

import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pylab as pl


def plot_egrids(age, e_grid, vf_work, c_worker, g_size):
    # Plot value corr. and policy on 
    # unrefined vs refined endogenous grid for age

    # get unrefined endogenous grid, value function and consumption
    # for worker at time t
    x = np.array(e_grid[age])
    vf = np.array(vf_work[age])
    c = np.array(c_worker[age])
    a_prime = np.array(cp.asset_grid_A)

    # generate refined grid, value function and policy using FUES
    x_clean, vf_clean, c_clean, a_prime_clean, dela \
        = FUES(x, vf, c, a_prime, 0.8)

    # make plots  
    pl.close()
    fig, ax = pl.subplots(1, 2)
    sns.set(
        style="white", rc={
            "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})

    ax[0].scatter(
        x,
        vf * cp.beta - cp.delta,
        s=20,
        facecolors='none',
        edgecolors='r')
    ax[0].plot(
        x_clean,
        vf_clean * cp.beta - cp.delta,
        color='black',
        linewidth=1,
        label='Value function')
    ax[0].scatter(
        x_clean,
        vf_clean * cp.beta - cp.delta,
        color='blue',
        s=15,
        marker='x',
        linewidth=0.75)

    ax[0].set_xlabel('Assets (t)', fontsize=11)
    ax[0].set_ylabel('Value', fontsize=11)
    ax[0].set_ylim(7.6, 8.5)
    ax[0].set_xlim(44, 55)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].legend(frameon=False, prop={'size': 10})
    ax[0].set_yticklabels(ax[0].get_yticks(), size=9)
    ax[0].set_xticklabels(ax[0].get_xticks(), size=9)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    ax[1].scatter(
        np.sort(x),
        np.take(
            x - c,
            np.argsort(x)),
        s=20,
        facecolors='none',
        edgecolors='r',
        label='EGM points')
    ax[1].scatter(
        np.sort(x_clean),
        np.take(
            x_clean - c_clean,
            np.argsort(x_clean)),
        s=20,
        color='blue',
        marker='x',
        linewidth=0.75,
        label='Optimal points')

    ax[1].set_ylim(20, 40)
    ax[1].set_xlim(44, 55)
    ax[1].set_ylabel('Assets (t+1)', fontsize=11)
    ax[1].set_xlabel('Assets (t)', fontsize=11)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_yticklabels(ax[1].get_yticks(), size=9)
    ax[1].set_xticklabels(ax[1].get_xticks(), size=9)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    fig.tight_layout()
    ax[1].legend(frameon=False, prop={'size': 10})
    fig.savefig(
        'plots/retirement/ret_vf_aprime_all_{}_{}.png'.format(age, g_size))
    pl.close()

    return None


def plot_cons_pol(sigma_work):
    # Plot consumption policy  for difference ages
    sns.set(style="whitegrid",
            rc={"font.size": 10,
                "axes.titlesize": 10,
                "axes.labelsize": 10})
    fig, ax = pl.subplots(1, 1)

    for t, col, lab in zip([17, 10, 0], ['blue', 'red', 'black'], [
            't=18', 't=10', 't=1']):

        cons_pol = np.copy(sigma_work[t])

        # remove jump joints for plotting only
        pos = np.where(np.abs(np.diff(cons_pol))\
                    /np.diff(cp.asset_grid_A)> 0.3)[0] + 1
        y1 = np.insert(cons_pol, pos, np.nan)
        x1 = np.insert(cp.asset_grid_A, pos, np.nan)

        ax.plot(x1, y1, color=col, label=lab)
        ax.set_xlim(0, 380)
        ax.set_ylim(0, 40)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticklabels(ax.get_yticks(), size=9)
        ax.set_xticklabels(ax.get_xticks(), size=9)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.set_ylabel('Consumption', fontsize=11)
        ax.set_xlabel('Assets (t)', fontsize=11)

    ax.legend(frameon=False, prop={'size': 10})
    fig.savefig('plots/retirement/ret_cons_all.png'.format(t))
    pl.close()

    return None


def plot_dcegm_cf(age, g_size, e_grid, vf_work, c_worker, a_prime,
                  plot=True):
    # get unrefined endogenous grid, value function and consumption
    # for worker at time t
    x = e_grid[age]
    vf = vf_work[age]
    c = c_worker[age]
    a_prime = cp.asset_grid_A
    time_start_dcegm = time.time()

#     start, end = calc_segments(x, vf)
    start, end = calc_nondecreasing_segments(x, vf)

    # generate refined grid, value function and policy using FUES
    x_clean, vf_clean, c_clean, a_prime_clean, dela = FUES(x, vf,
                                                           c, a_prime, m_bar=2)
    # interpolate
    vf_interp_fues = np.interp(x, x_clean, vf_clean)
    # len(vf_interp_fues[x_clean.searchsorted(x)])
    vf_interp_fues[x.searchsorted(x_clean)] = vf_clean

    # Plot them, and store them as [m, v] pairs
    segments = []
    c_segments = []
    a_segments = []
    m_segments = []
    v_segments = []

    for j in range(len(start)):
        idx = range(start[j], end[j] + 1)
        segments.append([x[idx], vf[idx]])
        c_segments.append(c[idx])
        a_segments.append(a_prime[idx])
        m_segments.append(x[idx])
        v_segments.append(vf[idx])

    m_upper, v_upper, inds_upper = upper_envelope(segments)
    vf_interp_fues = np.interp(m_upper, x_clean, vf_clean)
    a_interp_fues = np.interp(m_upper, x_clean, a_prime_clean)

    c1_env = np.zeros_like(m_upper) + np.nan
    a1_env = np.zeros_like(m_upper) + np.nan
    v1_env = np.zeros_like(m_upper) + np.nan

    for k, c_segm in enumerate(c_segments):
        c1_env[inds_upper == k] = c_segm[m_segments[k] .searchsorted(
            m_upper[inds_upper == k])]

    for k, a_segm in enumerate(a_segments):
        a1_env[inds_upper == k] = np.interp(m_upper[inds_upper == k],
                                            m_segments[k], a_segm)

    for k, v_segm in enumerate(v_segments):
        v1_env[inds_upper == k] = LinearInterp(
            m_segments[k], v_segm)(m_upper[inds_upper == k])

    a1_up = LinearInterp(m_upper, a1_env)
    indices = np.where(np.in1d(a1_env, a_prime))[0]
    a1_env2 = a1_env[indices]
    m_upper2 = m_upper[indices]

    if plot:

        pl.close()
        fig, ax = pl.subplots(1, 2)
        sns.set(
            style="whitegrid", rc={
                "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})

        ax[1].scatter(
            x,
            vf * cp.beta - cp.delta,
            s=20,
            facecolors='none',
            label='EGM points',
            edgecolors='r')

        ax[1].scatter(
            x_clean,
            vf_clean * cp.beta - cp.delta,
            color='blue',
            s=15,
            marker='x',    
            label='FUES-EGM points',
            linewidth=0.75)



        ax[0].scatter(
            x,
            a_prime,
            edgecolors='r',
            s=15,
            facecolors='none',
            label='EGM point',
            linewidth=0.75)

        ax[0].scatter(m_upper2, a1_env2,
                      edgecolors='red',
                      marker='o',
                      s=15,
                      label='DC-EGM point',
                      facecolors='none',
                      linewidth=0.75)


            # print(m_segments[k])

        #for k, v_segm in enumerate(v_segments):
        #    ax[1].plot(m_segments[k], v_segm * cp.beta - cp.delta,
        #               color='black',
        #               linestyle='--',
        #               linewidth=0.75)

        ax[0].scatter(
            x_clean,
            a_prime_clean,
            color='blue',
            s=15,
            marker='x',
            label='FUES-EGM points',
            linewidth=0.75)

        for k, a_segm in enumerate(a_segments):
            if k == 0:
                label1 = 'DC-EGM line seg.'
            else:
                label1 = None 

            ax[0].plot(m_segments[k], a_segm,
                       color='black',
                       linestyle='--',
                       label = label1,
                       linewidth=0.75)

        ax[1].set_ylim(7.5, 9.2)
        ax[1].set_xlim(40,80)
        ax[1].set_xlabel('Assets (t)', fontsize=11)
        ax[1].set_ylabel('Value', fontsize=11)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].legend(frameon=False, prop={'size': 10})

        ax[0].set_ylim(20, 60)
        ax[0].set_xlim(40, 80)
        ax[0].set_xlabel('Assets (t)', fontsize=11)
        ax[0].set_ylabel('Assets (t+1)', fontsize=11)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].legend(frameon=False, prop={'size': 10}, loc = 'upper left')

        fig.tight_layout()
        fig.savefig('plots/retirement/ret_vf_aprime_all_{}_cf_{}.png'
                    .format(g_size, age))

    return v_upper, v1_env, vf_interp_fues, a_prime_clean, m_upper, a1_env2


if __name__ == "__main__":

    from examples.retirement_choice import Operator_Factory, RetirementModel

    
    # Generate baseline parameter solution using FUES and make plots 

    # Create instance of RetirementModel
    g_size_baseline = 2000
    cp = RetirementModel(r=0.02,
                         beta= 0.98,
                         delta=1,
                         y=20,
                         b=1E-10,
                         grid_max_A=500,
                         grid_size=g_size_baseline,
                         T=20,
                         smooth_sigma=0)

    # Unpack solver operators 
    Ts_ret, Ts_work, iter_bell = Operator_Factory(cp)

    # Get optimal value and policy functions using FUES
    # by iterating on the Bellman equation 
    e_grid_worker_unref, vf_work_unref,vf_refined,\
             c_worker_unref,c_refined, iter_time_age = iter_bell(cp)

    # 1. Example use of FUES to refine EGM grids
    # get unrefined endogenous grid, value function and consumption
    # for worker at time age
    age = 17 
    x = np.array(e_grid_worker_unref[age])
    vf = np.array(vf_work_unref[age])
    c = np.array(c_worker_unref[age])
    a_prime = np.array(cp.asset_grid_A)

    # generate refined grid, value function and policy using FUES
    x_clean, vf_clean, c_clean, a_prime_clean, dela \
        = FUES(x, vf, c, a_prime, 2)


    # 2. Plot and save value function and policy on EGM grids
    # and refined EGM grids 
    plot_egrids(17, e_grid_worker_unref, vf_work_unref,\
                     c_worker_unref, g_size_baseline)

    # 3. Plot consumption function (for worker, 
    # but before next period work decision
    # made)
    plot_cons_pol(c_refined)

    # 4. Compute and plot comparison with DC-EGM 

    v_upper, v1_env, vf_interp_fues, a_interp_fues, m_upper, a1_env \
        = plot_dcegm_cf(age, g_size_baseline, e_grid_worker_unref,
                            vf_work_unref, c_worker_unref, cp.asset_grid_A,
                            plot=True)

    # 5. Evalute DC-EGM and FUES upper envelope for 
    # parms on a grid.  

    g_size = 300
    beta_min = 0.85
    g_size_min = 300
    g_size_max = 2000
    beta_max = 0.98
    N_params = 2
    y_min = 10
    y_max = 25
    delta_min = 0.5
    delta_max = 1.5

    betas = np.linspace(beta_min, beta_max, N_params)
    ys = np.linspace(y_min, y_max, N_params)
    gsizes = np.linspace(g_size_min, g_size_max, N_params)
    deltas = np.linspace(delta_min, delta_max, N_params)
    params = cartesian([betas,ys,deltas])


    # age at which to compcare DC-EGM with FUES
    age_dcegm = 17

    errors = np.empty(len(params))
    fues_times = np.empty(len(params))
    all_iter_times = np.empty(len(params))

    # Compare values policy from DC-EGM with FUES
    # Note we solve the model using FUES. Then at age_dcegm, we take the full
    # EGM grid and compute the upper envelope using DC-EGM and compare to FUES.
    # Comparison performed on EGM grid points selected by DC-EGM 
    # (not all EGM points, to avoid picking up interpolation 
    #  error due different interpolation grids 
    # used by DC-EGM and FUES 
    param_i = 0

    for p_list in range(len(params)):

        beta = params[p_list][0]
        delta = params[p_list][2]
        y = params[p_list][1]

        # Create instance of RetirementModel
        cp = RetirementModel(r=0.02,
                             beta=beta,
                             delta=delta,
                             y=y,
                             b=1E-1,
                             grid_max_A=500,
                             grid_size=g_size,
                             T=20,
                             smooth_sigma=0)

        # Unpack solvers
        Ts_ret, Ts_work, iter_bell = Operator_Factory(cp)

        # Get optimal value and policy functions using FUES
        e_grid, vf_work, vf_uncond, c_worker, sigma_work, mean_times\
             = iter_bell(cp)

        # calc upper envelope using DC-EGM and compare on EGM points to
        # FUES
        v_upper, v1_env, vf_interp_fues, a_interp_fues, m_upper, a1_env \
            = plot_dcegm_cf(age_dcegm, g_size, e_grid,
                            vf_work, c_worker, cp.asset_grid_A,
                            plot=False)

        if len(a1_env) == len(a_interp_fues):
            errors[param_i] = \
                np.max(np.abs(a1_env - a_interp_fues)) / len(a1_env)

        else:
            errors[param_i] =\
                np.max(np.abs(vf_interp_fues - v_upper)) / len(v_upper)
        fues_times[param_i] = mean_times[0]
        all_iter_times[param_i]  = mean_times[1]

        print(errors[param_i])

        param_i = param_i + 1

    print("Test DC-EGM vs. FUES on uniform grid of {} parameters:".format(N_params**3))
    print(' '    'beta: ({},{}), delta: ({},{}), y: ({},{})  \n '    ' exog. grid size: {}'\
            .format(beta_min, beta_max, y_min, y_max, delta_min, delta_max,g_size))
    print("Avg. error between DC-EGM and FUES: {0:.6f}"\
            .format(np.mean(errors)))
    print('Timings:')
    print(' '    'Avg. FUES time (secs): {0:.6f}'\
            .format(np.mean(fues_times)))
    print(' '    'Avg. worker iteration time (secs): {0:.6f}'\
            .format(np.mean(all_iter_times)))

