"""Plotting functions for retirement model results.

Author: Akshay Shanker, University of New South Wales, akshay.shanker@me.com
"""

import numpy as np
import time
import os
from HARK.interpolation import LinearInterp
from HARK.dcegm import calc_nondecreasing_segments, upper_envelope
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pylab as pl
import matplotlib.lines as mlines

from dcsmm.fues.fues import FUES as fues_alg


def _style_axis_spines(ax, *, color="0.65", linewidth=0.8, all_sides: bool = False) -> None:
    """Style axes spines (optionally on all four sides)."""
    sides = ("left", "bottom", "right", "top") if all_sides else ("left", "bottom")
    for side in sides:
        try:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(color)
            ax.spines[side].set_linewidth(linewidth)
        except Exception:
            pass
    # Match tick marks to spine color (labels remain default/black)
    try:
        ax.tick_params(axis="both", which="both", color=color)
    except Exception:
        pass


def plot_egrids(age, e_grid, vf_work, c_worker, del_a, g_size, cp, save_path, tag='sigma0'):
    """Plot unrefined vs refined endogenous grid for age.

    Left plot is value, right plot is policy points.
    Figure 4 in FUES paper.

    Parameters
    ----------
    age : int
        Age at which to plot value function and policy
    e_grid : dict
        Dictionary of endogenous grids for worker
    vf_work : dict
        Dictionary of value unrefined corrs. for worker by age
    c_worker : dict
        Dictionary of unrefined consumption points for worker by age
    del_a : dict
        Dictionary of unrefined derivative of policy function for worker by age
    g_size : int
        Grid size for the model for labeling.
    cp : RetirementModel
        RetirementModel instance
    save_path : str
        Path to save the plot
    tag : str, optional
        Tag for labeling, default is 'sigma0'.
    """
    # 1. Get unrefined endogenous grid, value function and consumption
    x = np.array(e_grid[age])
    vf = np.array(vf_work[age])
    c = np.array(c_worker[age])
    del_a = np.array(del_a[age])
    a_prime = np.array(cp.asset_grid_A)

    # 2. Generate refined grid, value function and policy using FUES
    fues_result, intersections = fues_alg(
        x, vf, c, a_prime, del_a, m_bar=1.0001,
        include_intersections=True,
        return_intersections_separately=True
    )
    x_clean, vf_clean, c_clean, a_prime_clean, del_a_clean = fues_result
    inter_e, inter_v, inter_p1, inter_p2, inter_d = intersections

    # 3. Make plots
    pl.close()
    fig, ax = pl.subplots(1, 2)
    sns.set(
        style="white", rc={
            "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})

    ax[0].scatter(x, vf * cp.beta - cp.delta, s=20, facecolors='none',
                  edgecolors='r', label='EGM points')
    ax[0].scatter(x_clean, vf_clean * cp.beta - cp.delta, color='blue',
                  s=15, marker='x', linewidth=0.75, label='FUES optimal points')
    ax[0].plot(x_clean, vf_clean * cp.beta - cp.delta, color='black',
               linewidth=1, label=r'Value function $V_t^{1}$')

    if len(inter_e) > 0:
        ax[0].scatter(inter_e, inter_v * cp.beta - cp.delta, color='green',
                      s=50, marker='*', linewidth=1, edgecolors='black',
                      label='Intersection points', zorder=5)

    ax[0].set_ylabel('Value', fontsize=11)
    ax[0].set_ylim(7.6, 8.4)
    ax[0].set_xlim(45, 55)
    # Full grey frame (all four spines) for each panel - matches plot_egm_csv.py
    _style_axis_spines(ax[0], color="0.65", linewidth=0.8, all_sides=True)
    ax[0].legend(frameon=False, prop={'size': 10})
    ax[0].tick_params(axis='y', labelsize=9)
    ax[0].tick_params(axis='x', labelsize=9)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax[0].grid(True)

    # Right plot
    ax[1].scatter(np.sort(x), np.take(x - c, np.argsort(x)), s=20,
                  facecolors='none', edgecolors='r', label='EGM points')
    ax[1].scatter(np.sort(x_clean), np.take(x_clean - c_clean, np.argsort(x_clean)),
                  s=20, color='blue', marker='x', linewidth=0.75, label='FUES optimal points')

    if len(inter_e) > 0:
        sort_idx = np.argsort(inter_e)
        ax[1].scatter(inter_e[sort_idx], inter_e[sort_idx] - inter_p1[sort_idx],
                      color='green', s=50, marker='*', linewidth=1,
                      edgecolors='black', label='Intersection points', zorder=5)

    ax[1].set_ylim(20, 40)
    ax[1].set_xlim(45, 55)
    ax[1].set_ylabel('Financial assets at time t+1', fontsize=11)
    _style_axis_spines(ax[1], color="0.65", linewidth=0.8, all_sides=True)
    ax[1].tick_params(axis='y', labelsize=9)
    ax[1].tick_params(axis='x', labelsize=9)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax[1].legend(frameon=False, prop={'size': 10})
    ax[1].grid(True)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.supxlabel('Endogenous grid of financial assets at time t', fontsize=11)
    fig.savefig(os.path.join(save_path, f'ret_vf_aprime_all_{age}_{g_size}_{tag}.png'))
    pl.close()


def plot_cons_pol(sigma_work, cp, save_path, ages=[17, 10, 0]):
    """Plot consumption policy for different ages.

    Parameters
    ----------
    sigma_work : dict
        Dictionary of consumption policy functions by age
    cp : RetirementModel
        RetirementModel instance
    save_path : str
        Path to save the plot
    ages : list
        List of ages to plot consumption policy for
    """
    sns.set(style="whitegrid", rc={"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10})
    fig, ax = pl.subplots(1, 1)

    for t, col, lab in zip(ages, ['blue', 'red', 'black'], ['t=18', 't=10', 't=1']):
        cons_pol = np.copy(sigma_work[t])
        pos = np.where(np.abs(np.diff(cons_pol)) / np.diff(cp.asset_grid_A) > 0.3)[0] + 1
        y1 = np.insert(cons_pol, pos, np.nan)
        x1 = np.insert(cp.asset_grid_A, pos, np.nan)

        ax.plot(x1, y1, color=col, label=lab)
        ax.set_xlim(0, 380)
        ax.set_ylim(0, 40)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=9)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.set_ylabel('Consumption at time $t$', fontsize=11)
        ax.set_xlabel('Financial assets at time $t$', fontsize=11)

    _style_axis_spines(ax)
    ax.legend(frameon=False, prop={'size': 10})
    fig.savefig(os.path.join(save_path, 'ret_cons_all.png'))
    pl.close()


def plot_dcegm_cf(age, g_size, e_grid, vf_work, c_worker, dela_worker, a_prime, cp,
                  save_path, tag='sigma05', plot=True):
    """Plot comparison of DC-EGM and FUES for worker at a specific age.

    Figure 5 in FUES paper.

    Parameters
    ----------
    age : int
        Age at which to plot value function and policy.
    g_size : int
        Grid size for the model for labeling.
    e_grid : dict
        Dictionary of endogenous grids for worker.
    vf_work : dict
        Dictionary of value unrefined correlations for worker by age.
    c_worker : dict
        Dictionary of unrefined consumption points for worker by age.
    dela_worker : dict
        Dictionary of unrefined derivative of policy function by age.
    a_prime : array
        Asset grid.
    cp : RetirementModel
        RetirementModel instance
    save_path : str
        Path to save the plot
    tag : str, optional
        Tag for labeling, default is 'sigma05'.
    plot : bool, optional
        If True, generates plot. Default is True.
    """
    x = e_grid[age]
    vf = vf_work[age]
    c = c_worker[age]
    a_prime = cp.asset_grid_A
    dela = dela_worker[age]

    x_clean, vf_clean, c_clean, a_prime_clean, dela_clean = fues_alg(
        x, vf, c, a_prime, dela, m_bar=1.2, endog_mbar=True
    )

    vf_interp_fues = np.interp(x, x_clean, vf_clean)
    vf_interp_fues[x.searchsorted(x_clean)] = vf_clean

    start, end = calc_nondecreasing_segments(x, vf)
    segments, c_segments, a_segments, m_segments = [], [], [], []
    v_segments, dela_segments = [], []

    for j in range(len(start)):
        idx = range(start[j], end[j] + 1)
        segments.append([x[idx], vf[idx]])
        c_segments.append(c[idx])
        a_segments.append(a_prime[idx])
        m_segments.append(x[idx])
        v_segments.append(vf[idx])
        dela_segments.append(dela[idx])

    m_upper, v_upper, inds_upper = upper_envelope(segments, calc_crossings=False)
    c1_env = np.zeros_like(m_upper) + np.nan
    a1_env = np.zeros_like(m_upper) + np.nan
    v1_env = np.zeros_like(m_upper) + np.nan
    d1_env = np.zeros_like(m_upper) + np.nan

    for k, c_segm in enumerate(c_segments):
        c1_env[inds_upper == k] = c_segm[m_segments[k].searchsorted(m_upper[inds_upper == k])]

    for k, a_segm in enumerate(a_segments):
        a1_env[inds_upper == k] = np.interp(m_upper[inds_upper == k], m_segments[k], a_segm)

    for k, v_segm in enumerate(v_segments):
        v1_env[inds_upper == k] = np.interp(m_upper[inds_upper == k], m_segments[k], v_segm)

    for k, dela_segm in enumerate(dela_segments):
        d1_env[inds_upper == k] = np.interp(m_upper[inds_upper == k], m_segments[k], dela_segm)

    a1_up = LinearInterp(m_upper, a1_env)
    indices = np.where(np.isin(a1_env, a_prime))[0]
    a1_env2 = a1_env[indices]
    m_upper2 = m_upper[indices]
    c_env2 = c1_env[indices]
    v_env2 = v1_env[indices]
    d_env2 = d1_env[indices]

    if plot:
        pl.close()
        fig, ax = pl.subplots(1, 2)
        sns.set(style="whitegrid", rc={"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})

        ax[0].scatter(x, vf * cp.beta - cp.delta, s=20, facecolors='none',
                      edgecolors='r', label='EGM points')
        ax[0].scatter(x_clean, vf_clean * cp.beta - cp.delta, color='blue',
                      s=15, marker='x', linewidth=0.75, label='FUES optimal points')

        ax[1].scatter(x, a_prime, edgecolors='r', s=15, facecolors='none',
                      label='EGM points', linewidth=0.75)

        for k, v_segm in enumerate(v_segments):
            x_values = m_segments[k]
            y_values = v_segm * cp.beta - cp.delta
            if len(x_values) == 1:
                ax[1].scatter(x_values, y_values, color='black', marker='x', linewidth=0.75)
            else:
                ax[1].plot(x_values, y_values, color='black', linewidth=0.75)
                ax[1].scatter([x_values[0], x_values[-1]], [y_values[0], y_values[-1]],
                              color='black', marker='x', linewidth=0.75)

        ax[1].scatter(x_clean, a_prime_clean, color='blue', s=15, marker='x',
                      label='FUES optimal points', linewidth=0.75)

        for k, v_segm in enumerate(v_segments):
            x_values = m_segments[k]
            y_values = v_segm * cp.beta - cp.delta
            if len(x_values) == 1:
                ax[0].scatter(x_values, y_values, color='black', marker='x', linewidth=0.75)
            else:
                ax[0].plot(x_values, y_values, color='black', linewidth=0.75)
                ax[0].scatter([x_values[0], x_values[-1]], [y_values[0], y_values[-1]],
                              color='black', marker='x', linewidth=0.75)

        for k, a_segm in enumerate(a_segments):
            x_values = m_segments[k]
            y_values = a_segm
            if len(x_values) == 1:
                ax[1].scatter(x_values, y_values, color='black', marker='x', linewidth=0.75)
            else:
                ax[1].plot(x_values, y_values, color='black', linewidth=0.75)
                ax[1].scatter([x_values[0], x_values[-1]], [y_values[0], y_values[-1]],
                              color='black', marker='x', linewidth=0.75)

        handles0, labels0 = ax[1].get_legend_handles_labels()
        line_x_end_handle = mlines.Line2D([0, 1], [0, 0], color='black', linestyle='-', linewidth=0.75)
        marker_start_handle = mlines.Line2D([0], [0], color='black', marker='x', linestyle='None', markersize=6)
        marker_end_handle = mlines.Line2D([0], [0], color='black', marker='x', linestyle='None', markersize=6)
        handles0.append((line_x_end_handle, marker_start_handle, marker_end_handle))
        labels0.append('DC-EGM segments')

        ax[0].set_ylim(30.5, 30.8)
        ax[0].set_xlim(55, 58.01)
        ax[1].set_ylim(20, 40)
        ax[1].set_xlim(44, 55.1)

        ax[0].tick_params(axis='y', labelsize=9)
        ax[0].tick_params(axis='x', labelsize=9)
        ax[1].tick_params(axis='y', labelsize=9)
        ax[1].tick_params(axis='x', labelsize=9)

        ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

        ax[0].set_ylabel('Value', fontsize=11)
        ax[1].set_ylabel('Financial assets at time t+1', fontsize=11)

        # Full dark-grey frame (all four spines) for each panel
        _style_axis_spines(ax[0], color="0.65", linewidth=0.8, all_sides=True)
        _style_axis_spines(ax[1], color="0.65", linewidth=0.8, all_sides=True)

        ax[0].legend(handles=handles0, labels=labels0, frameon=False, prop={'size': 10}, loc='upper left')
        ax[1].legend(handles=handles0, labels=labels0, frameon=False, prop={'size': 10}, loc='upper left')

        fig.supxlabel('Endogenous grid of financial assets at time t', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f'ret_vf_aprime_all_{g_size}_cf_{age}_{tag}.png'))

    return v_upper, v_env2, vf_clean, a_prime_clean, m_upper2, a1_env2
