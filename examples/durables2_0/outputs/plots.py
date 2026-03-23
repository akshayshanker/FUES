"""Policy and EGM grid plots for durables2_0.

Plot functions receive pre-computed data (nest, grids, savings).
No model objects (cp) — budget-constraint derivations belong
in diagnostics.derive_savings, called by the orchestrator.
"""

import warnings
import logging

warnings.filterwarnings('ignore', message='.*Font family.*not found.*')
warnings.filterwarnings('ignore', message='.*findfont.*')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import os


def _insert_nan_at_jumps(y, x, threshold):
    """Insert NaN where |dy| > threshold for clean line plots."""
    pos = np.where(np.abs(np.diff(y)) > threshold)[0] + 1
    return np.insert(y, pos, np.nan), np.insert(x, pos, np.nan)


def _clean_ax(ax):
    for sp in ['right', 'top', 'left', 'bottom']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True)
    ax.legend(frameon=False, prop={'size': 10})


def plot_policies(nest, grids, savings, output_dir=None,
                  plot_t=None, i_h_index=None):
    """Plot keeper and adjuster policies.

    Parameters
    ----------
    nest : dict
        From ``solve()``.
    grids : dict
    savings : dict
        From ``diagnostics.derive_savings``.
        ``{t: {'keeper': (n_z,n_a,n_h), 'adjuster': (n_z,n_w)}}``.
    output_dir : str, optional
    plot_t : int, optional
        Central age for plotting. Plots t-2, t-1, t.
    i_h_index : int, optional
        Housing grid index for keeper/value plots.
    """
    if output_dir is None:
        output_dir = 'results/durables2_0/plots'

    a_grid = grids['a']
    h_grid = grids['h']
    we_grid = grids['we']
    n_h = len(h_grid)

    new_by_t = {s['t']: s for s in nest['solutions']}
    all_t = sorted(new_by_t.keys())

    if plot_t is None:
        plot_t = all_t[-3] if len(all_t) >= 3 else all_t[-1]
    if i_h_index is None:
        i_h_index = n_h // 2

    available = [t for t in range(plot_t - 2, plot_t + 1)
                 if t in new_by_t]
    colors = ['blue', 'red', 'green']
    i_z = 0

    sns.set(style="white", rc={
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})

    age_dir = os.path.join(output_dir, f'age_{plot_t}')
    os.makedirs(age_dir, exist_ok=True)

    # 1. Adjuster housing
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        h_pol = new_by_t[t]['adjuster_cons']['dcsn']['h_choice'][i_z]
        y, x = _insert_nan_at_jumps(h_pol, we_grid, 2.0)
        ax.plot(x, y, color=colors[idx % 3],
                label=f't = {t}', linewidth=1)
    ax.set_xlabel(r'Total wealth at time $t$', fontsize=11)
    ax.set_ylabel(r'Housing assets at time $t+1$', fontsize=11)
    ax.set_xlim(0, we_grid[-1])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    _clean_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'policy_adj_housing.png'))
    plt.close(fig)

    # 2. Adjuster savings
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        a_pol = savings[t]['adjuster'][i_z]
        y, x = _insert_nan_at_jumps(a_pol, we_grid, 1.0)
        ax.plot(x, y, color=colors[idx % 3],
                label=f't = {t}', linewidth=0.75)
    ax.set_xlabel(r'Total wealth at time $t$', fontsize=11)
    ax.set_ylabel(r'End of period financial assets', fontsize=11)
    ax.set_xlim(0, 50)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    _clean_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'policy_adj_assets.png'))
    plt.close(fig)

    # 3. Keeper value
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        V = new_by_t[t]['keeper_cons']['dcsn']['V'][i_z, :, i_h_index]
        ax.plot(a_grid, V, color=colors[idx % 3],
                label=f't = {t}', linewidth=1)
    ax.set_xlabel(r'Financial assets at time $t$', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Keeper value ($h = {h_grid[i_h_index]:.1f}$)')
    _clean_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'value_keeper.png'))
    plt.close(fig)

    # 4. Keeper consumption
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        c = new_by_t[t]['keeper_cons']['dcsn']['c'][i_z, :, i_h_index]
        ax.plot(a_grid, c, color=colors[idx % 3],
                label=f't = {t}', linewidth=1)
    ax.set_xlabel(r'Financial assets at time $t$', fontsize=11)
    ax.set_ylabel(r'Consumption $c$', fontsize=11)
    ax.set_title(f'Keeper consumption ($h = {h_grid[i_h_index]:.1f}$)')
    _clean_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'policy_keeper_consumption.png'))
    plt.close(fig)

    # 5. Keeper savings
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        a_pol = savings[t]['keeper'][i_z, :, i_h_index]
        ax.plot(a_grid, a_pol, color=colors[idx % 3],
                label=f't = {t}', linewidth=1)
    ax.set_xlabel(r'Financial assets at time $t$', fontsize=11)
    ax.set_ylabel(r'End of period financial assets', fontsize=11)
    ax.set_title(f'Keeper savings ($h = {h_grid[i_h_index]:.1f}$)')
    _clean_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'policy_keeper_assets.png'))
    plt.close(fig)

    # 6. Adjuster consumption
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        c_pol = new_by_t[t]['adjuster_cons']['dcsn']['c'][i_z]
        y, x = _insert_nan_at_jumps(c_pol, we_grid, 1.0)
        ax.plot(x, y, color=colors[idx % 3],
                label=f't = {t}', linewidth=0.75)
    ax.set_xlabel(r'Total wealth at time $t$', fontsize=11)
    ax.set_ylabel(r'Consumption $c$', fontsize=11)
    ax.set_xlim(0, 50)
    _clean_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'policy_adj_consumption.png'))
    plt.close(fig)

    # 7. Discrete choice
    t_plot = available[-1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    adj_arr = new_by_t[t_plot]['tenure']['dcsn']['adj'][i_z, :, :]
    im = ax.pcolormesh(a_grid, h_grid, adj_arr.T,
                       cmap='RdBu', vmin=0, vmax=1, shading='auto')
    ax.set_xlabel(r'Financial assets $a$', fontsize=11)
    ax.set_ylabel(r'Housing $h$', fontsize=11)
    ax.set_title(f'Discrete choice (t = {t_plot})')
    ax.tick_params(labelsize=9)
    fig.colorbar(im, ax=ax, label=r'adj (0=keep, 1=adjust)')
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'discrete_choice.png'))
    plt.close(fig)

    print(f'Plots saved to {age_dir}/')


def plot_grids(nest, grids, output_dir=None, plot_t=None, i_z=0):
    """Plot EGM endogenous grids (unrefined vs FUES-refined).

    Matches the original paper format.
    Requires ``store_cntn=True`` and ``return_grids=True``.

    Parameters
    ----------
    nest : dict
    grids : dict
    output_dir : str, optional
    plot_t : int, optional
    i_z : int
    """
    if output_dir is None:
        output_dir = 'results/durables2_0/plots'

    new_by_t = {s['t']: s for s in nest['solutions']}
    all_t = sorted(new_by_t.keys())

    if plot_t is None:
        plot_t = all_t[-3] if len(all_t) >= 3 else all_t[-1]

    sol = new_by_t.get(plot_t)
    if sol is None:
        print(f"Age {plot_t} not in solution.")
        return

    age_dir = os.path.join(output_dir, f'age_{plot_t}')
    os.makedirs(age_dir, exist_ok=True)

    h_grid = grids['h']
    n_h = len(h_grid)

    # --- Keeper EGM grids ---
    keeper_cntn = sol.get('keeper_cons', {}).get('cntn')
    if keeper_cntn and 'm_endog' in keeper_cntn:
        i_h = n_h // 2
        key = (i_z, i_h)
        if key in keeper_cntn['m_endog']:
            m_ref = keeper_cntn['m_endog'][key]
            c_ref = keeper_cntn['c'][key]
            a_ref = m_ref - c_ref

            sns.set(style="white", rc={
                "font.size": 9, "axes.titlesize": 9,
                "axes.labelsize": 9})

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))

            ax[0].scatter(m_ref[1:], c_ref[1:], color='blue',
                          s=15, marker='x', linewidth=0.75,
                          label='FUES optimal points')
            ax[0].plot(m_ref[1:], c_ref[1:], linewidth=1,
                       label='Consumption function')
            ax[0].set_ylabel(r'Consumption $c$', fontsize=11)
            _clean_ax(ax[0])
            ax[0].yaxis.set_major_locator(MaxNLocator(6))

            ax[1].scatter(m_ref[1:], a_ref[1:], color='blue',
                          s=15, marker='x', linewidth=0.75,
                          label='FUES optimal points')
            ax[1].plot(m_ref[1:], a_ref[1:], linewidth=1,
                       label='Savings function')
            ax[1].set_ylabel(r"Savings $a'$", fontsize=11)
            _clean_ax(ax[1])
            ax[1].yaxis.set_major_locator(MaxNLocator(6))

            fig.supxlabel(r'Endogenous grid $\hat{m}$', fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(age_dir, 'egm_keeper.png'))
            plt.close(fig)
            print(f'Keeper EGM plot saved to {age_dir}/egm_keeper.png')

    # --- Adjuster EGM grids (paper format) ---
    adj_cntn = sol.get('adjuster_cons', {}).get('cntn')
    if adj_cntn and 'm_endog' in adj_cntn:
        m_raw = adj_cntn['m_endog'][i_z]
        a_eval = adj_cntn['a_nxt_eval'][i_z]
        h_eval = adj_cntn['h_nxt_eval'][i_z]

        mask = a_eval.ravel() > 0.0
        m_unref = m_raw.ravel()[mask]
        h_unref = h_eval.ravel()[mask]

        refined = adj_cntn.get('_refined', {}).get(i_z)

        # Plot 1: VF + housing (hous_vf_aprime_all_small)
        sns.set(style="white", rc={
            "font.size": 9, "axes.titlesize": 9,
            "axes.labelsize": 9})

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        if refined is not None:
            m_cl = refined['m_endog']
            vf_cl = refined['vf']
            h_cl = refined['h_nxt_eval']

            ax[0].scatter(m_cl[1:], vf_cl[1:], color='blue',
                          s=15, marker='x', linewidth=0.75,
                          label='FUES optimal points')
            ax[0].plot(m_cl[1:], vf_cl[1:], linewidth=1,
                       label='Value function')

        ax[0].set_ylabel('Value', fontsize=11)
        _clean_ax(ax[0])
        ax[0].yaxis.set_major_locator(MaxNLocator(6))
        ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

        ax[1].scatter(m_unref, h_unref, s=20, facecolors='none',
                      edgecolors='r', label='EGM points')
        if refined is not None:
            ax[1].scatter(m_cl, h_cl, s=20, color='blue',
                          marker='x', linewidth=0.75,
                          label='FUES optimal points')
        ax[1].set_ylabel(r'Housing assets at time $t+1$', fontsize=11)
        _clean_ax(ax[1])
        ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

        fig.supxlabel(r'Total wealth at time $t$', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(
            age_dir, 'hous_vf_aprime_all_small.png'))
        plt.close(fig)

        # Plot 2: h' vs m scatter (hous_vf_aprime_all_big)
        sns.set(style="white", rc={
            "font.size": 11, "axes.titlesize": 11,
            "axes.labelsize": 11})

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        ax.scatter(h_unref[1:], m_unref[1:], s=20,
                   facecolors='none', edgecolors='r',
                   label='EGM points')
        if refined is not None:
            ax.scatter(h_cl[1:], m_cl[1:], color='blue',
                       s=15, marker='x', linewidth=0.75,
                       label='FUES optimal points')
        ax.set_xlabel(
            r'Exogenous grid of housing assets at time $t+1$',
            fontsize=11)
        ax.set_ylabel(
            r'Endogenous grid of total wealth at time $t$',
            fontsize=11)
        _clean_ax(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(
            age_dir, 'hous_vf_aprime_all_big.png'))
        plt.close(fig)

        print(f'Adjuster EGM plots saved to {age_dir}/')


# ------------------------------------------------------------------
# Simulation lifecycle plots
# ------------------------------------------------------------------

def plot_lifecycle(sim_data, euler, cp, output_dir=None):
    """Plot lifecycle means with percentile bands from simulation.

    Produces a 3x2 panel: income, consumption, housing, liquid
    assets, adjustment rate, and Euler errors by age.

    Parameters
    ----------
    sim_data : dict
        From ``euler_errors()``.
    euler : ndarray(T, N)
        Log10 Euler errors.
    cp : ConsumerProblem
        For ``t0``, ``T``.
    output_dir : str, optional
    """
    if output_dir is None:
        output_dir = 'results/durables2_0/plots'

    t0 = cp.t0
    T = cp.T
    T_end = T - 1
    age = np.arange(t0, T)

    sns.set_style("white")

    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    axes = axes.flatten()

    # --- Panels: mean + 25-75 percentile band ---
    panels = [
        ('y',  'Income',        '$y_t$'),
        ('c',  'Consumption',   '$c_t$'),
        ('h',  'Housing',       '$h_t$'),
        ('a',  'Liquid Assets', '$a_t$'),
    ]

    for i, (key, title, ylabel) in enumerate(panels):
        ax = axes[i]
        data = sim_data[key][t0:T, :]
        mean = np.nanmean(data, axis=1)
        p25 = np.nanpercentile(data, 25, axis=1)
        p75 = np.nanpercentile(data, 75, axis=1)

        ax.plot(age, mean, lw=2, color='C0')
        ax.fill_between(age, p25, p75, alpha=0.2, color='C0')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        _clean_ax(ax)

    # --- Panel 5: adjustment rate ---
    ax_adj = axes[4]
    d = sim_data['discrete'][t0:T, :]
    adj_rate = np.nanmean(np.where(d >= 0, d, np.nan), axis=1)
    ax_adj.plot(age, adj_rate * 100, lw=2, color='C1')
    ax_adj.set_title('Adjustment Rate')
    ax_adj.set_ylabel('% adjusting')
    ax_adj.set_xlabel('Age')
    _clean_ax(ax_adj)

    # --- Panel 6: Euler errors ---
    ax_eu = axes[5]
    eu = euler[t0:T, :]
    eu_mean = np.nanmean(eu, axis=1)
    eu_p25 = np.nanpercentile(eu, 25, axis=1)
    eu_p75 = np.nanpercentile(eu, 75, axis=1)
    valid_mask = ~np.isnan(eu_mean)
    if np.any(valid_mask):
        ax_eu.plot(age[valid_mask], eu_mean[valid_mask],
                   lw=2, color='C2')
        ax_eu.fill_between(
            age[valid_mask],
            eu_p25[valid_mask], eu_p75[valid_mask],
            alpha=0.2, color='C2')
    ax_eu.set_title('Euler Errors')
    ax_eu.set_ylabel('log$_{10}$ error')
    ax_eu.set_xlabel('Age')
    _clean_ax(ax_eu)

    fig.tight_layout()

    sim_dir = os.path.join(output_dir, 'simulation')
    os.makedirs(sim_dir, exist_ok=True)
    path = os.path.join(sim_dir, 'lifecycle.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Lifecycle plot saved to {path}')
