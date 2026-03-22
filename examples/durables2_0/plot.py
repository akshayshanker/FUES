"""Policy plots for the durables2_0 DDSL pipeline.

Format matches examples/durables/plot.py (plot_pols).
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
from matplotlib.ticker import FormatStrFormatter
import os


def _insert_nan_at_jumps(y, x, threshold):
    """Insert NaN where |dy| > threshold for clean line plots."""
    pos = np.where(np.abs(np.diff(y)) > threshold)[0] + 1
    return np.insert(y, pos, np.nan), np.insert(x, pos, np.nan)


def plot_policies(nest, grids, output_dir=None,
                  plot_t=None, i_h_index=None):
    """Plot keeper and adjuster policies.

    Parameters
    ----------
    nest : dict
        From ``solve()``.
    grids : dict
    output_dir : str, optional
    plot_t : int, optional
        Central age for plotting. Plots t-2, t-1, t.
    i_h_index : int, optional
        Housing grid index for keeper/value plots.
    """
    if output_dir is None:
        output_dir = 'plots_durables2_0'

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

    # ================================================================
    # 1. Adjuster housing policy
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        h_pol = new_by_t[t]['adjuster_cons']['dcsn']['h'][i_z]
        y, x = _insert_nan_at_jumps(h_pol, we_grid, 2.0)
        ax.plot(x, y, color=colors[idx % 3],
                label=f't = {t}', linewidth=1)
    ax.set_xlabel(r'Total wealth at time $t$', fontsize=11)
    ax.set_ylabel(r'Housing assets at time $t+1$', fontsize=11)
    ax.set_xlim(0, we_grid[-1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.grid(True)
    ax.legend(frameon=False, prop={'size': 10})
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'policy_adj_housing.pdf'))
    plt.close(fig)

    # ================================================================
    # 2. Adjuster financial assets policy
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        a_pol = new_by_t[t]['adjuster_cons']['dcsn']['a'][i_z]
        y, x = _insert_nan_at_jumps(a_pol, we_grid, 1.0)
        ax.plot(x, y, color=colors[idx % 3],
                label=f't = {t}', linewidth=0.75)
    ax.set_xlabel(r'Total wealth at time $t$', fontsize=11)
    ax.set_ylabel(r'End of period financial assets', fontsize=11)
    ax.set_xlim(0, 50)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.grid(True)
    ax.legend(frameon=False, prop={'size': 10})
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'policy_adj_assets.pdf'))
    plt.close(fig)

    # ================================================================
    # 3. Value function (keeper V on asset grid)
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        V = new_by_t[t]['keeper_cons']['dcsn']['V'][i_z, :, i_h_index]
        ax.plot(a_grid, V, color=colors[idx % 3],
                label=f't = {t}', linewidth=1)
    ax.set_xlabel(r'Financial assets at time $t$', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True)
    ax.legend(frameon=False, prop={'size': 10})
    ax.set_title(f'Keeper value ($h = {h_grid[i_h_index]:.1f}$)')
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'value_keeper.pdf'))
    plt.close(fig)

    # ================================================================
    # 4. Keeper consumption
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        c = new_by_t[t]['keeper_cons']['dcsn']['c'][i_z, :, i_h_index]
        ax.plot(a_grid, c, color=colors[idx % 3],
                label=f't = {t}', linewidth=1)
    ax.set_xlabel(r'Financial assets at time $t$', fontsize=11)
    ax.set_ylabel(r'Consumption $c$', fontsize=11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True)
    ax.legend(frameon=False, prop={'size': 10})
    ax.set_title(f'Keeper consumption ($h = {h_grid[i_h_index]:.1f}$)')
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'policy_keeper_consumption.pdf'))
    plt.close(fig)

    # ================================================================
    # 5. Keeper savings
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        a_pol = new_by_t[t]['keeper_cons']['dcsn']['a'][i_z, :, i_h_index]
        ax.plot(a_grid, a_pol, color=colors[idx % 3],
                label=f't = {t}', linewidth=1)
    ax.set_xlabel(r'Financial assets at time $t$', fontsize=11)
    ax.set_ylabel(r'End of period financial assets', fontsize=11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True)
    ax.legend(frameon=False, prop={'size': 10})
    ax.set_title(f'Keeper savings ($h = {h_grid[i_h_index]:.1f}$)')
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'policy_keeper_assets.pdf'))
    plt.close(fig)

    # ================================================================
    # 6. Adjuster consumption
    # ================================================================
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        c_pol = new_by_t[t]['adjuster_cons']['dcsn']['c'][i_z]
        y, x = _insert_nan_at_jumps(c_pol, we_grid, 1.0)
        ax.plot(x, y, color=colors[idx % 3],
                label=f't = {t}', linewidth=0.75)
    ax.set_xlabel(r'Total wealth at time $t$', fontsize=11)
    ax.set_ylabel(r'Consumption $c$', fontsize=11)
    ax.set_xlim(0, 50)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True)
    ax.legend(frameon=False, prop={'size': 10})
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'policy_adj_consumption.pdf'))
    plt.close(fig)

    # ================================================================
    # 7. Discrete choice heatmap
    # ================================================================
    t_plot = available[-1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    d = new_by_t[t_plot]['tenure']['dcsn']['d'][i_z, :, :]
    im = ax.pcolormesh(a_grid, h_grid, d.T,
                       cmap='RdBu', vmin=0, vmax=1, shading='auto')
    ax.set_xlabel(r'Financial assets $a$', fontsize=11)
    ax.set_ylabel(r'Housing $h$', fontsize=11)
    ax.set_title(f'Discrete choice (t = {t_plot})')
    ax.tick_params(labelsize=9)
    fig.colorbar(im, ax=ax, label=r'$d$ (0=keep, 1=adjust)')
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'discrete_choice.pdf'))
    plt.close(fig)

    print(f'Plots saved to {age_dir}/')


if __name__ == '__main__':
    from .solve import solve
    nest, cp, grids, callables = solve(
        'examples/durables2_0/syntax', verbose=False)
    plot_policies(nest, grids)
