"""Run the durables2_0 DDSL pipeline and plot policies.

Also runs the original solver for comparison plots.
"""

from .solve import solve
from .plot import plot_policies


def _plot_original(cp, grids, plot_t, output_dir='plots_original'):
    """Run original solver and plot in same format."""
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter
    from examples.durables.durables_plot import solveEGM
    from .plot import _insert_nan_at_jumps

    old = solveEGM(cp, verbose=False)
    a_grid = grids['a']
    h_grid = grids['h']
    we_grid = grids['we']
    n_h = len(h_grid)
    i_h = n_h // 2
    i_z = 0

    available = [t for t in range(plot_t - 2, plot_t + 1) if t in old]
    colors = ['blue', 'red', 'green']

    sns.set(style="white", rc={
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})

    age_dir = os.path.join(output_dir, f'age_{plot_t}')
    os.makedirs(age_dir, exist_ok=True)

    # Adjuster housing
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        y, x = _insert_nan_at_jumps(old[t]['Hadj'][i_z], we_grid, 2.0)
        ax.plot(x, y, color=colors[idx % 3], label=f't = {t}', linewidth=1)
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

    # Adjuster assets
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        y, x = _insert_nan_at_jumps(old[t]['Aadj'][i_z], we_grid, 1.0)
        ax.plot(x, y, color=colors[idx % 3], label=f't = {t}', linewidth=0.75)
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

    # Keeper value
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        # old stores keeper V as 'Vadj'
        ax.plot(a_grid, old[t]['Vadj'][i_z, :, i_h],
                color=colors[idx % 3], label=f't = {t}', linewidth=1)
    ax.set_xlabel(r'Financial assets at time $t$', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True)
    ax.legend(frameon=False, prop={'size': 10})
    ax.set_title(f'Keeper value ($h = {h_grid[i_h]:.1f}$)')
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'value_keeper.pdf'))
    plt.close(fig)

    # Keeper consumption
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        ax.plot(a_grid, old[t]['Ckeeper'][i_z, :, i_h],
                color=colors[idx % 3], label=f't = {t}', linewidth=1)
    ax.set_xlabel(r'Financial assets at time $t$', fontsize=11)
    ax.set_ylabel(r'Consumption $c$', fontsize=11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True)
    ax.legend(frameon=False, prop={'size': 10})
    ax.set_title(f'Keeper consumption ($h = {h_grid[i_h]:.1f}$)')
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'policy_keeper_consumption.pdf'))
    plt.close(fig)

    # Keeper savings
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for idx, t in enumerate(available):
        ax.plot(a_grid, old[t]['Akeeper'][i_z, :, i_h],
                color=colors[idx % 3], label=f't = {t}', linewidth=1)
    ax.set_xlabel(r'Financial assets at time $t$', fontsize=11)
    ax.set_ylabel(r'End of period financial assets', fontsize=11)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True)
    ax.legend(frameon=False, prop={'size': 10})
    ax.set_title(f'Keeper savings ($h = {h_grid[i_h]:.1f}$)')
    fig.tight_layout()
    fig.savefig(os.path.join(age_dir, 'policy_keeper_assets.pdf'))
    plt.close(fig)

    # Discrete choice
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    t_plot = available[-1]
    d = old[t_plot]['D'][i_z, :, :]
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

    print(f'Original plots saved to {age_dir}/')


def run(syntax_dir='examples/durables2_0/syntax',
        output_dir='plots_durables2_0', verbose=True,
        compare=False):
    nest, cp, grids, callables = solve(
        syntax_dir, verbose=verbose)
    print(f'{len(nest["solutions"])} periods solved')

    all_t = sorted(s['t'] for s in nest['solutions'])
    plot_t = all_t[-3] if len(all_t) >= 3 else all_t[-1]

    plot_policies(nest, grids, output_dir=output_dir, plot_t=plot_t)

    if compare:
        _plot_original(cp, grids, plot_t,
                       output_dir='plots_original')

    return nest, cp, grids, callables


if __name__ == '__main__':
    import sys
    compare = '--compare' in sys.argv
    run(compare=compare)
