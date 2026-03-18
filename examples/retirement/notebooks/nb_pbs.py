"""PBS cluster timing comparison for the retirement notebook.

Parses a timing markdown file produced by ``run.py --run-timings``
and plots the results. Kept separate from ``outputs/plots.py``
because it's ad-hoc analysis, not reusable infrastructure.
"""

import numpy as np
from pathlib import Path


def parse_timing_md(path):
    """Parse a retirement_timing.md file into per-method means.

    Parameters
    ----------
    path : str or Path
        Path to the markdown timing table.

    Returns
    -------
    grid_sizes : list of int
        Sorted grid sizes found in the table.
    means : dict
        ``{method: [mean_ue_ms, ...]}``, one entry per grid size.
        Methods: FUES, DCEGM (was MSS), CONSAV (was LTM), RFC.
    """
    path = Path(path)
    col_methods = ['FUES', 'DCEGM', 'CONSAV', 'RFC']
    data = {}  # {grid: {method: [ue_times]}}
    current_grid = None

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith('|') or 'Grid' in line or '---' in line:
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) < 10:
            continue
        if parts[0] and not parts[0].startswith('0'):
            try:
                current_grid = int(parts[0])
            except ValueError:
                continue
        if current_grid is None:
            continue
        if current_grid not in data:
            data[current_grid] = {m: [] for m in col_methods}
        try:
            # UE columns: 2=FUES, 4=MSS->DCEGM, 6=LTM->CONSAV, 8=RFC
            ue_vals = [float(parts[2]), float(parts[4]),
                       float(parts[6]), float(parts[8])]
            for m, v in zip(col_methods, ue_vals):
                data[current_grid][m].append(v)
        except (ValueError, IndexError):
            continue

    grid_sizes = sorted(data.keys())
    means = {m: [np.mean(data[g][m]) for g in grid_sizes]
             for m in col_methods}
    return grid_sizes, means


def plot_pbs_scaling(path, ax=None):
    """Plot PBS timing results with reference lines.

    Parameters
    ----------
    path : str or Path
        Path to the markdown timing table.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from examples.retirement.outputs.plots import (
        _method_colors, _METHOD_MARKERS, _METHOD_LABELS, _style_nb_ax,
    )

    mc = _method_colors()

    grid_sizes, means = parse_timing_md(path)
    ns = np.array(grid_sizes, dtype=float)
    methods = ['FUES', 'DCEGM', 'CONSAV', 'RFC']

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    for m in methods:
        ax.loglog(ns, means[m], f'-{_METHOD_MARKERS.get(m, "o")}',
                  color=mc.get(m, 'gray'),
                  label=_METHOD_LABELS.get(m, m), markersize=6, linewidth=1.8)

    # Reference lines anchored at first point
    t0_lin = means['DCEGM'][0]
    ax.loglog(ns, t0_lin * (ns / ns[0]), '--',
              color='#9ca3af', linewidth=0.8, label='$O(n)$')

    t0_quad = means['CONSAV'][0]
    ax.loglog(ns, t0_quad * (ns / ns[0])**2, ':',
              color='#9ca3af', linewidth=0.8, label='$O(n^2)$')

    t0_fues = means['FUES'][0]
    ax.loglog(ns, t0_fues * (ns / ns[0]), '--',
              color=mc['FUES'], linewidth=0.7,
              alpha=0.4, label='$O(n)$ at FUES')

    # ── Speedup factor ticks on the right margin ──
    fues_last = means['FUES'][-1]
    ylim = ax.get_ylim()
    tick_ys = []
    for factor in [5, 10, 20, 50, 100]:
        y = fues_last * factor
        if y < ylim[0] or y > ylim[1]:
            continue
        tick_ys.append(y)
        ax.plot([ns[-1], ns[-1] * 1.06], [y, y],
                color='#6b7280', linewidth=0.7, clip_on=False,
                solid_capstyle='round')
        ax.text(ns[-1] * 1.09, y, f'{factor}\u00d7',
                fontsize=7.5, fontweight='600', color='#4b5563',
                va='center', ha='left', clip_on=False)
    if ylim[0] <= fues_last <= ylim[1]:
        tick_ys.append(fues_last)
        ax.plot([ns[-1], ns[-1] * 1.06], [fues_last, fues_last],
                color=mc['FUES'], linewidth=0.9, clip_on=False,
                solid_capstyle='round')
        ax.text(ns[-1] * 1.09, fues_last, '1\u00d7',
                fontsize=7.5, fontweight='700', color=mc['FUES'],
                va='center', ha='left', clip_on=False)
    if tick_ys:
        top_tick = max(tick_ys)
        ax.text(ns[-1] * 1.09, top_tick * 1.6,
                f'slowdown\nvs FUES',
                fontsize=6.5, fontweight='600', color='#4b5563',
                ha='left', va='bottom',
                clip_on=False, linespacing=1.2)

    import matplotlib.ticker as ticker

    # x-axis: explicit grid-size ticks in thousands
    xticks = [1000, 2000, 3000, 5000, 10000, 15000]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{x // 1000}k' for x in xticks])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(ns[0] * 0.85, ns[-1] * 1.08)

    # y-axis: sparse ms ticks
    yticks = [0.1, 1, 10, 100]
    vis_yticks = [y for y in yticks if ax.get_ylim()[0] <= y <= ax.get_ylim()[1]]
    ax.set_yticks(vis_yticks)
    ax.set_yticklabels([f'{y:g}' for y in vis_yticks])
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.grid(True, which='major', axis='y', alpha=0.12, linewidth=0.4)
    ax.grid(False, axis='x')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Grid size (number of EGM points)')
    ax.set_ylabel('Upper-envelope time per period (ms)')
    ax.set_title('Retirement choice model — upper-envelope scaling',
                 fontsize=10)
    _style_nb_ax(ax)
    ax.legend(fontsize=7, framealpha=0.7, edgecolor='none', ncol=2,
              loc='upper left')
    fig.subplots_adjust(right=0.82)
    return fig
