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
        _NORD, _METHOD_COLORS, _METHOD_MARKERS, _style_nb_ax,
    )

    grid_sizes, means = parse_timing_md(path)
    ns = np.array(grid_sizes, dtype=float)
    methods = ['FUES', 'DCEGM', 'CONSAV', 'RFC']

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    for m in methods:
        ax.loglog(ns, means[m], f'-{_METHOD_MARKERS.get(m, "o")}',
                  color=_METHOD_COLORS.get(m, 'gray'),
                  label=m, markersize=6, linewidth=1.8)

    # Reference lines anchored at first point
    t0_lin = means['DCEGM'][0]
    ax.loglog(ns, t0_lin * (ns / ns[0]), '--',
              color='#9ca3af', linewidth=0.8, label='$O(n)$')

    t0_quad = means['CONSAV'][0]
    ax.loglog(ns, t0_quad * (ns / ns[0])**2, ':',
              color='#9ca3af', linewidth=0.8, label='$O(n^2)$')

    t0_fues = means['FUES'][0]
    ax.loglog(ns, t0_fues * (ns / ns[0]), '--',
              color=_METHOD_COLORS['FUES'], linewidth=0.7,
              alpha=0.4, label='$O(n)$ at FUES')

    ax.set_xlabel('Grid size $n$')
    ax.set_ylabel('Mean UE time (ms, avg over $\\delta$)')
    ax.set_title('PBS cluster (Gadi) scaling')
    _style_nb_ax(ax)
    ax.legend(fontsize=7, framealpha=0.7, edgecolor='none', ncol=2)
    fig.tight_layout()
    return fig
