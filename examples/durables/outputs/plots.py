"""Policy and EGM grid plots for durables.

Plot functions receive pre-computed data (nest, grids, savings).
No legacy wrapper object — budget-constraint derivations belong
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import os


# ── Notebook theme (aligned with retirement notebook) ──

_NB_THEMES = {
    'light': {
        'fg': '#1a1a2e', 'bg': '#fafafa', 'panel': '#ffffff',
        'grid': '#e5e7eb', 'spine': '#d1d5db', 'muted': '#64748b',
        'accent': '#4361ee', 'accent2': '#e07c3e',
        'raw': '#b23a48', 'cross': '#2ec4b6',
        'reference': '#9ca3af',
    },
    'dark': {
        'fg': '#e8e8ed', 'bg': '#16161e', 'panel': '#1e1e2a',
        'grid': '#3b4252', 'spine': '#4b5565', 'muted': '#a7b0c0',
        'accent': '#7b8cde', 'accent2': '#ffb074',
        'raw': '#ff8fab', 'cross': '#54e1d1',
        'reference': '#8b95a7',
    },
}

def _nb_theme():
    return _NB_THEMES['light']


def _build_notebook_style_block():
    """Notebook CSS + JS aligned with MkDocs Material theme."""
    light = _NB_THEMES['light']
    dark = _NB_THEMES['dark']
    return f"""
<style id="durables-notebook-theme">
  .jp-RenderedHTMLCommon, .rendered_html, .jp-RenderedMarkdown, .text_cell_render {{
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    line-height: 1.7;
    color: {light['fg']};
  }}
  .jp-RenderedHTMLCommon h1, .jp-RenderedMarkdown h1, .rendered_html h1, .text_cell_render h1 {{
    font-weight: 800;
    letter-spacing: -0.02em;
    color: {light['fg']};
  }}
  .jp-RenderedHTMLCommon h2, .jp-RenderedMarkdown h2, .rendered_html h2, .text_cell_render h2 {{
    font-weight: 700;
    letter-spacing: -0.01em;
    border-bottom: 2px solid {light['accent']};
    padding-bottom: 0.25em;
  }}
  .jp-RenderedHTMLCommon code, .jp-RenderedMarkdown code, .rendered_html code, .text_cell_render code,
  .jp-OutputArea-output pre, .output_area pre {{
    font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
  }}
  .jp-RenderedMarkdown pre, .rendered_html pre, .text_cell_render pre, .jp-OutputArea-output pre, .output_area pre {{
    background: #f4f4f8;
    border: 1px solid {light['grid']};
    border-radius: 8px;
    padding: 0.9em 1em;
  }}
  .jp-RenderedHTMLCommon blockquote, .jp-RenderedMarkdown blockquote, .rendered_html blockquote, .text_cell_render blockquote {{
    border-left: 4px solid {light['accent']};
    background: rgba(67, 97, 238, 0.03);
    padding: 0.5em 1.1em;
    margin: 1.25em 0;
  }}
  .jp-RenderedHTMLCommon table, .jp-RenderedMarkdown table, .rendered_html table, .text_cell_render table {{
    border-collapse: collapse;
    font-size: 0.82rem;
    margin: 1em auto;
    max-width: 90%;
  }}
  .jp-RenderedHTMLCommon th, .jp-RenderedMarkdown th, .rendered_html th, .text_cell_render th {{
    background: #f0f0f5;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.7rem;
    font-weight: 600;
  }}
  .jp-RenderedHTMLCommon th, .jp-RenderedMarkdown th, .rendered_html th, .text_cell_render th,
  .jp-RenderedHTMLCommon td, .jp-RenderedMarkdown td, .rendered_html td, .text_cell_render td {{
    border: none;
    border-bottom: 1px solid {light['grid']};
    padding: 0.35em 0.65em;
  }}
  .jp-RenderedHTMLCommon thead, .jp-RenderedMarkdown thead, .rendered_html thead, .text_cell_render thead {{
    border-bottom: 2px solid {light['spine']};
  }}
  @media (prefers-color-scheme: dark) {{
    .jp-RenderedHTMLCommon, .rendered_html, .jp-RenderedMarkdown, .text_cell_render {{
      color: {dark['fg']};
    }}
    .jp-RenderedHTMLCommon h2, .jp-RenderedMarkdown h2, .rendered_html h2, .text_cell_render h2 {{
      border-bottom-color: {dark['accent']};
    }}
    .jp-RenderedMarkdown pre, .rendered_html pre, .text_cell_render pre, .jp-OutputArea-output pre, .output_area pre {{
      background: {dark['panel']};
      border-color: {dark['spine']};
      color: {dark['fg']};
    }}
    .jp-RenderedHTMLCommon th, .jp-RenderedMarkdown th, .rendered_html th, .text_cell_render th {{
      background: #242433;
    }}
    .jp-RenderedHTMLCommon th, .jp-RenderedMarkdown th, .rendered_html th, .text_cell_render th,
    .jp-RenderedHTMLCommon td, .jp-RenderedMarkdown td, .rendered_html td, .text_cell_render td {{
      border-color: {dark['spine']};
      border-bottom-color: {dark['spine']};
    }}
    .jp-RenderedHTMLCommon thead, .jp-RenderedMarkdown thead, .rendered_html thead, .text_cell_render thead {{
      border-bottom-color: {dark['fg']};
    }}
  }}
</style>
"""


def setup_nb_style():
    """Apply notebook-quality rcParams + CSS (matches retirement notebook)."""
    t = _nb_theme()
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Helvetica Neue', 'Arial'],
        'font.size': 10.5,
        'axes.titlesize': 11.5,
        'axes.titleweight': '700',
        'axes.labelsize': 10,
        'axes.titlelocation': 'left',
        'figure.facecolor': t['panel'],
        'axes.facecolor': t['panel'],
        'axes.grid': True,
        'grid.alpha': 0.55,
        'grid.linewidth': 0.6,
        'grid.color': t['grid'],
        'axes.edgecolor': t['spine'],
        'axes.linewidth': 0.8,
        'text.color': t['fg'],
        'axes.labelcolor': t['fg'],
        'xtick.color': t['fg'],
        'ytick.color': t['fg'],
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.framealpha': 0.9,
        'legend.edgecolor': t['spine'],
        'legend.facecolor': t['panel'],
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.facecolor': t['panel'],
        'savefig.edgecolor': t['panel'],
    })
    try:
        from IPython.display import HTML, display
        display(HTML(_build_notebook_style_block()))
    except Exception:
        pass


def _style_nb_ax(ax):
    """Apply notebook axis styling."""
    t = _nb_theme()
    for s in ax.spines.values():
        s.set_color(t['spine'])
        s.set_linewidth(0.8)
    ax.tick_params(colors=t['fg'], labelsize=9)
    ax.set_facecolor(t['panel'])


def _adjuster_euler_series(euler_results, methods, euler_key):
    """Adjuster rows only (discrete == 1), finite errors, per method."""
    out = {}
    for method in methods:
        euler = euler_results[method][euler_key]
        discrete = euler_results[method]['sim_data']['discrete']
        mask = (discrete == 1) & np.isfinite(euler)
        out[method] = euler[mask]
    return out


def _keeper_euler_series(euler_results, methods, euler_key):
    """Keeper rows only (discrete == 0), finite errors, per method."""
    out = {}
    for method in methods:
        euler = euler_results[method][euler_key]
        discrete = euler_results[method]['sim_data']['discrete']
        mask = (discrete == 0) & np.isfinite(euler)
        out[method] = euler[mask]
    return out


def _histogram_fues_negm_ax(ax, errors_by_method, methods, colors, t, *,
                           xlabel, title, empty_msg='No observations'):
    """Draw overlapping histograms on *ax*; bins from pooled 1–99 percentiles."""
    nonempty = [errors_by_method[m] for m in methods if len(errors_by_method[m]) > 0]
    if not nonempty:
        ax.set_title(title, fontweight='bold')
        ax.text(0.5, 0.5, empty_msg, ha='center', va='center',
                transform=ax.transAxes, color=t['fg'])
        _style_nb_ax(ax)
        return
    all_adj = np.concatenate(nonempty)
    lo = np.percentile(all_adj, 1)
    hi = np.percentile(all_adj, 99)
    bins = np.linspace(lo, hi, 50)

    for method in methods:
        vals = errors_by_method[method]
        if len(vals) == 0:
            continue
        mn = np.mean(vals)
        ax.hist(vals, bins=bins, color=colors[method], alpha=0.55,
                edgecolor='white', linewidth=0.3,
                label=f'{method} (mean = {mn:.2f}, n = {len(vals):,})')
        ax.axvline(mn, color=colors[method], ls='--', lw=1.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(title, fontweight='bold')
    ax.legend(frameon=False, fontsize=8)
    _style_nb_ax(ax)


def plot_euler_histogram(euler_results, output_dir=None):
    """FUES vs NEGM Euler error histograms (1×2 when dual keys present).

    Left: keeper consumption FOC (``discrete == 0``). Right: adjuster housing
    FOC (``discrete == 1``). Otherwise a single panel: ``euler_h`` or legacy
    ``euler`` on adjuster rows.

    Parameters
    ----------
    euler_results : dict
        Per method: ``euler_c``, ``euler_h`` (optional), legacy ``euler``,
        plus ``sim_data`` with ``discrete`` (0 = keeper, 1 = adjuster).
    output_dir : str, optional
    """
    t = _nb_theme()
    methods = [m for m in ['FUES', 'NEGM'] if m in euler_results]
    colors = {'FUES': t['accent'], 'NEGM': t['accent2']}

    sample = euler_results[methods[0]] if methods else {}
    dual = methods and all(
        'euler_c' in euler_results[m] and 'euler_h' in euler_results[m]
        for m in methods)

    if dual:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        ec_keep = _keeper_euler_series(euler_results, methods, 'euler_c')
        eh_adj = _adjuster_euler_series(euler_results, methods, 'euler_h')
        _histogram_fues_negm_ax(
            axes[0], ec_keep, methods, colors, t,
            xlabel=r'log$_{10}$ consumption Euler error',
            title='Keeper consumption FOC',
            empty_msg='No keeper Euler observations')
        _histogram_fues_negm_ax(
            axes[1], eh_adj, methods, colors, t,
            xlabel=r'log$_{10}$ housing FOC error',
            title='Adjuster housing FOC',
            empty_msg='No adjuster housing FOC observations')
        fig.suptitle('Euler FOC accuracy: FUES vs NEGM', fontweight='bold', y=1.02)
        fig.tight_layout()
        fname = 'euler_adjuster_histogram_c_h.png'
    else:
        euler_key = 'euler_h' if sample.get('euler_h') is not None else 'euler'
        adj_errors = _adjuster_euler_series(euler_results, methods, euler_key)
        fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
        is_h = euler_key == 'euler_h'
        _histogram_fues_negm_ax(
            ax, adj_errors, methods, colors, t,
            xlabel=(r'log$_{10}$ housing FOC error' if is_h
                    else r'log$_{10}$ adjuster Euler error'),
            title=('Adjuster housing FOC accuracy' if is_h
                   else 'Adjuster Euler error'))
        fig.tight_layout()
        fname = 'euler_adjuster_histogram.png'

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path
    return fig


# ====================================================================
# Notebook-style multi-panel plots (return fig, caller displays)
# ====================================================================

def nb_plot_policies_comparison(results, grids, plot_t, i_z=0, i_h=None):
    """2-row x 3-col comparison: FUES (top) vs NEGM (bottom).

    Columns: keeper consumption | adjuster housing | discrete choice.

    Parameters
    ----------
    results : dict  ``{'FUES': {'nest': ...}, 'NEGM': {'nest': ...}}``
    grids : dict
    plot_t : int  Age to plot.
    i_z : int  Shock index.
    i_h : int or None  Housing index for keeper (default: mid).

    Returns
    -------
    matplotlib.figure.Figure
    """
    t = _nb_theme()
    a_grid = grids['a']
    h_grid = grids['h']
    we_grid = grids['we']
    n_h = len(h_grid)
    if i_h is None:
        i_h = n_h // 2

    methods = [m for m in ['FUES', 'NEGM'] if m in results]
    colors = {'FUES': t['accent'], 'NEGM': t['accent2']}

    fig, axes = plt.subplots(len(methods), 3, figsize=(13, 3.5 * len(methods)),
                             squeeze=False)

    for row, method in enumerate(methods):
        nest = results[method]['nest']
        sol_by_t = {s['t']: s for s in nest['solutions']}
        if plot_t not in sol_by_t:
            continue
        sol = sol_by_t[plot_t]
        col = colors[method]

        # Keeper consumption
        ax = axes[row, 0]
        c_keep = sol['keeper_cons']['dcsn']['c'][i_z, :, i_h]
        y, x = _insert_nan_at_jumps(c_keep, a_grid, 0.3)
        ax.plot(x, y, color=col, linewidth=1.4)
        ax.set_xlabel('Financial assets $a$')
        ax.set_ylabel('Consumption $c$')
        ax.set_title(f'{method}: Keeper consumption ($h={h_grid[i_h]:.0f}$)',
                     fontweight='600')
        _style_nb_ax(ax)

        # Adjuster housing
        ax = axes[row, 1]
        h_adj = sol['adjuster_cons']['dcsn']['h_choice'][i_z]
        y, x = _insert_nan_at_jumps(h_adj, we_grid, 2.0)
        ax.plot(x, y, color=col, linewidth=1.4)
        ax.set_xlabel('Total wealth $w$')
        ax.set_ylabel('Housing choice $h\'$')
        ax.set_xlim(0, we_grid[-1])
        ax.set_title(f'{method}: Adjuster housing', fontweight='600')
        _style_nb_ax(ax)

        # Discrete choice
        ax = axes[row, 2]
        adj = sol['tenure']['dcsn']['adj'][i_z, :, :]
        im = ax.pcolormesh(a_grid, h_grid, adj.T,
                           cmap='RdBu_r', vmin=0, vmax=1, shading='auto')
        ax.set_xlabel('Financial assets $a$')
        ax.set_ylabel('Housing $h$')
        ax.set_title(f'{method}: Adjust region', fontweight='600')
        _style_nb_ax(ax)

    fig.tight_layout()
    return fig


def nb_plot_keeper_ages(results, grids, ages, i_z=0, i_h=None):
    """Keeper consumption at multiple ages, FUES vs NEGM overlaid.

    Parameters
    ----------
    results : dict
    grids : dict
    ages : list of int
    """
    t = _nb_theme()
    a_grid = grids['a']
    h_grid = grids['h']
    n_h = len(h_grid)
    if i_h is None:
        i_h = n_h // 2

    methods = [m for m in ['FUES', 'NEGM'] if m in results]
    colors = {'FUES': t['accent'], 'NEGM': t['accent2']}
    styles = {'FUES': '-', 'NEGM': '--'}

    fig, axes = plt.subplots(1, len(ages), figsize=(4.3 * len(ages), 3.5))
    if len(ages) == 1:
        axes = [axes]

    for ax, age in zip(axes, ages):
        for method in methods:
            sol_by_t = {s['t']: s for s in results[method]['nest']['solutions']}
            if age not in sol_by_t:
                continue
            c = sol_by_t[age]['keeper_cons']['dcsn']['c'][i_z, :, i_h]
            y, x = _insert_nan_at_jumps(c, a_grid, 0.3)
            ax.plot(x, y, color=colors[method], ls=styles[method],
                    linewidth=1.4, label=method)
        ax.set_xlabel('Financial assets $a$')
        ax.set_ylabel('Consumption $c$')
        ax.set_title(f'Age {age} ($h = {h_grid[i_h]:.0f}$)', fontweight='600')
        ax.legend(frameon=False, fontsize=8)
        _style_nb_ax(ax)

    fig.tight_layout()
    return fig


def _tau_from_results(results, methods):
    """Transaction cost ``tau`` from the first available solved nest."""
    if not methods:
        return 0.0
    st = results[methods[0]]['nest']['periods'][0]['stages']['keeper_cons']
    return float(st.calibration['tau'])


_METHOD_LABELS = {'FUES': 'EGM(FUES)', 'NEGM': 'NEGM(FUES)'}


def nb_plot_adjuster_comparison(results, grids, plot_t, i_z=0,
                                 methods_filter=None, xlim=14,
                                 ylim_a=None, ylim_h=None):
    """Two-panel adjuster comparison: financial assets $a'$ + housing.

    Savings on the wealth grid: $a' = m - c - (1+\\tau)h'$ with $m$ on ``we``.

    Parameters
    ----------
    results : dict  ``{'FUES': {'nest': ...}, 'NEGM': {'nest': ...}}``
    grids : dict
    plot_t : int or list of int
        Single age or list of ages. Multiple ages use distinct colours (viridis);
        methods differ by linestyle.
    i_z : int
    methods_filter : list of str, optional
        Subset of methods to plot (e.g. ``['FUES']``). Default: all available.
    xlim : float
        Maximum x-axis value (adjuster wealth m).

    Returns
    -------
    matplotlib.figure.Figure
    """
    t = _nb_theme()
    we_grid = grids['we']
    all_methods = [m for m in ['FUES', 'NEGM'] if m in results]
    methods = methods_filter if methods_filter else all_methods
    methods = [m for m in methods if m in results]
    base_colors = {'FUES': t['accent'], 'NEGM': t['accent2']}
    # Solid lines when only one method shown; dashed NEGM when overlaid
    if len(methods) == 1:
        styles = {methods[0]: '-'}
    else:
        styles = {'FUES': '-', 'NEGM': '--'}
    labels = _METHOD_LABELS
    tau = _tau_from_results(results, methods)
    tau_adj = 1.0 + tau

    ages = plot_t if isinstance(plot_t, (list, tuple)) else [plot_t]
    n_age = len(ages)
    age_colors = plt.cm.viridis(np.linspace(0.2, 0.85, max(n_age, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    for method in methods:
        sol_by_t = {s['t']: s for s in results[method]['nest']['solutions']}

        for ai, age in enumerate(ages):
            if age not in sol_by_t:
                continue
            adj = sol_by_t[age]['adjuster_cons']['dcsn']
            col = age_colors[ai] if n_age > 1 else base_colors[method]
            lbl = f'age {age}' if n_age > 1 else labels.get(method, method)

            c = adj['c'][i_z]
            h = adj['h_choice'][i_z]
            a_nxt = we_grid - c - tau_adj * h
            y, x = _insert_nan_at_jumps(a_nxt, we_grid, 1.0)
            axes[0].plot(x, y, color=col, ls=styles[method],
                         linewidth=1.4, label=lbl)

            yh, xh = _insert_nan_at_jumps(h, we_grid, 2.0)
            axes[1].plot(xh, yh, color=col, ls=styles[method],
                         linewidth=1.4, label=lbl)

    title_ages = ', '.join(str(a) for a in ages)
    method_title = labels.get(methods[0], methods[0]) if len(methods) == 1 else 'FUES vs NEGM'
    axes[0].set_xlabel('Adjuster wealth $m$')
    axes[0].set_ylabel("Financial assets $a'$")
    axes[0].set_title(f'{method_title}: savings (age {title_ages})', fontweight='600')
    axes[0].set_xlim(0, xlim)
    axes[0].set_ylim(*(ylim_a if ylim_a else (0, None)))
    if n_age > 1:
        age_handles = [
            Line2D([0], [0], color=age_colors[j], lw=1.4,
                   label=f'age {ages[j]}')
            for j in range(n_age)]
        axes[0].legend(handles=age_handles, frameon=False,
                       fontsize=8, loc='upper left')
        if len(methods) > 1:
            meth_handles = [
                Line2D([0], [0], color=t['muted'], ls=styles[m],
                       lw=1.4, label=labels.get(m, m))
                for m in methods]
            leg_age = axes[0].get_legend()
            axes[0].add_artist(leg_age)
            axes[0].legend(handles=meth_handles, frameon=False,
                           fontsize=8, title='Method',
                           loc='lower right')
    _style_nb_ax(axes[0])

    axes[1].set_xlabel('Adjuster wealth $m$')
    axes[1].set_ylabel("Housing choice $h'$")
    axes[1].set_title(
        f'{method_title}: housing (age {title_ages})',
        fontweight='600')
    axes[1].set_xlim(0, xlim)
    axes[1].set_ylim(*(ylim_h if ylim_h else (0, None)))
    if n_age > 1:
        age_handles_b = [
            Line2D([0], [0], color=age_colors[j], lw=1.4,
                   label=f'age {ages[j]}')
            for j in range(n_age)]
        axes[1].legend(handles=age_handles_b, frameon=False,
                       fontsize=8, loc='upper left')
        if len(methods) > 1:
            meth_handles_b = [
                Line2D([0], [0], color=t['muted'], ls=styles[m],
                       lw=1.4, label=labels.get(m, m))
                for m in methods]
            leg_age_b = axes[1].get_legend()
            axes[1].add_artist(leg_age_b)
            axes[1].legend(handles=meth_handles_b, frameon=False,
                           fontsize=8, title='Method',
                           loc='lower right')
    _style_nb_ax(axes[1])

    fig.tight_layout()
    return fig


def nb_plot_adjuster_egm(nest, grids, plot_t=None, i_z=0, xlim=14,
                          ylim_a=None, ylim_h=None, ylim_v=None):
    """Adjuster EGM scatter: raw points + FUES refined (notebook style).

    Shows why DCEGM/LTM cannot work: the 2D EGM correspondence
    from the partial EGM (over h_choice grid) produces a dense
    cloud of crossing segments that require a full upper-envelope
    scan, not a simple kink-detection heuristic.

    Requires ``store_cntn=True`` in settings.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    t = _nb_theme()
    sol_by_t = {s['t']: s for s in nest['solutions']}
    all_t = sorted(sol_by_t.keys())
    if plot_t is None:
        plot_t = all_t[-3] if len(all_t) >= 3 else all_t[-1]

    sol = sol_by_t.get(plot_t)
    if sol is None:
        print(f'Age {plot_t} not in solution')
        return None

    adj_cntn = sol.get('adjuster_cons', {}).get('cntn')
    if adj_cntn is None or 'm_endog' not in adj_cntn:
        print('No cntn data — run with store_cntn=True')
        return None

    m_raw = adj_cntn['m_endog'][i_z]
    a_eval = adj_cntn['a_nxt_eval'][i_z]
    h_eval = adj_cntn['h_nxt_eval'][i_z]

    mask = a_eval.ravel() > 0.0
    m_pts = m_raw.ravel()[mask]
    h_pts = h_eval.ravel()[mask]
    a_pts = a_eval.ravel()[mask]

    # Refined (post-FUES)
    refined = adj_cntn.get('_refined', {})
    has_refined = i_z in refined

    # Determine sensible y-axis caps from the FUES envelope (not raw outliers)
    a_ymax = h_ymax = None
    if has_refined:
        rf = refined[i_z]
        # Cap at 1.15x the envelope max within the xlim range
        env_mask = rf['m_endog'] <= xlim
        if np.any(env_mask):
            a_ymax = np.max(rf['a_nxt_eval'][env_mask]) * 1.15
            h_ymax = np.max(rf['h_nxt_eval'][env_mask]) * 1.15

    # Raw value (unrefined)
    v_raw_arr = adj_cntn.get('v_endog')
    if v_raw_arr is not None:
        v_pts = v_raw_arr[i_z].ravel()[mask]
    else:
        v_pts = None

    # y-axis cap for value from refined envelope
    v_ymin = v_ymax = None
    if has_refined and refined[i_z].get('vf') is not None:
        rf = refined[i_z]
        env_mask = rf['m_endog'] <= xlim
        if np.any(env_mask):
            vf_vis = rf['vf'][env_mask]
            vf_finite = vf_vis[np.isfinite(vf_vis)]
            if len(vf_finite) > 0:
                span = vf_finite.max() - vf_finite.min()
                v_ymin = vf_finite.min() - 0.1 * span
                v_ymax = vf_finite.max() + 0.05 * span

    if v_pts is None:
        print('Note: v_endog not stored — re-run solve to get the value EGM grid.')
    n_panels = 3 if v_pts is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

    # Panel 1: financial assets a'
    ax = axes[0]
    ax.scatter(m_pts, a_pts, s=3, alpha=0.25, color=t['raw'],
               label='Raw EGM', rasterized=True, edgecolors='none')
    if has_refined:
        rf = refined[i_z]
        sidx = np.argsort(rf['m_endog'])
        ax.plot(rf['m_endog'][sidx], rf['a_nxt_eval'][sidx],
                color=t['accent'], linewidth=1.3, label='FUES envelope',
                zorder=5)
        ax.scatter(rf['m_endog'], rf['a_nxt_eval'],
                   s=10, color=t['accent'], marker='x', linewidth=0.7,
                   zorder=6)
    ax.set_xlabel('Endogenous wealth $\\hat{m}$')
    ax.set_ylabel("Financial assets $a'$")
    ax.set_title(f'Financial assets (age {plot_t}, $z_{{{i_z}}}$)', fontweight='600')
    ax.set_xlim(0, xlim)
    ax.set_ylim(*(ylim_a if ylim_a else (0, a_ymax)))
    ax.legend(frameon=False, fontsize=8)
    _style_nb_ax(ax)

    # Panel 2: housing
    ax = axes[1]
    ax.scatter(m_pts, h_pts, s=3, alpha=0.25, color=t['raw'],
               label='Raw EGM', rasterized=True, edgecolors='none')
    if has_refined:
        rf = refined[i_z]
        sidx = np.argsort(rf['m_endog'])
        ax.plot(rf['m_endog'][sidx], rf['h_nxt_eval'][sidx],
                color=t['accent'], linewidth=1.3, label='FUES envelope',
                zorder=5)
        ax.scatter(rf['m_endog'], rf['h_nxt_eval'],
                   s=10, color=t['accent'], marker='x', linewidth=0.7,
                   zorder=6)
    ax.set_xlabel('Endogenous wealth $\\hat{m}$')
    ax.set_ylabel("Housing choice $h'$")
    ax.set_title(f'Housing (age {plot_t}, $z_{{{i_z}}}$)', fontweight='600')
    ax.set_xlim(0, xlim)
    ax.set_ylim(*(ylim_h if ylim_h else (0, h_ymax)))
    ax.legend(frameon=False, fontsize=8)
    _style_nb_ax(ax)

    # Panel 3: CE value (raw + refined)
    if v_pts is not None:
        _rho = float(nest['periods'][0]['stages'][
            'keeper_cons'].calibration.get('rho', 2.0))
        ce_raw = _ce_transform(v_pts, _rho)
        ce_mask = np.isfinite(ce_raw) & (ce_raw > 0)
        ax = axes[2]
        ax.scatter(m_pts[ce_mask], ce_raw[ce_mask], s=3,
                   alpha=0.25, color=t['raw'],
                   label='Raw EGM', rasterized=True,
                   edgecolors='none')
        if has_refined and refined[i_z].get('vf') is not None:
            rf = refined[i_z]
            ce_ref = _ce_transform(rf['vf'], _rho)
            sidx = np.argsort(rf['m_endog'])
            ax.plot(rf['m_endog'][sidx], ce_ref[sidx],
                    color=t['accent'], linewidth=1.3,
                    label='FUES envelope', zorder=5)
            ax.scatter(rf['m_endog'], ce_ref,
                       s=10, color=t['accent'], marker='x',
                       linewidth=0.7, zorder=6)
        ax.set_xlabel('Endogenous wealth $\\hat{m}$')
        ax.set_ylabel('CE composite good')
        ax.set_title(
            f'CE value (age {plot_t}, $z_{{{i_z}}}$)',
            fontweight='600')
        ax.set_xlim(0, xlim)
        if ylim_v:
            ax.set_ylim(*ylim_v)
        elif v_ymin is not None:
            ax.set_ylim(v_ymin, v_ymax)
        ax.legend(frameon=False, fontsize=8)
        _style_nb_ax(ax)

    fig.tight_layout()
    return fig


def nb_plot_keeper_egm(nest, grids, plot_t=None, i_z=0, i_h=None, xlim=7.5,
                        ylim_c=None, ylim_a=None):
    """Keeper EGM scatter: raw endogenous grid + refined policy.

    Requires ``store_cntn=True`` and the generic EGM path.
    """
    t = _nb_theme()
    a_grid = grids['a']
    h_grid = grids['h']
    n_h = len(h_grid)
    if i_h is None:
        i_h = n_h // 2
    sol_by_t = {s['t']: s for s in nest['solutions']}
    all_t = sorted(sol_by_t.keys())
    if plot_t is None:
        plot_t = all_t[-3] if len(all_t) >= 3 else all_t[-1]
    sol = sol_by_t.get(plot_t)
    if sol is None:
        print(f'Age {plot_t} not in solution')
        return None
    keep_cntn = sol.get('keeper_cons', {}).get('cntn')
    if keep_cntn is None or 'm_endog' not in keep_cntn:
        print('No keeper cntn data — run with store_cntn=True and generic EGM path')
        return None
    m_endog = keep_cntn['m_endog']
    c_data = keep_cntn['c']
    key = (i_z, i_h)
    if key not in m_endog:
        print(f'No keeper EGM data for (z={i_z}, h={i_h})')
        return None
    m_raw = m_endog[key]
    c_raw = c_data[key]
    c_ref = sol['keeper_cons']['dcsn']['c'][i_z, :, i_h]
    hk = h_grid[i_h]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Consumption
    ax = axes[0]
    ax.scatter(m_raw, c_raw, s=4, alpha=0.35, color=t['raw'],
               label='Raw EGM', rasterized=True, edgecolors='none')
    sidx = np.argsort(a_grid)
    y, x = _insert_nan_at_jumps(c_ref[sidx], a_grid[sidx], 0.3)
    ax.plot(x, y, color=t['accent'], linewidth=1.3, label='FUES envelope', zorder=5)
    ax.set_xlabel('Cash-on-hand $w_{\\mathrm{keep}}$')
    ax.set_ylabel('Consumption $c$')
    ax.set_title(f'Keeper consumption (age {plot_t}, $z_{{{i_z}}}$, '
                 f'$h={hk:.2f}$)', fontweight='600')
    ax.set_xlim(0, xlim)
    ax.set_ylim(*(ylim_c if ylim_c else (0, None)))
    ax.legend(frameon=False, fontsize=8)
    _style_nb_ax(ax)
    # Savings
    a_raw = m_raw - c_raw
    ax = axes[1]
    ax.scatter(m_raw, a_raw, s=4, alpha=0.35, color=t['raw'],
               label='Raw EGM', rasterized=True, edgecolors='none')
    a_sav_ref = a_grid[sidx] - c_ref[sidx]
    y, x = _insert_nan_at_jumps(a_sav_ref, a_grid[sidx], 0.3)
    ax.plot(x, y, color=t['accent'], linewidth=1.3, label='FUES envelope', zorder=5)
    ax.set_xlabel('Cash-on-hand $w_{\\mathrm{keep}}$')
    ax.set_ylabel("Financial assets $a'$")
    ax.set_title(f'Keeper savings (age {plot_t}, $z_{{{i_z}}}$, '
                 f'$h={hk:.2f}$)', fontweight='600')
    ax.set_xlim(0, xlim)
    ax.set_ylim(*(ylim_a if ylim_a else (0, None)))
    ax.legend(frameon=False, fontsize=8)
    _style_nb_ax(ax)
    fig.tight_layout()
    return fig


def nb_plot_keeper_policy(results, grids, plot_t, i_z=0, i_h=None, xlim=7.5,
                          methods_filter=None, ylim_c=None, ylim_a=None):
    """Keeper consumption and savings: FUES vs NEGM, multi-age."""
    t = _nb_theme()
    a_grid = grids['a']
    h_grid = grids['h']
    n_h = len(h_grid)
    if i_h is None:
        i_h = n_h // 2
    if isinstance(plot_t, int):
        plot_t = [plot_t]
    all_methods = [m for m in ['FUES', 'NEGM'] if m in results]
    methods = methods_filter if methods_filter else all_methods
    methods = [m for m in methods if m in results]
    labels = _METHOD_LABELS
    if len(methods) == 1:
        styles = {methods[0]: '-'}
    else:
        styles = {'FUES': '-', 'NEGM': '--'}
    age_colors = plt.cm.viridis(np.linspace(0.2, 0.85, max(len(plot_t), 1)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in methods:
        sol_by_t = {s['t']: s for s in results[method]['nest']['solutions']}
        ls = styles[method]
        for ai, age in enumerate(plot_t):
            if age not in sol_by_t:
                continue
            c = sol_by_t[age]['keeper_cons']['dcsn']['c'][i_z, :, i_h]
            col = age_colors[ai]
            lbl = f'age {age}' if method == methods[0] else None
            y, x = _insert_nan_at_jumps(c, a_grid, 0.3)
            axes[0].plot(x, y, color=col, ls=ls, linewidth=1.3, label=lbl)
            a_sav = a_grid - c
            y2, x2 = _insert_nan_at_jumps(a_sav, a_grid, 0.3)
            axes[1].plot(x2, y2, color=col, ls=ls, linewidth=1.3, label=lbl)
    hk = h_grid[i_h]
    method_title = labels.get(methods[0], methods[0]) if len(methods) == 1 else 'FUES vs NEGM'
    ylims = [ylim_c, ylim_a]
    for idx, (ax, ylabel, title) in enumerate([
        (axes[0], 'Consumption $c$', f'{method_title}: keeper consumption ($h={hk:.2f}$)'),
        (axes[1], "Financial assets $a'$", f'{method_title}: keeper savings ($h={hk:.2f}$)'),
    ]):
        ax.set_xlabel('Cash-on-hand $w_{\\mathrm{keep}}$')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='600')
        ax.set_xlim(0, xlim)
        ax.set_ylim(*(ylims[idx] if ylims[idx] else (0, None)))
        _style_nb_ax(ax)
    axes[0].legend(frameon=False, fontsize=8)
    if len(methods) > 1:
        from matplotlib.lines import Line2D as _L2D
        axes[1].legend(handles=[
            _L2D([0], [0], ls='-', color='grey', lw=1, label='EGM(FUES)'),
            _L2D([0], [0], ls='--', color='grey', lw=1, label='NEGM(FUES)')],
            frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def _ce_transform(V, rho):
    """Certainty-equivalent composite good: C_eq = ((1-rho)*V)^(1/(1-rho)).

    Maps hugely negative CRRA values to positive, interpretable units.
    For rho > 1 (the common case), V < 0 and (1-rho)*V > 0.
    """
    if abs(rho - 1.0) < 1e-8:
        return np.exp(V)
    inner = (1.0 - rho) * V
    # Guard: inner must be positive for real-valued power
    safe = np.where(inner > 0, inner, np.nan)
    return np.power(safe, 1.0 / (1.0 - rho))


def nb_plot_value_functions(results, grids, plot_t, i_z=0, i_h=None,
                            xlim_keep=7.5, xlim_adj=14,
                            ylim_keep=None, ylim_adj=None):
    """Certainty-equivalent value: keeper (left) and adjuster (right).

    Plots the CE composite good ``((1-rho)*V)^(1/(1-rho))`` so the
    y-axis is positive and in interpretable consumption-equivalent units.
    Both methods overlaid for direct comparison.

    Parameters
    ----------
    results : dict
    grids : dict
    plot_t : int  Age to plot.
    """
    t = _nb_theme()
    a_grid = grids['a']
    h_grid = grids['h']
    we_grid = grids['we']
    n_h = len(h_grid)
    if i_h is None:
        i_h = n_h // 2

    methods = [m for m in ['FUES', 'NEGM'] if m in results]
    colors = {'FUES': t['accent'], 'NEGM': t['accent2']}
    labels = _METHOD_LABELS

    # Read rho from calibration
    _st = results[methods[0]]['nest']['periods'][0]['stages']['keeper_cons']
    rho = float(_st.calibration.get('rho', 2.0))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for method in methods:
        sol_by_t = {s['t']: s for s in results[method]['nest']['solutions']}
        if plot_t not in sol_by_t:
            continue
        sol = sol_by_t[plot_t]
        col = colors[method]
        lbl = labels.get(method, method)

        # Keeper VF: slice (i_z, :, i_h)
        v_keep = sol['keeper_cons']['dcsn']['V'][i_z, :, i_h]
        ce_keep = _ce_transform(v_keep, rho)
        mask_k = (a_grid <= xlim_keep)
        axes[0].plot(a_grid[mask_k], ce_keep[mask_k], color=col,
                     linewidth=1.3, label=lbl)

        # Adjuster VF: slice (i_z, :)
        v_adj = sol['adjuster_cons']['dcsn']['V'][i_z]
        ce_adj = _ce_transform(v_adj, rho)
        mask_a = (we_grid <= xlim_adj)
        axes[1].plot(we_grid[mask_a], ce_adj[mask_a], color=col,
                     linewidth=1.3, label=lbl)

    hk = h_grid[i_h]
    axes[0].set_xlabel('Cash-on-hand $w_{\\mathrm{keep}}$')
    axes[0].set_ylabel('CE composite good')
    axes[0].set_title(f'Keeper CE value (age {plot_t}, $h={hk:.2f}$)',
                      fontweight='600')
    axes[0].set_xlim(0, xlim_keep)
    axes[0].set_ylim(*(ylim_keep if ylim_keep else (0, None)))
    axes[0].legend(frameon=False, fontsize=8)
    _style_nb_ax(axes[0])

    axes[1].set_xlabel('Adjuster wealth $m$')
    axes[1].set_ylabel('CE composite good')
    axes[1].set_title(f'Adjuster CE value (age {plot_t})', fontweight='600')
    axes[1].set_xlim(0, xlim_adj)
    axes[1].set_ylim(*(ylim_adj if ylim_adj else (0, None)))
    axes[1].legend(frameon=False, fontsize=8)
    _style_nb_ax(axes[1])

    fig.tight_layout()
    return fig


def _plotly_layout_defaults():
    """Plotly layout dict matching the notebook theme."""
    t = _nb_theme()
    axis = dict(
        gridcolor=t['grid'], gridwidth=0.6,
        linecolor=t['spine'], linewidth=0.8,
        zerolinecolor=t['grid'],
        tickfont=dict(color=t['fg']),
        title=dict(font=dict(color=t['fg'])),
    )
    return dict(
        font=dict(family='Inter, Helvetica Neue, Arial, sans-serif',
                  size=11, color=t['fg']),
        plot_bgcolor=t['panel'],
        paper_bgcolor=t['panel'],
        xaxis=axis, yaxis=axis,
    )


def nb_plot_adjuster_egm_interactive(nest, grids, plot_t=None, i_z=0, xlim=14):
    """Interactive plotly adjuster EGM: a', h', and value — raw + FUES refined.

    Three separate figures you can zoom/pan independently:
    (1) financial assets a', (2) housing h', (3) value.
    Each shows the raw EGM candidates (unrefined) and the FUES-selected
    envelope (refined).

    Requires ``store_cntn=True``.

    Returns
    -------
    tuple of (fig_a, fig_h, fig_v) — plotly Figures, or None
    """
    import plotly.graph_objects as go

    t = _nb_theme()
    sol_by_t = {s['t']: s for s in nest['solutions']}
    all_t = sorted(sol_by_t.keys())
    if plot_t is None:
        plot_t = all_t[-3] if len(all_t) >= 3 else all_t[-1]

    sol = sol_by_t.get(plot_t)
    if sol is None:
        print(f'Age {plot_t} not in solution')
        return None

    adj_cntn = sol.get('adjuster_cons', {}).get('cntn')
    if adj_cntn is None or 'm_endog' not in adj_cntn:
        print('No cntn data — run with store_cntn=True')
        return None

    m_raw = adj_cntn['m_endog'][i_z]
    a_eval = adj_cntn['a_nxt_eval'][i_z]
    h_eval = adj_cntn['h_nxt_eval'][i_z]

    mask = a_eval.ravel() > 0.0
    m_pts = m_raw.ravel()[mask]
    a_pts = a_eval.ravel()[mask]
    h_pts = h_eval.ravel()[mask]

    # Raw value (unrefined) — stored per (n_he, egm_n), ravel + mask
    v_raw_arr = adj_cntn.get('v_endog')
    if v_raw_arr is not None:
        v_pts = v_raw_arr[i_z].ravel()[mask]
    else:
        v_pts = None

    refined = adj_cntn.get('_refined', {})
    has_refined = i_z in refined

    layout_kw = _plotly_layout_defaults()

    def _make_fig(title, xlabel, ylabel, x_raw, y_raw, x_ref, y_ref,
                  y_range_mode='tozero'):
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=x_raw, y=y_raw, mode='markers',
            marker=dict(size=3, color=t['raw'], opacity=0.3),
            name='Raw EGM (unrefined)',
        ))
        if x_ref is not None:
            sidx = np.argsort(x_ref)
            fig.add_trace(go.Scattergl(
                x=x_ref[sidx], y=y_ref[sidx],
                mode='lines+markers',
                line=dict(color=t['accent'], width=1.5),
                marker=dict(size=5, color=t['accent'], symbol='x'),
                name='FUES refined',
            ))
        fig.update_xaxes(title_text=xlabel, range=[0, xlim])
        fig.update_yaxes(title_text=ylabel, rangemode=y_range_mode)
        fig.update_layout(
            title=title, height=400, width=620,
            dragmode='zoom',
            margin=dict(t=50, b=40, l=60, r=30),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)',
                        borderwidth=0, font=dict(size=9)),
            **layout_kw,
        )
        fig.update_layout(template={})
        return fig

    rf_m = refined[i_z]['m_endog'] if has_refined else None
    rf_a = refined[i_z]['a_nxt_eval'] if has_refined else None
    rf_h = refined[i_z]['h_nxt_eval'] if has_refined else None
    rf_v = refined[i_z].get('vf') if has_refined else None

    fig_a = _make_fig(
        f"Adjuster financial assets a' (age {plot_t}, z={i_z})",
        'Endogenous wealth m̂', "a'",
        m_pts, a_pts, rf_m, rf_a,
    )

    fig_h = _make_fig(
        f"Adjuster housing h' (age {plot_t}, z={i_z})",
        'Endogenous wealth m̂', "h'",
        m_pts, h_pts, rf_m, rf_h,
    )

    # CE value: raw + refined
    if v_pts is not None:
        _rho = float(nest['periods'][0]['stages'][
            'keeper_cons'].calibration.get('rho', 2.0))
        ce_raw = _ce_transform(v_pts, _rho)
        ce_mask = np.isfinite(ce_raw) & (ce_raw > 0)
        ce_rf = _ce_transform(rf_v, _rho) if rf_v is not None else None
        fig_v = _make_fig(
            f"Adjuster CE value (age {plot_t}, z={i_z})",
            'Endogenous wealth m̂', 'CE',
            m_pts[ce_mask], ce_raw[ce_mask],
            rf_m, ce_rf,
        )
    else:
        # Fallback: just the refined value on the wealth grid
        we_grid = grids['we']
        v_adj = sol['adjuster_cons']['dcsn']['V'][i_z]
        _st = nest['periods'][0]['stages']['keeper_cons']
        rho = float(_st.calibration.get('rho', 2.0))
        ce_adj = _ce_transform(v_adj, rho)
        mask_w = we_grid <= xlim
        fig_v = go.Figure()
        fig_v.add_trace(go.Scattergl(
            x=we_grid[mask_w], y=ce_adj[mask_w],
            mode='lines', line=dict(color=t['accent'], width=1.5),
            name='CE value (refined)',
        ))
        fig_v.update_xaxes(title_text='Adjuster wealth m', range=[0, xlim])
        fig_v.update_yaxes(title_text='CE composite good', rangemode='tozero')
        fig_v.update_layout(
            title=f"Adjuster CE value (age {plot_t}, z={i_z})",
            height=400, width=620, dragmode='zoom',
            margin=dict(t=50, b=40, l=60, r=30), **layout_kw,
        )
        fig_v.update_layout(template={})

    return fig_a, fig_h, fig_v


def _insert_nan_at_jumps(y, x, threshold):
    """Insert NaN where |dy| > *threshold* so discrete jumps render as gaps, not diagonals.

    Threshold is in the same units as *y* (e.g. ~0.3 for keeper *c* on ``a``;
    larger for housing or financial assets on ``we``).
    """
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
        output_dir = 'results/durables/plots'

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
        y, x = _insert_nan_at_jumps(c, a_grid, 0.3)
        ax.plot(x, y, color=colors[idx % 3],
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
        y, x = _insert_nan_at_jumps(a_pol, a_grid, 0.3)
        ax.plot(x, y, color=colors[idx % 3],
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
        output_dir = 'results/durables/plots'

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

            ax[0].scatter(m_ref[1:], a_ref[1:], color='blue',
                          s=15, marker='x', linewidth=0.75,
                          label='FUES optimal points')
            ax[0].plot(m_ref[1:], a_ref[1:], linewidth=1,
                       label='Financial assets')
            ax[0].set_ylabel(r"Financial assets $a'$", fontsize=11)
            _clean_ax(ax[0])
            ax[0].yaxis.set_major_locator(MaxNLocator(6))

            ax[1].scatter(m_ref[1:], c_ref[1:], color='blue',
                          s=15, marker='x', linewidth=0.75,
                          label='FUES optimal points')
            ax[1].plot(m_ref[1:], c_ref[1:], linewidth=1,
                       label='Consumption function')
            ax[1].set_ylabel(r'Consumption $c$', fontsize=11)
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

def plot_lifecycle(sim_data, euler, nest, output_dir=None):
    """Plot lifecycle means with percentile bands from simulation.

    Produces a 2x2 panel: consumption, housing, liquid assets,
    adjustment rate by age.

    Parameters
    ----------
    sim_data : dict
        From ``simulate_lifecycle()``.
    euler : ndarray(T, N)
        Log10 Euler errors (unused, kept for API compatibility).
    nest : dict
        Solved nest; ``t0`` and ``T`` read from the first period stage.
    output_dir : str, optional
    """
    if output_dir is None:
        output_dir = 'results/durables/plots'

    stage0 = nest["periods"][0]["stages"]["keeper_cons"]
    t0 = int(stage0.calibration["t0"])
    T = int(stage0.settings["T"])
    age = np.arange(t0, T)

    t = _nb_theme()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    # --- Panels: mean + 25-75 percentile band ---
    panels = [
        ('c',  'Consumption',       '$c_t$'),
        ('h',  'Housing',           '$H_t$'),
        ('a',  'Financial assets',  '$a_t$'),
    ]

    for i, (key, title, ylabel) in enumerate(panels):
        ax = axes[i]
        data = sim_data[key][t0:T, :]
        mean = np.nanmean(data, axis=1)
        p25 = np.nanpercentile(data, 25, axis=1)
        p75 = np.nanpercentile(data, 75, axis=1)

        ax.plot(age, mean, lw=1.8, color=t['accent'])
        ax.fill_between(age, p25, p75, alpha=0.18, color=t['accent'])
        ax.set_title(title, fontweight='600')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Age')
        _style_nb_ax(ax)

    # --- Panel 4: adjustment rate ---
    ax_adj = axes[3]
    d = sim_data['discrete'][t0:T, :]
    adj_rate = np.nanmean(np.where(d >= 0, d, np.nan), axis=1)
    ax_adj.plot(age, adj_rate * 100, lw=1.8, color=t['accent2'])
    ax_adj.set_title('Adjustment rate', fontweight='600')
    ax_adj.set_ylabel('% adjusting')
    ax_adj.set_xlabel('Age')
    _style_nb_ax(ax_adj)

    fig.tight_layout()

    sim_dir = os.path.join(output_dir, 'simulation')
    os.makedirs(sim_dir, exist_ok=True)
    path = os.path.join(sim_dir, 'lifecycle.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Lifecycle plot saved to {path}')
