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
from IPython.display import HTML, display

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


# ====================================================================
# Notebook plotting functions (interactive / exploratory)
#
# These are for the Jupyter notebook walkthrough, NOT for paper
# figures.  They use plotly for interactivity and matplotlib for
# static panels.  The paper-quality functions (plot_egrids,
# plot_cons_pol, plot_dcegm_cf) are above.
# ====================================================================

# ── Notebook theme palette (aligned with MkDocs Material + extra.css) ──
_NB_THEMES = {
    'light': {
        'fg': '#1a1a2e',
        'bg': '#fafafa',
        'panel': '#ffffff',
        'grid': '#e5e7eb',
        'spine': '#d1d5db',
        'muted': '#64748b',
        'accent': '#4361ee',
        'accent_soft': '#2d3561',
        'raw': '#b23a48',
        'cross': '#2ec4b6',
        'dcegm': '#e07c3e',
        'consav': '#9b5de5',
        'reference': '#9ca3af',
    },
    'dark': {
        'fg': '#e8e8ed',
        'bg': '#16161e',
        'panel': '#1e1e2a',
        'grid': '#3b4252',
        'spine': '#4b5565',
        'muted': '#a7b0c0',
        'accent': '#7b8cde',
        'accent_soft': '#a0b0ee',
        'raw': '#ff8fab',
        'cross': '#54e1d1',
        'dcegm': '#ffb074',
        'consav': '#c4a7ff',
        'reference': '#8b95a7',
    },
}
_NB_THEME_NAME = 'light'
_METHOD_MARKERS = {'FUES': 'o', 'DCEGM': 's', 'RFC': '^', 'CONSAV': 'D'}
# Display labels for plots (API uses DCEGM/CONSAV, paper uses MSS/LTM)
_METHOD_LABELS = {'FUES': 'FUES', 'DCEGM': 'MSS', 'RFC': 'RFC', 'CONSAV': 'LTM'}

def _resolve_nb_theme(theme='auto'):
    """Resolve notebook theme from explicit choice or environment."""
    if theme in _NB_THEMES:
        return theme
    env_theme = os.environ.get('FUES_NOTEBOOK_THEME', '').strip().lower()
    if env_theme in _NB_THEMES:
        return env_theme
    # Python cannot reliably observe browser dark/light mode. Default to light,
    # while injected notebook CSS/JS handles browser-side presentation.
    return 'light'


def _nb_theme(theme_name=None):
    return _NB_THEMES[theme_name or _NB_THEME_NAME]


def _method_colors(theme_name=None):
    t = _nb_theme(theme_name)
    return {
        'FUES': t['accent'],
        'DCEGM': t['dcegm'],
        'RFC': t['cross'],
        'CONSAV': t['consav'],
    }


def _plotly_layout_defaults(theme_name=None):
    t = _nb_theme(theme_name)
    return dict(
        font=dict(
            family='Inter, Helvetica Neue, Arial, sans-serif',
            size=11,
            color=t['fg'],
        ),
        plot_bgcolor=t['panel'],
        paper_bgcolor=t['panel'],
        xaxis=dict(
            gridcolor=t['grid'],
            gridwidth=0.6,
            linecolor=t['spine'],
            linewidth=0.8,
            zerolinecolor=t['grid'],
            tickfont=dict(color=t['fg']),
            title=dict(font=dict(color=t['fg'])),
        ),
        yaxis=dict(
            gridcolor=t['grid'],
            gridwidth=0.6,
            linecolor=t['spine'],
            linewidth=0.8,
            zerolinecolor=t['grid'],
            tickfont=dict(color=t['fg']),
            title=dict(font=dict(color=t['fg'])),
        ),
    )


def _build_notebook_style_block():
    """Notebook CSS + JS aligned with the MkDocs site theme."""
    light = _NB_THEMES['light']
    dark = _NB_THEMES['dark']
    return f"""
<style id="fues-notebook-theme">
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
    font-size: 0.92rem;
  }}
  .jp-RenderedHTMLCommon th, .jp-RenderedMarkdown th, .rendered_html th, .text_cell_render th {{
    background: #f0f0f5;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.76rem;
  }}
  .jp-RenderedHTMLCommon th, .jp-RenderedMarkdown th, .rendered_html th, .text_cell_render th,
  .jp-RenderedHTMLCommon td, .jp-RenderedMarkdown td, .rendered_html td, .text_cell_render td {{
    border: 1px solid {light['grid']};
    padding: 0.5em 0.75em;
  }}
  @media (prefers-color-scheme: dark) {{
    .jp-RenderedHTMLCommon, .rendered_html, .jp-RenderedMarkdown, .text_cell_render {{
      color: {dark['fg']};
    }}
    .jp-RenderedHTMLCommon h1, .jp-RenderedMarkdown h1, .rendered_html h1, .text_cell_render h1 {{
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
    .jp-RenderedHTMLCommon blockquote, .jp-RenderedMarkdown blockquote, .rendered_html blockquote, .text_cell_render blockquote {{
      border-left-color: {dark['accent']};
      background: rgba(123, 140, 222, 0.08);
    }}
    .jp-RenderedHTMLCommon th, .jp-RenderedMarkdown th, .rendered_html th, .text_cell_render th {{
      background: #242433;
    }}
    .jp-RenderedHTMLCommon th, .jp-RenderedMarkdown th, .rendered_html th, .text_cell_render th,
    .jp-RenderedHTMLCommon td, .jp-RenderedMarkdown td, .rendered_html td, .text_cell_render td {{
      border-color: {dark['spine']};
    }}
  }}
</style>
<script>
(function() {{
  const light = {{
    fg: "{light['fg']}",
    panel: "{light['panel']}",
    grid: "{light['grid']}",
    spine: "{light['spine']}"
  }};
  const dark = {{
    fg: "{dark['fg']}",
    panel: "{dark['panel']}",
    grid: "{dark['grid']}",
    spine: "{dark['spine']}"
  }};
  function activeTheme() {{
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? dark : light;
  }}
  function applyPlotlyTheme() {{
    if (!window.Plotly) return;
    const t = activeTheme();
    document.querySelectorAll('.js-plotly-plot').forEach((gd) => {{
      try {{
        window.Plotly.relayout(gd, {{
          paper_bgcolor: t.panel,
          plot_bgcolor: t.panel,
          'font.family': 'Inter, Helvetica Neue, Arial, sans-serif',
          'font.color': t.fg,
          'xaxis.gridcolor': t.grid,
          'xaxis.linecolor': t.spine,
          'xaxis.zerolinecolor': t.grid,
          'yaxis.gridcolor': t.grid,
          'yaxis.linecolor': t.spine,
          'yaxis.zerolinecolor': t.grid,
          'legend.font.color': t.fg,
          'legend.bgcolor': 'rgba(0,0,0,0)'
        }});
      }} catch (err) {{
      }}
    }});
  }}
  const mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  if (mq) {{
    if (mq.addEventListener) mq.addEventListener('change', applyPlotlyTheme);
    else if (mq.addListener) mq.addListener(applyPlotlyTheme);
  }}
  new MutationObserver(applyPlotlyTheme).observe(document.body, {{ childList: true, subtree: true }});
  setTimeout(applyPlotlyTheme, 0);
  setTimeout(applyPlotlyTheme, 300);
}})();
</script>
"""


def setup_nb_style(theme='auto'):
    """Apply notebook styling aligned with the MkDocs theme.

    Parameters
    ----------
    theme : {"auto", "light", "dark"}
        Matplotlib theme choice. ``auto`` falls back to light unless the
        ``FUES_NOTEBOOK_THEME`` environment variable is set. Browser-side CSS
        and Plotly styling still react to dark/light mode dynamically.
    """
    import matplotlib
    global _NB_THEME_NAME
    _NB_THEME_NAME = _resolve_nb_theme(theme)
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
        display(HTML(_build_notebook_style_block()))
    except Exception:
        pass


def _style_nb_ax(ax):
    """Apply notebook axis styling to a single matplotlib axes."""
    t = _nb_theme()
    for s in ax.spines.values():
        s.set_color(t['spine'])
        s.set_linewidth(0.8)
    ax.tick_params(colors=t['fg'], labelsize=9)
    ax.set_facecolor(t['panel'])


def nb_plot_egm_interactive(nest, model, age, pad=10):
    """Interactive plotly EGM grid plot, auto-centered on crossings.

    Shows raw EGM points, FUES-refined points, value function
    line, and crossing points.  Auto-pans to the median crossing.

    Parameters
    ----------
    nest : dict
        Solved nest from :func:`solve_nest`.
    model : RetirementModel
        Model instance.
    age : int
        Age (calendar time t) to plot.
    pad : float
        Padding around the median crossing point (grid units).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from dcsmm.fues.fues import FUES as fues_alg
    from .diagnostics import get_policy
    t = _nb_theme()

    e_grid = get_policy(nest, 'egrid', stage='work_cons')
    vf_unref = get_policy(nest, 'q_hat', stage='work_cons')
    c_unref = get_policy(nest, 'c_hat', stage='work_cons')
    da_unref = get_policy(nest, 'da_pre_ue', stage='work_cons')

    x_raw = np.array(e_grid[age])
    q_raw = np.array(vf_unref[age])
    c_raw = np.array(c_unref[age])
    da_raw = np.array(da_unref[age])

    fues_result, intersections = fues_alg(
        x_raw, q_raw, c_raw, model.asset_grid_A, da_raw,
        m_bar=1.0001, include_intersections=True,
        return_intersections_separately=True,
    )
    x_clean, vf_clean, c_clean, _, _ = fues_result
    inter_e, inter_v, inter_p1, _, _ = intersections

    v_raw = q_raw * model.beta - model.delta
    v_clean = vf_clean * model.beta - model.delta
    v_inter = inter_v * model.beta - model.delta if len(inter_e) > 0 else np.array([])

    sav_raw = x_raw - c_raw
    sav_clean = x_clean - c_clean
    sav_inter = inter_e - inter_p1 if len(inter_e) > 0 else np.array([])

    center = np.median(inter_e) if len(inter_e) > 0 else np.median(x_raw)
    x_lo, x_hi = center - pad, center + pad

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=[f'Value (age {age})', f'Savings policy (age {age})'])

    # Value panel
    fig.add_trace(go.Scattergl(
        x=x_raw, y=v_raw, mode='markers',
        marker=dict(size=5, color=t['raw'], opacity=0.35, symbol='circle-open'),
        name='Raw EGM',
    ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=x_clean, y=v_clean, mode='markers',
        marker=dict(size=6, color=t['accent'], symbol='x'),
        name='FUES optimal',
    ), row=1, col=1)
    sort_idx = np.argsort(x_clean)
    fig.add_trace(go.Scattergl(
        x=x_clean[sort_idx], y=v_clean[sort_idx],
        mode='lines', line=dict(color=t['fg'], width=1.3),
        name='Value function',
    ), row=1, col=1)
    if len(inter_e) > 0:
        fig.add_trace(go.Scattergl(
            x=inter_e, y=v_inter, mode='markers',
            marker=dict(size=10, color=t['cross'], symbol='star',
                        line=dict(width=1, color=t['fg'])),
            name='Crossing points',
        ), row=1, col=1)

    # Savings panel
    sort_raw = np.argsort(x_raw)
    fig.add_trace(go.Scattergl(
        x=x_raw[sort_raw], y=sav_raw[sort_raw], mode='markers',
        marker=dict(size=5, color=t['raw'], opacity=0.35, symbol='circle-open'),
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scattergl(
        x=x_clean[sort_idx], y=sav_clean[sort_idx], mode='markers',
        marker=dict(size=6, color=t['accent'], symbol='x'),
        showlegend=False,
    ), row=1, col=2)
    if len(inter_e) > 0:
        si = np.argsort(inter_e)
        fig.add_trace(go.Scattergl(
            x=inter_e[si], y=sav_inter[si], mode='markers',
            marker=dict(size=10, color=t['cross'], symbol='star',
                        line=dict(width=1, color=t['fg'])),
            showlegend=False,
        ), row=1, col=2)

    # Auto y-axis ranges from data visible in the x-window
    y_pad_frac = 0.05  # 5% padding
    in_view = (x_raw >= x_lo) & (x_raw <= x_hi)
    if in_view.any():
        v_vis = v_raw[in_view]
        v_span = max(v_vis.max() - v_vis.min(), 1e-6)
        v_lo = v_vis.min() - y_pad_frac * v_span
        v_hi = v_vis.max() + y_pad_frac * v_span
        s_vis = sav_raw[in_view]
        s_span = max(s_vis.max() - s_vis.min(), 1e-6)
        s_lo = s_vis.min() - y_pad_frac * s_span
        s_hi = s_vis.max() + y_pad_frac * s_span
    else:
        v_lo, v_hi, s_lo, s_hi = None, None, None, None

    fig.update_xaxes(title_text='Endogenous grid (assets)', range=[x_lo, x_hi])
    fig.update_yaxes(title_text='Value', range=[v_lo, v_hi], row=1, col=1)
    fig.update_yaxes(title_text='Next-period assets', range=[s_lo, s_hi], row=1, col=2)
    fig.update_layout(
        height=450,
        dragmode='zoom',
        margin=dict(t=40, b=40),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            font=dict(size=9, color=t['fg']),
        ),
        **_plotly_layout_defaults(),
    )
    return fig


def nb_plot_cons_ages(nest, model, ages=(10, 30, 40)):
    """Plot consumption policy at multiple ages (notebook style).

    Parameters
    ----------
    nest : dict
        Solved nest.
    model : RetirementModel
        Model instance.
    ages : tuple of int
        Ages to plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .diagnostics import get_policy
    t = _nb_theme()

    c_worker = get_policy(nest, 'c', stage='work_cons')
    colors = [t['accent'], t['dcegm'], t['cross']]

    fig, axes = pl.subplots(1, len(ages), figsize=(13, 4))
    if len(ages) == 1:
        axes = [axes]

    for ax, age, col in zip(axes, ages, colors):
        cons = np.copy(c_worker[age])
        jumps = np.where(
            np.abs(np.diff(cons)) / np.diff(model.asset_grid_A) > 0.3
        )[0] + 1
        y = np.insert(cons, jumps, np.nan)
        x = np.insert(model.asset_grid_A, jumps, np.nan)
        ax.plot(x, y, color=col, linewidth=1.4)
        ax.set_xlabel('Assets $a$')
        ax.set_ylabel('Consumption $c$')
        ax.set_title(f'Age $t = {age}$', fontweight='600')
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 40)
        _style_nb_ax(ax)

    fig.tight_layout()
    return fig


def nb_plot_scaling(grid_sizes, scaling, methods=None):
    """Plot UE scaling across grid sizes (notebook style).

    Parameters
    ----------
    grid_sizes : list of int
        Grid sizes swept.
    scaling : dict
        ``{method_name: [ue_time_ms, ...]}``.
    methods : list of str, optional
        Methods to plot (default: all keys in ``scaling``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if methods is None:
        methods = list(scaling.keys())
    method_colors = _method_colors()
    t = _nb_theme()

    fig, ax = pl.subplots(figsize=(7, 4.5))
    ns = np.array(grid_sizes, dtype=float)

    for m in methods:
        ax.loglog(ns, scaling[m],
                  f'-{_METHOD_MARKERS.get(m, "o")}',
                  color=method_colors.get(m, t['muted']),
                  label=_METHOD_LABELS.get(m, m), markersize=5, linewidth=1.6)

    # Reference lines anchored at first data point
    if 'DCEGM' in scaling:
        t0 = scaling['DCEGM'][0]
        ax.loglog(ns, t0 * (ns / ns[0]), '--',
                  color=t['reference'], linewidth=0.8, label='$O(n)$')

    if 'CONSAV' in scaling:
        t0 = scaling['CONSAV'][0]
        ax.loglog(ns, t0 * (ns / ns[0])**2, ':',
                  color=t['reference'], linewidth=0.8, label='$O(n^2)$')

    if 'FUES' in scaling:
        t0 = scaling['FUES'][0]
        ax.loglog(ns, t0 * (ns / ns[0]), '--',
                  color=method_colors['FUES'], linewidth=0.7,
                  alpha=0.4, label='$O(n)$ at FUES')

    # ── Speedup reference: secondary y-axis with factor labels ──
    # Place factor ticks on the right-hand side outside the plot area,
    # anchored to the FUES time at the largest grid size.
    if 'FUES' in scaling:
        fues_last = scaling['FUES'][-1]
        ylim = ax.get_ylim()

        # Thin colored bands spanning the full x-range
        band_color = '#e8ecfa'  # very light blue
        band_color_dark = '#252540'
        for factor in [5, 10, 20, 50, 100]:
            y = fues_last * factor
            if y < ylim[0] or y > ylim[1]:
                continue
            # Subtle horizontal tick mark on the right spine
            ax.plot([ns[-1], ns[-1] * 1.06], [y, y],
                    color='#6b7280', linewidth=0.7, clip_on=False,
                    solid_capstyle='round')
            # Label outside the axis
            ax.text(ns[-1] * 1.09, y, f'{factor}x',
                    fontsize=7.5, fontweight='600', color='#4b5563',
                    va='center', ha='left', clip_on=False)

        # FUES baseline tick (1×)
        if ylim[0] <= fues_last <= ylim[1]:
            ax.plot([ns[-1], ns[-1] * 1.06], [fues_last, fues_last],
                    color=method_colors.get('FUES', t['accent']),
                    linewidth=0.9, clip_on=False, solid_capstyle='round')
            ax.text(ns[-1] * 1.09, fues_last, '1x',
                    fontsize=7.5, fontweight='700',
                    color=method_colors.get('FUES', t['accent']),
                    va='center', ha='left', clip_on=False)

        # Reference grid size note — placed in figure coords above the axis
        n_ref = int(ns[-1])
        n_label = f'{n_ref // 1000}k' if n_ref >= 1000 else str(n_ref)
        ax.annotate(f'at $n$={n_label}',
                    xy=(1.09, 1.02), xycoords='axes fraction',
                    fontsize=6, color='#9ca3af',
                    va='bottom', ha='left',
                    style='italic', annotation_clip=False)

    ax.set_xlabel('Grid size $n$')
    ax.set_ylabel('Upper envelope time (ms)')
    ax.set_title('UE scaling (log-log)')
    _style_nb_ax(ax)
    ax.legend(fontsize=8, framealpha=0.7, edgecolor='none', ncol=2,
              loc='upper left')
    # Extra right margin for the factor labels
    fig.subplots_adjust(right=0.82)
    return fig


def nb_plot_egrids(nest, model, age, pad=10, xlim=None, ylim_v=None, ylim_s=None):
    """Static EGM grid plot for notebook (value + savings policy).

    Runs FUES with intersection points and plots:
    - Red open circles: raw EGM points
    - Blue crosses: FUES-refined optimal points
    - Black line: value function through optimal points
    - Green stars: estimated crossing points

    Parameters
    ----------
    nest : dict
        Solved nest.
    model : RetirementModel
        Model instance.
    age : int
        Age (calendar time t) to plot.
    pad : float
        Padding around median crossing for auto x-range.
    xlim : tuple of (lo, hi), optional
        Override x-axis range. If None, auto-centers on crossings.
    ylim_v : tuple of (lo, hi), optional
        Override y-axis range for value panel.
    ylim_s : tuple of (lo, hi), optional
        Override y-axis range for savings panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from dcsmm.fues.fues import FUES as fues_alg
    from .diagnostics import get_policy
    t = _nb_theme()

    e_grid = get_policy(nest, 'egrid', stage='work_cons')
    vf_unref = get_policy(nest, 'q_hat', stage='work_cons')
    c_unref = get_policy(nest, 'c_hat', stage='work_cons')
    da_unref = get_policy(nest, 'da_pre_ue', stage='work_cons')

    x_raw = np.array(e_grid[age])
    q_raw = np.array(vf_unref[age])
    c_raw = np.array(c_unref[age])
    da_raw = np.array(da_unref[age])

    fues_result, intersections = fues_alg(
        x_raw, q_raw, c_raw, model.asset_grid_A, da_raw,
        m_bar=1.0001, include_intersections=True,
        return_intersections_separately=True,
    )
    x_clean, vf_clean, c_clean, _, _ = fues_result
    inter_e, inter_v, inter_p1, _, _ = intersections

    v_raw = q_raw * model.beta - model.delta
    v_clean = vf_clean * model.beta - model.delta
    v_inter = inter_v * model.beta - model.delta if len(inter_e) > 0 else np.array([])

    sav_raw = x_raw - c_raw
    sav_clean = x_clean - c_clean
    sav_inter = inter_e - inter_p1 if len(inter_e) > 0 else np.array([])

    # x-range: user override or auto-center on median crossing
    if xlim is not None:
        x_lo, x_hi = xlim
    else:
        center = np.median(inter_e) if len(inter_e) > 0 else np.median(x_raw)
        x_lo, x_hi = center - pad, center + pad

    # Auto y-ranges from visible data (5% padding)
    def _auto_ylim(x_arr, y_arr, x_lo, x_hi):
        in_view = (x_arr >= x_lo) & (x_arr <= x_hi)
        if not in_view.any():
            return None, None
        y_vis = y_arr[in_view]
        span = max(y_vis.max() - y_vis.min(), 1e-6)
        return y_vis.min() - 0.05 * span, y_vis.max() + 0.05 * span

    if ylim_v is None:
        v_lo, v_hi = _auto_ylim(x_raw, v_raw, x_lo, x_hi)
    else:
        v_lo, v_hi = ylim_v

    if ylim_s is None:
        s_lo, s_hi = _auto_ylim(x_raw, sav_raw, x_lo, x_hi)
    else:
        s_lo, s_hi = ylim_s

    fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(10, 4))

    # Value panel
    ax1.scatter(x_raw, v_raw, s=12, facecolors='none', edgecolors=t['raw'],
                linewidths=0.5, alpha=0.4, label='Raw EGM', zorder=1)
    sort_idx = np.argsort(x_clean)
    ax1.plot(x_clean[sort_idx], v_clean[sort_idx], color=t['fg'],
             linewidth=1.3, label='Value function', zorder=2)
    ax1.scatter(x_clean, v_clean, s=15, color=t['accent'], marker='x',
                linewidths=0.8, label='FUES optimal', zorder=3)
    if len(inter_e) > 0:
        ax1.scatter(inter_e, v_inter, s=60, color=t['cross'], marker='*',
                    edgecolors=t['fg'], linewidths=0.5,
                    label='Crossing points', zorder=4)
    ax1.set_xlim(x_lo, x_hi)
    if v_lo is not None:
        ax1.set_ylim(v_lo, v_hi)
    ax1.set_xlabel('Endogenous grid (assets)')
    ax1.set_ylabel('Value')
    ax1.set_title(f'Value correspondence (age {age})')
    ax1.legend(fontsize=8, framealpha=0.7, edgecolor='none')
    _style_nb_ax(ax1)

    # Savings panel
    ax2.scatter(np.sort(x_raw), np.take(sav_raw, np.argsort(x_raw)),
                s=12, facecolors='none', edgecolors=t['raw'],
                linewidths=0.5, alpha=0.4, label='Raw EGM', zorder=1)
    ax2.scatter(x_clean[sort_idx], sav_clean[sort_idx],
                s=15, color=t['accent'], marker='x',
                linewidths=0.8, label='FUES optimal', zorder=3)
    if len(inter_e) > 0:
        si = np.argsort(inter_e)
        ax2.scatter(inter_e[si], sav_inter[si], s=60, color=t['cross'],
                    marker='*', edgecolors=t['fg'], linewidths=0.5,
                    label='Crossing points', zorder=4)
    ax2.set_xlim(x_lo, x_hi)
    if s_lo is not None:
        ax2.set_ylim(s_lo, s_hi)
    ax2.set_xlabel('Endogenous grid (assets)')
    ax2.set_ylabel('Next-period assets')
    ax2.set_title(f'Savings policy (age {age})')
    ax2.legend(fontsize=8, framealpha=0.7, edgecolor='none')
    _style_nb_ax(ax2)

    fig.tight_layout()
    return fig
