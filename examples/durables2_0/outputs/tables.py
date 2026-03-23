"""Comparison table generation for durables2_0 results.

Produces markdown and LaTeX tables for method comparison
(timing, Euler accuracy, adjustment rate) from results_row dicts.
"""

import os


_PARAM_ORDER = [
    'r', 'beta', 'T', 'alpha', 'gamma_c', 'gamma_h',
    'delta', 'kappa', 'tau', 'K', 'grid_max_A', 'm_bar',
]

_PARAM_LABELS_MD = {
    'r': 'r', 'beta': 'β', 'T': 'T',
    'alpha': 'α', 'gamma_c': 'γ_c', 'gamma_h': 'γ_h',
    'delta': 'δ', 'kappa': 'κ', 'tau': 'τ', 'K': 'K',
    'grid_max_A': 'A_max', 'm_bar': 'm̄',
}

_PARAM_LABELS_TEX = {
    'r': 'r', 'beta': r'$\beta$', 'T': 'T',
    'alpha': r'$\alpha$', 'gamma_c': r'$\gamma_c$',
    'gamma_h': r'$\gamma_h$', 'delta': r'$\delta$',
    'kappa': r'$\kappa$', 'tau': r'$\tau$', 'K': 'K',
    'grid_max_A': r'$A_{\max}$', 'm_bar': r'$\bar{m}$',
}

_TIMING_COLS = ['Method', 'Keeper (ms)', 'Adj (ms)', 'Total (ms)']

_EULER_COLS = ['Euler Combined', 'Euler Keeper', 'Euler Adjuster',
               'Adj Rate']

_EULER_LOG_COLS = {'Euler Combined', 'Euler Keeper', 'Euler Adjuster'}


def _has_euler(rows):
    return any('Euler Combined' in r for r in rows)


def _fmt_param(key, val, latex=False):
    labels = _PARAM_LABELS_TEX if latex else _PARAM_LABELS_MD
    label = labels.get(key, key)
    if isinstance(val, float) and abs(val) < 1e-6:
        return f"{label}={val:.1e}"
    if isinstance(val, float):
        return f"{label}={val:.4g}"
    return f"{label}={val}"


def _param_line(params, latex=False):
    if not params:
        return ""
    parts = [_fmt_param(k, params[k], latex)
             for k in _PARAM_ORDER if k in params]
    if not parts:
        return ""
    if latex:
        return "Model parameters: " + ", ".join(parts)
    return "**Model Parameters:** " + ", ".join(parts)


def _cell(val, col):
    """Format a single cell value."""
    if val is None or val == '':
        return ''
    if col in _EULER_LOG_COLS:
        return f"{val:.4f}"
    if col == 'Adj Rate':
        return f"{val:.1f}%"
    if isinstance(val, float):
        return f"{val:.1f}"
    return str(val)


def _md_table(rows, cols, caption, params):
    widths = {c: len(c) for c in cols}
    formatted = []
    for row in rows:
        cells = {c: _cell(row.get(c, ''), c) for c in cols}
        for c in cols:
            widths[c] = max(widths[c], len(cells[c]))
        formatted.append(cells)

    lines = []
    if caption:
        lines.append(f"# {caption}\n")

    hdr = '| ' + ' | '.join(c.ljust(widths[c]) for c in cols) + ' |'
    sep = '|' + '|'.join('-' * (widths[c] + 2) for c in cols) + '|'
    lines.extend([hdr, sep])

    for cells in formatted:
        line = '| ' + ' | '.join(
            cells[c].rjust(widths[c]) for c in cols) + ' |'
        lines.append(line)

    if params:
        lines.append('\n---')
        lines.append(_param_line(params))

    return '\n'.join(lines)


def _tex_table(rows, cols, caption, params):
    has_euler = any(c in _EULER_LOG_COLS for c in cols)
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    if caption:
        lines.append(r'\caption{' + caption + '}')
    lines.append(r'\label{tab:durables_comparison}')

    col_spec = 'l' + 'r' * (len(cols) - 1)
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    tex_headers = []
    for c in cols:
        if c in _EULER_LOG_COLS:
            label = c.replace('Euler ', '')
            tex_headers.append(r'$\log_{10}$ ' + label)
        else:
            tex_headers.append(c)
    lines.append(' & '.join(tex_headers) + r' \\')
    lines.append(r'\midrule')

    for row in rows:
        cells = [_cell(row.get(c, ''), c) for c in cols]
        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    if params:
        lines.append(r'\vspace{0.5em}')
        lines.append(r'\footnotesize{' + _param_line(params, latex=True) + '}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def generate_comparison_table(rows, fmt='md', caption=None, params=None):
    """Generate a method comparison table (timing + optional Euler).

    Parameters
    ----------
    rows : list of dict
        Each dict is a results_row from ``run_single``.
        Required keys: ``'Method'``, ``'Keeper (ms)'``, ``'Adj (ms)'``,
        ``'Total (ms)'``.
        Optional Euler keys: ``'Euler Combined'``, ``'Euler Keeper'``,
        ``'Euler Adjuster'``, ``'Adj Rate'``.
    fmt : str
        ``'md'`` for markdown, ``'tex'`` for LaTeX.
    caption : str, optional
    params : dict, optional
        Model parameters to show in footer.

    Returns
    -------
    str
        Formatted table.
    """
    cols = list(_TIMING_COLS)
    if _has_euler(rows):
        cols.extend(_EULER_COLS)

    if fmt == 'tex':
        return _tex_table(rows, cols, caption, params)
    return _md_table(rows, cols, caption, params)
