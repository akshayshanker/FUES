"""Comparison table generation for durables2_0 results.

Produces markdown and LaTeX tables for method comparison
(timing, Euler accuracy, adjustment rate) from results_row dicts.
"""

import math
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

_EULER_COLS = [
    'Euler c (keeper)', 'Euler c (adj)', 'Euler c (all)',
    'Euler h (keeper)', 'Euler h (adj)', 'Euler h (all)',
    'Adj Rate',
]

_EULER_LOG_COLS = {
    'Euler c (keeper)', 'Euler c (adj)', 'Euler c (all)',
    'Euler h (keeper)', 'Euler h (adj)', 'Euler h (all)',
    # legacy column names (still formatted as log10 means if present)
    'Euler Keeper (c FOC)', 'Euler Adjuster (h FOC)',
    'Euler Combined', 'Euler Keeper', 'Euler Adjuster', 'Euler Housing',
}


def _has_euler(rows):
    keys = (
        'Euler c (keeper)', 'Euler c (adj)', 'Euler c (all)',
        'Euler h (keeper)', 'Euler h (adj)', 'Euler h (all)',
        'Euler Keeper (c FOC)', 'Euler Adjuster (h FOC)',
        'Euler Combined', 'Euler Keeper', 'Euler Adjuster', 'Euler Housing',
    )
    return any(any(k in r for k in keys) for r in rows)


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
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return '—'
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


def generate_sweep_table(results_summary, fmt='tex', caption=None):
    """Paper-quality sweep table: grid × tau rows, FUES vs NEGM column groups.

    Adapted from the legacy durables plotting helper ``create_latex_table``.

    Parameters
    ----------
    results_summary : list of dict
        Each dict has keys: ``'Grid_Size'``, ``'Tau'``, ``'Method'``,
        ``'Avg_Keeper_ms'``, ``'Avg_Adj_ms'``,
        ``'Euler_Combined'``, ``'Euler_Keeper'``, ``'Euler_Adjuster'``.
        Optional HD variants: ``'Euler_Combined_HD'``, etc.
        Optional parameter keys: ``'beta'``, ``'gamma_c'``, etc.
    fmt : str
        ``'tex'`` for LaTeX, ``'md'`` for markdown.
    caption : str, optional

    Returns
    -------
    str
    """
    if fmt == 'md':
        return _sweep_table_md(results_summary, caption)
    return _sweep_table_tex(results_summary, caption)


def _sweep_table_tex(results_summary, caption=None):
    """LaTeX sweep table with booktabs, method column groups."""
    results_summary = sorted(results_summary,
                             key=lambda x: (x['Grid_Size'], x['Tau']))

    grid_sizes = sorted(set(r['Grid_Size'] for r in results_summary))
    tau_values = sorted(set(r['Tau'] for r in results_summary))

    def get_result(grid_size, tau, method):
        return next((r for r in results_summary
                     if r['Grid_Size'] == grid_size
                     and r['Tau'] == tau
                     and r['Method'] == method), None)

    table = "% Requires: \\usepackage{booktabs}\n"
    table += "\\begin{table}[htbp]\n\\centering\n"
    table += "\\begingroup\n\\small\n"
    table += "\\setlength{\\tabcolsep}{3pt}\n"
    table += "\\renewcommand{\\arraystretch}{1.05}\n"
    if caption:
        table += f"\\caption{{{caption}}}\n"
    else:
        table += "\\caption{Durables Model: Per-Period Timing and Accuracy}\n"
    table += "\\label{tab:durables_timing}\n"
    table += "\\begin{tabular}{@{}rr rrrrr rrrrr@{}}\n"
    table += "\\toprule\n"

    # Header row 1: method groups
    table += (
        "& & "
        "\\multicolumn{5}{c}{FUES} & "
        "\\multicolumn{5}{c}{NEGM} \\\\\n"
    )
    table += "\\cmidrule(lr){3-7} \\cmidrule(lr){8-12}\n"

    # Header row 2: Time and Euler subgroups
    table += (
        "& & "
        "\\multicolumn{2}{c}{Time} & \\multicolumn{3}{c}{Euler} & "
        "\\multicolumn{2}{c}{Time} & \\multicolumn{3}{c}{Euler} \\\\\n"
    )
    table += (
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-7} "
        "\\cmidrule(lr){8-9} \\cmidrule(lr){10-12}\n"
    )

    # Header row 3: individual column labels
    table += (
        "$N_A$ & $\\tau$ & "
        "Keep & Adj & Comb & Keep & Adj & "
        "Keep & Adj & Comb & Keep & Adj \\\\\n"
    )
    table += "\\midrule\n"

    current_grid = None
    for grid_size in grid_sizes:
        taus_for_grid = [t for t in tau_values
                         if get_result(grid_size, t, 'FUES') is not None]

        for i, tau in enumerate(taus_for_grid):
            fues = get_result(grid_size, tau, 'FUES')
            negm = get_result(grid_size, tau, 'NEGM')
            if fues is None or negm is None:
                continue

            if i == 0:
                if current_grid is not None:
                    table += "\\midrule\n"
                grid_str = f"{grid_size:,}"
                current_grid = grid_size
            else:
                grid_str = ""

            def _e(r, key):
                return r.get(key + '_HD', r.get(key, 0.0))

            table += (
                f"{grid_str} & {tau} & "
                f"{fues.get('Avg_Keeper_ms', 0):.0f} & "
                f"{fues.get('Avg_Adj_ms', 0):.0f} & "
                f"{_e(fues, 'Euler_Combined'):.2f} & "
                f"{_e(fues, 'Euler_Keeper'):.2f} & "
                f"{_e(fues, 'Euler_Adjuster'):.2f} & "
                f"{negm.get('Avg_Keeper_ms', 0):.0f} & "
                f"{negm.get('Avg_Adj_ms', 0):.0f} & "
                f"{_e(negm, 'Euler_Combined'):.2f} & "
                f"{_e(negm, 'Euler_Keeper'):.2f} & "
                f"{_e(negm, 'Euler_Adjuster'):.2f} "
                "\\\\\n"
            )

    table += "\\bottomrule\n\\end{tabular}\n"
    table += "\\vspace{0.3em}\n"

    # Parameter footnote
    if results_summary:
        p = results_summary[0]
        parts = []
        for k, tex in [('beta', r'\beta'), ('gamma_c', r'\gamma_c'),
                        ('gamma_h', r'\gamma_h'), ('alpha', r'\alpha'),
                        ('delta', r'\delta'), ('r', 'r'),
                        ('r_H', 'r_H'), ('phi_w', r'\rho_w'),
                        ('sigma_w', r'\sigma_w')]:
            if k in p:
                parts.append(f"${tex}={p[k]:.2g}$")
        if parts:
            table += (
                "\\par\\small \\textit{Notes:} "
                "Time in ms; Euler errors in $\\log_{10}$ scale. "
                f"Parameters: {', '.join(parts)}.\n"
            )

    table += "\\endgroup\n\\end{table}\n"
    return table


def _sweep_table_md(results_summary, caption=None):
    """Markdown sweep table with method column groups."""
    results_summary = sorted(results_summary,
                             key=lambda x: (x['Grid_Size'], x['Tau']))

    grid_sizes = sorted(set(r['Grid_Size'] for r in results_summary))
    tau_values = sorted(set(r['Tau'] for r in results_summary))

    def get_result(grid_size, tau, method):
        return next((r for r in results_summary
                     if r['Grid_Size'] == grid_size
                     and r['Tau'] == tau
                     and r['Method'] == method), None)

    lines = []
    if caption:
        lines.append(f"# {caption}\n")

    # Header
    lines.append(
        "| N_A | τ | "
        "FUES Keep | FUES Adj | FUES Euler | "
        "NEGM Keep | NEGM Adj | NEGM Euler |")
    lines.append(
        "|----:|---:|"
        "----------:|----------:|----------:|"
        "----------:|----------:|----------:|")

    for grid_size in grid_sizes:
        taus_for_grid = [t for t in tau_values
                         if get_result(grid_size, t, 'FUES') is not None]
        for i, tau in enumerate(taus_for_grid):
            fues = get_result(grid_size, tau, 'FUES')
            negm = get_result(grid_size, tau, 'NEGM')
            if fues is None or negm is None:
                continue
            grid_str = f"{grid_size:,}" if i == 0 else ""

            def _e(r, key):
                return r.get(key + '_HD', r.get(key, 0.0))

            lines.append(
                f"| {grid_str} | {tau} | "
                f"{fues.get('Avg_Keeper_ms', 0):.0f} | "
                f"{fues.get('Avg_Adj_ms', 0):.0f} | "
                f"{_e(fues, 'Euler_Combined'):.2f} | "
                f"{negm.get('Avg_Keeper_ms', 0):.0f} | "
                f"{negm.get('Avg_Adj_ms', 0):.0f} | "
                f"{_e(negm, 'Euler_Combined'):.2f} |")

    return '\n'.join(lines)


def generate_comparison_table(rows, fmt='md', caption=None, params=None):
    """Generate a method comparison table (timing + optional Euler).

    Parameters
    ----------
    rows : list of dict
        Each dict is a results_row from ``run_single``.
        Required keys: ``'Method'``, ``'Keeper (ms)'``, ``'Adj (ms)'``,
        ``'Total (ms)'``.
        Optional Euler keys: ``'Euler c (keeper|adj|all)'``, ``'Euler h (keeper|adj|all)'``
        (log10 means), and ``'Adj Rate'``. Legacy keys ``'Euler Keeper (c FOC)'`` /
        ``'Euler Adjuster (h FOC)'`` are still recognised for formatting.
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


def format_euler_detail(euler_stats_by_method):
    """Format Euler error detail as markdown.

    Parameters
    ----------
    euler_stats_by_method : dict
        {method_name: {'consumption': ec_stats, 'housing': eh_stats}}

    Returns
    -------
    str
        Markdown-formatted Euler detail report.
    """
    detail_lines = ['# Euler Error Detail\n']

    def _append_euler_block(title, es):
        detail_lines.append(f'### {title}\n')
        if 'combined' not in es:
            detail_lines.append(
                f"Mean: {es['mean']:.4f}, "
                f"Median: {es['median']:.4f}\n\n")
            return
        detail_lines.append(
            f"Adjustment rate: {es['pct_adjuster']:.2f}% "
            f"({es['n_adjuster']} adj, {es['n_keeper']} keep)\n")
        hdr = f"{'':12} {'Combined':>12} {'Adjuster':>12} {'Keeper':>12}"
        detail_lines.append(hdr)
        detail_lines.append('-' * 60)
        for key in ['mean', 'median', 'std', 'p5', 'p95']:
            detail_lines.append(
                f"{key:12} {es['combined'][key]:>12.4f} "
                f"{es['adjuster'][key]:>12.4f} "
                f"{es['keeper'][key]:>12.4f}")
        detail_lines.append(
            f"{'Frac>10^-3':12} "
            f"{es['combined']['frac_above_neg3']:>12.4f} "
            f"{es['adjuster']['frac_above_neg3']:>12.4f} "
            f"{es['keeper']['frac_above_neg3']:>12.4f}")
        detail_lines.append(
            f"{'Frac>10^-4':12} "
            f"{es['combined']['frac_above_neg4']:>12.4f} "
            f"{es['adjuster']['frac_above_neg4']:>12.4f} "
            f"{es['keeper']['frac_above_neg4']:>12.4f}")
        detail_lines.append(
            f"{'N obs':12} {es['combined']['n_obs']:>12} "
            f"{es['adjuster']['n_obs']:>12} "
            f"{es['keeper']['n_obs']:>12}")
        detail_lines.append('')

    for method, es in euler_stats_by_method.items():
        detail_lines.append(f'## {method}\n')
        ec = es.get('consumption')
        eh = es.get('housing')
        if ec is not None:
            _append_euler_block('Consumption Euler (c FOC)', ec)
        if eh is not None:
            _append_euler_block('Housing FOC (adjusters)', eh)

    return '\n'.join(detail_lines)


def write_euler_detail(euler_stats_by_method, output_dir):
    """Write Euler detail markdown to output_dir/euler_detail.md."""
    content = format_euler_detail(euler_stats_by_method)
    path = os.path.join(output_dir, 'euler_detail.md')
    with open(path, 'w') as f:
        f.write(content)
    return path
