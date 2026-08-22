"""Comparison table generation for durables results.

Produces markdown and LaTeX tables for method comparison
(timing, Euler accuracy, adjustment rate) from results_row dicts.
"""

import math
import os
import numpy as np


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
    """LaTeX sweep table — exact paper format (Table 2 in Dobrescu & Shanker).

    5 columns per method: NA(time) | Adj(time) | Comb | NA | Adj
    where Euler columns use the adjuster HOUSING Euler (Euler_H_Adjuster)
    for the Adj column, matching the paper's definition.

    Discovers method names from the data.
    """
    results_summary = sorted(results_summary,
                             key=lambda x: (x['Grid_Size'], x['Tau']))

    methods = sorted(set(r['Method'] for r in results_summary))
    grid_sizes = sorted(set(r['Grid_Size'] for r in results_summary))
    tau_values = sorted(set(r['Tau'] for r in results_summary))

    by_key = {}
    for r in results_summary:
        by_key[(r['Grid_Size'], r['Tau'], r['Method'])] = r

    def _ef(v):
        """Format Euler value; NaN → ---."""
        return f"{v:.2f}" if not np.isnan(v) else "---"

    n_methods = len(methods)
    col_spec = 'rr' + ' rrrrr' * n_methods

    table = "% Requires: \\usepackage{booktabs}\n"
    table += "\\begin{table}[htbp]\n\\centering\n"
    table += "\\begingroup\n\\small\n"
    table += "\\setlength{\\tabcolsep}{3pt}\n"
    table += "\\renewcommand{\\arraystretch}{1.05}\n"
    if caption:
        table += f"\\caption{{{caption}}}\n"
    else:
        table += "\\caption{Durables Model: Per-Period Timing and Accuracy}\n"
    table += "\\label{table:housing1}\n"
    table += "\\begin{tabular}{@{}" + col_spec + "@{}}\n"
    table += "\\toprule\n"

    # Header row 1: method groups
    method_headers = []
    cmidrules = []
    col_offset = 3
    for m in methods:
        method_headers.append(f"\\multicolumn{{5}}{{c}}{{{m}}}")
        cmidrules.append(
            f"\\cmidrule(lr){{{col_offset}-{col_offset + 4}}}")
        col_offset += 5
    table += "& & " + " & ".join(method_headers) + " \\\\\n"
    table += " ".join(cmidrules) + "\n"

    # Header row 2: Time and Euler subgroups
    time_euler = []
    te_rules = []
    col_offset = 3
    for m in methods:
        time_euler.append(
            f"\\multicolumn{{2}}{{c}}{{Time}} & \\multicolumn{{3}}{{c}}{{Euler}}")
        te_rules.append(
            f"\\cmidrule(lr){{{col_offset}-{col_offset + 1}}} "
            f"\\cmidrule(lr){{{col_offset + 2}-{col_offset + 4}}}")
        col_offset += 5
    table += "& & " + " & ".join(time_euler) + " \\\\\n"
    table += " ".join(te_rules) + "\n"

    # Header row 3: individual column labels
    col_labels = ["$N_A$ & $\\tau$"]
    for m in methods:
        col_labels.append("NA & Adj & Comb & NA & Adj")
    table += " & ".join(col_labels) + " \\\\\n"
    table += "\\midrule\n"

    # Data rows
    current_grid = None
    for grid_size in grid_sizes:
        taus_for_grid = [t for t in tau_values
                         if all((grid_size, t, m) in by_key for m in methods)]

        for i, tau in enumerate(taus_for_grid):
            if i == 0:
                if current_grid is not None:
                    table += "\\midrule\n"
                grid_str = f"{grid_size:,}"
                current_grid = grid_size
            else:
                grid_str = ""

            cells = [f"{grid_str} & {tau}"]
            for m in methods:
                r = by_key[(grid_size, tau, m)]
                keeper_ms = r.get('Avg_Keeper_ms', 0.0)
                adj_ms = r.get('Avg_Adj_ms', 0.0)
                euler_comb = r.get('Euler_Combined', np.nan)
                euler_keep = r.get('Euler_Keeper', np.nan)
                # Paper's "Adj" Euler = housing FOC for adjusters
                euler_adj = r.get('Euler_H_Adjuster', np.nan)
                cells.append(
                    f"{keeper_ms:.0f} & {adj_ms:.0f} & "
                    f"{_ef(euler_comb)} & {_ef(euler_keep)} & {_ef(euler_adj)}")
            table += " & ".join(cells) + " \\\\\n"

    table += "\\bottomrule\n\\end{tabular}\n"
    table += "\\vspace{0.3em}\n"

    # Parameter notes (exact paper format)
    if results_summary:
        p = results_summary[0]
        param_str = (
            f"$\\beta={p.get('beta', 0.93):.2f}$, "
            f"$\\gamma_c={p.get('gamma_c', 3):.1f}$, "
            f"$\\gamma_h={p.get('gamma_h', 3):.1f}$, "
            f"$\\alpha={p.get('alpha', 0.7):.2f}$, "
            f"$\\delta={p.get('delta', 0.0):.2f}$, "
            f"$r={p.get('r', 0.01):.2f}$, "
            f"$r_H={p.get('r_H', 0.0):.2f}$, "
            f"$\\rho_w={p.get('phi_w', 0.9):.2f}$, "
            f"$\\sigma_w={p.get('sigma_w', 0.08):.2f}$, "
            f"$N_{{sim}}={p.get('N_sim', 10000):,}$"
        )
        table += (
            "\\par\\small \\textit{Notes:} "
            "Time in ms; Euler errors in $\\log_{10}$ scale. "
            "\\textbf{Comb}: combined mean. "
            "\\textbf{NA}: non-adjusters. "
            "\\textbf{Adj}: adjusters. "
            f"Parameters: {param_str}.\n"
        )

    table += "\\endgroup\n\\end{table}\n"
    return table


def _sweep_table_md(results_summary, caption=None):
    """Markdown sweep table with method column groups.

    Discovers method names from the data rather than hard-coding.
    """
    results_summary = sorted(results_summary,
                             key=lambda x: (x['Grid_Size'], x['Tau']))

    methods = sorted(set(r['Method'] for r in results_summary))
    grid_sizes = sorted(set(r['Grid_Size'] for r in results_summary))
    tau_values = sorted(set(r['Tau'] for r in results_summary))

    by_key = {}
    for r in results_summary:
        by_key[(r['Grid_Size'], r['Tau'], r['Method'])] = r

    def _e(r, key):
        return r.get(key + '_HD', r.get(key, 0.0))

    lines = []
    if caption:
        lines.append(f"# {caption}\n")

    hdr_parts = ["| N_A | τ |"]
    sep_parts = ["|----:|---:|"]
    for m in methods:
        hdr_parts.append(f" {m} Keep | {m} Adj | {m} Euler(c) | {m} Euler(h) |")
        sep_parts.append("----------:|----------:|----------:|----------:|")
    lines.append(" ".join(hdr_parts))
    lines.append("".join(sep_parts))

    for grid_size in grid_sizes:
        taus_for_grid = [t for t in tau_values
                         if all((grid_size, t, m) in by_key for m in methods)]
        for i, tau in enumerate(taus_for_grid):
            grid_str = f"{grid_size:,}" if i == 0 else ""
            cells = [f"| {grid_str} | {tau} |"]
            for m in methods:
                r = by_key[(grid_size, tau, m)]
                eh_adj = r.get('Euler_H_Adjuster', float('nan'))
                eh_str = f"{eh_adj:.2f}" if not math.isnan(eh_adj) else "---"
                cells.append(
                    f" {r.get('Avg_Keeper_ms', 0):.0f} | "
                    f"{r.get('Avg_Adj_ms', 0):.0f} | "
                    f"{_e(r, 'Euler_Combined'):.2f} | "
                    f"{eh_str} |")
            lines.append(" ".join(cells))

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


def generate_vertical_comparison(rows, caption=None):
    """Vertical comparison table: metrics as rows, methods as columns.

    Designed for notebooks where horizontal space is limited.

    Parameters
    ----------
    rows : list of dict
        Each dict has keys: ``'Method'``, timing keys, Euler keys, ``'Adj Rate'``.
    caption : str, optional

    Returns
    -------
    str
        Markdown table.
    """
    methods = [r['Method'] for r in rows]
    by_method = {r['Method']: r for r in rows}

    # Build metric rows: (label, format_fn)
    metric_rows = []

    # Timing section
    metric_rows.append(('**Timing**', None))
    metric_rows.append(('Keeper (ms/period)', lambda r: f"{r.get('Keeper (ms)', 0):.0f}"))
    metric_rows.append(('Adjuster (ms/period)', lambda r: f"{r.get('Adj (ms)', 0):.0f}"))
    metric_rows.append(('Total (ms/period)', lambda r: f"{r.get('Total (ms)', 0):.0f}"))

    # Euler section
    has_euler = any(
        any(k in r for k in _EULER_COLS) for r in rows)
    if has_euler:
        metric_rows.append(('**Euler errors ($\\log_{10}$)**', None))
        euler_metrics = [
            ('Consumption: keeper', 'Euler c (keeper)'),
            ('Consumption: adjuster', 'Euler c (adj)'),
            ('Consumption: all', 'Euler c (all)'),
            ('Housing: adjuster', 'Euler h (adj)'),
        ]
        for label, key in euler_metrics:
            if any(key in r for r in rows):
                def _fmt(r, k=key):
                    v = r.get(k)
                    if v is None or v == '':
                        return '\u2014'
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        return '\u2014'
                    return f'{v:.2f}'
                metric_rows.append((label, _fmt))

    # Simulation section
    sim_keys = {'Adj Rate', 'CE Utility', 'Mean Consumption', 'Mean Housing',
                'Mean Financial Assets'}
    if any(any(k in r for k in sim_keys) for r in rows):
        metric_rows.append(('**Simulation**', None))
    if any('CE Utility' in r for r in rows):
        metric_rows.append(('CE utility', lambda r: f"{r.get('CE Utility', 0):,.2f}"))
    if any('Mean Consumption' in r for r in rows):
        metric_rows.append(('Mean consumption ($)', lambda r: f"{r.get('Mean Consumption', 0):,.0f}"))
    if any('Mean Financial Assets' in r for r in rows):
        metric_rows.append(('Mean fin. assets ($)', lambda r: f"{r.get('Mean Financial Assets', 0):,.0f}"))
    if any('Mean Housing' in r for r in rows):
        metric_rows.append(('Mean housing ($)', lambda r: f"{r.get('Mean Housing', 0):,.0f}"))
    if any('Adj Rate' in r for r in rows):
        metric_rows.append(('Adjustment rate (%)', lambda r: f"{r.get('Adj Rate', 0):.1f}"))

    # Build markdown
    lines = []
    if caption:
        lines.append(f'### {caption}\n')

    # Header
    hdr = '| | ' + ' | '.join(f'**{m}**' for m in methods) + ' |'
    sep = '|:---|' + '|'.join(':---:' for _ in methods) + '|'
    lines.extend([hdr, sep])

    for label, fmt_fn in metric_rows:
        if fmt_fn is None:
            # Section header row
            cells = ' | '.join('' for _ in methods)
            lines.append(f'| {label} | {cells} |')
        else:
            cells = ' | '.join(fmt_fn(by_method[m]) for m in methods)
            lines.append(f'| {label} | {cells} |')

    return '\n'.join(lines)


def generate_cohort_table(sim_data, t0, T, norm, cohort_width=5):
    """Mean and SD by age cohort for consumption, financial assets, housing.

    Returns markdown string (one table per call).

    Parameters
    ----------
    sim_data : dict
        ``(T_total, N)`` panels with keys ``'c'``, ``'a'``, ``'h'``.
    t0, T : int
        Age range (row index = absolute age in panels).
    norm : float
        Normalisation factor (model units -> dollars).
    cohort_width : int
    """
    import numpy as np

    bins = list(range(t0, T, cohort_width))
    if bins[-1] < T:
        bins.append(T)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        label = f'{lo}\u2013{hi - 1}'
        row = {'Age': label}
        for key, name in [('c', 'Consumption'), ('a', 'Financial assets'), ('h', 'Housing')]:
            panel = sim_data[key][lo:hi, :] * norm
            vals = panel[np.isfinite(panel)]
            if len(vals) > 0:
                row[f'{name} mean'] = np.mean(vals)
                row[f'{name} SD'] = np.std(vals)
            else:
                row[f'{name} mean'] = float('nan')
                row[f'{name} SD'] = float('nan')
        rows.append(row)

    def _fmt(v):
        if math.isnan(v):
            return '\u2014'
        return f'{v:,.2f}'

    hdr = '| Age | Mean c | SD c | Mean a | SD a | Mean H | SD H |'
    sep = '| :--- | ---: | ---: | ---: | ---: | ---: | ---: |'
    lines = [hdr, sep]
    for r in rows:
        lines.append(
            f'| {r["Age"]} '
            f'| {_fmt(r["Consumption mean"])} | {_fmt(r["Consumption SD"])} '
            f'| {_fmt(r["Financial assets mean"])} | {_fmt(r["Financial assets SD"])} '
            f'| {_fmt(r["Housing mean"])} | {_fmt(r["Housing SD"])} |'
        )
    return '\n'.join(lines)


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
