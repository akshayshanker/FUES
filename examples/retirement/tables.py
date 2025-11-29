"""Table generation functions for retirement model results.

Author: Akshay Shanker, University of New South Wales, akshay.shanker@me.com
"""

import os


def _format_params_caption_md(params):
    """Format model parameters as a markdown caption string."""
    if not params:
        return ""

    lines = ["\n---", "**Model Parameters:**"]
    param_strs = _format_params_list(params)
    lines.append(", ".join(param_strs))
    return "\n".join(lines)


def _format_params_caption_latex(params):
    """Format model parameters as a LaTeX caption string."""
    if not params:
        return ""

    param_strs = _format_params_list(params, latex=True)
    return "Model parameters: " + ", ".join(param_strs)


def _format_params_list(params, latex=False):
    """Format parameters as list of strings."""
    if not params:
        return []

    param_order = ['r', 'beta', 'T', 'y', 'b', 'grid_max_A', 'm_bar', 'smooth_sigma']
    if latex:
        param_labels = {
            'r': 'r',
            'beta': r'$\beta$',
            'T': 'T',
            'y': 'y',
            'b': 'b',
            'grid_max_A': r'$A_{\max}$',
            'm_bar': r'$\bar{m}$',
            'smooth_sigma': r'$\sigma$',
        }
    else:
        param_labels = {
            'r': 'r (interest)',
            'beta': 'β (discount)',
            'T': 'T (periods)',
            'y': 'y (income)',
            'b': 'b (borrow limit)',
            'grid_max_A': 'A_max',
            'm_bar': 'm̄ (FUES threshold)',
            'smooth_sigma': 'σ (smoothing)',
        }

    param_strs = []
    for key in param_order:
        if key in params:
            val = params[key]
            label = param_labels.get(key, key)
            if isinstance(val, float) and abs(val) < 1e-6:
                param_strs.append(f"{label}={val:.1e}")
            elif isinstance(val, float):
                param_strs.append(f"{label}={val:.4g}")
            else:
                param_strs.append(f"{label}={val}")

    return param_strs


def generate_timing_table(data, table_type, caption, results_dir="results", params=None):
    """Generate markdown and LaTeX tables with total solution time performance.

    Parameters
    ----------
    data : list of lists
        Data for RFC, FUES, DCEGM, CONSAV total solution timings.
        Each row: [grid_size, delta, rfc_time, fues_time, dcegm_time, consav_time]
    table_type : str
        Type of the table for labeling the output file.
    caption : str
        Caption/title for the table.
    results_dir : str
        Directory to save results. Default is "results".
    params : dict, optional
        Model parameters to include in caption.
    """
    os.makedirs(results_dir, exist_ok=True)

    # --- Markdown output ---
    md_lines = []
    md_lines.append(f"# {caption} - Total Solution Time (ms)\n")
    md_lines.append("| Grid | Delta | RFC | FUES | DCEGM | CONSAV |")
    md_lines.append("|------|-------|-----|------|-------|--------|")

    for row in data:
        grid_size = int(row[0])
        delta = row[1]
        rfc_time = row[2]
        fues_time = row[3]
        dcegm_time = row[4]
        consav_time = row[5]
        md_lines.append(
            f"| {grid_size} | {delta:.2f} | {rfc_time:.3f} | {fues_time:.3f} | "
            f"{dcegm_time:.3f} | {consav_time:.3f} |"
        )

    md_lines.append(_format_params_caption_md(params))

    md_path = os.path.join(results_dir, f"retirement_{table_type}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown table saved to: {md_path}")

    # --- LaTeX output ---
    tex_lines = []
    tex_lines.append(r"\begin{table}[htbp]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{" + caption + " -- Total Solution Time (ms)}")
    tex_lines.append(r"\label{tab:" + table_type + "}")
    tex_lines.append(r"\begin{tabular}{cccccc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"Grid & $\delta$ & RFC & FUES & DCEGM & CONSAV \\")
    tex_lines.append(r"\midrule")

    for row in data:
        grid_size = int(row[0])
        delta = row[1]
        rfc_time = row[2]
        fues_time = row[3]
        dcegm_time = row[4]
        consav_time = row[5]
        tex_lines.append(
            f"{grid_size} & {delta:.2f} & {rfc_time:.3f} & {fues_time:.3f} & "
            f"{dcegm_time:.3f} & {consav_time:.3f} \\\\"
        )

    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    if params:
        tex_lines.append(r"\vspace{0.5em}")
        tex_lines.append(r"\footnotesize{" + _format_params_caption_latex(params) + "}")
    tex_lines.append(r"\end{table}")

    tex_path = os.path.join(results_dir, f"retirement_{table_type}.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines))
    print(f"LaTeX table saved to: {tex_path}")

    return md_path, tex_path


def generate_results_table(data, errors, table_type, caption, results_dir="results", params=None):
    """Generate markdown and LaTeX tables with timing and Euler errors.

    Parameters
    ----------
    data : list of lists
        Data for RFC, FUES, DCEGM, CONSAV timings.
        Each row: [grid_size, delta, rfc_time, fues_time, dcegm_time, consav_time]
    errors : list of lists
        Data for corresponding Euler errors.
        Each row: [grid_size, delta, rfc_err, fues_err, dcegm_err, consav_err]
    table_type : str
        Type of the table for labeling the output file.
    caption : str
        Caption/title for the table.
    results_dir : str
        Directory to save results. Default is "results".
    params : dict, optional
        Model parameters to include in caption.
    """
    os.makedirs(results_dir, exist_ok=True)

    # --- Markdown output ---
    md_lines = []
    md_lines.append(f"# {caption} - Timing & Euler Errors\n")
    md_lines.append("| Grid | Delta | RFC (ms) | FUES (ms) | DCEGM (ms) | CONSAV (ms) | RFC Err | FUES Err | DCEGM Err | CONSAV Err |")
    md_lines.append("|------|-------|----------|-----------|------------|-------------|---------|----------|-----------|------------|")

    for row, error_row in zip(data, errors):
        grid_size = int(row[0])
        delta = row[1]
        rfc_time = row[2]
        fues_time = row[3]
        dcegm_time = row[4]
        consav_time = row[5]

        rfc_error = error_row[2]
        fues_error = error_row[3]
        dcegm_error = error_row[4]
        consav_error = error_row[5]

        md_lines.append(
            f"| {grid_size} | {delta:.2f} | {rfc_time:.3f} | {fues_time:.3f} | "
            f"{dcegm_time:.3f} | {consav_time:.3f} | {rfc_error:.3f} | "
            f"{fues_error:.3f} | {dcegm_error:.3f} | {consav_error:.3f} |"
        )

    md_lines.append(_format_params_caption_md(params))

    md_path = os.path.join(results_dir, f"retirement_{table_type}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown table saved to: {md_path}")

    # --- LaTeX output ---
    tex_lines = []
    tex_lines.append(r"\begin{table}[htbp]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{" + caption + " -- Timing and Euler Errors}")
    tex_lines.append(r"\label{tab:" + table_type + "}")
    tex_lines.append(r"\begin{tabular}{cccccccccc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"Grid & $\delta$ & \multicolumn{4}{c}{Time (ms)} & \multicolumn{4}{c}{Euler Error} \\")
    tex_lines.append(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}")
    tex_lines.append(r" & & RFC & FUES & DCEGM & CONSAV & RFC & FUES & DCEGM & CONSAV \\")
    tex_lines.append(r"\midrule")

    for row, error_row in zip(data, errors):
        grid_size = int(row[0])
        delta = row[1]
        rfc_time = row[2]
        fues_time = row[3]
        dcegm_time = row[4]
        consav_time = row[5]

        rfc_error = error_row[2]
        fues_error = error_row[3]
        dcegm_error = error_row[4]
        consav_error = error_row[5]

        tex_lines.append(
            f"{grid_size} & {delta:.2f} & {rfc_time:.3f} & {fues_time:.3f} & "
            f"{dcegm_time:.3f} & {consav_time:.3f} & {rfc_error:.3f} & "
            f"{fues_error:.3f} & {dcegm_error:.3f} & {consav_error:.3f} \\\\"
        )

    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    if params:
        tex_lines.append(r"\vspace{0.5em}")
        tex_lines.append(r"\footnotesize{" + _format_params_caption_latex(params) + "}")
    tex_lines.append(r"\end{table}")

    tex_path = os.path.join(results_dir, f"retirement_{table_type}.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines))
    print(f"LaTeX table saved to: {tex_path}")

    return md_path, tex_path
