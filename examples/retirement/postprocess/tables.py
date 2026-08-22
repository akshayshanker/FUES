"""Table generation functions for retirement model results.

Author: Akshay Shanker, University of New South Wales, akshay.shanker@me.com
"""

import os

# Column order for display: FUES, MSS (was DCEGM), LTM (was CONSAV), RFC
# Data rows arrive as [grid, delta, RFC, FUES, DCEGM, CONSAV] = indices 2,3,4,5
# Display order: FUES=3, MSS=4, LTM=5, RFC=2
_COL_ORDER = (3, 4, 5, 2)
_COL_NAMES = ("FUES", "MSS", "LTM", "RFC")


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

    param_order = ['r', 'beta', 'T', 'y', 'b', 'grid_max_A', 'm_bar', 'smooth_sigma',
                   'true_grid_size', 'true_method']
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
            'true_grid_size': r'$N_{\text{true}}$',
            'true_method': 'True method',
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
            'true_grid_size': 'N_true (ref grid)',
            'true_method': 'True method',
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


def generate_timing_table_combined(ue_data, total_data, table_type, caption,
                                   results_dir="results", params=None,
                                   latex_grids=None):
    """Generate timing table with UE and Total sub-columns per method.

    Parameters
    ----------
    ue_data : list of lists
        Upper envelope timing data.
        Each row: [grid_size, delta, rfc_ue, fues_ue, dcegm_ue, consav_ue]
    total_data : list of lists
        Total solution timing data.
        Each row: [grid_size, delta, rfc_tot, fues_tot, dcegm_tot, consav_tot]
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
    c = _COL_ORDER
    n = _COL_NAMES

    # Group data by grid size
    grid_groups = {}
    for ue_row, tot_row in zip(ue_data, total_data):
        grid = int(ue_row[0])
        if grid not in grid_groups:
            grid_groups[grid] = []
        grid_groups[grid].append((ue_row, tot_row))

    # --- Markdown output ---
    md_lines = []
    md_lines.append(f"# {caption} - Timing (ms)\n")
    md_lines.append(f"|      |       |    {n[0]}   |    {n[1]}    |    {n[2]}    |    {n[3]}    |")
    md_lines.append("| Grid | δ     | UE  | Tot | UE  | Tot | UE  | Tot | UE  | Tot |")
    md_lines.append("|------|-------|-----|-----|-----|-----|-----|-----|-----|-----|")

    for grid in sorted(grid_groups.keys()):
        rows = grid_groups[grid]
        for i, (ue_row, tot_row) in enumerate(rows):
            delta = ue_row[1]
            grid_str = str(grid) if i == 0 else ""
            md_lines.append(
                f"| {grid_str:4} | {delta:.2f} | "
                f"{ue_row[c[0]]:.2f} | {tot_row[c[0]]:.2f} | "
                f"{ue_row[c[1]]:.2f} | {tot_row[c[1]]:.2f} | "
                f"{ue_row[c[2]]:.2f} | {tot_row[c[2]]:.2f} | "
                f"{ue_row[c[3]]:.2f} | {tot_row[c[3]]:.2f} |"
            )

    md_lines.append(_format_params_caption_md(params))

    md_path = os.path.join(results_dir, f"retirement_{table_type}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown table saved to: {md_path}")

    # --- LaTeX output (filtered by latex_grids if provided) ---
    latex_grid_set = set(latex_grids) if latex_grids else None
    tex_lines = []
    tex_lines.append(r"\begin{table}[htbp]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{" + caption + " -- Timing (ms)}")
    tex_lines.append(r"\label{tab:" + table_type + "}")
    tex_lines.append(r"\begin{tabular}{cc|cc|cc|cc|cc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(
        r" & & \multicolumn{2}{c|}{" + n[0] + r"} & \multicolumn{2}{c|}{" + n[1] +
        r"} & \multicolumn{2}{c|}{" + n[2] + r"} & \multicolumn{2}{c}{" + n[3] + r"} \\")
    tex_lines.append(r"Grid & $\delta$ & UE & Tot & UE & Tot & UE & Tot & UE & Tot \\")
    tex_lines.append(r"\midrule")

    for grid in sorted(grid_groups.keys()):
        if latex_grid_set and grid not in latex_grid_set:
            continue
        rows = grid_groups[grid]
        for i, (ue_row, tot_row) in enumerate(rows):
            delta = ue_row[1]
            grid_str = str(grid) if i == 0 else ""
            tex_lines.append(
                f"{grid_str} & {delta:.2f} & "
                f"{ue_row[c[0]]:.2f} & {tot_row[c[0]]:.2f} & "
                f"{ue_row[c[1]]:.2f} & {tot_row[c[1]]:.2f} & "
                f"{ue_row[c[2]]:.2f} & {tot_row[c[2]]:.2f} & "
                f"{ue_row[c[3]]:.2f} & {tot_row[c[3]]:.2f} \\\\"
            )
        if grid != max(grid_groups.keys()):
            tex_lines.append(r"\addlinespace[0.5em]")

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


def generate_accuracy_table(euler_data, cdev_data, table_type, caption,
                            results_dir="results", params=None,
                            latex_grids=None):
    """Generate accuracy table with Euler and Cons.Dev sub-columns per method.

    Parameters
    ----------
    euler_data : list of lists
        Euler equation error data.
        Each row: [grid_size, delta, rfc_err, fues_err, dcegm_err, consav_err]
    cdev_data : list of lists
        Consumption deviation data.
        Each row: [grid_size, delta, rfc_cdev, fues_cdev, dcegm_cdev, consav_cdev]
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
    c = _COL_ORDER
    n = _COL_NAMES

    # Group data by grid size
    grid_groups = {}
    for euler_row, cdev_row in zip(euler_data, cdev_data):
        grid = int(euler_row[0])
        if grid not in grid_groups:
            grid_groups[grid] = []
        grid_groups[grid].append((euler_row, cdev_row))

    # --- Markdown output ---
    md_lines = []
    md_lines.append(f"# {caption} - Accuracy (log₁₀)\n")
    md_lines.append(f"|      |       |    {n[0]}    |    {n[1]}     |    {n[2]}     |    {n[3]}     |")
    md_lines.append("| Grid | δ     | Euler | Dev | Euler | Dev | Euler | Dev | Euler | Dev |")
    md_lines.append("|------|-------|-------|-----|-------|-----|-------|-----|-------|-----|")

    for grid in sorted(grid_groups.keys()):
        rows = grid_groups[grid]
        for i, (euler_row, cdev_row) in enumerate(rows):
            delta = euler_row[1]
            grid_str = str(grid) if i == 0 else ""
            md_lines.append(
                f"| {grid_str:4} | {delta:.2f} | "
                f"{euler_row[c[0]]:.2f} | {cdev_row[c[0]]:.2f} | "
                f"{euler_row[c[1]]:.2f} | {cdev_row[c[1]]:.2f} | "
                f"{euler_row[c[2]]:.2f} | {cdev_row[c[2]]:.2f} | "
                f"{euler_row[c[3]]:.2f} | {cdev_row[c[3]]:.2f} |"
            )

    md_lines.append(_format_params_caption_md(params))

    md_path = os.path.join(results_dir, f"retirement_{table_type}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown table saved to: {md_path}")

    # --- LaTeX output (filtered by latex_grids if provided) ---
    latex_grid_set = set(latex_grids) if latex_grids else None
    tex_lines = []
    tex_lines.append(r"\begin{table}[htbp]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{" + caption + r" -- Accuracy ($\log_{10}$)}")
    tex_lines.append(r"\label{tab:" + table_type + "}")
    tex_lines.append(r"\begin{tabular}{cc|cc|cc|cc|cc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(
        r" & & \multicolumn{2}{c|}{" + n[0] + r"} & \multicolumn{2}{c|}{" + n[1] +
        r"} & \multicolumn{2}{c|}{" + n[2] + r"} & \multicolumn{2}{c}{" + n[3] + r"} \\")
    tex_lines.append(r"Grid & $\delta$ & Euler & Dev & Euler & Dev & Euler & Dev & Euler & Dev \\")
    tex_lines.append(r"\midrule")

    for grid in sorted(grid_groups.keys()):
        if latex_grid_set and grid not in latex_grid_set:
            continue
        rows = grid_groups[grid]
        for i, (euler_row, cdev_row) in enumerate(rows):
            delta = euler_row[1]
            grid_str = str(grid) if i == 0 else ""
            tex_lines.append(
                f"{grid_str} & {delta:.2f} & "
                f"{euler_row[c[0]]:.2f} & {cdev_row[c[0]]:.2f} & "
                f"{euler_row[c[1]]:.2f} & {cdev_row[c[1]]:.2f} & "
                f"{euler_row[c[2]]:.2f} & {cdev_row[c[2]]:.2f} & "
                f"{euler_row[c[3]]:.2f} & {cdev_row[c[3]]:.2f} \\\\"
            )
        if grid != max(grid_groups.keys()):
            tex_lines.append(r"\addlinespace[0.5em]")

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


def generate_results_table(data, errors, table_type, caption, results_dir="results",
                           params=None, cdev_data=None):
    """Generate markdown and LaTeX tables with timing, Euler errors, and consumption deviation.

    Parameters
    ----------
    data : list of lists
        Each row: [grid_size, delta, rfc_time, fues_time, dcegm_time, consav_time]
    errors : list of lists
        Each row: [grid_size, delta, rfc_err, fues_err, dcegm_err, consav_err]
    table_type : str
        Type of the table for labeling the output file.
    caption : str
        Caption/title for the table.
    results_dir : str
        Directory to save results. Default is "results".
    params : dict, optional
        Model parameters to include in caption.
    cdev_data : list of lists, optional
        Each row: [grid_size, delta, rfc_cdev, fues_cdev, dcegm_cdev, consav_cdev]
    """
    os.makedirs(results_dir, exist_ok=True)
    c = _COL_ORDER
    n = _COL_NAMES
    include_cdev = cdev_data is not None and len(cdev_data) > 0

    # --- Markdown output ---
    md_lines = []
    if include_cdev:
        md_lines.append(f"# {caption} - Timing, Euler Errors & Cons. Deviation\n")
        md_lines.append(
            f"| Grid | Delta | {n[0]} (ms) | {n[1]} (ms) | {n[2]} (ms) | {n[3]} (ms) | "
            f"{n[0]} Err | {n[1]} Err | {n[2]} Err | {n[3]} Err | "
            f"{n[0]} CDev | {n[1]} CDev | {n[2]} CDev | {n[3]} CDev |")
        md_lines.append("|------|-------|----------|-----------|------------|-------------|"
                        "---------|----------|-----------|------------|"
                        "----------|-----------|------------|-------------|")
    else:
        md_lines.append(f"# {caption} - Timing & Euler Errors\n")
        md_lines.append(
            f"| Grid | Delta | {n[0]} (ms) | {n[1]} (ms) | {n[2]} (ms) | {n[3]} (ms) | "
            f"{n[0]} Err | {n[1]} Err | {n[2]} Err | {n[3]} Err |")
        md_lines.append("|------|-------|----------|-----------|------------|-------------|"
                        "---------|----------|-----------|------------|")

    for i, (row, error_row) in enumerate(zip(data, errors)):
        grid_size = int(row[0])
        delta = row[1]

        if include_cdev:
            cdev_row = cdev_data[i]
            md_lines.append(
                f"| {grid_size} | {delta:.2f} | "
                f"{row[c[0]]:.3f} | {row[c[1]]:.3f} | {row[c[2]]:.3f} | {row[c[3]]:.3f} | "
                f"{error_row[c[0]]:.3f} | {error_row[c[1]]:.3f} | {error_row[c[2]]:.3f} | {error_row[c[3]]:.3f} | "
                f"{cdev_row[c[0]]:.3f} | {cdev_row[c[1]]:.3f} | {cdev_row[c[2]]:.3f} | {cdev_row[c[3]]:.3f} |"
            )
        else:
            md_lines.append(
                f"| {grid_size} | {delta:.2f} | "
                f"{row[c[0]]:.3f} | {row[c[1]]:.3f} | {row[c[2]]:.3f} | {row[c[3]]:.3f} | "
                f"{error_row[c[0]]:.3f} | {error_row[c[1]]:.3f} | {error_row[c[2]]:.3f} | {error_row[c[3]]:.3f} |"
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
    if include_cdev:
        tex_lines.append(r"\caption{" + caption + " -- Timing, Euler Errors, and Cons. Deviation}")
        tex_lines.append(r"\label{tab:" + table_type + "}")
        tex_lines.append(r"\begin{tabular}{cccccccccccccc}")
        tex_lines.append(r"\toprule")
        tex_lines.append(
            r"Grid & $\delta$ & \multicolumn{4}{c}{Time (ms)} & "
            r"\multicolumn{4}{c}{Euler Error} & \multicolumn{4}{c}{Cons. Dev.} \\")
        tex_lines.append(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14}")
        tex_lines.append(
            f" & & {n[0]} & {n[1]} & {n[2]} & {n[3]} & "
            f"{n[0]} & {n[1]} & {n[2]} & {n[3]} & "
            f"{n[0]} & {n[1]} & {n[2]} & {n[3]} \\\\")
    else:
        tex_lines.append(r"\caption{" + caption + " -- Timing and Euler Errors}")
        tex_lines.append(r"\label{tab:" + table_type + "}")
        tex_lines.append(r"\begin{tabular}{cccccccccc}")
        tex_lines.append(r"\toprule")
        tex_lines.append(
            r"Grid & $\delta$ & \multicolumn{4}{c}{Time (ms)} & "
            r"\multicolumn{4}{c}{Euler Error} \\")
        tex_lines.append(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}")
        tex_lines.append(
            f" & & {n[0]} & {n[1]} & {n[2]} & {n[3]} & "
            f"{n[0]} & {n[1]} & {n[2]} & {n[3]} \\\\")
    tex_lines.append(r"\midrule")

    for i, (row, error_row) in enumerate(zip(data, errors)):
        grid_size = int(row[0])
        delta = row[1]

        if include_cdev:
            cdev_row = cdev_data[i]
            tex_lines.append(
                f"{grid_size} & {delta:.2f} & "
                f"{row[c[0]]:.3f} & {row[c[1]]:.3f} & {row[c[2]]:.3f} & {row[c[3]]:.3f} & "
                f"{error_row[c[0]]:.3f} & {error_row[c[1]]:.3f} & {error_row[c[2]]:.3f} & {error_row[c[3]]:.3f} & "
                f"{cdev_row[c[0]]:.3f} & {cdev_row[c[1]]:.3f} & {cdev_row[c[2]]:.3f} & {cdev_row[c[3]]:.3f} \\\\"
            )
        else:
            tex_lines.append(
                f"{grid_size} & {delta:.2f} & "
                f"{row[c[0]]:.3f} & {row[c[1]]:.3f} & {row[c[2]]:.3f} & {row[c[3]]:.3f} & "
                f"{error_row[c[0]]:.3f} & {error_row[c[1]]:.3f} & {error_row[c[2]]:.3f} & {error_row[c[3]]:.3f} \\\\"
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
