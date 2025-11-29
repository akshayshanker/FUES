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
                                   results_dir="results", params=None):
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
    md_lines.append("|      |       |    RFC    |    FUES   |   DCEGM   |  CONSAV   |")
    md_lines.append("| Grid | δ     | UE  | Tot | UE  | Tot | UE  | Tot | UE  | Tot |")
    md_lines.append("|------|-------|-----|-----|-----|-----|-----|-----|-----|-----|")

    for grid in sorted(grid_groups.keys()):
        rows = grid_groups[grid]
        for i, (ue_row, tot_row) in enumerate(rows):
            delta = ue_row[1]
            grid_str = str(grid) if i == 0 else ""
            md_lines.append(
                f"| {grid_str:4} | {delta:.2f} | "
                f"{ue_row[2]:.2f} | {tot_row[2]:.2f} | "
                f"{ue_row[3]:.2f} | {tot_row[3]:.2f} | "
                f"{ue_row[4]:.2f} | {tot_row[4]:.2f} | "
                f"{ue_row[5]:.2f} | {tot_row[5]:.2f} |"
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
    tex_lines.append(r"\caption{" + caption + " -- Timing (ms)}")
    tex_lines.append(r"\label{tab:" + table_type + "}")
    tex_lines.append(r"\begin{tabular}{cc|cc|cc|cc|cc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r" & & \multicolumn{2}{c|}{RFC} & \multicolumn{2}{c|}{FUES} & "
                     r"\multicolumn{2}{c|}{DCEGM} & \multicolumn{2}{c}{CONSAV} \\")
    tex_lines.append(r"Grid & $\delta$ & UE & Tot & UE & Tot & UE & Tot & UE & Tot \\")
    tex_lines.append(r"\midrule")

    for grid in sorted(grid_groups.keys()):
        rows = grid_groups[grid]
        for i, (ue_row, tot_row) in enumerate(rows):
            delta = ue_row[1]
            grid_str = str(grid) if i == 0 else ""
            tex_lines.append(
                f"{grid_str} & {delta:.2f} & "
                f"{ue_row[2]:.2f} & {tot_row[2]:.2f} & "
                f"{ue_row[3]:.2f} & {tot_row[3]:.2f} & "
                f"{ue_row[4]:.2f} & {tot_row[4]:.2f} & "
                f"{ue_row[5]:.2f} & {tot_row[5]:.2f} \\\\"
            )
        # Add slight spacing between grid groups
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
                            results_dir="results", params=None):
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
    md_lines.append("|      |       |     RFC     |    FUES     |    DCEGM    |   CONSAV    |")
    md_lines.append("| Grid | δ     | Euler | Dev | Euler | Dev | Euler | Dev | Euler | Dev |")
    md_lines.append("|------|-------|-------|-----|-------|-----|-------|-----|-------|-----|")

    for grid in sorted(grid_groups.keys()):
        rows = grid_groups[grid]
        for i, (euler_row, cdev_row) in enumerate(rows):
            delta = euler_row[1]
            grid_str = str(grid) if i == 0 else ""
            md_lines.append(
                f"| {grid_str:4} | {delta:.2f} | "
                f"{euler_row[2]:.2f} | {cdev_row[2]:.2f} | "
                f"{euler_row[3]:.2f} | {cdev_row[3]:.2f} | "
                f"{euler_row[4]:.2f} | {cdev_row[4]:.2f} | "
                f"{euler_row[5]:.2f} | {cdev_row[5]:.2f} |"
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
    tex_lines.append(r"\caption{" + caption + r" -- Accuracy ($\log_{10}$)}")
    tex_lines.append(r"\label{tab:" + table_type + "}")
    tex_lines.append(r"\begin{tabular}{cc|cc|cc|cc|cc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r" & & \multicolumn{2}{c|}{RFC} & \multicolumn{2}{c|}{FUES} & "
                     r"\multicolumn{2}{c|}{DCEGM} & \multicolumn{2}{c}{CONSAV} \\")
    tex_lines.append(r"Grid & $\delta$ & Euler & Dev & Euler & Dev & Euler & Dev & Euler & Dev \\")
    tex_lines.append(r"\midrule")

    for grid in sorted(grid_groups.keys()):
        rows = grid_groups[grid]
        for i, (euler_row, cdev_row) in enumerate(rows):
            delta = euler_row[1]
            grid_str = str(grid) if i == 0 else ""
            tex_lines.append(
                f"{grid_str} & {delta:.2f} & "
                f"{euler_row[2]:.2f} & {cdev_row[2]:.2f} & "
                f"{euler_row[3]:.2f} & {cdev_row[3]:.2f} & "
                f"{euler_row[4]:.2f} & {cdev_row[4]:.2f} & "
                f"{euler_row[5]:.2f} & {cdev_row[5]:.2f} \\\\"
            )
        # Add slight spacing between grid groups
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
                           params=None, l2_data=None):
    """Generate markdown and LaTeX tables with timing, Euler errors, and L2 deviation.

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
    l2_data : list of lists, optional
        Data for consumption deviation from true solution (20k grid DCEGM).
        Each row: [grid_size, delta, rfc_cdev, fues_cdev, dcegm_cdev, consav_cdev]
    """
    os.makedirs(results_dir, exist_ok=True)
    include_l2 = l2_data is not None and len(l2_data) > 0

    # --- Markdown output ---
    md_lines = []
    if include_l2:
        md_lines.append(f"# {caption} - Timing, Euler Errors & Cons. Deviation\n")
        md_lines.append("| Grid | Delta | RFC (ms) | FUES (ms) | DCEGM (ms) | CONSAV (ms) | "
                        "RFC Err | FUES Err | DCEGM Err | CONSAV Err | "
                        "RFC CDev | FUES CDev | DCEGM CDev | CONSAV CDev |")
        md_lines.append("|------|-------|----------|-----------|------------|-------------|"
                        "---------|----------|-----------|------------|"
                        "----------|-----------|------------|-------------|")
    else:
        md_lines.append(f"# {caption} - Timing & Euler Errors\n")
        md_lines.append("| Grid | Delta | RFC (ms) | FUES (ms) | DCEGM (ms) | CONSAV (ms) | "
                        "RFC Err | FUES Err | DCEGM Err | CONSAV Err |")
        md_lines.append("|------|-------|----------|-----------|------------|-------------|"
                        "---------|----------|-----------|------------|")

    for i, (row, error_row) in enumerate(zip(data, errors)):
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

        if include_l2:
            l2_row = l2_data[i]
            rfc_l2 = l2_row[2]
            fues_l2 = l2_row[3]
            dcegm_l2 = l2_row[4]
            consav_l2 = l2_row[5]
            md_lines.append(
                f"| {grid_size} | {delta:.2f} | {rfc_time:.3f} | {fues_time:.3f} | "
                f"{dcegm_time:.3f} | {consav_time:.3f} | {rfc_error:.3f} | "
                f"{fues_error:.3f} | {dcegm_error:.3f} | {consav_error:.3f} | "
                f"{rfc_l2:.3f} | {fues_l2:.3f} | {dcegm_l2:.3f} | {consav_l2:.3f} |"
            )
        else:
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
    if include_l2:
        tex_lines.append(r"\caption{" + caption + " -- Timing, Euler Errors, and Cons. Deviation}")
        tex_lines.append(r"\label{tab:" + table_type + "}")
        tex_lines.append(r"\begin{tabular}{cccccccccccccc}")
        tex_lines.append(r"\toprule")
        tex_lines.append(r"Grid & $\delta$ & \multicolumn{4}{c}{Time (ms)} & "
                         r"\multicolumn{4}{c}{Euler Error} & \multicolumn{4}{c}{Cons. Dev.} \\")
        tex_lines.append(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14}")
        tex_lines.append(r" & & RFC & FUES & DCEGM & CONSAV & RFC & FUES & DCEGM & CONSAV & "
                         r"RFC & FUES & DCEGM & CONSAV \\")
    else:
        tex_lines.append(r"\caption{" + caption + " -- Timing and Euler Errors}")
        tex_lines.append(r"\label{tab:" + table_type + "}")
        tex_lines.append(r"\begin{tabular}{cccccccccc}")
        tex_lines.append(r"\toprule")
        tex_lines.append(r"Grid & $\delta$ & \multicolumn{4}{c}{Time (ms)} & "
                         r"\multicolumn{4}{c}{Euler Error} \\")
        tex_lines.append(r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}")
        tex_lines.append(r" & & RFC & FUES & DCEGM & CONSAV & RFC & FUES & DCEGM & CONSAV \\")
    tex_lines.append(r"\midrule")

    for i, (row, error_row) in enumerate(zip(data, errors)):
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

        if include_l2:
            l2_row = l2_data[i]
            rfc_l2 = l2_row[2]
            fues_l2 = l2_row[3]
            dcegm_l2 = l2_row[4]
            consav_l2 = l2_row[5]
            tex_lines.append(
                f"{grid_size} & {delta:.2f} & {rfc_time:.3f} & {fues_time:.3f} & "
                f"{dcegm_time:.3f} & {consav_time:.3f} & {rfc_error:.3f} & "
                f"{fues_error:.3f} & {dcegm_error:.3f} & {consav_error:.3f} & "
                f"{rfc_l2:.3f} & {fues_l2:.3f} & {dcegm_l2:.3f} & {consav_l2:.3f} \\\\"
            )
        else:
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
