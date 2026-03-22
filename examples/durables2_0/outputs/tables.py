"""Table generation for durables2_0 benchmark results.

Produces markdown and LaTeX tables for timing and accuracy sweeps.
Adapted from examples/retirement/outputs/tables.py.
"""

import os

_COL_ORDER = (2, 3)
_COL_NAMES = ("FUES", "NEGM")


def _format_params_list(params, latex=False):
    if not params:
        return []

    param_order = [
        'r', 'beta', 'T', 'alpha', 'gamma_c', 'gamma_h',
        'delta', 'kappa', 'tau', 'K', 'grid_max_A', 'm_bar',
    ]
    if latex:
        param_labels = {
            'r': 'r', 'beta': r'$\beta$', 'T': 'T',
            'alpha': r'$\alpha$', 'gamma_c': r'$\gamma_c$',
            'gamma_h': r'$\gamma_h$', 'delta': r'$\delta$',
            'kappa': r'$\kappa$', 'tau': r'$\tau$', 'K': 'K',
            'grid_max_A': r'$A_{\max}$', 'm_bar': r'$\bar{m}$',
        }
    else:
        param_labels = {
            'r': 'r', 'beta': 'β', 'T': 'T',
            'alpha': 'α', 'gamma_c': 'γ_c', 'gamma_h': 'γ_h',
            'delta': 'δ', 'kappa': 'κ', 'tau': 'τ', 'K': 'K',
            'grid_max_A': 'A_max', 'm_bar': 'm̄',
        }

    strs = []
    for key in param_order:
        if key in params:
            val = params[key]
            label = param_labels.get(key, key)
            if isinstance(val, float) and abs(val) < 1e-6:
                strs.append(f"{label}={val:.1e}")
            elif isinstance(val, float):
                strs.append(f"{label}={val:.4g}")
            else:
                strs.append(f"{label}={val}")
    return strs


def _format_params_caption_md(params):
    if not params:
        return ""
    return "\n---\n**Model Parameters:** " + ", ".join(
        _format_params_list(params))


def _format_params_caption_latex(params):
    if not params:
        return ""
    return "Model parameters: " + ", ".join(
        _format_params_list(params, latex=True))


def generate_timing_table(timing_data, caption, results_dir="results",
                          params=None, col_names=None):
    """Generate timing table (markdown + LaTeX).

    Parameters
    ----------
    timing_data : list of lists
        Each row: ``[grid_size, tau, method1_ms, method2_ms, ...]``
    caption : str
    results_dir : str
    params : dict, optional
    col_names : tuple of str, optional
        Column names (default: ``_COL_NAMES``).
    """
    os.makedirs(results_dir, exist_ok=True)
    names = col_names or _COL_NAMES
    n_methods = len(names)

    # Group by grid size
    groups = {}
    for row in timing_data:
        grid = int(row[0])
        groups.setdefault(grid, []).append(row)

    # Markdown
    md = [f"# {caption} - Timing (ms)\n"]
    hdr = "| Grid | τ     | " + " | ".join(
        f"{n:>8}" for n in names) + " |"
    sep = "|------|-------|" + "|".join(
        "-" * 10 for _ in names) + "|"
    md.extend([hdr, sep])

    for grid in sorted(groups):
        for i, row in enumerate(groups[grid]):
            g_str = str(grid) if i == 0 else ""
            vals = " | ".join(
                f"{row[2 + j]:>8.2f}" for j in range(n_methods))
            md.append(f"| {g_str:4} | {row[1]:.3f} | {vals} |")

    md.append(_format_params_caption_md(params))

    md_path = os.path.join(results_dir, "durables_timing.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))

    # LaTeX
    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{" + caption + " -- Timing (ms)}")
    tex.append(r"\label{tab:durables_timing}")
    col_spec = "cc" + "c" * n_methods
    tex.append(r"\begin{tabular}{" + col_spec + "}")
    tex.append(r"\toprule")
    tex.append(
        r"Grid & $\tau$ & " + " & ".join(names) + r" \\")
    tex.append(r"\midrule")

    for grid in sorted(groups):
        for i, row in enumerate(groups[grid]):
            g_str = str(grid) if i == 0 else ""
            vals = " & ".join(
                f"{row[2 + j]:.2f}" for j in range(n_methods))
            tex.append(f"{g_str} & {row[1]:.3f} & {vals} \\\\")
        if grid != max(groups):
            tex.append(r"\addlinespace[0.5em]")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    if params:
        tex.append(r"\vspace{0.5em}")
        tex.append(
            r"\footnotesize{" + _format_params_caption_latex(params) + "}")
    tex.append(r"\end{table}")

    tex_path = os.path.join(results_dir, "durables_timing.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex))

    print(f"Timing tables saved to: {md_path}, {tex_path}")
    return md_path, tex_path


def generate_accuracy_table(euler_data, cdev_data, caption,
                            results_dir="results", params=None,
                            col_names=None):
    """Generate accuracy table with Euler and consumption deviation (markdown + LaTeX).

    Parameters
    ----------
    euler_data : list of lists
        Each row: ``[grid_size, tau, method1_euler, method2_euler, ...]``
    cdev_data : list of lists
        Each row: ``[grid_size, tau, method1_cdev, method2_cdev, ...]``
    caption : str
    results_dir : str
    params : dict, optional
    col_names : tuple of str, optional
    """
    os.makedirs(results_dir, exist_ok=True)
    names = col_names or _COL_NAMES
    n_methods = len(names)

    groups = {}
    for euler_row, cdev_row in zip(euler_data, cdev_data):
        grid = int(euler_row[0])
        groups.setdefault(grid, []).append((euler_row, cdev_row))

    # Markdown
    md = [f"# {caption} - Accuracy (log₁₀)\n"]
    sub_hdr = "|      |       | " + " | ".join(
        f"  {n}   " for n in names) + " |"
    col_hdr = "| Grid | τ     | " + " | ".join(
        "Euler | Dev" for _ in names) + " |"
    sep = "|------|-------|" + "|".join(
        "-------|-----" for _ in names) + "|"
    md.extend([sub_hdr, col_hdr, sep])

    for grid in sorted(groups):
        for i, (e_row, c_row) in enumerate(groups[grid]):
            g_str = str(grid) if i == 0 else ""
            vals = " | ".join(
                f"{e_row[2 + j]:.2f} | {c_row[2 + j]:.2f}"
                for j in range(n_methods))
            md.append(f"| {g_str:4} | {e_row[1]:.3f} | {vals} |")

    md.append(_format_params_caption_md(params))

    md_path = os.path.join(results_dir, "durables_accuracy.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))

    # LaTeX
    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{" + caption + r" -- Accuracy ($\log_{10}$)}")
    tex.append(r"\label{tab:durables_accuracy}")
    col_spec = "cc" + "|cc" * n_methods
    tex.append(r"\begin{tabular}{" + col_spec + "}")
    tex.append(r"\toprule")
    mc = " & ".join(
        r"\multicolumn{2}{c" + ("|" if j < n_methods - 1 else "") + "}{" + n + "}"
        for j, n in enumerate(names))
    tex.append(r" & & " + mc + r" \\")
    tex.append(
        r"Grid & $\tau$ & " + " & ".join(
            "Euler & Dev" for _ in names) + r" \\")
    tex.append(r"\midrule")

    for grid in sorted(groups):
        for i, (e_row, c_row) in enumerate(groups[grid]):
            g_str = str(grid) if i == 0 else ""
            vals = " & ".join(
                f"{e_row[2 + j]:.2f} & {c_row[2 + j]:.2f}"
                for j in range(n_methods))
            tex.append(f"{g_str} & {e_row[1]:.3f} & {vals} \\\\")
        if grid != max(groups):
            tex.append(r"\addlinespace[0.5em]")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    if params:
        tex.append(r"\vspace{0.5em}")
        tex.append(
            r"\footnotesize{" + _format_params_caption_latex(params) + "}")
    tex.append(r"\end{table}")

    tex_path = os.path.join(results_dir, "durables_accuracy.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex))

    print(f"Accuracy tables saved to: {md_path}, {tex_path}")
    return md_path, tex_path
