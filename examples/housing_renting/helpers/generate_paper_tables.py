#!/usr/bin/env python3
"""
Generate LaTeX and Markdown tables for the housing-renting model paper.

This script reads sweep results and outputs formatted tables matching
the retirement model table format.

Usage (on Gadi HPC):
    # After sweep completes, from the FUES repo root:
    python examples/housing_renting/helpers/generate_paper_tables.py --trial paper-v1
    
    # Or specify results path directly:
    python examples/housing_renting/helpers/generate_paper_tables.py \
        --results /scratch/tp66/as3442/FUES/solutions/housing_renting/paper-v1/sweep_results.csv

    # Specify custom output directory:
    python examples/housing_renting/helpers/generate_paper_tables.py \
        --trial paper-v1 --output /path/to/output
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path


# Default paths - can be overridden via environment or arguments
RESULTS_ROOT = Path(os.environ.get(
    "FUES_RESULTS_ROOT",
    Path(__file__).parent.parent.parent.parent / "results" / "housing_renting"
))
SOLUTIONS_ROOT = Path(os.environ.get(
    "FUES_SOLUTIONS_ROOT",
    "/scratch/tp66/as3442/FUES/solutions/housing_renting"
))


def load_sweep_results(results_path: Path) -> pd.DataFrame:
    """Load and validate sweep results CSV."""
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    df = pd.read_csv(results_path)
    
    # Rename columns for easier access
    column_map = {
        "master.methods.upper_envelope": "method",
        "master.settings.a_points": "grid",
        "master.settings.H_points": "H",
    }
    df = df.rename(columns=column_map)
    
    return df


def generate_accuracy_latex(df: pd.DataFrame, output_path: Path, model_params: str) -> None:
    """Generate LaTeX accuracy table with professional formatting."""
    
    # Get unique values - internal names for data lookup
    methods = ["FUES", "DCEGM", "CONSAV", "VFI"]
    # Display names for table headers
    method_display = {"FUES": "FUES", "DCEGM": "MSS", "CONSAV": "LTM", "VFI": "VFI"}
    
    grids = sorted(df["grid"].unique())
    H_sizes = sorted(df["H"].unique())
    
    # Check if dev_c_L2 is available
    has_dev = "dev_c_L2" in df.columns and df["dev_c_L2"].notna().any()
    
    # Build column spec based on available metrics
    # Use right-aligned columns for numeric data
    if has_dev:
        col_spec = r"\begin{tabular}{rr|rr|rr|rr|rr}"
        header1 = r" & & \multicolumn{2}{c|}{FUES} & \multicolumn{2}{c|}{MSS} & \multicolumn{2}{c|}{LTM} & \multicolumn{2}{c}{VFI} \\"
        cmidrule = r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}"
        header2 = r"Asset grid & $H$ & Euler & $L_2$ & Euler & $L_2$ & Euler & $L_2$ & Euler & $L_2$ \\"
    else:
        col_spec = r"\begin{tabular}{rr|r|r|r|r}"
        header1 = r" & & FUES & MSS & LTM & VFI \\"
        cmidrule = ""
        header2 = r"Asset grid & $H$ & Euler & Euler & Euler & Euler \\"
    
    # Start building LaTeX with professional formatting
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Housing-Renting Model: Solution Accuracy}",
        r"\label{tab:housing_accuracy}",
        col_spec,
        r"\toprule",
        header1,
    ]
    if cmidrule:
        lines.append(cmidrule)
    lines.extend([
        header2,
        r"\midrule",
    ])
    
    for i, grid in enumerate(grids):
        for j, H in enumerate(H_sizes):
            # Get row prefix - format grid with commas for thousands
            if j == 0:
                row_prefix = _format_number_with_commas(int(grid))
            else:
                row_prefix = ""
            
            # Collect values for each method
            vals = []
            for method in methods:
                mask = (df["method"] == method) & (df["grid"] == grid) & (df["H"] == H)
                row_data = df[mask]
                
                # Euler error
                if len(row_data) > 0 and pd.notna(row_data["euler_error"].values[0]):
                    euler = row_data["euler_error"].values[0]
                    vals.append(f"{euler:.2f}")
                else:
                    vals.append("--")
                
                # Dev error (if available) - L2 norm, show in scientific notation
                if has_dev:
                    if len(row_data) > 0 and pd.notna(row_data["dev_c_L2"].values[0]):
                        dev = row_data["dev_c_L2"].values[0]
                        vals.append(f"{dev:.2e}")
                    else:
                        vals.append("--")
            
            # Format row
            row = f"{row_prefix} & {H} & " + " & ".join(vals) + r" \\"
            lines.append(row)
        
        # Add spacing between grid groups (except for last)
        if i < len(grids) - 1:
            lines.append(r"\addlinespace[0.3em]")
    
    # Close table with footnote
    euler_note = r"\textbf{Euler}: Euler equation error in $\log_{10}$ units (more negative = more accurate). "
    dev_note = r"\textbf{$L_2$}: normalized consumption deviation from VFI baseline. " if has_dev else ""
    method_note = r"MSS = \citet{Iskhakov2017}; LTM = \citet{Druedahl2017}. "
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.5em}",
        r"\begin{minipage}{0.95\textwidth}",
        r"\footnotesize",
        r"\textit{Notes:} " + euler_note + dev_note + method_note + model_params,
        r"\end{minipage}",
        r"\end{table}",
    ])
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote: {output_path}")


def _format_number_with_commas(n: int) -> str:
    """Format integer with thousand separators for LaTeX."""
    return f"{n:,}".replace(",", "{,}")


def _format_float_with_commas(val: float, decimals: int = 2, latex: bool = True) -> str:
    """Format float with thousand separators.
    
    For values >= 1000, adds thousand separators.
    Always shows specified decimal places.
    
    Parameters
    ----------
    val : float
        The value to format
    decimals : int
        Number of decimal places (default: 2)
    latex : bool
        If True, use {,} for LaTeX. If False, use regular commas for Markdown.
    """
    if val >= 1000:
        # Split into integer and decimal parts
        int_part = int(val)
        dec_part = val - int_part
        # Format integer part with commas, then add decimal part
        int_str = f"{int_part:,}"
        if latex:
            int_str = int_str.replace(",", "{,}")
        dec_str = f"{dec_part:.{decimals}f}"[1:]  # Remove leading "0"
        return int_str + dec_str
    else:
        return f"{val:.{decimals}f}"


def generate_timing_latex(df: pd.DataFrame, output_path: Path, model_params: str) -> None:
    """Generate LaTeX timing table with professional formatting.
    
    U.env time is shown in milliseconds (×1000) for better readability.
    Period time remains in seconds.
    Cons. = average consumption stage time per period (shows VFI scaling with grid).
    Euler error is included as a compact accuracy column (replaces U.env%).
    
    Output is wrapped in landscape environment (requires pdflscape package).
    """
    
    # Get unique values - internal names for data lookup
    methods = ["FUES", "DCEGM", "CONSAV", "VFI"]
    grids = sorted(df["grid"].unique())
    H_sizes = sorted(df["H"].unique())
    
    # Professional LaTeX table with:
    # - Landscape orientation for wide table
    # - W column for asset grid size (compact rows)
    # - Right-aligned numeric columns (r) for better number alignment
    # - Caption width matched to table width
    lines = [
        r"% Requires: \usepackage{pdflscape, booktabs, caption}",
        r"\begin{landscape}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\captionsetup{width=0.75\linewidth}",
        r"\caption{Housing-Renting Model: Per-Period Timing and Accuracy}",
        r"\label{tab:housing_timing}",
        r"\begin{tabular}{@{}rr rrrr rrrr rrrr rrr@{}}",
        r"\toprule",
        r"\multicolumn{2}{c}{Grid} & \multicolumn{4}{c}{FUES} & \multicolumn{4}{c}{MSS} & \multicolumn{4}{c}{LTM} & \multicolumn{3}{c}{VFI} \\",
        r"\cmidrule(lr){1-2} \cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-14} \cmidrule(lr){15-17}",
        r"$W$ & $H$ & U.env & Cons. & Per. & E.Err & U.env & Cons. & Per. & E.Err & U.env & Cons. & Per. & E.Err & Cons. & Per. & E.Err \\",
        r"\midrule",
    ]
    
    for i, grid in enumerate(grids):
        for j, H in enumerate(H_sizes):
            # W value only on first H row for each grid
            if j == 0:
                w_str = _format_number_with_commas(int(grid))
            else:
                w_str = ""
            
            # Collect timing for each method
            timing_vals = []
            for method in methods:
                mask = (df["method"] == method) & (df["grid"] == grid) & (df["H"] == H)
                row_data = df[mask]
                
                if len(row_data) > 0:
                    # Euler error (log10 units, more negative = more accurate)
                    if "euler_error" in row_data.columns and pd.notna(row_data["euler_error"].values[0]):
                        euler_val = row_data["euler_error"].values[0]
                        euler_str = f"{euler_val:.2f}"
                    else:
                        euler_str = "--"

                    # Use per-period averages for timing metrics
                    if "avg_ue_time_per_period" in row_data.columns:
                        ue_time = row_data["avg_ue_time_per_period"].values[0]
                        total_time = row_data["avg_nonterminal_time_per_period"].values[0]
                    elif "total_ue_time" in row_data.columns:
                        # Fallback to totals for older results
                        ue_time = row_data["total_ue_time"].values[0]
                        total_time = row_data["total_nonterminal_time"].values[0]
                    else:
                        # No timing columns - bundles were loaded without solving
                        ue_time = np.nan
                        total_time = np.nan
                    
                    # Get average OWNC time
                    ownc_time = _get_avg_ownc_time(row_data)
                    
                    if pd.notna(total_time):
                        ownc_str = f"{ownc_time:.2f}" if pd.notna(ownc_time) else "--"
                        if method == "VFI":
                            # VFI has no upper envelope: keep the table compact by omitting the U.env column.
                            timing_vals.append(ownc_str)  # Cons. (s)
                            timing_vals.append(f"{total_time:.2f}")  # Period (s)
                            timing_vals.append(euler_str)  # Euler
                        elif pd.notna(ue_time):
                            # UE time in seconds, OWNC in seconds, total in seconds
                            timing_vals.append(f"{ue_time:.2f}")
                            timing_vals.append(ownc_str)
                            timing_vals.append(f"{total_time:.2f}")
                            timing_vals.append(euler_str)
                        else:
                            timing_vals.extend(["--", ownc_str, f"{total_time:.2f}", euler_str])
                    else:
                        if method == "VFI":
                            timing_vals.extend(["--", "--", euler_str])
                        else:
                            timing_vals.extend(["--", "--", "--", euler_str])
                else:
                    if method == "VFI":
                        timing_vals.extend(["--", "--", "--"])
                    else:
                        timing_vals.extend(["--", "--", "--", "--"])
            
            # Format row (W, H + timing data)
            row = f"{w_str} & {H} & " + " & ".join(timing_vals) + r" \\"
            lines.append(row)
        
        # Add midrule between grid groups
        lines.append(r"\midrule")
    
    # Close table with footnote - use exact text from user
    notes_text = (
        r"\textit{Notes:} All timings in seconds. "
        r"\textbf{U.env}: upper envelope time. "
        r"\textbf{Cons.}: owner consumption stage time. "
        r"\textbf{Per.}: total time per period. "
        r"\textbf{E.Err}: Euler equation error ($\log_{10}$). "
        + model_params
    )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.3em}",
        r"\par\small " + notes_text,
        r"\endgroup",
        r"\end{table}",
        r"\end{landscape}",
    ])
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote: {output_path}")


def generate_accuracy_markdown(df: pd.DataFrame, output_path: Path, model_params: str) -> None:
    """Generate Markdown accuracy table matching retirement format."""
    
    # Get unique values - internal names for data lookup
    methods = ["FUES", "DCEGM", "CONSAV", "VFI"]
    grids = sorted(df["grid"].unique())
    H_sizes = sorted(df["H"].unique())
    
    # Check if dev_c_L2 is available
    has_dev = "dev_c_L2" in df.columns and df["dev_c_L2"].notna().any()
    
    if has_dev:
        lines = [
            "# Housing-Renting Model: Solution Accuracy",
            "",
            "| Asset grid |    |    FUES     |     MSS     |     LTM     |     VFI     |",
            "|           | H  | Euler | L2  | Euler | L2  | Euler | L2  | Euler | L2  |",
            "|----------:|---:|------:|----:|------:|----:|------:|----:|------:|----:|",
        ]
    else:
        lines = [
            "# Housing-Renting Model: Solution Accuracy",
            "",
            "| Asset grid |    | FUES  |  MSS  |  LTM   | VFI   |",
            "|           | H  | Euler | Euler | Euler  | Euler |",
            "|----------:|---:|------:|------:|-------:|------:|",
        ]
    
    for grid in grids:
        for j, H in enumerate(H_sizes):
            # Get row prefix - format grid with commas for thousands
            if j == 0:
                row_prefix = f"{int(grid):,}"
            else:
                row_prefix = ""
            
            # Collect values for each method
            vals = []
            for method in methods:
                mask = (df["method"] == method) & (df["grid"] == grid) & (df["H"] == H)
                row_data = df[mask]
                
                # Euler error
                if len(row_data) > 0 and pd.notna(row_data["euler_error"].values[0]):
                    euler = row_data["euler_error"].values[0]
                    vals.append(f"{euler:.2f}")
                else:
                    vals.append("--")
                
                # Dev error (if available) - L2 norm, show in scientific notation
                if has_dev:
                    if len(row_data) > 0 and pd.notna(row_data["dev_c_L2"].values[0]):
                        dev = row_data["dev_c_L2"].values[0]
                        vals.append(f"{dev:.2e}")
                    else:
                        vals.append("--")
            
            # Format row
            row = f"| {row_prefix:6} | {H:2} | " + " | ".join(f"{v:8}" for v in vals) + " |"
            lines.append(row)
    
    dev_note = "L2 = normalized L2 consumption deviation from VFI baseline. " if has_dev else ""
    method_note = "MSS = Iskhakov et al. (2017); LTM = Druedahl & Jørgensen (2017). "
    lines.extend([
        "",
        "---",
        dev_note + method_note,
        "",
        "**Model Parameters:**",
        model_params.replace("$", "").replace(r"\beta", "β").replace(r"\phi", "φ")
                    .replace(r"\alpha", "α").replace(r"\delta", "δ").replace("_{", "_")
                    .replace("}", "").replace(r"\_", "_"),
    ])
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote: {output_path}")


def _get_avg_ownc_time(row_data: pd.DataFrame) -> float:
    """Get average OWNC time per period from pre-computed column.
    
    Uses the avg_ownc_time_per_period column that's computed during sweep
    from the stage_timings data before it's excluded from the CSV.
    """
    if "avg_ownc_time_per_period" in row_data.columns:
        val = row_data["avg_ownc_time_per_period"].values[0]
        if pd.notna(val):
            return val
    return np.nan


def generate_timing_markdown(df: pd.DataFrame, output_path: Path, model_params: str) -> None:
    """Generate Markdown timing table matching retirement format.
    
    U.env time is shown in milliseconds (×1000) for better readability.
    Period time remains in seconds.
    Cons. = average consumption stage time per period (to show VFI scaling).
    Euler error is included as a compact accuracy column (replaces U.env%).
    """
    
    # Get unique values - internal names for data lookup
    methods = ["FUES", "DCEGM", "CONSAV", "VFI"]
    grids = sorted(df["grid"].unique())
    H_sizes = sorted(df["H"].unique())
    
    lines = [
        "# Housing-Renting Model: Per-Period Timing Statistics",
        "",
        "| Asset grid |    |                    FUES                   |                    MSS                   |                    LTM                   |                    VFI                    |",
        "|           | H  | U.env(s) | Cons.(s) | Period(s) | Euler | U.env(s) | Cons.(s) | Period(s) | Euler | U.env(s) | Cons.(s) | Period(s) | Euler | Cons.(s) | Period(s) | Euler |",
        "|----------:|---:|--------:|--------:|---------:|------:|--------:|--------:|---------:|------:|--------:|--------:|---------:|------:|--------:|---------:|------:|",
    ]
    
    for grid in grids:
        for j, H in enumerate(H_sizes):
            # Get row prefix - format grid with commas for thousands
            if j == 0:
                row_prefix = f"{int(grid):,}"
            else:
                row_prefix = ""
            
            # Collect timing for each method
            timing_parts = []
            for method in methods:
                mask = (df["method"] == method) & (df["grid"] == grid) & (df["H"] == H)
                row_data = df[mask]
                
                if len(row_data) > 0:
                    # Euler error
                    if "euler_error" in row_data.columns and pd.notna(row_data["euler_error"].values[0]):
                        euler_val = row_data["euler_error"].values[0]
                        euler_str = f"{euler_val:5.2f}"
                    else:
                        euler_str = "   --"

                    # Use per-period averages for timing metrics
                    if "avg_ue_time_per_period" in row_data.columns:
                        ue_time = row_data["avg_ue_time_per_period"].values[0]
                        total_time = row_data["avg_nonterminal_time_per_period"].values[0]
                    elif "total_ue_time" in row_data.columns:
                        # Fallback to totals for older results
                        ue_time = row_data["total_ue_time"].values[0]
                        total_time = row_data["total_nonterminal_time"].values[0]
                    else:
                        # No timing columns - bundles were loaded without solving
                        ue_time = np.nan
                        total_time = np.nan
                    
                    # Get average OWNC time
                    ownc_time = _get_avg_ownc_time(row_data)
                    
                    if pd.notna(total_time):
                        ownc_str = f"{ownc_time:5.2f}" if pd.notna(ownc_time) else "   --"
                        if method == "VFI":
                            # VFI has no upper envelope: omit U.env column to keep table compact.
                            timing_parts.append(f"{ownc_str} | {total_time:5.2f} | {euler_str}")
                        elif pd.notna(ue_time):
                            # UE time in seconds
                            timing_parts.append(f"{ue_time:7.2f} | {ownc_str} | {total_time:5.2f} | {euler_str}")
                        else:
                            timing_parts.append(f"    -- | {ownc_str} | {total_time:5.2f} | {euler_str}")
                    else:
                        if method == "VFI":
                            timing_parts.append(f"    -- |    -- | {euler_str}")
                        else:
                            timing_parts.append(f"    -- |    -- |    -- | {euler_str}")
                else:
                    if method == "VFI":
                        timing_parts.append("    -- |    -- |    --")
                    else:
                        timing_parts.append("    -- |    -- |    -- |    --")
            
            # Format row
            row = f"| {row_prefix:6} | {H:2} | " + " | ".join(timing_parts) + " |"
            lines.append(row)
    
    # Add explanation of timing metrics and model parameters
    ue_explanation = (
        "**Timing Notes:** U.env (s) = average upper envelope time per period. "
        "Cons. (s) = average owner consumption stage solution time per period. "
        "Per. (s) = total solution time per period across all stages. "
        "Euler = Euler equation error (log₁₀; more negative is more accurate). "
        "VFI does not use an upper envelope. "
        "MSS = Iskhakov et al. (2017); LTM = Druedahl & Jørgensen (2017). "
    )
    lines.extend([
        "",
        "---",
        ue_explanation,
        "",
        "**Model Parameters:**",
        model_params.replace("$", "").replace(r"\beta", "β").replace(r"\phi", "φ")
                    .replace(r"\alpha", "α").replace(r"\delta", "δ").replace("_{", "_")
                    .replace("}", "").replace(r"\_", "_"),
    ])
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables from sweep results")
    parser.add_argument("--results", type=Path, help="Path to sweep_results.csv")
    parser.add_argument("--trial", type=str, default="paper-v1", 
                        help="Trial ID to load results from (default: paper-v1)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for tables (default: results/housing_renting/<trial>/)")
    parser.add_argument("--periods", type=int, default=None,
                        help="Number of periods T (reads from experiment YAML if not specified)")
    parser.add_argument("--experiment-set", type=str, default=None,
                        help="Experiment set name to read periods from (e.g., sweep_noPB_small)")
    args = parser.parse_args()
    
    # Determine results path
    if args.results:
        results_path = args.results
    else:
        results_path = SOLUTIONS_ROOT / args.trial / "sweep_results.csv"
    
    print(f"Loading results from: {results_path}")
    
    # Load data
    df = load_sweep_results(results_path)
    
    # Print summary
    print(f"\nLoaded {len(df)} configurations:")
    print(f"  Methods: {sorted(df['method'].unique())}")
    print(f"  Grid sizes: {sorted(df['grid'].unique())}")
    print(f"  H sizes: {sorted(df['H'].unique())}")
    
    # Determine periods (T) - from argument, experiment YAML, or default
    periods = args.periods
    if periods is None and args.experiment_set:
        # Try to read from experiment YAML
        try:
            import yaml
            exp_set_path = Path(__file__).parent.parent.parent.parent / "experiments" / "housing_renting" / "experiment_sets" / f"{args.experiment_set}.yml"
            if exp_set_path.exists():
                with open(exp_set_path) as f:
                    exp_config = yaml.safe_load(f)
                periods = exp_config.get("fixed", {}).get("periods", 5)
                print(f"  Read periods={periods} from {args.experiment_set}.yml")
        except Exception as e:
            print(f"  Warning: Could not read experiment set: {e}")
            periods = 5
    elif periods is None:
        periods = 5  # Default fallback
    
    # Model parameters string with proper LaTeX math formatting
    # Includes: preferences, grids, income process, and rental parameters
    model_params = (
        rf"Calibration: $r=0.06$, $\beta=0.93$, $T={periods}$, $\alpha=0.77$, "
        r"$\phi=0.07$, $\kappa=0.075$, $\iota=0.01$, "
        r"$A_{\max}=40$, $H_{\max}=5$, $w_{\max}=40$. "
        r"Income follows a 16-state Markov process ($\rho=0.977$, $\sigma_{\eta}=0.024$, $\sigma_{\varepsilon}=0.063$) as in \citet{Fella2014}. "
        r"Rental parameters: $P^{r}=0.1$, $\vartheta=0.5$."
    )
    
    # Generate tables - output in trial-specific subfolder
    if args.output:
        output_dir = args.output
    else:
        output_dir = RESULTS_ROOT / args.trial
    
    generate_accuracy_latex(df, output_dir / "housing_accuracy.tex", model_params)
    generate_timing_latex(df, output_dir / "housing_timing.tex", model_params)
    generate_accuracy_markdown(df, output_dir / "housing_accuracy.md", model_params)
    generate_timing_markdown(df, output_dir / "housing_timing.md", model_params)
    
    print(f"\nAll tables generated in: {output_dir}")


if __name__ == "__main__":
    main()
