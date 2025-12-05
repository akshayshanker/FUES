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
    """Generate LaTeX accuracy table matching retirement format."""
    
    # Get unique values
    methods = ["FUES", "DCEGM", "CONSAV", "VFI"]
    grids = sorted(df["grid"].unique())
    H_sizes = sorted(df["H"].unique())
    
    # Start building LaTeX
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Housing-renting model -- Accuracy ($\log_{10}$)}",
        r"\label{tab:housing_accuracy}",
        r"\begin{tabular}{cc|c|c|c|c}",
        r"\toprule",
        r" & & FUES & DCEGM & CONSAV & VFI \\",
        r"Grid & H & Euler & Euler & Euler & Euler \\",
        r"\midrule",
    ]
    
    for i, grid in enumerate(grids):
        for j, H in enumerate(H_sizes):
            # Get row prefix
            if j == 0:
                row_prefix = str(grid)
            else:
                row_prefix = ""
            
            # Collect euler errors for each method
            euler_vals = []
            for method in methods:
                mask = (df["method"] == method) & (df["grid"] == grid) & (df["H"] == H)
                row_data = df[mask]
                
                if len(row_data) > 0 and pd.notna(row_data["euler_error"].values[0]):
                    euler = row_data["euler_error"].values[0]
                    euler_vals.append(f"{euler:.2f}")
                else:
                    euler_vals.append("--")
            
            # Format row
            row = f"{row_prefix} & {H} & " + " & ".join(euler_vals) + r" \\"
            lines.append(row)
        
        # Add spacing between grid groups (except for last)
        if i < len(grids) - 1:
            lines.append(r"\addlinespace[0.5em]")
    
    # Close table with centered footnote
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.5em}",
        r"\begin{minipage}{\textwidth}",
        r"\centering",
        r"\footnotesize",
        r"\textit{Notes:} " + model_params,
        r"\end{minipage}",
        r"\end{table}",
    ])
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote: {output_path}")


def generate_timing_latex(df: pd.DataFrame, output_path: Path, model_params: str) -> None:
    """Generate LaTeX timing table matching retirement format.
    
    UE time is shown in milliseconds (×1000) for better readability.
    Total time remains in seconds.
    """
    
    # Get unique values
    methods = ["FUES", "DCEGM", "CONSAV", "VFI"]
    grids = sorted(df["grid"].unique())
    H_sizes = sorted(df["H"].unique())
    
    # Start building LaTeX
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Housing-renting model -- Timing (UE in ms, Tot in s)}",
        r"\label{tab:housing_timing}",
        r"\begin{tabular}{cc|cc|cc|cc|cc}",
        r"\toprule",
        r" & & \multicolumn{2}{c|}{FUES} & \multicolumn{2}{c|}{DCEGM} & \multicolumn{2}{c|}{CONSAV} & \multicolumn{2}{c}{VFI} \\",
        r"Grid & H & UE & Tot & UE & Tot & UE & Tot & UE & Tot \\",
        r"\midrule",
    ]
    
    for i, grid in enumerate(grids):
        for j, H in enumerate(H_sizes):
            # Get row prefix
            if j == 0:
                row_prefix = str(grid)
            else:
                row_prefix = ""
            
            # Collect timing for each method
            timing_vals = []
            for method in methods:
                mask = (df["method"] == method) & (df["grid"] == grid) & (df["H"] == H)
                row_data = df[mask]
                
                if len(row_data) > 0:
                    ue_time = row_data["total_ue_time"].values[0]
                    # Use non-terminal time as the total (excludes terminal period setup)
                    total_time = row_data["total_nonterminal_time"].values[0]
                    
                    if pd.notna(ue_time) and pd.notna(total_time):
                        # UE time in milliseconds (2 decimal places), total in seconds
                        timing_vals.append(f"{ue_time * 1000:.2f}")
                        timing_vals.append(f"{total_time:.2f}")
                    else:
                        timing_vals.extend(["--", "--"])
                else:
                    timing_vals.extend(["--", "--"])
            
            # Format row
            row = f"{row_prefix} & {H} & " + " & ".join(timing_vals) + r" \\"
            lines.append(row)
        
        # Add spacing between grid groups (except for last)
        if i < len(grids) - 1:
            lines.append(r"\addlinespace[0.5em]")
    
    # Close table with centered footnote
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.5em}",
        r"\begin{minipage}{\textwidth}",
        r"\centering",
        r"\footnotesize",
        r"\textit{Notes:} " + model_params,
        r"\end{minipage}",
        r"\end{table}",
    ])
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote: {output_path}")


def generate_accuracy_markdown(df: pd.DataFrame, output_path: Path, model_params: str) -> None:
    """Generate Markdown accuracy table matching retirement format."""
    
    # Get unique values
    methods = ["FUES", "DCEGM", "CONSAV", "VFI"]
    grids = sorted(df["grid"].unique())
    H_sizes = sorted(df["H"].unique())
    
    lines = [
        "# Housing-renting model - Accuracy (log₁₀)",
        "",
        "|      |    | FUES  | DCEGM | CONSAV | VFI   |",
        "| Grid | H  | Euler | Euler | Euler  | Euler |",
        "|------|----|----- -|-------|--------|-------|",
    ]
    
    for grid in grids:
        for j, H in enumerate(H_sizes):
            # Get row prefix
            if j == 0:
                row_prefix = str(grid)
            else:
                row_prefix = ""
            
            # Collect euler errors for each method
            euler_vals = []
            for method in methods:
                mask = (df["method"] == method) & (df["grid"] == grid) & (df["H"] == H)
                row_data = df[mask]
                
                if len(row_data) > 0 and pd.notna(row_data["euler_error"].values[0]):
                    euler = row_data["euler_error"].values[0]
                    euler_vals.append(f"{euler:.2f}")
                else:
                    euler_vals.append("--")
            
            # Format row
            row = f"| {row_prefix:4} | {H:2} | " + " | ".join(f"{v:5}" for v in euler_vals) + " |"
            lines.append(row)
    
    lines.extend([
        "",
        "---",
        "**Model Parameters:**",
        model_params.replace("$", "").replace(r"\beta", "β").replace(r"\phi", "φ")
                    .replace(r"\alpha", "α").replace(r"\delta", "δ").replace("_{", "_")
                    .replace("}", "").replace(r"\_", "_"),
    ])
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote: {output_path}")


def generate_timing_markdown(df: pd.DataFrame, output_path: Path, model_params: str) -> None:
    """Generate Markdown timing table matching retirement format.
    
    UE time is shown in milliseconds (×1000) for better readability.
    Total time remains in seconds.
    """
    
    # Get unique values
    methods = ["FUES", "DCEGM", "CONSAV", "VFI"]
    grids = sorted(df["grid"].unique())
    H_sizes = sorted(df["H"].unique())
    
    lines = [
        "# Housing-renting model - Timing (UE in ms, Tot in s)",
        "",
        "|      |    |     FUES    |    DCEGM    |   CONSAV    |     VFI     |",
        "| Grid | H  | UE(ms)| Tot | UE(ms)| Tot | UE(ms)| Tot | UE(ms)| Tot |",
        "|------|----|----- -|-----|-------|-----|-------|-----|-------|-----|",
    ]
    
    for grid in grids:
        for j, H in enumerate(H_sizes):
            # Get row prefix
            if j == 0:
                row_prefix = str(grid)
            else:
                row_prefix = ""
            
            # Collect timing for each method
            timing_parts = []
            for method in methods:
                mask = (df["method"] == method) & (df["grid"] == grid) & (df["H"] == H)
                row_data = df[mask]
                
                if len(row_data) > 0:
                    ue_time = row_data["total_ue_time"].values[0]
                    # Use non-terminal time as the total (excludes terminal period setup)
                    total_time = row_data["total_nonterminal_time"].values[0]
                    
                    if pd.notna(ue_time) and pd.notna(total_time):
                        # UE time in milliseconds (2 decimal places), total in seconds
                        timing_parts.append(f"{ue_time * 1000:6.2f} | {total_time:5.2f}")
                    else:
                        timing_parts.append("   -- |    --")
                else:
                    timing_parts.append("   -- |    --")
            
            # Format row
            row = f"| {row_prefix:4} | {H:2} | " + " | ".join(timing_parts) + " |"
            lines.append(row)
    
    lines.extend([
        "",
        "---",
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
    
    # Model parameters string with proper LaTeX math formatting
    model_params = (
        r"Model parameters: $r=0.06$, $\beta=0.93$, $T=5$, $\phi=0.07$, "
        r"$\alpha=0.77$, $\delta_{\mathrm{PB}}=1.0$, $A_{\max}=35$, $H_{\max}=5$, $w_{\max}=40$. "
        r"Reference method: DCEGM."
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
