"""
Table formatting utilities for housing model benchmarks.

Handles pretty table formatting and summary displays.
"""

import json
import tabulate
from collections import defaultdict
from pathlib import Path
import pandas as pd


def safely_parse(obj):
    """Safely parse JSON/literal string or return original object."""
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except:
            try:
                import ast
                return ast.literal_eval(obj)
            except:
                return {}
    elif isinstance(obj, dict):
        return obj
    else:
        return {}


def format_metrics_table(results_df):
    """Format run metrics into a table for display."""
    metrics_data = []
    
    for i, row in results_df.iterrows():
        method = row.get("master.methods.upper_envelope", "Unknown")
        
        # Get the main metrics
        total_time = row.get("total_solution_time", 0.0)
        non_terminal_time = row.get("total_nonterminal_time", 0.0)
        ue_time = row.get("total_ue_time", 0.0)
        ue_percent = row.get("ue_time_percent", 0.0)
        euler_error = row.get("euler_error", 0.0)
        VFI_L2_error = row.get("dev_c_L2",0.0)
        log10_mean_error = row.get("dev_c_log10_mean", 0.0)
        param_hash = row.get("param_hash", "Unknown")
        reference_bundle_hash = row.get("reference_bundle_hash", "Unknown")
        latest_time_id = row.get("latest_time_id", "Unknown")
        
        # Add to table data
        metrics_data.append([
            method,
            f"{total_time:.4f}s",
            f"{non_terminal_time:.4f}s",
            f"{ue_time:.4f}s",
            f"{ue_percent:.2f}%",
            f"{euler_error:.6f}",
            f"{VFI_L2_error:.6f}",
            f"{log10_mean_error:.6f}",
            param_hash,
            reference_bundle_hash,
            latest_time_id
        ])
    
    # Create table headers
    headers = [
        "Method",
        "Total Time",
        "Non-Terminal Time",
        "UE Time", 
        "UE/Non-Terminal",
        "Euler Error",
        "VFI L2 Error",
        "Log10 Mean Error",
        "Param Hash",
        "Reference Bundle Hash",
        "Latest Time ID"
    ]
    
    # Format as table
    return tabulate.tabulate(
        metrics_data,
        headers=headers,
        tablefmt="grid"
    )


def format_period_metrics(results_df):
    """Format period-by-period metrics into a table."""
    tables = []
    
    for i, row in results_df.iterrows():
        method = row.get("master.operator.upper_envelope", "Unknown")
        period_timings = safely_parse(row.get("period_timings", []))
        
        if not period_timings:
            tables.append(f"\n{method} - No period timing data available")
            continue
            
        # Create period data table
        period_data = []
        
        # Sort periods in ascending order (from last to first)
        try:
            for period in sorted(period_timings, key=lambda x: x.get("period", 0)):
                period_num = period.get("period", 0)
                is_terminal = period.get("is_terminal", False)
                total_time = period.get("time", 0.0)
                ue_time = period.get("ue_time", 0.0)
                
                # Calculate UE percentage for non-terminal periods
                ue_percent = (ue_time / total_time * 100) if not is_terminal and total_time > 0 else 0.0
                
                period_data.append([
                    period_num,
                    "Yes" if is_terminal else "No",
                    f"{total_time:.4f}s",
                    f"{ue_time:.4f}s" if not is_terminal else "N/A",
                    f"{ue_percent:.2f}%" if not is_terminal else "N/A"
                ])
        except Exception as e:
            tables.append(f"\n{method} - Error processing period timings: {e}")
            continue
        
        # Create table
        headers = ["Period", "Terminal", "Total Time", "UE Time", "UE Percent"]
        table = f"\n{method} - Period-by-Period Metrics:\n" + tabulate.tabulate(
            period_data,
            headers=headers,
            tablefmt="grid"
        )
        
        tables.append(table)
    
    return "\n".join(tables)


def format_stage_metrics(results_df):
    """Format average stage metrics by method, grouped by stage name."""
    method_tables = []
    
    for i, row in results_df.iterrows():
        method = row.get("master.methods.upper_envelope", "Unknown")
        stage_timings = safely_parse(row.get("stage_timings", []))
        
        if not stage_timings:
            method_tables.append(f"\n{method} - No stage timing data available")
            continue
        
        # Filter out terminal stages and group by stage name
        stage_groups = defaultdict(list)
        
        for stage_info in stage_timings:
            stage_name = stage_info.get("stage_name", "Unknown")
            is_terminal = stage_info.get("is_terminal", False)
            
            # Skip terminal stages
            if is_terminal:
                continue
                
            # Add to the appropriate group
            stage_groups[stage_name].append(stage_info)
        
        # Calculate averages for each stage
        avg_data = []
        
        for stage_name, stage_list in stage_groups.items():
            if not stage_list:
                continue
                
            # Calculate averages
            avg_total = sum(s.get("total_time", 0.0) for s in stage_list) / len(stage_list)
            avg_cntn_to_dcsn = sum(s.get("cntn_to_dcsn_time", 0.0) for s in stage_list) / len(stage_list)
            avg_dcsn_to_arvl = sum(s.get("dcsn_to_arvl_time", 0.0) for s in stage_list) / len(stage_list)
            avg_ue = sum(s.get("ue_time", 0.0) for s in stage_list) / len(stage_list)
            
            # Calculate UE percentage
            ue_percent = (avg_ue / avg_total * 100) if avg_total > 0 else 0.0
            
            # Count how many instances of this stage we averaged
            count = len(stage_list)
            
            avg_data.append([
                stage_name,
                count,
                f"{avg_total:.4f}s",
                f"{avg_cntn_to_dcsn:.4f}s",
                f"{avg_dcsn_to_arvl:.4f}s",
                f"{avg_ue:.4f}s",
                f"{ue_percent:.2f}%"
            ])
        
        # Sort by average total time (descending)
        avg_data.sort(key=lambda x: float(x[2][:-1]), reverse=True)
        
        # Create table
        headers = ["Stage", "Count", "Avg Total", "Avg cntn→dcsn", "Avg dcsn→arvl", "Avg UE Time", "UE Percent"]
        table = f"\n{method} - Average Stage Timings (Non-terminal only):\n" + tabulate.tabulate(
            avg_data,
            headers=headers,
            tablefmt="grid"
        )
        
        method_tables.append(table)
    
    return "\n".join(method_tables)


def print_summary(results_df: pd.DataFrame, 
                  output_root: Path, 
                  config_details: dict | None = None) -> None:
    """
    Print a formatted summary of the results and save it to disk.
    
    Args:
        results_df: DataFrame containing the full experimental results.
        output_root: The root directory to save the summary files.
        config_details: Optional dictionary of configuration details to print.
    """


    # --- Header with config details ---
    if config_details:
        print("\n" + "=" * 80)
        header = " | ".join(f"{k}: {v}" for k, v in config_details.items())
        print(f"** Run Summary **\n{header}")
        print("=" * 80)

    # --- Main results table ---
    main_table_str = format_metrics_table(results_df)
    print("\n=== Performance & Deviation Summary ===")
    print(main_table_str)
    
    # --- File I/O: Append results to existing files ---
    output_root.mkdir(parents=True, exist_ok=True)
    raw_csv_path = output_root / "raw_metrics.csv"
    summary_txt_path = output_root / "performance_summary.txt"

    # --- Handle raw metrics CSV (append and deduplicate) ---
    if raw_csv_path.exists():
        try:
            existing_df = pd.read_csv(raw_csv_path)
            id_cols = [c for c in ['master.methods.upper_envelope', 'master.parameters.beta'] 
                       if c in existing_df.columns and c in results_df.columns]
            if not id_cols:
                id_cols = ['master.methods.upper_envelope']

            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            results_df = combined_df.drop_duplicates(subset=id_cols, keep='last').sort_values(by=id_cols)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            pass # File is empty or not found, will be overwritten
    
    results_df.to_csv(raw_csv_path, index=False)
    
    # --- Handle performance summary text file (append and regenerate) ---
    # We use the full, deduplicated DataFrame to generate the final summary
    final_summary_table_str = format_metrics_table(results_df)
    summary_txt_path.write_text(final_summary_table_str, encoding="utf-8")
    
    # Print the final, complete summary to the console
    print("\n=== Performance & Deviation Summary ===")
    print(final_summary_table_str)

    print(f"\nResults updated in: {output_root}")
    
    # Optional detailed tables are always overwritten with the latest full data
    if len(results_df) > 0:
        period_table = format_period_metrics(results_df)
        stage_table = format_stage_metrics(results_df)
        
        (output_root / "period_metrics.txt").write_text(period_table, encoding="utf-8")
        (output_root / "stage_metrics.txt").write_text(stage_table, encoding="utf-8")
        
        print(f"Detailed metrics saved to: {output_root}") 



def generate_latex_table(res_df: pd.DataFrame, output_root: Path):
    """
    Generates a LaTeX table from the results DataFrame and saves it to a .tex file.
    
    The table will compare different solution methods based on key metrics.
    """
    # Define columns to include and their LaTeX header names
    cols_to_latex = {
        "master.methods.upper_envelope": "Method",
        "solve_time": "Solve Time (s)",
        "euler_error": "Euler Error",
        "dev_c_L2": r"$\|c - c_{base}\|_{L_2}$",
        "dev_c_log10_mean": r"$\overline{\log_{10}|c - c_{base}|}$",
        "VFI_L2_error": r"$\|V - V_{base}\|_{L_2}$"
    }
    
    # Select and rename columns that exist in the dataframe
    existing_cols = [col for col in cols_to_latex.keys() if col in res_df.columns]
    table_df = res_df[existing_cols].rename(columns=cols_to_latex)

    # Convert to LaTeX format
    latex_table = table_df.to_latex(
        index=False,
        float_format="%.4f",
        caption="Comparison of solution methods against the high-density grid baseline.",
        label="tab:method_comparison",
        column_format="lrrrr",
        escape=False  # To allow LaTeX commands in headers
    )

    # Save to file
    table_path = output_root / "comparison_table.tex"
    with open(table_path, "w") as f:
        f.write(latex_table)
    
    print(f"LaTeX comparison table saved to: {table_path}")