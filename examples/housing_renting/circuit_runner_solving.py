#!/usr/bin/env python
"""
Housing model with renting using CircuitRunner.

This script loads, initializes, and solves the housing model with renting
using the StageCraft and Heptapod-B architecture, but leverages the 
unified CircuitRunner and sampler utilities (DynX v1.6.12).

Usage:
    python circuit_runner_solving.py [--periods N] [--vfi-ngrid SIZE]

Options:
    --periods N      Number of periods to simulate (default: 3)
    --vfi-ngrid SIZE Dense-grid size for VFI (e.g. 10000, 4e4, 1e6)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tabulate
import time
import copy
import json
import logging
from pathlib import Path
from typing import Dict

VALID_METHODS = {"VFI_HDGRID", "VFI", "VFI_MPI", "FUES", "CONSAV", "DCEGM", "FUES2DEV", "VFI_POOL"}
FAST_METHODS  = ["FUES", "CONSAV", "DCEGM"]

BASELINE = "VFI_POOL"                                     

# -----------------------------------------------------------------------------
# Canonical imports (DynX ≥ 1.6.12)
# -----------------------------------------------------------------------------

# Core graph / StageCraft
from dynx.stagecraft.makemod import (
    initialize_model_Circuit,
    compile_all_stages,
)

# Heptapod-B functional layer
from dynx.stagecraft.io import load_config   # <-- new canonical loader
# Runner utilities
from dynx.runner import CircuitRunner, mpi_map
from dynx.runner.sampler import FixedSampler, build_design

try:
    # Plotting helpers and error metrics (local to this example)
    from .helpers.plots import generate_plots
    from .helpers.plots import plot_egm_grids, plot_dcsn_policy
    from .helpers.euler_error import euler_error_metric

    from .whisperer import (
        build_operators,
        solve_stage,
        run_time_iteration,
    )
except ImportError:
    # Stub functions for docs build or when the heavy deps aren't present
    def generate_plots(model, method, output_dir):
        """Print a message when plotting is unavailable."""
        print(f"Plotting unavailable: {method} for {output_dir}")
    
    def plot_egm_grids(model, vf, c, m, method, stage_name):
        """Stub for plot_egm_grids function."""
        pass
    
    def plot_dcsn_policy(model, policies, periods=None, stage_names=None):
        """Stub for plot_dcsn_policy function."""
        pass
    
    def euler_error_metric(model):
        """Stub for metric calculation."""
        return 0.0
    
    def build_operators(stage):
        """Stub for operator builder."""
        pass
    
    def solve_stage(stage, max_iter=None, tol=None, verbose=False):
        """Stub for stage solver."""
        return True
    
    def run_time_iteration(model_circuit, n_periods=None, verbose=False, verbose_timings=False, recorder=None):
        """Stub for time iteration."""
        return True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- BEGIN METRICS CSV HELPER FUNCTIONS ---
def save_bundle_metrics(bundle_dir: Path, metrics: Dict[str, float]) -> None:
    """Overwrite <bundle>/metrics_table.csv with the latest metrics."""
    try:
        df = pd.DataFrame([metrics])
        df.to_csv(bundle_dir / "metrics_table.csv", index=False)
    except Exception as exc:
        print(f"[warn] could not write bundle metrics: {exc}")

def append_experiment_metrics(root_dir: Path, bundle_dir: Path,
                              metrics: Dict[str, float],
                              dedup_key_column: str) -> None:
    """Append a row to <root>/experiment_metrics.csv (dedup on a specified key column)."""
    try:
        current_run_data = dict(metrics) # Make a copy
        # Ensure bundle_dir is relative to root_dir/bundles for consistent paths
        bundles_root = root_dir / "bundles"
        if not bundle_dir.is_relative_to(bundles_root):
            # This case should ideally not happen if bundle_dir is from runner._bundle_path
            # and output_root is consistent. For safety, use absolute path if not relative.
            print(f"[warn] bundle_dir {bundle_dir} is not relative to {bundles_root}. Using full path for bundle_dir in CSV.")
            current_run_data["bundle_dir"] = str(bundle_dir.resolve())
        else:
            current_run_data["bundle_dir"] = str(bundle_dir.relative_to(bundles_root))
        
        df_new_row = pd.DataFrame([current_run_data])
        output_csv_path = root_dir / "experiment_metrics.csv"

        if output_csv_path.exists():
            df_old = pd.read_csv(output_csv_path)
            # Check if the deduplication key column exists in the old DataFrame
            if dedup_key_column in df_old.columns:
                # Concatenate and then drop duplicates based on the method name column
                df_combined = (pd.concat([df_old, df_new_row], ignore_index=True)
                               .drop_duplicates(subset=[dedup_key_column], keep="last"))
            else:
                # If the dedup key column doesn't exist in old CSV, append without trying to drop based on it.
                # This might lead to duplicates if the script is run multiple times with evolving schemas.
                # For a fresh start, deleting the old CSV is recommended.
                print(f"[warn] Dedup key '{dedup_key_column}' not found in existing {output_csv_path}. Appending data.")
                df_combined = pd.concat([df_old, df_new_row], ignore_index=True)
        else: # File doesn't exist
            df_combined = df_new_row
        
        df_combined.to_csv(output_csv_path, index=False)

    except Exception as exc:
        print(f"[warn] could not update experiment dashboard: {exc}")
# --- END METRICS CSV HELPER FUNCTIONS ---


def load_configs():
    """Return the canonical config container dict."""
    cfg_dir = Path(__file__).parent / "config_HR"
    return load_config(cfg_dir)


def initialize_housing_model(cfg_container, n_periods=3, vf_ngrid=1E+3):
    """
    Initialize a housing model circuit with the specified configuration.
    
    Parameters
    ----------
    master_config : dict
        Master configuration
    stage_configs : dict
        Stage-level configurations
    connections_config : dict
        Connection configuration
    n_periods : int, optional
        Number of periods in the model
        
    Returns
    -------
    ModelCircuit
        Initialized model circuit
    """
    # Deep copy configs to avoid modifying originals
    cfg = copy.deepcopy(cfg_container)
    cfg["master"]["horizon"] = n_periods         # or "periods"
    cfg["master"]["settings"]["N_arg_grid_vfi"] = vf_ngrid

    # todo: we should not have to manually do this below. 
    if cfg["master"]["methods"]["upper_envelope"] == "VFI_HDGRID" or cfg["master"]["methods"]["upper_envelope"] == "VFI" or cfg["master"]["methods"]["upper_envelope"] == "VFI_POOL":
        cfg["stages"]["OWNC"]["stage"]["methods"]["solution"] = cfg["master"]["methods"]["upper_envelope"]
        cfg["stages"]["RNTC"]["stage"]["methods"]["solution"] = cfg["master"]["methods"]["upper_envelope"]
    else:
        cfg["stages"]["OWNC"]["stage"]["methods"]["solution"] = "EGM"
        cfg["stages"]["RNTC"]["stage"]["methods"]["solution"] = "EGM"
    

    mc = initialize_model_Circuit(
        master_config   = cfg["master"],
        stage_configs   = cfg["stages"],
        connections_config = cfg["connections"],
    )
    compile_all_stages(mc)
    return mc


def metric_function(model):
    """
    Extract metrics from a solved model.
    
    Parameters
    ----------
    model : ModelCircuit
        Solved model circuit
        
    Returns
    -------
    float
        Euler equation error as a quality metric
    """
    # Use the new euler_error_metric function
    #print(">>> euler_error_metric called")
    return euler_error_metric(model)


def format_metrics_table(results_df):
    """
    Format run metrics into a table for display.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame from mpi_map
        
    Returns
    -------
    str
        Formatted table string
    """
    # Ensure tabulate module is available
    try:
        import tabulate
    except ImportError:
        return "Error: tabulate module not available. Install with 'pip install tabulate'"

    # Create a more readable metrics table
    metrics_data = []
    
    for i, row in results_df.iterrows():
        method = row.get("master.methods.upper_envelope", "Unknown")
        
        # Get the main metrics
        total_time = row.get("total_solution_time", 0.0)
        non_terminal_time = row.get("total_nonterminal_time", 0.0)
        ue_time = row.get("total_ue_time", 0.0)
        ue_percent = row.get("ue_time_percent", 0.0)
        euler_error = row.get("euler_error", 0.0)
        
        # Add to table data
        metrics_data.append([
            method,
            f"{total_time:.4f}s",
            f"{non_terminal_time:.4f}s",
            f"{ue_time:.4f}s",
            f"{ue_percent:.2f}%",
            f"{euler_error:.6f}"
        ])
    
    # Create table headers
    headers = [
        "Method",
        "Total Time",
        "Non-Terminal Time",
        "UE Time", 
        "UE/Non-Terminal",
        "Euler Error"
    ]
    
    # Format as table
    table = tabulate.tabulate(
        metrics_data,
        headers=headers,
        tablefmt="grid"
    )
    
    return table


def format_period_metrics(results_df):
    """
    Format period-by-period metrics into a table.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame from mpi_map
        
    Returns
    -------
    str
        Formatted table string for each method
    """
    # Ensure tabulate module is available
    try:
        import tabulate
    except ImportError:
        return "Error: tabulate module not available. Install with 'pip install tabulate'"
        
    tables = []
    
    for i, row in results_df.iterrows():
        method = row.get("master.operator.upper_envelope", "Unknown")
        period_timings = row.get("period_timings", [])
        
        # Handle case where period_timings might be a string representation of a list
        if isinstance(period_timings, str):
            try:
                # First try to parse as JSON
                import json
                period_timings = json.loads(period_timings)
            except:
                # If not valid JSON, try ast.literal_eval
                try:
                    import ast
                    period_timings = ast.literal_eval(period_timings)
                except:
                    period_timings = []
        
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
    """
    Format average stage metrics by method, grouped by stage name.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame from mpi_map
        
    Returns
    -------
    str
        Formatted table of average stage timings by method
    """
    # Ensure tabulate module is available
    try:
        import tabulate
        import pandas as pd
        from collections import defaultdict
    except ImportError:
        return "Error: tabulate module not available. Install with 'pip install tabulate'"
        
    method_tables = []
    
    for i, row in results_df.iterrows():
        method = row.get("master.methods.upper_envelope", "Unknown")
        stage_timings = row.get("stage_timings", [])
        
        # Handle case where stage_timings might be a string representation of a list
        if isinstance(stage_timings, str):
            try:
                # First try to parse as JSON
                import json
                stage_timings = json.loads(stage_timings)
            except:
                # If not valid JSON, try ast.literal_eval
                try:
                    import ast
                    stage_timings = ast.literal_eval(stage_timings)
                except:
                    stage_timings = []
        
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


def main(argv: list[str] | None = None) -> None:
    """
    Solve the housing-with-renting model for a reference solver plus a set of
    fast solvers, report deviation metrics and (optionally) generate plots.
    """
    # ------------------------------------------------------------------
    # 0) CLI arguments
    # ------------------------------------------------------------------
    p = argparse.ArgumentParser(prog="circuit_runner_solving.py")
    p.add_argument("--periods", type=int, default=3,
                   help="number of time periods")
    p.add_argument("--ue-method", default="ALL",
                   help="comma-separated fast methods; 'ALL' = built-ins")
    p.add_argument("--plot", action="store_true",
                   help="force plot generation (even if --no-plots present)")
    p.add_argument("--no-plots", action="store_true",
                   help="skip plot generation regardless of other flags")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output-root", type=str, default="solutions/HR_test_v2",
                   help="output root directory")
    p.add_argument("--bundle-prefix", type=str, default="HR_test_v2",
                   help="bundle prefix")
    p.add_argument("--vfi-ngrid", type=str, default="10000",
                   help="dense-grid size for VFI (e.g. 10000, 4e4, 1e6)")
    
    # Check for old argument and provide clear error
    if "--vfi-ngrid-index" in (argv or sys.argv[1:]):
        raise argparse.ArgumentError(None, 
                                   "ERROR: --vfi-ngrid-index has been renamed to --vfi-ngrid. "
                                   "Use --vfi-ngrid with the actual grid size (e.g. --vfi-ngrid 10000)")
    
    args = p.parse_args(argv or sys.argv[1:])
    
    # Convert vfi-ngrid string to numeric value
    try:
        vfi_ngrid_num = int(float(args.vfi_ngrid))   # "1e6" → 1000000
        if vfi_ngrid_num < 2:
            raise ValueError
    except Exception:
        raise SystemExit("ERROR: --vfi-ngrid must be a positive number (e.g. 10000 or 1e6)")

    # Keep both forms
    vfi_ngrid_token = args.vfi_ngrid       # for folder names
    vfi_ngrid = vfi_ngrid_num              # for the solver

    # ------------------------------------------------------------------
    # 1) configuration & runner
    # ------------------------------------------------------------------
    cfg_container = load_configs()
    param_paths   = ["master.methods.upper_envelope"]      # includes solver flag
    output_root = Path(f"{args.output_root}_{vfi_ngrid_token}").expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # ---- metrics -----------------------------------------------------
    from dynx.runner.metrics.deviations import dev_c_L2
    metric_fns = {
        "euler_error": metric_function,
        "dev_c_L2":    dev_c_L2,
    }

    # ---- solver wrapper ---------------------------------------------
    def solver(model_circuit, recorder=None):
        final_prd = model_circuit.get_period(len(model_circuit.periods_list) - 1)
        final_prd.get_stage("OWNC").status_flags["is_terminal"] = True
        final_prd.get_stage("RNTC").status_flags["is_terminal"] = True
        run_time_iteration(model_circuit, n_periods=args.periods,
                           verbose=args.verbose, verbose_timings=args.verbose,
                           recorder=recorder)
        return model_circuit


    runner = CircuitRunner(
        base_cfg        = copy.deepcopy(cfg_container),
        param_paths     = param_paths,
        model_factory   = lambda cfg: initialize_housing_model(cfg, n_periods=args.periods, vf_ngrid = vfi_ngrid),
        solver          = solver,
        metric_fns      = metric_fns,
        output_root     = output_root,
        bundle_prefix   = args.bundle_prefix,
        save_by_default = True,
        load_if_exists  = False,
        cache           = False,
    )

    # ------------------------------------------------------------------
    # 2) design matrix: "reference + fast methods"
    # ------------------------------------------------------------------
    REF_METHOD   = "VFI_POOL"
    FAST_DEFAULT = ["FUES", "FUES2DEV", "CONSAV", "DCEGM"]

    if args.ue_method.upper() == "ALL":
        fast_methods = FAST_DEFAULT
    else:
        fast_methods = [m.strip().upper()
                        for m in args.ue_method.replace("(", "").replace(")", "").split(",")
                        if m.strip()]
    methods = [REF_METHOD] + [m for m in fast_methods if m != REF_METHOD]

    Xs, _ = build_design(
        param_paths,
        samplers=[FixedSampler(np.array([[m] for m in methods], dtype=object))],
        Ns=[None], meta={}, seed=0,
    )

    # ------------------------------------------------------------------
    # 3) run the sweep
    # ------------------------------------------------------------------
    need_models = (args.plot or not args.no_plots)
    print(f"\nSolving: {', '.join(methods)}  (periods={args.periods})")
    t0 = time.time()
    results_df, models = mpi_map(runner, Xs,
                                 mpi=False,
                                 return_models=need_models)
    print(f"Completed in {time.time() - t0:.1f}s\n")

    # --- BEGIN METRICS CSV SAVING ---
    if results_df is not None:
        for i in range(len(Xs)): # Assuming Xs and results_df rows correspond
            x_param_vector = Xs[i] # Changed from Xs.iloc[i] as Xs is a numpy array
            metrics_for_run = results_df.iloc[i].to_dict()

            # 1. Per-bundle CSV
            bundle_dir_path = runner._bundle_path(x_param_vector)
            
            if bundle_dir_path:
                bundle_dir_path.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                save_bundle_metrics(bundle_dir_path, metrics_for_run)

            # 2. Global dashboard
            if runner.output_root and bundle_dir_path:
                # The key for deduplication will be the method name itself, 
                # which is the first (and only) parameter name in this setup.
                dedup_column_name = runner.param_paths[0] # e.g., "master.methods.upper_envelope"
                append_experiment_metrics(
                    root_dir=runner.output_root,
                    bundle_dir=bundle_dir_path,
                    metrics=metrics_for_run,
                    dedup_key_column=dedup_column_name
                )
    # --- END METRICS CSV SAVING ---

    # ------------------------------------------------------------------
    # 4) pretty summary -------------------------------------------------
    rows = []
    for _, r in results_df.iterrows():

        # decode ↦ dict  (it may already be a dict or NaN)
        cst_raw = r.get("consumption_stage_time", {})
        if isinstance(cst_raw, str):
            try:
                cst = json.loads(cst_raw)
            except Exception:
                cst = {}
        elif isinstance(cst_raw, dict):
            cst = cst_raw
        else:
            cst = {}

        rows.append([
            r.get("master.methods.upper_envelope", "—"),

            f"{r['euler_error']:.3e}"            if pd.notna(r.get("euler_error"))         else "—",
            f"{r['dev_c_L2']:.3e}"               if pd.notna(r.get("dev_c_L2"))            else "—",

            f"{r['total_solution_time']:.2f}s"        if pd.notna(r.get("total_solution_time"))        else "—",
            f"{r['total_nonterminal_time']:.2f}s"     if pd.notna(r.get("total_nonterminal_time"))     else "—",

            f"{cst.get('avg_consumption_time', float('nan')):.2f}s"
            if 'avg_consumption_time' in cst else "—",
        ])

    headers = [
        "Method",
        "Euler err",
        "‖c–c★‖₂",
        "Total time",
        "Non-term time",
        "⌀ Cons. time",
    ]

    table_str = tabulate.tabulate(rows, headers=headers, tablefmt="grid")

    print("\n=== Performance & Deviation Summary ===")
    print(table_str)
    print("  (c★ = VFI_POOL reference)\n")

    # ------------------------------------------------------------------
    # 5) persist the pretty summary ------------------------------------
    out_dir = Path(output_root).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "performance_summary.txt").write_text(table_str, encoding="utf-8")
    pd.DataFrame(rows, columns=headers).to_csv(out_dir / "performance_summary.csv", index=False)



    # ------------------------------------------------------------------
    # 5) optional plotting ---------------------------------------------
    if need_models and models is not None:
        image_dir = output_root / "images"
        image_dir.mkdir(exist_ok=True)
        for mdl, meth in zip(models, methods):
            try:
                generate_plots(mdl, meth, image_dir)
            except Exception as err:
                if args.verbose:
                    print(f"[warn] plots for {meth} failed: {err}")

    

    # ------------------------------------------------------------------
    # 6) return (for debugger / notebooks)
    # ------------------------------------------------------------------
    return dict(results=results_df, methods=methods)


if __name__ == "__main__":
    # Fix argument parsing for Cursor debugger
    # >>> ONLY KEEP THIS SECTION FOR INTERACTIVE DEBUGGING <<<
    # Comment it out (or wrap in an env-check) when you submit a batch run


    
    if os.getenv("FUES_DEBUG_LOCAL"):          # example guard
        sys.argv = [
            "circuit_runner_solving.py",
            "--periods", "5",
            "--output-root", "/scratch/tp66/as3442/FUES/solutions/HR_test_v3",
            "--bundle-prefix", "HR_test_v3",
            "--vfi-ngrid", "1e4",
        ]

    # hand control to the real entry point
    debug_results = main()
