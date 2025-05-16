#!/usr/bin/env python
"""
Housing model with renting using CircuitRunner.

This script loads, initializes, and solves the housing model with renting
using the StageCraft and Heptapod-B architecture, but leverages the 
unified CircuitRunner and sampler utilities (DynX v1.6.12).

Usage:
    python circuit_runner_solving.py [--periods N]

Options:
    --periods N     Number of periods to simulate (default: 3)
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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# Canonical imports (DynX ≥ 1.6.12)
# -----------------------------------------------------------------------------

# Core graph / StageCraft
from dynx.stagecraft import Stage
from dynx.stagecraft.config_loader import (
    initialize_model_Circuit,
    compile_all_stages,
)

# Heptapod-B functional layer
from dynx.heptapodx.io.yaml_loader import load_config
from dynx.heptapodx.core.api import initialize_model
from dynx.heptapodx.num.generate import compile_num as generate_numerical_model

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


def load_configs():
    """Load all configuration files."""
    config_dir = os.path.join(os.path.dirname(__file__), "config_HR")
    master_path = os.path.join(config_dir, "housing_master.yml")
    ownh_path = os.path.join(config_dir, "OWNH_stage.yml") 
    ownc_path = os.path.join(config_dir, "OWNC_stage.yml")
    renth_path = os.path.join(config_dir, "RNTH_stage.yml")
    rentc_path = os.path.join(config_dir, "RNTC_stage.yml")
    tenu_path = os.path.join(config_dir, "TENU_stage.yml")
    connections_path = os.path.join(config_dir, "connections.yml")
    
    # Load configurations
    print("Loading configurations...")
    master_config = load_config(master_path)
    ownh_config = load_config(ownh_path)
    ownc_config = load_config(ownc_path)
    renth_config = load_config(renth_path)
    rentc_config = load_config(rentc_path)
    tenu_config = load_config(tenu_path)
    connections_config = load_config(connections_path)
    
    return {
        "master": master_config,
        "ownh": ownh_config,
        "ownc": ownc_config,
        "renth": renth_config,
        "rentc": rentc_config,
        "tenu": tenu_config,
        "connections": connections_config
    }


def initialize_housing_model(master_config, stage_configs, connections_config, n_periods=3):
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
    master_config = copy.deepcopy(master_config)
    stage_configs = copy.deepcopy(stage_configs)
    connections_config = copy.deepcopy(connections_config)
    
    # Set the number of periods in the master config
    master_config["periods"] = n_periods
    
    # Build the model circuit
    model_circuit = initialize_model_Circuit(
        master_config=master_config,
        stage_configs=stage_configs,
        connections_config=connections_config
    )
    
    compile_all_stages(model_circuit)
    print("test")
    return model_circuit


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
    Format stage-by-stage metrics into a table.
    
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
            tables.append(f"\n{method} - No stage timing data available")
            continue
            
        # Create stage data table - focus on slowest stages
        stage_data = []
        
        # Sort stages by total time (descending)
        try:
            sorted_stages = sorted(stage_timings, key=lambda x: x.get("total_time", 0), reverse=True)[:5]  # Top 5 slowest
            
            for stage in sorted_stages:
                stage_name = stage.get("stage_name", "Unknown")
                is_terminal = stage.get("is_terminal", False)
                total_time = stage.get("total_time", 0.0)
                cntn_to_dcsn_time = stage.get("cntn_to_dcsn_time", 0.0)
                dcsn_to_arvl_time = stage.get("dcsn_to_arvl_time", 0.0)
                ue_time = stage.get("ue_time", 0.0)
                
                # Calculate UE percentage for non-terminal stages
                ue_percent = (ue_time / total_time * 100) if not is_terminal and total_time > 0 else 0.0
                
                stage_data.append([
                    stage_name,
                    "Yes" if is_terminal else "No",
                    f"{total_time:.4f}s",
                    f"{cntn_to_dcsn_time:.4f}s",
                    f"{dcsn_to_arvl_time:.4f}s",
                    f"{ue_time:.4f}s" if not is_terminal else "N/A",
                    f"{ue_percent:.2f}%" if not is_terminal else "N/A"
                ])
        except Exception as e:
            tables.append(f"\n{method} - Error processing stage timings: {e}")
            continue
        
        # Create table
        headers = ["Stage", "Terminal", "Total Time", "cntn→dcsn", "dcsn→arvl", "UE Time", "UE Percent"]
        table = f"\n{method} - Top 5 Slowest Stages:\n" + tabulate.tabulate(
            stage_data,
            headers=headers,
            tablefmt="grid"
        )
        
        tables.append(table)
    
    return "\n".join(tables)


def main(argv=None):
    """Main driver function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Housing model with CircuitRunner")
    parser.add_argument("--periods", type=int, default=3, help="Number of periods to simulate")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--ue-method", type=str, choices=["FUES", "CONSAV", "DCEGM", "ALL"], 
                       default="ALL", help="Upper envelope method to use")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    
    # Load configurations
    configs = load_configs()
    
    # Create the unified base configuration for CircuitRunner
    base_cfg = {
        "master": configs["master"],
        "stages": {
            "OWNH": configs["ownh"],
            "OWNC": configs["ownc"],
            "RNTH": configs["renth"],
            "RNTC": configs["rentc"],
            "TENU": configs["tenu"],
        },
        "connections": configs["connections"],
    }
    
    # Define parameter paths
    param_paths = ["master.methods.upper_envelope"]
    
    # Define the solver function inline
    def solver(model_circuit, recorder=None):
        # Set terminal flags for last period's consumption stages
        final_period = model_circuit.get_period(len(model_circuit.periods_list) - 1)
        final_period.get_stage("OWNC").status_flags["is_terminal"] = True
        final_period.get_stage("RNTC").status_flags["is_terminal"] = True
        
        # Run time iteration
        all_stages_solved = run_time_iteration(model_circuit, verbose=args.verbose, 
                                              verbose_timings=args.verbose, recorder=recorder)
        return model_circuit
    
    # Create CircuitRunner with the new interface
    runner = CircuitRunner(
        base_cfg=base_cfg,
        param_paths=param_paths,
        model_factory=lambda cfg: initialize_housing_model(
            cfg["master"], cfg["stages"], cfg["connections"], n_periods=args.periods
        ),
        solver=solver,
        metric_fns={"euler_error": metric_function},
        cache=True,
    )
    
    # Create directory for images
    image_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(image_dir, exist_ok=True)
    
    # Run the parameter sweep
    print("\nRunning parameter sweep with CircuitRunner...")
    start_time = time.time()
    
    # Use sampler utilities to build the parameter design
    if args.ue_method == "ALL":
        methods = ["FUES", "CONSAV", "DCEGM"]
    else:
        methods = [args.ue_method]

    samplers = [FixedSampler(np.array([[m] for m in methods], dtype=object))]
    Xs, _ = build_design(param_paths, samplers, Ns=[None], meta={}, seed=0)
    
    # Suppress matplotlib warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, 
                          message="set_ticklabels() should only be used with")
    
    # Run models with selected methods and get the full metrics
    results_df, models = mpi_map(runner, Xs, mpi=False, return_models=True)


    
    # End of parameter sweep timing
    end_time = time.time()
    print(f"Parameter sweep completed in {end_time - start_time:.2f} seconds\n")
    
    # Display raw results
    print("Raw Results:")
    print(results_df)
    
    # Format and display metrics tables
    print("\nMetrics Summary:")
    print(format_metrics_table(results_df))
    print(format_period_metrics(results_df))
    print(format_stage_metrics(results_df))
    
    # Generate plots if requested or if plot flag is set
    if args.plot or (not args.no_plots and models is not None):
        for i, model in enumerate(models):
            # Get the method from the results dataframe
            method = results_df.iloc[i]["master.methods.upper_envelope"]

            # Safely generate plots, suppressing any errors
            try:
                generate_plots(model, method, image_dir)
            except Exception as plot_err:
                if args.verbose:
                    print(f"[Warning] Plot generation failed for method {method}: {plot_err}")
                # Continue without interrupting the rest of the script
    
    print("\nDone.")
    # Return the actual data instead of 0 for debugging
    return {
        'results_df': results_df,
        'models': models,
        'period_timings': [row.get('period_timings', []) for _, row in results_df.iterrows()],
        'stage_timings': [row.get('stage_timings', []) for _, row in results_df.iterrows()]
    }


if __name__ == "__main__":
    # Fix argument parsing for Cursor debugger
    import sys
    #sys.argv = ["circuit_runner_solving.py", "--periods", "4", "--ue-method", "FUES", "--verbose"]
    debug_results = main() 
