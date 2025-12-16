"""
Command-line interface argument parsing for solve_runner.py.
"""

import argparse


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for solve_runner.py.
    
    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all housing model options.
    """
    p = argparse.ArgumentParser(
        prog="solve_runner.py",
        description="Solve baseline and fast UE methods for the HR model"
    )
    
    # Core model parameters
    p.add_argument("--periods", type=int, default=3,
                   help="Number of periods to solve")
    p.add_argument("--ue-method", default="ALL",
                   help="Upper envelope method to use")
    p.add_argument("--delta-pb", default="1",
                   help="Probability of death parameter")
    
    # Grid settings
    p.add_argument("--vfi-ngrid", default="10000",
                   help="VFI grid size for baseline")
    p.add_argument("--HD-points", default="10000",
                   help="High-density grid points")
    p.add_argument("--grid-points", default="4000",
                   help="Standard grid points")
    
    # Output settings
    p.add_argument("--output-root", default="solutions/HR",
                   help="Root directory for output files")
    p.add_argument("--config-id", default="HR",
                   help="Identifier for config directory (config_HR/{config-id}/)")
    p.add_argument("--experiment-set", default="default",
                   help="Experiment set name from experiments/housing_renting/experiment_sets/")
    p.add_argument("--RUN-ID", default="",
                   help="Optional run identifier suffix")
    
    # Plotting and export
    p.add_argument("--plots", action="store_true",
                   help="Generate matplotlib plots")
    p.add_argument("--skip-egm-plots", action="store_true",
                   help="Skip EGM plots when --plots is enabled (keeps policy plots only, saves time)")
    p.add_argument("--csv-export", action="store_true",
                   help="Export plot data to CSV files (includes both policy and EGM data)")
    
    # Execution mode
    p.add_argument("--mpi", action="store_true",
                   help="Enable MPI parallelization")
    p.add_argument("--gpu", action="store_true",
                   help="Use GPU-accelerated solvers")
    p.add_argument("--sweep", action="store_true",
                   help="Enable sweep mode: distribute (method, grid_size, H_points) combinations across MPI ranks using mpi_map")
    p.add_argument("--precompile", action="store_true",
                   help="Run a small VFI job to pre-compile Numba functions")
    
    # Method selection
    p.add_argument("--baseline-method", default=None,
                   help="Baseline method to use (default: auto-detect based on --gpu flag)")
    p.add_argument("--fast-methods", default=None,
                   help="Comma-separated list of fast methods (default: FUES,CONSAV)")
    p.add_argument("--include-baseline", action="store_true",
                   help="Automatically include baseline method in the method list")
    
    # Recomputation flags
    p.add_argument("--fresh-fast", action="store_true",
                   help="Recompute fast methods even if cached")
    p.add_argument("--recompute-baseline", action="store_true",
                   help="Recompute baseline even if cached")
    
    # Metrics
    p.add_argument("--metrics", default="all",
                   help="Comma-separated list of metrics to compute: euler_error, dev_c_L2, plots, all (default: all)")
    p.add_argument("--comparison-metrics", default="dev_c_L2,plot_c_comparison,plot_v_comparison",
                   help="Comma-separated list of comparison metrics that require baseline loading (default: dev_c_L2,plot_c_comparison,plot_v_comparison)")
    p.add_argument("--stages-L2dev", default="OWNC",
                   help="Stages to use for L2 deviation calculation")
    
    # Memory management
    p.add_argument("--low-memory", action="store_true",
                   help="Enable low memory mode - clears Q and lambda arrays after solving to save memory")
    p.add_argument("--save-full-model", action="store_true",
                   help="Keep all solution data in memory for saving. Disables memory freeing during solve. Default: False (free memory)")
    p.add_argument("--skip-bundle-save", action="store_true",
                   help="Skip saving solution bundles to disk. Useful for timing runs where only metrics are needed.")
    
    # Debugging
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose output")
    p.add_argument("--trace", action="store_true",
                   help="Enable debug trace statements to help diagnose memory issues")
    
    return p












