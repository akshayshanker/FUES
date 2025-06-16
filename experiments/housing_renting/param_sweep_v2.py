#!/usr/bin/env python
"""
housing_renting_experiment_v2.py
==================================

Hierarchical MPI-based parameter sweep for the housing-with-renting model.

This script implements a two-level MPI parallelism strategy:
1.  **Inter-node parallelism (`COMM_TOP`)**: Distributes different parameter
    combinations (e.g., different β, γ values) to different compute nodes.
2.  **Intra-node parallelism (`COMM_SOLVER`)**: Uses all cores within a
    single node to solve the expensive `VFI_HDGRID` baseline for one
    specific parameter combination.

This approach allows for massive scaling across both parameter dimensions and
computational grid sizes. It leverages the memory-efficient solve-plot-delete
workflow from `solve_runner.py` to process each parameter combination
sequentially, minimizing memory footprint.

Key Features:
-   **Hierarchical MPI**: Scales to large parameter spaces and core counts.
-   **DynX Sampler Integration**: Robustly builds parameter design matrices.
-   **Memory-Efficient**: Processes one parameter combination at a time.
-   **Bundle Caching**: Automatically skips completed runs for easy restarts.
-   **Unified Workflow**: Combines parameter sweeps with the proven baseline +
    fast-method comparison logic.

Example Usage:
--------------

Run a 2x2 parameter sweep (β and γ) across 4 nodes, with each node using
45 cores for the VFI solver:

.. code-block:: bash

    mpiexec -np 180 \\
        python3 -m experiments.housing_renting.param_sweep_v2 \\
            --group-size 45 \\
            --param "master.parameters.beta=0.95,0.97" \\
            --param "master.parameters.gamma=2.0,4.0" \\
            --output "results/sweep_v2.csv" \\
            --plots

"""
from __future__ import annotations

import argparse
import copy
import gc
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

# ---------------------------------------------------------------------
# DynX and MPI imports
# ---------------------------------------------------------------------
from dynx.runner import CircuitRunner
from dynx.runner.sampler import Cartesian, Sampler
from dynx.stagecraft.io import load_config

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

# ---------------------------------------------------------------------
# Local helpers from the housing-renting example
# ---------------------------------------------------------------------
# Use solve_runner's factories and helpers directly
from examples.housing_renting.solve_runner import (
    BASE, FAST_METHODS, ALL_METHODS, CFG_DIR,
    make_housing_model, make_housing_solver,
    print_summary, generate_plots, euler_error_metric, dev_c_L2
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
#  Hierarchical MPI and Parameter Space Setup
# ────────────────────────────────────────────────────────────────────────────

def build_hierarchical_comms(group_size: int, mpi_enabled: bool) -> Tuple:
    """Build two-level MPI communicators for parameter sweeps."""
    if not mpi_enabled:
        return None, None, None, 0, 0

    world = MPI.COMM_WORLD
    rank = world.Get_rank()

    if world.Get_size() < group_size:
        if rank == 0:
            log.warning(
                f"World size ({world.Get_size()}) is smaller than group size "
                f"({group_size}). Running on a single node."
            )
        group_size = world.Get_size()

    node_id = rank // group_size
    intra_rank = rank % group_size

    # Communicator for all cores within a single node/group
    COMM_SOLVER = world.Split(color=node_id, key=intra_rank)

    # Communicator for the leaders (rank 0) of each node/group
    color_top = 0 if intra_rank == 0 else MPI.UNDEFINED
    COMM_TOP = world.Split(color=color_top, key=node_id)

    return world, COMM_SOLVER, COMM_TOP, node_id, intra_rank

def parse_param_spec(spec: str) -> Tuple[str, list]:
    """Parse param spec `PATH=v1,v2` or `PATH=min:max:N` into path and values."""
    if "=" not in spec:
        raise ValueError(f"Malformed --param '{spec}'. Expected PATH=...")
    path, rhs = spec.split("=", 1)
    path = path.strip()

    range_pat = re.compile(r"^([^:]+):([^:]+):([^:]+)$")
    m = range_pat.match(rhs)
    if m:
        lo, hi, n = float(m.group(1)), float(m.group(2)), int(m.group(3))
        vals = np.linspace(lo, hi, n).tolist()
    else:
        vals = [float(v) for v in rhs.split(",") if v.strip()]

    if not vals:
        raise ValueError(f"No values supplied for --param '{spec}'.")
    return path, vals


def build_parameter_space(param_specs: List[str]) -> Tuple[Dict, List[str]]:
    """Convert --param specifications to a Cartesian parameter space."""
    param_space_dict = {}
    for spec in param_specs:
        path, values = parse_param_spec(spec)
        param_space_dict[path] = values
    return Cartesian(param_space_dict), list(param_space_dict.keys())

def get_param_hash(runner: CircuitRunner, param_vec: np.ndarray) -> str:
    """Safely get a hash for a parameter vector."""
    # Per review note 2.7: Prefer a public API if it exists,
    # otherwise fall back to the private helper.
    if hasattr(runner, "param_hash"):
        return runner.param_hash(param_vec)
    return runner._hash_param_vec(param_vec)


# ────────────────────────────────────────────────────────────────────────────
#  Core Processing Logic
# ────────────────────────────────────────────────────────────────────────────

def process_parameter_combination(
    runner: CircuitRunner,
    base_params: np.ndarray,
    structural_param_paths: List[str],
    methods_to_run: List[str],
    args: argparse.Namespace,
    comm_solver: Any,
    output_root: Path
):
    """
    Process one parameter combination: solve baseline, then fast methods.
    Applies the memory-efficient solve-plot-delete-gc pattern.
    """
    all_metrics = []
    is_solver_root = comm_solver is None or comm_solver.rank == 0
    solver_rank = comm_solver.rank if comm_solver is not None else 0
    
    # Unpack for logging
    param_dict = dict(zip(structural_param_paths, base_params))
    if is_solver_root:
        log.info(f"Processing parameter combination: {param_dict}")

    HD_POINTS = int(args.HD_points)
    STD_POINTS = int(args.grid_points)

    # --- 1. Baseline Method ---
    if BASE in methods_to_run:
        runner.load_if_exists = not args.recompute_baseline
        if is_solver_root:
            log.info(f"  » Baseline ({BASE}): {'recompute' if args.recompute_baseline else 'load/solve'}")

        # Combine structural params with baseline-specific params
        baseline_full_params = np.concatenate([base_params, [BASE, HD_POINTS, HD_POINTS, HD_POINTS]])
        runner.ref_params = baseline_full_params

        try:
            metrics, model = runner.run(baseline_full_params, return_model=is_solver_root, rank=solver_rank)
            metrics["master.methods.upper_envelope"] = BASE
            metrics.update(param_dict)
            all_metrics.append(metrics)

            if is_solver_root and args.plots and model:
                img_dir = output_root / "images"
                img_dir.mkdir(exist_ok=True, parents=True)
                plot_name = f"{BASE}_{get_param_hash(runner, baseline_full_params)}"
                log.debug(f"    Generating plots for {plot_name}...")
                generate_plots(model, plot_name, img_dir)
        except Exception:
            log.error(f"  Failed to solve/process baseline for params: {param_dict}", exc_info=True)
        finally:
            if 'model' in locals() and model is not None:
                del model
                gc.collect()

    # --- 2. Fast Methods ---
    fast_methods_to_run = [m for m in FAST_METHODS if m in methods_to_run]
    if fast_methods_to_run:
        runner.load_if_exists = not args.fresh_fast
        if is_solver_root:
            log.info(f"  » Fast methods: {', '.join(fast_methods_to_run)}")

        for method in fast_methods_to_run:
            if is_solver_root:
                log.debug(f"    Solving {method}...")

            fast_full_params = np.concatenate([base_params, [method, STD_POINTS, STD_POINTS, STD_POINTS]])
            
            try:
                metrics, model = runner.run(fast_full_params, return_model=is_solver_root, rank=solver_rank)
                metrics["master.methods.upper_envelope"] = method
                metrics.update(param_dict)
                all_metrics.append(metrics)

                if is_solver_root and args.plots and model:
                    img_dir = output_root / "images"
                    img_dir.mkdir(exist_ok=True, parents=True)
                    plot_name = f"{method}_{get_param_hash(runner, fast_full_params)}"
                    log.debug(f"    Generating plots for {plot_name}...")
                    generate_plots(model, plot_name, img_dir)
            except Exception:
                log.error(f"  Failed to solve/process method {method} for params: {param_dict}", exc_info=True)
            finally:
                if 'model' in locals() and model is not None:
                    del model
                    gc.collect()

    return all_metrics

# ────────────────────────────────────────────────────────────────────────────
#  Main Runner
# ────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Hierarchical parameter sweep for the Housing-Renting model.")
    
    # --- MPI and Parallelism ---
    parser.add_argument("--mpi", action="store_true", help="Enable MPI parallelism.")
    parser.add_argument("--group-size", type=int, default=1, help="Number of cores per parameter combination (node size).")

    # --- Core Model and Sweep Parameters ---
    parser.add_argument("--param", action="append", default=[], metavar="PATH=VALS|min:max:N", help="Parameter to sweep.")
    parser.add_argument("--ue-methods", type=str, default="ALL", help="Comma-separated list of UE methods to run.")
    parser.add_argument("--periods", type=int, default=3, help="Number of periods.")

    # --- Grid and Precision ---
    parser.add_argument("--vfi-ngrid", type=int, default=10000, help="Choice grid points for VFI.")
    parser.add_argument("--HD-points", type=int, default=10000, help="Asset/wealth grid points for HD baseline.")
    parser.add_argument("--grid-points", type=int, default=4000, help="Asset/wealth grid points for fast methods.")
    
    # --- Caching and Output ---
    parser.add_argument("--output-root", type=str, default="solutions/HR_sweep", help="Root directory for sweep output.")
    parser.add_argument("--bundle-prefix", type=str, default="HR_sweep", help="Prefix for bundle directories.")
    parser.add_argument("--output", type=str, default=None, help="Output file for detailed results CSV.")
    parser.add_argument("--recompute-baseline", action="store_true", help="Force recomputation of baseline models.")
    parser.add_argument("--fresh-fast", action="store_true", help="Force recomputation of fast methods.")
    parser.add_argument("--plots", action="store_true", help="Generate plots for each run.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args(argv or sys.argv[1:])
    
    # --- 1. Setup MPI and Logging ---
    if args.verbose:
        log.setLevel(logging.DEBUG)

    use_mpi = args.mpi and HAS_MPI
    world, comm_solver, comm_top, node_id, intra_rank = build_hierarchical_comms(args.group_size, use_mpi)
    is_global_root = not use_mpi or world.rank == 0
    is_node_root = not use_mpi or comm_solver.rank == 0

    # --- 2. Build Parameter Space (on global root) ---
    design_matrix = None
    structural_param_paths = []
    if is_global_root:
        if not args.param:
            log.error("No parameters to sweep. Use --param to specify sweep dimensions.")
            sys.exit(1)
        param_space, structural_param_paths = build_parameter_space(args.param)
        design_matrix = Sampler(param_space).grid()
        log.info(f"Built design matrix with {design_matrix.shape[0]} parameter combinations.")
    
    # --- 3. Distribute Parameter Combinations to Nodes ---
    if use_mpi:
        # Bcast paths to all ranks
        structural_param_paths = world.bcast(structural_param_paths if is_global_root else None, root=0)

        # Node leaders get the design matrix and distribute work
        my_param_chunk = []
        if comm_top != MPI.COMM_NULL:
            design_matrix = comm_top.bcast(design_matrix if is_global_root else None, root=0)
            if design_matrix is not None:
                # Balanced round-robin distribution of parameter vectors to node leaders
                my_param_chunk = design_matrix[comm_top.rank::comm_top.Get_size()]

        # Each node leader now has a list of parameter vectors to process.
        # Broadcast this list to the workers within the node.
        my_param_chunk = comm_solver.bcast(my_param_chunk if is_node_root else None, root=0)
        
        # Now every rank has its list of work for its node.
        my_params_list = my_param_chunk

    else:
        # Serial run processes all combinations
        my_params_list = design_matrix
    
    # --- 4. Setup Runner and Process Combinations ---
    all_node_metrics = []

    # Each rank processes the parameter vectors assigned to its node
    for i, p_vec in enumerate(my_params_list):
        if is_node_root:
            log.info(f"--- Node {node_id} processing combination {i+1}/{len(my_params_list)} ---")
        
        # Configure runner for this specific parameter combination
        cfg_container = load_config(CFG_DIR)
        output_root = Path(args.output_root)
        
        # Define the full set of parameter paths for the runner
        all_param_paths = structural_param_paths + [
            "master.methods.upper_envelope", "master.settings.a_points",
            "master.settings.a_nxt_points", "master.settings.w_points"
        ]

        runner = CircuitRunner(
            base_cfg=cfg_container,
            param_paths=all_param_paths,
            model_factory=lambda cfg: make_housing_model(cfg, args.periods, int(args.vfi_ngrid), comm_solver),
            solver=make_housing_solver(args, use_mpi, comm_solver),
            metric_fns={"euler_error": euler_error_metric, "dev_c_L2": dev_c_L2},
            output_root=output_root,
            bundle_prefix=args.bundle_prefix,
            save_by_default=is_node_root,
            load_if_exists=True,
        )
        
        methods_to_run = ALL_METHODS if args.ue_methods.upper() == "ALL" else [m.strip().upper() for m in args.ue_methods.split(",")]

        node_metrics = process_parameter_combination(
            runner, p_vec, structural_param_paths, methods_to_run, args, comm_solver, output_root
        )
        if is_node_root:
            all_node_metrics.extend(node_metrics)

    # --- 5. Gather and Finalize Results ---
    if use_mpi:
        # Only node leaders (who have a valid COMM_TOP) participate in gathering.
        if comm_top != MPI.COMM_NULL:
            gathered_metrics = comm_top.gather(all_node_metrics, root=0)
        
        if is_global_root:
            # Flatten the list of lists
            final_metrics = [item for sublist in gathered_metrics if sublist for item in sublist]
        else:
            return # Workers are done
    else:
        final_metrics = all_node_metrics

    if is_global_root:
        if not final_metrics:
            log.warning("No metrics were collected. Exiting.")
            return

        df = pd.DataFrame(final_metrics)
        
        # Save detailed CSV
        output_file = args.output or Path(args.output_root) / "detailed_results.csv"
        df.to_csv(output_file, index=False)
        log.info(f"Detailed results saved to {output_file}")

        # Print summary table
        print_summary(df, Path(args.output_root))

if __name__ == "__main__":
    main() 