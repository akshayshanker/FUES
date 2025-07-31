#!/usr/bin/env python
"""
Housing model with renting – baseline HD grid search/fast-methods runner.

This script provides workflows for solving the housing-renting model using
different solution methods and comparing them against a reference method given
user-supplied metrics.

The script supports both serial and parallel execution using the Message
Passing Interface (MPI).

Workflows
---------
**Serial Execution (No MPI)**

1.  **Full Run: Fresh Baseline + Fast Methods**
    This computes the high-density baseline (`VFI_HDGRID`) and all fast
    methods from scratch.

    .. code-block:: bash

        python3 -m examples.housing_renting.solve_runner \\
            --periods 3 \\
            --ue-method ALL \\
            --output-root solutions/HR \\
            --bundle-prefix HR \\
            --vfi-ngrid 10000 \\
            --HD-points 10000 \\
            --grid-points 4000 \\
            --recompute-baseline \\
            --fresh-fast \\
            --plots

2.  **Fast Methods Only (with existing baseline)**
    This runs only the "fast" solvers (e.g., `FUES`, `CONSAV`), relying
    on a pre-computed baseline solution.

    .. code-block:: bash

        python3 -m examples.housing_renting.solve_runner \\
            --ue-method FUES,CONSAV \\
            --periods 3 \\
            --output-root solutions/HR \\
            --vfi-ngrid 100 \\
            --HD-points 10000 \\
            --fresh-fast \\
            --plots

    .. note::
       When ``--recompute-baseline`` is omitted, the runner searches for a
       baseline bundle matching the ``vfi-ngrid`` and ``HD-points``.

**Parallel Execution (MPI)**

For large-scale problems, MPI is recommended to accelerate the baseline
computation.

.. code-block:: bash

    mpiexec -np 45 \\
        python3 -m examples.housing_renting.solve_runner \\
          --periods 3 \\
          --ue-method ALL \\
          --output-root /scratch/tp66/$USER/FUES/solutions/HR_test_v4 \\
          --bundle-prefix HR_test_v4 \\
          --vfi-ngrid 100 \\
          --HD-points 650 \\
          --grid-points 500 \\
          --recompute-baseline \\
          --fresh-fast \\
          --mpi \\
          --plots
    .. note::
        The solvers currently require that the number of housing × income
        grid points be ≥ the number of MPI cores.

**GPU Execution**

For GPU-accelerated baseline computation:

.. code-block:: bash

    python3 -m examples.housing_renting.solve_runner \\
        --periods 3 \\
        --ue-method ALL \\
        --output-root solutions/HR_gpu \\
        --bundle-prefix HR_gpu \\
        --vfi-ngrid 10000 \\
        --HD-points 10000 \\
        --grid-points 4000 \\
        --recompute-baseline \\
        --fresh-fast \\
        --gpu \\
        --plots

Baseline Method Configuration
----------------------------
The baseline method can be configured in several ways:

1. **Automatic detection** (default):
   - Uses `VFI_HDGRID_GPU` when `--gpu` flag is present
   - Uses `VFI_HDGRID` otherwise

2. **Explicit specification**:
   .. code-block:: bash

       --baseline-method VFI_HDGRID  # Force non-GPU baseline
       --baseline-method VFI_HDGRID_GPU  # Force GPU baseline

3. **Loading existing baseline**:
   When running only fast methods with an existing baseline:
   
   .. code-block:: bash

       python3 -m examples.housing_renting.solve_runner \\
           --ue-method "FUES2DEV,CONSAV" \\
           --baseline-method VFI_HDGRID \\
           --include-baseline \\
           --fresh-fast

   The `--include-baseline` flag automatically adds the baseline method to the
   method list, enabling the runner to load the existing baseline bundle.

4. **Loading GPU baseline on CPU nodes**:
   To load a GPU-computed baseline when running on CPU-only nodes:
   
   .. code-block:: bash

       python3 -m examples.housing_renting.solve_runner \\
           --ue-method "FUES2DEV,CONSAV" \\
           --baseline-method VFI_HDGRID_GPU \\
           --include-baseline \\
           --fresh-fast

   This allows single-core jobs to leverage GPU-computed baselines without
   requiring GPU access.

Fast Methods Configuration
-------------------------
Custom fast methods can be specified using:

.. code-block:: bash

    --fast-methods "FUES,FUES2DEV,CONSAV,DCEGM"

If not specified, defaults to: `FUES2DEV,CONSAV`

Comparison Metrics Optimization
------------------------------
When running only the baseline method, comparison metrics (dev_c_L2, plot comparisons) 
are automatically skipped since comparing a model against itself is meaningless and wastes 
computational time. This behavior prevents walltime exceeded errors on baseline-only runs.

To customize which metrics are considered comparison metrics, use:
``--comparison-metrics "dev_c_L2,my_custom_comparison"``

Selective Model Loading
-----------------------
When loading existing models (e.g., for metric calculation), you can specify which
periods and stages to load to reduce memory usage and I/O time:

.. code-block:: bash

    # Load only periods 0 and 1 for Euler error calculation
    python3 -m examples.housing_renting.solve_runner \\
        --ue-method VFI_HDGRID_GPU \\
        --load-periods "0,1" \\
        --load-stages '{"0": ["OWNC"], "1": null}' \\
        --metrics euler_error

This loads only OWNC from period 0 and all stages from period 1, reducing loading
time by ~76% for a 5-period model.

Command-Line Arguments
---------------------
Key arguments for method configuration:

- ``--baseline-method``: Explicitly set baseline method (auto-detects based on --gpu if omitted)
- ``--fast-methods``: Comma-separated list of fast methods to compare
- ``--include-baseline``: Automatically include baseline in method list for loading
- ``--gpu``: Enable GPU acceleration (also affects baseline auto-detection)
- ``--mpi``: Enable MPI parallelization
- ``--recompute-baseline``: Force recomputation of baseline even if bundle exists
- ``--fresh-fast``: Force recomputation of fast methods
- ``--comparison-metrics``: Comma-separated list of metrics requiring baseline comparison (default: dev_c_L2,plot_c_comparison,plot_v_comparison)
- ``--load-periods``: Comma-separated list of period indices to load (e.g., '0,1' for Euler error)
- ``--load-stages``: JSON dict of period:stages to load (e.g., '{"0": ["OWNC"], "1": null}')

Directory Organization
---------------------
The runner uses a standardized directory structure for outputs:

- Base path: ``{output-root}/{bundle-prefix}/``
- With TRIAL_ID: ``{output-root}/{bundle-prefix}_{TRIAL_ID}/``

This allows organizing different experimental runs (e.g., GPU vs CPU baselines)
by setting the TRIAL_ID environment variable in job scripts.

"""

from __future__ import annotations

# Global trace flag - check for --trace in command line args early
import sys
_TRACE_ENABLED = '--trace' in sys.argv

def trace_print(message):
    """Print trace message only if tracing is enabled"""
    if _TRACE_ENABLED:
        print(f"[TRACE] {message}", flush=True)

trace_print("0.1: Starting imports")
import argparse
import copy
import sys
import time
from pathlib import Path
import gc
import resource
import json

trace_print("0.2: Basic imports done")
import numpy as np
import pandas as pd

trace_print("0.3: Numpy/Pandas imported")
from dynx.runner import CircuitRunner, write_design_matrix_csv
from dynx.stagecraft.io import load_config
from dynx.stagecraft.makemod import initialize_model_Circuit, compile_all_stages

trace_print("0.4: DynX imports done")

# ────────────────────────────────────────────────────────────────────────────
#  local helpers (imported lazily to keep fall-back stubs tiny)
# ────────────────────────────────────────────────────────────────────────────
try:
    from whisperer import build_operators_for_circuit, run_time_iteration
    from helpers.euler_error import euler_error_metric
    from helpers.plots import generate_plots
    from helpers.tables import print_summary, generate_latex_table
    from helpers.metrics import dev_c_L2, dev_v_L2, plot_comparison_factory
    from helpers.memory_utils import MemoryMonitor, log_memory_usage, cleanup_if_needed, get_memory_config
except ImportError:
    from .whisperer import build_operators_for_circuit, run_time_iteration
    from .helpers.euler_error import euler_error_metric
    from .helpers.plots import generate_plots
    from .helpers.tables import print_summary, generate_latex_table
    from .helpers.metrics import dev_c_L2, dev_v_L2, plot_comparison_factory
    from .helpers.memory_utils import MemoryMonitor, log_memory_usage, cleanup_if_needed, get_memory_config

trace_print("0.5: Local helpers imported")

CFG_DIR = Path(__file__).parent / "config_HR"
# Default baseline method - can be overridden by --baseline-method
DEFAULT_BASE = "VFI_HDGRID_GPU"
# All available methods
ALL_METHODS = ["VFI_HDGRID", "VFI_HDGRID_GPU", "FUES", "FUES2DEV", "CONSAV", "DCEGM", "FUES2DEV5"]
# Fast methods that are compared against baseline
DEFAULT_FAST_METHODS = ["FUES2DEV", "CONSAV", "FUES", "FUES2DEV5"]
# Pre-compilation parameters
PRE_COMPILE_PARAMS = np.array(["VFI_HDGRID_GPU", 500, 500, 500], dtype=object)

trace_print("0.6: Constants and globals set")

egm_bounds = {
      'value_h14': (2, 4, 2.5, 4),      # Left panel: x-axis 0-5, y-axis 0-3
      'assets_h14': (2, 4, 0.5, 3.5), # Right panel: x-axis 0-5, y-axis auto
  }


def cleanup_model(model, aggressive=False):
    """Clean up model data to free memory.
    
    Parameters
    ----------
    model : ModelCircuit
        The model to clean up
    aggressive : bool
        If True, clear all solution data. If False, keep only essential data.
    """
    if model is None:
        return
        
    try:
        # Clear large arrays from all periods and stages
        for period_idx in range(len(model.periods_list)):
            period = model.get_period(period_idx)
            for stage_name, stage in period.stages.items():
                for perch_name, perch in stage.perches.items():
                    if hasattr(perch, 'sol') and perch.sol is not None:
                        # Clear Q and lambda arrays which are typically not needed post-solve
                        if hasattr(perch.sol, '_jit'):
                            if aggressive:
                                # Clear everything
                                perch.sol._jit.Q = np.empty((0,), dtype=np.float64)
                                perch.sol._jit.lambda_ = np.empty((0,), dtype=np.float64)
                                perch.sol._jit.vlu = np.empty((0,), dtype=np.float64)
                            else:
                                # Just clear Q and lambda
                                perch.sol._jit.Q = np.empty((0,), dtype=np.float64)
                                perch.sol._jit.lambda_ = np.empty((0,), dtype=np.float64)
                        
                        # Clear EGM intermediate arrays if present
                        if hasattr(perch.sol, '_jit') and hasattr(perch.sol._jit, 'EGM'):
                            for layer in ["unrefined", "refined", "interpolated"]:
                                if layer in perch.sol._jit.EGM:
                                    perch.sol._jit.EGM[layer].clear()
    except Exception as e:
        print(f"Warning: Error during model cleanup: {e}")


def patch_cfg(args,cfg_container: dict, periods: int, vf_ngrid: int) -> dict:
    """
    Patch config with solution method and MPI compute settings.

    The upper-envelope method (`VFI_HDGRID`, `VFI_HDGRID_GPU`, etc.) determines
    whether the underlying solver should be VFI, EGM, or GPU. This helper
    sets the correct `solution` and `compute` methods on consumption
    stages without requiring extra CLI flags.
    """
    cfg = copy.deepcopy(cfg_container)
    cfg["master"]["horizon"] = periods
    cfg["master"]["settings"]["N_arg_grid_vfi"] = vf_ngrid

    sol = cfg["master"]["methods"]["upper_envelope"]
    target = sol if sol in ("VFI_HDGRID", "VFI", "VFI_POOL","VFI_HDGRID_GPU") else "EGM"
    cfg["stages"]["OWNC"]["stage"]["methods"]["solution"] = target
    cfg["stages"]["RNTC"]["stage"]["methods"]["solution"] = target

    if sol == "VFI_HDGRID" and args.mpi:
        cfg["stages"]["OWNC"]["stage"]["methods"]["compute"] = "MPI"
        cfg["stages"]["RNTC"]["stage"]["methods"]["compute"] = "MPI"
    elif sol == "VFI_HDGRID_GPU" and args.gpu:
        cfg["stages"]["OWNC"]["stage"]["methods"]["compute"] = "GPU"
        cfg["stages"]["RNTC"]["stage"]["methods"]["compute"] = "GPU"
        cfg["stages"]["OWNH"]["stage"]["methods"]["compute"] = "GPU"
        cfg["stages"]["RNTH"]["stage"]["methods"]["compute"] = "GPU"
    else:
        cfg["stages"]["OWNC"]["stage"]["methods"]["compute"] = "SINGLE"
        cfg["stages"]["RNTC"]["stage"]["methods"]["compute"] = "SINGLE"

    # compute h choice with GPU if avail in all cases. 
    if args.gpu:
        cfg["stages"]["OWNH"]["stage"]["methods"]["compute"] = "GPU"
        cfg["stages"]["RNTH"]["stage"]["methods"]["compute"] = "GPU"

    return cfg


def make_housing_model(args, cfg_container: dict, periods: int, vf_ngrid: int, comm=None):
    """
    Patch the config, build a ModelCircuit, and compile its stages.

    This helper prepares a model for the solver. It:
    1.  Patches the YAML config for the correct solution method.
    2.  Initializes the `ModelCircuit` skeleton.
    3.  Numerically compiles all stages (e.g., creating grids).

    Any compile-time warnings are logged, and hard failures are re-raised so
    that the calling runner can handle them cleanly.

    Returns
    -------
    ModelCircuit
        A fully initialized and compiled model circuit.
    """
    # 1. patch master + stage configs
    cfg = patch_cfg(args, cfg_container, periods, vf_ngrid)

    # 2. build ModelCircuit skeleton (no heavy maths yet)
    mc = initialize_model_Circuit(
        master_config=cfg["master"],
        stage_configs=cfg["stages"],
        connections_config=cfg["connections"],
    )

    # 3. numerically compile every Stage (grid creation, etc.)
    try:
        # 'force=False': skip already-compiled stages when re-loading bundles
        if comm is None or comm.rank == 0:
            compile_all_stages(mc, force=False)
    except Exception as exc:
        # logger.error("Stage compilation failed – aborting make_housing_model()", exc_info=True)
        raise

    return mc


# ---------------------------------------------------------------------
#  Solver-factory helper
# ---------------------------------------------------------------------
def make_housing_solver(args, use_mpi: bool, comm):
    """
    Returns an _solve(model, recorder) closure that the CircuitRunner calls.
    """
    def _solve(mc, recorder=None):
        """
        This closure correctly uses the MPI settings passed to the factory's scope
        """
        # Correctly use the use_mpi flag from the factory's scope
        build_operators_for_circuit(mc, use_mpi=use_mpi, comm=comm)

        # 1. mark last-period consumption stages as terminal
        final_period = mc.get_period(len(mc.periods_list) - 1)
        for tag in ("OWNC", "RNTC"):
            final_period.get_stage(tag).status_flags["is_terminal"] = True
        
        # 2. backward time iteration
        run_time_iteration(
            mc,
            n_periods=args.periods,
            verbose=args.verbose,
            verbose_timings=args.verbose,
            recorder=recorder,
        )
        return mc

    return _solve


# ────────────────────────────────────────────────────────────────────────────
#  CLI + main
# ────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    """
    Command-line interface for solving and benchmarking the Housing–Renting model.
    """
    global _TRACE_ENABLED
    
    trace_print("1: Starting main()")
    
    p = argparse.ArgumentParser(
        prog="solve_runner.py",
        description="Solve baseline and fast UE methods for the HR model"
    )
    p.add_argument("--periods", type=int, default=3)
    p.add_argument("--ue-method", default="ALL")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output-root", default="solutions/HR")
    p.add_argument("--bundle-prefix", default="HR")
    p.add_argument("--vfi-ngrid", default="10000")
    p.add_argument("--HD-points", default="10000")
    p.add_argument("--grid-points", default="4000")
    p.add_argument("--mpi", action="store_true")
    p.add_argument("--fresh-fast", action="store_true")
    p.add_argument("--recompute-baseline", action="store_true")
    p.add_argument("--precompile", action="store_true", help="Run a small VFI job to pre-compile Numba functions.")
    p.add_argument("--RUN-ID", default="")
    p.add_argument("--stages-L2dev", default="OWNC")
    p.add_argument("--delta-pb", default="1")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--baseline-method", default=None, 
                   help="Baseline method to use (default: auto-detect based on --gpu flag)")
    p.add_argument("--fast-methods", default=None,
                   help="Comma-separated list of fast methods (default: FUES2DEV,CONSAV)")
    p.add_argument("--include-baseline", action="store_true",
                   help="Automatically include baseline method in the method list")
    p.add_argument("--trace", action="store_true",
                   help="Enable debug trace statements to help diagnose memory issues")
    p.add_argument("--metrics", default="all",
                   help="Comma-separated list of metrics to compute: euler_error, dev_c_L2, plots, all (default: all)")
    p.add_argument("--low-memory", action="store_true",
                   help="Enable low memory mode - clears Q and lambda arrays after solving to save memory")
    p.add_argument("--comparison-metrics", default="dev_c_L2,plot_c_comparison,plot_v_comparison",
                   help="Comma-separated list of comparison metrics that require baseline loading (default: dev_c_L2,plot_c_comparison,plot_v_comparison)")
    p.add_argument("--load-periods", default=None,
                   help="Comma-separated list of period indices to load when loading existing models (e.g., '0,1' for Euler error). If not specified, loads all periods.")
    p.add_argument("--load-stages", default=None,
                   help="JSON-formatted dict of period:stages to load (e.g., '{\"0\": [\"OWNC\"], \"1\": null}' loads only OWNC in period 0, all stages in period 1)")
    
    args = p.parse_args(argv or sys.argv[1:])
    
    # Enable tracing based on command line flag
    _TRACE_ENABLED = args.trace
    
    # Configure memory management based on environment
    memory_config = get_memory_config("cluster")  # Default to cluster settings
    
    # Log initial memory usage
    log_memory_usage("at start of solve_runner")
    
    trace_print("2: Args parsed")

    #  MPI communicator
    from dc_smm.helpers.mpi_utils import get_comm
    comm = get_comm(args.mpi)
    trace_print(f"3: MPI comm created, rank={comm.rank if comm else 0}")
    
    if comm.size > 1 and not args.mpi and comm.rank != 0:
        comm.Barrier()
        sys.exit(0)

    #  parse grid size
    vf_ngrid = int(float(args.vfi_ngrid))
    pb_delta = float(args.delta_pb)
    trace_print("4: Grid sizes parsed")

    # Determine baseline method
    if args.baseline_method:
        BASE = args.baseline_method.upper()
    else:
        # Auto-detect based on GPU flag
        BASE = "VFI_HDGRID_GPU" if args.gpu else "VFI_HDGRID"
    trace_print(f"5: Baseline method: {BASE}")
    
    # Determine fast methods
    if args.fast_methods:
        FAST_METHODS = [m.strip().upper() for m in args.fast_methods.split(",")]
    else:
        FAST_METHODS = DEFAULT_FAST_METHODS
    trace_print(f"6: Fast methods: {FAST_METHODS}")

    #  method list
    methods = ALL_METHODS if args.ue_method.upper() == "ALL" \
        else [m.strip().upper() for m in args.ue_method.split(",")]
    
    # Automatically include baseline if requested and not already present
    if args.include_baseline and BASE not in methods:
        methods.insert(0, BASE)
    trace_print(f"7: Methods to run: {methods}")

    #  IO paths
    packroot = Path.cwd()
    output_root = packroot / f"{args.output_root}"
    output_root.mkdir(parents=True, exist_ok=True)
    cfg_dir_bundle = CFG_DIR /  f"{args.bundle_prefix}"
    trace_print("8: Paths created")
    
    #  set-up runner ------------------------------------------------------
    cfg_container = load_config(cfg_dir_bundle)
    trace_print("9: Config loaded")
    
    save_by_default = (comm is None) or (comm.rank == 0)
    is_root = (comm is None) or (comm.rank == 0)
    solver_rank = comm.rank if comm is not None else 0
    trace_print(f"10: is_root={is_root}, solver_rank={solver_rank}")

    # --- Optional Pre-compilation Step ---
    if args.precompile and is_root:
        trace_print("11: Starting precompilation")
        print("\n--- Running Numba Pre-compilation ---")
        
        # --- Create a minimal config for the pre-compilation run ---
        precompile_cfg = copy.deepcopy(cfg_container)
        precompile_cfg['master']['settings']['a_points'] = 100
        precompile_cfg['master']['settings']['w_points'] = 100
        precompile_cfg['master']['settings']['a_nxt_points'] = 100
        
        # Use the determined baseline method for precompilation
        precompile_params = np.array([BASE, 100, 100, 100,pb_delta ], dtype=object)

        precompile_runner = CircuitRunner(
            base_cfg=precompile_cfg, # Use the minimal config
            param_paths=[
                "master.methods.upper_envelope",
                "master.settings.a_points",
                "master.settings.a_nxt_points",
                "master.settings.w_points",
                "master.parameters.delta_pb",
            ],
            model_factory=lambda cfg: make_housing_model(args,cfg, 2, 100, comm),
            solver=make_housing_solver(argparse.Namespace(verbose=False, periods=2), use_mpi=args.mpi, comm=comm),
            metric_fns={},
            save_by_default=False,
            load_if_exists=False,
        )
        try:
            precompile_runner.run(precompile_params, rank=solver_rank)
            if is_root:
                print("--- Pre-compilation Complete ---\n")
        except Exception as e:
            if is_root:
                print(f"--- Pre-compilation Failed: {e} ---", file=sys.stderr)
        
        trace_print("12: Precompilation complete")

    if comm is not None:
        comm.Barrier()

    trace_print("13: Setting up metric functions")
    #  set-up plotting configuration -----------------------------------------------
    # Define the dimension labels for the model's policy arrays
    asset_dims = {
        0: 'w_idx',    # wealth capital (housing)
        1: 'h_idx',    # Liquid assets
        2: 'y_idx'    # The decision/choice axis
    }

    # Define which specific indices to generate plots for
    plots_of_interest = {
        'h_idx': [5, 10, 14]  # Generate plots only for these h indices (0-indexed)
    }

    # Available metrics mapping
    AVAILABLE_METRICS = {
        "euler_error": euler_error_metric,
        "dev_c_L2": dev_c_L2,
        "plot_c_comparison": plot_comparison_factory(
            decision_variable='c',
            dim_labels=asset_dims,
            plot_axis_label='w_idx',
            slice_config=plots_of_interest
        ),
        "plot_v_comparison": plot_comparison_factory(
            decision_variable='vlu',
            dim_labels=asset_dims,
            plot_axis_label='w_idx',
            slice_config=plots_of_interest,
            sol_attr='value'
        ),
    }

    # Parse requested metrics
    if args.metrics.lower() == "all":
        requested_metrics = list(AVAILABLE_METRICS.keys())
    else:
        requested_metrics = [m.strip() for m in args.metrics.split(",")]
        # Handle special case: "plots" adds all plot metrics
        if "plots" in requested_metrics:
            requested_metrics.remove("plots")
            requested_metrics.extend([k for k in AVAILABLE_METRICS.keys() if k.startswith("plot_")])

    # Build metric_fns based on request (plot metrics only included if explicitly requested in metrics)
    metric_fns = {}
    for metric in requested_metrics:
        if metric in AVAILABLE_METRICS:
            metric_fns[metric] = AVAILABLE_METRICS[metric]
        else:
            if is_root:
                print(f"Warning: Unknown metric '{metric}' requested, ignoring.")

    trace_print(f"14: Selected metrics: {list(metric_fns.keys())}")
    
    # Parse comparison metrics from command line
    comparison_metrics = set(m.strip() for m in args.comparison_metrics.split(",") if m.strip())
    trace_print(f"14.5: Comparison metrics: {comparison_metrics}")
    
    # Precompile Euler error calculation if it's requested
    if "euler_error" in metric_fns and is_root:
        try:
            from helpers.euler_error import precompile_euler_error_cpu
        except ImportError:
            from .helpers.euler_error import precompile_euler_error_cpu
        print("Precompiling Euler error calculation functions...")
        if precompile_euler_error_cpu():
            print("  Euler error functions precompiled successfully")
        else:
            print("  Warning: Euler error precompilation failed, will compile on first use")
    
    # Parse loading requirements from command line
    periods_to_load = None
    stages_to_load = None
    
    if args.load_periods:
        try:
            periods_to_load = [int(p.strip()) for p in args.load_periods.split(",")]
            trace_print(f"14.7: Periods to load: {periods_to_load}")
        except ValueError:
            if is_root:
                print(f"Warning: Invalid --load-periods format, ignoring: {args.load_periods}")
    
    if args.load_stages:
        try:
            stages_to_load = json.loads(args.load_stages)
            # Convert string keys to int keys
            stages_to_load = {int(k): v for k, v in stages_to_load.items()}
            trace_print(f"14.8: Stages to load: {stages_to_load}")
        except (json.JSONDecodeError, ValueError):
            if is_root:
                print(f"Warning: Invalid --load-stages format, ignoring: {args.load_stages}")

    trace_print("15: Creating CircuitRunner")
    #  set-up main runner ------------------------------------------------------
    runner = CircuitRunner(
        base_cfg=cfg_container,
        param_paths=[
            "master.methods.upper_envelope",
            "master.settings.a_points",
            "master.settings.a_nxt_points",
            "master.settings.w_points",
            "master.parameters.delta_pb",
        ],
        model_factory=lambda cfg: make_housing_model(args,cfg, args.periods, vf_ngrid, comm),
        solver=make_housing_solver(args, args.mpi, comm),
        metric_fns=metric_fns,
        output_root=output_root,
        bundle_prefix=args.bundle_prefix,
        save_by_default=save_by_default,
        load_if_exists=True,
    )
    trace_print("16: CircuitRunner created")
    
    # Store loading requirements on the runner
    if periods_to_load is not None or stages_to_load is not None:
        runner.periods_to_load = periods_to_load
        runner.stages_to_load = stages_to_load
        if is_root:
            print(f"Selective loading enabled: periods={periods_to_load}, stages={stages_to_load}")
    
    # Check if we need baseline loading for comparison metrics
    # (comparison_metrics was already parsed from command line args above)
    needs_baseline = any(metric in metric_fns for metric in comparison_metrics)
    trace_print(f"17: Needs baseline loading: {needs_baseline}")
    
    # NOTE: this has to be consistent with whatever the metric L2 actually wants to do!
    # TODO: make this more flexible, so that we can do L2dev on any stage. AND HARdwire it
    runner.stages_to_load = [args.stages_L2dev]
    runner.stages_to_save = [args.stages_L2dev]
    trace_print(f"18: Stages configured: load={runner.stages_to_load}, save={runner.stages_to_save}")

    all_metrics = []
    all_param_vectors = []  # Collect all parameter vectors for design matrix

    # --------------------------------------------------------------------
    #  1) Solve, plot, and process baseline immediately (only if needed)
    # --------------------------------------------------------------------
    HD_POINTS = int(float(args.HD_points))
    STD_POINTS = int(float(args.grid_points))
    
    # Determine grid points based on baseline method type
    # Use HD_POINTS only for VFI_HDGRID methods, STD_POINTS for fast methods
    if BASE in ["VFI_HDGRID", "VFI_HDGRID_GPU"]:
        baseline_points = HD_POINTS
    else:
        # For fast methods used as baseline (FUES, CONSAV, DCEGM, etc)
        baseline_points = STD_POINTS
    
    # Use the BASE variable, which could be VFI_HDGRID or VFI_HDGRID_GPU
    if BASE in methods and needs_baseline:
        with MemoryMonitor(f"Baseline computation ({BASE})", log_start=True, log_end=True):
            trace_print("19: Starting baseline computation")
            runner.load_if_exists = not args.recompute_baseline
            if is_root:  # print from root only
                print(f"\n» Baseline ({BASE}):", "recompute" if args.recompute_baseline else "load/solve-if-missing")

            ref_params = np.array([BASE, baseline_points, baseline_points, baseline_points,pb_delta], dtype=object)
            runner.ref_params = ref_params
            all_param_vectors.append(ref_params)  # Add to design matrix
            
            # Temporarily remove comparison metrics when running baseline
            # (baseline can't compare against itself)
            original_metrics = runner.metric_fns
            baseline_metrics = {k: v for k, v in original_metrics.items() if k not in comparison_metrics}
            runner.metric_fns = baseline_metrics
            if is_root and baseline_metrics != original_metrics:
                print(f"  Using non-comparison metrics for baseline: {list(baseline_metrics.keys())}")
            
            trace_print("20: Running baseline solver")
            ref_metrics, ref_model = runner.run(
                ref_params,
                return_model=is_root,
                rank=solver_rank
            )
            trace_print("21: Baseline solver complete")
            
            # Restore original metrics for fast methods
            runner.metric_fns = original_metrics
            
            ref_metrics["master.methods.upper_envelope"] = BASE
            ref_metrics["param_hash"] = runner._hash_param_vec(ref_params)
            ref_metrics["reference_bundle_hash"] = runner._hash_param_vec(ref_params)
            ref_metrics["latest_time_id"] = args.RUN_ID 
            all_metrics.append(ref_metrics)

            # Store the baseline model for plotting comparisons (temporarily)
            if is_root and ref_model is not None:
                runner.ref_model_for_plotting = ref_model

            # Generate plots immediately and delete model
            if is_root and args.plots and ref_model is not None:
                trace_print("22: Generating baseline plots")
                img_dir = output_root / "images" 
                img_dir.mkdir(exist_ok=True)
                
                # Clean up non-essential data before plotting if in low-memory mode
                if args.low_memory:
                    cleanup_model(ref_model, aggressive=False)
                    log_memory_usage("after baseline model cleanup")
                
                try:
                    print(f"  Generating plots for {BASE}...")
                    generate_plots(ref_model, BASE, img_dir, egm_bounds=egm_bounds)
                except Exception as err:
                    print(f"[warn] plot-gen for {BASE} failed: {err}")
                finally:
                    del ref_model
                    gc.collect()
                    trace_print("23: Baseline plots complete, model deleted")
            
            # Trigger cleanup if memory usage is high
            cleanup_if_needed(memory_config["cleanup_threshold_gb"])
            log_memory_usage("after baseline computation")
    
    # If baseline wasn't computed but we still have fast methods, create ref_params for them
    elif needs_baseline and 'ref_params' not in locals():
        ref_params = np.array([BASE, baseline_points, baseline_points, baseline_points, pb_delta], dtype=object)
        if is_root:
            print(f"\n» Baseline ({BASE}) will be loaded from existing bundles for comparison metrics")
        trace_print("24: Baseline ref_params created for loading")

    # --------------------------------------------------------------------
    #  2) Solve test methods one by one, processing each immediately  
    # --------------------------------------------------------------------
    
    fast_methods_to_run = [m for m in FAST_METHODS if m in methods]
    
    if fast_methods_to_run:
        trace_print(f"25: Starting fast methods: {fast_methods_to_run}")
        runner.load_if_exists = not args.fresh_fast
        
        # Use the selected metrics for fast methods
        runner.metric_fns = metric_fns
        
        # Only set ref_params if baseline comparison metrics are needed
        if needs_baseline:
            # Ensure ref_params is defined, even if baseline wasn't run
            if 'ref_params' not in locals():
                ref_params = np.array([BASE, baseline_points, baseline_points, baseline_points, pb_delta], dtype=object)
            runner.ref_params = ref_params
        else:
            # No baseline needed - don't set ref_params to avoid loading
            if is_root:
                print(f"\n» Skipping baseline loading - only computing: {', '.join(metric_fns.keys())}")
            trace_print("26: Skipping baseline loading")

        if is_root:  # print from root only
            print(f"\n» Fast methods: {', '.join(fast_methods_to_run)}")
            
        for i, method in enumerate(fast_methods_to_run):
            with MemoryMonitor(f"Fast method computation ({method})", log_start=True, log_end=True):
                trace_print(f"27.{i+1}: Starting {method}")
                if is_root:
                    print(f"  Solving {method}...")
                
                params = np.array([method, STD_POINTS, STD_POINTS, STD_POINTS, pb_delta], dtype=object)
                all_param_vectors.append(params)  # Add to design matrix
                
                trace_print(f"28.{i+1}: Running {method} solver")
                metrics, model = runner.run(params, return_model=is_root, rank=solver_rank)
                trace_print(f"29.{i+1}: {method} solver complete")
                
                metrics["master.methods.upper_envelope"] = method
                metrics["param_hash"] = runner._hash_param_vec(params)
                if needs_baseline and 'ref_params' in locals():
                    metrics["reference_bundle_hash"] = runner._hash_param_vec(ref_params)
                else:
                    metrics["reference_bundle_hash"] = "no_baseline"
                metrics["latest_time_id"] = args.RUN_ID 
                all_metrics.append(metrics)
                
                # Generate plots immediately and delete model
                if is_root and args.plots and model is not None:
                    trace_print(f"30.{i+1}: Generating {method} plots")
                    img_dir = output_root / "images"
                    img_dir.mkdir(exist_ok=True)
                    
                    # Clean up non-essential data before plotting if in low-memory mode
                    if args.low_memory:
                        cleanup_model(model, aggressive=False)
                        log_memory_usage(f"after {method} model cleanup")
                    
                    try:
                        print(f"  Generating plots for {method}...")
                        
                        # Debug: Check if this model has EGM grids
                        first_period = model.get_period(0)
                        ownc_stage = first_period.get_stage("OWNC")
                        if hasattr(ownc_stage.dcsn.sol, 'EGM'):
                            unrefined_keys = list(ownc_stage.dcsn.sol.EGM.unrefined.keys()) if hasattr(ownc_stage.dcsn.sol.EGM, 'unrefined') else []
                            refined_keys = list(ownc_stage.dcsn.sol.EGM.refined.keys()) if hasattr(ownc_stage.dcsn.sol.EGM, 'refined') else []
                            print(f"[DEBUG] Model passed to plotting for {method}: "
                                  f"unrefined={len(unrefined_keys)}, refined={len(refined_keys)}")
                            print(f"[DEBUG] Model object id: {id(model)}, ownc_stage.dcsn.sol id: {id(ownc_stage.dcsn.sol)}")
                        else:
                            print(f"[DEBUG] Model passed to plotting for {method}: NO EGM grids!")
                            print(f"[DEBUG] Model object id: {id(model)}, ownc_stage.dcsn.sol id: {id(ownc_stage.dcsn.sol)}")
                        
                        generate_plots(model, method, img_dir, egm_bounds=egm_bounds, y_idx_list = (0,1,2))
                    except Exception as err:
                        print(f"[warn] plot-gen for {method} failed: {err}")
                    finally:
                        del model
                        gc.collect()
                        trace_print(f"31.{i+1}: {method} plots complete, model deleted")
                
                # Trigger cleanup after each method to prevent memory accumulation
                cleanup_if_needed(memory_config["cleanup_threshold_gb"])
                
                # Log memory usage periodically
                if (i + 1) % 2 == 0:  # Log every 2 methods
                    log_memory_usage(f"after {i+1} fast methods")

    if comm is not None:
        comm.Barrier()

    trace_print("32: Creating final summary")
    # --------------------------------------------------------------------
    #  3) Create final summary table and save design matrix
    # --------------------------------------------------------------------
    if is_root and all_metrics:
        res_df = pd.DataFrame(all_metrics)
        print_summary(res_df, output_root)
        generate_latex_table(res_df, output_root) # Added call to generate LaTeX table
        
        # Save complete design matrix with all parameter vectors
        if all_param_vectors:
            design_matrix = np.array(all_param_vectors, dtype=object)
            write_design_matrix_csv(runner, design_matrix)
            if is_root:
                print(f"  Design matrix saved to {output_root}/design_matrix.csv")

    # Clean up the stored baseline model and perform final memory cleanup
    if hasattr(runner, 'ref_model_for_plotting'):
        del runner.ref_model_for_plotting
    
    # Clear any cached references in the runner
    if hasattr(runner, '_cache') and runner._cache is not None:
        runner._cache.clear()
    if hasattr(runner, 'ref_params'):
        del runner.ref_params
    if hasattr(runner, 'ref_model'):
        del runner.ref_model
    
    # Clear the runner's metric functions which might hold closures
    runner.metric_fns = {}
    
    # Final aggressive cleanup
    gc.collect()
    
    # Memory usage summary
    if is_root:
        print("\n" + "="*60)
        print("MEMORY USAGE SUMMARY")
        print("="*60)
        # These functions are already imported at the top
        current_mem = get_memory_usage()
        available_mem = get_available_memory()
        # Peak memory (platform-specific)
        try:
            peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Convert to GB (platform-specific: Linux is KB, macOS is bytes)
            import platform
            if platform.system() == 'Darwin':  # macOS
                peak_mem = peak_mem / 1024 / 1024 / 1024
            else:  # Linux
                peak_mem = peak_mem / 1024 / 1024
            print(f"Peak memory usage: {peak_mem:.2f} GB")
        except:
            pass
        print(f"Final memory usage: {current_mem:.2f} GB")
        print(f"Available memory: {available_mem:.2f} GB")
        if args.low_memory:
            print("Low memory mode: ENABLED (Q and lambda arrays cleared)")
        print("="*60)
    
    log_memory_usage("at end of solve_runner")
    
    trace_print("33: Main function complete")

if __name__ == "__main__":
    main()