#!/usr/bin/env python
"""
Housing model with renting – baseline HD grid search/fast-methods runner.

This script provides workflows for solving the housing-renting model using
different solution methods and comparing them against a reference method given
user-supplied metrics.

The script supports both serial and parallel execution using the Message
Passing Interface (MPI).

Selective Loading and Caching
-----------------------------
**Reference Model Caching**: The reference model (baseline) uses a superset caching 
strategy that always loads periods [0, 1] with all required stages. This ensures 
all metrics share the same cached reference model, dramatically reducing memory usage.

**Command Line Arguments Note**: The `--load-periods` and `--load-stages` arguments 
are currently stored but not actively used. They were intended for selective loading 
of primary models but are superseded by:
- Automatic metric-based selective loading for reference models
- Superset caching that ignores these arguments for reference models
These arguments may be removed or reimplemented in future versions.

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
    from helpers.metrics import dev_c_L2, dev_v_L2, dev_c_log10_mean, plot_comparison_factory
    from helpers.plot_csv_export import csv_plot_comparison_factory, csv_generate_plots
    from helpers.memory_utils import MemoryMonitor, log_memory_usage, cleanup_if_needed, get_memory_config, get_memory_usage, get_available_memory
    from helpers.execution_settings import ExecutionSettings
except ImportError:
    from .whisperer import build_operators_for_circuit, run_time_iteration
    from .helpers.euler_error import euler_error_metric
    from .helpers.plots import generate_plots
    from .helpers.tables import print_summary, generate_latex_table
    from .helpers.metrics import dev_c_L2, dev_v_L2, dev_c_log10_mean, plot_comparison_factory
    from .helpers.plot_csv_export import csv_plot_comparison_factory, csv_generate_plots
    from .helpers.memory_utils import MemoryMonitor, log_memory_usage, cleanup_if_needed, get_memory_config, get_memory_usage, get_available_memory
    from .helpers.execution_settings import ExecutionSettings

trace_print("0.5: Local helpers imported")

CFG_DIR = Path(__file__).parent / "config_HR"

trace_print("0.6: Constants and globals set")


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
def make_housing_solver(args, use_mpi: bool, comm, baseline_method=None):
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
        
        # 1b. Warm up GPU kernels if using GPU solver (happens once before solving)
        # Use baseline_method passed from outer scope
        solver_method = baseline_method or ""
        if solver_method.endswith("_GPU") and (comm is None or comm.rank == 0):
            try:
                from src.dc_smm.models.housing_renting.horses_c_gpu import warmup_gpu_kernels
                warmup_gpu_kernels()
            except ImportError:
                pass  # GPU module not available
        
        # 2. backward time iteration
        # Free memory during solving if we're not saving the model
        free_memory = not getattr(args, 'save_full_model', False)
        # For Euler error, we only need periods 0 and 1
        periods_to_keep = [0, 1] if free_memory else None
        run_time_iteration(
            mc,
            n_periods=args.periods,
            verbose=args.verbose,
            verbose_timings=args.verbose,
            recorder=recorder,
            free_memory=free_memory,
            periods_to_keep=periods_to_keep,
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
                   help="Comma-separated list of fast methods (default: FUES,CONSAV)")
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
    p.add_argument("--csv-export", action="store_true",
                   help="Export plot data to CSV files instead of generating matplotlib plots (for cluster use)")
    p.add_argument("--save-full-model", action="store_true",
                   help="Keep all solution data in memory for saving. Disables memory freeing during solve. Default: False (free memory)")
    
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

    # Initialize execution settings
    settings = ExecutionSettings(args, CFG_DIR)
    trace_print("4: Execution settings initialized")
    
    # Print settings summary
    is_root = (comm is None) or (comm.rank == 0)
    settings.print_configuration(is_root)
    
    # For backward compatibility, expose key variables
    vf_ngrid = settings.vf_ngrid
    output_root = settings.output_root
    cfg_dir_bundle = settings.cfg_dir_bundle
    trace_print("5: Execution settings complete")
    
    #  set-up runner ------------------------------------------------------
    model_config = load_config(settings.cfg_dir_bundle)
    trace_print("6: Model config loaded")
    
    save_by_default = is_root
    solver_rank = comm.rank if comm is not None else 0
    trace_print(f"7: is_root={is_root}, solver_rank={solver_rank}")

    # --- Optional Pre-compilation Step ---
    if args.precompile and is_root:
        trace_print("8: Starting precompilation")
        print("\n--- Running Numba Pre-compilation ---")
        
        # --- Create a minimal config for the pre-compilation run ---
        precompile_cfg = copy.deepcopy(model_config)
        precompile_cfg['master']['settings']['a_points'] = 100
        precompile_cfg['master']['settings']['w_points'] = 100
        precompile_cfg['master']['settings']['a_nxt_points'] = 100
        
        # Use the determined baseline method for precompilation
        precompile_params = np.array([settings.baseline_method, 100, 100, 100, settings.pb_delta], dtype=object)

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
            solver=make_housing_solver(argparse.Namespace(verbose=False, periods=2), use_mpi=args.mpi, comm=comm, baseline_method=settings.baseline_method),
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
        
        trace_print("9: Precompilation complete")

    if comm is not None:
        comm.Barrier()

    trace_print("10: Setting up metric functions")
    #  set-up plotting configuration -----------------------------------------------
    plot_config = settings.get_plot_config()
    
    # Choose between CSV export or matplotlib plotting based on flag
    if args.csv_export:
        plot_factory = csv_plot_comparison_factory
        egm_plot_func = csv_generate_plots
        if is_root:
            print("\n*** CSV EXPORT MODE: Plot data will be saved as CSV files ***\n")
    else:
        plot_factory = plot_comparison_factory
        egm_plot_func = generate_plots
    
    # Available metrics mapping
    AVAILABLE_METRICS = {
        "euler_error": euler_error_metric,
        "dev_c_L2": dev_c_L2,
        "dev_c_log10_mean": dev_c_log10_mean,
        "plot_c_comparison": plot_factory(
            decision_variable='c',
            dim_labels=plot_config['asset_dims'],
            plot_axis_label='w_idx',
            slice_config=plot_config['plots_of_interest']
        ),
        "plot_v_comparison": plot_factory(
            decision_variable='vlu',
            dim_labels=plot_config['asset_dims'],
            plot_axis_label='w_idx',
            slice_config=plot_config['plots_of_interest'],
            sol_attr='value'
        ),
    }

    # Build metric_fns based on requested metrics from config
    metric_fns = {}
    for metric in settings.requested_metrics:
        if metric in AVAILABLE_METRICS:
            metric_fns[metric] = AVAILABLE_METRICS[metric]
        else:
            if is_root:
                print(f"Warning: Unknown metric '{metric}' requested, ignoring.")

    trace_print(f"11: Selected metrics: {list(metric_fns.keys())}")
    
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
    
    trace_print("12: Creating CircuitRunner")
    #  set-up main runner ------------------------------------------------------
    runner = CircuitRunner(
        base_cfg=model_config,
        param_paths=[
            "master.methods.upper_envelope",
            "master.settings.a_points",
            "master.settings.a_nxt_points",
            "master.settings.w_points",
            "master.parameters.delta_pb",
        ],
        model_factory=lambda cfg: make_housing_model(args,cfg, args.periods, vf_ngrid, comm),
        solver=make_housing_solver(args, args.mpi, comm, baseline_method=settings.baseline_method),
        metric_fns=metric_fns,
        output_root=output_root,
        bundle_prefix=args.bundle_prefix,
        save_by_default=save_by_default,
        load_if_exists=True,
    )
    trace_print("13: CircuitRunner created")
    
    # Store loading requirements on the runner from config
    if settings.periods_to_load is not None or settings.stages_to_load is not None:
        runner.periods_to_load = settings.periods_to_load
        runner.stages_to_load = settings.stages_to_load
        if is_root:
            print(f"Selective loading enabled: periods={settings.periods_to_load}, stages={settings.stages_to_load}")
    
    trace_print(f"14: Needs baseline loading: {settings.needs_baseline}")
    
    # NOTE: this has to be consistent with whatever the metric L2 actually wants to do!
    # TODO: make this more flexible, so that we can do L2dev on any stage. AND HARdwire it
    runner.stages_to_load = [args.stages_L2dev]
    runner.stages_to_save = [args.stages_L2dev]
    trace_print(f"15: Stages configured: load={runner.stages_to_load}, save={runner.stages_to_save}")

    all_metrics = []
    all_param_vectors = []  # Collect all parameter vectors for design matrix

    # --------------------------------------------------------------------
    #  1) Solve, plot, and process baseline immediately (only if needed)
    # --------------------------------------------------------------------
    if settings.should_run_baseline:
        with MemoryMonitor(f"Baseline computation ({settings.baseline_method})", log_start=True, log_end=True):
            trace_print("16: Starting baseline computation")
            runner.load_if_exists = not args.recompute_baseline
            if is_root:  # print from root only
                print(f"\n» Baseline ({settings.baseline_method}):", "recompute" if args.recompute_baseline else "load/solve-if-missing")

            ref_params = settings.get_baseline_params()
            runner.ref_params = ref_params
            all_param_vectors.append(ref_params)  # Add to design matrix
            
            # Temporarily remove comparison metrics when running baseline
            # (baseline can't compare against itself)
            original_metrics = runner.metric_fns
            baseline_metrics = settings.get_baseline_metrics_filter(original_metrics)
            runner.metric_fns = baseline_metrics
            if is_root and baseline_metrics != original_metrics:
                print(f"  Using non-comparison metrics for baseline: {list(baseline_metrics.keys())}")
            
            # Print parameter hash and bundle path for baseline
            baseline_hash = runner._hash_param_vec(ref_params)
            baseline_bundle_path = runner._bundle_path(ref_params)
            if is_root:
                print(f"  Parameter hash: {baseline_hash}")
                if baseline_bundle_path:
                    print(f"  Bundle path: {baseline_bundle_path}")
            
            trace_print("17: Running baseline solver")
            ref_metrics, ref_model = runner.run(
                ref_params,
                return_model=is_root,
                rank=solver_rank
            )
            trace_print("18: Baseline solver complete")
            
            # Register baseline model in unified cache for reuse
            if is_root and ref_model is not None:
                try:
                    from dynx.runner.model_cache import register_baseline_model
                    # Register with periods [0, 1] since baseline loads full model
                    register_baseline_model(settings.baseline_method, ref_model, periods=[0, 1])
                    print(f"  Registered {settings.baseline_method} in unified cache for metric reuse")
                except ImportError:
                    trace_print("18.1: Unified model cache not available")
            
            # Restore original metrics for fast methods
            runner.metric_fns = original_metrics
            
            ref_metrics["master.methods.upper_envelope"] = settings.baseline_method
            ref_metrics["param_hash"] = runner._hash_param_vec(ref_params)
            ref_metrics["reference_bundle_hash"] = runner._hash_param_vec(ref_params)
            ref_metrics["latest_time_id"] = args.RUN_ID 
            all_metrics.append(ref_metrics)

            # Store the baseline model for plotting comparisons (temporarily)
            if is_root and ref_model is not None:
                runner.ref_model_for_plotting = ref_model

            # Generate plots immediately and delete model
            if is_root and args.plots and ref_model is not None:
                trace_print("19: Generating baseline plots")
                
                # Use bundle-specific directory for plots
                if baseline_bundle_path:
                    from pathlib import Path
                    baseline_plot_dir = Path(baseline_bundle_path) / "plots"
                    baseline_plot_dir.mkdir(parents=True, exist_ok=True)
                else:
                    # Fallback to original location if no bundle path
                    baseline_plot_dir = settings.img_dir
                    baseline_plot_dir.mkdir(exist_ok=True)
                
                # Clean up non-essential data before plotting if in low-memory mode
                if args.low_memory:
                    cleanup_model(ref_model, aggressive=False)
                    log_memory_usage("after baseline model cleanup")
                
                try:
                    print(f"  Generating plots for {settings.baseline_method}...")
                    if baseline_bundle_path:
                        print(f"  Plots will be saved to: {baseline_plot_dir}")
                    egm_plot_func(ref_model, settings.baseline_method, baseline_plot_dir, egm_bounds=plot_config['egm_bounds'])
                except Exception as err:
                    print(f"[warn] plot-gen for {settings.baseline_method} failed: {err}")
                finally:
                    del ref_model
                    gc.collect()
                    trace_print("20: Baseline plots complete, model deleted")
            
            # Trigger cleanup if memory usage is high
            mem_config = settings.get_memory_config()
            cleanup_if_needed(memory_config.get("cleanup_threshold_gb", 32))
            log_memory_usage("after baseline computation")
    
    # If baseline wasn't computed but we still have fast methods, create ref_params for them
    elif settings.needs_baseline and 'ref_params' not in locals():
        ref_params = settings.get_baseline_params()
        if is_root:
            print(f"\n» Baseline ({settings.baseline_method}) will be loaded from existing bundles for comparison metrics")
        trace_print("21: Baseline ref_params created for loading")

    # --------------------------------------------------------------------
    #  2) Solve test methods one by one, processing each immediately  
    # --------------------------------------------------------------------
    
    if settings.fast_methods_to_run:
        trace_print(f"22: Starting fast methods: {settings.fast_methods_to_run}")
        runner.load_if_exists = not args.fresh_fast
        
        # Use the selected metrics for fast methods
        runner.metric_fns = metric_fns
        
        # Only set ref_params if baseline comparison metrics are needed
        if settings.needs_baseline:
            # Ensure ref_params is defined, even if baseline wasn't run
            if 'ref_params' not in locals():
                ref_params = settings.get_baseline_params()
            runner.ref_params = ref_params
        else:
            # No baseline needed - don't set ref_params to avoid loading
            if is_root:
                print(f"\n» Skipping baseline loading - only computing: {', '.join(metric_fns.keys())}")
            trace_print("23: Skipping baseline loading")

        if is_root:  # print from root only
            print(f"\n» Fast methods: {', '.join(settings.fast_methods_to_run)}")
            
        for i, method in enumerate(settings.fast_methods_to_run):
            with MemoryMonitor(f"Fast method computation ({method})", log_start=True, log_end=True):
                trace_print(f"24.{i+1}: Starting {method}")
                if is_root:
                    print(f"  Solving {method}...")
                
                params = settings.get_method_params(method)
                all_param_vectors.append(params)  # Add to design matrix
                
                # Print parameter hash and bundle path for this method
                method_hash = runner._hash_param_vec(params)
                method_bundle_path = runner._bundle_path(params)
                if is_root:
                    print(f"  Parameter hash: {method_hash}")
                    if method_bundle_path:
                        print(f"  Bundle path: {method_bundle_path}")
                
                trace_print(f"25.{i+1}: Running {method} solver")
                metrics, model = runner.run(params, return_model=is_root, rank=solver_rank)
                trace_print(f"26.{i+1}: {method} solver complete")
                
                metrics["master.methods.upper_envelope"] = method
                metrics["param_hash"] = runner._hash_param_vec(params)
                if settings.needs_baseline and 'ref_params' in locals():
                    metrics["reference_bundle_hash"] = runner._hash_param_vec(ref_params)
                else:
                    metrics["reference_bundle_hash"] = "no_baseline"
                metrics["latest_time_id"] = args.RUN_ID 
                all_metrics.append(metrics)
                
                # Generate plots immediately and delete model
                if is_root and args.plots and model is not None:
                    trace_print(f"27.{i+1}: Generating {method} plots")
                    
                    # Use bundle-specific directory for plots
                    if method_bundle_path:
                        from pathlib import Path
                        bundle_plot_dir = Path(method_bundle_path) / "plots"
                        bundle_plot_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        # Fallback to original location if no bundle path
                        bundle_plot_dir = settings.img_dir
                        bundle_plot_dir.mkdir(exist_ok=True)
                    
                    # Clean up non-essential data before plotting if in low-memory mode
                    if args.low_memory:
                        cleanup_model(model, aggressive=False)
                        log_memory_usage(f"after {method} model cleanup")
                    
                    try:
                        print(f"  Generating plots for {method}...")
                        if method_bundle_path:
                            print(f"  Plots will be saved to: {bundle_plot_dir}")
                        
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
                        
                        egm_plot_func(model, method, bundle_plot_dir, egm_bounds=plot_config['egm_bounds'], y_idx_list=plot_config['y_idx_list'])
                    except Exception as err:
                        print(f"[warn] plot-gen for {method} failed: {err}")
                    finally:
                        del model
                        gc.collect()
                        trace_print(f"28.{i+1}: {method} plots complete, model deleted")
                
                # Trigger cleanup after each method to prevent memory accumulation
                cleanup_if_needed(memory_config.get("cleanup_threshold_gb", 32))
                
                # Log memory usage periodically
                if (i + 1) % 2 == 0:  # Log every 2 methods
                    log_memory_usage(f"after {i+1} fast methods")

    if comm is not None:
        comm.Barrier()

    trace_print("29: Creating final summary")
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
                print(f"  Design matrix saved to {settings.output_root}/design_matrix.csv")

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
    
    # Clear reference model cache
    try:
        from dynx.runner.reference_cache import clear_reference_cache
        clear_reference_cache()
        if is_root:
            print("  Reference model cache cleared")
    except ImportError:
        pass
    
    # Clear unified model cache
    try:
        from dynx.runner.model_cache import clear_model_cache
        clear_model_cache()
        if is_root:
            print("  Unified model cache cleared")
    except ImportError:
        pass
    
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
    
    trace_print("30: Main function complete")

if __name__ == "__main__":
    main()