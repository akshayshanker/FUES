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
**Reference Model Caching**: When metrics compare methods (e.g, FUES, CONSAV, etc.) 
against the baseline (e.g, VFI_HDGRID), the baseline model is loaded once with the union 
of all data required by any metric (all periods, all stages). This single load 
is then shared across all metrics, avoiding redundant disk reads. If the baseline 
was already solved in the same run, its model is reused directly from cache.

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
            --config-id HR \\
            --vfi-ngrid 10000 \\
            --HD-points 10000 \\
            --grid-points 4000 \\
            --recompute-baseline \\
            --fresh-fast \\
            --plots

2.  **Fast Methods Only (with existing baseline)**
    This runs only the "fast" solvers (e.g., `FUES`, `CONSAV`), relying
    on a pre-computed baseline solution for comparison metrics.

    .. code-block:: bash

        python3 -m examples.housing_renting.solve_runner \\
            --ue-method FUES,CONSAV \\
            --periods 3 \\
            --output-root solutions/HR \\
            --vfi-ngrid 100 \\
            --HD-points 10000 \\
            --baseline-method VFI_HDGRID \\
            --fresh-fast \\
            --plots

    .. note::
       If comparison metrics are requested (e.g., ``dev_c_L2``), the baseline 
       is loaded from disk. Use ``--metrics euler_error`` to skip baseline loading entirely.

**Parallel Execution (MPI)**

For large-scale problems, MPI is recommended to accelerate the baseline
computation.

.. code-block:: bash

    mpiexec -np 45 \\
        python3 -m examples.housing_renting.solve_runner \\
          --periods 3 \\
          --ue-method ALL \\
          --output-root /scratch/tp66/$USER/FUES/solutions/HR_test_v4 \\
          --config-id HR_test_v4 \\
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
        --config-id HR_gpu \\
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

If not specified, defaults to: `FUES,CONSAV,DCEGM`

Metrics Configuration
--------------------
Use ``--metrics`` to specify which metrics to compute:

.. code-block:: bash

    --metrics "euler_error"           # Euler error only (no baseline needed)
    --metrics "euler_error,dev_c_L2"  # Both (baseline loaded for dev_c_L2)
    --metrics "all"                   # All metrics (default)

**Comparison metrics** (``dev_c_L2``, ``plot_c_comparison``, ``plot_v_comparison``) require 
loading the baseline model from disk. Non-comparison metrics like ``euler_error`` do not.

When running only the baseline method, comparison metrics are automatically skipped since 
comparing a model against itself is meaningless.

To customize which metrics require baseline comparison, use:
``--comparison-metrics "dev_c_L2,my_custom_comparison"``

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
- ``--metrics``: Comma-separated list of metrics to compute (default: all). Options: euler_error, dev_c_L2, plot_c_comparison, plot_v_comparison
- ``--comparison-metrics``: Comma-separated list of metrics requiring baseline comparison (default: dev_c_L2,plot_c_comparison,plot_v_comparison)

Directory Organization
---------------------
The runner uses a standardized directory structure for outputs:

::

    {output-root}/
    ├── bundles/
    │   └── {hash}/                    # 8-char MD5 hash of grid params
    │       ├── VFI_HDGRID/            # Baseline method bundle
    │       │   └── images_{timestamp}/
    │       └── FUES/                  # Fast method bundle
    │           └── images_{timestamp}/
    ├── raw_metrics.csv                # All metrics (appended across runs)
    ├── performance_summary.txt
    └── comparison_table.tex

Experiment grouping is done via ``--output-root``. For example, PBS scripts 
set ``OUTPUT_DIR="/scratch/.../housing_renting/${VERSION_TAG}_${TRIAL_ID}"``
to group related experiments together.

Note: ``--config-id`` is used for config directory naming only, not bundle paths.

"""

from __future__ import annotations

# Global trace flag - check for --trace in command line args early
import sys
_TRACE_ENABLED = '--trace' in sys.argv
_MPI_RANK = 0  # Will be set once MPI comm is available

def set_mpi_rank(rank: int):
    """Set the MPI rank for trace filtering (only rank 0 prints)"""
    global _MPI_RANK
    _MPI_RANK = rank

# Trace verbosity: 0=off, 1=major milestones only, 2=all details
_TRACE_VERBOSITY = 1

def trace_print(message, force_all_ranks=False, detail=False):
    """Print trace message only if tracing is enabled and rank is 0.
    
    Args:
        message: Message to print
        force_all_ranks: Print on all MPI ranks (default: only rank 0)
        detail: If True, only print when verbosity >= 2 (default: False)
    """
    if not _TRACE_ENABLED:
        return
    if detail and _TRACE_VERBOSITY < 2:
        return
    if force_all_ranks or _MPI_RANK == 0:
        print(f"[TRACE] {message}", flush=True)

# Note: Early trace prints (before MPI init) go to rank 0 since _MPI_RANK defaults to 0
# These are detail traces - only shown at verbosity 2
import argparse
try:
    from helpers.cli import create_argument_parser
except ImportError:
    from .helpers.cli import create_argument_parser
import copy
import sys
import time
from datetime import datetime
from pathlib import Path
import gc
import resource

trace_print("0.2: Basic imports done", detail=True)
import numpy as np
import pandas as pd

trace_print("0.3: Numpy/Pandas imported", detail=True)
from dynx.runner import CircuitRunner, write_design_matrix_csv
from dynx.stagecraft.io import load_config
from dynx.stagecraft.makemod import initialize_model_Circuit, compile_all_stages

trace_print("0.4: DynX imports done", detail=True)

# ────────────────────────────────────────────────────────────────────────────
#  local helpers (imported lazily to keep fall-back stubs tiny)
# ────────────────────────────────────────────────────────────────────────────
try:
    from whisperer import build_operators_for_circuit, run_time_iteration, run_recursive_iteration
    from helpers.euler_error import euler_error_metric
    from helpers.plots import generate_plots, plot_compare_consumption_policy
    from helpers.tables import print_summary, generate_latex_table
    from helpers.metrics import dev_c_L2, dev_v_L2, dev_c_log10_mean, plot_comparison_factory
    from helpers.plot_csv_export import csv_plot_comparison_factory, csv_generate_plots
    from helpers.memory_utils import MemoryMonitor, log_memory_usage, cleanup_if_needed, get_memory_config, get_memory_usage, get_available_memory
    from helpers.execution_settings import ExecutionSettings
except ImportError:
    from .whisperer import build_operators_for_circuit, run_time_iteration, run_recursive_iteration
    from .helpers.euler_error import euler_error_metric
    from .helpers.plots import generate_plots, plot_compare_consumption_policy
    from .helpers.tables import print_summary, generate_latex_table
    from .helpers.metrics import dev_c_L2, dev_v_L2, dev_c_log10_mean, plot_comparison_factory
    from .helpers.plot_csv_export import csv_plot_comparison_factory, csv_generate_plots
    from .helpers.memory_utils import MemoryMonitor, log_memory_usage, cleanup_if_needed, get_memory_config, get_memory_usage, get_available_memory
    from .helpers.execution_settings import ExecutionSettings

trace_print("0.5: Local helpers imported", detail=True)

CFG_DIR = Path(__file__).parent / "config_HR"

trace_print("0.6: Constants and globals set", detail=True)


def cleanup_model(model, aggressive=False, preserve_period_0=True):
    """Clean up model data to free memory.

    Parameters
    ----------
    model : ModelCircuit
        The model to clean up
    aggressive : bool
        If True, clear all solution data. If False, keep only essential data.
    preserve_period_0 : bool
        If True, preserve period 0 data for plotting (Q, lambda, EGM grids).
        Default: True.
    """
    if model is None:
        return

    try:
        # Clear large arrays from all periods and stages
        for period_idx in range(len(model.periods_list)):
            # Skip period 0 if preserving for plots
            if preserve_period_0 and period_idx == 0:
                continue
                
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

    # Apply use_taxes from CLI (must happen after patch_cfg deep copy)
    use_taxes_flag = getattr(args, 'use_taxes', False)
    if use_taxes_flag:
        cfg["master"]["parameters"]["use_taxes"] = True
        print(f"[TAX DEBUG] make_housing_model: use_taxes set to True in cfg")
        print(f"[TAX DEBUG] tax_table in cfg: {'yes' if cfg['master']['parameters'].get('tax_table') else 'no'}")

    # 2. build ModelCircuit skeleton (no heavy maths yet)
    mc = initialize_model_Circuit(
        master_config=cfg["master"],
        stage_configs=cfg["stages"],
        connections_config=cfg["connections"],
    )

    # Store tax config for injection after compilation
    _tax_table = cfg["master"]["parameters"].get("tax_table")
    _use_taxes = use_taxes_flag or cfg["master"]["parameters"].get("use_taxes", False)
    if _use_taxes:
        print(f"[TAX DEBUG] Injecting tax params onto model stages (before compile)...")
    # Optional: log memory right after circuit skeleton creation
    if getattr(args, "trace", False):
        try:
            log_memory_usage("after initialize_model_Circuit (skeleton)", verbose=True)
        except Exception:
            pass

    # 3. numerically compile every Stage (grid creation, etc.)
    # Skip if lazy_compile is enabled - compilation will happen per-period in run_time_iteration
    lazy_compile = getattr(args, 'low_memory', False)
    
    if lazy_compile:
        print("[INFO] Lazy compilation enabled - skipping upfront compile_all_stages")
        # Still need to set num_rep on stages for lazy compilation to work
        from dynx.heptapodx.core.api import generate_numerical_model
        for period_idx in range(len(mc.periods_list)):
            period = mc.get_period(period_idx)
            for stage_name, stage in period.stages.items():
                if stage.num_rep is None:
                    stage.num_rep = generate_numerical_model
                # IMPORTANT:
                # DynX may allocate per-period grids during circuit initialization.
                # In low-memory + lazy-compile mode we want the *period* to remain a light
                # skeleton until `_compile_period_stages()` runs, otherwise memory can
                # still scale with `T` even when we "compile lazily".
                try:
                    for perch_name in ("arvl", "dcsn", "cntn"):
                        perch = getattr(stage, perch_name, None)
                        if perch is not None:
                            if hasattr(perch, "grid"):
                                perch.grid = None
                    # Do NOT clear `stage.model.num` here.
                    # DynX stage compilation can expect parts of the model scaffold
                    # to exist when `build_computational_model()` runs; clearing it at
                    # factory time can trigger KeyErrors like "'functions' not found".
                    stage.status_flags["compiled"] = False
                except Exception:
                    pass
        if getattr(args, "trace", False):
            try:
                log_memory_usage("after lazy-init grid stripping", verbose=True)
            except Exception:
                pass
    else:
        try:
            # Always compile when comm=None (sweep mode), otherwise only rank 0
            # Use force=True to ensure grids are rebuilt with current config
            if comm is None:
                compile_all_stages(mc, force=True)  # Sweep mode: each rank compiles fresh
            elif comm.rank == 0:
                compile_all_stages(mc, force=False)  # Non-sweep: rank 0 compiles, skip if cached
        except Exception as exc:
            # logger.error("Stage compilation failed – aborting make_housing_model()", exc_info=True)
            raise
    
    # WORKAROUND: Dynx model.param may be reconstructed on each access, losing injected attrs.
    # Instead, inject use_taxes and tax_table into settings_dict (a regular dict that persists).
    # Also inject into mover.model.settings_dict since operator factories use mover.model.
    if _use_taxes:
        print(f"[TAX DEBUG] Injecting tax params into settings_dict (after compile)...")
        # Track unique model objects to avoid double-counting
        injected_models = set()
        for period_idx in range(len(mc.periods_list)):
            period = mc.get_period(period_idx)
            for stage_name, stage in period.stages.items():
                # Collect all model objects: stage.model and mover.model for all movers
                models_to_inject = []
                if hasattr(stage, 'model'):
                    models_to_inject.append((f"{stage_name}.model", stage.model))
                for mover_name in ['cntn_to_dcsn', 'dcsn_to_arvl', 'arvl_to_cntn']:
                    mover = getattr(stage, mover_name, None)
                    if mover is not None and hasattr(mover, 'model'):
                        models_to_inject.append((f"{stage_name}.{mover_name}.model", mover.model))
                
                # Inject into each unique model's settings_dict
                for model_path, model in models_to_inject:
                    model_id = id(model)
                    if model_id not in injected_models:
                        if hasattr(model, 'settings_dict'):
                            model.settings_dict['use_taxes'] = True
                            model.settings_dict['tax_table'] = _tax_table
                            injected_models.add(model_id)
                        else:
                            # Renter stages (RNTH, RNTC) don't need taxes - skip silently
                            if 'RNT' not in model_path:
                                print(f"[TAX DEBUG] {model_path} has no settings_dict!")
        print(f"[TAX DEBUG] Tax params injected into {len(injected_models)} unique model objects")
    
    # DEBUG: Print model info to trace period mismatch
    print(f"[DEBUG] make_housing_model: periods param={periods}, model has {len(mc.periods_list)} periods, cfg horizon={cfg['master'].get('horizon', 'NOT SET')}")

    # 4. Add runtime flags to model settings for operators to use
    # This allows solver operators to skip expensive operations when not needed
    if hasattr(args, 'skip_egm_plots'):
        # Inject skip_egm_plots flag into each period's model settings
        for period_idx in range(len(mc.periods_list)):
            period = mc.get_period(period_idx)
            for stage_name in period.stages.keys():
                stage = period.get_stage(stage_name)
                if hasattr(stage, 'model') and hasattr(stage.model, 'settings_dict'):
                    stage.model.settings_dict['skip_egm_plots'] = args.skip_egm_plots

    return mc


def make_sweep_model(cfg, args, sweep_periods: int, sweep_vfi_ngrid: int):
    """
    Model factory for sweep mode that syncs H_points -> S_points and applies grid multiplier.
    
    This is used by CircuitRunner in parameter sweep mode. Each configuration
    gets its own model with appropriate grid settings based on the method type.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary with 'master', 'stages', 'connections' keys.
    args : argparse.Namespace
        Command-line arguments.
    sweep_periods : int
        Number of periods to solve (from sweep config or args).
    sweep_vfi_ngrid : int
        VFI grid size (from sweep config or args).
    
    Returns
    -------
    ModelCircuit
        A fully initialized and compiled model circuit.
    """
    # DEBUG: Print horizon values to trace period mismatch
    print(f"[DEBUG] make_sweep_model: sweep_periods={sweep_periods}, cfg horizon={cfg['master'].get('horizon', 'NOT SET')}")
    
    # Get grid multiplier from settings (default to 1 if not set)
    a_grid_mult = cfg["master"]["settings"].get("a_grid_multiplier", 1)
    
    # Get the method from config to check if it's VFI-based
    method = cfg["master"]["methods"].get("upper_envelope", "")
    
    # Apply multiplier to a_points and a_nxt_points (w_points stays at base)
    # BUT skip for VFI methods - they need all grids to match exactly
    a_points = cfg["master"]["settings"]["a_points"]
    H_points = cfg["master"]["settings"]["H_points"]
    
    if method in ["VFI_HDGRID", "VFI_HDGRID_GPU"]:
        # VFI methods: all grid sizes must be identical
        cfg["master"]["settings"]["a_points"] = a_points
        cfg["master"]["settings"]["a_nxt_points"] = a_points
        cfg["master"]["settings"]["w_points"] = a_points
    else:
        # EGM methods: apply multiplier to a_points/a_nxt_points, keep w_points at base
        cfg["master"]["settings"]["a_points"] = a_points * a_grid_mult
        cfg["master"]["settings"]["a_nxt_points"] = a_points * a_grid_mult
        cfg["master"]["settings"]["w_points"] = a_points
    
    # Sync H_points -> S_points (both are in settings, not parameters)
    cfg["master"]["settings"]["S_points"] = H_points
    
    # NOTE: Pass comm=None so each MPI rank compiles its own model independently.
    # In sweep mode with mpi_map, each rank works on different configs in parallel.
    # Use sweep_periods and sweep_vfi_ngrid from fixed params (not args)
    model = make_housing_model(args, cfg, sweep_periods, sweep_vfi_ngrid, comm=None)
    
    # Store original params on model for correct bundle path computation in metrics
    model._sweep_params = {
        'method': method,
        'a_points': a_points,
        'H_points': H_points,
    }
    return model


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
        # Detect method from model config (works for sweep mode where baseline_method=None)
        # The method is set in the config by patch_cfg() during model creation
        if baseline_method:
            solver_method = baseline_method
        else:
            # Get method from stage model (reliable across all modes)
            try:
                stage = mc.get_period(0).get_stage("OWNC")
                solver_method = stage.model.methods.get("upper_envelope", "")
            except (AttributeError, KeyError):
                solver_method = ""
        
        if solver_method.endswith("_GPU") and (comm is None or comm.rank == 0):
            try:
                from src.dc_smm.models.housing_renting.horses_c_gpu import warmup_gpu_kernels
                warmup_gpu_kernels()
            except ImportError:
                pass  # GPU module not available
        
        # 2. backward time iteration
        # Check if recursive iteration is enabled (infinite horizon mode)
        print("[DEBUG] Entering recursive iteration detection block")  # UNCONDITIONAL DEBUG
        recursive_mode = False
        convergence_tol = 1e-6
        max_iterations = 500

        try:
            # Get settings from stage model (same pattern as whisperer.py _get_model_setting)
            stage = mc.get_period(0).get_stage("TENU")
            model = stage.model

            # Try settings_dict first (like _get_model_setting does)
            settings_dict = getattr(model, 'settings_dict', None)
            settings = getattr(model, 'settings', None)

            print(f"[DEBUG] settings_dict type: {type(settings_dict)}, keys: {list(settings_dict.keys()) if isinstance(settings_dict, dict) else 'N/A'}")
            print(f"[DEBUG] settings type: {type(settings)}, keys: {list(settings.keys()) if isinstance(settings, dict) else 'N/A'}")

            # Extract from settings_dict first (higher priority), then settings
            if isinstance(settings_dict, dict) and 'recursive_iteration' in settings_dict:
                recursive_mode = settings_dict.get("recursive_iteration", False)
                convergence_tol = settings_dict.get("convergence_tol", 1e-6)
                max_iterations = settings_dict.get("max_iterations", 500)
                print(f"[DEBUG] Got values from settings_dict: recursive_mode={recursive_mode}")
            elif isinstance(settings, dict) and 'recursive_iteration' in settings:
                recursive_mode = settings.get("recursive_iteration", False)
                convergence_tol = settings.get("convergence_tol", 1e-6)
                max_iterations = settings.get("max_iterations", 500)
                print(f"[DEBUG] Got values from settings: recursive_mode={recursive_mode}")
            elif hasattr(settings, 'get'):
                recursive_mode = settings.get("recursive_iteration", False)
                convergence_tol = settings.get("convergence_tol", 1e-6)
                max_iterations = settings.get("max_iterations", 500)
                print(f"[DEBUG] Got values from settings.get(): recursive_mode={recursive_mode}")

            # Ensure correct types (YAML can return lists for unresolved references)
            if isinstance(recursive_mode, list):
                recursive_mode = False
            if isinstance(convergence_tol, list):
                convergence_tol = 1e-6
            if isinstance(max_iterations, list):
                max_iterations = 500

            # Convert to proper types
            if recursive_mode is not False:
                recursive_mode = bool(recursive_mode)
            max_iterations = int(max_iterations)
            convergence_tol = float(convergence_tol)

        except (AttributeError, KeyError) as e:
            if args.verbose:
                print(f"[DEBUG] Could not get recursive iteration settings: {e}")
            recursive_mode = False
            convergence_tol = 1e-6
            max_iterations = 500

        # Always print for debugging (remove args.verbose check temporarily)
        if comm is None or comm.rank == 0:
            print(f"[DEBUG] FINAL: recursive_mode={recursive_mode}, max_iterations={max_iterations}, convergence_tol={convergence_tol}")

        if recursive_mode:
            # Infinite horizon: recursive iteration on single period
            if args.verbose and (comm is None or comm.rank == 0):
                print("Using recursive iteration (infinite horizon mode)")
            run_recursive_iteration(
                mc,
                max_iterations=max_iterations,
                convergence_tol=convergence_tol,
                verbose=args.verbose,
                recorder=recorder,
            )
        else:
            # Finite horizon: standard backward time iteration
            # Free memory during solving if we're not saving the model
            free_memory = not getattr(args, 'save_full_model', False)
            # Keep periods 0 and 1: period 0 for plots, both for Euler error calculation
            periods_to_keep = [0, 1] if free_memory else None
            # Enable lazy compilation in low-memory mode to reduce peak memory usage
            # This compiles each period's grids just before solving, then clears them
            lazy_compile = getattr(args, 'low_memory', False)

            run_time_iteration(
                mc,
                n_periods=args.periods,
                verbose=args.verbose,
                verbose_timings=args.verbose,
                recorder=recorder,
                free_memory=free_memory,
                periods_to_keep=periods_to_keep,
                lazy_compile=lazy_compile,
            )
        return mc

    return _solve


def cleanup_model_complete(model):
    """Aggressively free ALL memory used by a model.
    
    Call this when an MPI process is done with a model and won't need it again.
    This goes beyond cleanup_model() by deleting the entire model structure.
    """
    if model is None:
        return
    
    import gc
    
    try:
        # Clear all periods and stages completely
        for period_idx in range(len(model.periods_list)):
            period = model.get_period(period_idx)
            for stage_name, stage in list(period.stages.items()):
                # Clear all perch solutions
                for perch_name in ["arvl", "dcsn", "cntn"]:
                    perch = getattr(stage, perch_name, None)
                    if perch is not None:
                        if hasattr(perch, 'sol'):
                            perch.sol = None
                        if hasattr(perch, 'model'):
                            perch.model = None
                
                # Clear stage model and numerical data
                if hasattr(stage, 'model'):
                    if hasattr(stage.model, 'num'):
                        stage.model.num = None
                    stage.model = None
                
                # Clear operators
                if hasattr(stage, '_ops'):
                    stage._ops = None
        
        # Clear periods list
        if hasattr(model, 'periods_list'):
            model.periods_list.clear()
        
    except Exception as e:
        print(f"[cleanup_model_complete] Warning: {e}")
    
    # Force garbage collection
    gc.collect()


# ────────────────────────────────────────────────────────────────────────────
#  CLI + main
# ────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    """
    Command-line interface for solving and benchmarking the Housing–Renting model.
    """
    global _TRACE_ENABLED
    
    trace_print("1: Starting main()")
    
    p = create_argument_parser()
    args = p.parse_args(argv or sys.argv[1:])

    # Generate timestamp suffix for image directories (shared across all plots in this run)
    # Always generate timestamp to preserve old image files
    timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.plots or args.csv_export:
        print(f"Images will be saved with timestamp suffix: {timestamp_suffix} (preserving previous runs)")

    # Enable tracing based on command line flag
    _TRACE_ENABLED = args.trace
    
    # Configure memory management based on environment
    memory_config = get_memory_config("cluster")  # Default to cluster settings
    
    trace_print("2: Args parsed", detail=True)

    #  MPI communicator
    from dc_smm.helpers.mpi_utils import get_comm
    comm = get_comm(args.mpi)
    mpi_rank = comm.rank if comm else 0
    set_mpi_rank(mpi_rank)  # Set rank for trace_print filtering
    trace_print(f"3: MPI comm created, rank={mpi_rank}", detail=True)
    
    # Log initial memory usage (rank 0 only)
    if mpi_rank == 0:
        log_memory_usage("at start of solve_runner")
    
    if comm.size > 1 and not args.mpi and comm.rank != 0:
        comm.Barrier()
        sys.exit(0)

    # Initialize execution settings
    settings = ExecutionSettings(args, CFG_DIR, timestamp_suffix=timestamp_suffix)
    trace_print("4: Execution settings initialized", detail=True)
    
    # Print settings summary
    is_root = (comm is None) or (comm.rank == 0)
    settings.print_configuration(is_root)
    
    # For backward compatibility, expose key variables
    vf_ngrid = settings.vf_ngrid
    output_root = settings.output_root
    cfg_dir_bundle = settings.cfg_dir_bundle
    trace_print("5: Execution settings complete", detail=True)
    
    #  set-up runner ------------------------------------------------------
    model_config = load_config(settings.cfg_dir_bundle)
    trace_print("6: Model config loaded", detail=True)

    # Apply use_taxes from CLI to model config
    if getattr(args, 'use_taxes', False):
        model_config["master"]["parameters"]["use_taxes"] = True

    # Apply a_grid_multiplier in standard (non-sweep) runs:
    a_grid_mult = model_config["master"]["settings"].get("a_grid_multiplier", 1)

    # Skip bundle saving if --skip-bundle-save flag is set (useful for timing runs)
    skip_bundle_save = getattr(args, 'skip_bundle_save', False)
    save_by_default = is_root and not skip_bundle_save
    solver_rank = comm.rank if comm is not None else 0
    trace_print(f"7: is_root={is_root}, solver_rank={solver_rank}", detail=True)

    # --- Optional Pre-compilation Step ---
    if args.precompile and is_root:
        trace_print("8: Starting precompilation", detail=True)
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
            param_paths=ExecutionSettings.get_param_paths(args.experiment_set),
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
        
        trace_print("9: Precompilation complete", detail=True)

    if comm is not None:
        comm.Barrier()

    trace_print("10: Setting up metric functions", detail=True)
    #  set-up plotting configuration -----------------------------------------------
    plot_config = settings.get_plot_config()
    
    # Set up plotting and CSV export functions
    # Note: We'll handle both plots and CSV in the plotting section
    if is_root:
        if args.csv_export and args.plots:
            print("\n*** BOTH MODE: Generating plots AND exporting CSV data ***\n")
        elif args.csv_export:
            print("\n*** CSV EXPORT MODE: Plot data will be saved as CSV files ***\n")
        elif args.plots:
            print("\n*** PLOT MODE: Generating matplotlib plots ***\n")
    
    # For metrics, use CSV factory if CSV export is requested
    if args.csv_export:
        plot_factory = csv_plot_comparison_factory
    else:
        plot_factory = plot_comparison_factory
    
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
    if is_root:
        print(f"  [Metrics] requested_metrics from settings: {settings.requested_metrics}")
        print(f"  [Metrics] AVAILABLE_METRICS keys: {list(AVAILABLE_METRICS.keys())}")
    for metric in settings.requested_metrics:
        if metric in AVAILABLE_METRICS:
            metric_fns[metric] = AVAILABLE_METRICS[metric]
        else:
            if is_root:
                print(f"Warning: Unknown metric '{metric}' requested, ignoring.")

    if is_root:
        print(f"  [Metrics] Final metric_fns: {list(metric_fns.keys())}")
    trace_print(f"11: Selected metrics: {list(metric_fns.keys())}", detail=True)
    
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
    
    # ====================================================================
    #  SWEEP MODE: Use mpi_map to distribute across MPI ranks
    # ====================================================================
    if args.sweep:
        trace_print("SWEEP: Starting sweep mode")
        from dynx.runner import mpi_map
        
        # Build design matrix from experiment set
        design_matrix, sweep_param_paths, sweep_config = ExecutionSettings.build_sweep_design_matrix(args.experiment_set)
        fixed_params = sweep_config.get("fixed", {})
        
        if is_root:
            print("\n" + "="*60)
            print("SWEEP MODE")
            print("="*60)
            print(f"Experiment set: {args.experiment_set}")
            print(f"param_paths: {sweep_param_paths}")
            print(f"Design matrix: {len(design_matrix)} configurations")
            print(f"  Methods: {sweep_config['methods']}")
            print(f"  Grid sizes: {sweep_config['grid_sizes']}")
            print(f"  H sizes: {sweep_config['H_sizes']}")
            print(f"  Fixed: {fixed_params}")
            print("="*60 + "\n")
        
        # Create sweep-specific runner with the right param_paths
        # Need to patch base config with fixed params (H_points -> S_points sync)
        sweep_base_cfg = copy.deepcopy(model_config)
        
        # Set fixed parameters from sweep config (override command-line args)
        if "delta_pb" in fixed_params:
            sweep_base_cfg["master"]["parameters"]["delta_pb"] = fixed_params["delta_pb"]

        # Handle use_taxes from sweep config or fall back to CLI arg
        if "use_taxes" in fixed_params:
            sweep_base_cfg["master"]["parameters"]["use_taxes"] = fixed_params["use_taxes"]
            if is_root:
                print(f"  Using use_taxes from sweep config: {fixed_params['use_taxes']}")

        # Use periods from sweep config if specified, otherwise fall back to args
        sweep_periods = fixed_params.get("periods", args.periods)
        if "periods" in fixed_params:
            sweep_base_cfg["master"]["horizon"] = fixed_params["periods"]
            # CRITICAL: Update args.periods so solver uses correct value
            args.periods = sweep_periods
            if is_root:
                print(f"  Using periods from sweep config: {sweep_periods}")
        
        # Use vfi_ngrid from sweep config if specified, otherwise fall back to args
        sweep_vfi_ngrid = int(float(fixed_params.get("vfi_ngrid", args.vfi_ngrid)))
        if "vfi_ngrid" in fixed_params and is_root:
            print(f"  Using vfi_ngrid from sweep config: {sweep_vfi_ngrid}")
        
        # Reference method for baseline comparison (used by fast method sweeps)
        ref_method = sweep_config.get("ref_method", None)
        ref_params_override = sweep_config.get("ref_params_override", {})
        
        if is_root and ref_method:
            print(f"  ref_method: {ref_method}")
            print(f"  ref_params_override: {ref_params_override}")
        
        # NOTE: In sweep mode with mpi_map, each rank solves different configs independently.
        # Pass use_mpi=False and comm=None so each rank:
        # 1. Compiles its own model (done via make_sweep_model's comm=None)
        # 2. Initializes its own terminal values (solver needs comm=None)
        # 3. Solves without MPI synchronization within a single model
        sweep_runner = CircuitRunner(
            base_cfg=sweep_base_cfg,
            param_paths=sweep_param_paths,
            method_param_path=sweep_param_paths[0],  # Exclude method from hash
            model_factory=lambda cfg: make_sweep_model(cfg, args, sweep_periods, sweep_vfi_ngrid),
            solver=make_housing_solver(args, use_mpi=False, comm=None, baseline_method=None),
            metric_fns=metric_fns,
            output_root=output_root,
            save_by_default=not skip_bundle_save,  # Skip if --skip-bundle-save flag is set
            load_if_exists=not args.fresh_fast,
        )
        sweep_runner.stages_to_load = [args.stages_L2dev]
        sweep_runner.stages_to_save = [args.stages_L2dev]
        
        # Store ref_method for baseline comparison
        # Read design_matrix.csv from VFI sweep to get correct bundle paths (avoids hash mismatch)
        # Note: pd (pandas) is imported at module level, Path is already imported
        if ref_method:
            design_csv = output_root / "design_matrix.csv"
            ref_bundle_by_H = {}  # Map H -> bundle path
            ref_grid = ref_params_override.get("grid_sizes", 20000)  # Required grid size
            
            if is_root:
                print(f"  [Setup] Looking for design_matrix.csv at: {design_csv}")
                print(f"  [Setup] design_csv.exists(): {design_csv.exists()}")
            
            if design_csv.exists():
                df = pd.read_csv(design_csv)
                if is_root:
                    print(f"  [Setup] CSV columns: {list(df.columns)}")
                    print(f"  [Setup] CSV rows: {len(df)}")
                    print(f"  [Setup] Methods in CSV: {df['master.methods.upper_envelope'].unique().tolist() if 'master.methods.upper_envelope' in df.columns else 'N/A'}")
                
                # Filter for ref_method AND ref_grid
                mask = (df['master.methods.upper_envelope'] == ref_method)
                if 'master.settings.a_points' in df.columns:
                    mask = mask & (df['master.settings.a_points'] == ref_grid)
                vfi_rows = df[mask]
                
                if is_root:
                    print(f"  [Setup] Filtered rows: {len(vfi_rows)}")
                
                for _, row in vfi_rows.iterrows():
                    H = int(row['master.settings.H_points'])
                    # Use bundle_dir if available, otherwise construct from param_hash
                    bundle_dir = row.get('bundle_dir', '')
                    if bundle_dir:
                        # bundle_dir may be relative (hash/method) - prepend bundles path
                        bundle_path = Path(bundle_dir)
                        if not bundle_path.is_absolute():
                            bundle_path = output_root / "bundles" / bundle_dir
                        ref_bundle_by_H[H] = bundle_path
                    elif 'param_hash' in row:
                        # Construct path from hash
                        h = row['param_hash']
                        ref_bundle_by_H[H] = output_root / "bundles" / h / ref_method
            
            sweep_runner.ref_bundle_by_H = ref_bundle_by_H
            
            if is_root:
                print(f"  [Setup] Looking for {ref_method} with grid={ref_grid}")
                print(f"  [Setup] Loaded {len(ref_bundle_by_H)} VFI baselines from design_matrix.csv")
                for H, path in sorted(ref_bundle_by_H.items()):
                    exists = path.exists() if path else False
                    print(f"  [Setup] H={H}: {path}, exists={exists}")

            

        
        trace_print(f"SWEEP: Running mpi_map with {len(design_matrix)} configurations")
        
        # Add plot generation metric if plots requested (runs locally on each rank)
        if args.plots or args.csv_export:
            plot_config = settings.get_plot_config()
            sweep_methods = sweep_config["methods"]  # Capture for closure
            
            def sweep_plot_metric(model):
                """Generate plots for each sweep config - runs locally on each MPI rank."""
                try:
                    from pathlib import Path

                    # Get params from model (stored by make_sweep_model for correct bundle hashing)
                    if hasattr(model, '_sweep_params'):
                        method = model._sweep_params['method']
                        a_points = model._sweep_params['a_points']
                        H_points = model._sweep_params['H_points']
                    else:
                        # Fallback to extracting from grid (may produce incorrect bundle paths)
                        first_period = model.get_period(0)
                        ownc_stage = first_period.get_stage("OWNC")
                        a_points = ownc_stage.dcsn.grid.w.shape[0] if hasattr(ownc_stage.dcsn.grid, 'w') else 0
                        H_points = len(ownc_stage.dcsn.grid.H_nxt) if hasattr(ownc_stage.dcsn.grid, 'H_nxt') else 0
                        if len(sweep_methods) == 1:
                            method = sweep_methods[0]
                        else:
                            method = ownc_stage.dcsn.model.get("upper_envelope", sweep_methods[0]) if hasattr(ownc_stage.dcsn, 'model') and isinstance(ownc_stage.dcsn.model, dict) else sweep_methods[0]
                    
                    # Get bundle path from runner (hash excludes method via method_param_path)
                    params = np.array([method, a_points, H_points], dtype=object)
                    bundle_path = sweep_runner._bundle_path(params)
                    
                    if bundle_path:
                        bundle_images_dir = Path(bundle_path) / f"images_{timestamp_suffix}"
                    else:
                        # Fallback if _bundle_path returns None
                        bundle_images_dir = output_root / "images" / f"{method}_{a_points}_{H_points}_{timestamp_suffix}"
                    bundle_images_dir.mkdir(parents=True, exist_ok=True)
                    
                    if args.csv_export:
                        print(f"  [Sweep] Exporting CSV for {method} to {bundle_images_dir}")
                        csv_generate_plots(model, method, bundle_images_dir,
                                         egm_bounds=plot_config['egm_bounds'],
                                         skip_egm_plots=args.skip_egm_plots)
                    
                    if args.plots:
                        print(f"  [Sweep] Generating plots for {method} to {bundle_images_dir}")
                        generate_plots(model, method, bundle_images_dir,
                                     egm_bounds=plot_config['egm_bounds'],
                                     skip_egm_plots=args.skip_egm_plots,
                                     policy_config=plot_config.get('policy_config'))

                    # Aggressive cleanup after plotting to free memory for next config
                    # Always do complete cleanup in sweep mode to prevent memory accumulation
                    cleanup_model_complete(model)

                    return 1
                except Exception as err:
                    import traceback
                    print(f"[warn] sweep plot-gen failed: {err}")
                    traceback.print_exc()
                    return 0

            # Add to metric functions (runs on each rank for its assigned configs)
            sweep_runner.metric_fns["_sweep_plots"] = sweep_plot_metric

        # Always add an explicit cleanup metric in sweep mode.
        # Rationale: even if per-period memory is freed inside `run_time_iteration`,
        # DynX/runner internals may still retain references across configurations
        # (e.g., caches). Making the model object "small" prevents memory growth
        # over long sweeps (especially when periods is large).
        def sweep_cleanup_metric(model):
            try:
                cleanup_model_complete(model)
                return 1
            except Exception:
                return 0

        # Insert last so it runs after all other metrics.
        sweep_runner.metric_fns["_sweep_cleanup"] = sweep_cleanup_metric
        
        # Run sweep using mpi_map
        # Note: mpi_map returns None on non-root ranks when MPI is enabled
        result = mpi_map(
            sweep_runner, 
            design_matrix, 
            mpi=args.mpi, 
            comm=comm,
            return_models=False  # Don't gather models - plots generated via metric
        )
        
        # Handle None return on non-root MPI ranks
        if result is None:
            # Non-root ranks just wait at barrier and exit
            if comm is not None:
                comm.Barrier()
            trace_print("SWEEP: Non-root rank complete")
            return
        
        results_df, _ = result
        
        # Only root has the aggregated results
        if is_root and results_df is not None:
            trace_print("SWEEP: Saving results")
            
            # Note: avg_ownc_time_per_period is now computed directly in whisperer.py
            # and recorded as a simple float (nested stage_timings is filtered by dynx)
            
            # Save results DataFrame
            results_path = output_root / "sweep_results.csv"
            results_df.to_csv(results_path, index=False)
            print(f"\nSweep results saved to: {results_path}")
            
            # Print summary table (exclude verbose columns)
            print("\n" + "="*60)
            print("SWEEP RESULTS SUMMARY")
            print("="*60)
            # Exclude columns with large nested data (period_timings, stage_timings, _sweep_plots)
            verbose_cols = ['period_timings', 'stage_timings', '_sweep_plots']
            summary_cols = [c for c in results_df.columns if c not in verbose_cols]
            print(results_df[summary_cols].to_string())
            print("="*60 + "\n")
            
            # Also save design matrix
            write_design_matrix_csv(sweep_runner, design_matrix)
            print(f"Design matrix saved to: {output_root}/design_matrix.csv")
        
        if comm is not None:
            comm.Barrier()
        
        trace_print("SWEEP: Complete")
        return  # Exit after sweep mode
    
    # ====================================================================
    #  STANDARD MODE: Sequential execution (original behavior)
    # ====================================================================
    
    # Set up main runner for standard mode
    trace_print("12: Creating CircuitRunner", detail=True)
    param_paths = ExecutionSettings.get_param_paths(args.experiment_set)
    if is_root:
        print(f"Using experiment set: {args.experiment_set}")
        print(f"  param_paths: {param_paths}")
    
    runner = CircuitRunner(
        base_cfg=model_config,
        param_paths=param_paths,
        model_factory=lambda cfg: make_housing_model(args, cfg, args.periods, vf_ngrid, comm),
        solver=make_housing_solver(args, args.mpi, comm, baseline_method=settings.baseline_method),
        metric_fns=metric_fns,
        output_root=output_root,
        bundle_prefix=args.config_id,
        save_by_default=save_by_default,
        load_if_exists=True,
    )
    trace_print("13: CircuitRunner created", detail=True)
    
    # Configure stages to load/save for L2 deviation calculation
    runner.stages_to_load = [args.stages_L2dev]
    runner.stages_to_save = [args.stages_L2dev]
    trace_print(f"14: Stages configured: load={runner.stages_to_load}, save={runner.stages_to_save}", detail=True)
    
    all_metrics = []
    all_param_vectors = []  # Collect all parameter vectors for design matrix

    # --------------------------------------------------------------------
    #  1) Solve, plot, and process baseline immediately (only if needed)
    # --------------------------------------------------------------------
    if settings.should_run_baseline:
        with MemoryMonitor(f"Baseline computation ({settings.baseline_method})", log_start=is_root, log_end=is_root):
            trace_print("16: Starting baseline computation", detail=True)
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
            
            trace_print("17: Running baseline solver", detail=True)
            ref_metrics, ref_model = runner.run(
                ref_params,
                return_model=is_root,
                rank=solver_rank
            )
            trace_print("18: Baseline solver complete", detail=True)
            
            # Register baseline model in unified cache for reuse
            if is_root and ref_model is not None:
                try:
                    from dynx.runner.model_cache import register_baseline_model
                    # Register with periods [0, 1] since baseline loads full model
                    register_baseline_model(settings.baseline_method, ref_model, periods=[0, 1])
                    print(f"  Registered {settings.baseline_method} in unified cache for metric reuse")
                except ImportError:
                    trace_print("18.1: Unified model cache not available", detail=True)
            
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

            # Generate plots and/or CSV exports immediately and delete model
            if is_root and (args.plots or args.csv_export) and ref_model is not None:
                trace_print("19: Generating baseline plots", detail=True)
                
                # Use organized directory structure for outputs
                if baseline_bundle_path:
                    from pathlib import Path
                    # Create clean directory structure: bundle/VFI_HDGRID_GPU/images_YYYYMMDD_HHMMSS/...
                    baseline_images_dir = Path(baseline_bundle_path) / f"images_{timestamp_suffix}"
                    baseline_images_dir.mkdir(parents=True, exist_ok=True)
                    
                    # For CSV export mode, data goes to appropriate subdirectories
                    if args.csv_export:
                        baseline_output_dir = baseline_images_dir  # CSV function will handle subdirs
                    else:
                        # For regular plots, use policy_plots subdirectory
                        baseline_output_dir = baseline_images_dir / "policy_plots"
                        baseline_output_dir.mkdir(parents=True, exist_ok=True)
                else:
                    # Fallback to original location if no bundle path (already has timestamp suffix)
                    baseline_output_dir = settings.img_dir
                    baseline_output_dir.mkdir(exist_ok=True)
                
                # Clean up non-essential data before plotting if in low-memory mode
                if args.low_memory:
                    cleanup_model(ref_model, aggressive=False)
                    log_memory_usage("after baseline model cleanup", verbose=is_root)
                
                try:
                    # Handle CSV export
                    if args.csv_export:
                        print(f"  Exporting CSV data for {settings.baseline_method}...")
                        if baseline_bundle_path:
                            if args.skip_egm_plots:
                                print(f"  CSV data will be saved to: {baseline_images_dir}/policy_csv (EGM plots skipped)")
                            else:
                                print(f"  CSV data will be saved to: {baseline_images_dir}/egm_csv and {baseline_images_dir}/policy_csv")
                        csv_generate_plots(ref_model, settings.baseline_method, baseline_images_dir, 
                                         egm_bounds=plot_config['egm_bounds'],
                                         skip_egm_plots=args.skip_egm_plots)
                    
                    # Handle matplotlib plots
                    if args.plots:
                        print(f"  Generating plots for {settings.baseline_method}...")
                        if baseline_bundle_path:
                            plot_output_dir = baseline_images_dir / "policy_plots"
                            plot_output_dir.mkdir(parents=True, exist_ok=True)
                            print(f"  Plots will be saved to: {plot_output_dir}")
                        else:
                            plot_output_dir = baseline_output_dir
                        
                        generate_plots(ref_model, settings.baseline_method, plot_output_dir,
                                     egm_bounds=plot_config['egm_bounds'],
                                     skip_egm_plots=args.skip_egm_plots,
                                     policy_config=plot_config.get('policy_config'))
                except Exception as err:
                    print(f"[warn] plot-gen for {settings.baseline_method} failed: {err}")
                finally:
                    cleanup_model_complete(ref_model)
                    del ref_model
                    gc.collect()
                    trace_print("20: Baseline plots complete, model deleted", detail=True)
            
            # Trigger cleanup if memory usage is high
            mem_config = settings.get_memory_config()
            cleanup_if_needed(memory_config.get("cleanup_threshold_gb", 32))
            log_memory_usage("after baseline computation", verbose=is_root)
    
    # If baseline wasn't computed but we still have fast methods, create ref_params for them
    elif settings.needs_baseline and 'ref_params' not in locals():
        ref_params = settings.get_baseline_params()
        if is_root:
            print(f"\n» Baseline ({settings.baseline_method}) will be loaded from existing bundles for comparison metrics")
        trace_print("21: Baseline ref_params created for loading", detail=True)

    # --------------------------------------------------------------------
    #  2) Solve test methods one by one, processing each immediately  
    # --------------------------------------------------------------------
    
    if settings.fast_methods_to_run:
        trace_print(f"22: Starting fast methods: {settings.fast_methods_to_run}", detail=True)
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
            trace_print("23: Skipping baseline loading", detail=True)

        if is_root:  # print from root only
            print(f"\n» Fast methods: {', '.join(settings.fast_methods_to_run)}")
            
        for i, method in enumerate(settings.fast_methods_to_run):
            with MemoryMonitor(f"Fast method computation ({method})", log_start=is_root, log_end=is_root):
                trace_print(f"24.{i+1}: Starting {method}", detail=True)
                if is_root:
                    print(f"  Solving {method}...")
                
                params = settings.get_method_params(method)
                # Apply a_grid_multiplier for non-VFI methods (hash reflects scaled asset grids; w_points unchanged)
                if not args.sweep and a_grid_mult != 1 and method not in ["VFI_HDGRID", "VFI_HDGRID_GPU"]:
                    params[1] = params[1] * a_grid_mult  # a_points
                    params[2] = params[2] * a_grid_mult  # a_nxt_points

                print(params)
                
                all_param_vectors.append(params)  # Add to design matrix
                
                # Print parameter hash and bundle path for this method
                method_hash = runner._hash_param_vec(params)
                method_bundle_path = runner._bundle_path(params)
                if is_root:
                    print(f"  Parameter hash: {method_hash}")
                    if method_bundle_path:
                        print(f"  Bundle path: {method_bundle_path}")
                
                trace_print(f"25.{i+1}: Running {method} solver", detail=True)
                metrics, model = runner.run(params, return_model=is_root, rank=solver_rank)
                trace_print(f"26.{i+1}: {method} solver complete", detail=True)
                
                metrics["master.methods.upper_envelope"] = method
                metrics["param_hash"] = runner._hash_param_vec(params)
                if settings.needs_baseline and 'ref_params' in locals():
                    metrics["reference_bundle_hash"] = runner._hash_param_vec(ref_params)
                else:
                    metrics["reference_bundle_hash"] = "no_baseline"
                metrics["latest_time_id"] = args.RUN_ID 
                all_metrics.append(metrics)
                
                # Generate plots and/or CSV exports immediately and delete model
                if is_root and (args.plots or args.csv_export) and model is not None:
                    trace_print(f"27.{i+1}: Generating {method} plots", detail=True)
                    
                    # Use organized directory structure for outputs
                    if method_bundle_path:
                        from pathlib import Path
                        # Create clean directory structure: bundle/FUES/images_YYYYMMDD_HHMMSS/...
                        bundle_images_dir = Path(method_bundle_path) / f"images_{timestamp_suffix}"
                        bundle_images_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        # Fallback to original location if no bundle path (already has timestamp suffix)
                        bundle_images_dir = settings.img_dir
                        bundle_images_dir.mkdir(exist_ok=True)
                    
                    # Clean up non-essential data before plotting if in low-memory mode
                    if args.low_memory:
                        cleanup_model(model, aggressive=False)
                        log_memory_usage(f"after {method} model cleanup", verbose=is_root)
                    
                    try:
                        # Handle CSV export
                        if args.csv_export:
                            print(f"  Exporting CSV data for {method}...")
                            if method_bundle_path:
                                if args.skip_egm_plots:
                                    print(f"  CSV data will be saved to: {bundle_images_dir}/policy_csv (EGM plots skipped)")
                                else:
                                    print(f"  CSV data will be saved to: {bundle_images_dir}/egm_csv and {bundle_images_dir}/policy_csv")
                            csv_generate_plots(model, method, bundle_images_dir, 
                                             egm_bounds=plot_config['egm_bounds'], 
                                             y_idx_list=plot_config['y_idx_list'],
                                             skip_egm_plots=args.skip_egm_plots)
                        
                        # Handle matplotlib plots (including policy plots when both flags are set)
                        if args.plots:
                            print(f"  Generating plots for {method}...")
                            if method_bundle_path:
                                # Pass base images dir, generate_plots will create subdirs
                                plot_output_dir = bundle_images_dir
                                print(f"  Plots will be saved to: {plot_output_dir}/{method}/")
                            else:
                                plot_output_dir = bundle_images_dir

                            generate_plots(model, method, plot_output_dir,
                                         egm_bounds=plot_config['egm_bounds'],
                                         y_idx_list=plot_config['y_idx_list'],
                                         skip_egm_plots=args.skip_egm_plots,
                                         policy_config=plot_config.get('policy_config'))

                            # Generate consumption policy comparison plot (FUES vs VFI side-by-side)
                            # Only for EGM-based methods when baseline is available
                            print(f"  [DEBUG] Checking comparison plot: method={method}, needs_baseline={settings.needs_baseline}, has_ref_params={hasattr(runner, 'ref_params')}")
                            if method in ['FUES', 'CONSAV', 'DCEGM']:
                                try:
                                    from dynx.stagecraft.io import load_circuit
                                    # Try to get baseline bundle path
                                    baseline_bundle_path = None
                                    if hasattr(runner, 'ref_params') and runner.ref_params is not None:
                                        baseline_bundle_path = runner._bundle_path(runner.ref_params)
                                        print(f"  [DEBUG] baseline_bundle_path from ref_params: {baseline_bundle_path}")
                                    elif hasattr(settings, 'baseline_method') and settings.baseline_method:
                                        # Try to construct baseline params from settings
                                        baseline_params = settings.get_baseline_params()
                                        baseline_bundle_path = runner._bundle_path(baseline_params)
                                        print(f"  [DEBUG] baseline_bundle_path from settings: {baseline_bundle_path}")

                                    if baseline_bundle_path and baseline_bundle_path.exists():
                                        print(f"  Generating FUES vs VFI consumption comparison plot...")
                                        baseline_model = load_circuit(baseline_bundle_path)
                                        if baseline_model is not None:
                                            plot_compare_consumption_policy(
                                                model_fues=model,
                                                model_vfi=baseline_model,
                                                image_dir=plot_output_dir,
                                                plot_period=0,
                                                bounds=plot_config.get('egm_bounds', {})
                                            )
                                            print(f"  Comparison plot saved to: {plot_output_dir}")
                                            # Clean up baseline model after comparison plot
                                            del baseline_model
                                            gc.collect()
                                        else:
                                            print(f"  [DEBUG] baseline_model is None after load_circuit")
                                    else:
                                        print(f"  [DEBUG] baseline_bundle_path doesn't exist or is None: {baseline_bundle_path}")
                                except Exception as comp_err:
                                    import traceback
                                    print(f"[warn] consumption comparison plot failed: {comp_err}")
                                    traceback.print_exc()
                    except Exception as err:
                        print(f"[warn] plot-gen for {method} failed: {err}")
                    finally:
                        cleanup_model_complete(model)
                        del model
                        gc.collect()
                        trace_print(f"28.{i+1}: {method} plots complete, model deleted", detail=True)
                
                # Trigger cleanup after each method to prevent memory accumulation
                cleanup_if_needed(memory_config.get("cleanup_threshold_gb", 32))
                
                # Log memory usage periodically
                if (i + 1) % 2 == 0 and is_root:  # Log every 2 methods, rank 0 only
                    log_memory_usage(f"after {i+1} fast methods")

    if comm is not None:
        comm.Barrier()

    trace_print("29: Creating final summary", detail=True)
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
            print("Low memory mode: ENABLED (Q and lambda arrays cleared, period 0 preserved for plots)")
        print("="*60)
    
    if is_root:
        log_memory_usage("at end of solve_runner")
    
    trace_print("30: Main function complete", detail=True)

if __name__ == "__main__":
    main()