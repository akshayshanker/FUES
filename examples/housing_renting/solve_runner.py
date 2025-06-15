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

"""

from __future__ import annotations
import argparse
import copy
import sys
import time
from pathlib import Path
import gc

import numpy as np
import pandas as pd

from dynx.runner import CircuitRunner, write_design_matrix_csv
from dynx.stagecraft.io import load_config
from dynx.stagecraft.makemod import initialize_model_Circuit, compile_all_stages

# ────────────────────────────────────────────────────────────────────────────
#  local helpers (imported lazily to keep fall-back stubs tiny)
# ────────────────────────────────────────────────────────────────────────────
try:
    from whisperer import build_operators_for_circuit, run_time_iteration
    from helpers.euler_error import euler_error_metric
    from helpers.plots import generate_plots
    from helpers.tables import print_summary
    from helpers.metrics import dev_c_L2
except ImportError:
    from .whisperer import build_operators_for_circuit, run_time_iteration
    from .helpers.euler_error import euler_error_metric
    from .helpers.plots import generate_plots
    from .helpers.tables import print_summary
    from .helpers.metrics import dev_c_L2


CFG_DIR = Path(__file__).parent / "config_HR"
BASE = "VFI_HDGRID"
ALL_METHODS = ["VFI_HDGRID", "FUES", "FUES2DEV", "CONSAV", "DCEGM"]
FAST_METHODS = ["FUES"]
PRE_COMPILE_PARAMS = np.array(["VFI_HDGRID", 500, 500, 500], dtype=object)


def patch_cfg(cfg_container: dict, periods: int, vf_ngrid: int) -> dict:
    """
    Patch config with solution method and MPI compute settings.

    The upper-envelope method (`VFI_HDGRID`, `FUES`, etc.) determines
    whether the underlying solver should be VFI or EGM. This helper
    sets the correct `solution` and `compute` methods on consumption
    stages without requiring extra CLI flags.
    """
    cfg = copy.deepcopy(cfg_container)
    cfg["master"]["horizon"] = periods
    cfg["master"]["settings"]["N_arg_grid_vfi"] = vf_ngrid

    sol = cfg["master"]["methods"]["upper_envelope"]
    target = sol if sol in ("VFI_HDGRID", "VFI", "VFI_POOL") else "EGM"
    cfg["stages"]["OWNC"]["stage"]["methods"]["solution"] = target
    cfg["stages"]["RNTC"]["stage"]["methods"]["solution"] = target

    if sol == "VFI_HDGRID":
        cfg["stages"]["OWNC"]["stage"]["methods"]["compute"] = "MPI"
        cfg["stages"]["RNTC"]["stage"]["methods"]["compute"] = "MPI"
    else:
        cfg["stages"]["OWNC"]["stage"]["methods"]["compute"] = "SINGLE"
        cfg["stages"]["RNTC"]["stage"]["methods"]["compute"] = "SINGLE"
    return cfg


def make_housing_model(cfg_container: dict, periods: int, vf_ngrid: int, comm=None):
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
    cfg = patch_cfg(cfg_container, periods, vf_ngrid)

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
        # Crash early with a clear message – better than a silent mis-compile
        logger.error("Stage compilation failed – aborting make_housing_model()", exc_info=True)
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
        This closure:
        - attaches MPI-aware operators once per ModelCircuit
        - marks last-period consumption stages as terminal
        - runs backward time iteration
        """

        # 0. attach MPI-aware operators once per ModelCircuit
        ## TODO: This is redudnant as MPI use can be specified in congig file. 
        build_operators_for_circuit(mc, use_mpi=use_mpi, comm=comm)

        # 1. mark last-period consumption stages as terminal
        final_period = mc.get_period(len(mc.periods_list) - 1)
        for tag in ("OWNC", "RNTC"):
            final_period.get_stage(tag).status_flags["is_terminal"] = True
        
        #print(args.verbose)

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

    Parameters
    ----------
    argv : list[str] | None, optional
        Custom argument vector (for unit tests). If *None* (default),
        ``sys.argv[1:]`` is used.

    Key CLI options
    ---------------
    --periods INT
        Number of model periods to solve (e.g., 3 for periods 2, 1, 0).
    --ue-method STR
        Comma-separated list of UE methods to run, or ``ALL``.
    --output-root PATH
        Directory for all bundles, plots, and metrics.
    --bundle-prefix STR
        Prefix for bundle filenames within ``output-root``.
    --vfi-ngrid INT
        Number of choice-grid points used by the VFI baseline.
    --HD-points INT
        State-grid size (a, w, a_nxt) for the HD-grid baseline.
    --grid-points INT
        Same as ``--HD-points`` but for fast methods.
    --recompute-baseline
        Ignore any existing baseline bundle and recompute it.
    --fresh-fast
        Re-solve fast methods even if their bundles already exist.
    --plots
        Create diagnostic figures for every solved model.
    --mpi
        Activate MPI mode; the script will auto-detect the communicator.
    --verbose
        Print detailed progress and timing information.
    --precompile
        Run a small VFI job to pre-compile Numba functions.

    Behaviour
    ---------
    1.  Build or load the **baseline** (``VFI_HDGRID``), unless excluded.
    2.  Solve each requested **fast method** individually.
    3.  On rank 0:
        *   Generate plots (if ``--plots``).
        *   Compute Euler-error and consumption-norm metrics.
        *   Print a summary table.
        *   Write the design matrix to ``design_matrix.csv``.

    Notes
    -----
    **Parameter Hash**: Bundles are keyed by grid sizes only; the UE method
    itself is *not* part of the hash. Consequently, fast-method runs must
    specify the hash of the reference baseline via ``runner.ref_params`` so
    that deviations are computed against the correct bundle.

    (The ``method_param_path`` argument in `CircuitRunner` is used to
    specify the parameter path to the method that is not included in the hash.)
    """
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
    args = p.parse_args(argv or sys.argv[1:])

    #  MPI communicator
    from dc_smm.helpers.mpi_utils import get_comm
    comm = get_comm(args.mpi)
    if comm.size > 1 and not args.mpi and comm.rank != 0:
        comm.Barrier()
        sys.exit(0)

    #  parse grid size
    vf_ngrid = int(float(args.vfi_ngrid))

    #  method list
    methods = ALL_METHODS if args.ue_method.upper() == "ALL" \
        else [m.strip().upper() for m in args.ue_method.split(",")]

    #  IO paths
    packroot = Path.cwd()
    output_root = packroot / f"{args.output_root}"
    output_root.mkdir(parents=True, exist_ok=True)
    cfg_dir_bundle = CFG_DIR /  f"{args.bundle_prefix}"
    #  set-up runner ------------------------------------------------------
    cfg_container = load_config(cfg_dir_bundle)
    save_by_default = (comm is None) or (comm.rank == 0)
    is_root = (comm is None) or (comm.rank == 0)
    solver_rank = comm.rank if comm is not None else 0

    # --- Optional Pre-compilation Step ---
    if args.precompile:
        if is_root:
            print("\n--- Running Numba Pre-compilation for VFI_HDGRID ---")
        precompile_runner = CircuitRunner(
            base_cfg=cfg_container,
            param_paths=[
                "master.methods.upper_envelope",
                "master.settings.a_points",
                "master.settings.a_nxt_points",
                "master.settings.w_points",
            ],
            model_factory=lambda cfg: make_housing_model(cfg, 2, 100, comm), # Minimal settings
            solver=make_housing_solver(argparse.Namespace(verbose=False, periods=2), use_mpi=args.mpi, comm=comm),
            metric_fns={},
            save_by_default=False,
            load_if_exists=False,
        )
        try:
            precompile_runner.run(PRE_COMPILE_PARAMS, rank=solver_rank)
            if is_root:
                print("--- Pre-compilation Complete ---\n")
        except Exception as e:
            if is_root:
                print(f"--- Pre-compilation Failed: {e} ---", file=sys.stderr)

    comm.Barrier()

    #  set-up main runner ------------------------------------------------------
    runner = CircuitRunner(
        base_cfg=cfg_container,
        param_paths=[
            "master.methods.upper_envelope",
            "master.settings.a_points",
            "master.settings.a_nxt_points",
            "master.settings.w_points",
        ],
        model_factory=lambda cfg: make_housing_model(cfg, args.periods, vf_ngrid, comm),
        solver=make_housing_solver(args, args.mpi, comm),
        metric_fns={
            "euler_error": euler_error_metric,
            "dev_c_L2": dev_c_L2,
        },
        output_root=output_root,
        bundle_prefix=args.bundle_prefix,
        save_by_default=save_by_default,
        load_if_exists=True,
    )
    # NOTE: this has to be consistent with whatever the metric L2 actually wants to do!
    # TODO: make this more flexible, so that we can do L2dev on any stage. AND HARdwire it
    runner.stages_to_load = [args.stages_L2dev]
    runner.stages_to_save = [args.stages_L2dev]

    all_metrics = []
    all_param_vectors = []  # Collect all parameter vectors for design matrix

    # --------------------------------------------------------------------
    #  1) Solve, plot, and process baseline immediately
    # --------------------------------------------------------------------
    
    HD_POINTS = int(float(args.HD_points))
    if BASE in methods:
        runner.load_if_exists = not args.recompute_baseline
        if is_root:  # print from root only
            print("\n» Baseline:", "recompute" if args.recompute_baseline else "load/solve-if-missing")

        ref_params = np.array([BASE, HD_POINTS, HD_POINTS, HD_POINTS], dtype=object)
        runner.ref_params = ref_params
        all_param_vectors.append(ref_params)  # Add to design matrix
        
        ref_metrics, ref_model = runner.run(
            ref_params,
            return_model=is_root,
            rank=solver_rank
        )
        ref_metrics["master.methods.upper_envelope"] = BASE
        ref_metrics["param_hash"] = runner._hash_param_vec(ref_params)
        ref_metrics["reference_bundle_hash"] = runner._hash_param_vec(ref_params)
        ref_metrics["latest_time_id"] = args.RUN_ID 
        all_metrics.append(ref_metrics)

        # Generate plots immediately and delete model
        if is_root and args.plots and ref_model is not None:
            img_dir = output_root / "images" 
            img_dir.mkdir(exist_ok=True)
            try:
                print(f"  Generating plots for {BASE}...")
                generate_plots(ref_model, BASE, img_dir)
            except Exception as err:
                print(f"[warn] plot-gen for {BASE} failed: {err}")
            finally:
                del ref_model
                gc.collect()

    # --------------------------------------------------------------------
    #  2) Solve test methods one by one, processing each immediately  
    # --------------------------------------------------------------------
    
    STD_POINTS = int(float(args.grid_points))
    fast_methods_to_run = [m for m in FAST_METHODS if m in methods]
    
    if fast_methods_to_run:
        runner.load_if_exists = not args.fresh_fast
        runner.ref_params = ref_params if BASE in methods else None
        
        if is_root:  # print from root only
            print(f"\n» Fast methods: {', '.join(fast_methods_to_run)}")
            
        for method in fast_methods_to_run:
            if is_root:
                print(f"  Solving {method}...")
                
            params = np.array([method, STD_POINTS, STD_POINTS, STD_POINTS], dtype=object)
            all_param_vectors.append(params)  # Add to design matrix
            
            metrics, model = runner.run(params, return_model=is_root, rank=solver_rank)
            metrics["master.methods.upper_envelope"] = method
            metrics["param_hash"] = runner._hash_param_vec(params)
            metrics["reference_bundle_hash"] = runner._hash_param_vec(ref_params)
            metrics["latest_time_id"] = args.RUN_ID 
            all_metrics.append(metrics)
            
            # Generate plots immediately and delete model
            if is_root and args.plots and model is not None:
                img_dir = output_root / "images"
                img_dir.mkdir(exist_ok=True) 
                try:
                    print(f"  Generating plots for {method}...")
                    generate_plots(model, method, img_dir)
                except Exception as err:
                    print(f"[warn] plot-gen for {method} failed: {err}")
                finally:
                    del model
                    gc.collect()

    if comm is not None:
        comm.Barrier()

    # --------------------------------------------------------------------
    #  3) Create final summary table and save design matrix
    # --------------------------------------------------------------------
    if is_root and all_metrics:
        res_df = pd.DataFrame(all_metrics)
        print_summary(res_df, output_root)
        
        # Save complete design matrix with all parameter vectors
        if all_param_vectors:
            design_matrix = np.array(all_param_vectors, dtype=object)
            write_design_matrix_csv(runner, design_matrix)
            if is_root:
                print(f"  Design matrix saved to {output_root}/design_matrix.csv")

if __name__ == "__main__":
    main()
