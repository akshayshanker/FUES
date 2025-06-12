#!/usr/bin/env python
"""
Housing model with renting – baseline HD grid search/fast-methods runner.

This script provides workflows for solving the housing-renting model using different
solution methods, and comparing the methods against a baseline methods. 

The script support for both serial and parallel (MPI) execution.

Workflows
---------

**Serial Execution (No MPI)**

1.  **Full Run: Fresh Baseline + Fast Methods**
    This workflow computes the high-density baseline (`VFI_HDGRID`) and all
    fast methods from scratch. This is useful for a complete, fresh build.

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
    This runs only the "fast" solvers (e.g., `FUES`, `CONSAV`). It relies on a
    pre-computed baseline solution (with the same `vfi-ngrid` and `HD-points` 
    as entered in the command line) being present in the output directory.
    If a matching baseline is found, it's loaded for comparison.

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
       When `--recompute-baseline` is omitted, the runner will search for an
       existing baseline bundle matching the `vfi-ngrid` and `HD-points`
       settings in the `output-root`.

**Parallel Execution (MPI)**

For large-scale problems, running with MPI is recommended to accelerate the
`VFI_HDGRID` baseline computation.

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
        The solvers currently require that the number of housing x income grid points
        >= the number of mpicores. Ideally, run this s.t. each core gets a single 
        housing-income point to work on in the HD grid search. 
"""

from __future__ import annotations
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dynx.runner import CircuitRunner, mpi_map
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


CFG_DIR = Path(__file__).parent / "config_HR"
BASE = "VFI_HDGRID"
ALL_METHODS = ["VFI_HDGRID", "FUES", "FUES2DEV", "CONSAV", "DCEGM"]
FAST_METHODS = ["FUES"]


def patch_cfg(cfg_container: dict, periods: int, vf_ngrid: int) -> dict:
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
    Patch the YAML config → build a ModelCircuit → compile its stages.

    All compile-time warnings are logged; any hard failure is re-raised so
    that the calling code (runner) can handle / abort cleanly.
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
        # 0. attach MPI-aware operators once per ModelCircuit
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
    output_root = packroot / f"{args.output_root}_{args.vfi_ngrid}"
    output_root.mkdir(parents=True, exist_ok=True)

    #  set-up runner ------------------------------------------------------
    cfg_container = load_config(CFG_DIR)
    save_by_default = (comm is None) or (comm.rank == 0)
    is_root = (comm is None) or (comm.rank == 0)
    solver_rank = comm.rank if comm is not None else 0

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

    # --------------------------------------------------------------------
    #  1) make sure baseline is available / solved
    # --------------------------------------------------------------------
    
    HD_POINTS = int(args.HD_points)
    if BASE in methods:
        runner.load_if_exists = not args.recompute_baseline
        if is_root:  # print from root only
            print("\n» Baseline:", "recompute" if args.recompute_baseline else "load/solve-if-missing")

        base_params = np.array([BASE, HD_POINTS, HD_POINTS, HD_POINTS], dtype=object)
        runner.ref_params = base_params
        base_metrics, base_model = runner.run(
            base_params,
            return_model=is_root,
            rank=solver_rank
        )
        base_metrics["master.methods.upper_envelope"] = BASE

        # Attach the baseline's parameters to the runner so that deviation
        # metrics (e.g. dev_c_L2) can find the correct reference bundle.
        # if is_root:
        #     runner.ref_params = base_params

        # Write design matrix for the baseline run
        # if is_root:
        #     _write_design_matrix(runner, base_params)

    # --------------------------------------------------------------------
    #  2) run fast methods (optionally fresh)
    # --------------------------------------------------------------------
    
    STD_POINTS = int(args.grid_points)
    if FAST_METHODS:
        runner.model_factory = lambda cfg: make_housing_model(cfg, args.periods, vf_ngrid, comm)
        runner.load_if_exists = not args.fresh_fast
        runner.ref_params = base_params
        # print
        xs = np.asarray([[m, STD_POINTS, STD_POINTS, STD_POINTS] for m in FAST_METHODS], dtype=object)
        if is_root:  # print from root only
            print(f"\n» Fast methods: {', '.join(FAST_METHODS)}")
        t0 = time.time()
        res_df, models = mpi_map(
            runner,
            xs,
            mpi=False,
            comm=None,
            comm_solver=comm,
            return_models=is_root
        )
        # print(f"✓ completed in {time.time()-t0:.1f}s")
    else:
        res_df, models = None, None

    if comm is not None:
        comm.Barrier()

    # ----------------------------------------------------------
    #  include baseline row (if we just computed / loaded it)
    # ----------------------------------------------------------
    if BASE in methods and is_root:
        base_df = pd.DataFrame([base_metrics])
        if res_df is not None:
            res_df = pd.concat([base_df, res_df], ignore_index=True)
        else:
            res_df = base_df

    # --------------------------------------------------------------------
    #  3) summary + plots
    # --------------------------------------------------------------------
    if is_root and res_df is not None and len(res_df):
        print_summary(res_df, output_root)

    # prepend baseline model/name if we have one
    if is_root and args.plots:
        models = [base_model] + models
        plot_seq = [BASE] + FAST_METHODS

        if models is not None:
            img_dir = output_root / "images"
            img_dir.mkdir(exist_ok=True)
            for m_obj, m_name in zip(models, plot_seq):
                try:
                    generate_plots(m_obj, m_name, img_dir)
                except Exception as err:
                    print(f"[warn] plot-gen for {m_name} failed: {err}")

if __name__ == "__main__":
    main()
