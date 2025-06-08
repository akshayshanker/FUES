#!/usr/bin/env python
"""
Housing model with renting – baseline/fast-method runner.

Workflows
---------
1) Build / refresh baseline
   $ python -m examples.housing_renting.circuit_runner_solving \
       --ue-method VFI_HDGRID --mpi                 # <– fastest on many cores
   # or force overwrite:
   $ ... --recompute-baseline

2) Fast methods (auto-load baseline, no MPI needed)
   $ python -m examples.housing_renting.circuit_runner_solving \
       --ue-method FUES,CONSAV --periods 3         # loads baseline if present
   # fresh run of the fast solvers only
   $ ... --fresh-fast

3) Full benchmark
   $ python -m examples.housing_renting.circuit_runner_solving --ue-method ALL
"""

from __future__ import annotations
from pathlib import Path
import argparse, copy, sys, time

import numpy as np
import pandas as pd

from dynx.runner       import CircuitRunner, mpi_map
from dynx.stagecraft.io import load_config
from dynx.stagecraft.makemod import initialize_model_Circuit, compile_all_stages
from dynx.runner.metrics.deviations import dev_c_L2

# ────────────────────────────────────────────────────────────────────────────
#  local helpers (imported lazily to keep fall-back stubs tiny)
# ────────────────────────────────────────────────────────────────────────────
try:
    from whisperer import build_operators_for_circuit, run_time_iteration
    from helpers.euler_error import euler_error_metric
    from helpers.plots        import generate_plots
    from helpers.tables       import print_summary
except ImportError:
    from .whisperer import build_operators_for_circuit, run_time_iteration
    from .helpers.euler_error import euler_error_metric
    from .helpers.plots        import generate_plots
    from .helpers.tables       import print_summary


# ────────────────────────────────────────────────────────────────────────────
#  configuration utilities
# ────────────────────────────────────────────────────────────────────────────
def load_configs() -> dict:
    cfg_dir = Path(__file__).parent / "config_HR"
    return load_config(cfg_dir)


def patch_cfg(cfg_container: dict, periods: int, vf_ngrid: int) -> dict:
    cfg = copy.deepcopy(cfg_container)
    cfg["master"]["horizon"]                       = periods
    cfg["master"]["settings"]["N_arg_grid_vfi"]    = vf_ngrid

    sol = cfg["master"]["methods"]["upper_envelope"]
    target = sol if sol in ("VFI_HDGRID", "VFI", "VFI_POOL") else "EGM"
    cfg["stages"]["OWNC"]["stage"]["methods"]["solution"] = target
    cfg["stages"]["RNTC"]["stage"]["methods"]["solution"] = target
    return cfg


def init_model(cfg_container: dict, periods: int, vf_ngrid: int):
    cfg = patch_cfg(cfg_container, periods, vf_ngrid)
    mc  = initialize_model_Circuit(
            master_config      = cfg["master"],
            stage_configs      = cfg["stages"],
            connections_config = cfg["connections"])
    compile_all_stages(mc)
    return mc


def solver_factory(args, use_mpi, comm):
    def _solve(mc, recorder=None):
        build_operators_for_circuit(mc, use_mpi=use_mpi, comm=comm)
        final = mc.get_period(len(mc.periods_list)-1)
        for tag in ("OWNC", "RNTC"):
            final.get_stage(tag).status_flags["is_terminal"] = True
        run_time_iteration(mc, n_periods=args.periods,
                           verbose=args.verbose,
                           verbose_timings=args.verbose,
                           recorder=recorder)
        return mc
    return _solve


# ────────────────────────────────────────────────────────────────────────────
#  CLI + main
# ────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
            prog="circuit_runner_solving.py",
            description="Solve baseline and fast UE methods for the HR model")
    p.add_argument("--periods", type=int, default=3)
    p.add_argument("--ue-method", default="ALL")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output-root", default="solutions/HR")
    p.add_argument("--bundle-prefix", default="HR")
    p.add_argument("--vfi-ngrid", default="10000")
    p.add_argument("--mpi", action="store_true")
    p.add_argument("--fresh-fast", action="store_true")
    p.add_argument("--recompute-baseline", action="store_true")
    args = p.parse_args(argv or sys.argv[1:])

    #  MPI communicator
    from dc_smm.helpers.mpi_utils import get_comm
    comm = get_comm(args.mpi)
    if comm.size > 1 and not args.mpi and comm.rank != 0:
        comm.Barrier(); sys.exit(0)

    #  parse grid size
    vf_ngrid = int(float(args.vfi_ngrid))

    #  method list
    all_methods = ["VFI_HDGRID", "FUES", "FUES2DEV", "CONSAV", "DCEGM"]
    methods = all_methods if args.ue_method.upper()=="ALL" \
              else [m.strip().upper() for m in args.ue_method.split(",")]

    #  IO paths
    packroot    = Path.cwd()
    output_root = packroot / f"{args.output_root}_{args.vfi_ngrid}"
    output_root.mkdir(parents=True, exist_ok=True)

    #  set-up runner ------------------------------------------------------
    cfg_container = load_configs()
    runner = CircuitRunner(
        base_cfg      = cfg_container,
        param_paths   = ["master.methods.upper_envelope"],
        model_factory = lambda cfg: init_model(cfg, args.periods, vf_ngrid),
        solver        = solver_factory(args, args.mpi, comm),
        metric_fns    = {
            "euler_error": euler_error_metric,
            "dev_c_L2"   : dev_c_L2,
        },
        output_root   = output_root,
        bundle_prefix = args.bundle_prefix,
        save_by_default = True,
        load_if_exists  = True,
    )

    # --------------------------------------------------------------------
    #  1) make sure baseline is available / solved
    # --------------------------------------------------------------------
    BASE = "VFI_HDGRID"
    if BASE in methods:
        runner.load_if_exists = not args.recompute_baseline
        print("\n» Baseline:", "recompute" if args.recompute_baseline else "load/solve-if-missing")
        base_metrics = runner.run(np.array([BASE], dtype=object))
        need_model   = args.plots and not args.mpi          # only gather model in 1-proc mode
        base_metrics, base_model = runner.run(
                np.array([BASE], dtype=object),
                return_model = need_model)
        base_metrics["master.methods.upper_envelope"] = BASE



    # --------------------------------------------------------------------
    #  2) run fast methods (optionally fresh)
    # --------------------------------------------------------------------
    fast_methods = [m for m in methods if m != BASE]
    if fast_methods:
        runner.load_if_exists = not args.fresh_fast
        xs = np.asarray([[m] for m in fast_methods], dtype=object)
        need_models = args.plots and not args.mpi          # only gather models in single proc
        print(f"\n» Fast methods: {', '.join(fast_methods)}")
        t0 = time.time()
        res_df, models = mpi_map(runner, xs,
                                 mpi=args.mpi, comm=comm,
                                 return_models=need_models)
        print(f"✓ completed in {time.time()-t0:.1f}s")
    else:
        res_df, models = None, None

    # ----------------------------------------------------------
    #  include baseline row (if we just computed / loaded it)
    # ----------------------------------------------------------
    if BASE in methods:
        base_df = pd.DataFrame([base_metrics])
        if res_df is not None:
            res_df = pd.concat([base_df, res_df], ignore_index=True)
        else:
            res_df = base_df


    # --------------------------------------------------------------------
    #  3) summary + plots
    # --------------------------------------------------------------------
    if res_df is not None and len(res_df):
        print_summary(res_df, output_root)

    # prepend baseline model/name if we have one
    if args.plots and not args.mpi:
        models   = [base_model] + models
        plot_seq = [BASE]       + fast_methods
    else:
        plot_seq = fast_methods

    if args.plots and models is not None:
        img_dir = output_root / "images"
        img_dir.mkdir(exist_ok=True)
        for m_obj, m_name in zip(models, plot_seq):
            try:
                generate_plots(m_obj, m_name, img_dir)
            except Exception as err:
                print(f"[warn] plot-gen for {m_name} failed: {err}")

if __name__ == "__main__":
    main()
