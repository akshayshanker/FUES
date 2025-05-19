#!/usr/bin/env python
"""
housing_renting_experiment.py
=============================

Generic parameter sweep for the housing-with-renting model.
All model parameters are provided through `--param` (list or range
syntax) and one or more upper-envelope methods are selected with
`--ue-method / --ue-methods`.  Metrics are averaged per method and
printed at the end.

Requires DynX ≥ 1.6.12.
"""



import os
import time
import copy
import argparse
import logging
import sys
import functools
import operator
from pathlib import Path
from typing import Any
import datetime
import re
import numpy as np
import pandas as pd
from dynx.runner import CircuitRunner, mpi_map
from dynx.runner.sampler import FixedSampler, FullGridSampler, build_design
from tabulate import tabulate

# ---------------------------------------------------------------------
# Local helpers and model-specific utilities  (from examples directory)
# ---------------------------------------------------------------------

from examples.housing_renting.helpers.euler_error import euler_error_metric
from examples.housing_renting.whisperer import run_time_iteration
# Import from the appropriate module where these functions are defined
from examples.housing_renting.circuit_runner_solving import load_configs, initialize_housing_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Solver wrapper
# ---------------------------------------------------------------------
def solver(model_circuit: Any, *, recorder=None):
    """Time-iteration solver with terminal flags already set."""
    final_prd = model_circuit.get_period(len(model_circuit.periods_list) - 1)
    final_prd.get_stage("OWNC").status_flags["is_terminal"] = True
    final_prd.get_stage("RNTC").status_flags["is_terminal"] = True
    run_time_iteration(
        model_circuit,
        verbose=False,
        verbose_timings=False,
        recorder=recorder,
    )
    return model_circuit


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Housing model with CircuitRunner")

    # -----------------------------------------------------------------
    # Parallel execution
    # -----------------------------------------------------------------
    parser.add_argument("--use-mpi",  action="store_true",
                        help="Run the sweep with MPI (mpi4py)")
    parser.add_argument("--n-procs",  type=int, default=None,
                        help="Hint for the number of MPI ranks; "
                             "only informative – `mpiexec` still controls "
                             "the actual process count.")
    parser.add_argument(
        "--periods",
        type=int,
        default=3,
        help="Number of periods to simulate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (CSV)",
    )

    # ------------------------------------------------------------
    # Upper-envelope method(s)
    #   • --ue-methods  FUES,CONSAV
    #   • --ue-method   FUES   --ue-method CONSAV
    # ------------------------------------------------------------
    parser.add_argument("--ue-methods", type=str, default=None,
                        help="Comma-separated list of UE methods (FUES,CONSAV,DCEGM)")
    parser.add_argument("--ue-method",  action="append", dest="ue_methods_list",
                        default=[],
                        help="Repeatable flag to specify UE method(s)")

    # ------------------------------------------------------------
    # Generic extra parameters
    # ------------------------------------------------------------
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="PATH=VALS|min:max:N",
        help=(
            "Add a parameter to the sweep. Either give an explicit list "
            "(`…=0.1,0.2,0.3`) or a range `…=min:max:N` which expands to "
            "N equally spaced points.  Repeat --param for multiple entries."
        ),
    )

    args = parser.parse_args(argv)

    cfgs = load_configs()  # helper from previous script

    # ---- baseline config assembled into a single dict ----------------
    base_cfg = {
        "master": cfgs["master"],
        "stages": {
            "OWNH": cfgs["ownh"],
            "OWNC": cfgs["ownc"],
            "RNTH": cfgs["renth"],
            "RNTC": cfgs["rentc"],
            "TENU": cfgs["tenu"],
        },
        "connections": cfgs["connections"],
    }

    # -----------------------------------------------------------------
    # Build dictionary   path → [values …]   (only from --param now)
    # -----------------------------------------------------------------
    param_vals_map: dict[str, list] = {}

    def _num_or_str(s: str):
        try:
            return float(s)
        except ValueError:
            return s

    range_pat = re.compile(r"^([^:]+):([^:]+):([^:]+)$")

    for spec in args.param:
        if "=" not in spec:
            raise ValueError(f"Malformed --param '{spec}'. Expected PATH=…")
        path, rhs = spec.split("=", 1)
        path = path.strip()

        # range syntax PATH=min:max:N
        m = range_pat.match(rhs)
        if m:
            lo, hi, n = float(m.group(1)), float(m.group(2)), int(m.group(3))
            vals = np.linspace(lo, hi, n).tolist()
        else:
            # list syntax PATH=v1,v2,…
            vals = [_num_or_str(v) for v in rhs.split(",") if v.strip()]
            if not vals:
                raise ValueError(f"No values supplied for --param '{spec}'.")

        param_vals_map[path] = vals

    # Final list of parameter paths (upper-envelope + whatever user supplied)
    # -----------------------------------------------------------------
    #  UE-method CLI handling  (unchanged)
    # -----------------------------------------------------------------
    _cli_methods = (
        args.ue_methods.split(",") if args.ue_methods else []
    ) + (args.ue_methods_list or [])
    _cli_methods = [m.strip().upper() for m in _cli_methods if m.strip()]

    methods = (
        ["FUES", "CONSAV", "DCEGM"]        # default set
        if not _cli_methods or "ALL" in _cli_methods
        else sorted(set(_cli_methods))
    )


    # -----------------------------------------------------------------
    #  Build a single Cartesian grid = methods  ×  every --param
    # -----------------------------------------------------------------
    param_vals_map_full = {
            "master.methods.upper_envelope": methods,   # e.g. ['FUES']
            **param_vals_map,                          # β, γ, …
                }

    #print(param_vals_map_full)

    samplers    = [FullGridSampler(param_vals_map_full)]
    Ns          = [None]                 # placeholder required by build_design
    param_paths = list(param_vals_map_full)

    # meta tells build_design which paths are numeric vs categorical
    meta = {
    }
    
    print(meta)
    xs, _ = build_design(param_paths, samplers, Ns, meta, seed=0)
    log.info("Design matrix built: %s rows", xs.shape[0])




    # ---- create runner ----------------------------------------------
    runner = CircuitRunner(
        base_cfg=base_cfg,
        param_paths=param_paths,
        model_factory=lambda cfg: initialize_housing_model(
            cfg["master"], cfg["stages"], cfg["connections"], n_periods=args.periods
        ),
        solver=solver,
        metric_fns={"euler_error": euler_error_metric},
        cache=True,
    )

    # ---- run sweep ---------------------------------------------------
    print(f"\nRunning parameter sweep with {len(methods)} method(s): "
          f"{', '.join(methods)}")
    if param_vals_map:
        dims = ", ".join(f"{Path(p).name}:{len(v)}"
                         for p, v in param_vals_map.items())
        print(f"Grid dims : {dims}")
    print(f"Total runs : {xs.shape[0]}")
    print(f"MPI active : {args.use_mpi}")
    
    if args.use_mpi and args.n_procs:
        if xs.shape[0] != args.n_procs:
            log.error(
                "Number of parameter draws (%d) must equal --n-procs (%d) "
                "when --use-mpi is specified!",
                xs.shape[0], args.n_procs
            )
            sys.exit(1)
    
    tic = time.time()
    if args.use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    else:
        comm = None
    result = mpi_map(runner, xs,
                         mpi=args.use_mpi,
                         comm=comm,
                         return_models=True)

    if comm.Get_rank() == 0:                        # only master has something
        df, models = result              # <- safe to unpack
        # … continue with tables / plots / file output …
    else:
        return                            # or just pass
    toc = time.time()
    log.info("Sweep finished in %.1f s", toc - tic)

    # ---- aggregate & show -------------------------------------------
    # 1. which numeric metrics exist?
    metric_cols = [
        c for c in df.columns
        if c.startswith("total_")
        or c.endswith("_time")
        or c == "euler_error"
        or c.endswith("_percent")
    ]

    # 2. average per method
    agg = (
        df.groupby("master.methods.upper_envelope")[metric_cols]
        .mean()
        .reset_index()
    )

    # 3. pretty column names
    col_map = {
        "master.methods.upper_envelope": "Method",
        "euler_error"            : "Euler Error",
        "total_solution_time"    : "Total Time (s)",
        "total_nonterminal_time" : "Non-Terminal Time (s)",
        "total_ue_time"          : "UE Time (s)",
        "ue_time_percent"        : "UE Time (%)",
    }
    agg = agg.rename(columns=col_map)

    # 4. desired presentation order
    display_order = [
        "Method",
        "Euler Error",
        "Total Time (s)",
        "Non-Terminal Time (s)",
        "UE Time (s)",
        "UE Time (%)",
    ]
    present_cols = [c for c in display_order if c in agg.columns]
    agg = agg[present_cols]

    # 5. number formatting
    for c in agg.columns:
        if c == "Method":
            continue
        if "Time" in c and "%" not in c:                 # seconds
            agg[c] = agg[c].map("{:.4f}".format)
        elif c.endswith("(%)"):                          # percentages
            agg[c] = agg[c].map("{:.2f}".format)
        else:                                            # euler error etc.
            agg[c] = agg[c].map("{:.6f}".format)

    # 6. print
    print("\nPerformance Summary by Method")
    print("=" * 80)
    print(tabulate(agg, headers="keys", tablefmt="grid", showindex=False))
    print("=" * 80)

    

    result_metrics = {
        "summary": agg,
        "detailed": df,
        "models": models,
        "parameters": {
            "methods": methods,
            "periods": args.periods
        }
    }
    
    
    # Save detailed results if output file specified
    if args.output:
        output_file = args.output
    else:
        # Default filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"housing_sweep_{timestamp}.csv"
    
    # Save full results to CSV
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to '{output_file}'")

    # ------------------------------------------------------------------
    # Save LaTeX summary table in the *same* folder as the CSV
    # ------------------------------------------------------------------
    out_path  = Path(output_file).expanduser().resolve()
    tex_file  = out_path.with_suffix(".tex")          # same name, .tex extension

    # pandas ≥ 1.3 lets you write a nicely formatted tabularx-friendly table
    agg.to_latex(
        tex_file,
        index=False,
        caption="Performance summary by upper–envelope method",
        label="tab:fues_summary",
        column_format="l" + "r" * (agg.shape[1] - 1)  # left-align 1st, right others
    )

    print(f"LaTeX summary saved to '{tex_file}'")
    
    return {
        "summary": result_metrics,
        "detailed": df,
        "models": models,
        "parameters": {
            "methods": methods,
            "periods": args.periods
        }
    }


if __name__ == "__main__":
    main()
