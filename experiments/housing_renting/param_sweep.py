#!/usr/bin/env python
"""
housing_renting_experiment.py
=============================

Runs the housing-with-renting model over a 5 × 5 uniform grid of
(`policy.beta`, `utility.gamma`) values and all three upper-envelope
methods (`FUES`, `CONSAV`, `DCEGM`).  Metrics are averaged per method
and printed at the end.

Requires DynX ≥ 1.6.12.
"""

from __future__ import annotations

import os
import time
import copy
import argparse
import logging
from pathlib import Path
from typing import Any
import datetime

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
    parser.add_argument(
        "--ue-method",
        type=str,
        choices=["FUES", "CONSAV", "DCEGM", "ALL"],
        default="ALL",
        help="Upper-envelope method to use",
    )
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

    param_paths = [
        "master.methods.upper_envelope",  # categorical
        "master.parameters.beta",         # numeric
        "master.parameters.gamma",        # numeric
    ]

    # ---- build design ------------------------------------------------
    beta_vals = np.linspace(0.90, 0.99, 2).round(4).tolist()
    gamma_vals = np.linspace(1.5, 5.0, 2).round(3).tolist()

    # Decide which methods to run, based on CLI flag
    methods = (
        ["FUES", "CONSAV", "DCEGM"]    # --ue-method ALL
        if args.ue_method == "ALL"
        else [args.ue_method]          # --ue-method FUES  (or CONSAV, DCEGM)
    )

    # Build a properly shaped array with placeholders for other parameters
    rows = np.full((len(methods), 3), np.nan, dtype=object)
    rows[:, 0] = methods    # First column = method names (cols 1-2 left as NaN)

    samplers = [
        FixedSampler(rows),                  # provides categorical column
        FullGridSampler({
            "master.parameters.beta": beta_vals,
            "master.parameters.gamma": gamma_vals,
        }),
    ]
    Ns = [None, None]  # Fixed rows + full grid

    meta = {
        # only the chosen methods
        "master.methods.upper_envelope": {"enum": methods},
        "master.parameters.beta":  {"values": beta_vals},
        "master.parameters.gamma": {"values": gamma_vals},
    }

    xs, _ = build_design(param_paths, samplers, Ns, meta, seed=0)
    expected_rows = len(methods) * len(beta_vals) * len(gamma_vals)
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
    print(f"\nRunning parameter sweep with {len(methods)} method(s): {', '.join(methods)}")
    print(f"Grid: {len(beta_vals)}×{len(gamma_vals)} (beta × gamma) = {expected_rows} combinations")
    print(f"Total runs: {xs.shape[0]}")
    
    tic = time.time()
    df, models = mpi_map(runner, xs, mpi=False, return_models=True)
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
            "beta_vals": beta_vals,
            "gamma_vals": gamma_vals,
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
    
    return {
        "summary": result_metrics,
        "detailed": df,
        "models": models,
        "parameters": {
            "methods": methods,
            "beta_vals": beta_vals,
            "gamma_vals": gamma_vals,
            "periods": args.periods
        }
    }


if __name__ == "__main__":
    main()
