"""Run the durables DDSL pipeline via kikku (``parse_cli`` + ``sweep``).

Pipeline visible in ``main``::

    parse_cli  →  make_solve_test  →  make_metric_fns  →  sweep  →  write_outputs

Per-θ kernel (inside ``make_solve_test``)::

    solve  →  simulate?  →  render_plots?  →  pack
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

from kikku.run import parse_cli, sweep
from kikku.run.types import RunSpec, TestSpec

from examples._mpi import get_mpi_comm as _get_mpi_comm
from .horses.simulate import simulate_one
from .outputs import (
    derive_savings,
    get_timing,
    write_outputs,
)
from .solve import read_scheme_method, solve


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _parse_plot_ages(raw):
    """Convert plot_ages setting (list, string, int, or empty) to list or None."""
    if isinstance(raw, str):
        return [int(a) for a in raw.split(",") if a.strip()]
    if isinstance(raw, (int, float)):
        return [int(raw)]
    if isinstance(raw, list) and raw:
        return [int(a) for a in raw]
    return None


def _method_label(nest) -> str:
    """Read the upper-envelope method tag from the solved adjuster_cons stage."""
    return read_scheme_method(
        nest["periods"][0]["stages"]["adjuster_cons"], "upper_envelope"
    )


def _m_n_a(r: dict) -> float:
    return float(
        r["nest"]["periods"][0]["stages"]["keeper_cons"].settings.get(
            "n_a", np.nan
        )
    )


def _m_tau(r: dict) -> float:
    return float(
        r["nest"]["periods"][0]["stages"]["keeper_cons"].calibration.get(
            "tau", float("nan")
        )
    )


# ---------------------------------------------------------------------------
# Plotting (rank-0 only — caller gates)
# ---------------------------------------------------------------------------


def render_plots(run: RunSpec, nest, grids, result: dict) -> None:
    """Plot per-age policies/grids and (optionally) lifecycle. Caller is rank-0."""
    stage0 = nest["periods"][0]["stages"]["keeper_cons"]
    plot_ages = _parse_plot_ages(stage0.calibration.get("plot_ages", []))
    if plot_ages is None:
        all_t = sorted(s["t"] for s in nest["solutions"])
        plot_ages = [all_t[-3] if len(all_t) >= 3 else all_t[-1]]

    all_t = sorted(s["t"] for s in nest["solutions"])
    store_cntn = bool(int(stage0.settings.get("store_cntn", 0)))
    plots_dir = os.path.join(str(run.output_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    tau = float(stage0.calibration["tau"])
    savings = derive_savings(nest, grids, tau)

    from .outputs.plots import plot_grids, plot_policies

    for age in plot_ages:
        if age not in all_t:
            continue
        plot_policies(nest, grids, savings, output_dir=plots_dir, plot_t=age)
        if store_cntn:
            plot_grids(nest, grids, output_dir=plots_dir, plot_t=age)

    if run.sim is not None and run.sim.plots and "sim_data" in result:
        from .outputs.plots import plot_lifecycle

        # plot_lifecycle wants the consumption-Euler array; re-derive from sim_data.
        from .horses.simulate import evaluate_euler_c

        euler_c = evaluate_euler_c(result["sim_data"], nest, grids)
        plot_lifecycle(result["sim_data"], euler_c, nest, output_dir=plots_dir)


# ---------------------------------------------------------------------------
# Builders: per-θ kernel and metric functions
# ---------------------------------------------------------------------------


def make_solve_test(run: RunSpec, comm) -> Callable[[TestSpec], dict]:
    """Per-θ kernel: ``solve → simulate? → render_plots? → pack``."""
    is_root = comm is None or comm.Get_rank() == 0

    def solve_test(t: TestSpec) -> dict:
        nest, grids = solve(str(run.base_spec), **t.slots, verbose=False)
        result: dict = {
            "nest":         nest,
            "grids":        grids,
            "timing":       get_timing(nest),
            "method_label": _method_label(nest),
        }
        if run.sim is not None:
            result.update(simulate_one(nest, grids, run.sim))
        if is_root:
            render_plots(run, nest, grids, result)
        return result

    return solve_test


def make_metric_fns(with_sim: bool) -> dict:
    """Sweep metric functions; sim-only metrics added when ``with_sim``."""
    fns = {
        "solve_ms":  lambda r: r["timing"]["solve_time"] * 1000,
        "keeper_ms": lambda r: r["timing"]["keeper_ms"],
        "adj_ms":    lambda r: r["timing"]["adj_ms"],
        "method":    lambda r: r["method_label"],
        "n_a":       _m_n_a,
        "tau":       _m_tau,
    }
    if with_sim:
        fns.update({
            "euler_c_mean":     lambda r: r.get("euler_c_all",    np.nan),
            "euler_c_keeper":   lambda r: r.get("euler_c_keeper", np.nan),
            "euler_c_adjuster": lambda r: r.get("euler_c_adj",    np.nan),
            "euler_h_mean":     lambda r: r.get("euler_h_all",    np.nan),
            "euler_h_keeper":   lambda r: r.get("euler_h_keeper", np.nan),
            "euler_h_adjuster": lambda r: r.get("euler_h_adj",    np.nan),
            "adj_rate":         lambda r: r.get("adj_rate",       np.nan),
        })
    return fns


# ---------------------------------------------------------------------------
# main — thin combinator
# ---------------------------------------------------------------------------


def main() -> None:
    run = parse_cli(
        name="durables",
        base_spec="examples/durables/mod/separable",
        modes=["compare", "sweep", "simulate"],
        output="results/durables",
    )
    comm    = _get_mpi_comm()
    is_root = comm is None or comm.Get_rank() == 0
    if is_root:
        print(f"Output directory: {run.output_dir}")

    solve_test = make_solve_test(run, comm)
    metric_fns = make_metric_fns(with_sim=run.sim is not None)

    results = sweep(
        solve_test, list(run.test_set), metric_fns,
        n_reps=(1 if run.mode == "single" else run.sweep_runs),
        warmup=(False if run.mode == "single" else run.warmup),
        best="min", on_error="raise", comm=comm, verbose=run.verbose,
    )

    if is_root and results:
        write_outputs(results, run)


if __name__ == "__main__":
    main()
