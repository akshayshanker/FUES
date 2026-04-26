"""Run the durables DDSL pipeline via kikku (``parse_cli`` + ``sweep``)."""

from __future__ import annotations

import os

import numpy as np

from kikku.run import parse_cli, sweep
from kikku.run.types import TestSpec

from examples._mpi import get_mpi_comm as _get_mpi_comm
from .horses.simulate import (
    evaluate_euler_c,
    evaluate_euler_h,
    simulate_lifecycle,
)
from .outputs import (
    compute_euler_stats,
    get_timing,
    print_euler_stats,
    derive_savings,
    write_outputs,
)
from .solve import read_scheme_method, solve


def _parse_plot_ages(raw):
    """Convert plot_ages setting (list, string, int, or empty) to list or None."""
    if isinstance(raw, str):
        return [int(a) for a in raw.split(",") if a.strip()]
    if isinstance(raw, (int, float)):
        return [int(raw)]
    if isinstance(raw, list) and raw:
        return [int(a) for a in raw]
    return None


def main() -> None:
    run = parse_cli(
        name="durables",
        base_spec="examples/durables/mod/separable",
        modes=["compare", "sweep", "simulate"],
        output="results/durables",
    )
    comm = _get_mpi_comm()
    is_root = comm is None or comm.Get_rank() == 0
    if is_root:
        print(f"Output directory: {run.output_dir}")

    def solve_test(t: TestSpec) -> dict:
        nest, grids = solve(str(run.base_spec), **t.slots, verbose=False)
        adj0 = nest["periods"][0]["stages"]["adjuster_cons"]
        method_label = read_scheme_method(adj0, "upper_envelope")
        if is_root:
            print(f'{len(nest["solutions"])} periods solved')
        timing = get_timing(nest)
        if is_root:
            print(
                f'Mean timing — keeper: {timing["keeper_ms"]:.1f}ms, '
                f'adj: {timing["adj_ms"]:.1f}ms'
            )
        out: dict = {
            "nest": nest,
            "grids": grids,
            "method_label": method_label,
            "timing": timing,
        }
        stage0 = nest["periods"][0]["stages"]["keeper_cons"]
        store_cntn = bool(int(stage0.settings.get("store_cntn", 0)))
        plot_ages = _parse_plot_ages(
            stage0.calibration.get("plot_ages", [])
        )
        if plot_ages is None:
            all_t = sorted(s["t"] for s in nest["solutions"])
            plot_ages = [all_t[-3] if len(all_t) >= 3 else all_t[-1]]
        use_empirical_init = (
            stage0.calibration.get("init_method", "standard") == "empirical"
        )
        base_dir = str(run.output_dir)
        plots_dir = os.path.join(base_dir, "plots")
        if is_root:
            os.makedirs(plots_dir, exist_ok=True)
        tau = float(stage0.calibration["tau"])
        savings = derive_savings(nest, grids, tau)
        if is_root and plot_ages:
            from .outputs.plots import plot_grids, plot_policies

            for age in plot_ages:
                all_t = sorted(s["t"] for s in nest["solutions"])
                if age not in all_t:
                    if is_root:
                        print(
                            f"Age {age} not in solution "
                            f"(range {all_t[0]}–{all_t[-1]}), skipping."
                        )
                    continue
                plot_policies(
                    nest, grids, savings, output_dir=plots_dir, plot_t=age
                )
                if store_cntn:
                    plot_grids(nest, grids, output_dir=plots_dir, plot_t=age)
            if is_root and plot_ages:
                print(f"Plots saved to {plots_dir}/")

        if run.sim is not None:
            sim_data = simulate_lifecycle(
                nest,
                grids,
                N=run.sim.n_sim,
                seed=run.sim.seed,
                use_empirical_init=use_empirical_init,
            )
            euler_c = evaluate_euler_c(sim_data, nest, grids)
            euler_h = evaluate_euler_h(sim_data, nest, grids)
            d = sim_data["discrete"]
            ec_stats = compute_euler_stats(euler_c, d)
            eh_stats = compute_euler_stats(euler_h, d)
            out["euler_c_stats"] = ec_stats
            out["euler_h_stats"] = eh_stats
            if is_root:
                print("Consumption Euler (c FOC):")
                print_euler_stats(ec_stats)
                print("\nHousing FOC (adjusters):")
                print_euler_stats(eh_stats)
            if is_root and "npv_utility" in sim_data:
                npv = sim_data["npv_utility"]
                print(
                    f"  NPV utility: mean={np.mean(npv):.4f}, "
                    f"std={np.std(npv):.4f}"
                )
            if is_root and run.sim.plots:
                from .outputs.plots import plot_lifecycle

                plot_lifecycle(sim_data, euler_c, nest, output_dir=plots_dir)
            if "combined" in ec_stats:
                out["euler_c_all"] = ec_stats["combined"]["mean"]
                out["euler_c_keeper"] = ec_stats["keeper"]["mean"]
                out["euler_c_adj"] = ec_stats["adjuster"]["mean"]
            else:
                out["euler_c_all"] = float("nan")
                out["euler_c_keeper"] = float("nan")
                out["euler_c_adj"] = float("nan")
            if "combined" in eh_stats:
                out["euler_h_keeper"] = eh_stats["keeper"]["mean"]
                out["euler_h_adj"] = eh_stats["adjuster"]["mean"]
                out["euler_h_all"] = eh_stats["combined"]["mean"]
            else:
                out["euler_h_keeper"] = float("nan")
                out["euler_h_adj"] = float("nan")
                out["euler_h_all"] = float("nan")
            out["adj_rate"] = float(np.mean(d[d >= 0]) * 100)
        return out

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

    metric_fns: dict = {
        "solve_ms": lambda r: r["timing"]["solve_time"] * 1000,
        "keeper_ms": lambda r: r["timing"]["keeper_ms"],
        "adj_ms": lambda r: r["timing"]["adj_ms"],
        "method": lambda r: r["method_label"],
        "n_a": _m_n_a,
        "tau": _m_tau,
    }
    if run.sim is not None:
        metric_fns.update(
            {
                "euler_c_mean": lambda r: r.get("euler_c_all", np.nan),
                "euler_c_keeper": lambda r: r.get("euler_c_keeper", np.nan),
                "euler_c_adjuster": lambda r: r.get("euler_c_adj", np.nan),
                "euler_h_mean": lambda r: r.get("euler_h_all", np.nan),
                "euler_h_keeper": lambda r: r.get("euler_h_keeper", np.nan),
                "euler_h_adjuster": lambda r: r.get("euler_h_adj", np.nan),
                "adj_rate": lambda r: r.get("adj_rate", np.nan),
            }
        )

    n_reps_use = 1 if run.mode == "single" else run.sweep_runs
    warmup_use = False if run.mode == "single" else run.warmup
    results = sweep(
        solve_test,
        list(run.test_set),
        metric_fns,
        n_reps=n_reps_use,
        warmup=warmup_use,
        best="min",
        on_error="raise",
        comm=comm,
        verbose=run.verbose,
    )

    if is_root and results:
        write_outputs(results, run)


if __name__ == "__main__":
    main()
