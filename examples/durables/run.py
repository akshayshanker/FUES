"""Run the durables DDSL pipeline via kikku v3 (``parse_cli`` + ``sweep``)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

from kikku.run import parse_cli, sweep, write_results_table
from kikku.run.sweep import SweepResult
from kikku.run.types import RunSpec, TestSpec

from examples._mpi import get_mpi_comm as _get_mpi_comm
from .horses.simulate import (
    evaluate_euler_c,
    evaluate_euler_h,
    simulate_lifecycle,
)
from .outputs import (
    compute_euler_stats,
    generate_comparison_table,
    generate_sweep_table,
    get_timing,
    print_euler_stats,
    derive_savings,
    write_euler_detail,
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


def _draw_cal_and_settings(
    t: TestSpec, base_calib: dict, base_settings: dict
) -> tuple[dict, dict]:
    """Map ``$draw`` slot to calibration/settings overlays (flat or tier-wrapped)."""
    d = t.slots.get("draw", {}) or {}
    if d and set(d) <= {"calibration", "settings", "methods"}:
        return (d.get("calibration") or {}), (d.get("settings") or {})
    c: dict = {}
    s: dict = {}
    for k, v in d.items():
        if k in base_calib:
            c[k] = v
        if k in base_settings:
            s[k] = v
    return c, s


def _method_label_from_testspec(t: TestSpec) -> str | None:
    ms = t.slots.get("method_switch")
    if not ms:
        return None
    if isinstance(ms, str):
        return ms
    if isinstance(ms, dict):
        methods = ms.get("methods") or []
        tags: set[str] = set()
        for ent in methods:
            for sch in (ent or {}).get("schemes", []) or []:
                m = sch.get("method")
                if m is not None and m != "":
                    tags.add(str(m))
        if len(tags) == 1:
            return next(iter(tags))
    return None


def _load_baseline_calib(base: Path) -> dict:
    cands = [base / "calibration.yaml", base / "calibration" / "main.yaml"]
    for p in cands:
        if p.is_file():
            raw = yaml.safe_load(p.read_text()) or {}
            if "calibration" in raw and isinstance(raw["calibration"], dict):
                return dict(raw["calibration"])
            return dict(raw) if raw else {}
    return {}


def _load_baseline_settings(base: Path) -> dict:
    cands = [base / "settings.yaml", base / "settings" / "default.yaml"]
    for p in cands:
        if p.is_file():
            raw = yaml.safe_load(p.read_text()) or {}
            if "settings" in raw and isinstance(raw["settings"], dict):
                return dict(raw["settings"])
            return dict(raw) if raw else {}
    return {}


def _write_durables_latex(results: list[SweepResult], run: RunSpec) -> None:
    """Bespoke per-row LaTeX summary (sweep) — rank-0 caller only."""
    if not results:
        return
    base = Path(run.base_spec)
    base_calib = _load_baseline_calib(base)
    base_st_file = _load_baseline_settings(base)
    tdir = os.path.join(str(run.output_dir), "tables")
    os.makedirs(tdir, exist_ok=True)
    n_sim = run.sim.n_sim if run.sim is not None else 0
    summaries = []
    for sr in results:
        row = sr.metrics
        t = sr.point
        p_ovl, s_ovl = _draw_cal_and_settings(t, base_calib, base_st_file)
        t_settings = {**base_st_file, **s_ovl}
        cal_params = {
            k: base_calib[k]
            for k in (
                "beta",
                "gamma_c",
                "gamma_h",
                "alpha",
                "delta",
                "R",
                "R_H",
                "phi_w",
                "sigma_w",
            )
            if k in base_calib
        }
        cal_params = {**cal_params, **p_ovl}
        if "R" in cal_params:
            cal_params["r"] = cal_params.pop("R")
        if "R_H" in cal_params:
            cal_params["r_H"] = cal_params.pop("R_H")
        cal_params["N_sim"] = n_sim
        summaries.append(
            {
                "Grid_Size": int(
                    row.get("n_a", t_settings.get("n_a", 0))
                ),
                "Tau": float(
                    row.get("tau", p_ovl.get("tau", cal_params.get("tau", 0.0)))
                ),
                "Method": row.get("method", ""),
                "Avg_Keeper_ms": row.get("keeper_ms", 0.0),
                "Avg_Adj_ms": row.get("adj_ms", 0.0),
                "Euler_Combined": row.get("euler_c_mean", np.nan),
                "Euler_Keeper": row.get("euler_c_keeper", np.nan),
                "Euler_Adjuster": row.get("euler_c_adjuster", np.nan),
                "Euler_H_Adjuster": row.get("euler_h_mean", np.nan),
                **cal_params,
            }
        )
    tex = generate_sweep_table(
        summaries, fmt="tex", caption="Durables Model: Per-Period Timing and Accuracy"
    )
    preamble = (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\begin{document}\n"
        "\\pagestyle{empty}\n"
    )
    with open(os.path.join(tdir, "sweep.tex"), "w", encoding="utf-8") as f:
        f.write(preamble)
        f.write(tex)
        f.write("\n\\end{document}\n")


def _comparison_rows_from_sweep(
    results: list[SweepResult], run: RunSpec
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for sr in results:
        t = sr.point
        m = dict(sr.metrics)
        label = t.label
        if not label:
            label = m.get("method", "")
        row = {
            "Method": label,
            "Keeper (ms)": m.get("keeper_ms", 0.0),
            "Adj (ms)": m.get("adj_ms", 0.0),
            "Total (ms)": m.get("keeper_ms", 0.0) + m.get("adj_ms", 0.0),
        }
        for k, outk in [
            ("euler_c_keeper", "Euler c (keeper)"),
            ("euler_c_adjuster", "Euler c (adj)"),
            ("euler_c_mean", "Euler c (all)"),
            ("euler_h_keeper", "Euler h (keeper)"),
            ("euler_h_adjuster", "Euler h (adj)"),
            ("euler_h_mean", "Euler h (all)"),
            ("adj_rate", "Adj Rate"),
        ]:
            if k in m and not (isinstance(m[k], float) and m[k] != m[k]):
                row[outk] = m[k]
        rows.append(row)
    base = Path(run.base_spec)
    base_calib = _load_baseline_calib(base)
    base_st = _load_baseline_settings(base)
    t0 = results[0].point if results else None
    p0, _s0 = (
        _draw_cal_and_settings(t0, base_calib, base_st) if t0 is not None else ({}, {})
    )
    first = {**base_calib, **p0}
    return rows, first


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

    _reg_base = Path(run.base_spec)
    _base_cal0 = _load_baseline_calib(_reg_base)
    _base_st0 = _load_baseline_settings(_reg_base)

    def solve_test(t: TestSpec) -> dict:
        nest, grids = solve(str(run.base_spec), **t.slots, verbose=False)
        adj0 = nest["periods"][0]["stages"]["adjuster_cons"]
        method_label = _method_label_from_testspec(
            t
        ) or read_scheme_method(adj0, "upper_envelope")
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
        _, st_draw = _draw_cal_and_settings(t, _base_cal0, _base_st0)
        store_cntn = bool(
            int({**_base_st0, **st_draw}.get("store_cntn", 0))
        )
        stage0 = nest["periods"][0]["stages"]["keeper_cons"]
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

    if not is_root or not results:
        return

    tdir = os.path.join(str(run.output_dir), "tables")
    os.makedirs(tdir, exist_ok=True)
    write_results_table(
        results, os.path.join(tdir, "sweep.md"), fmt="markdown"
    )

    if run.mode == "compare":
        rows, params = _comparison_rows_from_sweep(results, run)
        from kikku.run.metrics import format_table

        if rows:
            print("\n" + format_table(rows, list(rows[0].keys())))
        md_table = generate_comparison_table(
            rows, fmt="md", caption="Durables Model Comparison", params=params
        )
        with open(
            os.path.join(tdir, "comparison.md"), "w", encoding="utf-8"
        ) as f:
            f.write(md_table)
        tex_table = generate_comparison_table(
            rows, fmt="tex", caption="Durables Model Comparison", params=params
        )
        with open(
            os.path.join(tdir, "comparison.tex"), "w", encoding="utf-8"
        ) as f:
            f.write(tex_table)
        print(f"Tables saved to {tdir} (comparison.md, comparison.tex)")
        if run.sim is not None:
            euler_by_label = {}
            for r in results:
                if r.result is None:
                    continue
                lab = r.point.label
                if not lab:
                    lab = r.result.get("method_label", "")
                euler_by_label[lab] = {
                    "consumption": r.result.get("euler_c_stats", {}),
                    "housing": r.result.get("euler_h_stats", {}),
                }
            if euler_by_label:
                p = write_euler_detail(euler_by_label, tdir)
                print(f"Euler detail saved to {p}")
    else:
        from kikku.run.metrics import format_table

        if len(results) == 1:
            sr = results[0]
            r0 = sr.result
            m = {**sr.metrics}
            if r0 is not None and run.sim is not None and "euler_c_all" in r0:
                m["Euler c (all)"] = r0.get("euler_c_all", np.nan)
            row = {
                "Method": m.get("method", ""),
                "Keeper (ms)": m.get("keeper_ms", 0.0),
                "Adj (ms)": m.get("adj_ms", 0.0),
                "Total (ms)": m.get("keeper_ms", 0.0) + m.get("adj_ms", 0.0),
            }
            for k, outk in [
                ("euler_c_keeper", "Euler c (keeper)"),
                ("euler_c_adjuster", "Euler c (adj)"),
                ("euler_c_mean", "Euler c (all)"),
                ("euler_h_keeper", "Euler h (keeper)"),
                ("euler_h_adjuster", "Euler h (adj)"),
                ("euler_h_mean", "Euler h (all)"),
                ("adj_rate", "Adj Rate"),
            ]:
                if k in m:
                    row[outk] = m[k]
            print("\n" + format_table([row], list(row.keys())))
        if run.mode == "sweep" and len(results) > 1:
            _write_durables_latex(results, run)


if __name__ == "__main__":
    main()
