#!/usr/bin/env python3
"""Run retirement model experiments via the canonical kikku v3 pipeline."""

from __future__ import annotations

import os
import sys
from dataclasses import replace

from kikku.run import parse_cli, sweep
from kikku.run.sweep import SweepResult
from kikku.run.types import RunSpec, TestSpec

from examples._mpi import get_mpi_comm as _get_mpi_comm
from examples.retirement import benchmark as rbench
from examples.retirement.outputs import (
    consumption_deviation,
    euler,
    get_policy,
    get_timing,
)
from examples.retirement.solve import (
    METHOD_SHORTCUT,
    expand_method_shortcut,
    solve_nest,
)

UE_METHODS = ("RFC", "FUES", "DCEGM", "CONSAV")

RE_EXTRA = {
    "--latex-grids": {
        "type": str,
        "default": None,
        "help": "Comma list of grid sizes for LaTeX timing/accuracy table output.",
    },
}


def _dkey(x: float) -> float:
    return round(float(x), 10)


def _draw_cal_settings(
    t: TestSpec, base_c: dict, base_s: dict
) -> tuple[dict, dict]:
    d = t.slots.get("draw", {}) or {}
    if d and set(d) <= {"calibration", "settings", "methods"}:
        return (d.get("calibration") or {}), (d.get("settings") or {})
    p: dict = {}
    s: dict = {}
    for k, v in d.items():
        if k in base_c:
            p[k] = v
        if k in base_s:
            s[k] = v
    return p, s


def _row_delta(t: TestSpec, base_c: dict) -> float:
    p, _ = _draw_cal_settings(t, base_c, {})
    return _dkey(p.get("delta", base_c.get("delta", 1.0)))


def _expand_default_ue_grid(run: RunSpec) -> RunSpec:
    """If every row has no method_switch, fan out across UE_METHODS (spec §6.2)."""
    if any(
        "method_switch" in t.slots and t.slots.get("method_switch")
        for t in run.test_set
    ):
        return run
    new: list[TestSpec] = []
    for t in run.test_set:
        for m in UE_METHODS:
            new_slots = {
                **t.slots,
                "method_switch": expand_method_shortcut(m, METHOD_SHORTCUT),
            }
            new.append(TestSpec(slots=new_slots, label=m))
    return replace(run, test_set=tuple(new))


def _sweep_is_only_method_vary(test_set: tuple[TestSpec, ...]) -> bool:
    """All rows share the same draw (except method_switch); only UE method differs."""
    if len(test_set) <= 1:
        return True

    def _strip_ms(ts: TestSpec) -> dict:
        s = dict(ts.slots)
        s.pop("method_switch", None)
        return s

    s0 = _strip_ms(test_set[0])
    for t in test_set[1:]:
        if _strip_ms(t) != s0:
            return False
    return True


def _latex_int_list(ex: dict, key: str) -> list[int] | None:
    v = (ex or {}).get(key)
    if v is None or v == "":
        return None
    return [int(x) for x in str(v).split(",") if str(x).strip()]


def _grid_size_from_draw(
    t: TestSpec, base_c: dict, base_s: dict
) -> int | None:
    d = t.slots.get("draw", {}) or {}
    if d and set(d) <= {"calibration", "settings", "methods"}:
        gs = (d.get("settings") or {}).get("grid_size")
        if gs is not None:
            return int(gs)
        return None
    s = {k: v for k, v in d.items() if k in base_s}
    if s.get("grid_size") is not None:
        return int(s["grid_size"])
    return None


def _max_grid_in_testset(
    test_set: tuple[TestSpec, ...], base_c: dict, base_s: dict
) -> int:
    g = 0
    for t in test_set:
        gs = _grid_size_from_draw(t, base_c, base_s)
        if gs is not None:
            g = max(g, gs)
    return g or 3000


def _one_delta_for_plots(
    test_set: tuple[TestSpec, ...], base_c: dict, default: float
) -> float:
    ds: set[float] = set()
    for t in test_set:
        d = t.slots.get("draw", {}) or {}
        if d and set(d) <= {"calibration", "settings", "methods"}:
            c = d.get("calibration") or {}
            if "delta" in c:
                ds.add(_dkey(c["delta"]))
        elif "delta" in d:
            ds.add(_dkey(d["delta"]))
    if ds:
        return sorted(ds)[0]
    return default


def _post_timing_plots(
    run: RunSpec,
    *,
    max_grid: int,
    ref_delta: float,
    calib0: dict,
    settings0: dict,
    save_path: str,
) -> None:
    """On rank 0, four one-off solves on ``max_grid`` for EGM / policy figures."""
    from examples.retirement.outputs import plot_cons_pol, plot_dcegm_cf, plot_egrids

    solutions: dict = {}
    for m in UE_METHODS:
        cal = dict(calib0)
        cal["delta"] = ref_delta
        stg = {**settings0, "grid_size": max_grid, "padding_mbar": -0.011}
        nest, model, _, _ = solve_nest(
            str(run.base_spec),
            method_switch=m,
            draw={"calibration": cal, "settings": stg},
        )
        c_ref = get_policy(nest, "c", stage="labour_mkt_decision")
        tim = get_timing(nest)
        solutions[m] = {
            "nest": nest,
            "model": model,
            "endog_grid": get_policy(nest, "x_dcsn_hat", stage="work_cons"),
            "vf_unrefined": get_policy(nest, "v_dcsn_hat", stage="work_cons"),
            "c_unrefined": get_policy(nest, "c_dcsn_hat", stage="work_cons"),
            "dela_unrefined": get_policy(
                nest, "dela_dcsn_hat", stage="work_cons"
            ),
            "c_refined": c_ref,
            "c_worker": get_policy(nest, "c", stage="work_cons"),
            "timing": tim,
        }

    rfc = solutions["RFC"]
    model = rfc["model"]
    smooth_sigma = model.smooth_sigma
    sigma_tag = (
        "sigma0" if abs(smooth_sigma) < 1e-12 else f"sigma{int(round(smooth_sigma * 100)):02d}"
    )
    grid_size = int(max_grid)

    print(f"Generating plots to {save_path}...")
    plot_egrids(
        int(settings0.get("plot_age", 5)),
        rfc["endog_grid"],
        rfc["vf_unrefined"],
        rfc["c_unrefined"],
        rfc["dela_unrefined"],
        grid_size,
        model,
        save_path,
        tag=sigma_tag,
    )
    plot_cons_pol(solutions["FUES"]["c_worker"], model, save_path)
    plot_dcegm_cf(
        int(settings0.get("plot_age", 5)),
        grid_size,
        rfc["endog_grid"],
        rfc["vf_unrefined"],
        rfc["c_unrefined"],
        rfc["dela_unrefined"],
        model.asset_grid_A,
        model,
        save_path,
        tag=sigma_tag,
    )
    return


def main() -> None:
    run = parse_cli(
        name="retirement",
        base_spec="examples/retirement/syntax",
        modes=["compare", "sweep", "simulate"],
        output="results/retirement",
        extra_args=RE_EXTRA,
    )
    ex = run.extra_args or {}
    if run.test_set is None or len(run.test_set) < 1:
        raise SystemExit("parse_cli did not return a test_set (internal error)")

    comm = _get_mpi_comm()
    is_root = comm is None or comm.Get_rank() == 0
    argv_s = " ".join(sys.argv)
    has_range = "--slot-range" in argv_s
    base_c, base_s = rbench.load_baseline()
    st0 = run.test_set[0]
    calib0, settings0 = _draw_cal_settings(st0, base_c, base_s)
    run = _expand_default_ue_grid(run)
    # 4-UE "plot only": same draw (except method_switch) on every row, UE differs only;
    # no explicit --slot-range (otherwise timing+tables path).
    only_methods = _sweep_is_only_method_vary(run.test_set) and not has_range
    st0 = run.test_set[0]
    calib0, settings0 = _draw_cal_settings(st0, base_c, base_s)

    run_dir = str(run.output_dir)
    if is_root:
        print(f"Model dir: {run.base_spec}")
        print(f"Output directory: {run_dir}")
        if calib0:
            print(f"Params: {calib0}")
        if settings0:
            print(f"Settings: {settings0}")

    if is_root:
        os.makedirs(os.path.join(run_dir, "tables"), exist_ok=True)
    save_path = os.path.join(run_dir, "plots")
    if is_root:
        os.makedirs(save_path, exist_ok=True)

    # --- True solutions for (grid, delta) × methods sweeps; not for 4-UE only ---
    true_solutions: dict | None = None
    if not only_methods:
        deltas = sorted({_row_delta(t, base_c) for t in run.test_set})
        ocal = {**base_c, **calib0}
        oset = {**base_s, **settings0}
        true_solutions = rbench.precompute_true_solutions(
            deltas,
            20000,
            "DCEGM",
            ocal,
            oset,
            comm=comm,
        )

    # --- build solve_test ---
    n_reps = 1 if only_methods else run.sweep_runs
    use_comm = None if only_methods else comm
    wdir = str(run.base_spec).replace("\\", "/")

    def _solve_test_timing(t: TestSpec) -> dict:
        if true_solutions is None:
            raise RuntimeError("timing solve_test: missing true_solutions")
        nest, model, _, _ = solve_nest(wdir, **t.slots)
        c_ref = get_policy(nest, "c", stage="labour_mkt_decision")
        tim = get_timing(nest)
        err = euler(model, c_ref)
        dk = _row_delta(t, base_c)
        ts = true_solutions[dk]
        cdev = consumption_deviation(
            model, c_ref, ts["c_true"], ts["a_grid"]
        )
        return {
            "ue_time": float(tim[0]),
            "total_time": float(tim[1]),
            "error": float(err),
            "cdev": float(cdev),
        }

    def _label_from_method_slot(t: TestSpec) -> str | None:
        ms = t.slots.get("method_switch")
        if not ms:
            return None
        if isinstance(ms, str):
            return ms
        for ent in (ms or {}).get("methods", []) or []:
            for sch in (ent or {}).get("schemes", []) or []:
                m = sch.get("method")
                if m is not None:
                    return str(m)
        return None

    def _solve_test_plots(t: TestSpec) -> dict:
        _label = t.label
        if not _label:
            _label = _label_from_method_slot(t) or ""
        nest, model, _, _ = solve_nest(wdir, **t.slots)
        c_ref = get_policy(nest, "c", stage="labour_mkt_decision")
        err = euler(model, c_ref)
        tim = get_timing(nest)
        return {
            "nest": nest,
            "model": model,
            "endog_grid": get_policy(nest, "x_dcsn_hat", stage="work_cons"),
            "vf_unrefined": get_policy(nest, "v_dcsn_hat", stage="work_cons"),
            "c_unrefined": get_policy(nest, "c_dcsn_hat", stage="work_cons"),
            "dela_unrefined": get_policy(
                nest, "dela_dcsn_hat", stage="work_cons"
            ),
            "c_refined": c_ref,
            "c_worker": get_policy(nest, "c", stage="work_cons"),
            "timing": tim,
            "euler_error": err,
            "label": _label,
        }

    if only_methods:
        solve_test = _solve_test_plots
        metric_fns = {
            "euler_error": lambda r: r["euler_error"],
            "ue_ms": lambda r: r["timing"][0] * 1000,
            "tot_ms": lambda r: r["timing"][1] * 1000,
        }
    else:
        solve_test = _solve_test_timing
        metric_fns = {
            "error": lambda r: r["error"],
            "ue_time": lambda r: r["ue_time"],
            "total_time": lambda r: r["total_time"],
            "cdev": lambda r: r["cdev"],
        }

    if only_methods and comm is not None and not is_root:
        return

    results = sweep(
        solve_test,
        list(run.test_set),
        metric_fns,
        n_reps=n_reps,
        warmup=run.warmup,
        best="min",
        on_error="raise",
        comm=use_comm,
        verbose=run.verbose,
    )
    if not results:
        return

    if not only_methods and is_root:
        bparams = {
            **base_c,
            **base_s,
            **calib0,
            **settings0,
            "true_grid_size": 20000,
            "true_method": "DCEGM",
        }
        lgx = _latex_int_list(ex, "latex_grids")
        rbench.write_timing_sweep_tables(
            results,
            os.path.join(run_dir, "tables"),
            benchmark_params=bparams,
            latex_grids=lgx,
        )
        print("Timing / accuracy tables written to tables/")
        max_g = _max_grid_in_testset(run.test_set, base_c, base_s)
        d0 = _one_delta_for_plots(
            run.test_set, base_c, float(calib0.get("delta", base_c.get("delta", 1.0)))
        )
        _post_timing_plots(
            run,
            max_grid=max_g,
            ref_delta=d0,
            calib0=calib0,
            settings0=settings0,
            save_path=save_path,
        )
        return

    # ---- plot-only: same rank-0 table print + figures from sweep payloads ----
    if not is_root:
        return

    def _sweepresult_to_block(sr: SweepResult) -> dict:
        r = sr.result
        if r is None:
            raise RuntimeError("missing result payload from solve_test")
        m = r["label"] or (sr.point.label or "UNK")
        return {
            m: {
                "nest": r["nest"],
                "model": r["model"],
                "endog_grid": r["endog_grid"],
                "vf_unrefined": r["vf_unrefined"],
                "c_unrefined": r["c_unrefined"],
                "dela_unrefined": r["dela_unrefined"],
                "c_refined": r["c_refined"],
                "c_worker": r["c_worker"],
                "timing": r["timing"],
            }
        }

    solutions: dict = {}
    for sr in results:
        solutions.update(_sweepresult_to_block(sr))

    model = solutions[UE_METHODS[0]]["model"]
    grid_size = model.grid_size
    smooth_sigma = model.smooth_sigma
    sigma_tag = (
        "sigma0" if abs(smooth_sigma) < 1e-12 else f"sigma{int(round(smooth_sigma * 100)):02d}"
    )

    errors: dict = {}
    for m in UE_METHODS:
        errors[m] = euler(model, solutions[m]["c_refined"])

    print()
    print("| Method | Euler Error    | Avg UE time(ms) | Total time(ms) |")
    print("|--------|----------------|-----------------|----------------|")
    for m in UE_METHODS:
        t_ = solutions[m]["timing"]
        print(
            f"| {m:<6s} | {errors[m]:<14.6f} "
            f"| {t_[0]*1000:<15.3f} | {t_[1]*1000:<14.3f} |"
        )
    print()

    from examples.retirement.outputs import plot_cons_pol, plot_dcegm_cf, plot_egrids

    print(f"Generating plots to {save_path}...")
    rfc = solutions["RFC"]
    plot_egrids(
        int(settings0.get("plot_age", 5)),
        rfc["endog_grid"],
        rfc["vf_unrefined"],
        rfc["c_unrefined"],
        rfc["dela_unrefined"],
        grid_size,
        model,
        save_path,
        tag=sigma_tag,
    )
    plot_cons_pol(solutions["FUES"]["c_worker"], model, save_path)
    plot_dcegm_cf(
        int(settings0.get("plot_age", 5)),
        grid_size,
        rfc["endog_grid"],
        rfc["vf_unrefined"],
        rfc["c_unrefined"],
        rfc["dela_unrefined"],
        model.asset_grid_A,
        model,
        save_path,
        tag=sigma_tag,
    )

    print("Done!")


if __name__ == "__main__":
    main()
