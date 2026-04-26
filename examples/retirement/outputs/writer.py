"""Runner-side output orchestration for the retirement example.

Two entry points dispatched from ``run.main`` based on mode:

- ``write_timing_outputs``: timing/accuracy tables (markdown + LaTeX) plus
  four-method reference figures solved at ``max_grid``.
- ``write_plot_outputs``: console method table plus four-method figures
  rendered from the existing sweep payloads.

Both share ``_render_method_figures`` so the plotting routine has one
home; the only difference is whether the per-method solutions come from
fresh solves or from the sweep result objects.
"""

from __future__ import annotations

import os

from kikku.run.sweep import SweepResult
from kikku.run.types import RunSpec

from examples.retirement import benchmark as rbench
from examples.retirement.outputs import (
    euler,
    get_policy,
    get_timing,
)
from examples.retirement.solve import solve_nest

UE_METHODS = ("RFC", "FUES", "DCEGM", "CONSAV")


# ---------------------------------------------------------------------------
# Timing-mode dispatch: tables + four-method reference figures
# ---------------------------------------------------------------------------


def write_timing_outputs(
    results: list[SweepResult],
    run: RunSpec,
    *,
    benchmark_params: dict,
    latex_grids: list[int] | None,
    save_path: str,
    max_grid: int,
    ref_delta: float,
    calib0: dict,
    settings0: dict,
) -> None:
    """Tables (timing + accuracy) and four-method figures at ``max_grid``."""
    rbench.write_timing_sweep_tables(
        results,
        os.path.join(str(run.output_dir), "tables"),
        benchmark_params=benchmark_params,
        latex_grids=latex_grids,
    )
    print("Timing / accuracy tables written to tables/")

    solutions = _solve_for_reference(
        str(run.base_spec),
        max_grid=max_grid,
        ref_delta=ref_delta,
        calib0=calib0,
        settings0=settings0,
    )
    _render_method_figures(
        solutions,
        save_path=save_path,
        plot_age=int(settings0.get("plot_age", 5)),
        grid_size=max_grid,
    )


# ---------------------------------------------------------------------------
# Plot-only dispatch: print method table + figures from sweep payloads
# ---------------------------------------------------------------------------


def write_plot_outputs(
    results: list[SweepResult],
    *,
    settings0: dict,
    save_path: str,
) -> None:
    """Console method table + figures rendered from sweep payloads."""
    solutions = _solutions_from_sweep(results)
    if not solutions:
        return
    model = solutions[UE_METHODS[0]]["model"]
    _print_method_table(solutions, model)
    _render_method_figures(
        solutions,
        save_path=save_path,
        plot_age=int(settings0.get("plot_age", 5)),
        grid_size=int(model.grid_size),
    )
    print("Done!")


# ---------------------------------------------------------------------------
# Solutions: from sweep payloads or freshly solved at max_grid
# ---------------------------------------------------------------------------


def _solutions_from_sweep(results: list[SweepResult]) -> dict[str, dict]:
    """Pack each sweep row's result payload into ``solutions[label]``."""
    out: dict[str, dict] = {}
    for sr in results:
        r = sr.result
        if r is None:
            raise RuntimeError("missing result payload from solve_test")
        label = r.get("label") or sr.point.label or "UNK"
        out[label] = {
            "nest":           r["nest"],
            "model":          r["model"],
            "endog_grid":     r["endog_grid"],
            "vf_unrefined":   r["vf_unrefined"],
            "c_unrefined":    r["c_unrefined"],
            "dela_unrefined": r["dela_unrefined"],
            "c_refined":      r["c_refined"],
            "c_worker":       r["c_worker"],
            "timing":         r["timing"],
        }
    return out


def _solve_for_reference(
    base_spec: str,
    *,
    max_grid: int,
    ref_delta: float,
    calib0: dict,
    settings0: dict,
) -> dict[str, dict]:
    """Solve every UE method at ``max_grid`` for the reference figure."""
    cal = {**calib0, "delta": ref_delta}
    stg = {**settings0, "grid_size": max_grid, "padding_mbar": -0.011}
    solutions: dict[str, dict] = {}
    for m in UE_METHODS:
        nest, model, _, _ = solve_nest(
            base_spec,
            method_switch=m,
            draw={"calibration": cal, "settings": stg},
        )
        solutions[m] = {
            "nest":           nest,
            "model":          model,
            "endog_grid":     get_policy(nest, "x_dcsn_hat", stage="work_cons"),
            "vf_unrefined":   get_policy(nest, "v_dcsn_hat", stage="work_cons"),
            "c_unrefined":    get_policy(nest, "c_dcsn_hat", stage="work_cons"),
            "dela_unrefined": get_policy(nest, "dela_dcsn_hat", stage="work_cons"),
            "c_refined":      get_policy(nest, "c", stage="labour_mkt_decision"),
            "c_worker":       get_policy(nest, "c", stage="work_cons"),
            "timing":         get_timing(nest),
        }
    return solutions


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _print_method_table(solutions: dict[str, dict], model) -> None:
    """Console-print Method × (Euler, UE time, Total time) for UE_METHODS."""
    print()
    print("| Method | Euler Error    | Avg UE time(ms) | Total time(ms) |")
    print("|--------|----------------|-----------------|----------------|")
    for m in UE_METHODS:
        if m not in solutions:
            continue
        err = euler(model, solutions[m]["c_refined"])
        t = solutions[m]["timing"]
        print(
            f"| {m:<6s} | {err:<14.6f} "
            f"| {t[0] * 1000:<15.3f} | {t[1] * 1000:<14.3f} |"
        )
    print()


def _render_method_figures(
    solutions: dict[str, dict],
    *,
    save_path: str,
    plot_age: int,
    grid_size: int,
) -> None:
    """Three paper figures: egrids, cons_pol, dcegm_cf — RFC + FUES based."""
    from examples.retirement.outputs import (
        plot_cons_pol,
        plot_dcegm_cf,
        plot_egrids,
    )

    if "RFC" not in solutions or "FUES" not in solutions:
        return
    rfc = solutions["RFC"]
    model = rfc["model"]
    sigma_tag = _sigma_tag(model.smooth_sigma)

    print(f"Generating plots to {save_path}...")
    plot_egrids(
        plot_age,
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
        plot_age,
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


def _sigma_tag(smooth_sigma: float) -> str:
    if abs(smooth_sigma) < 1e-12:
        return "sigma0"
    return f"sigma{int(round(smooth_sigma * 100)):02d}"
