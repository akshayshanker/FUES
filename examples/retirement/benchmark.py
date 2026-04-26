"""Benchmarking helpers: true-solution precompute and timing-table layout.

The retirement ``run.py`` path uses the canonical kikku ``sweep`` over
``run.test_set``; this module holds the cross-tab post-processing and
the reference-solution precompute for consumption-deviation metrics.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import yaml
from kikku.run.mpi import bcast_item, is_root
from kikku.run.sweep import SweepResult
from kikku.run.types import TestSpec

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pathlib import Path  # noqa: E402

from .outputs import (  # noqa: E402
    generate_accuracy_table,
    generate_timing_table_combined,
    get_policy,
)
from .solve import solve_nest  # noqa: E402

SYNTAX_DIR = Path(__file__).resolve().parent / "syntax"
_cal_path = SYNTAX_DIR / "calibration.yaml"
_set_path = SYNTAX_DIR / "settings.yaml"

# Data row layout MUST match outputs/tables expectations:
#   "Data rows arrive as [grid, delta, RFC, FUES, DCEGM, CONSAV] = indices 2,3,4,5"
# tables._COL_ORDER = (3, 4, 5, 2) remaps to display order (FUES, MSS=DCEGM, LTM=CONSAV, RFC).
METHODS = ("RFC", "FUES", "DCEGM", "CONSAV")


def load_baseline() -> tuple[dict, dict]:
    with open(_cal_path) as f:
        cal = yaml.safe_load(f)["calibration"]
    with open(_set_path) as f:
        settings = yaml.safe_load(f)["settings"]
    return cal, settings


def precompute_true_solutions(
    deltas: list[float],
    true_grid_size: int,
    true_method: str,
    base_params: dict,
    base_settings: dict,
    *,
    comm,
) -> dict[float, dict]:
    """High-grid reference policy per *delta*; rank 0 only then broadcast.

    Each value is ``{'c_true': ..., 'a_grid': ...}`` for ``consumption_deviation``.
    """
    trues: dict | None
    if is_root(comm):
        out: dict[float, dict] = {}
        for d in deltas:
            dk = _dkey(d)
            cal_ov = {**base_params, "delta": d}
            cfg_ov = {
                **base_settings,
                "grid_size": int(true_grid_size),
                "padding_mbar": -0.011,
            }
            nest, model, _, _ = solve_nest(
                SYNTAX_DIR,
                method_switch=true_method,
                draw={"calibration": cal_ov, "settings": cfg_ov},
            )
            out[dk] = {
                "c_true": get_policy(nest, "c"),
                "a_grid": model.asset_grid_A,
            }
        trues = out
    else:
        trues = None
    b = bcast_item(trues, comm, root=0)
    if b is None:
        raise RuntimeError("precompute_true_solutions: broadcast failed")
    return b


def _dkey(x: float) -> float:
    return round(float(x), 10)


def _params_settings_from_testspec(t: TestSpec) -> tuple[dict, dict]:
    base_c, base_s = load_baseline()
    d = t.slots.get("draw", {}) or {}
    if d and set(d) <= {"calibration", "settings", "methods"}:
        return (d.get("calibration") or {}), (d.get("settings") or {})
    p, s = {}, {}
    for k, v in d.items():
        if k in base_c:
            p[k] = v
        if k in base_s:
            s[k] = v
    return p, s


def _method_tag(t: TestSpec) -> str:
    ms = t.slots.get("method_switch")
    if ms and isinstance(ms, str):
        return str(ms)
    if ms and isinstance(ms, dict):
        for ent in (ms or {}).get("methods", []) or []:
            for sch in (ent or {}).get("schemes", []) or []:
                m = sch.get("method")
                if m is not None:
                    return str(m)
    if t.label:
        return str(t.label)
    return "UNK"


def format_timing_sweep_for_tables(
    results: list[SweepResult],
    *,
    method_order: tuple[str, ...] = METHODS,
) -> dict[str, list]:
    """Turn flat ``SweepResult`` rows into the row-lists the LaTeX writers expect.

    Returns keys ``errors``, ``ue_ms``, ``total_ms``, ``cdev``; each row is
    ``[grid_size, delta, m0, m1, m2, m3]`` in ``method_order``.
    """
    base_c, base_s = load_baseline()
    by_key: dict[tuple[int, float, str], Any] = {}
    for sr in results:
        t = sr.point
        if not isinstance(t, TestSpec):
            raise TypeError("format_timing_sweep_for_tables expected TestSpec points")
        p, s = _params_settings_from_testspec(t)
        settings_row = {**base_s, **s}
        if "grid_size" not in settings_row:
            raise ValueError("Timing sweep rows need draw→settings grid_size (or in base settings)")
        gs = int(settings_row["grid_size"])
        d = _dkey(p.get("delta", base_c.get("delta", 1.0)))
        m = _method_tag(t)
        by_key[(gs, d, m)] = sr.metrics

    latex_errors: list = []
    latex_ue: list = []
    latex_tot: list = []
    latex_cdev: list = []

    gset: set[int] = set()
    dset: set[float] = set()
    for sr in results:
        p0 = sr.point
        if isinstance(p0, TestSpec):
            pp, sp = _params_settings_from_testspec(p0)
            srow = {**base_s, **sp}
            if "grid_size" in srow:
                gset.add(int(srow["grid_size"]))
            dset.add(_dkey(pp.get("delta", base_c.get("delta", 1.0))))

    for gs in sorted(gset):
        for d in sorted(dset):
            if not all((gs, d, m) in by_key for m in method_order):
                continue
            e_row, ue_row, tot_row, cd_row = [], [], [], []
            for meth in method_order:
                m = by_key[(gs, d, meth)]
                e_row.append(m.get("error", float("nan")))
                ue_row.append(m.get("ue_time", float("nan")) * 1000.0)
                tot_row.append(m.get("total_time", float("nan")) * 1000.0)
                cd_row.append(m.get("cdev", float("nan")))
            latex_errors.append([gs, d, *e_row])
            latex_ue.append([gs, d, *ue_row])
            latex_tot.append([gs, d, *tot_row])
            latex_cdev.append([gs, d, *cd_row])

    return {
        "errors": latex_errors,
        "ue_ms": latex_ue,
        "total_ms": latex_tot,
        "cdev": latex_cdev,
    }


def write_timing_sweep_tables(
    results: list[SweepResult],
    results_dir: str,
    *,
    benchmark_params: dict,
    latex_grids: list[int] | None,
) -> None:
    """Reshape + write markdown/LaTeX timing and accuracy tables."""
    shaped = format_timing_sweep_for_tables(results)
    os.makedirs(results_dir, exist_ok=True)
    generate_timing_table_combined(
        shaped["ue_ms"],
        shaped["total_ms"],
        "timing",
        "Retirement model",
        results_dir,
        params=benchmark_params,
        latex_grids=latex_grids,
    )
    generate_accuracy_table(
        shaped["errors"],
        shaped["cdev"],
        "accuracy",
        "Retirement model",
        results_dir,
        params=benchmark_params,
        latex_grids=latex_grids,
    )
