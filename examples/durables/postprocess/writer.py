"""Runner-side output orchestration for the durables example.

Pipeline (all pure-builder + IO-writer; no compute):

  results ──► row-builders ──► table renderers ──► writers ──► disk
              (this file)      (tables.py)         (this file)

The dispatcher ``write_outputs`` picks the right combination based on
``run.mode``. All calibration / settings values come from each row's
solved stage — spec_factory has already applied the row's slot overrides,
so re-classifying the slot bundle is unnecessary.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from kikku.run import write_results_table
from kikku.run.metrics import format_table
from kikku.run.sweep import SweepResult
from kikku.run.types import RunSpec

from .tables import (
    generate_comparison_table,
    generate_sweep_table,
    write_euler_detail,
)

# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def write_outputs(results: list[SweepResult], run: RunSpec) -> None:
    """Dispatch results to the right writer for the run mode.

    Always writes ``sweep.md``. Then:

    - ``compare`` → ``comparison.md`` + ``comparison.tex`` + Euler detail
    - ``sweep``  → ``sweep.tex`` if multiple rows
    - else       → single-row console print
    """
    if not results:
        return
    tdir = Path(run.output_dir) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    write_results_table(results, str(tdir / "sweep.md"), fmt="markdown")

    if run.mode == "compare":
        _write_compare(results, run, tdir)
    elif run.mode == "sweep" and len(results) > 1:
        _write_sweep_latex(results, run, tdir)
    else:
        _print_single(results, run)


# ---------------------------------------------------------------------------
# Compare mode — comparison.md / .tex + console table + Euler detail
# ---------------------------------------------------------------------------


def _write_compare(
    results: list[SweepResult], run: RunSpec, tdir: Path
) -> None:
    rows = _compare_rows(results)
    params = _first_row_calibration(results)
    if rows:
        print("\n" + format_table(rows, list(rows[0].keys())))
    md = generate_comparison_table(
        rows, fmt="md", caption="Durables Model Comparison", params=params
    )
    (tdir / "comparison.md").write_text(md, encoding="utf-8")
    tex = generate_comparison_table(
        rows, fmt="tex", caption="Durables Model Comparison", params=params
    )
    (tdir / "comparison.tex").write_text(tex, encoding="utf-8")
    print(f"Tables saved to {tdir} (comparison.md, comparison.tex)")
    if run.sim is not None:
        euler = _euler_by_label(results)
        if euler:
            p = write_euler_detail(euler, str(tdir))
            print(f"Euler detail saved to {p}")


# ---------------------------------------------------------------------------
# Sweep mode — bespoke per-row LaTeX summary
# ---------------------------------------------------------------------------


_LATEX_CAL_KEYS = (
    "beta", "gamma_c", "gamma_h", "alpha", "delta",
    "R", "R_H", "phi_w", "sigma_w",
)

_LATEX_PREAMBLE = (
    "\\documentclass[11pt]{article}\n"
    "\\usepackage{booktabs}\n"
    "\\usepackage[margin=1in]{geometry}\n"
    "\\begin{document}\n"
    "\\pagestyle{empty}\n"
)


def _write_sweep_latex(
    results: list[SweepResult], run: RunSpec, tdir: Path
) -> None:
    n_sim = run.sim.n_sim if run.sim is not None else 0
    rows = _latex_summary_rows(results, n_sim)
    tex = generate_sweep_table(
        rows, fmt="tex", caption="Durables Model: Per-Period Timing and Accuracy"
    )
    (tdir / "sweep.tex").write_text(
        _LATEX_PREAMBLE + tex + "\n\\end{document}\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Single mode — console print of one row
# ---------------------------------------------------------------------------


def _print_single(results: list[SweepResult], run: RunSpec) -> None:
    if len(results) != 1:
        return
    row = _single_row(results[0], with_sim=run.sim is not None)
    print("\n" + format_table([row], list(row.keys())))


# ---------------------------------------------------------------------------
# Pure builders: SweepResult(s) → row dict(s)
# ---------------------------------------------------------------------------


_EULER_OUT_KEYS = [
    ("euler_c_keeper", "Euler c (keeper)"),
    ("euler_c_adjuster", "Euler c (adj)"),
    ("euler_c_mean", "Euler c (all)"),
    ("euler_h_keeper", "Euler h (keeper)"),
    ("euler_h_adjuster", "Euler h (adj)"),
    ("euler_h_mean", "Euler h (all)"),
    ("adj_rate", "Adj Rate"),
]


def _compare_rows(results: list[SweepResult]) -> list[dict]:
    rows: list[dict] = []
    for sr in results:
        m = sr.metrics
        label = sr.point.label or m.get("method", "")
        row: dict[str, Any] = {
            "Method": label,
            "Keeper (ms)": m.get("keeper_ms", 0.0),
            "Adj (ms)": m.get("adj_ms", 0.0),
            "Total (ms)": m.get("keeper_ms", 0.0) + m.get("adj_ms", 0.0),
        }
        for k, outk in _EULER_OUT_KEYS:
            v = m.get(k)
            if v is not None and not (isinstance(v, float) and v != v):
                row[outk] = v
        rows.append(row)
    return rows


def _single_row(sr: SweepResult, with_sim: bool) -> dict:
    m = dict(sr.metrics)
    if with_sim and sr.result is not None and "euler_c_all" in sr.result:
        m["Euler c (all)"] = sr.result.get("euler_c_all", np.nan)
    row: dict[str, Any] = {
        "Method": m.get("method", ""),
        "Keeper (ms)": m.get("keeper_ms", 0.0),
        "Adj (ms)": m.get("adj_ms", 0.0),
        "Total (ms)": m.get("keeper_ms", 0.0) + m.get("adj_ms", 0.0),
    }
    for k, outk in _EULER_OUT_KEYS:
        if k in m:
            row[outk] = m[k]
    return row


def _latex_summary_rows(
    results: list[SweepResult], n_sim: int
) -> list[dict]:
    rows: list[dict] = []
    for sr in results:
        if sr.result is None:
            continue
        stage0 = sr.result["nest"]["periods"][0]["stages"]["keeper_cons"]
        cal, st = stage0.calibration, stage0.settings
        m = sr.metrics
        cal_params = {k: cal[k] for k in _LATEX_CAL_KEYS if k in cal}
        if "R" in cal_params:
            cal_params["r"] = cal_params.pop("R")
        if "R_H" in cal_params:
            cal_params["r_H"] = cal_params.pop("R_H")
        cal_params["N_sim"] = n_sim
        rows.append(
            {
                "Grid_Size": int(m.get("n_a", st.get("n_a", 0))),
                "Tau": float(m.get("tau", cal.get("tau", 0.0))),
                "Method": m.get("method", ""),
                "Avg_Keeper_ms": m.get("keeper_ms", 0.0),
                "Avg_Adj_ms": m.get("adj_ms", 0.0),
                "Euler_Combined": m.get("euler_c_mean", np.nan),
                "Euler_Keeper": m.get("euler_c_keeper", np.nan),
                "Euler_Adjuster": m.get("euler_c_adjuster", np.nan),
                "Euler_H_Adjuster": m.get("euler_h_mean", np.nan),
                **cal_params,
            }
        )
    return rows


def _first_row_calibration(results: list[SweepResult]) -> dict:
    """Calibration dict from the first solved row, for comparison-table headers."""
    for sr in results:
        if sr.result is not None:
            stage0 = sr.result["nest"]["periods"][0]["stages"]["keeper_cons"]
            return dict(stage0.calibration)
    return {}


def _euler_by_label(results: list[SweepResult]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sr in results:
        if sr.result is None:
            continue
        label = sr.point.label or sr.result.get("method_label", "")
        out[label] = {
            "consumption": sr.result.get("euler_c_stats", {}),
            "housing": sr.result.get("euler_h_stats", {}),
        }
    return out
