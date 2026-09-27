"""Estimation parameter and moment-fit tables (spec 7-E).

Oracles
-------
* Parameter cells: ``theta_best.json`` and ``theta_se.json`` of the saved
  example run, compared at the printed precision (four decimals for
  ``beta`` and ``beta_bar``, three otherwise). The cell regex is spec 7-E.
* Row order: the module-level ``PARAMETER_ORDER``, then any other name
  alphabetically. A synthetic two-run table is the oracle for ``n/a``.
* Fit contributions: checked on the full-precision
  ``fit_<label>.csv`` written by ``write_estimation_tables``.
  ``contribution == w (simulated - data)^2`` with
  ``w = 1 / data^2`` when ``|data| >= 1`` and ``w = 1`` otherwise, the
  default weights of ``kikku.run.estimate.make_criterion``
  (``kikku/run/estimate.py`` lines 138-143). Tolerance ``1e-9``.
  The displayed Markdown table drops the raw contribution column and
  prints data, simulated and residual at four significant figures
  (comma-grouped integers when ``|value| > 1e4``) and
  ``contribution_pct`` to two decimals, sorted by contribution
  descending. The sum-to-100 property of ``contribution_pct`` is not
  an oracle.
* Caption provenance: spec 5.4 / 7-E, when ``fit_table_at_best.csv`` is
  absent.
* Gadi guard: spec 5.7; ``os.path.isdir("/scratch/tp66")`` is patched.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from examples.durables.postprocess.estimation_tables import (  # noqa: E402
    LATEX_SYMBOLS,
    PARAMETER_ORDER,
    fit_table,
    main,
    parameter_table,
    read_run,
    write_estimation_tables,
)

FIXTURE_ROOT = REPO / "tests" / "fixtures" / "estimation_run"
RUN_DIR = sorted(FIXTURE_ROOT.glob("est_*"))[0]
TYPES_ROOT = REPO / "tests" / "fixtures" / "estimation_run_types"

CELL_RE = re.compile(r"^-?\d+\.\d{3,4} \(\d+\.\d+\)$")
FOUR_DEC = {"beta", "beta_bar"}

# Display cells: comma-grouped integers above 1e4, otherwise '{:.4g}'.
FIT_NUM_RE = re.compile(
    r"^-?(?:\d{1,3}(?:,\d{3})+|(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)$"
)
FIT_PCT_RE = re.compile(r"^-?\d+\.\d{2}$")


def _md_rows(text):
    """Parse a GitHub-style markdown table into a list of dicts."""
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    if len(lines) < 2:
        return []
    headers = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def _decimals(name):
    return 4 if name in FOUR_DEC else 3


def _param_row_names(rows):
    footer = {"loss", "converged", "iterations", "n_samples", "grid"}
    names = []
    for row in rows:
        name = row.get("Parameter") or row.get("parameter")
        if name is None:
            continue
        if name in footer or name.startswith("nodes") or "nodes" in name:
            continue
        if name.endswith(" nodes") or name.endswith(" shares"):
            continue
        names.append(name)
    return names


def test_parameter_table_md_cells_match_saved_run():
    """One row per free parameter; cells match theta_best and theta_se."""
    run = read_run(RUN_DIR)
    text = parameter_table([("saved", run)], fmt="md")
    rows = _md_rows(text)
    theta_best = json.loads((RUN_DIR / "theta_best.json").read_text())
    theta_se = json.loads((RUN_DIR / "theta_se.json").read_text())

    param_names = _param_row_names(rows)
    assert param_names == [n for n in PARAMETER_ORDER if n in theta_best]
    assert set(param_names) == set(theta_best)

    by_name = {r["Parameter"]: r for r in rows if r.get("Parameter") in param_names}
    for name in param_names:
        cell = by_name[name]["saved"]
        nd = _decimals(name)
        if cell != "n/a":
            assert CELL_RE.match(cell), (name, cell)
            best_txt, se_txt = cell.split()
            se_txt = se_txt.strip("()")
            assert len(best_txt.split(".")[1]) == nd
            assert float(best_txt) == float(f"{theta_best[name]:.{nd}f}")
            assert float(se_txt) == float(f"{theta_se[name]:.{nd}f}")


def test_parameter_table_tex_has_table_and_run_symbols():
    """LaTeX starts with a table environment and carries the run's symbols."""
    run = read_run(RUN_DIR)
    tex = parameter_table([("saved", run)], fmt="tex")
    assert tex.startswith(r"\begin{table}")
    assert r"\gamma_c" in tex
    run_with_rho = {
        "theta_best": {"rho": 2.0},
        "theta_se": {"rho": 0.1},
        "objective": 1.0,
        "converged": True,
        "n_iter": 1,
        "manifest": None,
        "summary": {},
        "beta_types": None,
    }
    tex_rho = parameter_table([("cd", run_with_rho)], fmt="tex")
    assert r"\rho" in tex_rho
    assert LATEX_SYMBOLS["rho"] == r"\rho"


def test_fit_table_csv_weight_identity_and_md_rounding(tmp_path):
    """CSV keeps full precision for the SMM identity; Markdown is rounded.

    The weight identity is ``contribution == w (simulated - data)^2``
    with ``w = 1/data^2`` when ``|data| >= 1`` else 1, checked at 1e-9
    on ``fit_<label>.csv``. The Markdown table drops the raw
    contribution column, prints data/simulated/residual at four
    significant figures (comma-grouped integers when ``|value| > 1e4``),
    prints ``contribution_pct`` to two decimals, and follows the CSV
    sort order (contribution descending).
    """
    run = read_run(RUN_DIR)
    write_estimation_tables([("saved", run)], tmp_path, fits=[("saved", RUN_DIR)])
    csv_path = tmp_path / "fit_saved.csv"
    assert csv_path.is_file()
    with csv_path.open(newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows
    assert set(csv_rows[0]) >= {
        "moment",
        "data",
        "simulated",
        "residual",
        "contribution",
        "contribution_pct",
    }

    contributions = []
    for row in csv_rows:
        data = float(row["data"])
        sim = float(row["simulated"])
        contrib = float(row["contribution"])
        w = (1.0 / data ** 2) if abs(data) >= 1.0 else 1.0
        expected = w * (sim - data) ** 2
        assert abs(contrib - expected) < 1e-9, (row["moment"], contrib, expected)
        contributions.append(contrib)
    assert contributions == sorted(contributions, reverse=True)

    text = fit_table(RUN_DIR, fmt="md")
    md_rows = _md_rows(text)
    assert md_rows
    headers = set(md_rows[0])
    assert {"moment", "data", "simulated", "residual", "contribution_pct"} <= headers
    assert "contribution" not in headers
    assert [r["moment"] for r in md_rows] == [r["moment"] for r in csv_rows]

    for row in md_rows:
        for col in ("data", "simulated", "residual"):
            cell = row[col]
            assert FIT_NUM_RE.match(cell), (row["moment"], col, cell)
        assert FIT_PCT_RE.match(row["contribution_pct"]), (
            row["moment"],
            row["contribution_pct"],
        )


def test_fit_table_caption_states_last_evaluation_when_at_best_absent():
    """Without fit_table_at_best.csv the caption names the last CE evaluation."""
    assert not (RUN_DIR / "fit_table_at_best.csv").exists()
    text = fit_table(RUN_DIR, fmt="md")
    lowered = text.lower()
    assert "last" in lowered and "cross-entropy" in lowered
    assert "theta_best" in lowered
    assert "not" in lowered


def test_write_estimation_tables_writes_expected_files(tmp_path):
    """parameters.md/.tex and one fit pair per fits entry land in out_dir."""
    run = read_run(RUN_DIR)
    paths = write_estimation_tables(
        [("saved", run)], tmp_path, fits=[("saved", RUN_DIR)]
    )
    names = {p.name for p in paths}
    assert names == {
        "parameters.md",
        "parameters.tex",
        "fit_saved.csv",
        "fit_saved.md",
        "fit_saved.tex",
    }
    for p in paths:
        assert p.is_file() and p.stat().st_size > 0
        assert p.read_text()


def test_parameter_table_orders_rows_and_prints_na_for_absent():
    """Two-run table follows PARAMETER_ORDER and prints n/a where missing."""
    run_rho = {
        "theta_best": {"rho": 2.0, "theta": 0.5},
        "theta_se": {"rho": 0.1, "theta": 0.02},
        "objective": 1.0,
        "converged": True,
        "n_iter": 3,
        "manifest": None,
        "summary": {},
        "beta_types": None,
    }
    run_gc = {
        "theta_best": {"gamma_c": 3.0},
        "theta_se": {"gamma_c": 0.2},
        "objective": 2.0,
        "converged": False,
        "n_iter": 1,
        "manifest": None,
        "summary": {},
        "beta_types": None,
    }
    text = parameter_table([("cd", run_rho), ("sep", run_gc)], fmt="md")
    rows = _md_rows(text)
    names = _param_row_names(rows)
    assert names == ["gamma_c", "rho", "theta"]
    by_name = {r["Parameter"]: r for r in rows if r.get("Parameter") in names}
    assert by_name["gamma_c"]["cd"] == "n/a"
    assert CELL_RE.match(by_name["gamma_c"]["sep"])
    assert CELL_RE.match(by_name["rho"]["cd"])
    assert by_name["rho"]["sep"] == "n/a"
    assert CELL_RE.match(by_name["theta"]["cd"])
    assert by_name["theta"]["sep"] == "n/a"


def test_gadi_guard_exits_when_scratch_exists_and_out_omitted(monkeypatch):
    """On Gadi the command refuses the repository default and asks for --out."""
    real = os.path.isdir

    def fake_isdir(path):
        if path == "/scratch/tp66":
            return True
        return real(path)

    monkeypatch.setattr(os.path, "isdir", fake_isdir)
    with pytest.raises(SystemExit, match="--out"):
        main(["--run", f"saved={RUN_DIR}"])


@pytest.mark.skipif(
    not TYPES_ROOT.exists(),
    reason="types fixture not yet produced (Task E2)",
)
def test_types_footer_shows_beta_nodes():
    """A types run shows beta_bar and sigma_beta rows and a nodes footer."""
    ests = sorted(TYPES_ROOT.glob("est_*"))
    types_dir = ests[0] if ests else TYPES_ROOT
    run = read_run(types_dir)
    text = parameter_table([("types", run)], fmt="md")
    assert "beta_bar" in text
    assert "sigma_beta" in text
    assert "nodes" in text.lower()
