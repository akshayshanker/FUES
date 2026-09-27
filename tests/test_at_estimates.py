"""Lifecycle profiles and moment fit at a finished estimation run.

Task B / check D of the Cobb-Douglas estimation and post-processing spec
(``AI/devspecs/28092026/cd_estimation_and_postprocess.md``, Sections 5.3,
5.7 and 7-D). The saved example run is a separable estimation at
``n_a = n_h = n_w = 30``, ``t0 = 20``, 500 simulated agents and seed 99;
the manifest records these.

Oracles
-------
* ``load_run`` fields: the fixture ``summary.json`` and ``manifest.json``.
* Simulated consumption means: ``numpy.nanmean`` of ``sim_data['c']`` on
  exactly the rows ``kikku.run.moments._age_group_masks`` assigns to each
  age group, times ``1 / normalisation``. That pins kikku's four-of-five-
  years convention rather than assuming a five-year bin.
* Fit contributions: ``w * (sim - data)^2`` with
  ``w = 1 / data^2`` when ``|data| >= 1`` else ``1``, the weight rule in
  ``kikku.run.estimate.make_criterion``.
* Objective reconciliation: the sum of those contributions at the
  manifest's own grid, ``N_sim`` and seed equals
  ``summary.json['objective']`` within ``1e-8``.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kikku.run.estimate import load_estimation_spec
from kikku.run.moments import _age_group_masks

from examples.durables.postprocess import at_estimates

FIXTURE_ROOT = REPO / "tests" / "fixtures" / "estimation_run"
RUN_DIRS = sorted(p for p in FIXTURE_ROOT.glob("est_*") if p.is_dir())
assert RUN_DIRS, f"no est_* folder under {FIXTURE_ROOT}"
RUN_DIR = RUN_DIRS[0]

TYPES_FIXTURE_ROOT = REPO / "tests" / "fixtures" / "estimation_run_types"
TYPES_RUN_DIRS = sorted(
    p for p in TYPES_FIXTURE_ROOT.glob("est_*") if p.is_dir()
) if TYPES_FIXTURE_ROOT.is_dir() else []

REGISTRY_ROOT = REPO / "examples" / "durables" / "syntax"

# Level moments in AUD sit well above 100 and well below 10 million at
# this calibration (normalisation 1e-5). The bounds are the spec's
# acceptance window, not an economic claim about the estimates.
_LEVEL_AUD_LO = 1e2
_LEVEL_AUD_HI = 1e7

# Same prefixes the estimation driver uses to decide which moment keys
# are levels (examples.durables.estimate._LEVEL_PREFIXES).
_LEVEL_PREFIXES = (
    "mean_", "sd_", "av_", "cond_discrete_0_mean_",
    "cond_discrete_1_mean_", "cond_discrete_0_sd_",
    "cond_discrete_1_sd_",
)


def _is_level_key(key):
    base = key.rsplit("__age", 1)[0] if "__age" in key else key
    return any(base.startswith(p) for p in _LEVEL_PREFIXES)


def _weight(data):
    """kikku's default SMM weight: 1/data^2 when |data| >= 1, else 1."""
    return 1.0 / (data * data) if abs(data) >= 1.0 else 1.0


def _stage(nest):
    return nest["periods"][0]["stages"]["keeper_cons"]


@pytest.fixture(scope="module")
def run():
    return at_estimates.load_run(RUN_DIR)


@pytest.fixture(scope="module")
def solved(run):
    return at_estimates.solve_at_estimates(run, REGISTRY_ROOT)


@pytest.fixture(scope="module")
def sim_data_n200(run, solved):
    return at_estimates.simulate_at_estimates(run, solved, N=200)


@pytest.fixture(scope="module")
def moment_bundle(run, solved, sim_data_n200):
    spec_path = (
        REGISTRY_ROOT / run["registry"] / "estimation"
        / f"{run['spec_name']}.yaml"
    )
    spec = load_estimation_spec(str(spec_path))
    sett = _stage(solved[0][1]).settings
    sim_moments = at_estimates.simulated_moments_at_estimates(
        sim_data_n200, spec["moment_spec"], sett)
    return spec, sett, sim_moments


def test_load_run_fields(run):
    """Every field Section 5.3 names is present and matches the fixture."""
    import json
    summary = json.loads((RUN_DIR / "summary.json").read_text())
    manifest = json.loads((RUN_DIR / "manifest.json").read_text())

    assert run["theta_best"] == summary["theta_best"]
    assert run["theta_mean"] == summary["theta_mean"]
    assert run["theta_se"] == summary["theta_se"]
    assert run["objective"] == summary["objective"]
    assert run["converged"] is False
    assert run["n_iter"] == 2
    assert run["calib_overrides"] == {"t0": 20}
    assert run["registry"] == "separable"
    assert run["spec_name"] == "baseline_large_egm"
    assert run["spec_factory"] == "spec_factory.yaml"
    assert run["grid"] == {"n_a": 30, "n_h": 30, "n_w": 30}
    assert run["N_sim"] == 500
    assert run["simulation_seed"] == 99
    assert run["solver_method"] is None
    assert run["beta_types"] is None
    assert run["manifest"] == manifest


def test_types_spec_absent_on_separable_run(run):
    """The fixture spec has no types block."""
    assert at_estimates.types_spec_for_run(run, REGISTRY_ROOT) is None


def test_solve_at_estimates_one_entry(solved):
    """A run without types produces one solved model with share 1."""
    assert len(solved) == 1
    share, nest, grids = solved[0]
    assert share == 1.0
    assert nest is not None
    assert grids is not None


def test_solve_raises_without_grid(run):
    """ValueError when neither the call nor the run supplies a grid."""
    bare = dict(run)
    bare["grid"] = None
    with pytest.raises(ValueError, match="grid"):
        at_estimates.solve_at_estimates(bare, REGISTRY_ROOT, grid=None)


def test_simulate_shape_n200(sim_data_n200):
    """Panels at N=200 have shape (T, 200) with T = 70."""
    for key in ("c", "a", "h", "discrete"):
        assert sim_data_n200[key].shape == (70, 200), key


def test_simulated_moments_finite_aud(moment_bundle):
    """Each target and age group inside the horizon is finite; levels in AUD.

    Identification conditionals (adjuster means in a bin with no
    adjusters) can be NaN; those are not targets and are not scored.
    """
    spec, _, sim_moments = moment_bundle
    setup = spec["moment_spec"]["setup"]
    t0 = int(setup.get("t0", 0))
    T = int(setup.get("T", 70))
    masks = _age_group_masks(T, setup["age_groups"], t0)
    n_checked = 0
    for target in spec["moment_spec"].get("targets") or []:
        key = target["key"]
        for group, rows in masks.items():
            if not rows:
                continue
            full_key = f"{key}__age{group}"
            if full_key not in sim_moments:
                continue
            value = sim_moments[full_key]
            assert np.isfinite(value), full_key
            if _is_level_key(full_key):
                assert _LEVEL_AUD_LO < value < _LEVEL_AUD_HI, (full_key, value)
            n_checked += 1
    assert n_checked > 0


def test_c_means_match_age_group_masks(run, solved, sim_data_n200, moment_bundle):
    """Consumption means equal a direct numpy average on kikku's rows."""
    spec, sett, sim_moments = moment_bundle
    setup = spec["moment_spec"]["setup"]
    t0 = int(setup.get("t0", 0))
    T = int(setup.get("T", sim_data_n200["c"].shape[0]))
    masks = _age_group_masks(T, setup["age_groups"], t0)
    denorm = 1.0 / float(sett["normalisation"])
    c = sim_data_n200["c"]
    for group, rows in masks.items():
        if not rows:
            continue
        key = f"mean_c__age{group}"
        if key not in sim_moments:
            continue
        direct = float(np.nanmean(c[np.array(rows, dtype=int), :])) * denorm
        assert sim_moments[key] == pytest.approx(direct, rel=0, abs=0.0)


def test_fit_weight_rule(run, moment_bundle):
    """Each contribution equals the kikku weight rule at 1e-9."""
    spec, _, sim_moments = moment_bundle
    rows = at_estimates.fit_at_estimates(
        run, sim_moments, spec["data_moments"])
    assert rows
    for row in rows:
        data = float(row["data"])
        sim = float(row["simulated"])
        expected = _weight(data) * (sim - data) ** 2
        assert row["contribution"] == pytest.approx(expected, abs=1e-9), row[
            "moment"]


def test_write_outputs_and_objective_reconciliation(run, tmp_path):
    """Figure and both CSVs are written; contributions sum to the objective.

    Re-solves at the manifest grid (30^3) and simulates 500 agents, the
    same evaluation the estimation reported as ``objective``. About a
    minute on this machine.
    """
    paths = at_estimates.write_outputs(
        run, REGISTRY_ROOT, tmp_path,
        grid=run["grid"], N=run["N_sim"], seed=run["simulation_seed"],
    )
    fig = Path(paths["lifecycle_at_estimates.png"])
    fit_csv = Path(paths["fit_table_at_best.csv"])
    cohort_csv = Path(paths["cohort_means.csv"])
    assert fig.is_file() and fig.stat().st_size > 0
    assert fit_csv.is_file()
    assert cohort_csv.is_file()

    with fit_csv.open(newline="") as f:
        fit_rows = list(csv.DictReader(f))
    total = sum(float(r["contribution"]) for r in fit_rows)
    assert total == pytest.approx(run["objective"], abs=1e-8)

    with cohort_csv.open(newline="") as f:
        cohort_rows = list(csv.DictReader(f))
    assert cohort_rows
    assert set(cohort_rows[0]) >= {"age group", "variable", "data", "simulated"}


def test_gadi_guard_requires_out(monkeypatch, run):
    """On Gadi the tool refuses to write inside the checkout without --out."""
    real_isdir = os.path.isdir

    def fake_isdir(path):
        if os.path.abspath(path) == "/scratch/tp66" or path == "/scratch/tp66":
            return True
        return real_isdir(path)

    monkeypatch.setattr(os.path, "isdir", fake_isdir)
    with pytest.raises(SystemExit):
        at_estimates.main([str(RUN_DIR)])


@pytest.mark.skipif(
    not TYPES_RUN_DIRS,
    reason="types fixture not yet produced (Task E2)",
)
def test_types_fixture(tmp_path):
    """Same checks on the types saved-example run, when Task E2 has written it."""
    types_dir = TYPES_RUN_DIRS[0]
    run = at_estimates.load_run(types_dir)
    assert run["beta_types"] is not None
    k = int(run["beta_types"]["n"])
    assert k >= 2
    types_spec = at_estimates.types_spec_for_run(run, REGISTRY_ROOT)
    assert types_spec is not None
    solved = at_estimates.solve_at_estimates(run, REGISTRY_ROOT)
    assert len(solved) == k
    sim_data = at_estimates.simulate_at_estimates(run, solved, N=200)
    assert "type_idx" in sim_data
    for key in ("c", "a", "h"):
        assert sim_data[key].shape == (70, 200)
    paths = at_estimates.write_outputs(
        run, REGISTRY_ROOT, tmp_path,
        grid=run["grid"], N=run["N_sim"], seed=run["simulation_seed"],
    )
    with Path(paths["fit_table_at_best.csv"]).open(newline="") as f:
        fit_rows = list(csv.DictReader(f))
    total = sum(float(r["contribution"]) for r in fit_rows)
    assert total == pytest.approx(run["objective"], abs=1e-8)
