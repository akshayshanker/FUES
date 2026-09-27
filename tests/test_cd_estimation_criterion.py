"""The estimation driver's criterion, parameter-name guard and types hooks.

Task A2 of the Cobb-Douglas estimation and post-processing spec
(``AI/devspecs/28092026/cd_estimation_and_postprocess.md``, Sections 5.2,
5.6 and 7-B).  The criterion under test is the driver's own
``build_criterion``; the model is the separable registry at a 30^3 grid,
``t0 = 20`` (50 periods) and 500 simulated agents, for the female and the
male calibration chains.

Oracles
-------
* Finite loss and no NaN target: the spec's verified fact that the registries
  give every target moment without NaN at ``t0 = 20`` (Section 2).
* Sensitivity to ``beta``: an override that reaches the solver changes the
  simulated moments; the Cobb-Douglas ``gamma_c`` incident (Section 2) is the
  failure this guards against, where an ignored name changed nothing.
* Calibration names: the ``calibration:`` block of
  ``cobb_douglas/calibration/main.yaml`` read directly from the file, not
  through ``make_spec``.
* ``denorm``: ``normalisation: 1.0e-05`` in ``separable/settings.yaml``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kikku.run.estimate import BIG_LOSS, load_estimation_spec  # noqa: E402

SYNTAX = REPO_ROOT / "examples" / "durables" / "syntax"
SEPARABLE = SYNTAX / "separable"
COBB_DOUGLAS = SYNTAX / "cobb_douglas"
SPEC_NAME = "baseline_large_egm.yaml"
FACTORIES = ["spec_factory.yaml", "spec_factory_males.yaml"]

GRID = {"n_a": 30, "n_h": 30, "n_w": 30}
CALIB_OVERRIDES = {"t0": 20}
N_SIM = 500
SIMULATION_SEED = 99  # the separable specs' simulation_seed

# The unit conversion the driver applies to level moments: 1/normalisation,
# with normalisation 1.0e-05 in separable/settings.yaml (line 71).
DENORM_EXPECTED = 1.0e5

TYPES_SPEC = {"beta": {"n": 4, "location": "beta_bar", "spread": "sigma_beta"}}


def _driver():
    """Import the driver lazily so a missing name fails the test, not collection."""
    import examples.durables.estimate as est
    return est


def _base_theta(spec):
    """Midpoint of every free parameter's bounds, with beta at 0.945."""
    theta = {
        name: 0.5 * (float(b["bounds"][0]) + float(b["bounds"][1]))
        for name, b in spec["free"].items()
    }
    theta["beta"] = 0.945
    return theta


@pytest.fixture(scope="module")
def spec():
    return load_estimation_spec(str(SEPARABLE / "estimation" / SPEC_NAME))


@pytest.fixture(scope="module")
def criteria(spec):
    """The driver's criterion for the female and the male calibration chain."""
    est = _driver()
    return {
        factory: est.build_criterion(
            SEPARABLE, spec, factory, None, dict(CALIB_OVERRIDES), dict(GRID),
            N_SIM, SIMULATION_SEED, None,
        )
        for factory in FACTORIES
    }


# ---------------------------------------------------------------------------
# (a) one evaluation is finite and no target moment is NaN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", FACTORIES)
def test_one_evaluation_is_finite_with_no_nan_target(criteria, spec, factory):
    criterion, moment_fn, data_moments, denorm = criteria[factory]
    assert denorm == pytest.approx(DENORM_EXPECTED)
    assert len(data_moments) > 0

    theta = _base_theta(spec)
    loss = criterion(theta)
    # kikku scores a failed trial at BIG_LOSS, which is finite: the loss must
    # be strictly below the penalty and the failure count must be zero.
    assert np.isfinite(loss) and loss < BIG_LOSS
    assert criterion.n_failures == 0

    sim = moment_fn(criterion.trial(theta))
    missing = sorted(k for k in data_moments if k not in sim)
    not_finite = sorted(k for k in data_moments if k in sim and not np.isfinite(sim[k]))
    assert missing == [], f"{factory}: simulated moments lack {missing}"
    assert not_finite == [], f"{factory}: NaN or inf simulated moments {not_finite}"


# ---------------------------------------------------------------------------
# (b) the moments respond to beta
# ---------------------------------------------------------------------------

def test_moments_respond_to_beta(criteria, spec):
    criterion, moment_fn, data_moments, _ = criteria["spec_factory.yaml"]
    theta_hi = {**_base_theta(spec), "beta": 0.945}
    theta_lo = {**_base_theta(spec), "beta": 0.90}
    m_hi = moment_fn(criterion.trial(theta_hi))
    m_lo = moment_fn(criterion.trial(theta_lo))
    # Level moments are in AUD (tens of thousands); 1e-3 is far below any
    # response, so the assertion is that the override reaches the solver at
    # all, not that the response has a particular size.
    max_abs_diff = max(abs(m_hi[k] - m_lo[k]) for k in data_moments)
    assert max_abs_diff > 1e-3, f"max |diff| = {max_abs_diff}"


# ---------------------------------------------------------------------------
# (c) the parameter-name guard
# ---------------------------------------------------------------------------

def _yaml_calibration_keys(*paths):
    keys = set()
    for p in paths:
        raw = yaml.safe_load(Path(p).read_text())
        keys |= set((raw.get("calibration") or {}).keys())
    return keys


def test_resolved_calibration_keys_match_the_calibration_files():
    est = _driver()
    cd_keys = est.resolved_calibration_keys(COBB_DOUGLAS, "spec_factory.yaml")
    assert cd_keys == _yaml_calibration_keys(COBB_DOUGLAS / "calibration" / "main.yaml")
    assert "rho" in cd_keys and "gamma_c" not in cd_keys and "gamma_h" not in cd_keys

    sep_keys = est.resolved_calibration_keys(SEPARABLE, "spec_factory.yaml")
    assert sep_keys == _yaml_calibration_keys(SEPARABLE / "calibration" / "main.yaml")
    sep_male_keys = est.resolved_calibration_keys(SEPARABLE, "spec_factory_males.yaml")
    assert sep_male_keys == _yaml_calibration_keys(
        SEPARABLE / "calibration" / "main.yaml",
        SEPARABLE / "calibration" / "main_males.yaml",
    )


def test_guard_rejects_unknown_calibration_name_and_passes_known_ones(spec):
    est = _driver()
    cd_keys = est.resolved_calibration_keys(COBB_DOUGLAS, "spec_factory.yaml")
    cd_free = ["beta", "alpha", "rho", "tau", "theta"]
    with pytest.raises(ValueError, match="gamma_c"):
        est.check_parameter_names(cd_keys, cd_free, {"t0": 20, "gamma_c": 3.0}, None)
    # A stale free name is caught as well.
    with pytest.raises(ValueError, match="gamma_h"):
        est.check_parameter_names(cd_keys, cd_free + ["gamma_h"], {"t0": 20}, None)

    sep_keys = est.resolved_calibration_keys(SEPARABLE, "spec_factory.yaml")
    est.check_parameter_names(sep_keys, list(spec["free"]), {"t0": 20}, None)
    est.check_parameter_names(sep_keys, list(spec["free"]), {"t0": 20, "tau": 0.1}, None)


def test_guard_types_rules():
    est = _driver()
    keys = {"beta", "alpha", "tau", "t0"}

    # The mapped names are free and are not calibration keys; the target is a
    # calibration key and is not free: accepted.
    est.check_parameter_names(keys, ["beta_bar", "sigma_beta", "alpha"], {"t0": 20}, TYPES_SPEC)

    # beta under free together with a types block naming it.
    with pytest.raises(ValueError, match=r"'beta'"):
        est.check_parameter_names(keys, ["beta", "beta_bar", "sigma_beta"], {}, TYPES_SPEC)

    # A mapped name that is a calibration key would shadow a model parameter.
    shadowing = {"beta": {"n": 4, "location": "alpha", "spread": "sigma_beta"}}
    with pytest.raises(ValueError, match=r"'alpha'"):
        est.check_parameter_names(keys, ["alpha", "sigma_beta"], {}, shadowing)

    # A mapped name may not reach the solver through the overrides.
    with pytest.raises(ValueError, match=r"'beta_bar'"):
        est.check_parameter_names(keys, ["beta_bar", "sigma_beta"], {"beta_bar": 0.9}, TYPES_SPEC)

    # A types target that is not a calibration key.
    stray = {"kappa": {"n": 4, "location": "kappa_bar", "spread": "sigma_kappa"}}
    with pytest.raises(ValueError, match=r"'kappa'"):
        est.check_parameter_names(keys, ["kappa_bar", "sigma_kappa"], {}, stray)

    # Every offending name is listed in one message.
    with pytest.raises(ValueError) as excinfo:
        est.check_parameter_names(keys, ["beta", "gamma_c"], {"gamma_h": 2.0}, TYPES_SPEC)
    message = str(excinfo.value)
    assert "'beta'" in message and "'gamma_c'" in message and "'gamma_h'" in message


# ---------------------------------------------------------------------------
# (d) load_types_spec, (e) refusal of types with self-generated data
# ---------------------------------------------------------------------------

def test_load_types_spec_is_none_for_the_existing_spec():
    est = _driver()
    raw = yaml.safe_load((SEPARABLE / "estimation" / SPEC_NAME).read_text())
    assert est.load_types_spec(raw["estimation"]) is None
    assert est.load_types_spec({}) is None
    assert est.load_types_spec({"types": TYPES_SPEC}) == TYPES_SPEC


def test_types_with_selfgen_data_are_refused_naming_both_keys():
    est = _driver()
    with pytest.raises(ValueError) as excinfo:
        est.check_types_data_source(TYPES_SPEC, {"data_source": "selfgen"})
    message = str(excinfo.value)
    assert "estimation.types" in message and "moments.data_source" in message
    # Accepted: types with precomputed data; self-generated data without types.
    est.check_types_data_source(TYPES_SPEC, {"data_source": "precomputed"})
    est.check_types_data_source(None, {"data_source": "selfgen"})


# ---------------------------------------------------------------------------
# (f) a types spec without a group never runs a single beta silently
# ---------------------------------------------------------------------------

def test_types_spec_without_group_raises_and_is_counted(spec, capsys):
    est = _driver()
    criterion, moment_fn, data_moments, denorm = est.build_criterion(
        SEPARABLE, spec, "spec_factory.yaml", None, dict(CALIB_OVERRIDES), dict(GRID),
        N_SIM, SIMULATION_SEED, None, types_spec=TYPES_SPEC, group=None,
    )
    theta = {**{k: v for k, v in _base_theta(spec).items() if k != "beta"},
             "beta_bar": 0.94, "sigma_beta": 0.1}
    with pytest.raises(NotImplementedError, match="Task E2"):
        criterion.trial(theta)
    assert criterion.n_failures == 1

    # Through kikku's closure the failure becomes the penalty loss, and the
    # driver has printed it before re-raising (the 28 March 2026 incident
    # was a silent NameError scored at BIG_LOSS).
    assert criterion(theta) == BIG_LOSS
    assert criterion.n_failures == 2
    out = capsys.readouterr().out
    assert "NotImplementedError" in out and "beta_bar" in out


# ---------------------------------------------------------------------------
# The saved example run (end-to-end run C) carries the new manifest fields
# ---------------------------------------------------------------------------

def test_saved_run_manifest_records_spec_factory_and_method():
    fixtures = REPO_ROOT / "tests" / "fixtures" / "estimation_run"
    runs = sorted(p for p in fixtures.glob("est_*") if p.is_dir())
    assert runs, f"no saved estimation run under {fixtures}"
    for run in runs:
        for fname in ("summary.json", "theta_best.json", "theta_mean.json",
                      "theta_se.json", "fit_table.csv", "convergence.csv",
                      "manifest.json"):
            assert (run / fname).is_file(), f"{run.name} lacks {fname}"
        manifest = yaml.safe_load((run / "manifest.json").read_text())
        assert manifest["spec_factory"] == "spec_factory.yaml"
        assert manifest["solver_method"] is None
        assert manifest["beta_types"] is None
        assert manifest["n_samples"] == 6 and manifest["n_elite"] == 2
        assert manifest["grid"] == GRID and manifest["N_sim"] == N_SIM
        assert "git_commit" in manifest
