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

A second group runs the same grid and sample on the Cobb-Douglas registry
(``baseline_large_egm.yaml`` / ``baseline_large_egm_males.yaml``): one
finite evaluation with every target moment finite (70 keys on the female
factory; 66 on the male factory, where four age-group columns are absent
from the precomputed CSV), sensitivity of those moments to ``rho``, and
the start-up guard rejecting ``gamma_c``.
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

# The generalised types block (spec 5.6): n members; member k takes the k-th
# equiprobable quantile of every listed parameter on its transform scale.
TYPES_SPEC = {
    "n": 4,
    "parameters": {
        "beta": {"location": "beta_bar", "spread": "sigma_beta", "transform": "logit"},
    },
}

# The two equiprobable normal quantiles for K = 2, Phi^{-1}(0.25) and
# Phi^{-1}(0.75), written as numbers so the oracle is independent of the code.
Z2 = np.array([-0.6744897501960817, 0.6744897501960817])


def _logit_nodes(beta_bar, sigma_beta):
    """Closed form of the type nodes on the logit scale, increasing in beta."""
    x = np.log(1.0 / beta_bar - 1.0) + sigma_beta * Z2
    return np.sort(1.0 / (1.0 + np.exp(x)))


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
    shadowing = {"n": 4, "parameters": {
        "beta": {"location": "alpha", "spread": "sigma_beta"}}}
    with pytest.raises(ValueError, match=r"'alpha'"):
        est.check_parameter_names(keys, ["alpha", "sigma_beta"], {}, shadowing)

    # A mapped name may not reach the solver through the overrides.
    with pytest.raises(ValueError, match=r"'beta_bar'"):
        est.check_parameter_names(keys, ["beta_bar", "sigma_beta"], {"beta_bar": 0.9}, TYPES_SPEC)

    # A types target that is not a calibration key.
    stray = {"n": 4, "parameters": {
        "kappa": {"location": "kappa_bar", "spread": "sigma_kappa"}}}
    with pytest.raises(ValueError, match=r"'kappa'"):
        est.check_parameter_names(keys, ["kappa_bar", "sigma_kappa"], {}, stray)

    # Two heterogeneous parameters: every location and spread is mapped.
    two = {"n": 4, "parameters": {
        "beta": {"location": "beta_bar", "spread": "sigma_beta"},
        "alpha": {"location": "alpha_bar", "spread": "sigma_alpha"}}}
    est.check_parameter_names(
        keys, ["beta_bar", "sigma_beta", "alpha_bar", "sigma_alpha", "tau"], {"t0": 20}, two)
    with pytest.raises(ValueError, match=r"'alpha'"):
        est.check_parameter_names(keys, ["beta_bar", "sigma_beta", "alpha_bar", "sigma_alpha", "alpha"], {}, two)

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

    # The committed types spec of each registry loads with n = 4 and beta.
    for registry in (SEPARABLE, COBB_DOUGLAS):
        raw_types = yaml.safe_load(
            (registry / "estimation" / "baseline_large_egm_types.yaml").read_text())
        block = est.load_types_spec(raw_types["estimation"])
        assert block["n"] == 4 and list(block["parameters"]) == ["beta"]
        assert block["parameters"]["beta"]["location"] == "beta_bar"
        assert block["parameters"]["beta"]["spread"] == "sigma_beta"
        assert block["parameters"]["beta"]["transform"] == "logit"
        assert set(raw_types["estimation"]["free"]) >= {"beta_bar", "sigma_beta"}
        assert "beta" not in raw_types["estimation"]["free"]

    # n is required, as is a non-empty parameters mapping with location and
    # spread for every entry.
    with pytest.raises(ValueError, match=r"'n'"):
        est.load_types_spec({"types": {"parameters": TYPES_SPEC["parameters"]}})
    with pytest.raises(ValueError, match="parameters"):
        est.load_types_spec({"types": {"n": 4}})
    with pytest.raises(ValueError, match="spread"):
        est.load_types_spec({"types": {"n": 4, "parameters": {"beta": {"location": "beta_bar"}}}})


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
# (f) a types spec without a group solves the K members in sequence
# ---------------------------------------------------------------------------

def test_types_spec_without_group_solves_the_members_in_sequence(spec, capsys):
    """Serial mode (no communicator): K solves inside one trial, then pooling.

    Oracle: the two node values are the closed form ``_logit_nodes``; every
    agent's ``beta`` equals the node of its ``type_idx``; both types are
    present among 500 agents with equal shares; the panels keep the
    single-model shape ``(T, N)`` so the moment function applies unchanged.
    """
    est = _driver()
    two_types = {"n": 2, "parameters": TYPES_SPEC["parameters"]}
    criterion, moment_fn, data_moments, denorm = est.build_criterion(
        SEPARABLE, spec, "spec_factory.yaml", None, dict(CALIB_OVERRIDES), dict(GRID),
        N_SIM, SIMULATION_SEED, None, types_spec=two_types, group=None,
    )
    theta = {**{k: v for k, v in _base_theta(spec).items() if k != "beta"},
             "beta_bar": 0.94, "sigma_beta": 0.1}

    panels = criterion.trial(theta)
    assert criterion.n_failures == 0
    assert panels["c"].shape == (70, N_SIM)
    assert panels["type_idx"].dtype == np.int64
    counts = np.bincount(panels["type_idx"], minlength=2)
    assert counts.shape == (2,) and counts.min() > 0 and counts.sum() == N_SIM
    # The two node values agree with the closed form to 1e-12: the driver's
    # nodes come from the algebraically identical form
    # beta_bar / (beta_bar + (1 - beta_bar) exp(sigma z)), which differs from
    # the literal 1/(1 + exp(x)) by one unit in the last place (spec 5.6).
    nodes = _logit_nodes(0.94, 0.1)
    pooled_nodes = np.unique(panels["beta"])
    np.testing.assert_allclose(pooled_nodes, nodes, rtol=0, atol=1e-12)
    # Every agent carries exactly the node of its type (nodes increase in k).
    assert np.array_equal(panels["beta"], pooled_nodes[panels["type_idx"]])
    out = capsys.readouterr().out
    assert "[types] serial member solve times" in out

    loss = criterion(theta)
    assert np.isfinite(loss) and loss < BIG_LOSS
    assert criterion.n_failures == 0

    # A candidate that carries beta beside the location and spread is a
    # failure, printed and counted before kikku scores it at the penalty.
    assert criterion({**theta, "beta": 0.95}) == BIG_LOSS
    assert criterion.n_failures == 1
    out = capsys.readouterr().out
    assert "[trial failure]" in out and "ValueError" in out and "'beta'" in out


# ---------------------------------------------------------------------------
# Cobb-Douglas registry: finite loss, rho sensitivity, gamma_c guard
# ---------------------------------------------------------------------------

CD_SPEC_NAMES = {
    "spec_factory.yaml": "baseline_large_egm.yaml",
    "spec_factory_males.yaml": "baseline_large_egm_males.yaml",
}

# Calibration values from cobb_douglas/calibration/main.yaml; rho is the
# name that must reach the solver (the gamma_c incident of Section 2).
CD_THETA = {
    "beta": 0.945,
    "alpha": 0.7,
    "rho": 2.0,
    "tau": 0.12,
    "theta": 1.3498,
}

# Filtered precomputed keys after build_criterion's target-prefix match:
# the female CSV contributes 70; the male CSV contributes 66 because four
# age-group columns present for the _0 (female) suffixes are absent for _1.
N_TARGET_MOMENTS = {
    "spec_factory.yaml": 70,
    "spec_factory_males.yaml": 66,
}


@pytest.fixture(scope="module")
def cd_specs():
    return {
        factory: load_estimation_spec(str(COBB_DOUGLAS / "estimation" / name))
        for factory, name in CD_SPEC_NAMES.items()
    }


@pytest.fixture(scope="module")
def cd_criteria(cd_specs):
    """The driver's criterion for each Cobb-Douglas calibration chain."""
    est = _driver()
    return {
        factory: est.build_criterion(
            COBB_DOUGLAS, cd_specs[factory], factory, None, dict(CALIB_OVERRIDES), dict(GRID),
            N_SIM, SIMULATION_SEED, None,
        )
        for factory in FACTORIES
    }


@pytest.fixture(scope="module")
def cd_evals(cd_criteria):
    """One solve per factory at rho=2.0; one extra trial at rho=3.0.

    The rho=2.0 ``criterion`` call is the evaluation reused by the finite-loss
    check; the rho=3.0 trial is only for the sensitivity comparison.
    """
    out = {}
    for factory, (criterion, moment_fn, data_moments, denorm) in cd_criteria.items():
        loss = criterion(dict(CD_THETA))
        moments_rho2 = dict(criterion.last_sim_moments)
        moments_rho3 = moment_fn(criterion.trial({**CD_THETA, "rho": 3.0}))
        out[factory] = {
            "criterion": criterion,
            "data_moments": data_moments,
            "denorm": denorm,
            "loss": loss,
            "moments_rho2": moments_rho2,
            "moments_rho3": moments_rho3,
        }
    return out


@pytest.mark.parametrize("factory", FACTORIES)
def test_cd_one_evaluation_is_finite_with_70_finite_targets(cd_evals, factory):
    ev = cd_evals[factory]
    assert ev["denorm"] == pytest.approx(DENORM_EXPECTED)
    assert len(ev["data_moments"]) == N_TARGET_MOMENTS[factory]
    assert np.isfinite(ev["loss"]) and ev["loss"] < BIG_LOSS
    assert ev["criterion"].n_failures == 0

    sim = ev["moments_rho2"]
    missing = sorted(k for k in ev["data_moments"] if k not in sim)
    not_finite = sorted(
        k for k in ev["data_moments"] if k in sim and not np.isfinite(sim[k])
    )
    assert missing == [], f"{factory}: simulated moments lack {missing}"
    assert not_finite == [], f"{factory}: NaN or inf simulated moments {not_finite}"


@pytest.mark.parametrize("factory", FACTORIES)
def test_cd_moments_respond_to_rho(cd_evals, factory):
    ev = cd_evals[factory]
    m_lo = ev["moments_rho2"]
    m_hi = ev["moments_rho3"]
    max_abs_diff = max(abs(m_lo[k] - m_hi[k]) for k in ev["data_moments"])
    assert max_abs_diff > 1e-3, f"{factory}: max |diff| = {max_abs_diff}"


@pytest.mark.parametrize("factory", FACTORIES)
def test_cd_guard_rejects_gamma_c_as_calibration_override(cd_specs, factory):
    est = _driver()
    keys = est.resolved_calibration_keys(COBB_DOUGLAS, factory)
    with pytest.raises(ValueError, match="gamma_c"):
        est.check_parameter_names(
            keys, list(cd_specs[factory]["free"]), {"t0": 20, "gamma_c": 3.0}, None,
        )


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
