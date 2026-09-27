"""Discount-factor types: distribution, birth draw, pooling, and the
bit-for-bit equivalence of the typed simulator with today's simulator.

Checks G and H of ``AI/devspecs/28092026/cd_estimation_and_postprocess.md``.
Oracles are closed forms written here as literals, hand-built arrays, and the
golden digests recorded from the untouched code (main tree, 28 Sep 2026) at
``n_a = n_h = n_w = 30``, ``t0 = 55``, ``N = 200``, ``seed = 99``.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from examples.durables.solve import solve  # noqa: E402
from examples.durables.solvers.simulate import (  # noqa: E402
    _base_stage,
    make_initial_particles,
    simulate_lifecycle,
    simulate_type_subset,
)
from examples.durables.solvers.beta_types import (  # noqa: E402
    discretise_beta,
    draw_types,
    expand_types,
    pool_by_type,
)

SYNTAX = REPO_ROOT / "examples" / "durables" / "syntax"

# Settings of the golden run (spec check H).
GRID = {"n_a": 30, "n_h": 30, "n_w": 30}
T0 = 55
N_GOLD = 200
SEED_GOLD = 99

# Golden digests: sha256 of np.ascontiguousarray(arr).tobytes(), with shape
# and dtype, recorded from the untouched code before simulate.py was changed
# (/tmp/fues-verify/golden_sim_digest.json, 28 Sep 2026, commit fb4c757).
GOLDEN = {
    "separable": {
        "a": ((70, 200), "float64", "1f2991626171cde5113601d75bca2789ddd74531dd28f38b3e0f09c303d60a1d"),
        "a_nxt": ((70, 200), "float64", "b4b3117dd9fb9145deabfa1876dd4bb4367428187c3213b8a57a664c0d001bcf"),
        "c": ((70, 200), "float64", "92f1547c362f0fd34a955375a9f531af148ea1291b320734a1405a4d2384f62b"),
        "discrete": ((70, 200), "int64", "96d2636b7f8cb08683b83b64d4492edf50953fec70f9208ef481064a200151c3"),
        "h": ((70, 200), "float64", "cbd5843301488e537efeef09bee11375044553170e4dd16fd0c9b911f088beca"),
        "h_nxt": ((70, 200), "float64", "a6d2eb6a688f78ed61192605b598e5d52e14803d151e2184a5e454098e201309"),
        "n_adj_periods": ((200,), "int64", "43ed205b25f3b1288c4f95079d00d8c5104a9102a2d80cd19e70c6269c06b2b1"),
        "n_keep_periods": ((200,), "int64", "bf677275ab9f96a5963290752cc0ae3a84e4ea3bd6eb137d8a9ec9d90201c400"),
        "npv_utility": ((200,), "float64", "297e04265da34a8899187a99a4174d05bfc682e44efa224552ac9d42148db828"),
        "npv_utility_adj": ((200,), "float64", "db9975b042dd3be6cd94cb0ecf872347e182eb566f56a5e3f31f3b3125553bb3"),
        "npv_utility_keep": ((200,), "float64", "040ca75133a087e076ae24d9d5728b991c27b90a7ef63751d25781cae8f9c20e"),
        "y": ((70, 200), "float64", "35f0a94e336098f0c4ee769446d81223e9824b5f9c3ed34cb98cb71dedd87fa0"),
        "z_idx": ((70, 200), "int64", "83e22a2cc3765037fa2b71e8d579d885b2e7e359fe1791e6ba66dc1a3a7ad0e3"),
    },
    "cobb_douglas": {
        "a": ((70, 200), "float64", "3dfa7cd694ef0ba7d646c4986b21d606f7c914e095f88112f940b3b72363056a"),
        "a_nxt": ((70, 200), "float64", "ed06ee988433f82acad87e302c54a705beba7cfc09ebc8921fee7c71352bbc2c"),
        "c": ((70, 200), "float64", "467372946f1762f1498ee1ef3bd775e42104e51b02c91aff195e052487f9f150"),
        "discrete": ((70, 200), "int64", "a44a402a668e33cf47afca8947fc420cd2b6d89e488fa644b40a0b7b4b887755"),
        "h": ((70, 200), "float64", "bc874839b7cf309a2dad86556e03633b0ac4be5ee7172329435076649e8fa28e"),
        "h_nxt": ((70, 200), "float64", "ea9dbb4e8135eb6ac8f96d23bf3610c20d10cfa723d1b9e82d0615c3d194b49d"),
        "n_adj_periods": ((200,), "int64", "1c61f9ed4507f94ffed5d6e9208620f3ce4eebe956112bf43f75f08d75122987"),
        "n_keep_periods": ((200,), "int64", "f9d339530e0eeee3584a80c1160c9a0b9196cb1e05005cabe7f10d97fae466cf"),
        "npv_utility": ((200,), "float64", "2e43ccf8078dbf266cae1f5da5e5049457eac2a787b0a74700ef5895b71f51ca"),
        "npv_utility_adj": ((200,), "float64", "e520883bf38d4f1f9f95f408d2d01b46133dcb0ea1bf8c4efb00e16502f95121"),
        "npv_utility_keep": ((200,), "float64", "df9319089a3aeaa2f246d44aed39d348c4b1486111c6c681c0d4eeaf7291f42e"),
        "y": ((70, 200), "float64", "50d6e0cf1fd23a2422b7774c7864aee0c211e0e5a862faf443b418c080afe19e"),
        "z_idx": ((70, 200), "int64", "aba248c9d38619c967653d7730219a657d923c269d4cac34bcde378c1d041763"),
    },
}

# The four equiprobable standard-normal quantiles, norm.ppf((k + 0.5) / 4),
# written to ten decimals so the oracle does not call the function under test.
Z4 = np.array([-1.1503493804, -0.3186393639, 0.3186393639, 1.1503493804])

# Ten-decimal quantiles carry an error below 5e-11 in x; with sigma_beta = 0.2
# and d beta / d x = -beta (1 - beta) < 0.25 the node error is below 3e-12.
# 1e-10 leaves a margin of thirty over that.
TOL_CLOSED_FORM = 1e-10


def closed_form_nodes(beta_bar, sigma_beta):
    """Spec 5.6 written out: x_bar = ln(1/beta_bar - 1), beta = 1/(1+exp(x)).

    Returned increasing in beta (the quantiles are increasing in x, and beta
    is decreasing in x, so the order is reversed).
    """
    x_bar = np.log(1.0 / beta_bar - 1.0)
    beta_desc = 1.0 / (1.0 + np.exp(x_bar + sigma_beta * Z4))
    return beta_desc[::-1]


def sha256_of(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def solve_registry(registry, **calibration):
    nest, grids = solve(
        str(SYNTAX / registry),
        draw={"calibration": {"t0": T0, **calibration}, "settings": dict(GRID)},
        verbose=False,
    )
    return nest, grids


@pytest.fixture(scope="module")
def solved():
    """One solved model per registry at the golden settings."""
    return {reg: solve_registry(reg) for reg in ("separable", "cobb_douglas")}


@pytest.fixture(scope="module")
def solved_two_betas():
    """Two separable models that differ only in beta (0.90 and 0.97)."""
    return {
        0.90: solve_registry("separable", beta=0.90),
        0.97: solve_registry("separable", beta=0.97),
    }


# --------------------------------------------------------------------------
# G: type distribution, expansion, birth draw, pooling (no solve)
# --------------------------------------------------------------------------

def test_discretise_beta_matches_closed_form():
    nodes, shares = discretise_beta(0.94, 0.2, 4)
    expected = closed_form_nodes(0.94, 0.2)
    assert nodes.dtype == np.float64
    assert nodes.shape == (4,)
    np.testing.assert_allclose(nodes, expected, rtol=0, atol=TOL_CLOSED_FORM)
    assert np.all(np.diff(nodes) > 0)
    # 1/4 is exactly representable, so the shares are exact.
    assert np.array_equal(shares, np.array([0.25, 0.25, 0.25, 0.25]))


@pytest.mark.parametrize("beta_bar", [0.88, 0.94, 0.99])
def test_discretise_beta_zero_spread_is_exact(beta_bar):
    # Review focus item 3: sigma_beta = 0.0 exactly (a clipped candidate)
    # must give n identical types equal to beta_bar, not NaN.
    nodes, shares = discretise_beta(beta_bar, 0.0, 4)
    assert np.array_equal(nodes, np.full(4, beta_bar))
    assert np.array_equal(shares, np.full(4, 0.25))
    # At sigma_beta = 1e-9 the nodes move by at most
    # beta (1 - beta) * sigma_beta * |z| < 0.25 * 1e-9 * 1.16 < 1e-9.
    nodes_small, _ = discretise_beta(beta_bar, 1e-9, 4)
    assert np.all(np.abs(nodes_small - beta_bar) <= 1e-9)
    assert np.all(np.isfinite(nodes_small))


@pytest.mark.parametrize("beta_bar", [0.88, 0.94, 0.99])
@pytest.mark.parametrize("sigma_beta", [0.0, 0.05, 0.3, 0.5])
def test_discretise_beta_nodes_increasing_inside_unit_interval(beta_bar, sigma_beta):
    nodes, shares = discretise_beta(beta_bar, sigma_beta, 4)
    assert nodes.shape == (4,) and shares.shape == (4,)
    assert np.all(nodes > 0.0) and np.all(nodes < 1.0)
    if sigma_beta > 0.0:
        assert np.all(np.diff(nodes) > 0.0)
    else:
        assert np.all(np.diff(nodes) == 0.0)
    assert np.array_equal(shares, np.full(4, 0.25))
    np.testing.assert_allclose(
        nodes, closed_form_nodes(beta_bar, sigma_beta), rtol=0, atol=TOL_CLOSED_FORM)


def test_expand_types_drops_location_and_spread_and_sets_beta():
    theta = {"beta_bar": 0.94, "sigma_beta": 0.2, "alpha": 0.6, "tau": 0.12}
    types_spec = {"beta": {"n": 4, "location": "beta_bar", "spread": "sigma_beta"}}
    records = expand_types(theta, types_spec)
    assert len(records) == 4
    expected = closed_form_nodes(0.94, 0.2)
    for k, (share, calib) in enumerate(records):
        assert share == 0.25
        assert set(calib) == {"alpha", "tau", "beta"}
        assert calib["alpha"] == 0.6 and calib["tau"] == 0.12
        assert abs(calib["beta"] - expected[k]) <= TOL_CLOSED_FORM
    # theta is not mutated
    assert theta == {"beta_bar": 0.94, "sigma_beta": 0.2, "alpha": 0.6, "tau": 0.12}

    with pytest.raises(ValueError):
        expand_types({"beta_bar": 0.94, "alpha": 0.6}, types_spec)
    with pytest.raises(ValueError):
        expand_types({"sigma_beta": 0.2, "alpha": 0.6}, types_spec)
    with pytest.raises(ValueError):
        expand_types({**theta, "beta": 0.95}, types_spec)


def test_draw_types_frequencies_and_determinism():
    N = 10000
    shares = np.full(4, 0.25)
    types = draw_types(N, shares, 99)
    assert types.dtype == np.int64
    assert types.shape == (N,)
    assert types.min() >= 0 and types.max() <= 3
    freq = np.bincount(types, minlength=4) / N
    # Binomial standard deviation at N = 10000 is 0.0043; 0.02 is 4.6 of them.
    assert np.all(np.abs(freq - 0.25) < 0.02), freq
    assert np.array_equal(types, draw_types(N, shares, 99))
    # A different seed gives a different assignment.
    assert not np.array_equal(types, draw_types(N, shares, 100))


def _chi2_independence(a, b, n_a, n_b):
    table = np.zeros((n_a, n_b), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    res = stats.chi2_contingency(table)
    return float(res.statistic), float(res.pvalue)


def test_draw_types_independent_of_initial_income_state(solved):
    """The birth draw must not be related to the initial income draw.

    ``make_initial_particles`` seeds ``default_rng(seed + 1)``; ``draw_types``
    uses ``SeedSequence(seed, spawn_key=(1,))``, a separate stream. Independence
    is tested within an agent (type of i against z_idx of i) and across the
    neighbouring agents that a shared PCG64 stream would tie together (type of
    i against z_idx of 2i and 2i + 1; see the collision test below).
    """
    nest, grids = solved["separable"]
    N = 10000
    n_z = len(grids["z"])
    shares = np.full(4, 0.25)
    z0 = make_initial_particles(N, grids, nest, seed=99)["z_idx"]
    types = draw_types(N, shares, 99)
    stat, p = _chi2_independence(types, z0, 4, n_z)
    assert p > 0.01, (stat, p)
    half = N // 2
    i = np.arange(half)
    for shift in (0, 1):
        stat_x, p_x = _chi2_independence(types[:half], z0[2 * i + shift], 4, n_z)
        assert p_x > 0.01, (shift, stat_x, p_x)


def test_seed_plus_one_collision_is_detected(solved):
    """The failure the spawn_key stream guards against.

    A birth draw taken from ``default_rng(seed + 1)`` shares its PCG64 stream
    with the initial income draw of ``make_initial_particles``. The two
    consumers read that stream differently: ``Generator.choice`` draws bounded
    integers through 32-bit halves of successive 64-bit words (low half first),
    while ``Generator.random`` takes whole words. Agent i's collided type is
    therefore a function of the top bits of word i, which also set the income
    state of agent 2i + 1; the contingency table of type[i] against
    z_idx[2i + 1] is diagonal. Within an agent (type[i] against z_idx[i]) the
    same statistic does not reject: observed chi-square 10.12 on 9 degrees of
    freedom, p = 0.34, on 28 Sep 2026 with numpy 2.3.5. The dependence the
    collision creates is across neighbouring agents, and that is what is
    asserted here; the spawn_key stream is tested against both pairings above.
    """
    nest, grids = solved["separable"]
    N = 10000
    n_z = len(grids["z"])
    shares = np.full(4, 0.25)
    z0 = make_initial_particles(N, grids, nest, seed=99)["z_idx"]
    u = np.random.default_rng(99 + 1).random(N)
    types_collided = np.minimum(
        np.searchsorted(np.cumsum(shares), u, side="right"), 3).astype(np.int64)
    half = N // 2
    i = np.arange(half)
    stat_cross, p_cross = _chi2_independence(
        types_collided[:half], z0[2 * i + 1], 4, n_z)
    assert p_cross < 0.01, (stat_cross, p_cross)


def test_pool_by_type_restores_rows_and_marks_type():
    # Hand-made truth: agent i has type type_idx[i]; every array is a known
    # function of (t, i), so a wrong scatter changes a value.
    N, T = 7, 3
    type_idx = np.array([0, 2, 1, 0, 2, 2, 1], dtype=np.int64)
    betas = np.array([0.90, 0.94, 0.97])
    t = np.arange(T)[:, None]
    i = np.arange(N)[None, :]
    truth = {
        "c": (100.0 * t + i).astype(np.float64),                 # (T, N) float
        "z_idx": (-10 * (t + 1) - i).astype(np.int64),            # (T, N) int
        "npv_utility": (1.5 * np.arange(N)).astype(np.float64),  # (N,) float
        "n_adj_periods": np.arange(N, dtype=np.int64),           # (N,) int
    }
    parts = []
    for k in range(3):
        idx = np.flatnonzero(type_idx == k)
        parts.append((idx, {key: arr[..., idx] for key, arr in truth.items()}))

    pooled = pool_by_type(parts, type_idx, betas, N)
    for key, arr in truth.items():
        assert pooled[key].dtype == arr.dtype, key
        assert pooled[key].shape == arr.shape, key
        assert np.array_equal(pooled[key], arr), key
    assert pooled["beta_type"].dtype == np.int64
    assert np.array_equal(pooled["beta_type"], type_idx)
    assert pooled["beta"].dtype == np.float64
    assert np.array_equal(pooled["beta"], betas[type_idx])
    assert set(pooled) == set(truth) | {"beta_type", "beta"}

    # A type with no agents contributes empty arrays and changes nothing.
    empty = (np.zeros(0, dtype=np.int64),
             {key: arr[..., :0] for key, arr in truth.items()})
    pooled4 = pool_by_type(parts + [empty], type_idx, np.append(betas, 0.99), N)
    for key, arr in truth.items():
        assert np.array_equal(pooled4[key], arr), key
    assert np.array_equal(pooled4["beta_type"], type_idx)

    # Parts that do not cover every agent are refused.
    with pytest.raises(ValueError):
        pool_by_type(parts[:2], type_idx, betas, N)


# --------------------------------------------------------------------------
# H: the typed simulator is bit-identical to today's when types are absent
# --------------------------------------------------------------------------

def _assert_matches_golden(sim, registry):
    gold = GOLDEN[registry]
    for key, (shape, dtype, digest) in gold.items():
        arr = sim[key]
        assert arr.shape == shape, (key, arr.shape)
        assert str(arr.dtype) == dtype, (key, arr.dtype)
        assert sha256_of(arr) == digest, key


@pytest.mark.parametrize("registry", ["separable", "cobb_douglas"])
def test_golden_digest_without_types(solved, registry):
    nest, grids = solved[registry]
    sim = simulate_lifecycle(nest, grids, N=N_GOLD, seed=SEED_GOLD)
    assert set(sim) == set(GOLDEN[registry])
    _assert_matches_golden(sim, registry)


@pytest.mark.parametrize("registry", ["separable", "cobb_douglas"])
def test_golden_digest_with_a_single_type(solved, registry):
    nest, grids = solved[registry]
    beta = float(_base_stage(nest).calibration["beta"])
    sim = simulate_lifecycle(nest, grids, N=N_GOLD, seed=SEED_GOLD,
                             types=[(1.0, nest, grids)])
    assert set(sim) == set(GOLDEN[registry]) | {"beta_type", "beta"}
    _assert_matches_golden(sim, registry)
    assert sim["beta_type"].dtype == np.int64
    assert np.array_equal(sim["beta_type"], np.zeros(N_GOLD, dtype=np.int64))
    assert sim["beta"].dtype == np.float64
    assert np.array_equal(sim["beta"], np.full(N_GOLD, beta))


@pytest.mark.parametrize("registry", ["separable", "cobb_douglas"])
def test_golden_digest_via_subset_and_pool(solved, registry):
    nest, grids = solved[registry]
    beta = float(_base_stage(nest).calibration["beta"])
    everyone = np.arange(N_GOLD)
    part = simulate_type_subset(nest, grids, everyone, N_GOLD, SEED_GOLD)
    assert set(part) == set(GOLDEN[registry])
    _assert_matches_golden(part, registry)
    pooled = pool_by_type([(everyone, part)], np.zeros(N_GOLD, dtype=np.int64),
                          [beta], N_GOLD)
    _assert_matches_golden(pooled, registry)


def test_initial_particles_do_not_depend_on_beta(solved_two_betas):
    # Spec 5.6, assumption to confirm: the initial conditions drawn from any
    # type's nest are the same, so each type may draw its own.
    nest_lo, grids_lo = solved_two_betas[0.90]
    nest_hi, grids_hi = solved_two_betas[0.97]
    p_lo = make_initial_particles(N_GOLD, grids_lo, nest_lo, seed=SEED_GOLD)
    p_hi = make_initial_particles(N_GOLD, grids_hi, nest_hi, seed=SEED_GOLD)
    assert set(p_lo) == set(p_hi) == {"a", "h", "z_idx", "_idx"}
    for key in p_lo:
        assert p_lo[key].dtype == p_hi[key].dtype, key
        assert np.array_equal(p_lo[key], p_hi[key]), key


def test_two_types_walk_each_agent_with_its_own_policy(solved_two_betas):
    nest_lo, grids_lo = solved_two_betas[0.90]
    nest_hi, grids_hi = solved_two_betas[0.97]
    assert float(_base_stage(nest_lo).calibration["beta"]) == 0.90
    assert float(_base_stage(nest_hi).calibration["beta"]) == 0.97
    types = [(0.5, nest_lo, grids_lo), (0.5, nest_hi, grids_hi)]

    sim = simulate_lifecycle(nest_lo, grids_lo, N=N_GOLD, seed=SEED_GOLD, types=types)
    type_idx = draw_types(N_GOLD, np.array([0.5, 0.5]), SEED_GOLD)
    assert np.array_equal(sim["beta_type"], type_idx)
    assert set(np.unique(sim["beta"]).tolist()) == {0.90, 0.97}
    assert np.array_equal(sim["beta"], np.where(type_idx == 0, 0.90, 0.97))
    assert 0 < type_idx.sum() < N_GOLD  # both types present

    # Oracle: each type's agents equal, agent by agent, the single-beta
    # simulation at that beta (same shocks and initial conditions for a given
    # agent whatever its type).
    sim_lo = simulate_lifecycle(nest_lo, grids_lo, N=N_GOLD, seed=SEED_GOLD)
    sim_hi = simulate_lifecycle(nest_hi, grids_hi, N=N_GOLD, seed=SEED_GOLD)
    for k, single in ((0, sim_lo), (1, sim_hi)):
        idx = np.flatnonzero(type_idx == k)
        for key in GOLDEN["separable"]:
            assert sim[key].dtype == single[key].dtype, key
            assert np.array_equal(sim[key][..., idx], single[key][..., idx],
                                  equal_nan=True), (k, key)

    # The two types consume differently: the pooled panel is not the panel of
    # either single-beta model.
    for single in (sim_lo, sim_hi):
        diff = np.abs(sim["c"] - single["c"])
        assert np.nanmax(diff) > 1e-6
