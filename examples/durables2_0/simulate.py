"""Forward simulation and post-hoc Euler evaluation for durables2_0.

Orchestrator — a thin combinator that composes:

  1. **Stage forward operators** from ``horses/`` into per-period
     bundles (pure simulation; no Euler side channels).
  2. **kikku.asva.simulate.simulate** for the topology-driven
     lifecycle walk.
  3. **Records-to-panels** transform for the output arrays.

Euler errors are evaluated after simulation via
:func:`evaluate_euler_c` and :func:`evaluate_euler_h` using the
same scalar kernels as before (:func:`keeper_euler`,
:func:`adjuster_euler`).
"""

import numpy as np
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
from dcsmm.fues.helpers.math_funcs import interp_as_scalar

from kikku.asva.simulate import simulate, draw_shocks

from .horses.branching import make_tenure_forward
from .horses.keeper_egm import make_keeper_forward
from .horses.adjuster_egm import make_adjuster_forward


def _base_stage(nest):
    """First period in the accretion list (terminal age); shared period calibration."""
    return nest["periods"][0]["stages"]["keeper_cons"]


# ------------------------------------------------------------------
# Scalar primitives: policy evaluators + Euler kernels
#
# These are the per-agent operations that the forward operators
# in horses/ compose into vectorised stage callables.
# ------------------------------------------------------------------

def _eval_adj(a, h, iz, adj_t, UG):
    """Interpolate tenure discrete choice at (a, h)."""
    pt = np.array([a, h])
    adj_val = eval_linear(UG, adj_t[iz], pt, xto.LINEAR)
    return int(min(max(round(adj_val), 0), 1))


def _eval_keeper_c(w, h, iz, C_t, UG):
    """Interpolate keeper consumption at (w, h)."""
    pt = np.array([w, h])
    c = eval_linear(UG, C_t[iz], pt, xto.LINEAR)
    return max(c, 1e-10)


def keeper_euler(c, a_nxt, h_nxt, iz,
                 edata_next, grids, callables):
    """Keeper consumption Euler error (scalar, one agent).

    FOC: du_c(c) = E_z'[beta * R * du_c(c')]
    """
    du_c = callables['keeper_cons']['d_c_u']
    euler_error_c = callables['keeper_cons']['euler_error_c']
    marginal_a = callables['tenure']['marginalBellman_d_a']
    Pi = grids['Pi']
    z_vals = grids['z']
    UG = grids['UGgrid_all']
    we_grid = grids['we']
    n_z = len(z_vals)

    income_trans_next = edata_next['income_trans']
    age_next = edata_next['age']
    adj_next = edata_next['adj']
    C_keep_next = edata_next['C_keep']
    C_adj_next = edata_next['C_adj']

    rhs = 0.0
    for iz2 in range(n_z):
        prob = Pi[iz, iz2]
        z2 = z_vals[iz2]
        d2 = _eval_adj(a_nxt, h_nxt, iz2, adj_next, UG)
        if d2 == 1:
            w2 = income_trans_next['adj_w'](a_nxt, h_nxt, z2, age_next)
            c2 = max(interp_as_scalar(we_grid, C_adj_next[iz2], w2), 1e-10)
        else:
            w2 = income_trans_next['keep_w'](a_nxt, z2, age_next)
            c2 = _eval_keeper_c(w2, h_nxt, iz2, C_keep_next, UG)
        rhs += prob * marginal_a(du_c(max(c2, 1e-10)))

    return euler_error_c(c, rhs)


def adjuster_euler(c, h_nxt, a_nxt, iz,
                   edata_next, grids, callables):
    """Adjuster housing Euler error (scalar, one agent).

    InvEuler FOC2 (adjuster_cons): lhs matches invEuler_foc_h_rhs.
    """
    euler_error_h = callables['adjuster_cons']['euler_error_h']
    UG = grids['UGgrid_all']

    d_hV_arvl = edata_next['d_hV_arvl']
    pt = np.array([a_nxt, h_nxt])
    phi = eval_linear(UG, d_hV_arvl[iz], pt, xto.LINEAR)
    if np.isnan(phi) or np.isinf(phi):
        phi = 0.0

    return euler_error_h(c, h_nxt, phi)


# ------------------------------------------------------------------
# Compile: nest solutions -> per-period forward operator bundles
#
#   period[t] = tenure(D_t, trans_t)
#               ∘ { keep -> keeper(C_t)
#                   adj  -> adjuster(C_t, H_t) }
# ------------------------------------------------------------------

def build_period_pushforwards(nest, grids):
    """Per-period forward operator bundles (pure simulation)."""
    sol_by_t = {sol['t']: sol for sol in nest['solutions']}

    pushforward_by_t = {}
    for t, sol in sol_by_t.items():
        c_t = sol["callables"]
        period_stages = nest["periods"][sol["h"]]["stages"]

        pushforward_by_t[t] = {
            'tenure': make_tenure_forward(
                sol['tenure']['dcsn']['adj'], c_t, grids, t),
            'keeper_cons': make_keeper_forward(
                sol['keeper_cons']['dcsn']['c'], c_t, grids,
                period_stages["keeper_cons"]),
            'adjuster_cons': make_adjuster_forward(
                sol['adjuster_cons']['dcsn']['c'],
                sol['adjuster_cons']['dcsn']['h_choice'],
                c_t, grids, period_stages["adjuster_cons"]),
        }

    return pushforward_by_t


def _build_lookahead(sol_next):
    """Next-period policy bundle for Euler kernels (post-hoc)."""
    c_next = sol_next["callables"]
    income_trans = {
        "keep_w": c_next["tenure"]["transitions"]["keep_w"],
        "adj_w": c_next["tenure"]["transitions"]["adj_w"],
    }
    return {
        'adj': sol_next['tenure']['dcsn']['adj'],
        'C_keep': sol_next['keeper_cons']['dcsn']['c'],
        'C_adj': sol_next['adjuster_cons']['dcsn']['c'],
        'H_adj': sol_next['adjuster_cons']['dcsn']['h_choice'],
        'd_hV_arvl': sol_next['tenure']['arvl']['d_hV'],
        'income_trans': income_trans,
        'age': sol_next['t'],
    }


def evaluate_euler_c(sim_data, nest, grids):
    """Consumption Euler error for all agents (log10 relative), shape (T, N)."""
    T, N = sim_data['c'].shape
    cal = _base_stage(nest).calibration
    t0 = int(cal["t0"])
    euler_c = np.full((T, N), np.nan)
    sol_by_t = {sol['t']: sol for sol in nest['solutions']}

    for t in range(t0, T - 1):
        if t not in sol_by_t or (t + 1) not in sol_by_t:
            continue
        edata_next = _build_lookahead(sol_by_t[t + 1])
        callables_t = sol_by_t[t]["callables"]

        for i in range(N):
            c_ti = sim_data['c'][t, i]
            a_ti = sim_data['a_nxt'][t, i]
            h_ti = sim_data['h_nxt'][t, i]
            z_ti = int(sim_data['z_idx'][t, i])
            if np.isnan(c_ti) or c_ti <= 0.1 or a_ti <= 0.1:
                continue
            euler_c[t, i] = keeper_euler(
                c_ti, a_ti, h_ti, z_ti,
                edata_next, grids, callables_t)

    return euler_c


def evaluate_euler_h(sim_data, nest, grids):
    """Housing FOC error for adjusters only (log10 relative), shape (T, N)."""
    T, N = sim_data['c'].shape
    cal = _base_stage(nest).calibration
    t0 = int(cal["t0"])
    euler_h = np.full((T, N), np.nan)
    sol_by_t = {sol['t']: sol for sol in nest['solutions']}

    for t in range(t0, T - 1):
        if t not in sol_by_t or (t + 1) not in sol_by_t:
            continue
        edata_next = _build_lookahead(sol_by_t[t + 1])
        callables_t = sol_by_t[t]["callables"]

        for i in range(N):
            if sim_data['discrete'][t, i] != 1:
                continue
            c_ti = sim_data['c'][t, i]
            a_ti = sim_data['a_nxt'][t, i]
            h_ti = sim_data['h_nxt'][t, i]
            z_ti = int(sim_data['z_idx'][t, i])
            if np.isnan(c_ti) or c_ti <= 0.1 or a_ti <= 0.1:
                continue
            euler_h[t, i] = adjuster_euler(
                c_ti, h_ti, a_ti, z_ti,
                edata_next, grids, callables_t)

    return euler_h


# ------------------------------------------------------------------
# Exogenous shock: inter-period Markov z draw
# ------------------------------------------------------------------

def _make_markov_draw(grids):
    """Build Markov chain draw callable."""
    Pi_cumsum = np.cumsum(grids['Pi'], axis=1)

    def markov_draw(particles, draws_next):
        z_idx = particles['z_idx']
        u = draws_next.get('markov', np.random.random(len(z_idx)))
        N = len(z_idx)
        z_new = np.empty(N, dtype=np.int64)
        for i in range(N):
            z_new[i] = np.searchsorted(
                Pi_cumsum[int(z_idx[i])], u[i])
        return {'z_idx': z_new}

    return markov_draw


# ------------------------------------------------------------------
# Records -> panels: pure reshape of kikku history into (T, N) arrays
#
# Per-period, per-stage records become per-variable lifecycle panels.
# Poststates (a_nxt, h_nxt) are read directly from records —
# no budget-constraint re-derivation needed.
# ------------------------------------------------------------------

def _records_to_panels(history, nest, N):
    """Reshape kikku history records into (T, N) lifecycle panels."""
    stage0 = _base_stage(nest)
    cal = stage0.calibration
    sett = stage0.settings
    T = int(sett["T"])
    t0 = int(cal["t0"])
    R = float(cal["R"])
    R_H = float(cal["R_H"])
    delta = float(cal["delta"])

    a = np.full((T, N), np.nan)
    h = np.full((T, N), np.nan)
    c = np.full((T, N), np.nan)
    y = np.full((T, N), np.nan)
    z_idx = np.full((T, N), -1, dtype=np.int64)
    adj_choice = np.full((T, N), -1, dtype=np.int64)
    a_nxt = np.full((T, N), np.nan)
    h_nxt = np.full((T, N), np.nan)

    for t in range(t0, T):
        if t not in history:
            continue
        rec = history[t]

        ts = rec.get('tenure', {}).get('states', {})
        if 'a' in ts:
            a[t] = ts['a']
        if 'h' in ts:
            h[t] = ts['h']
        if 'z_idx' in ts:
            z_idx[t] = ts['z_idx'].astype(np.int64)

        labels = rec.get('tenure', {}).get('labels')
        if labels is not None:
            adj_choice[t] = np.where(labels == 'keep', 0, 1)

        keep = (adj_choice[t] == 0)
        adj = (adj_choice[t] == 1)

        # keeper: c from controls, a_nxt/h_nxt from poststates
        kr = rec.get('keeper_cons', {})
        kc = kr.get('controls', {})
        ks = kr.get('states', {})
        kp = kr.get('poststates', {})
        if 'c' in kc and np.any(keep):
            c[t, keep] = kc['c'][keep]
        if 'w_keep' in ks and np.any(keep):
            y[t, keep] = ks['w_keep'][keep] - R * a[t, keep]
        if 'a_nxt' in kp and np.any(keep):
            a_nxt[t, keep] = kp['a_nxt'][keep]
        if 'h_nxt' in kp and np.any(keep):
            h_nxt[t, keep] = kp['h_nxt'][keep]

        # adjuster: c, h_choice from controls, a_nxt/h_nxt from poststates
        ar = rec.get('adjuster_cons', {})
        ac = ar.get('controls', {})
        ast = ar.get('states', {})
        ap = ar.get('poststates', {})
        if 'c' in ac and np.any(adj):
            c[t, adj] = ac['c'][adj]
        if 'w_adj' in ast and np.any(adj):
            y[t, adj] = (ast['w_adj'][adj] - R * a[t, adj]
                         - R_H * (1 - delta) * h[t, adj])
        if 'a_nxt' in ap and np.any(adj):
            a_nxt[t, adj] = ap['a_nxt'][adj]
        if 'h_nxt' in ap and np.any(adj):
            h_nxt[t, adj] = ap['h_nxt'][adj]

    return {
        'a': a, 'h': h, 'c': c, 'y': y,
        'z_idx': z_idx, 'discrete': adj_choice,
        'a_nxt': a_nxt, 'h_nxt': h_nxt,
    }


# ------------------------------------------------------------------
# Utility stats
# ------------------------------------------------------------------

def _compute_utility_stats(sim_data, nest, callables):
    """NPV utility and per-branch period counts."""
    T, N = sim_data['a'].shape
    cal = _base_stage(nest).calibration
    t0 = int(cal["t0"])
    beta = float(cal["beta"])
    # chi is captured inside u_fn; if chi > 0, adjuster needs its own u_adj callable
    u_fn = callables['keeper_cons']['u']

    npv = np.zeros(N)
    npv_adj = np.zeros(N)
    npv_keep = np.zeros(N)
    n_adj = np.zeros(N, dtype=np.int64)
    n_keep = np.zeros(N, dtype=np.int64)

    for t in range(t0, T):
        discount = beta ** (t - t0)
        for i in range(N):
            ci = sim_data['c'][t, i]
            hi = sim_data['h_nxt'][t, i]
            di = sim_data['discrete'][t, i]
            if np.isnan(ci) or np.isnan(hi) or ci <= 0.1:
                continue
            util = u_fn(ci, hi)
            if util > -1e10:
                npv[i] += discount * util
                if di == 1:
                    npv_adj[i] += discount * util
                    n_adj[i] += 1
                else:
                    npv_keep[i] += discount * util
                    n_keep[i] += 1

    return {
        'npv_utility': npv, 'npv_utility_adj': npv_adj,
        'npv_utility_keep': npv_keep,
        'n_adj_periods': n_adj, 'n_keep_periods': n_keep,
    }


# ------------------------------------------------------------------
# I/O boundary: initial conditions
# ------------------------------------------------------------------

def make_initial_particles(N, grids, nest, seed=42,
                           use_empirical=False,
                           dispersion=0.0, gender='male'):
    """Construct initial particle dict (I/O boundary).

    Returns ``{'a', 'h', 'z_idx', '_idx'}`` clamped to grid bounds.
    """
    n_z = len(grids['z'])
    stage0 = _base_stage(nest)
    cal = stage0.calibration
    sett = stage0.settings
    b = float(sett["b"])
    gA = float(sett["a_max"])
    gH = float(sett["h_max"])
    t0 = int(cal["t0"])
    normalisation = float(sett["normalisation"])

    rng = np.random.default_rng(seed + 1)
    z_idx = rng.choice(n_z, size=N).astype(np.int64)

    if use_empirical:
        try:
            from examples.durables.init_conditions import (
                initialize_simulation,
            )
            init = initialize_simulation(
                N, t0, gender=gender,
                normalisation=normalisation,
                dispersion=dispersion, seed=seed)
            a = init['a_init']
            h = init['h_init']
        except Exception as e:
            print(f"Warning: empirical init failed: {e}")
            print("  Falling back to default (0.5, 0.5)")
            a = np.full(N, 0.5)
            h = np.full(N, 0.5)
    else:
        a = np.full(N, 0.5)
        h = np.full(N, 0.5)

    return {
        'a': np.clip(a, b, gA).astype(np.float64),
        'h': np.clip(h, b, gH).astype(np.float64),
        'z_idx': z_idx,
        '_idx': np.arange(N, dtype=np.int64),
    }


# ------------------------------------------------------------------
# Main entry point: thin combinator
#
# Composition:
#   initial_particles
#     -> build_period_pushforwards  [stage ops]
#     -> simulate(graph, twister, pushforward_by_t, ...)   [kikku]
#     -> _records_to_panels(history)                  [reshape]
#     -> _compute_utility_stats(panels)               [aggregate]
# ------------------------------------------------------------------

def simulate_lifecycle(nest, grids,
                       N=10000, seed=42,
                       use_empirical_init=False,
                       init_dispersion=0.0, init_gender='male'):
    """Forward-simulate the lifecycle (no Euler; use evaluate_euler_* post-hoc).

    Parameters
    ----------
    nest : dict
        Solved nest from ``solve()``.  Must contain ``'graph'``
        and ``'inter_conn'``. Each solution entry carries ``'callables'``.
    grids : dict
    N : int
    seed : int
    use_empirical_init, init_dispersion, init_gender :
        Forwarded to ``make_initial_particles``.

    Returns
    -------
    sim_data : dict
        ``(T, N)`` panels including ``c``, ``a_nxt``, ``h_nxt``,
        ``z_idx``, ``discrete``, utility aggregates, etc.
    """
    stage0 = _base_stage(nest)
    t0 = int(stage0.calibration["t0"])
    T_end = int(stage0.settings["T"]) - 1

    # topology from the solved nest
    graph = nest['graph']
    ic = nest['inter_conn']
    twister_rename = (
        {k: v for d in ic for k, v in d.items()}
        if isinstance(ic, list) else dict(ic or {}))

    pushforward_by_t = build_period_pushforwards(nest, grids)

    # exogenous shocks
    markov_draw = _make_markov_draw(grids)
    draws = draw_shocks(
        N, t0, T_end,
        {'markov': lambda n, rng: rng.random(n)},
        rng_seed=seed)

    # I/O boundary: initial conditions
    particles = make_initial_particles(
        N, grids, nest, seed=seed,
        use_empirical=use_empirical_init,
        dispersion=init_dispersion, gender=init_gender)

    # lifecycle walk (kikku generic)
    history, _ = simulate(
        graph=graph, twister=twister_rename,
        stage_forwards_by_t=pushforward_by_t,
        twister_fn=markov_draw,
        initial_particles=particles,
        t0=t0, T_end=T_end, draws=draws)

    # records -> panels -> utility stats
    sim_data = _records_to_panels(history, nest, N)
    sim_data.update(_compute_utility_stats(
        sim_data, nest, nest["solutions"][0]["callables"]))

    return sim_data
