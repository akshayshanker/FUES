"""Forward simulation and Euler errors for durables2_0.

Orchestrator — a thin combinator that composes:

  1. **Stage forward operators** from ``horses/`` (same modules
     that house the backward ops) into per-period operator
     bundles.
  2. **kikku.asva.simulate.simulate** for the topology-driven
     lifecycle walk.
  3. **Records-to-panels** transform for the output arrays.

The period graph and inter-period connector come from ``nest``
(stored by ``solve()``), so the forward simulator is guaranteed
to use the same topology that was solved.

Euler errors are computed inline inside each leaf stage's
``dcsn_to_cntn`` and written to a pre-allocated panel via
agent indices (``_idx``).  See ``horses/keeper_egm.py`` and
``horses/adjuster_egm.py`` for the forward-operator construction.
"""

import numpy as np
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
from dcsmm.fues.helpers.math_funcs import interp_as_scalar

from kikku.asva.simulate import simulate, draw_shocks

from .callables import make_y_func
from .horses.branching import make_tenure_forward
from .horses.keeper_egm import make_keeper_forward
from .horses.adjuster_egm import make_adjuster_forward


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
    adj_next = edata_next['adj']
    C_keep_next = edata_next['C_keep']
    C_adj_next = edata_next['C_adj']

    rhs = 0.0
    for iz2 in range(n_z):
        prob = Pi[iz, iz2]
        z2 = z_vals[iz2]
        d2 = _eval_adj(a_nxt, h_nxt, iz2, adj_next, UG)
        if d2 == 1:
            w2 = income_trans_next['adj_w'](a_nxt, h_nxt, z2)
            c2 = max(interp_as_scalar(we_grid, C_adj_next[iz2], w2), 1e-10)
        else:
            w2 = income_trans_next['keep_w'](a_nxt, z2)
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
# Two-pass:
#   pass 1  collect Euler-lookahead data for every period
#   pass 2  compose StageForward / BranchingForward per period
#           binding edata[t+1] + euler_panel into each leaf stage
#
# The composition is:
#   period[t] = tenure(D_t, trans_t)
#               ∘ { keep -> keeper(C_t, edata[t+1])
#                   adj  -> adjuster(C_t, H_t, edata[t+1]) }
# ------------------------------------------------------------------

def build_period_pushforwards(nest, cp, grids, callables, settings, euler_panel):
    """Per-period forward operator bundles with inline Euler.

    Returns
    -------
    pushforward_by_t : dict[int, dict]
    stage_solution_by_t : dict[int, dict]
    """
    sol_by_t = {sol['t']: sol for sol in nest['solutions']}
    UG = grids['UGgrid_all']
    we_grid = grids['we']

    keep_h_fn = callables["tenure"]["transitions"]["keep_h"]

    # --- pass 1: Euler lookahead data ---
    stage_solution_by_t = {}
    for t, sol in sol_by_t.items():
        y_func_t = make_y_func(cp, age=t)
        income_trans = callables["tenure"]["make_income_transitions"](y_func_t)
        stage_solution_by_t[t] = {
            'adj': sol['tenure']['dcsn']['adj'],
            'C_keep': sol['keeper_cons']['dcsn']['c'],
            'C_adj': sol['adjuster_cons']['dcsn']['c'],
            'H_adj': sol['adjuster_cons']['dcsn']['h_choice'],
            'd_hV_arvl': sol['tenure']['arvl']['d_hV'],
            'income_trans': income_trans,
        }

    # --- pass 2: compose forward operators ---
    pushforward_by_t = {}
    for t in stage_solution_by_t:
        ed = stage_solution_by_t[t]
        edata_next = stage_solution_by_t.get(t + 1)

        pushforward_by_t[t] = {
            'tenure': make_tenure_forward(
                ed['adj'], ed['income_trans'], keep_h_fn, grids, UG),
            'keeper_cons': make_keeper_forward(
                ed['C_keep'], UG,
                edata_next, grids, callables, settings,
                euler_panel, t),
            'adjuster_cons': make_adjuster_forward(
                ed['C_adj'], ed['H_adj'], we_grid,
                edata_next, grids, callables, settings,
                euler_panel, t),
        }

    return pushforward_by_t, stage_solution_by_t


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

def _records_to_panels(history, cp, N):
    """Reshape kikku history records into (T, N) lifecycle panels."""
    T = cp.T
    t0 = cp.t0

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
            y[t, keep] = ks['w_keep'][keep] - cp.R * a[t, keep]
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
            y[t, adj] = (ast['w_adj'][adj] - cp.R * a[t, adj]
                         - cp.R_H * (1 - cp.delta) * h[t, adj])
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

def _compute_utility_stats(sim_data, cp, callables):
    """NPV utility and per-branch period counts."""
    T, N = sim_data['a'].shape
    t0 = cp.t0
    # chi is captured inside u_fn; if chi > 0, adjuster needs its own u_adj callable
    u_fn = callables['keeper_cons']['u']

    npv = np.zeros(N)
    npv_adj = np.zeros(N)
    npv_keep = np.zeros(N)
    n_adj = np.zeros(N, dtype=np.int64)
    n_keep = np.zeros(N, dtype=np.int64)

    for t in range(t0, T):
        discount = cp.beta ** (t - t0)
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

def make_initial_particles(N, cp, grids, seed=42,
                           use_empirical=False,
                           dispersion=0.0, gender='male'):
    """Construct initial particle dict (I/O boundary).

    Returns ``{'a', 'h', 'z_idx', '_idx'}`` clamped to grid bounds.
    """
    n_z = len(grids['z'])
    b, gA, gH = cp.b, cp.grid_max_A, cp.grid_max_H

    rng = np.random.default_rng(seed + 1)
    z_idx = rng.choice(n_z, size=N).astype(np.int64)

    if use_empirical:
        try:
            from examples.durables.init_conditions import (
                initialize_simulation,
            )
            normalisation = getattr(cp, 'normalisation', 1.0)
            init = initialize_simulation(
                N, cp.t0, gender=gender,
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
#     -> build_period_pushforwards(nest, ..., euler_panel)  [stage ops]
#     -> simulate(graph, twister, pushforward_by_t, ...)   [kikku]
#     -> _records_to_panels(history)                  [reshape]
#     -> _compute_utility_stats(panels)               [aggregate]
# ------------------------------------------------------------------

def simulate_and_euler(nest, cp, grids, callables, settings,
                       N=10000, seed=42,
                       use_empirical_init=False,
                       init_dispersion=0.0, init_gender='male'):
    """Forward-simulate and compute Euler errors.

    Parameters
    ----------
    nest : dict
        Solved nest from ``solve()``.  Must contain ``'graph'``
        and ``'inter_conn'``.
    cp, grids, callables, settings : from ``solve()``
    N : int
    seed : int
    use_empirical_init, init_dispersion, init_gender :
        Forwarded to ``make_initial_particles``.

    Returns
    -------
    euler : ndarray(T, N)
    sim_data : dict
    """
    T, t0, T_end = cp.T, cp.t0, cp.T - 1

    # topology from the solved nest
    graph = nest['graph']
    ic = nest['inter_conn']
    twister_rename = (
        {k: v for d in ic for k, v in d.items()}
        if isinstance(ic, list) else dict(ic or {}))

    # Euler side-channel
    euler_panel = np.full((T, N), np.nan)

    # compose per-period forward operators (horses/)
    pushforward_by_t, _ = build_period_pushforwards(
        nest, cp, grids, callables, settings, euler_panel)

    # exogenous shocks
    markov_draw = _make_markov_draw(grids)
    draws = draw_shocks(
        N, t0, T_end,
        {'markov': lambda n, rng: rng.random(n)},
        rng_seed=seed)

    # I/O boundary: initial conditions
    particles = make_initial_particles(
        N, cp, grids, seed=seed,
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
    sim_data = _records_to_panels(history, cp, N)
    sim_data.update(_compute_utility_stats(sim_data, cp, callables))

    return euler_panel, sim_data
