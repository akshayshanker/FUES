"""Backward induction for the durables model (DDSL).

Accretive build+solve: each step h creates a fresh
calibrated period, solves it, and accretes into the nest.

Solution convention (see solution_syntax.md):
  vlu_cntn — continuation value {V, dV: {a, h}}
  vlu_dcsn — decision-perch value
  vlu_arvl — arrival-perch value (E_z conditioned)
  pol      — policy {c, a_nxt, h_nxt, d}
"""

import numpy as np
import time
from pathlib import Path
from kikku.dynx import (
    load_syntax, instantiate_period,
    period_to_graph, backward_paths,
)
from dolo.compiler.calibration import (
    calibrate as calibrate_stage,
)

from .model import DurablesModel
from .operators import build_stage_ops


# ============================================================
# Helpers
# ============================================================

def _terminal_vlu_cntn(model):
    """Terminal condition: consume everything."""
    cp = model.cp
    shape = (len(cp.z_vals), len(cp.asset_grid_A),
             len(cp.asset_grid_H))
    V = np.empty(shape)
    dV_a = np.empty(shape)
    dV_h = np.empty(shape)

    for state in range(len(cp.X_all)):
        i_z, i_a, i_h = cp.X_all[state]
        a = cp.asset_grid_A[i_a]
        h = cp.asset_grid_H[i_h]
        w = cp.R_H * (1 - cp.delta) * h + cp.R * a
        V[i_z, i_a, i_h] = cp.term_u(w)
        dV_a[i_z, i_a, i_h] = (
            cp.beta * cp.R * cp.term_du(w))
        dV_h[i_z, i_a, i_h] = (
            cp.beta * cp.R_H * cp.term_du(w)
            * (1 - cp.delta))

    return {'V': V, 'dV': {'a': dV_a, 'h': dV_h}}


def _apply_twister(prev_sol, twister):
    """Twister: previous vlu_arvl -> next vlu_cntn."""
    return prev_sol['tenure']['vlu_arvl']


def recalibrate_period(period, calib_h):
    """Pure fn: return new period with updated calibration.

    Only re-runs calibrate (skips load/methodize/configure).
    """
    new_stages = {}
    for name, stage in period['stages'].items():
        new_stages[name] = calibrate_stage(stage, calib_h)
    return {
        'stages': new_stages,
        'connectors': period.get('connectors', []),
    }


def make_params_for_age(base_calibration, T):
    """Age-specific calibration schedule."""
    def params_for_age(h):
        calib = dict(base_calibration)
        calib['age'] = T - h
        return calib
    return params_for_age


# ============================================================
# Solve one period
# ============================================================

def solve_period(stage_ops, vlu_cntn, t, model,
                 dV_h_hd_cntn=None, m_bar=1.4,
                 verbose=False):
    """Solve one period in wave order.

    Wave 0: keeper_cons + adjuster_cons (B then I).
    Wave 1: tenure (B then I).

    Parameters
    ----------
    stage_ops : dict
    vlu_cntn : dict
        ``{'V', 'dV': {'a', 'h'}}``.
    t : int
    model : DurablesModel
    dV_h_hd_cntn : ndarray or None
        HD d_h V continuation.
    m_bar : float
    verbose : bool

    Returns
    -------
    dict
        Solution with ``vlu_dcsn``, ``vlu_arvl``, ``pol``
        per stage.
    """
    cp = model.cp
    use_hd = model.N_HD_LAMBDA > 1
    t0 = time.perf_counter()

    # --- Tenure branch transitions (arvl_to_dcsn) ---
    br_trans = stage_ops['tenure']['branch_transitions']()
    h_keep_grid = br_trans['h_keep_grid']

    # --- Wave 0: keeper_cons ---
    A_keep, C_keep, V_keep = stage_ops['keeper_cons'][
        'dcsn_mover'](vlu_cntn, h_keep_grid, t, m_bar)
    pol_keep, vlu_keep = stage_ops['keeper_cons'][
        'arvl_mover'](A_keep, C_keep, V_keep,
                      vlu_cntn, t)
    t_keeper = time.perf_counter() - t0

    # --- Wave 0: adjuster_cons (B then I) ---
    t1 = time.perf_counter()
    x_a, v_a, a_nxt_a, h_nxt_a = stage_ops[
        'adjuster_cons']['dcsn_mover'](vlu_cntn, t)
    pol_adj, vlu_adj = stage_ops[
        'adjuster_cons']['arvl_mover'](
            x_a, v_a, a_nxt_a, h_nxt_a,
            vlu_cntn, t, m_bar=m_bar)
    t_adj = time.perf_counter() - t1

    # --- Wave 1: tenure (B then I) ---
    t2 = time.perf_counter()

    branches = {
        'keep':   {'pol': pol_keep, 'vlu': vlu_keep},
        'adjust': {'pol': pol_adj,  'vlu': vlu_adj},
    }

    vlu_dcsn, pol_dcsn = stage_ops[
        'tenure']['dcsn_mover'](
            t, branches)
    t_discrete = time.perf_counter() - t2

    # arvl_mover: E_z conditioning
    vlu_arvl = stage_ops[
        'tenure']['arvl_mover'](vlu_dcsn)

    # HD path (deferred to Phase 3)
    dV_h_hd = np.zeros((1, 1, 1))
    dV_h_hd_arvl = None

    solve_time = time.perf_counter() - t0

    if verbose:
        print(f"  [h={model.T - t}] t={t},"
              f" keeper: {t_keeper*1000:.1f}ms,"
              f" adj: {t_adj*1000:.1f}ms,"
              f" discrete: {t_discrete*1000:.1f}ms")

    return {
        't': t,
        'keeper_cons': {
            'pol': pol_keep,
            'vlu': vlu_keep,
        },
        'adjuster_cons': {
            'pol': pol_adj,
            'vlu': vlu_adj,
        },
        'tenure': {
            'pol': pol_dcsn,
            'vlu_dcsn': vlu_dcsn,
            'vlu_arvl': vlu_arvl,
            'dV_h_hd': dV_h_hd,
            'dV_h_hd_arvl': dV_h_hd_arvl,
        },
        'solve_time': solve_time,
    }


# ============================================================
# Accretive build + solve
# ============================================================

def accrete_and_solve(
    H, period_inst, inter_conn, params_for_age,
    model, stage_ops, waves, verbose=False,
):
    """Accrete backward, solving each period.

    At each h: recalibrate -> accrete -> solve.
    """
    nest = {"periods": [], "twisters": [], "solutions": []}
    vlu_cntn = _terminal_vlu_cntn(model)
    dV_h_hd_cntn = None

    for h in range(H + 1):
        t = model.T - h

        # Recalibrate (pure fn, new period)
        calib_h = params_for_age(h)
        period_h = recalibrate_period(
            period_inst, calib_h)

        # Accrete
        nest["periods"].append(period_h)
        nest["twisters"].append(
            None if h == 0 else inter_conn)

        # Continuation from previous solution
        if h > 0:
            vlu_cntn = _apply_twister(
                nest["solutions"][h - 1],
                nest["twisters"][h])
            prev_ad = nest["solutions"][h - 1][
                'tenure']
            dV_h_hd_cntn = prev_ad.get('dV_h_hd_arvl')

        # Solve
        sol = solve_period(
            stage_ops, vlu_cntn, t, model,
            dV_h_hd_cntn=dV_h_hd_cntn,
            m_bar=model.m_bar, verbose=verbose)
        sol['h'] = h
        nest["solutions"].append(sol)

    return nest


# ============================================================
# Top-level entry point
# ============================================================

def solve(syntax_dir, method='FUES',
          calib_overrides=None, config_overrides=None,
          model=None, stage_ops=None, waves=None,
          verbose=False):
    """Full DDSL pipeline: load -> accrete+solve."""
    syntax_dir = Path(syntax_dir)

    calibration, settings, stage_sources, \
        period_template, inter_conn = load_syntax(
            syntax_dir, calib_overrides,
            config_overrides)

    period_inst = instantiate_period(
        calibration, settings,
        stage_sources, period_template)

    if waves is None:
        graph = period_to_graph(period_inst)
        waves = backward_paths(graph, inter_conn)

    if model is None:
        model = DurablesModel.from_period(
            period_inst, calibration, settings)

    if stage_ops is None:
        stage_ops = build_stage_ops(model)

    H = model.T - model.t0
    params_for_age = make_params_for_age(
        calibration, model.T)

    nest = accrete_and_solve(
        H, period_inst, inter_conn, params_for_age,
        model, stage_ops, waves, verbose=verbose)

    return nest, model, stage_ops, waves
