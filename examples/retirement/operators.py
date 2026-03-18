"""Stage operator factories for the retirement choice model.

Each stage is decomposed into two movers following bellman calculus:
  - dcsn_mover (B): continuation → decision
  - arvl_mover (I): decision → arrival

The composed stage operator calls both movers internally,
so solve.py does not need to know about the split.

Grids are passed as arguments so the same compiled operators
work with any grid size — no JIT recompilation needed.
"""

import numpy as np
import time
from numba import njit
from dcsmm.fues.helpers.math_funcs import interp_as_2, interp_as_3
from dcsmm.uenvelope import EGM_UE as egm_ue_global
from kikku.asva.egm_1d import make_egm_1d
from kikku.asva.compose_interp import make_compose_interp
from .model import (
    WORKER_EGM_FNS, RETIREE_EGM_FNS,
    g_arvl_to_dcsn_worker, g_arvl_to_dcsn_retiree,
)


def _read_ue_method(period):
    """Extract the UE method tag from the work_cons stage's methods."""
    stages = period["stages"] if "stages" in period else period
    work = stages.get("work_cons")
    if work and hasattr(work, "methods"):
        mover = work.methods.get("cntn_to_dcsn_mover", {})
        for scheme in mover.get("schemes", []):
            if scheme.get("scheme") == "upper_envelope":
                tag = scheme.get("method", {})
                if isinstance(tag, dict):
                    return tag.get("__yaml_tag__", "FUES")
                return str(tag)
    return "FUES"


def make_stage_operators(model, period=None, equations=None):
    """Build stage operators for the retirement model.

    Returns
    -------
    dict
        ``{'retire_cons': ..., 'work_cons': ...,
           'labour_mkt_decision': ...}``
    """
    beta, delta = model.beta, model.delta
    y = model.y
    smooth_sigma = model.smooth_sigma
    R = model.R
    m_bar = model.m_bar

    if equations is None:
        equations = {}
    u = equations.get('u', model.u)
    du = equations.get('du', model.du)
    uc_inv = equations.get('uc_inv', model.uc_inv)
    ddu = equations.get('ddu', model.ddu)

    egm_params = np.array([beta, R, delta, y])

    # ==============================================================
    # retire_cons: retiree EGM (no upper envelope)
    # ==============================================================

    _egm_retire = make_egm_1d(
        RETIREE_EGM_FNS['inv_euler'],
        RETIREE_EGM_FNS['bellman_rhs'],
        RETIREE_EGM_FNS['cntn_to_dcsn'],
        RETIREE_EGM_FNS['concavity'],
        egm_params,
    )

    _compose_ret = make_compose_interp(
        g_arvl_to_dcsn_retiree, interp_as_3)

    def dcsn_mover_ret(c_cntn, v_cntn, dlambda_cntn, grid):
        """B: continuation → decision for retire_cons."""
        c_dcsn, v_dcsn, x_dcsn, dela_dcsn = _egm_retire(
            du(c_cntn), dlambda_cntn, v_cntn, grid, 0.0)
        return x_dcsn, v_dcsn, c_dcsn, dela_dcsn

    @njit
    def _ret_constrained(x_dcsn, v_arvl, c_arvl, da_arvl,
                         grid, v_cntn_0, params):
        """Patch constrained region for retiree."""
        beta_p = params[0]
        R_p = params[1]
        min_a_val = x_dcsn[0] / R_p
        constrained_idx = np.where(grid <= min_a_val)
        c_arvl[constrained_idx] = grid[constrained_idx]
        v_arvl[constrained_idx] = u(
            grid[constrained_idx]) + beta_p * v_cntn_0
        da_arvl[constrained_idx] = 0
        dlambda_arvl = ddu(c_arvl) * (R_p - da_arvl)
        return c_arvl, v_arvl, da_arvl, dlambda_arvl

    def solver_retiree_stage(c_cntn, v_cntn, dlambda_cntn, grid):
        """Composed stage operator for retire_cons."""
        x_dcsn, v_dcsn, c_dcsn, dela_dcsn = dcsn_mover_ret(
            c_cntn, v_cntn, dlambda_cntn, grid)
        c_arvl, v_arvl, da_arvl = _compose_ret(
            x_dcsn, c_dcsn, v_dcsn, dela_dcsn, grid, egm_params)
        c_arvl, v_arvl, da_arvl, dlambda_arvl = _ret_constrained(
            x_dcsn, v_arvl, c_arvl, da_arvl,
            grid, v_cntn[0], egm_params)
        return c_arvl, v_arvl, da_arvl, dlambda_arvl

    # ==============================================================
    # work_cons: worker EGM + upper envelope
    # ==============================================================

    _invert_euler = make_egm_1d(
        WORKER_EGM_FNS['inv_euler'],
        WORKER_EGM_FNS['bellman_rhs'],
        WORKER_EGM_FNS['cntn_to_dcsn'],
        WORKER_EGM_FNS['concavity'],
        egm_params,
    )

    ue_method = _read_ue_method(period) if period else "FUES"

    _compose_work = make_compose_interp(
        g_arvl_to_dcsn_worker, interp_as_2)

    def dcsn_mover_work(dv_cntn, ddv_cntn, v_cntn, grid):
        """B: continuation → decision for work_cons (EGM + UE)."""
        c_hat, v_hat, x_dcsn_hat, del_a = _invert_euler(
            dv_cntn, ddv_cntn, v_cntn, grid, 0.0)

        t_ue = time.time()
        refined, _, _ = egm_ue_global(
            x_dcsn_hat, v_hat,
            beta * v_cntn - delta, c_hat,
            grid, grid,
            du, {"func": u, "args": {}},
            ue_method=ue_method,
            m_bar=m_bar, lb=10,
            rfc_radius=0.75, rfc_n_iter=40,
        )
        ue_time = time.time() - t_ue

        x_dcsn = refined["x_dcsn_ref"]
        v_dcsn = refined["v_dcsn_ref"]
        c_dcsn = refined["kappa_ref"]
        dela_dcsn = np.zeros_like(refined["x_cntn_ref"])

        return (x_dcsn, v_dcsn, c_dcsn, dela_dcsn, ue_time,
                c_hat, v_hat, x_dcsn_hat, del_a)

    @njit
    def _work_constrained(v_arvl, c_arvl, x_dcsn, v_cntn,
                          grid, params):
        """Patch constrained region for worker."""
        R_p, y_p = params[1], params[3]
        beta_p, delta_p = params[0], params[2]
        w_grid = R_p * grid + y_p
        min_a = x_dcsn[0]
        constrained = np.where(w_grid < min_a)
        c_arvl[constrained] = w_grid[constrained] - grid[0]
        v_arvl[constrained] = u(
            w_grid[constrained]) + beta_p * v_cntn[0] - delta_p
        return v_arvl, c_arvl

    def solver_worker_stage(dv_cntn, ddv_cntn, v_cntn, grid):
        """Composed stage operator for work_cons."""
        (x_dcsn, v_dcsn, c_dcsn, dela_dcsn, ue_time,
         c_hat, v_hat, x_dcsn_hat, del_a) = dcsn_mover_work(
            dv_cntn, ddv_cntn, v_cntn, grid)

        v_arvl, c_arvl = _compose_work(
            x_dcsn, v_dcsn, c_dcsn, grid, egm_params)

        v_arvl, c_arvl = _work_constrained(
            v_arvl, c_arvl, x_dcsn, v_cntn, grid, egm_params)

        da_arvl = np.zeros(len(grid))

        return (v_arvl, c_arvl, da_arvl, ue_time,
                c_hat, v_hat, x_dcsn_hat, del_a)

    # ==============================================================
    # labour_mkt_decision: branching max/logit
    # ==============================================================

    @njit
    def lab_mkt_choice_stage(
        v_work, v_ret, c_work, c_ret, da_work, da_ret,
    ):
        """Branching stage: discrete work/retire choice."""
        if smooth_sigma == 0:
            p = v_work > v_ret
        else:
            e_w = np.exp(v_work / smooth_sigma)
            e_r = np.exp(v_ret / smooth_sigma)
            p = e_w / (e_r + e_w)
            p = np.where(np.isnan(p) | np.isinf(p), 0, p)

        c = p * c_work + (1 - p) * c_ret
        c[np.where(c < 0.0001)] = 0.0001
        v = p * v_work + (1 - p) * v_ret
        da = p * da_work + (1 - p) * da_ret

        dv = du(c)
        ddv = ddu(c) * (R - da)
        return v, c, dv, ddv

    return {
        'retire_cons': solver_retiree_stage,
        'work_cons': solver_worker_stage,
        'labour_mkt_decision': lab_mkt_choice_stage,
    }
