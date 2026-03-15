"""Stage operator factories for the retirement choice model.

Each factory builds a closure over calibrated scalar parameters
and equation callables.  Grids are passed as arguments so the
same compiled operators work with any grid size — no JIT
recompilation needed when the grid changes.

Separated from ``model.py`` so that the numModel (params + grid)
and the operators (computation) live in different modules —
different abstraction per design-principles Rule 1.
"""

import numpy as np
import time
from numba import njit
from dcsmm.fues.helpers.math_funcs import interp_as_2, interp_as_3
from dcsmm.uenvelope import EGM_UE as egm_ue_global


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

    Scalar parameters (beta, R, delta, etc.) and equation
    callables are captured in closures.  The asset grid is
    passed as an argument to each operator, so the same
    compiled operators can be reused across grid sizes
    without JIT recompilation.

    Parameters
    ----------
    model : RetirementModel
        Model instance (scalars + callables).
    period : dict, optional
        Canonical period dict (UE method read from here).
    equations : dict, optional
        Override equation callables.

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

    # ----------------------------------------------------------
    # retire_cons: retiree EGM (no upper envelope)
    # ----------------------------------------------------------

    @njit
    def solver_retiree_stage(c_cntn, v_cntn, dlambda_cntn, grid):
        """Retiree consumption stage (EGM, no upper envelope).

        ``grid`` is the asset grid (arrival/poststate).
        """
        n = len(grid)
        c_dcsn_hat = np.zeros(n)
        v_dcsn_hat = np.zeros(n)
        endog_grid = np.zeros(n)
        da_dcsn_hat = np.zeros(n)

        for i in range(n):
            b_ret = grid[i]
            uc_cntn = beta * R * du(c_cntn[i])
            c_dcsn = uc_inv(uc_cntn)
            endog_grid[i] = (c_dcsn + b_ret) / R
            c_dcsn_hat[i] = c_dcsn
            v_dcsn_hat[i] = u(c_dcsn) + beta * v_cntn[i]
            da_dcsn_hat[i] = R * ddu(c_dcsn) / \
                (ddu(c_dcsn) + beta * R * dlambda_cntn[i])

        min_a_val = endog_grid[0]
        c_arvl, v_arvl, da_arvl = interp_as_3(
            endog_grid, c_dcsn_hat, v_dcsn_hat, da_dcsn_hat,
            grid)

        constrained_idx = np.where(grid <= min_a_val)
        c_arvl[constrained_idx] = grid[constrained_idx]
        v_arvl[constrained_idx] = u(
            grid[constrained_idx]) + beta * v_cntn[0]
        da_arvl[constrained_idx] = 0

        dlambda_arvl = ddu(c_arvl) * (R - da_arvl)
        return c_arvl, v_arvl, da_arvl, dlambda_arvl

    # ----------------------------------------------------------
    # work_cons: worker EGM + upper envelope
    # ----------------------------------------------------------

    @njit
    def _invert_euler(lambda_cntn, dlambda_cntn, v_cntn, grid):
        """Inverse Euler step for the worker stage (pre-UE)."""
        n = len(grid)
        cons_hat = np.zeros(n)
        q_hat = np.zeros(n)
        endog_grid = np.zeros(n)
        del_a = np.zeros(n)

        for i in range(n):
            c = uc_inv(beta * R * lambda_cntn[i])
            q_hat[i] = u(c) + beta * v_cntn[i] - delta
            cons_hat[i] = c
            endog_grid[i] = c + grid[i]
            del_a[i] = R * ddu(c) / \
                (ddu(c) + beta * R * dlambda_cntn[i])

        return cons_hat, q_hat, endog_grid, del_a

    @njit
    def _interp_to_arrival(egrid, vf, sigma, dela, min_a,
                           v_cntn, grid):
        """Interpolate refined worker policies to arrival grid."""
        w_grid = R * grid + y
        n = len(grid)
        vf_arvl, c_arvl = interp_as_2(egrid, vf, sigma, w_grid)
        da_arvl = np.zeros(n)

        constrained = np.where(w_grid < min_a)
        c_arvl[constrained] = w_grid[constrained] - grid[0]
        vf_arvl[constrained] = u(
            w_grid[constrained]) + beta * v_cntn[0] - delta

        return vf_arvl, c_arvl, da_arvl

    ue_method = _read_ue_method(period) if period else "FUES"

    def solver_worker_stage(lambda_cntn, dlambda_cntn,
                            v_cntn, grid):
        """Worker consumption stage (EGM + upper envelope).

        ``grid`` is the asset grid (poststate / arrival).
        UE method is bound at operator construction time.
        """
        cons_hat, q_hat, endog_grid, del_a = _invert_euler(
            lambda_cntn, dlambda_cntn, v_cntn, grid)

        min_a_val = endog_grid[0]

        t_ue = time.time()
        refined, _, _ = egm_ue_global(
            endog_grid, q_hat,
            beta * v_cntn - delta, cons_hat,
            grid, grid,
            du, {"func": u, "args": {}},
            ue_method=ue_method,
            m_bar=m_bar, lb=10,
            rfc_radius=0.75, rfc_n_iter=40,
        )
        ue_time = time.time() - t_ue

        egrid1 = refined["x_dcsn_ref"]
        q_ref = refined["v_dcsn_ref"]
        c_ref = refined["kappa_ref"]
        dela_ref = np.zeros_like(refined["x_cntn_ref"])

        v_arvl, c_arvl, da_arvl = _interp_to_arrival(
            egrid1, q_ref, c_ref, dela_ref, min_a_val,
            v_cntn, grid)

        return (v_arvl, c_arvl, da_arvl, ue_time,
                cons_hat, q_hat, endog_grid, del_a)

    # ----------------------------------------------------------
    # labour_mkt_decision: branching max/logit
    # ----------------------------------------------------------

    @njit
    def lab_mkt_choice_stage(
        v_work, v_ret, c_work, c_ret, da_work, da_ret,
    ):
        """Branching stage: discrete work/retire choice.

        No grid argument needed — operates on arrays from
        the two branch stages.
        """
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
