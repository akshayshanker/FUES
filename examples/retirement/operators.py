"""Stage operator factories for the retirement choice model.

Each factory builds a closure over calibrated parameters and
equation callables.  The closures are the actual stage operators
that the backward-solve loop calls.

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

    The UE method is read from the stage's methodization
    data (set during the pipeline), not passed as a
    separate argument.

    Parameters
    ----------
    model : RetirementModel
        Model instance with calibrated parameters and
        grids.
    period : dict, optional
        Canonical period dict.  If provided, the UE method
        is read from the ``work_cons`` stage's methods
        (a methodization concern).  Defaults to ``FUES``.
    equations : dict, optional
        Override equation callables.

    Returns
    -------
    dict
        ``{'retire_cons':  ...,
           'work_cons':    ...,
           'labour_mkt_decision': ...}``
    """
    beta, delta = model.beta, model.delta
    asset_grid_A = model.asset_grid_A
    y = model.y
    smooth_sigma = model.smooth_sigma
    grid_size = model.grid_size
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
    def solver_retiree_stage(c_cntn, v_cntn, dlambda_cntn):
        """Retiree consumption stage (EGM, no upper envelope).

        1. InvEuler on the poststate grid
        2. cntn_to_dcsn transition (endogenous grid)
        3. Bellman
        4. Asset derivative
        5. Interpolate to exogenous arrival grid
        6. Constrained region
        7. MarginalBellman
        """
        c_dcsn_hat = np.zeros(grid_size)
        v_dcsn_hat = np.zeros(grid_size)
        endog_grid = np.zeros(grid_size)
        da_dcsn_hat = np.zeros(grid_size)

        for i in range(len(asset_grid_A)):
            b_ret = asset_grid_A[i]
            c_next = c_cntn[i]
            uc_cntn = beta * R * du(c_next)
            c_dcsn = uc_inv(uc_cntn)
            a_ret_hat = (c_dcsn + b_ret) / R
            endog_grid[i] = a_ret_hat
            c_dcsn_hat[i] = c_dcsn
            v_dcsn_hat[i] = u(c_dcsn) + beta * v_cntn[i]
            da_dcsn_hat[i] = R * ddu(c_dcsn) / \
                (ddu(c_dcsn) + beta * R * dlambda_cntn[i])

        min_a_val = endog_grid[0]
        c_arvl, v_arvl, da_arvl = interp_as_3(
            endog_grid, c_dcsn_hat, v_dcsn_hat, da_dcsn_hat,
            asset_grid_A)

        constrained_idx = np.where(asset_grid_A <= min_a_val)
        c_arvl[constrained_idx] = asset_grid_A[constrained_idx]
        v_arvl[constrained_idx] = u(
            asset_grid_A[constrained_idx]) + beta * v_cntn[0]
        da_arvl[constrained_idx] = 0

        dlambda_arvl = ddu(c_arvl) * (R - da_arvl)
        return c_arvl, v_arvl, da_arvl, dlambda_arvl

    # ----------------------------------------------------------
    # work_cons: worker EGM + upper envelope
    # ----------------------------------------------------------

    @njit
    def _invert_euler(lambda_cntn, dlambda_cntn, v_cntn):
        """Inverse Euler step for the worker stage (pre-UE)."""
        cons_hat = np.zeros(grid_size)
        q_hat = np.zeros(grid_size)
        endog_grid = np.zeros(grid_size)
        del_a = np.zeros(grid_size)

        for i in range(grid_size):
            c = uc_inv(beta * R * lambda_cntn[i] )     #control function 
            q_hat[i] = u(c) + beta * v_cntn[i] - delta #RHS of bellman 
            cons_hat[i] = c
            endog_grid[i] = c + asset_grid_A[i]
            del_a[i] = R * ddu(c) / \
                (ddu(c) + beta * R * dlambda_cntn[i])

        return cons_hat, q_hat, endog_grid, del_a

    @njit
    def _interp_to_arrival(egrid, vf, sigma, dela, min_a, v_cntn):
        """Interpolate refined worker policies to arrival grid."""
        w_grid = R * asset_grid_A + y
        vf_arvl, c_arvl = interp_as_2(egrid, vf, sigma, w_grid)
        da_arvl = np.zeros(grid_size)

        constrained = np.where(w_grid < min_a)
        c_arvl[constrained] = w_grid[constrained] - asset_grid_A[0]
        vf_arvl[constrained] = u(
            w_grid[constrained]) + beta * v_cntn[0] - delta

        return vf_arvl, c_arvl, da_arvl

    ue_method = _read_ue_method(period) if period else "FUES"

    def solver_worker_stage(lambda_cntn, dlambda_cntn, v_cntn):
        """Worker consumption stage (EGM + upper envelope).

        UE method is bound at operator construction time.

        1. Invert Euler (pre-UE)
        2. Upper envelope (method bound at construction)
        3. Interpolate to arrival grid
        """
        cons_hat, q_hat, endog_grid, del_a = _invert_euler(
            lambda_cntn, dlambda_cntn, v_cntn)

        min_a_val = endog_grid[0]

        t_ue = time.time()
        refined, _, _ = egm_ue_global(
            endog_grid, q_hat,
            beta * v_cntn - delta, cons_hat,
            asset_grid_A, asset_grid_A,
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
            egrid1, q_ref, c_ref, dela_ref, min_a_val, v_cntn)

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

        Hard max (smooth_sigma=0) or logit smoothing.
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
