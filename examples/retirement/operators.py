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
from kikku.asva.egm_1d import make_egm_1d
from .model import WORKER_EGM_FNS, RETIREE_EGM_FNS


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

    egm_params = np.array([beta, R, delta])

    # ----------------------------------------------------------
    # retire_cons: retiree EGM (no upper envelope)
    # ----------------------------------------------------------

    _egm_retire = make_egm_1d(
        RETIREE_EGM_FNS['inv_euler'],
        RETIREE_EGM_FNS['bellman_rhs'],
        RETIREE_EGM_FNS['cntn_to_dcsn'],
        RETIREE_EGM_FNS['concavity'],
        egm_params,
    )

    @njit
    def _retire_interp(endog_grid, c_hat, v_hat, da_hat,
                       grid, v_cntn_0):
        """Interpolate retiree EGM results to arrival grid + constrained region."""
        min_a_val = endog_grid[0]
        c_arvl, v_arvl, da_arvl = interp_as_3(
            endog_grid, c_hat, v_hat, da_hat, grid)

        constrained_idx = np.where(grid <= min_a_val)
        c_arvl[constrained_idx] = grid[constrained_idx]
        v_arvl[constrained_idx] = u(
            grid[constrained_idx]) + beta * v_cntn_0
        da_arvl[constrained_idx] = 0

        dlambda_arvl = ddu(c_arvl) * (R - da_arvl)
        return c_arvl, v_arvl, da_arvl, dlambda_arvl

    def solver_retiree_stage(c_cntn, v_cntn, dlambda_cntn, grid):
        """Retiree consumption stage (EGM, no upper envelope).

        Note: the retiree's cntn_to_dcsn transition produces an
        endogenous grid scaled by 1/R: a_ret = (c + b_ret) / R.
        The EGM step handles this via fn_cntn_to_dcsn_ret.
        """
        c_hat, v_hat, endog_grid, da_hat = _egm_retire(
            du(c_cntn), dlambda_cntn, v_cntn, grid, 0.0)
        return _retire_interp(
            endog_grid, c_hat, v_hat, da_hat, grid, v_cntn[0])

    # ----------------------------------------------------------
    # work_cons: worker EGM + upper envelope
    # ----------------------------------------------------------

    _invert_euler = make_egm_1d(
        WORKER_EGM_FNS['inv_euler'],
        WORKER_EGM_FNS['bellman_rhs'],
        WORKER_EGM_FNS['cntn_to_dcsn'],
        WORKER_EGM_FNS['concavity'],
        egm_params,
    )

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

    def solver_worker_stage(dv_cntn, ddv_cntn, v_cntn, grid):
        """Worker consumption stage (EGM + upper envelope).

        ``grid`` is the asset grid (poststate / arrival).
        UE method is bound at operator construction time.
        EGM sub-equations are from model.WORKER_EGM_FNS.
        """
        c_hat, v_hat, x_dcsn_hat, del_a = _invert_euler(
            dv_cntn, ddv_cntn, v_cntn, grid, 0.0)

        min_a_val = x_dcsn_hat[0]

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

        egrid1 = refined["x_dcsn_ref"]
        v_ref = refined["v_dcsn_ref"]
        c_ref = refined["kappa_ref"]
        dela_ref = np.zeros_like(refined["x_cntn_ref"])

        v_arvl, c_arvl, da_arvl = _interp_to_arrival(
            egrid1, v_ref, c_ref, dela_ref, min_a_val,
            v_cntn, grid)

        return (v_arvl, c_arvl, da_arvl, ue_time,
                c_hat, v_hat, x_dcsn_hat, del_a)

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
