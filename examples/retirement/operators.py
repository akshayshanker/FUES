"""Stage operator factories for the retirement choice model.

Each stage has its own factory returning dcsn_mover (B) and
arvl_mover (I).  The composed stage operator T = I ∘ B is
also returned for use by the solver.

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
from .model import g_arvl_to_dcsn_worker, g_arvl_to_dcsn_retiree


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


# ==============================================================
# retire_cons
# ==============================================================

def make_retire_cons(model, callables):
    """Factory for retire_cons stage operators.

    Parameters
    ----------
    model : RetirementModel
    callables : dict
        EGM recipe: ``{'inv_euler', 'bellman_rhs',
        'cntn_to_dcsn', 'concavity'}``.

    Returns
    -------
    dict with 'dcsn_mover', 'arvl_mover', 'stage_op'
    """
    beta = model.beta
    R = model.R
    u, du, ddu = model.u, model.du, model.ddu

    egm_params = np.array([beta, R, model.delta, model.y])

    _egm_retire = make_egm_1d(
        callables['inv_euler'],
        callables['bellman_rhs'],
        callables['cntn_to_dcsn'],
        callables['concavity'],
        egm_params,
    )

    _compose = make_compose_interp(
        g_arvl_to_dcsn_retiree, interp_as_3)

    def dcsn_mover(c_cntn, v_cntn, dlambda_cntn, grid):
        """B: continuation → decision."""
        c_dcsn, v_dcsn, x_dcsn, dela_dcsn = _egm_retire(
            du(c_cntn), dlambda_cntn, v_cntn, grid, 0.0)
        return x_dcsn, v_dcsn, c_dcsn, dela_dcsn

    def arvl_mover(x_dcsn, v_dcsn, c_dcsn, dela_dcsn,
                   grid, v_cntn_0):
        """I: decision → arrival."""
        c_arvl, v_arvl, da_arvl = _compose(
            x_dcsn, c_dcsn, v_dcsn, dela_dcsn, grid, egm_params)

        min_a_val = x_dcsn[0] / R
        constrained_idx = np.where(grid <= min_a_val)
        c_arvl[constrained_idx] = grid[constrained_idx]
        v_arvl[constrained_idx] = u(
            grid[constrained_idx]) + beta * v_cntn_0
        da_arvl[constrained_idx] = 0

        dlambda_arvl = ddu(c_arvl) * (R - da_arvl)
        return c_arvl, v_arvl, da_arvl, dlambda_arvl

    def stage_op(c_cntn, v_cntn, dlambda_cntn, grid):
        """T = I ∘ B."""
        x_dcsn, v_dcsn, c_dcsn, dela_dcsn = dcsn_mover(
            c_cntn, v_cntn, dlambda_cntn, grid)
        return arvl_mover(
            x_dcsn, v_dcsn, c_dcsn, dela_dcsn, grid, v_cntn[0])

    return {'dcsn_mover': dcsn_mover,
            'arvl_mover': arvl_mover,
            'stage_op': stage_op}


# ==============================================================
# work_cons
# ==============================================================

def make_work_cons(model, callables, period=None):
    """Factory for work_cons stage operators.

    Parameters
    ----------
    model : RetirementModel
    callables : dict
        EGM recipe: ``{'inv_euler', 'bellman_rhs',
        'cntn_to_dcsn', 'concavity'}``.
    period : dict, optional
        Canonical period dict (UE method read from here).

    Returns
    -------
    dict with 'dcsn_mover', 'arvl_mover', 'stage_op'
    """
    beta, delta = model.beta, model.delta
    y = model.y
    R = model.R
    m_bar = model.m_bar
    u, du, ddu = model.u, model.du, model.ddu

    egm_params = np.array([beta, R, delta, y])

    _invert_euler = make_egm_1d(
        callables['inv_euler'],
        callables['bellman_rhs'],
        callables['cntn_to_dcsn'],
        callables['concavity'],
        egm_params,
    )

    ue_method = _read_ue_method(period) if period else "FUES"

    _compose = make_compose_interp(
        g_arvl_to_dcsn_worker, interp_as_2)

    def dcsn_mover(dv_cntn, ddv_cntn, v_cntn, grid):
        """B: continuation → decision (EGM + UE)."""
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

    def arvl_mover(x_dcsn, v_dcsn, c_dcsn, v_cntn, grid):
        """I: decision → arrival."""
        v_arvl, c_arvl = _compose(
            x_dcsn, v_dcsn, c_dcsn, grid, egm_params)

        w_grid = R * grid + y
        min_a = x_dcsn[0]
        constrained = np.where(w_grid < min_a)
        c_arvl[constrained] = w_grid[constrained] - grid[0]
        v_arvl[constrained] = u(
            w_grid[constrained]) + beta * v_cntn[0] - delta

        da_arvl = np.zeros(len(grid))
        return v_arvl, c_arvl, da_arvl

    def stage_op(dv_cntn, ddv_cntn, v_cntn, grid):
        """T = I ∘ B."""
        (x_dcsn, v_dcsn, c_dcsn, dela_dcsn, ue_time,
         c_hat, v_hat, x_dcsn_hat, del_a) = dcsn_mover(
            dv_cntn, ddv_cntn, v_cntn, grid)

        v_arvl, c_arvl, da_arvl = arvl_mover(
            x_dcsn, v_dcsn, c_dcsn, v_cntn, grid)

        return (v_arvl, c_arvl, da_arvl, ue_time,
                c_hat, v_hat, x_dcsn_hat, del_a)

    return {'dcsn_mover': dcsn_mover,
            'arvl_mover': arvl_mover,
            'stage_op': stage_op}


# ==============================================================
# labour_mkt_decision
# ==============================================================

def make_labour_mkt_decision(model):
    """Factory for labour_mkt_decision stage operators.

    Returns
    -------
    dict with 'stage_op'
    """
    R = model.R
    smooth_sigma = model.smooth_sigma
    du, ddu = model.du, model.ddu

    @njit
    def stage_op(v_work, v_ret, c_work, c_ret, da_work, da_ret):
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

    return {'stage_op': stage_op}
