"""Tenure stage: transitions + eval + max + chain rule.

dcsn_mover: interpolate keeper/adjuster at transition
            points, max over branches, chain rule.
arvl_mover: E_z conditioning.
"""

import numpy as np
from numba import njit
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
from dcsmm.fues.helpers.math_funcs import interp_as_scalar


@njit(cache=True)
def _tenure_clamp_scalar(val, min_val, max_val, nan_replacement):
    """Match kikku ``clamp_scalar`` (NaN/inf handling) for tenure kernel."""
    if np.isnan(val):
        return nan_replacement
    if np.isinf(val):
        return max_val if val > 0 else min_val
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val


@njit
def _tenure_dcsn_kernel(
    z_vals,
    a_grid,
    h_grid,
    h_keep,
    we_grid,
    V_cntn,
    Vkeeper,
    dVw_keeper,
    phi_keeper,
    Aadj,
    Hadj,
    dVw_adj,
    UGgrid_all,
    b,
    grid_max_A,
    grid_max_H,
    extrap_flag,
    xto_mode,
    age,
    g_keep_w,
    g_adj_w,
    u_fn,
    bellman_obj_fn,
    housing_cost_fn,
    marginal_a_fn,
    marginal_h_keep_fn,
    marginal_h_adj_fn,
):
    n_z = z_vals.shape[0]
    n_a = a_grid.shape[0]
    n_h = h_grid.shape[0]

    V_out = np.empty((n_z, n_a, n_h))
    adj_out = np.empty((n_z, n_a, n_h))
    dV_a_out = np.empty((n_z, n_a, n_h))
    dV_h_out = np.empty((n_z, n_a, n_h))

    for iz in range(n_z):
        z = z_vals[iz]
        for ia in range(n_a):
            a = a_grid[ia]
            w_k = g_keep_w(a, z, age)
            for ih in range(n_h):
                h = h_grid[ih]

                v_k = _tenure_clamp_scalar(
                    interp_as_scalar(a_grid, Vkeeper[iz, :, ih], w_k, extrap=extrap_flag),
                    -1e10,
                    1e10,
                    -1e10,
                )

                w_adj = g_adj_w(a, h, z, age)
                a_a = _tenure_clamp_scalar(
                    interp_as_scalar(we_grid, Aadj[iz], w_adj, extrap=extrap_flag),
                    b,
                    2.0 * grid_max_A,
                    b,
                )
                h_a = _tenure_clamp_scalar(
                    interp_as_scalar(we_grid, Hadj[iz], w_adj, extrap=extrap_flag),
                    b,
                    2.0 * grid_max_H,
                    b,
                )
                c_a = _tenure_clamp_scalar(
                    w_adj - a_a - housing_cost_fn(h_a),
                    1e-10,
                    1e10,
                    1e-10,
                )
                pts = np.empty(2, dtype=np.float64)
                pts[0] = min(max(a_a, b), 2.0 * grid_max_A)
                pts[1] = min(max(h_a, b), 2.0 * grid_max_H)
                Ev_raw = eval_linear(UGgrid_all, V_cntn[iz], pts, xto_mode)
                if np.isnan(Ev_raw) or np.isinf(Ev_raw):
                    Ev_raw = -1e10
                v_a = bellman_obj_fn(
                    u_fn(c_a, h_a),
                    Ev_raw,
                )

                adj = 1.0 if v_a >= v_k else 0.0
                V_out[iz, ia, ih] = adj * v_a + (1.0 - adj) * v_k
                adj_out[iz, ia, ih] = adj

                dvw_k = interp_as_scalar(a_grid, dVw_keeper[iz, :, ih], w_k, extrap=extrap_flag)
                dvw_a = interp_as_scalar(we_grid, dVw_adj[iz], w_adj, extrap=extrap_flag)
                pk = interp_as_scalar(a_grid, phi_keeper[iz, :, ih], w_k, extrap=extrap_flag)

                dV_a_out[iz, ia, ih] = marginal_a_fn(
                    adj * dvw_a + (1.0 - adj) * dvw_k
                )
                dV_h_out[iz, ia, ih] = (
                    adj * marginal_h_adj_fn(dvw_a)
                    + (1.0 - adj) * marginal_h_keep_fn(pk)
                )

    return V_out, adj_out, dV_a_out, dV_h_out


@njit(cache=True)
def _branch_policy_adj_vals(UG, adj_t, a, h, z_idx):
    N = a.shape[0]
    out = np.empty(N, dtype=np.float64)
    for i in range(N):
        pt = np.empty(2, dtype=np.float64)
        pt[0] = a[i]
        pt[1] = h[i]
        iz = int(z_idx[i])
        out[i] = eval_linear(UG, adj_t[iz], pt, xto.LINEAR)
    return out


@njit
def _dcsn_to_cntn_keep_kernel(a, h, z_idx, z_vals, age, keep_w, keep_h):
    N = a.shape[0]
    w = np.empty(N, dtype=np.float64)
    hk = np.empty(N, dtype=np.float64)
    for i in range(N):
        iz = int(z_idx[i])
        z = z_vals[iz]
        w[i] = keep_w(a[i], z, age)
        hk[i] = keep_h(h[i])
    return w, hk


@njit
def _dcsn_to_cntn_adj_kernel(a, h, z_idx, z_vals, age, adj_w):
    N = a.shape[0]
    w = np.empty(N, dtype=np.float64)
    for i in range(N):
        iz = int(z_idx[i])
        z = z_vals[iz]
        w[i] = adj_w(a[i], h[i], z, age)
    return w


def make_tenure_ops(callables, grids, stage,
                    condition_V, condition_V_HD):
    """Build tenure operators.

    Parameters
    ----------
    callables : dict
        Full per-period callables; uses ``tenure`` and its ``transitions``.
    grids : dict
    stage : dolo.compiler.model.SymbolicModel
        Dolo+ stage.
    condition_V, condition_V_HD : callable
        E_z conditioning.
    """
    tenure = callables["tenure"]
    income_transitions = tenure["transitions"]
    sett = stage.settings
    b = float(sett["b"])
    grid_max_A = float(sett["a_max"])
    grid_max_H = float(sett["h_max"])
    extrap = bool(int(sett.get("extrap_policy", 1)))
    _xto_mode = xto.LINEAR if extrap else xto.CONSTANT
    u_fn = tenure["u"]
    bellman_obj_fn = tenure["bellman_obj"]
    housing_cost_fn = tenure["housing_cost"]
    marginal_a_fn = tenure["marginalBellman_d_a"]
    marginal_h_keep_fn = tenure["marginalBellman_d_h_keep"]
    marginal_h_adj_fn = tenure["marginalBellman_d_h_adj"]
    g_keep_w = income_transitions["keep_w"]
    g_keep_h = income_transitions["keep_h"]
    g_adj_w = income_transitions["adj_w"]
    h_grid = grids["h"]
    h_keep = np.array([g_keep_h(hv) for hv in h_grid], dtype=np.float64)

    def dcsn_mover(vlu_cntn, grids, age,
                   Akeeper, Ckeeper, Vkeeper,
                   dVw_keeper, phi_keeper,
                   Aadj, Cadj, Hadj, Vadj, dVw_adj):
        """Tenure cntn_to_dcsn: transitions + eval + max.

        Receives raw keeper (on asset grid per h slice)
        and adjuster (on wealth grid per z) outputs.
        Computes branch transitions, interpolates at
        transition points, takes max, chain rule.
        """
        z_vals = np.asarray(grids["z"], dtype=np.float64)
        a_grid = np.asarray(grids["a"], dtype=np.float64)
        we_grid = np.asarray(grids["we"], dtype=np.float64)
        UGgrid_all = grids["UGgrid_all"]
        h_grid_loc = np.asarray(h_grid, dtype=np.float64)

        V = vlu_cntn["V"]

        V_out, adj_out, dV_a_out, dV_h_out = _tenure_dcsn_kernel(
            z_vals,
            a_grid,
            h_grid_loc,
            h_keep,
            we_grid,
            V,
            Vkeeper,
            dVw_keeper,
            phi_keeper,
            Aadj,
            Hadj,
            dVw_adj,
            UGgrid_all,
            b,
            grid_max_A,
            grid_max_H,
            extrap,
            _xto_mode,
            float(age),
            g_keep_w,
            g_adj_w,
            u_fn,
            bellman_obj_fn,
            housing_cost_fn,
            marginal_a_fn,
            marginal_h_keep_fn,
            marginal_h_adj_fn,
        )

        return (
            {"V": V_out, "d_aV": dV_a_out, "d_hV": dV_h_out},
            {"adj": adj_out},
        )

    def arvl_mover(vlu_dcsn):
        """E_z conditioning."""
        Ev, Edv_a, Edv_h = condition_V(
            vlu_dcsn["V"],
            vlu_dcsn["d_aV"],
            vlu_dcsn["d_hV"])
        return {"V": Ev, "d_aV": Edv_a, "d_hV": Edv_h}

    def arvl_mover_hd(dV_h_hd):
        return condition_V_HD(dV_h_hd)

    return dcsn_mover, arvl_mover, arvl_mover_hd


# ------------------------------------------------------------------
# Forward (simulation) operator
# ------------------------------------------------------------------

def make_tenure_forward(adj_t, callables, grids, age):
    """BranchingForward for the tenure stage.

    Composes: arvl_to_dcsn (identity) -> branch_policy (D interp)
    -> dcsn_to_cntn per branch (transition callables).

    Passes ``_idx`` through both branches so downstream leaf
    stages can map back to the full population.
    """
    from kikku.asva.simulate import BranchingForward

    UG = grids["UGgrid_all"]
    income_trans_t = callables["tenure"]["transitions"]
    g_keep_h = income_trans_t["keep_h"]
    g_keep_w = income_trans_t["keep_w"]
    g_adj_w = income_trans_t["adj_w"]
    z_vals = np.asarray(grids["z"], dtype=np.float64)
    age_f = float(age)

    def arvl_to_dcsn(particles, shocks):
        return particles

    def branch_policy(particles):
        a = particles["a"]
        h = particles["h"]
        z_idx = particles["z_idx"]
        N = len(a)
        adj_vals = _branch_policy_adj_vals(UG, adj_t, a, h, z_idx)
        labels = np.empty(N, dtype="<U7")
        for i in range(N):
            av = adj_vals[i]
            r = round(min(max(av, 0.0), 1.0))
            labels[i] = "adjust" if r == 1 else "keep"
        return labels

    def dcsn_to_cntn_keep(particles, shocks):
        a, h, z_idx = particles["a"], particles["h"], particles["z_idx"]
        w, hk = _dcsn_to_cntn_keep_kernel(
            a, h, z_idx, z_vals, age_f, g_keep_w, g_keep_h
        )
        out = {"w_keep": w, "h_keep": hk, "z_idx": z_idx.copy()}
        if "_idx" in particles:
            out["_idx"] = particles["_idx"].copy()
        return out

    def dcsn_to_cntn_adj(particles, shocks):
        a, h, z_idx = particles["a"], particles["h"], particles["z_idx"]
        w = _dcsn_to_cntn_adj_kernel(a, h, z_idx, z_vals, age_f, g_adj_w)
        out = {"w_adj": w, "z_idx": z_idx.copy()}
        if "_idx" in particles:
            out["_idx"] = particles["_idx"].copy()
        return out

    return BranchingForward(
        arvl_to_dcsn=arvl_to_dcsn,
        branch_policy=branch_policy,
        dcsn_to_cntn={"keep": dcsn_to_cntn_keep,
                      "adjust": dcsn_to_cntn_adj},
        shock_draw_arvl=None,
        shock_draw_cntn=None,
    )
