"""Tenure stage: transitions + eval + max + chain rule.

dcsn_mover: interpolate keeper/adjuster at transition
            points, max over branches, chain rule.
arvl_mover: E_z conditioning.
"""

import numpy as np
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
from dcsmm.fues.helpers.math_funcs import interp_as_scalar
from kikku.asva.numerics import clamp_scalar as _clamp


def make_tenure_ops(cp, callables, transitions, condition_V, condition_V_HD):
    """Build tenure operators.

    Parameters
    ----------
    cp : ConsumerProblem
    callables : dict
    transitions : dict
        Transition callables from make_transitions().
    condition_V, condition_V_HD : callable
        E_z conditioning.
    """
    R = cp.R
    R_H = cp.R_H
    delta = cp.delta
    beta = cp.beta
    b = cp.b
    tau = cp.tau
    chi = cp.chi
    u_fn = callables["u"]
    g_keep_w = transitions["tenure"]["dcsn_to_cntn"]["keep_w"]
    g_keep_h = transitions["tenure"]["dcsn_to_cntn"]["keep_h"]
    g_adj_w = transitions["tenure"]["dcsn_to_cntn"]["adj_w"]
    def dcsn_mover(vlu_cntn, grids,
                   Akeeper, Ckeeper, Vkeeper,
                   dVw_keeper, phi_keeper,
                   Aadj, Cadj, Hadj, Vadj, dVw_adj):
        """Tenure cntn_to_dcsn: transitions + eval + max.

        Receives raw keeper (on asset grid per h slice)
        and adjuster (on wealth grid per z) outputs.
        Computes branch transitions, interpolates at
        transition points, takes max, chain rule.
        """
        z_vals = grids["z"]
        a_grid = grids["a"]
        h_grid = grids["h"]
        we_grid = grids["we"]
        UGgrid_all = grids["UGgrid_all"]
        n_z = len(z_vals)
        n_a = len(a_grid)
        n_h = len(h_grid)
        h_keep = (1 - delta) * h_grid

        V = vlu_cntn['V']

        V_out = np.empty((n_z, n_a, n_h))
        D_out = np.empty((n_z, n_a, n_h))
        dV_a_out = np.empty((n_z, n_a, n_h))
        dV_h_out = np.empty((n_z, n_a, n_h))

        for iz in range(n_z):
            z = z_vals[iz]
            for ia in range(n_a):
                a = a_grid[ia]
                w_k = g_keep_w(a, z)

                for ih in range(n_h):
                    h = h_grid[ih]
                    hk = h_keep[ih]

                    # keep branch: value only
                    v_k = _clamp(interp_as_scalar(
                        a_grid, Vkeeper[iz, :, ih],
                        w_k), -1e10, 1e10, -1e10)

                    # adjust branch
                    w_adj = g_adj_w(a, h, z)
                    a_a = _clamp(interp_as_scalar(
                        we_grid, Aadj[iz],
                        w_adj), b, 1e10, b)
                    h_a = _clamp(interp_as_scalar(
                        we_grid, Hadj[iz],
                        w_adj), b, 1e10, b)
                    c_a = _clamp(
                        w_adj - a_a - h_a * (1 + tau),
                        1e-10, 1e10, 1e-10)
                    pts = np.array([a_a, h_a])
                    v_a = (u_fn(c_a, h_a, chi)
                           + beta * eval_linear(
                               UGgrid_all, V[iz],
                               pts, xto.LINEAR))

                    # max
                    d = 1 if v_a >= v_k else 0
                    V_out[iz, ia, ih] = (
                        d * v_a + (1 - d) * v_k)
                    D_out[iz, ia, ih] = d

                    # chain rule: marginals from leaves
                    dvw_k = interp_as_scalar(
                        a_grid,
                        dVw_keeper[iz, :, ih], w_k)
                    dvw_a = interp_as_scalar(
                        we_grid,
                        dVw_adj[iz], w_adj)
                    pk = interp_as_scalar(
                        a_grid,
                        phi_keeper[iz, :, ih], w_k)

                    dV_a_out[iz, ia, ih] = (
                        beta * R
                        * (d * dvw_a + (1 - d) * dvw_k))
                    dV_h_out[iz, ia, ih] = (
                        beta * R_H * (1 - delta)
                        * (d * dvw_a + (1 - d) * pk))

        return (
            {'V': V_out, 'd_aV': dV_a_out, 'd_hV': dV_h_out},
            {'d': D_out},
        )

    def arvl_mover(vlu_dcsn):
        """E_z conditioning."""
        Ev, Edv_a, Edv_h = condition_V(
            vlu_dcsn['V'],
            vlu_dcsn['d_aV'],
            vlu_dcsn['d_hV'])
        return {'V': Ev, 'd_aV': Edv_a, 'd_hV': Edv_h}

    def arvl_mover_hd(dV_h_hd):
        return condition_V_HD(dV_h_hd)

    return dcsn_mover, arvl_mover, arvl_mover_hd


# ------------------------------------------------------------------
# Forward (simulation) operator
# ------------------------------------------------------------------

def make_tenure_forward(D_t, trans_t, grids, UG):
    """BranchingForward for the tenure stage.

    Composes: arvl_to_dcsn (identity) -> branch_policy (D interp)
    -> dcsn_to_cntn per branch (transition callables).

    Passes ``_idx`` through both branches so downstream leaf
    stages can map back to the full population.
    """
    from kikku.asva.simulate import BranchingForward

    z_vals = grids['z']

    def arvl_to_dcsn(particles, shocks):
        return particles

    def branch_policy(particles):
        a = particles['a']
        h = particles['h']
        z_idx = particles['z_idx']
        N = len(a)
        labels = np.empty(N, dtype='<U6')
        for i in range(N):
            pt = np.array([a[i], h[i]])
            d = eval_linear(UG, D_t[int(z_idx[i])], pt, xto.LINEAR)
            labels[i] = "adjust" if round(min(max(d, 0), 1)) == 1 else "keep"
        return labels

    def dcsn_to_cntn_keep(particles, shocks):
        a, h, z_idx = particles['a'], particles['h'], particles['z_idx']
        N = len(a)
        w = np.empty(N)
        hk = np.empty(N)
        for i in range(N):
            z = z_vals[int(z_idx[i])]
            w[i] = trans_t['tenure']['dcsn_to_cntn']['keep_w'](a[i], z)
            hk[i] = trans_t['tenure']['dcsn_to_cntn']['keep_h'](h[i])
        out = {'w_keep': w, 'h_keep': hk, 'z_idx': z_idx.copy()}
        if '_idx' in particles:
            out['_idx'] = particles['_idx'].copy()
        return out

    def dcsn_to_cntn_adj(particles, shocks):
        a, h, z_idx = particles['a'], particles['h'], particles['z_idx']
        N = len(a)
        w = np.empty(N)
        for i in range(N):
            z = z_vals[int(z_idx[i])]
            w[i] = trans_t['tenure']['dcsn_to_cntn']['adj_w'](a[i], h[i], z)
        out = {'w_adj': w, 'z_idx': z_idx.copy()}
        if '_idx' in particles:
            out['_idx'] = particles['_idx'].copy()
        return out

    return BranchingForward(
        arvl_to_dcsn=arvl_to_dcsn,
        branch_policy=branch_policy,
        dcsn_to_cntn={"keep": dcsn_to_cntn_keep,
                      "adjust": dcsn_to_cntn_adj},
        shock_draw_arvl=None,
        shock_draw_cntn=None,
    )
