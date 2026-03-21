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


def make_tenure_ops(model, condition_V, condition_V_HD):
    """Build tenure operators.

    Parameters
    ----------
    model : DurablesModel
    condition_V, condition_V_HD : callable
        E_z conditioning.
    """
    cp = model.cp
    R = cp.R
    R_H = cp.R_H
    delta = cp.delta
    beta = cp.beta
    b = cp.b
    y_func = cp.y_func
    z_vals = cp.z_vals
    a_grid = cp.asset_grid_A
    h_grid = cp.asset_grid_H
    UGgrid_all = cp.UGgrid_all
    n_z = len(z_vals)
    n_a = len(a_grid)
    n_h = len(h_grid)
    h_keep = (1 - delta) * h_grid

    def dcsn_mover(t, vlu_cntn,
                   Akeeper, Ckeeper, Vkeeper,
                   dVw_keeper, phi_keeper,
                   Aadj, Cadj, Hadj, Vadj, dVw_adj):
        """Tenure cntn_to_dcsn: transitions + eval + max.

        Receives raw keeper (on asset grid per h slice)
        and adjuster (on wealth grid per z) outputs.
        Computes branch transitions, interpolates at
        transition points, takes max, chain rule.
        """
        V = vlu_cntn['V']

        V_out = np.empty((n_z, n_a, n_h))
        D_out = np.empty((n_z, n_a, n_h))
        dV_a_out = np.empty((n_z, n_a, n_h))
        dV_h_out = np.empty((n_z, n_a, n_h))

        for iz in range(n_z):
            z = z_vals[iz]
            for ia in range(n_a):
                a = a_grid[ia]
                w_k = R * a + y_func(t, z)
                w_a = (R * a + R_H * (1 - delta)
                       * h_grid[0] + y_func(t, z))

                for ih in range(n_h):
                    h = h_grid[ih]
                    hk = h_keep[ih]

                    # keep branch: value only
                    v_k = _clamp(interp_as_scalar(
                        a_grid, Vkeeper[iz, :, ih],
                        w_k), -1e10, 1e10, -1e10)

                    # adjust branch
                    w_adj = (R * a
                             + R_H * (1 - delta) * h
                             + y_func(t, z))
                    a_a = _clamp(interp_as_scalar(
                        cp.asset_grid_WE, Aadj[iz],
                        w_adj), b, 1e10, b)
                    h_a = _clamp(interp_as_scalar(
                        cp.asset_grid_WE, Hadj[iz],
                        w_adj), b, 1e10, b)
                    c_a = _clamp(
                        w_adj - a_a - h_a * (1 + cp.tau),
                        1e-10, 1e10, 1e-10)
                    pts = np.array([a_a, h_a])
                    v_a = (cp.u(c_a, h_a, cp.chi)
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
                        cp.asset_grid_WE,
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
            {'V': V_out,
             'dV': {'a': dV_a_out, 'h': dV_h_out}},
            {'d': D_out},
        )

    def arvl_mover(vlu_dcsn):
        """E_z conditioning."""
        Ev, Edv_a, Edv_h = condition_V(
            vlu_dcsn['V'],
            vlu_dcsn['dV']['a'],
            vlu_dcsn['dV']['h'])
        return {'V': Ev, 'dV': {'a': Edv_a, 'h': Edv_h}}

    def arvl_mover_hd(dV_h_hd):
        return condition_V_HD(dV_h_hd)

    return dcsn_mover, arvl_mover, arvl_mover_hd
