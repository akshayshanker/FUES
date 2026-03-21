"""Stage operator factories for the durables model.

Three stages:
  keeper_cons: EGM + FUES (from horses/keeper_egm.py)
  adjuster_cons: partial EGM + FUES (from Operator_Factory)
  tenure: transitions + eval + max + chain rule + E_z
"""

import numpy as np
from numba import njit
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
from dcsmm.fues.helpers.math_funcs import interp_as_scalar
from kikku.asva.numerics import clamp_scalar as _clamp
from examples.durables.durables import Operator_Factory
from .horses.keeper_egm import make_keeper_ops


def build_stage_ops(model):
    """Build all three stage operators."""
    cp = model.cp
    (_, _, condition_V,
     condition_V_HD, _, internals) = Operator_Factory(cp)

    _adjEGM = internals['_adjEGM']
    _refine_adj = internals['refine_adj']
    _adjuster_to_grid = internals['_adjuster_to_state_grid']
    return_grids = cp.return_grids

    # Keeper horse (self-contained, no Operator_Factory)
    _keeper_dcsn = make_keeper_ops(model)

    # Tenure closure variables
    R = cp.R
    R_H = cp.R_H
    delta = cp.delta
    beta = cp.beta
    b = cp.b
    y_func = cp.y_func
    du_c = cp.du_c
    du_h = cp.du_h
    z_vals = cp.z_vals
    a_grid = cp.asset_grid_A
    h_grid = cp.asset_grid_H
    UGgrid_all = cp.UGgrid_all
    n_z = len(z_vals)
    n_a = len(a_grid)
    n_h = len(h_grid)

    h_keep = (1 - delta) * h_grid

    # AC grid for keeper pre-eval
    bg = np.zeros(n_a)
    bg[:] = b
    asset_grid_AC = np.concatenate((bg, a_grid))
    n_ac = len(asset_grid_AC)

    @njit
    def _eval_2d_at_h(arr_2d, h_val, a_grid_loc):
        n = len(a_grid_loc)
        out = np.zeros(n)
        for i in range(n):
            pt = np.array([a_grid_loc[i], h_val])
            out[i] = eval_linear(
                UGgrid_all, arr_2d, pt, xto.LINEAR)
        return out

    # --- keeper_cons ---

    def keeper_dcsn_mover(vlu_cntn, t):
        """Keeper EGM + FUES.

        Pre-evaluates continuation at h_keep, then calls
        the keeper horse.
        """
        dV_a = vlu_cntn['dV']['a']
        V = vlu_cntn['V']

        dv_keep = np.empty((n_z, n_ac, n_h))
        v_keep = np.empty((n_z, n_ac, n_h))
        for iz in range(n_z):
            for ih in range(n_h):
                dv_keep[iz, :, ih] = _eval_2d_at_h(
                    dV_a[iz], h_keep[ih], asset_grid_AC)
                v_keep[iz, :, ih] = _eval_2d_at_h(
                    V[iz], h_keep[ih], asset_grid_AC)

        vlu_cntn_keep = {
            'dv': dv_keep, 'v': v_keep,
            'ac': asset_grid_AC, 'h_keep': h_keep,
        }
        return _keeper_dcsn(vlu_cntn_keep)

    # --- adjuster_cons ---

    def adjuster_dcsn_mover(vlu_cntn, t, m_bar=1.4):
        """Adjuster partial EGM + root-finding + FUES."""
        egrid, vf, a_nxt, h_nxt = _adjEGM(
            vlu_cntn['dV']['a'], vlu_cntn['dV']['h'],
            vlu_cntn['V'], t)
        (Aadj, Cadj, Hadj, Vadj,
         _, _, _, _, _, _, _, _) = _refine_adj(
            egrid, vf, a_nxt, h_nxt,
            m_bar=m_bar, return_grids=return_grids)
        return Aadj, Cadj, Hadj, Vadj

    # --- tenure ---

    def tenure_dcsn_mover(t, vlu_cntn,
                          Akeeper, Ckeeper, Vkeeper,
                          Aadj, Cadj, Hadj, Vadj):
        """Tenure cntn_to_dcsn: transitions + eval + max.

        Per YAML dcsn_to_cntn_transition:
          keep:   w_keep = R*a + y(z), h_keep = (1-delta)*h
          adjust: w_adj = R*a + R_H*(1-delta)*h + y(z)

        Evaluates keeper/adjuster at transition points,
        takes pointwise max, applies chain rule for
        marginal values.
        """
        V_out = np.empty((n_z, n_a, n_h))
        C_out = np.empty((n_z, n_a, n_h))
        A_out = np.empty((n_z, n_a, n_h))
        H_out = np.empty((n_z, n_a, n_h))
        D_out = np.empty((n_z, n_a, n_h))
        dV_a_out = np.empty((n_z, n_a, n_h))
        dV_h_out = np.empty((n_z, n_a, n_h))

        for iz in range(n_z):
            z = z_vals[iz]
            for ia in range(n_a):
                a = a_grid[ia]
                for ih in range(n_h):
                    h = h_grid[ih]

                    # --- keep branch ---
                    w_k = R * a + y_func(t, z)
                    hk = h_keep[ih]
                    v_k = _clamp(interp_as_scalar(
                        a_grid, Vkeeper[iz, :, ih], w_k),
                        -1e10, 1e10, -1e10)
                    c_k = _clamp(interp_as_scalar(
                        a_grid, Ckeeper[iz, :, ih], w_k),
                        1e-10, 1e10, 1e-10)
                    a_k = _clamp(interp_as_scalar(
                        a_grid, Akeeper[iz, :, ih], w_k),
                        b, 1e10, b)

                    # --- adjust branch ---
                    w_a = (R * a + R_H * (1 - delta) * h
                           + y_func(t, z))
                    a_a = _clamp(interp_as_scalar(
                        cp.asset_grid_WE, Aadj[iz, :], w_a),
                        b, 1e10, b)
                    h_a = _clamp(interp_as_scalar(
                        cp.asset_grid_WE, Hadj[iz, :], w_a),
                        b, 1e10, b)
                    c_a = _clamp(
                        w_a - a_a - h_a * (1 + cp.tau),
                        1e-10, 1e10, 1e-10)
                    pts = np.array([a_a, h_a])
                    v_a = (cp.u(c_a, h_a, cp.chi)
                           + beta * eval_linear(
                               UGgrid_all, vlu_cntn['V'][iz],
                               pts, xto.LINEAR))

                    # --- max ---
                    if v_a >= v_k:
                        d = 1
                    else:
                        d = 0

                    V_out[iz, ia, ih] = d * v_a + (1 - d) * v_k
                    C_out[iz, ia, ih] = d * c_a + (1 - d) * c_k
                    A_out[iz, ia, ih] = d * a_a + (1 - d) * a_k
                    H_out[iz, ia, ih] = d * h_a + (1 - d) * hk
                    D_out[iz, ia, ih] = d

                    # --- chain rule for marginals ---
                    dV_a_out[iz, ia, ih] = (
                        beta * R * (d * du_c(c_a)
                                    + (1 - d) * du_c(c_k)))

                    # phi_keep for keeper branch
                    pt_k = np.array([a_k, hk])
                    edvh = eval_linear(
                        UGgrid_all,
                        vlu_cntn['dV']['h'][iz],
                        pt_k, xto.LINEAR)
                    phi_k = du_h(hk) + edvh

                    dV_h_out[iz, ia, ih] = (
                        beta * R_H * (1 - delta)
                        * (d * du_c(c_a)
                           + (1 - d) * phi_k))

        vlu_dcsn = {
            'V': V_out,
            'dV': {'a': dV_a_out, 'h': dV_h_out},
        }
        pol = {
            'c': C_out, 'a_nxt': A_out,
            'h_nxt': H_out, 'd': D_out,
        }
        return vlu_dcsn, pol

    def tenure_arvl_mover(vlu_dcsn):
        """E_z conditioning: Pi @ arrays."""
        Ev, Edv_a, Edv_h = condition_V(
            vlu_dcsn['V'],
            vlu_dcsn['dV']['a'],
            vlu_dcsn['dV']['h'])
        return {'V': Ev, 'dV': {'a': Edv_a, 'h': Edv_h}}

    def tenure_arvl_mover_hd(dV_h_hd):
        return condition_V_HD(dV_h_hd)

    return {
        'keeper_cons': {
            'dcsn_mover': keeper_dcsn_mover,
        },
        'adjuster_cons': {
            'dcsn_mover': adjuster_dcsn_mover,
        },
        'tenure': {
            'dcsn_mover': tenure_dcsn_mover,
            'arvl_mover': tenure_arvl_mover,
            'arvl_mover_hd': tenure_arvl_mover_hd,
        },
    }
