"""Stage operator factories for the durables model.

Three stages matching the YAML period template:

- ``keeper_cons``: keeper EGM + FUES + interp to state grid
- ``adjuster_cons``: partial EGM + FUES + Bellman eval on state grid
- ``tenure``: branching max + E_z

Each factory returns ``{'dcsn_mover', 'arvl_mover'}``.

Phase 2: leaf arvl_movers interpolate onto the common
(z, a, h) grid. The branching stage receives only
branch-keyed results — no continuation values.
"""

import numpy as np
from numba import njit
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
from examples.durables.durables import Operator_Factory
from .horses.keeper_egm import make_keeper_ops


def build_stage_ops(model):
    """Build all three stage operators.

    Keeper: from horses/keeper_egm.py (make_egm_1d + FUES).
    Adjuster + tenure: from Operator_Factory internals.

    Parameters
    ----------
    model : DurablesModel

    Returns
    -------
    dict
        Stage operators keyed by YAML stage name.
    """
    cp = model.cp
    (_, _, condition_V,
     condition_V_HD, _, internals) = Operator_Factory(cp)

    _adjEGM = internals['_adjEGM']
    _refine_adj = internals['refine_adj']
    _adjuster_to_grid = internals['_adjuster_to_state_grid']
    _branching_max = internals['_branching_max']

    return_grids = cp.return_grids
    delta = cp.delta

    # --- keeper_cons (from horse) ---

    _keeper_dcsn = make_keeper_ops(model)

    # keeper_dcsn_mover is _keeper_dcsn directly.
    # Caller (solve_period) passes h_keep_grid from
    # the tenure branch transition.

    from dcsmm.fues.helpers.math_funcs import interp_as_scalar
    from kikku.asva.numerics import clamp_scalar as _clamp

    du_h = cp.du_h
    UGgrid_all_op = cp.UGgrid_all
    a_grid = cp.asset_grid_A
    h_grid = cp.asset_grid_H
    b_val = cp.b
    gmax_A = cp.grid_max_A

    def keeper_arvl_mover(Akeeper, Ckeeper, Vkeeper,
                          w_keep, h_keep, vlu_cntn, t):
        """I: interp keeper onto (z,a,h) via w_keep + phi.

        w_keep[iz, ia] = R*a + y(t,z) from tenure.
        h_keep[ih] = (1-delta)*h from tenure.
        """
        n_z = len(cp.z_vals)
        n_a = len(a_grid)
        n_h = len(h_grid)

        v_out = np.empty((n_z, n_a, n_h))
        c_out = np.empty((n_z, n_a, n_h))
        a_out = np.empty((n_z, n_a, n_h))
        h_out = np.empty((n_z, n_a, n_h))
        phi_out = np.empty((n_z, n_a, n_h))

        dV_h = vlu_cntn['dV']['h']

        for iz in range(n_z):
            for ih in range(n_h):
                for ia in range(n_a):
                    w = w_keep[iz, ia]
                    hk = h_keep[ih]

                    v_out[iz, ia, ih] = _clamp(
                        interp_as_scalar(a_grid, Vkeeper[iz, :, ih], w),
                        -1e10, 1e10, -1e10)
                    c_val = _clamp(
                        interp_as_scalar(a_grid, Ckeeper[iz, :, ih], w),
                        1e-10, 1e10, 1e-10)
                    c_out[iz, ia, ih] = c_val
                    a_val = _clamp(
                        interp_as_scalar(a_grid, Akeeper[iz, :, ih], w),
                        b_val, gmax_A * 2, b_val)
                    a_out[iz, ia, ih] = a_val
                    h_out[iz, ia, ih] = hk

                    # phi = du_h(h_keep) + E_z[d_h V](a', h_keep)
                    pt = np.array([a_val, hk])
                    edvh = eval_linear(UGgrid_all_op, dV_h[iz], pt, xto.LINEAR)
                    phi_out[iz, ia, ih] = du_h(hk) + edvh

        return (
            {'c': c_out, 'a_nxt': a_out, 'h_nxt': h_out},
            {'V': v_out, 'phi': phi_out},
        )

    # --- adjuster_cons ---

    def adjuster_dcsn_mover(vlu_cntn, t, m_bar=1.4):
        """B: partial EGM + root-finding + FUES.

        Returns refined policies on the wealth grid.
        """
        egrid, vf, a_nxt, h_nxt = _adjEGM(
            vlu_cntn['dV']['a'], vlu_cntn['dV']['h'],
            vlu_cntn['V'], t)
        (Aadj, Cadj, Hadj, Vadj,
         _, _, _, _, _, _, _, _) = _refine_adj(
            egrid, vf, a_nxt, h_nxt,
            m_bar=m_bar, return_grids=return_grids)
        return Aadj, Cadj, Hadj, Vadj

    def adjuster_arvl_mover(Aadj, Cadj, Hadj, Vadj,
                            w_adj, vlu_cntn, t,
                            c_from_budget=1):
        """I: interp adjuster onto state grid via w_adj.

        w_adj comes from tenure branch_transitions.
        """
        v_out, c_out, a_out, h_out = _adjuster_to_grid(
            t, Aadj, Cadj, Hadj,
            vlu_cntn['V'], c_from_budget)
        return (
            {'c': c_out, 'a_nxt': a_out, 'h_nxt': h_out},
            {'V': v_out},
        )

    # --- tenure ---

    R = cp.R
    R_H = cp.R_H
    y_func = cp.y_func

    UGgrid_all = cp.UGgrid_all

    @njit
    def _eval_2d_at_h(arr_2d, h_val, a_grid):
        """Evaluate 2D array at fixed h along a_grid."""
        n = len(a_grid)
        out = np.zeros(n)
        for i in range(n):
            pt = np.array([a_grid[i], h_val])
            out[i] = eval_linear(
                UGgrid_all, arr_2d, pt, xto.LINEAR)
        return out

    def tenure_arvl_to_dcsn(t, vlu_cntn):
        """Tenure arvl_to_dcsn: compute branch grids +
        pre-evaluate continuation at branch coordinates.

        Per YAML dcsn_to_cntn_transition:
          keep:   w_keep = R*a + y(z), h_keep = (1-delta)*h
          adjust: w_adj = R*a + R_H*(1-delta)*h + y(z)

        Pre-evaluates dV_a and V at h_keep for the keeper
        so it receives pure 1D arrays (no 2D interpolation).
        """
        z = cp.z_vals
        a = cp.asset_grid_A
        h = cp.asset_grid_H
        n_z, n_a, n_h = len(z), len(a), len(h)

        # Branch grids
        w_keep = np.empty((n_z, n_a))
        for iz in range(n_z):
            w_keep[iz, :] = R * a + y_func(t, z[iz])

        w_adj = np.empty((n_z, n_a, n_h))
        for iz in range(n_z):
            for ih in range(n_h):
                w_adj[iz, :, ih] = (
                    R * a + R_H * (1 - delta) * h[ih]
                    + y_func(t, z[iz]))

        h_keep = (1 - delta) * h

        # Pre-evaluate continuation at h_keep for keeper
        # dV_a_keep[iz, :, ih] = dV_a(a_grid, h_keep[ih])
        bg = np.zeros(n_a)
        bg[:] = cp.b
        asset_grid_AC = np.concatenate((bg, a))

        dV_a = vlu_cntn['dV']['a']
        V = vlu_cntn['V']
        n_ac = len(asset_grid_AC)

        dv_keep = np.empty((n_z, n_ac, n_h))
        v_keep = np.empty((n_z, n_ac, n_h))
        for iz in range(n_z):
            for ih in range(n_h):
                dv_keep[iz, :, ih] = _eval_2d_at_h(
                    dV_a[iz], h_keep[ih], asset_grid_AC)
                v_keep[iz, :, ih] = _eval_2d_at_h(
                    V[iz], h_keep[ih], asset_grid_AC)

        return {
            'w_keep': w_keep,
            'h_keep': h_keep,
            'w_adj': w_adj,
            'dv_keep': dv_keep,
            'v_keep': v_keep,
            'asset_grid_AC': asset_grid_AC,
        }

    def tenure_dcsn_mover(branches):
        """B: max over keep/adjust branches.

        Clean DDSL interface: receives only branch-keyed
        results from the leaf stages.

        Parameters
        ----------
        branches : dict
            ``{'keep':   {'pol': {...}, 'vlu': {...}},
               'adjust': {'pol': {...}, 'vlu': {...}}}``

        Returns
        -------
        vlu_dcsn : dict
        pol : dict
        """
        keep = branches['keep']
        adj = branches['adjust']

        V, A, H, C, D, dV_a, dV_h = _branching_max(
            keep['vlu']['V'],
            keep['pol']['c'],
            keep['pol']['a_nxt'],
            keep['pol']['h_nxt'],
            keep['vlu']['phi'],
            adj['vlu']['V'],
            adj['pol']['c'],
            adj['pol']['a_nxt'],
            adj['pol']['h_nxt'],
        )

        vlu_dcsn = {'V': V, 'dV': {'a': dV_a, 'h': dV_h}}
        pol = {'c': C, 'a_nxt': A, 'h_nxt': H, 'd': D}
        return vlu_dcsn, pol

    def tenure_arvl_mover(vlu_dcsn):
        """I: E_z conditioning (Pi @ arrays)."""
        Ev, Edv_a, Edv_h = condition_V(
            vlu_dcsn['V'],
            vlu_dcsn['dV']['a'],
            vlu_dcsn['dV']['h'])
        return {'V': Ev, 'dV': {'a': Edv_a, 'h': Edv_h}}

    def tenure_arvl_mover_hd(dV_h_hd):
        """I: E_z conditioning for HD grid."""
        return condition_V_HD(dV_h_hd)

    return {
        'keeper_cons': {
            'dcsn_mover': _keeper_dcsn,
            'arvl_mover': keeper_arvl_mover,
        },
        'adjuster_cons': {
            'dcsn_mover': adjuster_dcsn_mover,
            'arvl_mover': adjuster_arvl_mover,
        },
        'tenure': {
            'arvl_to_dcsn': tenure_arvl_to_dcsn,
            'dcsn_mover': tenure_dcsn_mover,
            'arvl_mover': tenure_arvl_mover,
            'arvl_mover_hd': tenure_arvl_mover_hd,
        },
    }
