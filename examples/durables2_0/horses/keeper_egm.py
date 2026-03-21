"""Keeper consumption stage: EGM + FUES via kikku.

Uses ``make_egm_1d`` from kikku with module-level recipe
callables from ``model.py``.

The keeper receives ``h_keep`` (already depreciated by the
tenure stage's branch transition ``h_keep = (1-delta)*h``).
The keeper does NOT know about delta.

dcsn_mover = EGM + FUES. Returns refined (A, C, V) on
             the arrival asset grid. No separate arvl_mover
             — interpolation is inside dcsn_mover.
"""

import numpy as np
from numba import njit
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
from kikku.asva.egm_1d import make_egm_1d
from dcsmm.fues import FUES_jit
from dcsmm.fues.fues_v0dev import uniqueEG
from dcsmm.fues.fues_v0_2dev import (
    EPS_D as _EPS_D, EPS_SEP as _EPS_SEP,
    PARALLEL_GUARD as _PAR_GUARD,
)
from dcsmm.fues.helpers.math_funcs import interp_as
from kikku.asva.numerics import clamp_value, clamp_policy

from ..model import KEEPER_EGM_FNS


def make_keeper_ops(model):
    """Build keeper dcsn_mover + arvl_mover.

    Parameters
    ----------
    model : DurablesModel

    Returns
    -------
    dcsn_mover : callable
        ``(vlu_cntn, h_keep_grid, t, m_bar)``
        -> ``(Akeeper, Ckeeper, Vkeeper)`` on the asset
        grid (post-EGM + FUES + interpolation).
    """
    cp = model.cp

    # Grids
    z_vals = cp.z_vals
    asset_grid_A = cp.asset_grid_A
    asset_grid_H = cp.asset_grid_H
    UGgrid_all = cp.UGgrid_all

    # Scalars
    b = cp.b
    delta = cp.delta
    grid_max_A = cp.grid_max_A
    m_bar = cp.m_bar
    n_a = len(asset_grid_A)
    n_h = len(asset_grid_H)

    # h_keep + AC grid
    h_keep = (1 - delta) * asset_grid_H
    bg = np.zeros(n_a)
    bg[:] = b
    asset_grid_AC = np.concatenate((bg, asset_grid_A))

    @njit
    def _eval_2d_at_h(arr_2d, h_val, a_grid):
        n = len(a_grid)
        out = np.zeros(n)
        for i in range(n):
            pt = np.array([a_grid[i], h_val])
            out[i] = eval_linear(
                UGgrid_all, arr_2d, pt, xto.LINEAR)
        return out

    # Params + recipe callables
    egm_params = model.keeper_egm_params
    fns = KEEPER_EGM_FNS
    _egm_step = make_egm_1d(
        fns['inv_euler'], fns['bellman_rhs'],
        fns['cntn_to_dcsn'], fns['concavity'],
        egm_params,
    )

    # --- dcsn_mover: EGM + FUES ---

    def dcsn_mover(vlu_cntn):
        """B: EGM + FUES per (z, h) slice.

        Parameters
        ----------
        vlu_cntn : dict
            ``{'V': (n_z, n_a, n_h),
               'dV': {'a': ..., 'h': ...}}``.

        Returns
        -------
        Akeeper, Ckeeper, Vkeeper : ndarray (n_z, n_a, n_h)
        """
        dV_a = vlu_cntn['dV']['a']
        V = vlu_cntn['V']
        n_z_loc = len(z_vals)
        n_ac = len(asset_grid_AC)

        Akeeper = np.empty((n_z_loc, n_a, n_h))
        Ckeeper = np.empty((n_z_loc, n_a, n_h))
        Vkeeper = np.empty((n_z_loc, n_a, n_h))

        for iz in range(n_z_loc):
            for ih in range(n_h):
                hk = h_keep[ih]

                # Pre-eval 2D continuation at h_keep
                dv_1d = _eval_2d_at_h(
                    dV_a[iz], hk, asset_grid_AC)
                v_1d = _eval_2d_at_h(
                    V[iz], hk, asset_grid_AC)

                # --- EGM: constrained region ---
                c0 = fns['inv_euler'](
                    dv_1d[0], hk, egm_params)
                C_arr = np.linspace(
                    1e-08, max(1e-08, c0 - 1e-10), n_a)

                egrid = np.ones(n_a * 2)
                vf = np.ones(n_a * 2)
                c_raw = np.ones(n_a * 2)

                for k in range(n_a):
                    vf[k] = fns['bellman_rhs'](
                        C_arr[k], v_1d[0],
                        hk, egm_params)
                    egrid[k] = C_arr[k] + b
                    c_raw[k] = C_arr[k]

                # --- EGM: unconstrained via make_egm_1d ---
                x_cntn = asset_grid_AC[n_a:]
                c_hat, v_hat, x_hat, _ = _egm_step(
                    dv_1d[n_a:], np.zeros(n_a),
                    v_1d[n_a:], x_cntn, hk)

                for k in range(n_a):
                    idx = n_a + k
                    egrid[idx] = x_hat[k]
                    vf[idx] = v_hat[k]
                    c_raw[idx] = c_hat[k]

                # --- FUES refinement ---
                uid = uniqueEG(egrid, vf)
                eg_u = egrid[uid]
                vf_u = vf[uid]
                c_u = c_raw[uid]
                ac_u = asset_grid_AC[uid]

                sidx = np.argsort(eg_u)
                (eg_ref, vf_ref, c_ref,
                 a_ref, _) = FUES_jit(
                    eg_u[sidx], vf_u[sidx],
                    c_u[sidx], ac_u[sidx],
                    ac_u[sidx].copy(),
                    m_bar, 10,
                    False, 0.0,
                    False, True, False, True,
                    _EPS_D, _EPS_SEP, 0.05,
                    _PAR_GUARD)

                # --- Interpolate to asset grid ---
                a_interp = interp_as(
                    eg_ref, a_ref, asset_grid_A,
                    extrap=True)
                c_interp = interp_as(
                    eg_ref, c_ref, asset_grid_A,
                    extrap=True)
                v_interp = interp_as(
                    eg_ref, vf_ref, asset_grid_A,
                    extrap=True)

                Akeeper[iz, :, ih] = clamp_policy(
                    a_interp, b, grid_max_A * 2)
                Ckeeper[iz, :, ih] = clamp_policy(
                    c_interp, 1e-10, 1e10)
                Vkeeper[iz, :, ih] = clamp_value(
                    v_interp)

        return Akeeper, Ckeeper, Vkeeper

    return dcsn_mover
