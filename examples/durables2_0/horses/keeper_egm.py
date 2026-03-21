"""Keeper consumption stage: EGM + FUES via kikku.

Uses ``make_egm_1d`` from kikku with module-level recipe
callables from ``model.py``.

The keeper receives ``h_keep`` (already depreciated by the
tenure stage's branch transition ``h_keep = (1-delta)*h``).
The keeper does NOT know about delta.

dcsn_mover = EGM + FUES (returns refined arrays on
             the endogenous grid).
arvl_mover = interpolation onto the arrival asset grid.
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
        -> ``(A_ref, C_ref, V_ref)`` on the endogenous
        grid (post-FUES).
    arvl_mover : callable
        ``(A_ref, C_ref, V_ref)``
        -> ``(pol, vlu)`` on the arrival asset grid.
    """
    cp = model.cp

    # Grids
    z_vals = cp.z_vals
    asset_grid_A = cp.asset_grid_A
    asset_grid_H = cp.asset_grid_H
    bg = np.zeros(len(asset_grid_A))
    bg.fill(cp.b)
    asset_grid_AC = np.concatenate((bg, asset_grid_A))
    UGgrid_all = cp.UGgrid_all

    # Scalars
    b = cp.b
    grid_max_A = cp.grid_max_A

    # Params + recipe callables
    egm_params = model.keeper_egm_params
    fns = KEEPER_EGM_FNS
    _egm_step = make_egm_1d(
        fns['inv_euler'], fns['bellman_rhs'],
        fns['cntn_to_dcsn'], fns['concavity'],
        egm_params,
    )

    @njit
    def _eval_at_hprime(arr_2d, h_keep, a_grid):
        """Evaluate 2D array at fixed h_keep along a_grid."""
        n = len(a_grid)
        out = np.zeros(n)
        for i in range(n):
            pt = np.array([a_grid[i], h_keep])
            out[i] = eval_linear(
                UGgrid_all, arr_2d, pt, xto.LINEAR)
        return out

    # --- dcsn_mover: EGM + FUES ---

    def dcsn_mover(vlu_cntn, h_keep_grid, t, m_bar=1.1):
        """B: EGM + FUES per (z, h) slice.

        Parameters
        ----------
        vlu_cntn : dict
            ``{'V', 'dV': {'a'}}``.
        h_keep_grid : ndarray (n_h,)
            Depreciated housing grid, from tenure's
            branch transition: ``(1-delta) * asset_grid_H``.
        t : int
        m_bar : float

        Returns
        -------
        Akeeper, Ckeeper, Vkeeper : ndarray (n_z, n_a, n_h)
            Refined policies on the arrival asset grid.
        """
        n_z = len(z_vals)
        n_a = len(asset_grid_A)
        n_h = len(asset_grid_H)

        Akeeper = np.empty((n_z, n_a, n_h))
        Ckeeper = np.empty((n_z, n_a, n_h))
        Vkeeper = np.empty((n_z, n_a, n_h))

        dV_a = vlu_cntn['dV']['a']
        V = vlu_cntn['V']

        for iz in range(n_z):
            for ih in range(n_h):
                h_keep = h_keep_grid[ih]

                # Pre-eval continuation at h_keep
                dv_1d = _eval_at_hprime(
                    dV_a[iz], h_keep, asset_grid_AC)
                v_1d = _eval_at_hprime(
                    V[iz], h_keep, asset_grid_AC)

                # --- EGM: constrained region ---
                c0 = fns['inv_euler'](
                    dv_1d[0], h_keep, egm_params)
                C_arr = np.linspace(
                    1e-08, max(1e-08, c0 - 1e-10), n_a)

                egrid = np.ones(n_a * 2)
                vf = np.ones(n_a * 2)
                c_raw = np.ones(n_a * 2)

                for k in range(n_a):
                    vf[k] = fns['bellman_rhs'](
                        C_arr[k], v_1d[0],
                        h_keep, egm_params)
                    egrid[k] = C_arr[k] + b
                    c_raw[k] = C_arr[k]

                # --- EGM: unconstrained via make_egm_1d ---
                x_cntn = asset_grid_AC[n_a:]
                c_hat, v_hat, x_hat, _ = _egm_step(
                    dv_1d[n_a:], np.zeros(n_a),
                    v_1d[n_a:], x_cntn, h_keep)

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

    # --- arvl_mover: interpolation to state grid ---

    def arvl_mover(Akeeper, Ckeeper, Vkeeper,
                   vlu_cntn, h_keep_grid, t):
        """I: interpolate keeper onto (z, a, h) state grid.

        Computes wealth = R*a + y(t,z) per state point,
        interpolates keeper policies, and returns Phi_t
        (housing marginal).

        Returns
        -------
        pol : dict
            ``{'c', 'a_nxt', 'h_nxt'}`` on state grid.
        vlu : dict
            ``{'V', 'phi'}`` on state grid.
        """
        # arvl_mover delegates to interp_to_grid
        # (extracted separately in Phase 3)
        # For now, return the keeper arrays directly
        # (they're already on the asset grid from dcsn_mover)
        return (
            {'c': Ckeeper, 'a_nxt': Akeeper,
             'h_nxt': np.broadcast_to(
                 h_keep_grid[np.newaxis, np.newaxis, :],
                 Akeeper.shape).copy()},
            {'V': Vkeeper},
        )

    return dcsn_mover, arvl_mover
