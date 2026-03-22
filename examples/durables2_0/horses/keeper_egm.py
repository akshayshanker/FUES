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


def make_keeper_ops(cp, callables):
    """Build keeper dcsn_mover + arvl_mover.

    Parameters
    ----------
    cp : ConsumerProblem
    callables : dict

    Returns
    -------
    dcsn_mover : callable
        ``(vlu_cntn, h_keep_grid, t, m_bar)``
        -> ``(Akeeper, Ckeeper, Vkeeper)`` on the asset
        grid (post-EGM + FUES + interpolation).
    """
    # Scalars + callables
    b = cp.b
    delta = cp.delta
    beta = cp.beta
    grid_max_A = cp.grid_max_A
    m_bar = cp.m_bar
    du_c = callables["du_c"]
    du_h = callables["du_h"]
    # Params + recipe callables (built here, not on a wrapper object)
    egm_params = np.array([
        cp.beta,      # [0]
        cp.alpha,     # [1]
        cp.gamma_c,   # [2]
        cp.gamma_h,   # [3]
        cp.kappa,     # [4]
    ])
    fns = KEEPER_EGM_FNS
    _egm_step = make_egm_1d(
        fns['inv_euler'], fns['bellman_rhs'],
        fns['cntn_to_dcsn'], fns['concavity'],
        egm_params,
    )

    # --- dcsn_mover: EGM + FUES ---

    def dcsn_mover(vlu_cntn, grids):
        """EGM + FUES per (z, h) slice.

        Parameters
        ----------
        vlu_cntn : dict
            ``{'V': (n_z, n_a, n_h),
               'dV': {'a': ..., 'h': ...}}``.

        Returns
        -------
        Akeeper, Ckeeper, Vkeeper, dVw_keep, phi_keep
        """
        dV_a = vlu_cntn['d_aV']
        dV_h = vlu_cntn['d_hV']
        V = vlu_cntn['V']
        z_vals = grids["z"]
        asset_grid_A = grids["a"]
        asset_grid_H = grids["h"]
        n_z_loc = len(z_vals)
        n_a = len(asset_grid_A)
        n_h = len(asset_grid_H)
        h_keep = (1 - delta) * asset_grid_H

        Akeeper = np.empty((n_z_loc, n_a, n_h))
        Ckeeper = np.empty((n_z_loc, n_a, n_h))
        Vkeeper = np.empty((n_z_loc, n_a, n_h))
        dVw_keep = np.empty((n_z_loc, n_a, n_h))
        phi_keep = np.empty((n_z_loc, n_a, n_h))

        for iz in range(n_z_loc):
            for ih in range(n_h):
                hk = h_keep[ih]

                # Slice continuation at ih (housing is identity)
                dv_slice = dV_a[iz, :, ih]
                v_slice = V[iz, :, ih]

                # --- EGM: constrained region ---
                c0 = fns['inv_euler'](
                    dv_slice[0], hk, egm_params)
                C_arr = np.linspace(
                    1e-08, max(1e-08, c0 - 1e-10), n_a)

                egrid = np.ones(n_a * 2)
                vf = np.ones(n_a * 2)
                c_raw = np.ones(n_a * 2)

                for k in range(n_a):
                    vf[k] = fns['bellman_rhs'](
                        C_arr[k], v_slice[0],
                        hk, egm_params)
                    egrid[k] = C_arr[k] + b
                    c_raw[k] = C_arr[k]

                # --- EGM: unconstrained via make_egm_1d ---
                c_hat, v_hat, x_hat, _ = _egm_step(
                    dv_slice, np.zeros(n_a),
                    v_slice, asset_grid_A, hk)

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

                # a_cntn for FUES: constrained = b, unconstrained = a_grid
                ac_arr = np.concatenate((
                    np.full(n_a, b), asset_grid_A))
                ac_u = ac_arr[uid]

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

                a_clamped = clamp_policy(
                    a_interp, b, grid_max_A * 2)
                c_clamped = clamp_policy(
                    c_interp, 1e-10, 1e10)

                Akeeper[iz, :, ih] = a_clamped
                Ckeeper[iz, :, ih] = c_clamped
                Vkeeper[iz, :, ih] = clamp_value(
                    v_interp)

                # dV_w = du_c(c) for each grid point
                for ia in range(n_a):
                    dVw_keep[iz, ia, ih] = du_c(
                        c_clamped[ia])

                # phi = du_h(h_keep) + dV_h_cntn(a', h_keep)
                # dV_h already includes beta from recursion
                dv_h_slice = dV_h[iz, :, ih]
                for ia in range(n_a):
                    edvh = np.interp(
                        a_clamped[ia], asset_grid_A,
                        dv_h_slice)
                    phi_keep[iz, ia, ih] = du_h(hk) + edvh

        return Akeeper, Ckeeper, Vkeeper, dVw_keep, phi_keep

    return dcsn_mover
