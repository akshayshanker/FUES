"""Keeper consumption stage: EGM via kikku.

Uses ``make_egm_1d`` from kikku with module-level recipe
callables defined in ``model.py``. Per (z, h) slice, the
keeper is a standard 1D EGM on liquid assets with housing
as a fixed state (pass-through dimension).
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

    Recipe callables are module-level @njit from model.py.
    ``make_egm_1d`` from kikku compiles the generic step.

    Parameters
    ----------
    model : DurablesModel

    Returns
    -------
    dcsn_mover, arvl_mover : callable
    """
    cp = model.cp

    # Grids (closure scope)
    z_vals = cp.z_vals
    asset_grid_A = cp.asset_grid_A
    asset_grid_H = cp.asset_grid_H
    bg = np.zeros(len(asset_grid_A))
    bg.fill(cp.b)
    asset_grid_AC = np.concatenate((bg, asset_grid_A))
    UGgrid_all = cp.UGgrid_all

    # Scalars
    delta = cp.delta
    b = cp.b
    grid_max_A = cp.grid_max_A

    # Params array for recipe callables
    egm_params = model.keeper_egm_params

    # Build generic EGM step from model-level callables
    fns = KEEPER_EGM_FNS
    _egm_step = make_egm_1d(
        fns['inv_euler'], fns['bellman_rhs'],
        fns['cntn_to_dcsn'], fns['concavity'],
        egm_params,
    )

    # Pre-evaluate 2D arrays at off-grid h'

    @njit
    def _eval_at_hprime(arr_2d, h_prime, a_grid):
        n = len(a_grid)
        out = np.zeros(n)
        for i in range(n):
            pt = np.array([a_grid[i], h_prime])
            out[i] = eval_linear(
                UGgrid_all, arr_2d, pt, xto.LINEAR)
        return out

    # --- dcsn_mover ---

    def dcsn_mover(vlu_cntn, t):
        """B: EGM per (z, h) slice via make_egm_1d."""
        n_z = len(z_vals)
        n_a = len(asset_grid_A)
        n_h = len(asset_grid_H)
        egrid = np.ones((n_z, n_a * 2, n_h))
        vf = np.ones((n_z, n_a * 2, n_h))
        c_out = np.ones((n_z, n_a * 2, n_h))

        dV_a = vlu_cntn['dV']['a']
        V = vlu_cntn['V']

        for iz in range(n_z):
            for ih in range(n_h):
                h_prime = asset_grid_H[ih] * (1 - delta)

                # Pre-eval continuation at off-grid h'
                dv_1d = _eval_at_hprime(
                    dV_a[iz], h_prime, asset_grid_AC)
                v_1d = _eval_at_hprime(
                    V[iz], h_prime, asset_grid_AC)

                # Constrained region: a' = b
                c0 = fns['inv_euler'](
                    dv_1d[0], h_prime, egm_params)
                C_arr = np.linspace(
                    1e-08, max(1e-08, c0 - 1e-10), n_a)
                v_at_b = egm_params[0] * v_1d[0]

                for k in range(n_a):
                    vf[iz, k, ih] = fns['bellman_rhs'](
                        C_arr[k], v_1d[0],
                        h_prime, egm_params)
                    egrid[iz, k, ih] = C_arr[k] + b
                    c_out[iz, k, ih] = C_arr[k]

                # Unconstrained: generic EGM step
                x_cntn = asset_grid_AC[n_a:]
                dv_unc = dv_1d[n_a:]
                v_unc = v_1d[n_a:]
                ddv_unc = np.zeros(n_a)

                c_hat, v_hat, x_hat, _ = _egm_step(
                    dv_unc, ddv_unc, v_unc,
                    x_cntn, h_prime)

                for k in range(n_a):
                    idx = n_a + k
                    egrid[iz, idx, ih] = x_hat[k]
                    vf[iz, idx, ih] = v_hat[k]
                    c_out[iz, idx, ih] = c_hat[k]

        return egrid, vf, c_out

    # --- arvl_mover ---

    def arvl_mover(egrid, vf, c, vlu_cntn, t, m_bar=1.1):
        """I: FUES + interpolation to arrival grid."""
        n_z = len(z_vals)
        n_h = len(asset_grid_H)
        n_a = len(asset_grid_A)

        Akeeper = np.empty((n_z, n_a, n_h))
        Ckeeper = np.empty((n_z, n_a, n_h))
        Vkeeper = np.empty((n_z, n_a, n_h))
        e_clean = np.empty((n_z, n_a, n_h))
        v_clean = np.empty((n_z, n_a, n_h))
        c_clean = np.empty((n_z, n_a, n_h))
        a_clean = np.empty((n_z, n_a, n_h))

        for iz in range(n_z):
            for ih in range(n_h):
                eg_raw = egrid[iz, :, ih]
                vf_raw = vf[iz, :, ih]
                c_raw = c[iz, :, ih]

                uid = uniqueEG(eg_raw, vf_raw)
                eg_u = eg_raw[uid]
                vf_u = vf_raw[uid]
                c_u = c_raw[uid]
                ac_u = asset_grid_AC[uid]

                sidx = np.argsort(eg_u)
                eg_s = eg_u[sidx]
                vf_s = vf_u[sidx]
                c_s = c_u[sidx]
                a_s = ac_u[sidx]
                da_s = a_s.copy()

                (eg_ref, vf_ref, c_ref,
                 a_ref, _) = FUES_jit(
                    eg_s, vf_s, c_s, a_s, da_s,
                    m_bar, 10,
                    False, 0.0,
                    False, True, False, True,
                    _EPS_D, _EPS_SEP, 0.05, _PAR_GUARD)

                a_interp = interp_as(
                    eg_ref, a_ref, asset_grid_A,
                    extrap=True)
                c_interp = interp_as(
                    eg_ref, c_ref, asset_grid_A,
                    extrap=True)
                v_interp = interp_as(
                    eg_ref, vf_ref, asset_grid_A,
                    extrap=True)

                a_interp = clamp_policy(
                    a_interp, b, grid_max_A * 2)
                c_interp = clamp_policy(
                    c_interp, 1e-10, 1e10)
                v_interp = clamp_value(v_interp)

                Ckeeper[iz, :, ih] = c_interp
                Vkeeper[iz, :, ih] = v_interp
                Akeeper[iz, :, ih] = a_interp

                e_clean[iz, :, ih] = v_interp
                v_clean[iz, :, ih] = v_interp
                c_clean[iz, :, ih] = c_interp
                a_clean[iz, :, ih] = a_interp

        return (Akeeper, Ckeeper, Vkeeper,
                e_clean, v_clean, c_clean, a_clean)

    return dcsn_mover, arvl_mover
