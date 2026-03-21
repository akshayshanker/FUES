"""Keeper consumption stage EGM kernels.

Factory function that creates @njit closures for the keeper
EGM step and FUES refinement. The closures capture model
grids and callables — this is necessary because numba @njit
cannot receive Python dicts or callables as arguments.

The factory pattern is the same as kikku/asva/egm_1d.py:
a Python function creates and returns @njit closures that
close over the model's numerical resources.
"""

import numpy as np
from numba import njit
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
from dcsmm.fues import FUES_jit
from dcsmm.fues.fues_v0dev import uniqueEG
from dcsmm.fues.fues_v0_2dev import (
    EPS_D as _EPS_D, EPS_SEP as _EPS_SEP,
    PARALLEL_GUARD as _PAR_GUARD,
)
from dcsmm.fues.helpers.math_funcs import interp_as
from kikku.asva.numerics import (
    clamp_value, clamp_policy,
)


def make_keeper_egm(model):
    """Build keeper EGM + FUES refinement closures.

    Parameters
    ----------
    model : DurablesModel

    Returns
    -------
    keeper_egm : callable
        ``(dV_a, V, t) -> (egrid, vf, c)``
    refine_keeper : callable
        ``(egrid, vf, c, V, t, m_bar) -> (A, C, V, ...)``
    """
    # Unpack into closure scope — must match the types
    # that Operator_Factory uses, so numba sees the same
    # captured-variable types for @njit closures.
    cp = model.cp
    z_vals = cp.z_vals
    asset_grid_A = cp.asset_grid_A
    asset_grid_H = cp.asset_grid_H
    bg = np.zeros(len(cp.asset_grid_A))
    bg.fill(cp.b)
    asset_grid_AC = np.concatenate((bg, cp.asset_grid_A))
    UGgrid_all = cp.UGgrid_all
    X_all = cp.X_all

    beta = cp.beta
    delta = cp.delta
    b = cp.b
    grid_max_A = cp.grid_max_A

    u = cp.u
    du_c_inv = cp.du_c_inv

    return_grids = cp.return_grids

    # --- InvEuler (2D interpolation for off-grid h') ---

    @njit
    def _inv_euler_2d(a_prime, h_prime, dV_a_slice):
        """InvEuler: interpolate dV_a at (a', h'), invert."""
        point = np.array([a_prime, h_prime])
        dV_val = eval_linear(
            UGgrid_all, dV_a_slice, point, xto.LINEAR)
        c = du_c_inv(dV_val)
        return c + a_prime, c

    # --- Keeper EGM kernel ---

    @njit
    def keeper_egm(dV_a, V, t):
        """EGM step for keeper (per z, h slice).

        Returns unrefined endogenous grid, value, and
        consumption on a 2x-sized grid (constrained +
        unconstrained regions).
        """
        n_z = len(z_vals)
        n_a = len(asset_grid_A)
        n_h = len(asset_grid_H)
        egrid = np.ones((n_z, n_a * 2, n_h))
        vf = np.ones((n_z, n_a * 2, n_h))
        c = np.ones((n_z, n_a * 2, n_h))

        for iz in range(n_z):
            for ih in range(n_h):
                h_prime = asset_grid_H[ih] * (1 - delta)

                # Constrained region: a' = b
                _, c0 = _inv_euler_2d(
                    b, h_prime, dV_a[iz])
                C_arr = np.linspace(
                    1e-08, max(1e-08, c0 - 1e-10), n_a)
                pt = np.array([b, h_prime])
                v_prime = beta * eval_linear(
                    UGgrid_all, V[iz], pt, xto.LINEAR)

                for k in range(n_a):
                    vf[iz, k, ih] = (
                        u(C_arr[k], h_prime, 0)
                        + v_prime)
                    egrid[iz, k, ih] = C_arr[k] + b
                    c[iz, k, ih] = C_arr[k]

                # Unconstrained region
                for ia in range(n_a):
                    idx = n_a + ia
                    a_prime = asset_grid_AC[idx]
                    hp = asset_grid_H[ih] * (1 - delta)

                    egm_a, c_val = _inv_euler_2d(
                        a_prime, hp, dV_a[iz])
                    egrid[iz, idx, ih] = egm_a
                    c[iz, idx, ih] = c_val

                    pt2 = np.array([a_prime, hp])
                    vp = beta * eval_linear(
                        UGgrid_all, V[iz], pt2,
                        xto.LINEAR)
                    vf[iz, idx, ih] = (
                        u(c_val, hp, 0) + vp)

        return egrid, vf, c

    # --- FUES refinement ---

    def refine_keeper(egrid, vf, c, V, t, m_bar=1.1):
        """FUES + interpolation to arrival grid.

        Not @njit because FUES_jit is called per slice.
        """
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

                # Remove duplicates (uniqueEG returns indices)
                uid = uniqueEG(eg_raw, vf_raw)
                eg_u = eg_raw[uid]
                vf_u = vf_raw[uid]
                c_u = c_raw[uid]
                ac_u = asset_grid_AC[uid]

                # Sort by endogenous grid
                sidx = np.argsort(eg_u)
                eg_s = eg_u[sidx]
                vf_s = vf_u[sidx]
                c_s = c_u[sidx]
                a_s = ac_u[sidx]
                da_s = a_s.copy()

                # FUES
                (eg_ref, vf_ref, c_ref,
                 a_ref, _) = FUES_jit(
                    eg_s, vf_s, c_s, a_s, da_s,
                    m_bar, 10,
                    False, 0.0,
                    False, True, False, True,
                    _EPS_D, _EPS_SEP, 0.05, _PAR_GUARD)

                # Interpolate to asset grid (extrap=True)
                a_interp = interp_as(
                    eg_ref, a_ref, asset_grid_A,
                    extrap=True)
                c_interp = interp_as(
                    eg_ref, c_ref, asset_grid_A,
                    extrap=True)
                v_interp = interp_as(
                    eg_ref, vf_ref, asset_grid_A,
                    extrap=True)

                # Clamp
                a_interp = clamp_policy(
                    a_interp, b, grid_max_A * 2)
                c_interp = clamp_policy(
                    c_interp, 1e-10, 1e10)
                v_interp = clamp_value(v_interp)

                a_pol = a_interp

                Ckeeper[iz, :, ih] = c_interp
                Vkeeper[iz, :, ih] = v_interp
                Akeeper[iz, :, ih] = a_pol

                if return_grids:
                    e_clean[iz, :, ih] = interp_as(
                        eg_ref, eg_ref, asset_grid_A)
                    v_clean[iz, :, ih] = v_interp
                    c_clean[iz, :, ih] = c_interp
                    a_clean[iz, :, ih] = a_pol

        return (Akeeper, Ckeeper, Vkeeper,
                e_clean, v_clean, c_clean, a_clean)

    return keeper_egm, refine_keeper
