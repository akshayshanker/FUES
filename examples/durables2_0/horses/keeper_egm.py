"""Keeper consumption stage: EGM via kikku.

Uses ``make_egm_1d`` from kikku with model-specific recipe
callables. Per (z, h) slice, the keeper is a standard 1D
EGM on liquid assets with housing as a fixed state.
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


def make_keeper_ops(model):
    """Build keeper dcsn_mover + arvl_mover.

    Uses ``make_egm_1d`` from kikku with recipe callables
    derived from the model's equation system.

    Parameters
    ----------
    model : DurablesModel

    Returns
    -------
    keeper_egm : callable
        dcsn_mover: ``(vlu_cntn, t) -> (egrid, vf, c)``
    refine_keeper : callable
        arvl_mover: ``(egrid, vf, c, vlu_cntn, t, m_bar)``
        -> ``(pol, vlu)``
    """
    cp = model.cp

    # --- Grids (closure scope) ---
    z_vals = cp.z_vals
    asset_grid_A = cp.asset_grid_A
    asset_grid_H = cp.asset_grid_H
    bg = np.zeros(len(asset_grid_A))
    bg.fill(cp.b)
    asset_grid_AC = np.concatenate((bg, asset_grid_A))
    UGgrid_all = cp.UGgrid_all

    # --- Scalars ---
    beta = cp.beta
    delta = cp.delta
    b = cp.b
    grid_max_A = cp.grid_max_A

    # --- Callables from model ---
    u = cp.u
    du_c_inv = cp.du_c_inv

    # --- EGM recipe callables ---
    # params = [beta] (used by bellman_rhs)
    egm_params = np.array([beta])

    @njit
    def fn_inv_euler(dv_cntn_i, fixed_state, params):
        """InvEuler: c = (du_c)^{-1}(dv_cntn_i)."""
        return du_c_inv(dv_cntn_i)

    @njit
    def fn_bellman_rhs(c_i, v_cntn_i, fixed_state, params):
        """Bellman: v = u(c, h') + beta * V_cntn."""
        h_prime = fixed_state
        beta_p = params[0]
        return u(c_i, h_prime, 0.0) + beta_p * v_cntn_i

    @njit
    def fn_cntn_to_dcsn(c_i, x_cntn_i, fixed_state, params):
        """Transition: w = c + a' (endogenous grid)."""
        return c_i + x_cntn_i

    @njit
    def fn_concavity(c_i, ddv_cntn_i, fixed_state, params):
        """Concavity diagnostic (unused for keeper, return a')."""
        return 0.0  # keeper doesn't use del_a

    # --- Build the generic EGM step ---
    _egm_step = make_egm_1d(
        fn_inv_euler, fn_bellman_rhs,
        fn_cntn_to_dcsn, fn_concavity,
        egm_params,
    )

    # --- Pre-evaluate dV_a at off-grid h' ---

    @njit
    def _eval_dv_at_hprime(dV_a_2d, h_prime, a_grid):
        """Evaluate 2D dV_a at fixed h' along a_grid.

        Returns 1D array: dV_a(a_grid[i], h_prime).
        """
        n = len(a_grid)
        dv_1d = np.zeros(n)
        for i in range(n):
            pt = np.array([a_grid[i], h_prime])
            dv_1d[i] = eval_linear(
                UGgrid_all, dV_a_2d, pt, xto.LINEAR)
        return dv_1d

    @njit
    def _eval_v_at_hprime(V_2d, h_prime, a_grid):
        """Evaluate 2D V at fixed h' along a_grid."""
        n = len(a_grid)
        v_1d = np.zeros(n)
        for i in range(n):
            pt = np.array([a_grid[i], h_prime])
            v_1d[i] = eval_linear(
                UGgrid_all, V_2d, pt, xto.LINEAR)
        return v_1d

    # --- dcsn_mover: keeper EGM ---

    def keeper_dcsn_mover(vlu_cntn, t):
        """B: EGM per (z, h) slice via make_egm_1d.

        For each slice: pre-evaluate dV_a and V at the
        off-grid h' = (1-delta)*h, then run the generic
        1D EGM step. Concatenates constrained + unconstrained
        regions into a 2x-sized grid.
        """
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

                # Pre-evaluate continuation at off-grid h'
                dv_1d = _eval_dv_at_hprime(
                    dV_a[iz], h_prime, asset_grid_AC)
                v_1d = _eval_v_at_hprime(
                    V[iz], h_prime, asset_grid_AC)

                # Constrained region: a' = b (first n_a pts)
                # dv_1d[:n_a] all evaluated at a'=b
                c0 = du_c_inv(dv_1d[0])
                C_arr = np.linspace(
                    1e-08, max(1e-08, c0 - 1e-10), n_a)
                v_at_b = beta * v_1d[0]

                for k in range(n_a):
                    vf[iz, k, ih] = (
                        u(C_arr[k], h_prime, 0.0)
                        + v_at_b)
                    egrid[iz, k, ih] = C_arr[k] + b
                    c_out[iz, k, ih] = C_arr[k]

                # Unconstrained: use generic EGM step
                # x_cntn = asset_grid_AC[n_a:]
                x_cntn = asset_grid_AC[n_a:]
                dv_unc = dv_1d[n_a:]
                v_unc = v_1d[n_a:]
                ddv_unc = np.zeros(n_a)  # unused

                c_hat, v_hat, x_hat, _ = _egm_step(
                    dv_unc, ddv_unc, v_unc,
                    x_cntn, h_prime)

                for k in range(n_a):
                    idx = n_a + k
                    egrid[iz, idx, ih] = x_hat[k]
                    vf[iz, idx, ih] = v_hat[k]
                    c_out[iz, idx, ih] = c_hat[k]

        return egrid, vf, c_out

    # --- arvl_mover: FUES refinement ---

    def keeper_arvl_mover(egrid, vf, c, vlu_cntn,
                          t, m_bar=1.1):
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

                e_clean[iz, :, ih] = eg_ref[0] if len(
                    eg_ref) else 0.0
                v_clean[iz, :, ih] = v_interp
                c_clean[iz, :, ih] = c_interp
                a_clean[iz, :, ih] = a_interp

        return (Akeeper, Ckeeper, Vkeeper,
                e_clean, v_clean, c_clean, a_clean)

    return keeper_dcsn_mover, keeper_arvl_mover
