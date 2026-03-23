"""Keeper consumption stage: EGM + FUES via kikku.

Uses ``make_egm_1d`` from kikku with EGM recipe callables from
``make_callables`` (keeper_egm_fns).

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
from dcsmm.fues import FUES_jit
from dcsmm.fues.fues_v0dev import uniqueEG
from dcsmm.fues.fues_v0_2dev import (
    EPS_D as _EPS_D, EPS_SEP as _EPS_SEP,
    PARALLEL_GUARD as _PAR_GUARD,
)
from dcsmm.fues.helpers.math_funcs import interp_as
from kikku.asva.numerics import clamp_value, clamp_policy
from kikku.asva.egm_1d import make_egm_1d


def make_keeper_ops(callables, keep_h_fn, grids, settings):
    """Build keeper dcsn_mover.

    Parameters
    ----------
    callables : dict
        Stage dict for keeper_cons (includes keeper_egm_fns).
    keep_h_fn : callable
        ``h -> (1-delta)*h`` from tenure transitions.
    grids : dict
    settings : dict

    Returns
    -------
    dcsn_mover : callable
        ``(vlu_cntn, grids)``
        -> ``(Akeeper, Ckeeper, Vkeeper, dVw_keep, phi_keep, cntn_data)``
        where cntn_data is None or dict per solution_scheme.md.
    """
    b = settings["b"]
    g_keep_h = keep_h_fn
    grid_max_A = settings["grid_max_A"]
    m_bar = settings["m_bar"]
    return_grids = settings["return_grids"]
    d_c_u = callables["d_c_u"]
    d_h_u = callables["d_h_u"]
    fns = callables["keeper_egm_fns"]
    _egm_step = make_egm_1d(
        fns["inv_euler"],
        fns["bellman_rhs"],
        fns["cntn_to_dcsn"],
        fns["concavity"],
    )

    def dcsn_mover(vlu_cntn, grids):
        """EGM + FUES per (z, h) slice.

        Returns
        -------
        Akeeper, Ckeeper, Vkeeper, dVw_keep, phi_keep, cntn_data
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
        h_keep = np.array([g_keep_h(h) for h in asset_grid_H])

        Akeeper = np.empty((n_z_loc, n_a, n_h))
        Ckeeper = np.empty((n_z_loc, n_a, n_h))
        Vkeeper = np.empty((n_z_loc, n_a, n_h))
        dVw_keep = np.empty((n_z_loc, n_a, n_h))
        phi_keep = np.empty((n_z_loc, n_a, n_h))

        if return_grids:
            cntn_c = {}
            cntn_m_endog = {}
        else:
            cntn_c = None
            cntn_m_endog = None

        for iz in range(n_z_loc):
            for ih in range(n_h):
                hk = h_keep[ih]

                dv_slice = dV_a[iz, :, ih]
                v_slice = V[iz, :, ih]

                # --- EGM: constrained region ---
                c0 = fns['inv_euler'](dv_slice[0], hk)
                C_arr = np.linspace(
                    1e-08, max(1e-08, c0 - 1e-10), n_a)

                egrid = np.ones(n_a * 2)
                vf = np.ones(n_a * 2)
                c_raw = np.ones(n_a * 2)

                for k in range(n_a):
                    vf[k] = fns['bellman_rhs'](C_arr[k], v_slice[0], hk)
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

                if cntn_c is not None:
                    cntn_c[(iz, ih)] = c_ref.copy()
                    cntn_m_endog[(iz, ih)] = eg_ref.copy()

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

                for ia in range(n_a):
                    dVw_keep[iz, ia, ih] = d_c_u(
                        c_clamped[ia])

                dv_h_slice = dV_h[iz, :, ih]
                for ia in range(n_a):
                    edvh = np.interp(
                        a_clamped[ia], asset_grid_A,
                        dv_h_slice)
                    phi_keep[iz, ia, ih] = d_h_u(hk) + edvh

        cntn_data = None
        if cntn_c is not None:
            cntn_data = {
                'c': cntn_c,
                'm_endog': cntn_m_endog,
            }

        return Akeeper, Ckeeper, Vkeeper, dVw_keep, phi_keep, cntn_data

    return dcsn_mover


# ------------------------------------------------------------------
# Forward (simulation) operator
# ------------------------------------------------------------------

def make_keeper_forward(C_keep_t, UG,
                        edata_next, grids, callables, settings,
                        euler_panel, t_val):
    """StageForward for keeper_cons with inline Euler.

    Composes: arvl_to_dcsn (identity) -> policy (C interp)
    -> dcsn_to_cntn (budget constraint + Euler side-channel).

    Euler values are written to ``euler_panel[t_val, idx]``
    via the ``_idx`` side-channel, NOT emitted in poststates.
    """
    from kikku.asva.simulate import StageForward
    from ..simulate import keeper_euler, _eval_keeper_c

    b = settings["b"]
    gA = settings["grid_max_A"]
    gH = settings["grid_max_H"]

    def arvl_to_dcsn(particles, shocks):
        return particles

    def policy(particles):
        w = particles['w_keep']
        h = particles['h_keep']
        z_idx = particles['z_idx']
        N = len(w)
        c = np.empty(N)
        for i in range(N):
            c[i] = _eval_keeper_c(w[i], h[i], int(z_idx[i]),
                                   C_keep_t, UG)
        return {'c': c}

    def dcsn_to_cntn(particles, controls, shocks):
        w = particles['w_keep']
        h_keep = particles['h_keep']
        c = controls['c']
        z_idx = particles['z_idx']
        N = len(w)
        a_nxt = np.clip(w - c, b, gA)
        h_nxt = np.clip(h_keep, b, gH)

        idx = particles.get('_idx')
        if edata_next is not None and idx is not None:
            for i in range(N):
                ci, ai, hi = c[i], a_nxt[i], h_nxt[i]
                if ci > 0.1 and ai > 0.1:
                    euler_panel[t_val, int(idx[i])] = keeper_euler(
                        ci, ai, hi, int(z_idx[i]),
                        edata_next, grids, callables)

        out = {'a_nxt': a_nxt, 'h_nxt': h_nxt,
               'z_idx': z_idx.copy()}
        if idx is not None:
            out['_idx'] = idx.copy()
        return out

    return StageForward(
        arvl_to_dcsn=arvl_to_dcsn, policy=policy,
        dcsn_to_cntn=dcsn_to_cntn,
        shock_draw_arvl=None, shock_draw_cntn=None,
    )
