"""Keeper consumption stage: EGM + FUES via kikku.

Uses ``make_egm_1d`` from kikku with EGM recipe callables from
``make_callables`` (keeper_egm_fns).

The keeper receives ``h_keep`` (already depreciated by the
tenure stage's branch transition ``h_keep = (1-delta)*h``).
The keeper does NOT know about delta.

dcsn_mover = EGM + FUES. Returns refined (A, C, V) on
             the arrival asset grid. No separate arvl_mover
             — interpolation is inside dcsn_mover.

For upper-envelope methods FUES / FUES_V0*DEV, the keeper uses a single
``@njit(parallel=True)`` kernel that calls ``FUES_jit`` directly (same
pattern as ``adjuster_egm.fues_refine``). Other UE methods still go
through ``EGM_UE``.
"""

import numpy as np
from numba import njit, prange
from dcsmm.fues.fues_v0dev import uniqueEG
from dcsmm.fues.fues_v0_2dev import (
    FUES_jit,
    EPS_D as _FUES_EPS_D,
    EPS_SEP as _FUES_EPS_SEP,
    PARALLEL_GUARD as _FUES_PAR_GUARD,
)
from dcsmm.fues.helpers.math_funcs import interp_as, interp_as_scalar
from dcsmm.uenvelope import EGM_UE
from kikku.asva.numerics import clamp_value, clamp_policy
from kikku.asva.egm_1d import make_egm_1d

# Adjuster FUES_jit uses 0.05 for forward/back scan proximity (not EPS_fwd_back=0.5).
_FUES_EPS_FWD_BACK = 0.05

_FUES_FAST_METHODS = frozenset({
    "FUES",
    "FUES_V0DEV",
    "FUES_V0_1DEV",
    "FUES_V0_2DEV",
})


def make_keeper_ops(callables, grids, stage):
    """Build keeper dcsn_mover (FUES fast path or generic EGM_UE)."""
    from ..solve import read_scheme_method

    ue_method = read_scheme_method(stage, "upper_envelope")
    if str(ue_method).upper() in _FUES_FAST_METHODS:
        return _make_keeper_fast(callables, grids, stage)
    return _make_keeper_generic(callables, grids, stage)


def _make_keeper_fast(callables, grids, stage):
    """FUES path: one parallel kernel, ``FUES_jit`` (no ``EGM_UE``).

    When ``return_grids`` is True, EGM diagnostics are not collected
    (``cntn_data`` is None); use the generic path for full diagnostics.
    """
    keeper = callables["keeper_cons"]
    g_keep_h = callables["tenure"]["transitions"]["keep_h"]
    sett = stage.settings
    b = float(sett["b"])
    grid_max_A = float(sett["a_max"])
    m_bar = float(sett["m_bar"])
    d_c_u = keeper["d_c_u"]
    d_h_u = keeper["d_h_u"]
    fns = keeper["keeper_egm_fns"]

    inv_euler_fn = fns["inv_euler"]
    bellman_rhs_fn = fns["bellman_rhs"]
    egm_step = make_egm_1d(
        inv_euler_fn,
        bellman_rhs_fn,
        fns["cntn_to_dcsn"],
        fns["concavity"],
    )

    lb_fues = int(sett.get("fues_lb", 10))
    k_include_inter = bool(int(sett.get("keeper_include_intersections", 1)))
    k_no_double = bool(int(sett.get("keeper_no_double_jumps", 1)))
    k_single_inter = bool(int(sett.get("keeper_single_intersection", 0)))
    k_disable_jumps = bool(int(sett.get("keeper_disable_jump_checks", 0)))
    fues_eps_d = float(sett.get("fues_eps_d", _FUES_EPS_D))
    fues_eps_sep = float(sett.get("fues_eps_sep", _FUES_EPS_SEP))
    fues_eps_fwd_back = float(sett.get("fues_eps_fwd_back", _FUES_EPS_FWD_BACK))
    fues_parallel_guard = float(sett.get("fues_parallel_guard", _FUES_PAR_GUARD))
    extrap = bool(int(sett.get("extrap_policy", 1)))
    clamp_fac = float(sett.get("clamp_max_factor", 2.0))

    @njit(parallel=True)
    def _keeper_fues_kernel(
        dV_a,
        V,
        dV_h,
        asset_grid_A,
        h_keep,
        b_in,
        m_bar_in,
        LB,
        grid_max_A_in,
        incl_inter,
        no_dbl,
        single_inter,
        disable_jmp,
        eps_d,
        eps_sep,
        eps_fwd_back,
        parallel_guard,
        extrap_flag,
        clamp_fac_in,
        Akeeper,
        Ckeeper,
        Vkeeper,
        dVw_keep,
        phi_keep,
    ):
        n_z = dV_a.shape[0]
        n_a = dV_a.shape[1]
        n_h = dV_a.shape[2]

        for flat in prange(n_z * n_h):
            iz = flat // n_h
            ih = flat % n_h
            hk = h_keep[ih]

            dv_slice = dV_a[iz, :, ih]
            v_slice = V[iz, :, ih]

            n_pad = max(2, n_a // 10)
            egrid = np.ones(n_pad + n_a)
            vf = np.ones(n_pad + n_a)
            c_raw = np.ones(n_pad + n_a)

            c0 = inv_euler_fn(dv_slice[0], hk)
            C_arr = np.linspace(1e-08, max(1e-08, c0 - 1e-10), n_pad)
            for k in range(n_pad):
                vf[k] = bellman_rhs_fn(C_arr[k], v_slice[0], hk)
                egrid[k] = C_arr[k] + b_in
                c_raw[k] = C_arr[k]

            zwork = np.zeros(n_a)
            c_hat, v_hat, x_hat, _ = egm_step(
                dv_slice, zwork, v_slice, asset_grid_A, hk
            )
            for k in range(n_a):
                egrid[n_pad + k] = x_hat[k]
                vf[n_pad + k] = v_hat[k]
                c_raw[n_pad + k] = c_hat[k]

            ac_arr = np.concatenate((np.full(n_pad, b_in), asset_grid_A))

            uid = uniqueEG(egrid, vf)
            eg_u = egrid[uid]
            vf_u = vf[uid]
            c_u = c_raw[uid]
            ac_u = ac_arr[uid]

            sidx = np.argsort(eg_u)
            eg_s = eg_u[sidx]
            vf_s = vf_u[sidx]
            c_s = c_u[sidx]
            ac_s = ac_u[sidx]
            del_a = np.zeros(len(eg_s))

            eg_ref, vf_ref, c_ref, ac_ref, _ = FUES_jit(
                eg_s,
                vf_s,
                c_s,
                ac_s,
                del_a,
                m_bar_in,
                LB,
                False,               # endog_mbar
                0.0,                 # padding_mbar
                incl_inter,
                no_dbl,
                single_inter,
                disable_jmp,
                eps_d,
                eps_sep,
                eps_fwd_back,
                parallel_guard,
            )

            a_interp = interp_as(eg_ref, ac_ref, asset_grid_A, extrap_flag)
            c_interp = interp_as(eg_ref, c_ref, asset_grid_A, extrap_flag)
            v_interp = interp_as(eg_ref, vf_ref, asset_grid_A, extrap_flag)

            a_clamped = clamp_policy(a_interp, b_in, grid_max_A_in * clamp_fac_in)
            c_clamped = clamp_policy(c_interp, 1e-10, 1e10)
            v_clamped = clamp_value(v_interp)

            for ia in range(n_a):
                Akeeper[iz, ia, ih] = a_clamped[ia]
                Ckeeper[iz, ia, ih] = c_clamped[ia]
                Vkeeper[iz, ia, ih] = v_clamped[ia]
                dVw_keep[iz, ia, ih] = d_c_u(c_clamped[ia], hk)

            dv_h_slice = dV_h[iz, :, ih]
            for ia in range(n_a):
                edvh = interp_as_scalar(
                    asset_grid_A, dv_h_slice, Akeeper[iz, ia, ih],
                    extrap=extrap_flag)
                phi_keep[iz, ia, ih] = d_h_u(c_clamped[ia], hk) + edvh

    def dcsn_mover(vlu_cntn, grids):
        dV_a = vlu_cntn["d_aV"]
        dV_h = vlu_cntn["d_hV"]
        V = vlu_cntn["V"]
        asset_grid_A = grids["a"]
        asset_grid_H = grids["h"]
        n_z = len(grids["z"])
        n_a = len(asset_grid_A)
        n_h = len(asset_grid_H)
        h_keep = np.array([g_keep_h(h) for h in asset_grid_H], dtype=np.float64)

        Akeeper = np.empty((n_z, n_a, n_h))
        Ckeeper = np.empty((n_z, n_a, n_h))
        Vkeeper = np.empty((n_z, n_a, n_h))
        dVw_keep = np.empty((n_z, n_a, n_h))
        phi_keep = np.empty((n_z, n_a, n_h))

        _keeper_fues_kernel(
            dV_a,
            V,
            dV_h,
            asset_grid_A,
            h_keep,
            b,
            m_bar,
            lb_fues,
            grid_max_A,
            k_include_inter,
            k_no_double,
            k_single_inter,
            k_disable_jumps,
            fues_eps_d,
            fues_eps_sep,
            fues_eps_fwd_back,
            fues_parallel_guard,
            extrap,
            clamp_fac,
            Akeeper,
            Ckeeper,
            Vkeeper,
            dVw_keep,
            phi_keep,
        )

        # Fast path: no per-slice EGM diagnostics (use generic path if needed).
        cntn_data = None

        return Akeeper, Ckeeper, Vkeeper, dVw_keep, phi_keep, cntn_data

    return dcsn_mover


def _make_keeper_generic(callables, grids, stage):
    """Build keeper dcsn_mover (EGM_UE dispatch: DCEGM, RFC, CONSAV, FUES, …).

    Parameters
    ----------
    callables : dict
        Full per-period callables from ``make_callables``; uses
        ``keeper_cons`` and ``tenure.transitions.keep_h``.
    grids : dict
    stage : dolo.compiler.model.SymbolicModel
        Dolo+ stage; economics from ``stage.calibration``, grid bounds
        from ``stage.settings``.

    Returns
    -------
    dcsn_mover : callable
        ``(vlu_cntn, grids)``
        -> ``(Akeeper, Ckeeper, Vkeeper, dVw_keep, phi_keep, cntn_data)``
        where cntn_data is None or dict per solution_scheme.md.
    """
    keeper = callables["keeper_cons"]
    g_keep_h = callables["tenure"]["transitions"]["keep_h"]
    cal = stage.calibration
    sett = stage.settings
    b = float(sett["b"])
    grid_max_A = float(sett["a_max"])
    m_bar = float(sett["m_bar"])
    fues_lb = int(sett.get("fues_lb", 10))
    return_grids = cal.get("return_grids", False)
    d_c_u = keeper["d_c_u"]
    d_h_u = keeper["d_h_u"]
    fns = keeper["keeper_egm_fns"]

    from ..solve import read_scheme_method
    ue_method = read_scheme_method(stage, "upper_envelope")
    extrap = bool(int(sett.get("extrap_policy", 1)))
    clamp_fac = float(sett.get("clamp_max_factor", 2.0))

    def _d_c_u_arr(c_arr, hk_val=0.0):
        """Vectorised marginal utility for EGM_UE (accepts arrays).
        hk_val passed for CD compatibility; separable ignores it."""
        return np.array([d_c_u(c, hk_val) for c in c_arr])

    inv_euler_fn = fns["inv_euler"]
    bellman_rhs_fn = fns["bellman_rhs"]
    _egm_step = make_egm_1d(
        inv_euler_fn,
        bellman_rhs_fn,
        fns["cntn_to_dcsn"],
        fns["concavity"],
    )

    @njit(parallel=True)
    def _keeper_egm_slice(dv_slice, v_slice, asset_grid_A, hk, n_a):
        """JIT-compiled EGM for one (z, h) slice: constrained + unconstrained.

        Returns raw endogenous grid, value, consumption, and assets
        (pre–upper-envelope, unsorted, with duplicates).
        """
        egrid = np.ones(n_a * 2)
        vf = np.ones(n_a * 2)
        c_raw = np.ones(n_a * 2)

        # Constrained region
        c0 = inv_euler_fn(dv_slice[0], hk)
        C_arr = np.linspace(1e-08, max(1e-08, c0 - 1e-10), n_a)
        for k in prange(n_a):
            vf[k] = bellman_rhs_fn(C_arr[k], v_slice[0], hk)
            egrid[k] = C_arr[k] + b
            c_raw[k] = C_arr[k]

        # Unconstrained via make_egm_1d
        c_hat, v_hat, x_hat, _ = _egm_step(
            dv_slice, np.zeros(n_a), v_slice, asset_grid_A, hk)
        for k in prange(n_a):
            egrid[n_a + k] = x_hat[k]
            vf[n_a + k] = v_hat[k]
            c_raw[n_a + k] = c_hat[k]

        ac_arr = np.concatenate((np.full(n_a, b), asset_grid_A))

        # De-duplicate
        uid = uniqueEG(egrid, vf)
        return egrid[uid], vf[uid], c_raw[uid], ac_arr[uid]

    def dcsn_mover(vlu_cntn, grids):
        """EGM (jitted) + upper-envelope dispatch per (z, h) slice.

        EGM inversion is JIT-compiled (_keeper_egm_slice).
        Upper-envelope refinement uses EGM_UE, which dispatches to
        FUES, MSS, DCEGM, RFC, etc. based on stage.methods.

        Returns
        -------
        Akeeper, Ckeeper, Vkeeper, dVw_keep, phi_keep, cntn_data
        """
        dV_a = vlu_cntn["d_aV"]
        dV_h = vlu_cntn["d_hV"]
        V = vlu_cntn["V"]
        asset_grid_A = grids["a"]
        asset_grid_H = grids["h"]
        n_z = len(grids["z"])
        n_a = len(asset_grid_A)
        n_h = len(asset_grid_H)
        h_keep = np.array([g_keep_h(h) for h in asset_grid_H])

        Akeeper = np.empty((n_z, n_a, n_h))
        Ckeeper = np.empty((n_z, n_a, n_h))
        Vkeeper = np.empty((n_z, n_a, n_h))
        dVw_keep = np.empty((n_z, n_a, n_h))
        phi_keep = np.empty((n_z, n_a, n_h))

        cntn_c = {} if return_grids else None
        cntn_m_endog = {} if return_grids else None

        for iz in range(n_z):
            for ih in range(n_h):
                hk = h_keep[ih]

                # --- EGM (jitted): raw endogenous grid ---
                eg_u, vf_u, c_u, ac_u = _keeper_egm_slice(
                    dV_a[iz, :, ih], V[iz, :, ih],
                    asset_grid_A, hk, n_a)

                # --- Upper envelope (EGM_UE dispatch) ---
                refined, _, _ = EGM_UE(
                    eg_u, vf_u, None,
                    c_u, ac_u,
                    asset_grid_A,
                    _d_c_u_arr,
                    None,  # u_func only needed by CONSAV engine
                    ue_method=ue_method,
                    m_bar=m_bar, lb=fues_lb,
                )

                eg_ref = refined["x_dcsn_ref"]
                vf_ref = refined["v_dcsn_ref"]
                c_ref = refined["kappa_ref"]
                a_ref = refined["x_cntn_ref"]

                if cntn_c is not None:
                    cntn_c[(iz, ih)] = c_ref.copy()
                    cntn_m_endog[(iz, ih)] = eg_ref.copy()

                # --- Interpolate to asset grid ---
                a_interp = interp_as(eg_ref, a_ref, asset_grid_A, extrap=extrap)
                c_interp = interp_as(eg_ref, c_ref, asset_grid_A, extrap=extrap)
                v_interp = interp_as(eg_ref, vf_ref, asset_grid_A, extrap=extrap)

                a_clamped = clamp_policy(a_interp, b, grid_max_A * clamp_fac)
                c_clamped = clamp_policy(c_interp, 1e-10, 1e10)

                Akeeper[iz, :, ih] = a_clamped
                Ckeeper[iz, :, ih] = c_clamped
                Vkeeper[iz, :, ih] = clamp_value(v_interp)

                for ia in range(n_a):
                    dVw_keep[iz, ia, ih] = d_c_u(c_clamped[ia], hk)

                dv_h_slice = dV_h[iz, :, ih]
                for ia in range(n_a):
                    edvh = interp_as_scalar(
                        asset_grid_A, dv_h_slice, a_clamped[ia],
                        extrap=extrap)
                    phi_keep[iz, ia, ih] = d_h_u(c_clamped[ia], hk) + edvh

        cntn_data = None
        if cntn_c is not None:
            cntn_data = {"c": cntn_c, "m_endog": cntn_m_endog}

        return Akeeper, Ckeeper, Vkeeper, dVw_keep, phi_keep, cntn_data

    return dcsn_mover


# ------------------------------------------------------------------
# Forward (simulation) operator
# ------------------------------------------------------------------


def make_keeper_forward(C_keep_t, callables, grids, stage):
    """StageForward for keeper_cons (pure simulation, no Euler).

    Composes: arvl_to_dcsn (identity) -> policy (C interp)
    -> dcsn_to_cntn (budget constraint).

    ``callables`` is the full per-period dict (reserved for parity with
    ops makers); interpolation uses ``grids['UGgrid_all']``.
    """
    from kikku.asva.simulate import StageForward
    from .simulate import _eval_keeper_c

    _a_grid = np.asarray(grids["a"], dtype=np.float64)
    _h_grid = np.asarray(grids["h"], dtype=np.float64)
    sett = stage.settings
    b = float(sett["b"])
    gA = float(sett["a_max"])
    gH = float(sett["h_max"])
    _extrap = bool(int(sett.get("extrap_policy", 1)))

    def arvl_to_dcsn(particles, shocks):
        return particles

    def policy(particles):
        w = particles["w_keep"]
        h = particles["h_keep"]
        z_idx = particles["z_idx"]
        N = len(w)
        c = np.empty(N)
        for i in range(N):
            c[i] = _eval_keeper_c(w[i], h[i], int(z_idx[i]),
                                  C_keep_t, _a_grid, _h_grid)
        return {"c": c}

    def dcsn_to_cntn(particles, controls, shocks):
        w = particles["w_keep"]
        h_keep = particles["h_keep"]
        c = controls["c"]
        z_idx = particles["z_idx"]
        a_nxt = np.clip(w - c, b, gA)
        h_nxt = np.clip(h_keep, b, gH)

        out = {"a_nxt": a_nxt, "h_nxt": h_nxt,
               "z_idx": z_idx.copy()}
        idx = particles.get("_idx")
        if idx is not None:
            out["_idx"] = idx.copy()
        return out

    return StageForward(
        arvl_to_dcsn=arvl_to_dcsn, policy=policy,
        dcsn_to_cntn=dcsn_to_cntn,
        shock_draw_arvl=None, shock_draw_cntn=None,
    )
