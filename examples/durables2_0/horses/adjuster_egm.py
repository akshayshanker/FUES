"""Adjuster consumption stage: EGM + FUES from 2D post-state.

The dual InvEuler constrains (a_nxt, h_choice) to a 1D curve.
EGM inverts from this 1D subspace onto a 1D endogenous wealth
grid m[>]. FUES resolves the upper envelope. Interpolation maps
to the decision-perch Cartesian grid.

YAML sub-equations implemented:
  InvEuler:  solve_{c[>], h_choice[>]}{two coupled FOCs}
  cntn_to_dcsn_transition:  m[>] = a_nxt + c[>] + (1+tau)*h_choice[>]
  MarginalBellman:  d_{w}V = d_{c}u(c)
"""

import numpy as np
from numba import njit, prange
from dcsmm.fues import FUES_jit
from dcsmm.fues.fues_v0dev import uniqueEG
from dcsmm.fues.fues_v0_2dev import (
    EPS_D as _FUES_EPS_D,
    EPS_SEP as _FUES_EPS_SEP,
    PARALLEL_GUARD as _FUES_PAR_GUARD,
)
from dcsmm.fues.helpers.math_funcs import (
    interp_as_scalar,
    interp_as,
    find_roots_piecewise_linear,
    correct_jumps1d_arr,
)
from kikku.asva.numerics import clamp_value, clamp_policy
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
from consav import golden_section_search


def _make_negm_adjuster(callables, grids, stage):
    """NEGM path: nested optimisation (golden-section over h, keeper for c)."""
    adj = callables["adjuster_cons"]
    cal = stage.calibration
    sett = stage.settings
    b = float(cal["b"])
    grid_max_A = float(sett["a_max"])
    grid_max_H = float(sett["h_max"])
    UGgrid_all = grids["UGgrid_all"]

    u_fn = adj["u"]
    du_c_fn = adj["d_c_u"]
    bellman_obj = adj["bellman_obj"]
    housing_cost = adj["housing_cost"]
    fac_housing = float(housing_cost(1.0))

    _keeper_c = [None]
    _keeper_a = [None]

    def inject_keeper(C_keep, A_keep):
        _keeper_c[0] = C_keep
        _keeper_a[0] = A_keep

    @njit
    def _obj_interior(h_prime, wealth, i_z, V_cntn, keeper_c_iz):
        w_residual = wealth - housing_cost(h_prime)

        if w_residual <= 0 or h_prime <= 0 or h_prime > grid_max_H:
            return -np.inf
        if w_residual > grid_max_A:
            return -np.inf

        pt = np.array([w_residual, h_prime])
        c_keeper = eval_linear(UGgrid_all, keeper_c_iz, pt, xto.LINEAR)

        if c_keeper <= 0 or c_keeper > w_residual:
            return -np.inf

        a_prime = max(b, w_residual - c_keeper)
        if a_prime > grid_max_A:
            return -np.inf

        pt_v = np.array([a_prime, h_prime])
        Ev = eval_linear(UGgrid_all, V_cntn[i_z], pt_v, xto.LINEAR)

        return bellman_obj(u_fn(c_keeper, h_prime), Ev)

    @njit
    def _obj_interior_neg(h_prime, wealth, i_z, V_cntn, keeper_c_iz):
        return -_obj_interior(h_prime, wealth, i_z, V_cntn, keeper_c_iz)

    @njit
    def _obj_boundary(h_prime, wealth, i_z, V_cntn, keeper_c_iz):
        w_residual = wealth - housing_cost(h_prime)

        if w_residual <= 0 or h_prime < b or h_prime > grid_max_H:
            return -np.inf

        c_val = w_residual - b
        if c_val <= 0:
            return -np.inf

        pt = np.array([b, h_prime])
        Ev = eval_linear(UGgrid_all, V_cntn[i_z], pt, xto.LINEAR)

        return bellman_obj(u_fn(c_val, h_prime), Ev)

    @njit
    def _obj_boundary_neg(h_prime, wealth, i_z, V_cntn, keeper_c_iz):
        return -_obj_boundary(h_prime, wealth, i_z, V_cntn, keeper_c_iz)

    @njit
    def _gs_max(obj, obj_neg, lo, hi, args, n_sections=1, xtol=1e-6):
        if hi - lo < xtol:
            return lo, obj(lo, *args)

        x_opts = np.zeros(n_sections)
        vals = np.full(n_sections, -1e250)

        section_w = (hi - lo) / n_sections
        for i in range(n_sections):
            s_lo = lo + i * section_w
            s_hi = lo + (i + 1) * section_w
            if s_hi - s_lo < xtol:
                continue
            x_opt = golden_section_search.optimizer(
                obj_neg, s_lo, s_hi, args=args, tol=xtol)
            x_opts[i] = x_opt
            vals[i] = obj(x_opt, *args)

        best_idx = 0
        best_val = vals[0]
        for i in range(1, n_sections):
            if vals[i] > best_val:
                best_val = vals[i]
                best_idx = i

        if best_val < -1e200:
            mid = (lo + hi) / 2.0
            return mid, obj(mid, *args)

        return x_opts[best_idx], best_val

    def dcsn_mover(vlu_cntn, grids):
        V_cntn = vlu_cntn["V"]
        we_grid = grids["we"]
        z_vals = grids["z"]
        n_z = len(z_vals)
        n_w = len(we_grid)

        keeper_c = _keeper_c[0]
        keeper_a = _keeper_a[0]
        if keeper_c is None:
            raise RuntimeError(
                "inject_keeper() must be called before NEGM adjuster")

        a_nxt = np.empty((n_z, n_w))
        c = np.empty((n_z, n_w))
        h_choice = np.empty((n_z, n_w))
        V = np.empty((n_z, n_w))

        for iw in range(n_w):
            wealth = we_grid[iw]
            for iz in range(n_z):
                h_lo = b
                h_hi = min(wealth / fac_housing + b, grid_max_H)

                args = (wealth, iz, V_cntn, keeper_c[iz])

                h_star, v_star = _gs_max(
                    _obj_interior, _obj_interior_neg,
                    h_lo, h_hi, args, n_sections=1, xtol=1e-6)

                h_bound, v_bound = _gs_max(
                    _obj_boundary, _obj_boundary_neg,
                    h_lo, h_hi, args, n_sections=1, xtol=1e-6)

                if v_bound > v_star:
                    h_opt = h_bound
                    w_res = wealth - housing_cost(h_opt)
                    a_nxt[iz, iw] = b
                    c[iz, iw] = max(1e-10, w_res - b)
                    V[iz, iw] = v_bound
                else:
                    h_opt = h_star
                    w_res = wealth - housing_cost(h_opt)
                    pt = np.array([w_res, h_opt])
                    a_val = eval_linear(
                        UGgrid_all, keeper_a[iz], pt, xto.LINEAR)
                    a_nxt[iz, iw] = min(max(a_val, b), grid_max_A * 2)
                    c[iz, iw] = max(1e-10, w_res - a_nxt[iz, iw])
                    V[iz, iw] = v_star

                h_choice[iz, iw] = min(max(h_opt, b), grid_max_H * 2)

        d_wV = np.empty((n_z, n_w))
        for iz in range(n_z):
            for iw in range(n_w):
                d_wV[iz, iw] = du_c_fn(c[iz, iw], h_choice[iz, iw])

        return a_nxt, c, h_choice, V, d_wV, None

    return {"dcsn_mover": dcsn_mover, "inject_keeper": inject_keeper}


def _make_egm_adjuster(callables, grids, stage):
    """EGM + FUES path for adjuster ``dcsn_mover``.

    Parameters
    ----------
    callables : dict
        Full per-period callables; uses ``adjuster_cons``.
    grids : dict
    stage : dolo.compiler.model.SymbolicModel
        Dolo+ stage.

    Returns
    -------
    dict
        ``{'dcsn_mover': callable}``
    """
    adj = callables["adjuster_cons"]
    cal = stage.calibration
    sett = stage.settings
    b = float(cal["b"])
    m_bar = float(sett["m_bar"])
    grid_max_A = float(sett["a_max"])
    grid_max_H = float(sett["h_max"])
    return_grids = cal.get("return_grids", False)
    root_eps = float(sett["root_eps"])
    egm_n = int(sett["egm_n"])

    z_vals = grids["z"]
    a_grid = grids["a"]
    h_choice_grid = grids["h_choice"]

    u_fn = adj["u"]
    du_c_fn = adj["d_c_u"]
    du_c_inv_fn = adj["d_c_u_inv"]
    du_h_fn = adj["d_h_u"]
    bellman_discount = adj["bellman_discount"]
    housing_cost = adj["housing_cost"]
    invEuler_foc_h_residual = adj["invEuler_foc_h_residual"]
    invEuler_foc_h_rhs = adj["invEuler_foc_h_rhs"]
    fac_housing = float(housing_cost(1.0))

    # ================================================================
    # InvEuler: housing Euler residual root-finder
    #
    # For fixed h_choice, finds a_nxt roots where the housing FOC
    # holds: (1+tau)*d_c u(c) - d_h u(h_choice) - beta*d_{h_nxt}V[>] = 0
    # ================================================================

    @njit
    def inv_euler_h_residual(h_choice, dv_a_cntn, dv_h_cntn):
        """Find a_nxt roots of housing Euler for fixed h_choice.

        Returns
        -------
        a_nxt_cntn : ndarray
            Poststate a_nxt values on the constraint curve.
        m_cntn_raw : ndarray
            Endogenous wealth m[>] at each root.
        """
        n_a = len(a_grid)
        grid_range = a_grid[-1] - a_grid[0]
        avg_spacing = grid_range / (n_a - 1)

        # root_eps from settings: 0 = full grid (exact), >0 = coarse scan
        if root_eps > avg_spacing:
            step = int(root_eps / avg_spacing)
            n_samples = (n_a - 1) // step + 1
            sample_grid = np.empty(n_samples)
            resid = np.empty(n_samples)
            for k in range(n_samples):
                idx = min(k * step, n_a - 1)
                sample_grid[k] = a_grid[idx]
                resid[k] = invEuler_foc_h_residual(
                    dv_a_cntn[idx], dv_h_cntn[idx], h_choice
                )
            a_nxt_cntn, n_roots = find_roots_piecewise_linear(
                resid, sample_grid, egm_n, 0.0
            )
        else:
            # Full grid: exact piecewise-linear roots (root_eps=0 recommended)
            resid = np.empty(n_a)
            for j in range(n_a):
                resid[j] = invEuler_foc_h_residual(
                    dv_a_cntn[j], dv_h_cntn[j], h_choice
                )
            a_nxt_cntn, n_roots = find_roots_piecewise_linear(
                resid, a_grid, egm_n, 0.0
            )

        # Recover c and endogenous wealth at each root.
        # c from the housing FOC: c = du_c_inv((dv_h(a') + du_h(h')) / (1+tau)).
        # At interior roots both FOCs hold so this equals the asset Euler;
        # at the borrowing constraint only the housing FOC holds.
        # Matches old/durables/durables.py root_H_UPRIME_func_fast.
        du_h_val = du_h_fn(h_choice)
        m_cntn_raw = np.zeros(egm_n)
        for j in range(n_roots):
            a_p = a_nxt_cntn[j]
            if a_p > 0.0:
                dv_h_val = interp_as_scalar(a_grid, dv_h_cntn, a_p)
                c = du_c_inv_fn(invEuler_foc_h_rhs(du_h_val, dv_h_val))
                m_cntn_raw[j] = c + a_p + housing_cost(h_choice)

        # Add borrowing constraint point
        has_b = False
        for j in range(n_roots):
            if a_nxt_cntn[j] == b:
                has_b = True
                break
        if not has_b:
            dv_h_val_b = interp_as_scalar(a_grid, dv_h_cntn, b)
            c_at_b = du_c_inv_fn(invEuler_foc_h_rhs(du_h_val, dv_h_val_b))
            a_nxt_cntn[-1] = b
            m_cntn_raw[-1] = c_at_b + b + housing_cost(h_choice)

        return a_nxt_cntn, m_cntn_raw

    # ================================================================
    # cntn_to_dcsn_egm: full EGM step (InvEuler + endogenous grid)
    # ================================================================

    @njit
    def cntn_to_dcsn_egm(dv_a_cntn, dv_h_cntn, v_cntn):
        """Unrefined EGM: loop over h_choice grid, root-find, build m[>].

        Returns raw (pre-FUES) arrays on the continuation grid.
        """
        n_z = len(z_vals)
        n_he = len(h_choice_grid)

        m_cntn_raw = np.zeros((n_z, n_he, egm_n))
        v_cntn_raw = np.zeros((n_z, n_he, egm_n))
        a_nxt_cntn = np.zeros((n_z, n_he, egm_n))
        h_choice_cntn = np.zeros((n_z, n_he, egm_n))

        for ihp in range(n_he):
            h_choice = h_choice_grid[ihp]
            h_cost = housing_cost(h_choice)

            for iz in range(n_z):
                dv_a_1d = dv_a_cntn[iz, :, ihp]
                dv_h_1d = dv_h_cntn[iz, :, ihp]
                v_1d = v_cntn[iz, :, ihp]

                a_roots, m_roots = inv_euler_h_residual(
                    h_choice, dv_a_1d, dv_h_1d
                )

                m_cntn_raw[iz, ihp] = m_roots
                a_nxt_cntn[iz, ihp] = a_roots

                for i in range(len(a_roots)):
                    a_p = a_roots[i]
                    if a_p > 0.0:
                        c_val = m_roots[i] - h_cost - a_p
                        v_at_ap = interp_as_scalar(a_grid, v_1d, a_p)
                        v_cntn_raw[iz, ihp, i] = (
                            u_fn(c_val, h_choice) + bellman_discount(v_at_ap)
                        )
                        h_choice_cntn[iz, ihp, i] = h_choice

        return m_cntn_raw, v_cntn_raw, a_nxt_cntn, h_choice_cntn

    # ================================================================
    # fues_refine: upper envelope + interpolation to m_grid
    # ================================================================

    def fues_refine(m_cntn_raw, v_cntn_raw, a_nxt_cntn, h_choice_cntn,
                    m_grid):
        """Ravel 2D EGM points, FUES envelope, interpolate to m_grid.

        Parameters
        ----------
        m_cntn_raw, v_cntn_raw, a_nxt_cntn, h_choice_cntn : (n_z, n_he, egm_n)
            Raw EGM outputs from cntn_to_dcsn_egm.
        m_grid : (n_w,)
            Decision-perch wealth grid for interpolation.

        Returns
        -------
        a_nxt, c, h_choice, V : (n_z, n_w)
        refined : dict
            Per-iz FUES-refined arrays.
        """
        n_z = len(z_vals)
        n_w = len(m_grid)

        a_nxt = np.empty((n_z, n_w))
        h_choice_out = np.empty((n_z, n_w))
        V = np.empty((n_z, n_w))
        c = np.empty((n_z, n_w))
        refined = {}

        for iz in range(n_z):
            a_flat = a_nxt_cntn[iz].ravel()
            mask = a_flat > 0.0

            v_pts = v_cntn_raw[iz].ravel()[mask]
            h_pts = h_choice_cntn[iz].ravel()[mask]
            a_pts = a_flat[mask]
            m_pts = m_cntn_raw[iz].ravel()[mask]
            c_pts = m_pts - h_pts * fac_housing - a_pts

            uid = uniqueEG(m_pts, v_pts)
            m_pts = m_pts[uid]
            v_pts = v_pts[uid]
            c_pts = c_pts[uid]
            a_pts = a_pts[uid]
            h_pts = h_pts[uid]

            sidx = np.argsort(m_pts)
            m_s = m_pts[sidx]
            v_s = v_pts[sidx]
            a_s = a_pts[sidx]
            h_s = h_pts[sidx]
            c_s = c_pts[sidx]

            m_clean, v_clean, h_clean, a_clean, _ = FUES_jit(
                m_s, v_s, h_s, a_s, c_s,
                m_bar, 10,
                False, 0.0,
                False, True, True, True,
                _FUES_EPS_D, _FUES_EPS_SEP, 0.05,
                _FUES_PAR_GUARD,
            )

            refined[iz] = {
                'm_endog': m_clean.copy(),
                'vf': v_clean.copy(),
                'h_nxt_eval': h_clean.copy(),
                'a_nxt_eval': a_clean.copy(),
            }

            c_clean = m_clean - h_clean * fac_housing - a_clean

            a_nxt[iz] = clamp_policy(
                interp_as(m_clean, a_clean, m_grid, extrap=True),
                b, grid_max_A * 2)
            h_choice_out[iz] = clamp_policy(
                interp_as(m_clean, h_clean, m_grid, extrap=True),
                b, grid_max_H * 2)
            V[iz] = clamp_value(
                interp_as(m_clean, v_clean, m_grid, extrap=True))
            c[iz] = clamp_policy(
                interp_as(m_clean, c_clean, m_grid, extrap=True),
                1e-10, 1e10)

            c[iz], V[iz], h_choice_out[iz], a_nxt[iz] = correct_jumps1d_arr(
                c[iz], m_grid, m_bar,
                V[iz], h_choice_out[iz], a_nxt[iz],
            )

        return a_nxt, c, h_choice_out, V, refined

    # ================================================================
    # dcsn_mover: full backward step (B operator)
    # ================================================================

    def dcsn_mover(vlu_cntn, grids):
        """cntn_to_dcsn_mover: EGM + FUES + MarginalBellman.

        Parameters
        ----------
        vlu_cntn : dict
            ``{'V', 'd_aV', 'd_hV'}`` on continuation grid.
        grids : dict

        Returns
        -------
        a_nxt, c, h_choice, V, d_wV, cntn_data
            cntn_data is None or dict per solution_scheme.md.
        """
        m_grid = grids["we"]

        m_cntn_raw, v_cntn_raw, a_nxt_cntn, h_choice_cntn = cntn_to_dcsn_egm(
            vlu_cntn["d_aV"], vlu_cntn["d_hV"], vlu_cntn["V"]
        )

        a_nxt, c, h_choice, V, refined = fues_refine(
            m_cntn_raw, v_cntn_raw, a_nxt_cntn, h_choice_cntn, m_grid,
        )

        n_z, n_w = c.shape
        d_wV = np.empty((n_z, n_w))
        for iz in range(n_z):
            for iw in range(n_w):
                d_wV[iz, iw] = du_c_fn(c[iz, iw])

        cntn_data = None
        if return_grids:
            m_arr = np.asarray(m_cntn_raw)
            a_arr = np.asarray(a_nxt_cntn)
            h_arr = np.asarray(h_choice_cntn)
            c_raw = m_arr - h_arr * fac_housing - a_arr
            cntn_data = {
                'c': c_raw,
                'm_endog': m_arr,
                'a_nxt_eval': a_arr,
                'h_nxt_eval': h_arr,
                '_refined': refined,
            }

        return a_nxt, c, h_choice, V, d_wV, cntn_data

    return {"dcsn_mover": dcsn_mover}


def make_adjuster_ops(callables, grids, stage):
    """Build adjuster ``dcsn_mover``; reads ``upper_envelope`` method from stage."""
    from ..solve import read_scheme_method

    ue_method = read_scheme_method(stage, 'upper_envelope')
    if ue_method == 'NEGM':
        return _make_negm_adjuster(callables, grids, stage)
    return _make_egm_adjuster(callables, grids, stage)


# ------------------------------------------------------------------
# Forward (simulation) operator
# ------------------------------------------------------------------

def make_adjuster_forward(C_adj_t, H_adj_t, callables, grids, stage):
    """StageForward for adjuster_cons (pure simulation, no Euler).

    Composes: arvl_to_dcsn (identity) -> policy (C, H interp)
    -> dcsn_to_cntn (budget constraint).
    """
    from kikku.asva.simulate import StageForward
    from dcsmm.fues.helpers.math_funcs import interp_as_scalar

    we_grid = grids["we"]
    fac_housing = float(callables["adjuster_cons"]["housing_cost"](1.0))
    sett = stage.settings
    b = float(sett["b"])
    gA = float(sett["a_max"])
    gH = float(sett["h_max"])

    def arvl_to_dcsn(particles, shocks):
        return particles

    def policy(particles):
        w = particles['w_adj']
        z_idx = particles['z_idx']
        N = len(w)
        c = np.empty(N)
        h_choice = np.empty(N)
        for i in range(N):
            ci = interp_as_scalar(we_grid, C_adj_t[int(z_idx[i])], w[i])
            hi = interp_as_scalar(we_grid, H_adj_t[int(z_idx[i])], w[i])
            c[i] = max(ci, 1e-10)
            h_choice[i] = max(hi, 1e-10)
        return {'c': c, 'h_choice': h_choice}

    def dcsn_to_cntn(particles, controls, shocks):
        w = particles['w_adj']
        c = controls['c']
        hc = controls['h_choice']
        z_idx = particles['z_idx']
        N = len(w)
        a_nxt = np.clip(w - c - fac_housing * hc, b, gA)
        h_nxt = np.clip(hc, b, gH)

        idx = particles.get('_idx')
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
