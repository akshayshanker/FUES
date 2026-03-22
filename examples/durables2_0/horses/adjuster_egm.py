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
from numba import njit
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


def make_adjuster_ops(cp, callables):
    """Build age-invariant adjuster dcsn_mover.

    Parameters
    ----------
    cp : ConsumerProblem
    callables : dict
        Structural callables (u, du_c, du_c_inv, du_h).

    Returns
    -------
    dict
        ``{'dcsn_mover': callable}``
    """
    b = cp.b
    beta = cp.beta
    tau = cp.tau
    chi = cp.chi
    m_bar = cp.m_bar
    grid_max_A = cp.grid_max_A
    grid_max_H = cp.grid_max_H
    return_grids = cp.return_grids
    root_eps = cp.root_eps
    egm_n = cp.EGM_N

    z_vals = cp.z_vals
    a_grid = cp.asset_grid_A
    h_choice_grid = cp.asset_grid_HE

    u_fn = callables["u"]
    du_c_fn = callables["du_c"]
    du_c_inv_fn = callables["du_c_inv"]
    du_h_fn = callables["du_h"]

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
        du_h_val = du_h_fn(h_choice)

        n_a = len(a_grid)
        grid_range = a_grid[-1] - a_grid[0]
        avg_spacing = grid_range / (n_a - 1)

        if root_eps > avg_spacing:
            step = int(root_eps / avg_spacing)
            n_samples = (n_a - 1) // step + 1
            sample_grid = np.empty(n_samples)
            resid = np.empty(n_samples)
            for k in range(n_samples):
                idx = min(k * step, n_a - 1)
                sample_grid[k] = a_grid[idx]
                resid[k] = (
                    dv_a_cntn[idx] * (1.0 + tau)
                    - dv_h_cntn[idx]
                    - du_h_val
                )
            a_nxt_cntn, n_roots = find_roots_piecewise_linear(
                resid, sample_grid, egm_n, 0.0
            )
        else:
            resid = dv_a_cntn * (1.0 + tau) - dv_h_cntn - du_h_val
            a_nxt_cntn, n_roots = find_roots_piecewise_linear(
                resid, a_grid, egm_n, 0.0
            )

        # Recover c and endogenous wealth at each root
        m_cntn_raw = np.zeros(egm_n)
        for j in range(n_roots):
            a_p = a_nxt_cntn[j]
            if a_p > 0.0:
                dv_h_val = interp_as_scalar(a_grid, dv_h_cntn, a_p)
                c = du_c_inv_fn((dv_h_val + du_h_val) / (1.0 + tau))
                m_cntn_raw[j] = c + a_p + h_choice * (1.0 + tau)

        # Add borrowing constraint point
        has_b = False
        for j in range(n_roots):
            if a_nxt_cntn[j] == b:
                has_b = True
                break
        if not has_b:
            dv_h_val = interp_as_scalar(a_grid, dv_h_cntn, b)
            c_at_b = du_c_inv_fn((dv_h_val + du_h_val) / (1.0 + tau))
            a_nxt_cntn[-1] = b
            m_cntn_raw[-1] = c_at_b + b + h_choice * (1.0 + tau)

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
        tau_adj = 1.0 + tau

        m_cntn_raw = np.zeros((n_z, n_he, egm_n))
        v_cntn_raw = np.zeros((n_z, n_he, egm_n))
        a_nxt_cntn = np.zeros((n_z, n_he, egm_n))
        h_choice_cntn = np.zeros((n_z, n_he, egm_n))

        for ihp in range(n_he):
            h_choice = h_choice_grid[ihp]
            h_cost = h_choice * tau_adj

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
                        v_prime = beta * interp_as_scalar(a_grid, v_1d, a_p)
                        v_cntn_raw[iz, ihp, i] = u_fn(c_val, h_choice, chi) + v_prime
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
        tau_adj = 1.0 + tau

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
            c_pts = m_pts - h_pts * tau_adj - a_pts

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
                m_bar, 5,
                False, 0.0,
                False, True, False, True,
                _FUES_EPS_D, _FUES_EPS_SEP, 0.05,
                _FUES_PAR_GUARD,
            )

            refined[iz] = {
                'm_endog': m_clean.copy(),
                'vf': v_clean.copy(),
                'h_nxt_eval': h_clean.copy(),
                'a_nxt_eval': a_clean.copy(),
            }

            c_clean = m_clean - h_clean * tau_adj - a_clean

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
            tau_adj = 1.0 + tau
            m_arr = np.asarray(m_cntn_raw)
            a_arr = np.asarray(a_nxt_cntn)
            h_arr = np.asarray(h_choice_cntn)
            c_raw = m_arr - h_arr * tau_adj - a_arr
            cntn_data = {
                'c': c_raw,
                'm_endog': m_arr,
                'a_nxt_eval': a_arr,
                'h_nxt_eval': h_arr,
                '_refined': refined,
            }

        return a_nxt, c, h_choice, V, d_wV, cntn_data

    return {"dcsn_mover": dcsn_mover}


# ------------------------------------------------------------------
# Forward (simulation) operator
# ------------------------------------------------------------------

def make_adjuster_forward(C_adj_t, H_adj_t, cp, we_grid,
                          edata_next, grids, callables,
                          euler_panel, t_val):
    """StageForward for adjuster_cons with inline Euler.

    Composes: arvl_to_dcsn (identity) -> policy (C, H interp)
    -> dcsn_to_cntn (budget constraint + Euler side-channel).

    Euler values are written to ``euler_panel[t_val, idx]``
    via the ``_idx`` side-channel, NOT emitted in poststates.
    """
    from kikku.asva.simulate import StageForward
    from dcsmm.fues.helpers.math_funcs import interp_as_scalar
    from ..simulate import adjuster_euler

    b, gA, gH, tau = cp.b, cp.grid_max_A, cp.grid_max_H, cp.tau

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
        a_nxt = np.clip(w - c - (1 + tau) * hc, b, gA)
        h_nxt = np.clip(hc, b, gH)

        idx = particles.get('_idx')
        if edata_next is not None and idx is not None:
            for i in range(N):
                ci, ai, hi = c[i], a_nxt[i], h_nxt[i]
                if ci > 0.1 and ai > 0.1:
                    euler_panel[t_val, int(idx[i])] = adjuster_euler(
                        ci, hi, ai, int(z_idx[i]),
                        edata_next, grids, cp, callables)

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
