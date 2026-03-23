"""Adjuster consumption stage: NEGM (nested optimisation).

Golden-section maximisation over h_choice, nesting the keeper's
consumption solution for c. Same YAML spec as EGM adjuster —
only the methodization differs.

dcsn_mover signature matches adjuster_egm: (vlu_cntn, grids) -> 6-tuple.
Keeper output injected via inject_keeper() between waves.
"""

import numpy as np
from numba import njit
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
from consav import golden_section_search


def make_adjuster_negm_ops(callables, grids, settings):
    """Build NEGM adjuster with closure-captured keeper dependency.

    Parameters
    ----------
    callables : dict
        adjuster_cons stage callables.
    grids : dict
    settings : dict

    Returns
    -------
    dict
        ``{'dcsn_mover': callable, 'inject_keeper': callable}``
    """
    b = settings["b"]
    grid_max_A = settings["grid_max_A"]
    grid_max_H = settings["grid_max_H"]
    UGgrid_all = grids["UGgrid_all"]

    u_fn = callables["u"]
    du_c_fn = callables["d_c_u"]
    bellman_obj = callables["bellman_obj"]
    housing_cost = callables["housing_cost"]
    fac_housing = float(housing_cost(1.0))

    # Mutable containers for keeper injection
    _keeper_c = [None]
    _keeper_a = [None]

    def inject_keeper(C_keep, A_keep):
        """Inject keeper output before adjuster runs."""
        _keeper_c[0] = C_keep
        _keeper_a[0] = A_keep

    # ================================================================
    # Objective functions (closures over cp params)
    # ================================================================

    @njit
    def _obj_interior(h_prime, wealth, i_z, V_cntn, keeper_c_iz):
        """Interior objective: max_h u(c, h) + beta*V(a', h).

        c from keeper interpolation at (w_residual, h_prime).
        """
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
        """Boundary objective: a' = b (borrowing constraint)."""
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

    # ================================================================
    # Golden-section multi-section maximiser
    # ================================================================

    @njit
    def _gs_max(obj, obj_neg, lo, hi, args, n_sections=1, xtol=1e-6):
        """Golden-section max over [lo, hi], divided into n_sections."""
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

    # ================================================================
    # dcsn_mover
    # ================================================================

    def dcsn_mover(vlu_cntn, grids):
        """NEGM adjuster: golden-section over h, keeper for c.

        Returns
        -------
        a_nxt, c, h_choice, V, d_wV, cntn_data (None)
        """
        V_cntn = vlu_cntn["V"]
        we_grid = grids["we"]
        z_vals = grids["z"]
        n_z = len(z_vals)
        n_w = len(we_grid)

        keeper_c = _keeper_c[0]
        keeper_a = _keeper_a[0]
        if keeper_c is None:
            raise RuntimeError("inject_keeper() must be called before NEGM adjuster")

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

                # Interior solution
                h_star, v_star = _gs_max(
                    _obj_interior, _obj_interior_neg,
                    h_lo, h_hi, args, n_sections=1, xtol=1e-6)

                # Boundary solution (a' = b)
                h_bound, v_bound = _gs_max(
                    _obj_boundary, _obj_boundary_neg,
                    h_lo, h_hi, args, n_sections=1, xtol=1e-6)

                # Take best
                if v_bound > v_star:
                    h_opt = h_bound
                    w_res = wealth - housing_cost(h_opt)
                    a_nxt[iz, iw] = b
                    c[iz, iw] = max(1e-10, w_res - b)
                    V[iz, iw] = v_bound
                else:
                    h_opt = h_star
                    w_res = wealth - housing_cost(h_opt)
                    # Get a' from keeper
                    pt = np.array([w_res, h_opt])
                    a_val = eval_linear(
                        UGgrid_all, keeper_a[iz], pt, xto.LINEAR)
                    a_nxt[iz, iw] = min(max(a_val, b), grid_max_A * 2)
                    c[iz, iw] = max(1e-10, w_res - a_nxt[iz, iw])
                    V[iz, iw] = v_star

                h_choice[iz, iw] = min(max(h_opt, b), grid_max_H * 2)

        # MarginalBellman: d_{w}V = d_{c}u(c)
        d_wV = np.empty((n_z, n_w))
        for iz in range(n_z):
            for iw in range(n_w):
                d_wV[iz, iw] = du_c_fn(c[iz, iw])

        return a_nxt, c, h_choice, V, d_wV, None

    return {"dcsn_mover": dcsn_mover, "inject_keeper": inject_keeper}
