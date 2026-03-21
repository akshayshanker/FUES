"""Interpolate branch policies onto common (z,a,h) grid.

These functions bridge the leaf stage outputs (on their
own grids) to the common state grid needed by the
branching stage.
"""

import numpy as np
from numba import njit
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
from dcsmm.fues.helpers.math_funcs import interp_as_scalar
from kikku.asva.numerics import clamp_scalar


@njit
def keeper_to_state_grid(
    t, Akeeper, Ckeeper, Vkeeper,
    ELambdaHnxt, ELambdaH_HD_nxt,
    X_all, asset_grid_A, asset_grid_H, z_vals,
    UGgrid_all, UGgrid_HD, use_hd_lambda,
    y_func, du_h,
    R, delta, b, grid_max_A,
):
    """Interpolate keeper onto (z,a,h) + compute Phi_t.

    Parameters
    ----------
    Akeeper, Ckeeper, Vkeeper : ndarray (n_z, n_a, n_h)
        Keeper arrival policies on the asset grid.
    ELambdaHnxt : ndarray (n_z, n_a, n_h)
        E_z[d_h V] continuation.
    ELambdaH_HD_nxt : ndarray
        HD variant (or dummy).

    Returns
    -------
    v, c, a_nxt, h_nxt, phi : ndarray (n_z, n_a, n_h)
    """
    shape = (len(z_vals), len(asset_grid_A),
             len(asset_grid_H))
    v = np.empty(shape)
    c = np.empty(shape)
    a_nxt = np.empty(shape)
    h_nxt = np.empty(shape)
    phi = np.empty(shape)

    for state in range(len(X_all)):
        a = asset_grid_A[X_all[state][1]]
        h = asset_grid_H[X_all[state][2]]
        i_a = X_all[state][1]
        i_h = X_all[state][2]
        i_z = int(X_all[state][0])
        z = z_vals[i_z]

        wealth = R * a + y_func(t, z)
        v_val = interp_as_scalar(
            asset_grid_A, Vkeeper[i_z, :, i_h], wealth)
        c_val = interp_as_scalar(
            asset_grid_A, Ckeeper[i_z, :, i_h], wealth)
        a_val = interp_as_scalar(
            asset_grid_A, Akeeper[i_z, :, i_h], wealth)
        h_val = (1 - delta) * h

        v_val = clamp_scalar(v_val, -1e10, 1e10, -1e10)
        c_val = clamp_scalar(c_val, 1e-10, 1e10, 1e-10)
        a_val = clamp_scalar(a_val, b, grid_max_A * 2, b)

        v[i_z, i_a, i_h] = v_val
        c[i_z, i_a, i_h] = c_val
        a_nxt[i_z, i_a, i_h] = a_val
        h_nxt[i_z, i_a, i_h] = h_val

        point = np.array([a_val, h_val])
        if use_hd_lambda:
            phi_val = du_h(h_val) + eval_linear(
                UGgrid_HD, ELambdaH_HD_nxt[i_z],
                point, xto.LINEAR)
        else:
            phi_val = du_h(h_val) + eval_linear(
                UGgrid_all, ELambdaHnxt[i_z],
                point, xto.LINEAR)
        phi[i_z, i_a, i_h] = phi_val

    return v, c, a_nxt, h_nxt, phi


@njit
def adjuster_to_state_grid(
    t, Aadj, Cadj, Hadj, V_prime,
    X_all, asset_grid_A, asset_grid_H,
    asset_grid_WE, z_vals,
    UGgrid_all,
    y_func, u,
    R, R_H, delta, tau, b, chi,
    grid_max_A, grid_max_H,
    c_from_budget=1,
):
    """Interpolate adjuster onto (z,a,h) + Bellman eval.

    Parameters
    ----------
    Aadj, Cadj, Hadj : ndarray (n_z, n_w)
        Adjuster policies on wealth grid.
    V_prime : ndarray (n_z, n_a, n_h)
        Continuation value (for Bellman).

    Returns
    -------
    v, c, a_nxt, h_nxt : ndarray (n_z, n_a, n_h)
    """
    shape = (len(z_vals), len(asset_grid_A),
             len(asset_grid_H))
    v_out = np.empty(shape)
    c_out = np.empty(shape)
    a_out = np.empty(shape)
    h_out = np.empty(shape)
    tau_adj = 1.0 + tau
    beta = 0.0  # will be set below

    for state in range(len(X_all)):
        a = asset_grid_A[X_all[state][1]]
        h = asset_grid_H[X_all[state][2]]
        i_a = X_all[state][1]
        i_h = X_all[state][2]
        i_z = int(X_all[state][0])
        z = z_vals[i_z]

        wealth = (R * a + R_H * h * (1 - delta)
                  + y_func(t, z))
        a_val = interp_as_scalar(
            asset_grid_WE, Aadj[i_z, :], wealth)
        h_val = interp_as_scalar(
            asset_grid_WE, Hadj[i_z, :], wealth)

        a_val = clamp_scalar(
            a_val, b, grid_max_A * 2, b)
        h_val = clamp_scalar(
            h_val, b, grid_max_H * 2, b)

        if c_from_budget == 1:
            c_val = wealth - a_val - h_val * tau_adj
        else:
            c_val = interp_as_scalar(
                asset_grid_WE, Cadj[i_z, :], wealth)
        c_val = clamp_scalar(
            c_val, 1e-10, 1e10, 1e-10)

        points = np.array([a_val, h_val])
        # Note: beta is captured from the calling scope
        # via the operator factory closure. Here we
        # hardcode the Bellman: v = u(c,h') + beta*V(a',h')
        # The operator factory passes beta when calling.
        v_val = (u(c_val, h_val, chi)
                 + eval_linear(
                     UGgrid_all, V_prime[i_z],
                     points, xto.LINEAR))

        v_out[i_z, i_a, i_h] = v_val
        c_out[i_z, i_a, i_h] = c_val
        a_out[i_z, i_a, i_h] = a_val
        h_out[i_z, i_a, i_h] = h_val

    return v_out, c_out, a_out, h_out
