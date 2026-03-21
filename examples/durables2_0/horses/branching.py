"""Branching max: tenure stage dcsn_mover.

Pure max/argmax over keep vs adjust branches + marginal
value computation. No interpolation, no continuation eval.
Receives pre-interpolated values on the common (z,a,h) grid.
"""

import numpy as np
from numba import njit


@njit
def branching_max(v_keep, c_keep, a_keep, h_keep,
                  phi_keep,
                  v_adj, c_adj, a_adj, h_adj,
                  X_all, du_c,
                  beta, R, R_H, delta):
    """Max over keep/adjust + marginal values.

    Parameters
    ----------
    v_keep, c_keep, a_keep, h_keep : ndarray (n_z,n_a,n_h)
        Keeper arrival values on the state grid.
    phi_keep : ndarray (n_z,n_a,n_h)
        Keeper housing marginal Phi_t.
    v_adj, c_adj, a_adj, h_adj : ndarray (n_z,n_a,n_h)
        Adjuster arrival values on the state grid.
    X_all : ndarray
        State space indices.
    du_c : callable
        Marginal utility of consumption.
    beta, R, R_H, delta : float
        Model parameters.

    Returns
    -------
    V, A, H, C, D, dV_a, dV_h : ndarray (n_z,n_a,n_h)
    """
    shape = v_keep.shape
    V = np.empty(shape)
    C = np.empty(shape)
    A = np.empty(shape)
    H = np.empty(shape)
    D = np.empty(shape)
    dV_a = np.empty(shape)
    dV_h = np.empty(shape)

    for state in range(len(X_all)):
        i_z = int(X_all[state][0])
        i_a = X_all[state][1]
        i_h = X_all[state][2]

        vk = v_keep[i_z, i_a, i_h]
        va = v_adj[i_z, i_a, i_h]

        if va >= vk:
            d = 1
        else:
            d = 0

        D[i_z, i_a, i_h] = d
        ck = c_keep[i_z, i_a, i_h]
        ca = c_adj[i_z, i_a, i_h]
        V[i_z, i_a, i_h] = d * va + (1 - d) * vk
        C[i_z, i_a, i_h] = d * ca + (1 - d) * ck
        A[i_z, i_a, i_h] = (
            d * a_adj[i_z, i_a, i_h]
            + (1 - d) * a_keep[i_z, i_a, i_h])
        H[i_z, i_a, i_h] = (
            d * h_adj[i_z, i_a, i_h]
            + (1 - d) * h_keep[i_z, i_a, i_h])

        dV_a[i_z, i_a, i_h] = (
            beta * R * (d * du_c(ca)
                        + (1 - d) * du_c(ck)))
        dV_h[i_z, i_a, i_h] = (
            beta * R_H * (1 - delta)
            * (d * du_c(ca)
               + (1 - d) * phi_keep[i_z, i_a, i_h]))

    return V, A, H, C, D, dV_a, dV_h
