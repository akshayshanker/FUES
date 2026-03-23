"""Shock conditioning helpers for tenure arrival values."""

import numpy as np
from numba import njit


def make_conditioners(Pi):
    """Build E_z conditioning operators from a Markov transition matrix."""

    @njit
    def condition_V(Vcurr, d_aV_curr, d_hV_curr):
        n_z, n_a, n_h = Vcurr.shape
        new_V = np.zeros((n_z, n_a, n_h))
        new_d_aV = np.zeros((n_z, n_a, n_h))
        new_d_hV = np.zeros((n_z, n_a, n_h))

        for i_a in range(n_a):
            for i_h in range(n_h):
                v_slice = np.ascontiguousarray(Vcurr[:, i_a, i_h])
                a_slice = np.ascontiguousarray(d_aV_curr[:, i_a, i_h])
                h_slice = np.ascontiguousarray(d_hV_curr[:, i_a, i_h])
                new_V[:, i_a, i_h] = np.dot(Pi, v_slice)
                new_d_aV[:, i_a, i_h] = np.dot(Pi, a_slice)
                new_d_hV[:, i_a, i_h] = np.dot(Pi, h_slice)

        return new_V, new_d_aV, new_d_hV

    @njit
    def condition_V_HD(d_h_hd_curr):
        n_z, n_a_hd, n_h_hd = d_h_hd_curr.shape
        new_hd = np.zeros((n_z, n_a_hd, n_h_hd))

        for i_a in range(n_a_hd):
            for i_h in range(n_h_hd):
                hd_slice = np.ascontiguousarray(d_h_hd_curr[:, i_a, i_h])
                new_hd[:, i_a, i_h] = np.dot(Pi, hd_slice)

        return new_hd

    return condition_V, condition_V_HD
