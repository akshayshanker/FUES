"""Functions to implement fast upper-envelope scantime

Author: Akshay Shanker, University of Sydney, akshay.shanker@me.com.

"""


import numpy as np
from numba import njit


@njit
def FUES(e_grid, vf, c, a_prime, m_bar = 2):

    vf = np.take(vf, np.argsort(e_grid))
    c = np.take(c, np.argsort(e_grid))
    a_prime = np.take(a_prime, np.argsort(e_grid))

    e_grid = np.sort(e_grid)

    upper_env_maxes = np.copy(vf)
    e_grid = e_grid[np.where(~np.isnan(upper_env_maxes))]
    c = c[np.where(~np.isnan(upper_env_maxes))]
    a_prime = a_prime[np.where(~np.isnan(upper_env_maxes))]
    upper_env_maxes = upper_env_maxes[np.where(~np.isnan(upper_env_maxes))]

    e_grid, vf_clean, c, a_prime, dela = _scan(
        e_grid, upper_env_maxes, c, a_prime, m_bar)

    return e_grid[np.where(~np.isnan(vf_clean))], vf_clean[np.where(~np.isnan(
        vf_clean))], c[np.where(~np.isnan(vf_clean))], dela[np.where(~np.isnan(vf_clean))]


@njit
def _scan(e_grid, vf, c, a_prime, m_bar):

    # leading value that is checked is j+1
    # leading value to be checked is i+2
    turn = False
    g_minus_2 = 0
    dela = np.zeros(len(vf))

    vf_full = np.copy(vf)

    for i in range(len(e_grid) - 2):
        if i <= 1:
            j = i
        else:
            g_minus_1 = (vf_full[j] - vf_full[j - 1]) / \
                (e_grid[j] - e_grid[j - 1])
            g_1 = (vf_full[i + 1] - vf[j]) / (e_grid[i + 1] - e_grid[j])

            if np.isnan(vf_full[i + 1]):
                pass
            else:
                if g_1 <= g_minus_1 and np.abs(
                        (a_prime[i + 1] - a_prime[j]) / (e_grid[i + 1] - e_grid[j])) > m_bar:
                    vf[i + 1] = np.nan
                elif turn and g_1 < g_minus_2:
                    vf[i + 1] = np.nan
                else:
                    dela[i + 1] = np.abs((a_prime[i + 1] -
                                          a_prime[j]) / (e_grid[i + 1] - e_grid[j]))
                    g_minus_2 = g_minus_1
                    j = i + 1
            if g_1 > g_minus_1:
                turn = True
            else:
                turn = False

    return e_grid, vf, c, a_prime, dela
