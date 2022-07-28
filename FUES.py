"""Functions to implement fast upper-envelope scan 

Author: Akshay Shanker, University of Sydney, akshay.shanker@me.com.

"""


import numpy as np
from numba import njit

@njit
def FUES(e_grid, vf, c, a_prime, m_bar = 2):
    """
    FUES function 

    Parameters
    ----------
    e_grid: 1D array 
            unrefined endogenous grid
    vf: 1d array
            value points at endogenous grid
    c: 1D array 
            policy 1 points at endogenous grid
    a_prime: 1D array
            policy 2 points at endogenous grid
    m_bar: float
            jump detection threshold

    returns
    ------- 
    e_grid_clean: 1D array
                    refined endogenous grid
    vf_clean: 1D array
    c_clean: 1D array
    a_prime_clean: 1D array
    dela: 1D array
            a_prime(i+1) - a_prime(i) values along
            refined grid 
    """

    # sort policy and vf by e grid order 
    vf = np.take(vf, np.argsort(e_grid))
    c = np.take(c, np.argsort(e_grid))
    a_prime = np.take(a_prime, np.argsort(e_grid))
    e_grid = np.sort(e_grid)

    # remove any NaN values in vf 
    e_grid = e_grid[np.where(~np.isnan(vf))]
    c = c[np.where(~np.isnan(vf))]
    a_prime = a_prime[np.where(~np.isnan(vf))]

    # scan attaches NaN value to vf at all sub-optimal points 
    e_grid_clean, vf_with_nans, c_clean, a_prime_clean, dela\
    = _scan(e_grid, vf, c, a_prime, m_bar)

    return e_grid_clean[np.where(~np.isnan(vf_with_nans))],\
            vf_with_nans[np.where(~np.isnan(vf_with_nans))],\
            c_clean[np.where(~np.isnan(vf_with_nans))],\
            a_prime_clean[np.where(~np.isnan(a_prime_clean))],\
            dela[np.where(~np.isnan(vf_with_nans))]


@njit
def _scan(e_grid, vf, c, a_prime, m_bar):

    # leading value that is checked is j
    # leading value to be checked is i+1

   

    # create copy of value function that
    # remains as an unrefined set of points 
    vf_full = np.copy(vf)

    # empty array to stor policy function change
    dela = np.zeros(len(vf))

    for i in range(len(e_grid) - 2):

        # inital two points on clean grid 
        if i <= 1:
            j = i
        else:
            g_minus_1 = (vf_full[j] - vf_full[j - 1]) / \
                (e_grid[j] - e_grid[j - 1])

            g_1 = (vf_full[i + 1] - vf[j]) / (e_grid[i + 1] - e_grid[j])

            if np.isnan(vf_full[i + 1]):
                pass
            else:
                # right turn is made and jump registered 
                # remove point
                if g_1 <= g_minus_1 and np.abs(
                        (a_prime[i + 1] - a_prime[j]) / (e_grid[i + 1] - e_grid[j])) > m_bar:
                    vf[i + 1] = np.nan

                # left turn is made or right turn with no jump 
                # keep point 
                else:
                    dela[i + 1] = np.abs((a_prime[i + 1] -
                                          a_prime[j]) / (e_grid[i + 1] - e_grid[j]))

                    # iterate clean grid point by one 
                    j = i + 1

    return e_grid, vf, c, a_prime, dela
