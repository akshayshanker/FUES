"""Functions to implement fast upper-envelope scan by
Dobrescu and Shanker (2023)

Author: Akshay Shanker, University of Sydney, akshay.shanker@me.com.


Credits
-------

Line segment intersection code from:

https://web.archive.org/web/20111108065352/https://www.cs.mun.ca/%7Erod
/2500/notes/numpy-arrays/numpy-arrays.html


Notes
-----

EGM grid should not have duplicate values. Run uniqueEG before passing
endogenous grid through FUES other will throw up division by zero errors. 


"""


import numpy as np
from numba import njit
import copy


@njit
def uniqueEG(egrid, vf):
    egrid_rounded = np.round_(egrid, 10)
    unique_values = np.unique(egrid_rounded)
    max_vf_indices = np.full_like(egrid, -1)  # Initialize with -1

    for value in unique_values:
        if np.isnan(value):
            continue
        else:
            indices = np.where(egrid_rounded == value)[0]
            max_index = indices[np.argmax(vf[indices])]
            max_vf_indices[max_index] = max_index

    mask = (max_vf_indices != -1)
    return mask


@njit
def append_push(x_array, m):
    """ Delete first value of array,
        pushes back index of all undeleted
        values and appends m to final index"""
    # Shift values in-place to avoid unnecessary reallocation
    x_array[:-1] = x_array[1:]
    x_array[-1] = m
    return x_array


@njit
def back_scan_gradients(m_array, a_prime, vf_full, e_grid, j, q):
    """ Compute gradients of value correspondence points
        and policy points with respect to all m values and policy
        points in m_array """
    
    num_elements = len(m_array)
    gradients_m_vf = np.empty(num_elements)
    gradients_m_a = np.empty(num_elements)

    for m in range(num_elements):
        m_int = int(m_array[m])
        delta_e_grid = e_grid[j] - e_grid[m_int]  # Cache repeated subtraction
        gradients_m_vf[m] = (vf_full[j] - vf_full[m_int]) / delta_e_grid
        gradients_m_a[m] = np.abs((a_prime[q] - a_prime[m_int]) / delta_e_grid)

    return gradients_m_vf, gradients_m_a

@njit
def fwd_scan_gradients(a_prime, vf_full, e_grid, j, q, LB):
    """ Computes gradients of value correspondence points
        and  policy points with respect to values and policy
        points for next LB points in grid

        See Figure 5, left panel in DS (2023)"""

    gradients_f_vf = np.empty(LB)
    gradients_f_a = np.empty(LB)

    for f in range(LB):
        delta_e_grid = e_grid[q] - e_grid[q + 1 + f]  # Cache repeated subtraction
        gradients_f_vf[f] = (vf_full[q] - vf_full[q + 1 + f]) / delta_e_grid
        gradients_f_a[f] = np.abs((a_prime[j] - a_prime[q + 1 + f]) / delta_e_grid)

    return gradients_f_vf, gradients_f_a


@njit
def perp(a):
    """ Finds perpendicilar line to 1D line

    Parameters
    ----------
    a: 1D array
        points (b, 1/m)

    Returns
    -------
    b: 1D array
        b[0] = -1/m, b[1]= b

    """
    b = np.empty(np.shape(a))
    b[0] = -a[1]
    b[1] = a[0]

    return b


@njit
def seg_intersect(a1, a2, b1, b2):
    """Intersection of two 1D line segments."""
    
    # Compute deltas
    da_x = a2[0] - a1[0]
    da_y = a2[1] - a1[1]
    db_x = b2[0] - b1[0]
    db_y = b2[1] - b1[1]
    dp_x = a1[0] - b1[0]
    dp_y = a1[1] - b1[1]
    
    # Compute perpendicular vector to da (perpendicular to line segment a1-a2)
    dap_x = -da_y
    dap_y = da_x

    # Compute dot products
    denom = dap_x * db_x + dap_y * db_y  # dot(dap, db)
    num = dap_x * dp_x + dap_y * dp_y    # dot(dap, dp)
    
    # Avoid division by zero (parallel lines case)
    if denom == 0:
        return np.array([np.nan, np.nan])

    # Intersection point calculation
    t = num / denom
    intersect_x = t * db_x + b1[0]
    intersect_y = t * db_y + b1[1]

    return np.array([intersect_x, intersect_y])

@njit
def FUES(e_grid, vf, policy_1, policy_2,del_a, b=1e-10, m_bar=2, LB=4, endog_mbar = False):
    """
    FUES function returns refined EGM grid, value function and
    policy function

    Parameters
    ----------
    e_grid: 1D array
            unrefined endogenous grid
    vf: 1D array
            value correspondence points at endogenous grid
    policy_1: 1D array
            policy 1 points at endogenous grid
    policy_2: 1D array
            policy 2 points at endogenous grid
    del_a: 1D array
            derivative of policy function 1
    b: float64
        lower bound for the endogenous grid

    m_bar: float64
            fixed jump detection threshold
    LB: int
         length of bwd and fwd scan search
    edog_mbar: Bool
         flag if m_bar is enodgenous using 
         policy function derivative 

    Returns
    -------
    e_grid_clean: 1D array
                    refined endogenous grid
    vf_clean: 1D array
                value function on refined grid
    policy_1_clean: 1D array
                policy 1 on refined grid
    policy_2_clean: 1D array
                    policy 2 on refined grid
    del_a_clean: 1D array
            gradient of policy 2 on refined grid

    Notes
    -----
    Policy 2 is used to determine jumps in policy.

    FUES attaches NaN values to vf array
    where points are sub-optimal and not to be retained.

    The code below checks to see if multiple EGM points equal the lower
    bound of the endogenous grid. If multiple EGM points equal the lower bound,
    the one yielding the highest value is retained. So far in applications 
    in DS (2023),the only multiple EGM values occur on the 
    lower bound (see Application 2 for DS, 2023).

    If endogenous M_bar is used, then policy_a is assumed to 
    be convex conditional on all future discrete choices. 

    Todo
    ----
    Incorporate explicit check for multiple
    equal EGM grid values (other than the lb).

    Incorporate full functionality to attach crossing points.

    For the forward and backward scans we are still
    using the exogenously specified M_bar. This should be 
    replaced by the enodgenously determined maximumum
    gradients of the policy function. 

    """

    # Sort policy and vf by e_grid order
    sort_indices = np.argsort(e_grid)

    # Sort e_grid and other arrays using the sorted indices
    e_grid = e_grid[sort_indices]
    vf = vf[sort_indices]
    policy_1 = policy_1[sort_indices]
    policy_2 = policy_2[sort_indices]
    del_a = del_a[sort_indices]


    # Scan attaches NaN to vf at all sub-optimal points
    e_grid_clean, vf_with_nans = \
        _scan(e_grid, vf,policy_1, del_a, m_bar, LB, endog_mbar=endog_mbar)

    non_nan_indices = np.where(~np.isnan(vf_with_nans))
    
    return (e_grid_clean[non_nan_indices],
        vf[non_nan_indices],
        policy_1[non_nan_indices],
        policy_2[non_nan_indices],
        del_a[non_nan_indices])
        

@njit
def _scan(e_grid, vf, a_prime,del_a, m_bar, LB, fwd_scan_do=True, endog_mbar= True):
    """" Implements the scan for FUES"""

    # leading index for optimal values j
    # leading index for value to be `checked' is i+1

    # create copy of value function
    # this copy remains intact as the unrefined set of points
    vf_full = np.copy(vf)

    # empty array to store policy function gradient
    #dela = np.zeros(len(vf))

    # array of previously sub-optimal indices to be used in backward scan
    m_array = np.zeros(LB)

    # FUES scan
    for i in range(len(e_grid) - 2):

        # inital two points are optimal (assumption)
        if i <= 1:
            j = i
            k = j-1
            #previous_opt_is_intersect = False
            #k_minus_1 = np.copy(np.array([k]))[0] - 1

        else:
            # value function gradient betweeen previous two optimal points
            g_j_minus_1 = (vf_full[j] - vf_full[k]) / \
                (e_grid[j] - e_grid[k])

            # gradient with leading index to be checked
            denom_egrid = e_grid[i + 1] - e_grid[j]
            g_1 = (vf_full[i + 1] - vf_full[j]) / (denom_egrid)

            # Absolute gradients of policy function at current index 
            # and at testing point
            M_L = np.abs(del_a[j])
            M_U = np.abs(del_a[i+1])
            M_max = max(M_L, M_U) + 0.001

            # policy gradient with leading index to be checked
            d_p_prime = a_prime[i + 1] - a_prime[j]
            g_tilde_a = np.abs((d_p_prime)\
                               / (denom_egrid))

            # Set detection threshold to m_bar if fixed m_bar used 
            if endog_mbar == False:
                M_max = m_bar 

            # if right turn is made and jump registered
            # remove point or perform forward scan

            if g_1 < g_j_minus_1 and  g_tilde_a > M_max:
                keep_i_1_point = False

                if fwd_scan_do:
                    gradients_f_vf, gradients_f_a\
                        = fwd_scan_gradients(a_prime, vf_full,
                                             e_grid, j, i + 1, LB)

                    # get index of closest next point with same
                    # discrete choice as point j
                    if len(np.where(gradients_f_a < m_bar)[0]) > 0:
                        m_index_fwd = np.where(gradients_f_a < m_bar)[0][0]
                        g_m_vf = gradients_f_vf[m_index_fwd]
                        g_m_a = gradients_f_a[m_index_fwd]

                        if g_1 > g_m_vf:
                            keep_i_1_point = True
                        else:
                            pass
                    else:
                        pass

                    if not keep_i_1_point:
                        vf[i + 1] = np.nan
                        m_array = append_push(m_array, i + 1)
                    else:
                        previous_opt_is_intersect = True
                        k = j
                        j = i + 1

            # If value falls, remove points
            elif vf_full[i + 1] - vf_full[j] < 0:
                vf[i + 1] = np.nan
                # append index array of previously deleted points
                m_array = append_push(m_array, i + 1)

            # assume value is monotone in policy and delete if not
            # satisfied
            elif g_1 < g_j_minus_1 and d_p_prime < 0:
                vf[i + 1] = np.nan
                m_array = append_push(m_array, i + 1)

            # if left turn is made or right turn with no jump, then
            # keep point provisionally and conduct backward scan
            else:
                # backward scan
                # compute value gradients (from i+1) and
                # policy gradients (from j)
                # wrt to LB previously deleted values
                gradients_m_vf, gradients_m_a \
                    = back_scan_gradients(m_array,
                                          a_prime, vf_full, e_grid, j, i + 1)
                keep_j_point = True

                # index m of last point that is deleted and does not jump from
                # leading point where left turn is made.
                # this is the closest previous point on the same
                # discrete choice specific
                # policy as the leading value we have just jumped to
                if len(np.where(gradients_m_a < m_bar)[0]) > 0:
                    m_index_bws = np.where(gradients_m_a < m_bar)[0][-1]

                    # gradient of vf and policy to the m'th point
                    g_m_vf = gradients_m_vf[m_index_bws]
                    g_m_a = gradients_m_a[m_index_bws]

                else:
                    m_index_bws = 0
                    keep_j_point = True

                # index of m'th point on the e_grid
                m_ind = int(m_array[m_index_bws])

                # if the gradient joining the leading point i+1 (we have just
                # jumped to) and the point m(the last point on the same
                # choice specific policy) is shallower than the
                # gradient joining the i+1 and j, then delete j'th point

                if g_1 > g_j_minus_1 and g_1 >= g_m_vf and g_tilde_a > m_bar:
                    keep_j_point = False

                if not keep_j_point:
                    pj = np.array([e_grid[j], vf_full[j]])
                    pi1 = np.array([e_grid[i + 1], vf_full[i + 1]])
                    pk = np.array([e_grid[k], vf_full[k]])
                    pm = np.array([e_grid[m_ind], vf_full[m_ind]])
                    intrsect = seg_intersect(pj, pk, pi1, pm)

                    vf[j] = np.nan
                    vf_full[j] = intrsect[1]
                    e_grid[j] = intrsect[0]
                    previous_opt_is_intersect = True
                    j = i + 1

                else:

                    previous_opt_is_intersect = False
                    if g_1 > g_j_minus_1:
                        previous_opt_is_intersect = True

                    k = j
                    j = i + 1

    return e_grid, vf