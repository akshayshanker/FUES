"""Functions to implement fast upper-envelope scan as described by 
Dobrescu and Shanker (2022)

Author: Akshay Shanker, University of Sydney, akshay.shanker@me.com.


Credits
-------

Line segment intersection code from:

https://web.archive.org/web/20111108065352/https://www.cs.mun.ca/%7Erod
/2500/notes/numpy-arrays/numpy-arrays.html


"""


import numpy as np
from numba import njit
import copy 

@njit 
def append_push(x_array, m):
    """ Deletes first value of array, pushes back index of all undeleted
        values and appends m to final index"""

    for i in range(len(x_array)-1):
        x_array[i] = x_array[i+1]

    x_array[-1] = m
    return x_array

@njit
def back_scan_gradients(m_array, a_prime,vf_full, e_grid,j, q):
    """ Computes gradients of j'ith value function q'th policy 
        with respect to all m'th value and policy respectively 
        in list m_array""" 
   
    gradients_m_vf = np.zeros(len(m_array))
    gradients_m_a  = np.zeros(len(m_array))

    for m in range(len(gradients_m_a)):
        m_int = int(m_array[m])
        gradients_m_vf[m] = (vf_full[j] - vf_full[m_int]) \
                                / (e_grid[j] - e_grid[m_int])
        gradients_m_a[m]  = np.abs((a_prime[q] - a_prime[m_int])\
                                 / (e_grid[q] - e_grid[m_int]))

    return gradients_m_vf, gradients_m_a
                

@njit
def perp(a) :
    """ Finds perpendicilar line to 1D line 

    Parameters
    ----------
    a: 1D array
        points b 1/m

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
def seg_intersect(a1,a2, b1,b2):
    """ Intersection of two 1D line segments

    Parameters
    ----------
    a1: 1D array
         First point of first line seg
    a2: 1D array
         Second point of first line seg
    b1: 1D array
         First point of first line seg
    b2: 1D array
         Second point of first line seg
    
    Returns
    -------
    c: 1D array
        intersection point 

    """
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom)*db + b1

@njit
def FUES(e_grid, vf, c, a_prime, b = 1e-10, m_bar = 2, LB = 10):
    """
    FUES function returns refined EGM grid, value function and 
    policy function 

    Parameters
    ----------
    e_grid: 1D array 
            unrefined endogenous grid
    vf: 1D array
            value correspondence points at endogenous grid
    c: 1D array 
            policy 1 points at endogenous grid
    a_prime: 1D array
            policy 2 points at endogenous grid
    b: float64
        lower bound for the endogenous grid 
    m_bar: float
            jump detection threshold
    LB: int
         length of backward scan search 

    Returns
    ------- 
    e_grid_clean: 1D array
                    refined endogenous grid
    vf_clean: 1D array
                value function on refined grid
    c_clean: 1D array
                policy 1 on refined grid
    a_prime_clean: 1D array
                    policy 2 on refined grid 
    dela: 1D array
            gradient of policy 2 on refined grid 

    Notes
    -----
    Policy 2 is used to determine jumps in policy. 

    The procedure below attaches NaN values to vf array where points are 
    sub-optimal.

    The code below checks to see if multiple EGM points equal the lower
    bound of the endogenous grid. If multiple EGM points equal the lower bound,
    the one yielding the highest value is retained. 

    So far in applications in DS (2022), 
    the only multiple EGM values occur on the lower bound (see Application 2
    for DS, 2022).
     

    Todo
    ----
    Incorporate explicit step to check for multiple
    equal EGM grid values (other than the lb). 

    Incorporate full functionality to attach crossing points. 

    """


    # determine locations where enogenous grid points are 
    # equal the lower bound 
    if len(vf[np.where(e_grid<=b)])>0:
        vf_lb_max =  max(vf[np.where(e_grid<=b)])

        # remove sub-optimal lb EGM points 
        for i in range(len(e_grid)):
            if e_grid[i] <= b and vf[i]<vf_lb_max:
                vf[i] = np.nan

        
    # remove any NaN values from vf array  
    e_grid = e_grid[np.where(~np.isnan(vf))]
    c = c[np.where(~np.isnan(vf))]
    a_prime = a_prime[np.where(~np.isnan(vf))]
    vf = vf[np.where(~np.isnan(vf))]

    # sort policy and vf by e_grid order 
    vf = np.take(vf, np.argsort(e_grid))
    c = np.take(c, np.argsort(e_grid))
    a_prime = np.take(a_prime, np.argsort(e_grid))
    e_grid = np.sort(e_grid)

    # scan attaches NaN to vf at all sub-optimal points 
    e_grid_clean, vf_with_nans,  c_clean, a_prime_clean, dela\
    = _scan(e_grid, vf, c, a_prime, m_bar, LB)

    return e_grid_clean[np.where(~np.isnan(vf_with_nans))],\
            vf[np.where(~np.isnan(vf_with_nans))],\
            c_clean[np.where(~np.isnan(vf_with_nans))],\
            a_prime_clean[np.where(~np.isnan(vf_with_nans))],\
            dela


@njit
def _scan(e_grid, vf, c, a_prime, m_bar, LB):
    """" Implements the scan for FUES"""


    # leading index for optimal values  j
    # leading index for value to be `checked' is i+1

    # create copy of value function
    # this copy remains intact as the unrefined set of points 
    vf_full = np.copy(vf)

    # empty array to store policy function gradient 
    dela = np.zeros(len(vf))

    # array of previously deleted indices to be used in backward scan
    m_array = np.zeros(LB)

    # FUES scan 
    for i in range(len(e_grid) - 2):

        # inital two points are optimal (assumption)
        if i <= 1:
            j = np.copy(np.array([i]))[0]
            k = np.copy(np.array([j-1]))[0]
            previous_opt_is_intersect = False
            k_minus_1 = np.copy(np.array([k]))[0] - 1
        
        else:

            # value function gradient betweeen previous two optimal points 
            g_j_minus_1 = (vf_full[j] - vf_full[k]) / \
                        (e_grid[j] - e_grid[k])

            # gradient with leading index to be checked 
            g_1 = (vf_full[i + 1] - vf_full[j]) / (e_grid[i + 1] - e_grid[j])

            #policy gradient with leading index to be checked 
            g_tilde_a = np.abs((a_prime[i + 1] - a_prime[j])\
                            / (e_grid[i + 1] - e_grid[j]))

            # If right turn is made and jump registered 
            # remove point
            if g_1 < g_j_minus_1 and g_tilde_a > m_bar:
                vf[i + 1] = np.nan

                # append index array of previously deleted points
                m_array = append_push(m_array,i+1)

            # If value falls, remove points 
            elif vf_full[i + 1] - vf_full[j]<0:
                vf[i + 1] = np.nan

                # append index array of previously deleted points
                m_array = append_push(m_array,i+1)

            # assume value is monotone in policy and delete if not 
            # satisfied
            elif g_1 < g_j_minus_1 and a_prime[i + 1] - a_prime[j] < 0:
                vf[i + 1] = np.nan
                m_array = append_push(m_array,i+1)

            # if left turn is made or right turn with no jump, then
            # keep point provisionally and conduct backward scan 
            else:
                
                # backward scan 
                # compute value gradients (i+1) and policy gradients (j)
                # wrt to LB previously deleted values 
                gradients_m_vf, gradients_m_a = back_scan_gradients(m_array,\
                                         a_prime,vf_full,e_grid, j, i+1)

                # index m of last that is deleted that does not jump from 
                # leading point where left turn is made 
                # this is the closest previous point on the same 
                # discrete specific 
                # policy as the leading value we have just jumped to
                if len(np.where(gradients_m_a< m_bar)[0])> 0:
                    m_index_bws = np.where(gradients_m_a< m_bar)[0][-1]

                    # gradient of vf and policy to the m'th point
                    g_m_vf = gradients_m_vf[m_index_bws]
                    g_m_a = gradients_m_a[m_index_bws]
                
                else:
                    m_index_bws = 0
                    g_m_vf = gradients_m_vf[m_index_bws]
                    g_m_a = gradients_m_a[m_index_bws]

                # index of m'th point on the e_grid 
                m_ind = int(m_array[m_index_bws])

                # if the gradient joining the leading point i+1 (we have just
                # jumped to) and the point m(the last point on the same 
                # choice specific policy) is shallower than the 
                # gradient joining the

                if g_1 > g_j_minus_1 and g_1>=g_m_vf:
                    pj = np.copy(np.array([e_grid[j], vf_full[j]]))
                    pi1 = np.copy(np.array([e_grid[i+1], vf_full[i+1]]))
                    pk = np.copy(np.array([e_grid[k], vf_full[k]]))
                    pm =  np.copy(np.array([e_grid[m_ind], vf_full[m_ind]]))
                    intrsect = seg_intersect(pj,pk,pi1, pm)

                    vf[j]  =  np.nan
                    vf_full[j]=  intrsect[1]
                    e_grid[j] = intrsect[0]
                    previous_opt_is_intersect = True
                    j = np.copy(np.array([i]))[0] + 1

                else:

                    previous_opt_is_intersect = False

                    if g_1 > g_j_minus_1:
                       previous_opt_is_intersect = True
                       #print(g_1)
                       #print(g_m_vf)
                       #print(e_grid[i+1])
                       #print(e_grid[j])
                       #print(e_grid[m_ind])
                    k = np.copy(np.array([j]))[0]
                    j = np.copy(np.array([i]))[0] + 1


    return e_grid, vf, c, a_prime, dela
