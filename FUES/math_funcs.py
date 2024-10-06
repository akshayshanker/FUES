import math
from numba import njit, vectorize, prange
import numpy as np
from quantecon.optimize.root_finding import brentq
#from HARK.dcegm import calc_segments, calc_multiline_envelope, calc_cross_points
from HARK.dcegm import calc_nondecreasing_segments, upper_envelope, calc_linear_crossing

@njit
def rootsearch(f,a,b,dx, h_prime,z, Ud_prime_a, Ud_prime_h,t):
    x1 = a; f1 = f(a, h_prime,z, Ud_prime_a, Ud_prime_h,t)
    x2 = a + dx; f2 = f(x2, h_prime,z, Ud_prime_a, Ud_prime_h,t)
    
    while f1*f2 > 0.0:
        if x1 >= b:
            return np.nan,np.nan
        x1 = x2; f1 = f2
        x2 = x1 + dx; f2 = f(x2,h_prime,z, Ud_prime_a, Ud_prime_h,t)
        #print(x2)
    return x1,x2

def bisect(f,x1,x2,switch=0,epsilon=1.0e-9):
    f1 = f(x1)
    if f1 == 0.0:
        return x1
    f2 = f(x2)
    if f2 == 0.0:
        return x2
    if f1*f2 > 0.0:
        print('Root is not bracketed')
        return None
    n = int(math.ceil(math.log(abs(x2 - x1)/epsilon)/math.log(2.0)))
    for i in range(n):
        x3 = 0.5*(x1 + x2); f3 = f(x3)
        if (switch == 1) and (abs(f3) >abs(f1)) and (abs(f3) > abs(f2)):
            return None
        if f3 == 0.0:
            return x3
        if f2*f3 < 0.0:
            x1 = x3
            f1 = f3
        else:
            x2 =x3
            f2 = f3
    return (x1 + x2)/2.0


@njit 
def f(x):
    return x * np.cos(x-4)



@njit
def interp_as(xp, yp, x, extrap=False):
    """Function  interpolates 1D
    with linear extraplolation

    Parameters
    ----------
    xp : 1D array
            points of x values
    yp : 1D array
            points of y values
    x  : 1D array
            points to interpolate

    Returns
    -------
    evals: 1D array
            y values at x

    """

    evals = np.zeros(len(x))
    if extrap and len(xp) > 1:
        for i in range(len(x)):
            if x[i] < xp[0]:
                if (xp[1] - xp[0]) != 0:
                    evals[i] = yp[0] + (x[i] - xp[0]) * (yp[1] - yp[0])\
                        / (xp[1] - xp[0])
                else:
                    evals[i] = yp[0]

            elif x[i] > xp[-1]:
                if (xp[-1] - xp[-2]) != 0:
                    evals[i] = yp[-1] + (x[i] - xp[-1]) * (yp[-1] - yp[-2])\
                        / (xp[-1] - xp[-2])
                else:
                    evals[i] = yp[-1]
            else:
                evals[i] = np.interp(x[i], xp, yp)
    else:
        evals = np.interp(x, xp, yp)
    return evals


def upper_envelope(segments,  calc_crossings=False):
    """

    Cloned HARK line segment upper_envelope function  

    Finds the upper envelope of a list of non-decreasing segments
    Parameters
    ----------
    segments : list of segments. Segments are tuples of arrays, with item[0]
        containing the x coordninates and item[1] the y coordinates of the
        points that confrom the segment item.
    calc_crossings : Bool, optional
        Indicates whether the crossing points at which the "upper" segment
        changes should be computed. The default is True.
    Returns
    -------
    x : np.array of floats
        x coordinates of the points that conform the upper envelope.
    y : np.array of floats
        y coordinates of the points that conform the upper envelope.
    env_inds : np array of ints
        Array of the same length as x and y. It indicates which of the
        provided segments is the "upper" one at every returned (x,y) point.
    """
    n_seg = len(segments)

    # Collect the x points of all segments in an ordered array, removing duplicates
    x = np.unique(np.concatenate([x[0] for x in segments]))

    # Interpolate all segments on every x point, without extrapolating.
    y_cond = np.zeros((n_seg, len(x)))
    for i in range(n_seg):

        if len(segments[i][0]) == 1:
            # If the segment is a single point, we can only know its value
            # at the observed point.
            row = np.repeat(np.nan, len(x))
            ind = np.searchsorted(x, segments[i][0][0])
            row[ind] = segments[i][1][0]
        else:
            # If the segment has more than one point, we can interpolate
            row = np.interp(x,segments[i][0], segments[i][1])
            extrap = np.logical_or(x < segments[i][0][0], x > segments[i][0][-1])
            row[extrap] = np.nan

        y_cond[i, :] = row

    # Take the maximum to get the upper envelope.
    env_inds = np.nanargmax(y_cond, 0)
    y = y_cond[env_inds, range(len(x))]

    # Get crossing points if needed
    if calc_crossings:

        xing_points, xing_lines = calc_cross_points(x, y_cond, env_inds)

        if len(xing_points) > 0:

            # Extract x and y coordinates
            xing_x = np.array([p[0] for p in xing_points])
            xing_y = np.array([p[1] for p in xing_points])

            # To capture the discontinuity, we'll add the successors of xing_x to
            # the grid
            succ = np.nextafter(xing_x, xing_x + 1)

            # Collect points to add to grids
            xtra_x = np.concatenate([xing_x, succ])
            # if there is a crossing, y will be the same on both segments
            xtra_y = np.concatenate([xing_y, xing_y])
            xtra_lines = np.concatenate([xing_lines[:, 0], xing_lines[:, 1]])

            # Insert them
            idx = np.searchsorted(x, xtra_x)
            x = np.insert(x, idx, xtra_x)
            y = np.insert(y, idx, xtra_y)
            env_inds = np.insert(env_inds, idx, xtra_lines)

    return x, y, env_inds


@njit
def calculate_gradient_1d(data, x):
    gradients = np.empty_like(data, dtype=np.float64)
    for i in range(1, len(data)):
        gradients[i] = (data[i] - data[i - 1]) / (x[i] - x[i - 1])
    gradients[0] = gradients[1]  # assuming continuous gradient at the start
    return gradients

@njit
def correct_jumps1d(data, x, gradient_jump_threshold, policy_value_funcs):
    """
    Removes jumps in a 1D array based on gradient jump threshold and applies the same correction to
    policy and value function arrays stored in a dictionary.

    Args:
        data (numpy.ndarray): The input 1D data array.
        x (numpy.ndarray): The 1D array of x values corresponding to `data`.
        gradient_jump_threshold (float): Threshold for detecting jumps in gradient.
        policy_value_funcs (dict): A dictionary of additional 1D arrays for policy, value functions, etc.

    Returns:
        tuple: Corrected 1D data array and the updated policy_value_funcs dictionary.
    """
    corrected_data = np.copy(data)
    gradients = calculate_gradient_1d(data, x)

    # Ensure policy and value arrays are also copied to avoid in-place modification
    corrected_policy_value_funcs = {key: np.copy(value) for key, value in policy_value_funcs.items()}

    for i in range(1, len(data) - 1):
        left_jump = np.abs(gradients[i]) > gradient_jump_threshold
        right_jump = np.abs(gradients[i + 1]) > gradient_jump_threshold

        # If a jump is detected, correct the main data and policy/value functions
        if left_jump and right_jump:
            slope = (corrected_data[i - 1] - corrected_data[i - 2]) / (x[i - 1] - x[i - 2])
            correction = slope * (x[i] - x[i - 1])
            corrected_data[i] = corrected_data[i - 1] + correction
            
            # Apply the same correction to all policy and value functions
            for key in corrected_policy_value_funcs:
                slope_extra = (corrected_policy_value_funcs[key][i - 1] - corrected_policy_value_funcs[key][i - 2]) / (x[i - 1] - x[i - 2])
                correction_extra = slope_extra * (x[i] - x[i - 1])
                corrected_policy_value_funcs[key][i] = corrected_policy_value_funcs[key][i - 1] + correction_extra

        elif np.isnan(corrected_data[i]):
            slope = (corrected_data[i - 2] - corrected_data[i - 3]) / (x[i - 2] - x[i - 3])
            corrected_data[i] = corrected_data[i - 2] + slope * (x[i] - x[i - 2])
            
            # Handle NaN for policy and value functions similarly
            for key in corrected_policy_value_funcs:
                slope_extra = (corrected_policy_value_funcs[key][i - 2] - corrected_policy_value_funcs[key][i - 3]) / (x[i - 2] - x[i - 3])
                corrected_policy_value_funcs[key][i] = corrected_policy_value_funcs[key][i - 2] + slope_extra * (x[i] - x[i - 2])

    return corrected_data, corrected_policy_value_funcs

def mask_jumps(data, threshold=1.1):
    """
    Mask the data by introducing NaNs where there are large jumps/discontinuities.
    
    Use in plotting. 

    Parameters:
    data : np.ndarray
        The data array to be masked.
    threshold : float, optional
        The threshold for detecting jumps. Any jump larger than this value will be masked.
        
    Returns:
    np.ndarray
        The masked data array.
    """
    masked_data = np.copy(data)
    diffs = np.abs(np.diff(data))
    
    # Mask out the points after large jumps
    masked_data[1:][diffs > threshold] = np.nan
    return masked_data
