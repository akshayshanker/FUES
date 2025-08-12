import math
from numba import njit
import numpy as np

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


@njit(cache=True)
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

@njit(cache=True)
def interp_clean(xp, yp, x, extrap=False):
    """Clean interpolation function for FUES (non-ConSav case).
    
    Simpler and more robust than interp_as, with better handling of edge cases.
    Uses numpy's interp for the core interpolation and simplified extrapolation.
    
    Parameters
    ----------
    xp : 1D array
        x coordinates of data points (must be increasing)
    yp : 1D array
        y coordinates of data points
    x : 1D array
        x coordinates where we want interpolated values
    extrap : bool
        If True, extrapolate linearly outside bounds
        If False, use boundary values outside bounds
        
    Returns
    -------
    evals : 1D array
        Interpolated y values at x positions
    """
    n = len(xp)
    
    # Handle edge cases
    if n == 0:
        return np.zeros(len(x))
    if n == 1:
        return np.full(len(x), yp[0])
    
    # Use numpy's interp for the core interpolation
    evals = np.interp(x, xp, yp)
    
    # Handle extrapolation if requested
    if extrap:
        # Left extrapolation
        mask_left = x < xp[0]
        if np.any(mask_left):
            # Use slope from first two points
            slope_left = (yp[1] - yp[0]) / (xp[1] - xp[0])
            evals[mask_left] = yp[0] + slope_left * (x[mask_left] - xp[0])
        
        # Right extrapolation  
        mask_right = x > xp[-1]
        if np.any(mask_right):
            # Use slope from last two points
            slope_right = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2])
            evals[mask_right] = yp[-1] + slope_right * (x[mask_right] - xp[-1])
    
    return evals

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

def mask_jumps(data, threshold=0.9):
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


# ============== FUES Intersection Helpers ==============

# Constants
EPS_D = 1e-50
EPS_SEP = 1e-10
PARALLEL_GUARD = 1e-12

@njit(inline="always")
def _clip_open(x, lo, hi, eps):
    """Clip x into (lo+eps, hi-eps); if interval collapsed, return nan"""
    if lo > hi:
        lo, hi = hi, lo
    if (hi - lo) <= 2.0*eps:
        return np.nan
    if x <= lo + eps:
        return lo + eps
    if x >= hi - eps:
        return hi - eps
    return x

@njit(inline="always")
def _force_crossing_inside(
    L_x1, L_y1, L_x2, L_y2,
    R_x1, R_y1, R_x2, R_y2,
    e_lo, e_hi, eps, eps_d=EPS_D, parallel_guard=PARALLEL_GUARD
):
    """
    Force a crossing point strictly inside (e_lo, e_hi).
    """
    lo = e_lo if e_lo <= e_hi else e_hi
    hi = e_hi if e_hi >= e_lo else e_lo

    dxL = L_x2 - L_x1; dyL = L_y2 - L_y1
    dxR = R_x2 - R_x1; dyR = R_y2 - R_y1
    denom = dxL * dyR - dyL * dxR

    if np.abs(denom) >= parallel_guard:
        s = ((R_x1 - L_x1) * dyR - (R_y1 - L_y1) * dxR) / denom
        x_star = L_x1 + s * dxL
    else:
        x_star = 0.5 * (lo + hi)

    x = _clip_open(x_star, lo, hi, eps)
    dxL_safe = dxL if np.abs(dxL) > eps_d else (eps_d if dxL >= 0.0 else -eps_d)
    dxR_safe = dxR if np.abs(dxR) > eps_d else (eps_d if dxR >= 0.0 else -eps_d)
    sL = dyL / dxL_safe
    sR = dyR / dxR_safe
    yL = L_y1 + sL * (x - L_x1)
    yR = R_y1 + sR * (x - R_x1)

    y = 0.5 * (yL + yR)
    y_min = yL if yL < yR else yR
    y_max = yR if yR > yL else yL
    if y < y_min:
        y = y_min
    elif y > y_max:
        y = y_max

    return (x, y)

@njit
def add_intersection_from_pairs_with_sep(
    intersections, n_inter, intr_x, intr_y, sep,
    L, R, eps_d=EPS_D
):
    """
    Add two intersection points with ADAPTIVE separation.
    L and R format: (x1, y1, a1, p21, d1, x2, y2, a2, p22, d2)
    """
    L_x1, L_y1, L_a1, L_p21, L_d1, L_x2, L_y2, L_a2, L_p22, L_d2 = L
    R_x1, R_y1, R_a1, R_p21, R_d1, R_x2, R_y2, R_a2, R_p22, R_d2 = R
    
    if not np.isnan(intr_x) and n_inter + 1 < intersections.shape[0]:
        intersections[n_inter, 0] = intr_x - sep
        intersections[n_inter, 1] = intr_y
        
        denom_L = L_x2 - L_x1
        if np.abs(denom_L) < eps_d:
            denom_L = eps_d if denom_L >= 0.0 else -eps_d
        tL = (intr_x - sep - L_x1) / denom_L
        intersections[n_inter, 2] = L_a1 + tL * (L_a2 - L_a1)
        intersections[n_inter, 3] = L_p21 + tL * (L_p22 - L_p21)
        intersections[n_inter, 4] = L_d1 + tL * (L_d2 - L_d1)

        intersections[n_inter+1, 0] = intr_x + sep
        intersections[n_inter+1, 1] = intr_y
        
        denom_R = R_x2 - R_x1
        if np.abs(denom_R) < eps_d:
            denom_R = eps_d if denom_R >= 0.0 else -eps_d
        tR = (intr_x + sep - R_x1) / denom_R
        intersections[n_inter+1, 2] = R_a1 + tR * (R_a2 - R_a1)
        intersections[n_inter+1, 3] = R_p21 + tR * (R_p22 - R_p21)
        intersections[n_inter+1, 4] = R_d1 + tR * (R_d2 - R_d1)
        
        return n_inter + 2
    return n_inter

@njit(inline="always")
def _forced_intersection_twopoint(
    intersections, n_inter,
    e_lo, e_hi, sep_cap,
    L, R,
    eps_d, eps_sep, parallel_guard,
    dbg_i, dbg_j
):
    """
    Forced crossing of two line segments with adaptive separation (FUES/DC‑EGM helper).

    Purpose
    -------
    Compute a numerically robust crossing between two segments, force the x‑coordinate
    strictly inside (e_lo, e_hi), and write **two** intersection rows slightly to the
    left and right of the crossing. This is used when constructing the upper envelope
    of choice‑specific value functions: the small left/right separation avoids kinks and
    “flapping” when branches are nearly parallel.

    Interface (drop‑in)
    -------------------
    This function preserves the original signature and return values so it can replace
    existing inlined geometry blocks without changing call sites.

    Parameters
    ----------
    intersections : 2d float array, shape (M, 5)
        Preallocated buffer for intersection rows:
        [e, value, policy_1 (a'), policy_2, del_a].
    n_inter : int
        Current count of filled rows in `intersections`.
    e_lo, e_hi : float
        Open interval endpoints for the x‑coordinate of the crossing. Order is not
        assumed elsewhere, but here we **use them as provided** (callers should pass
        j < i+1 in EGM loops so e_hi > e_lo).
    sep_cap : float
        Optional upper bound on the left/right separation. If <= 0, cap is disabled.
    L, R : tuple[10 floats] or 1d float arrays (length 10)
        Segment endpoints and interpolands:
        (x1, y1, a1, p21, d1,   x2, y2, a2, p22, d2)
        for the LEFT (L) and RIGHT (R) branches.
        * Seeding is always done from the LEFT branch (EGM convention).
    eps_d : float
        Division guard (denominator floor) that **preserves sign** of dx.
    eps_sep : float
        Base epsilon for intersection separation and open‑interval padding when
        computing the forced crossing.
    parallel_guard : float
        Threshold for treating the two infinite lines as “near parallel” in the
        cross‑product denominator. Midpoint fallback is used if |den| < parallel_guard.
    dbg_i, dbg_j : int
        Loop indices used only in the rare fallback debug print.

    Returns
    -------
    n_inter_new : int
        Updated number of filled rows in `intersections` (n_inter or n_inter+2).
    intr_x, intr_y : float
        Crossing coordinates (after forcing inside (e_lo, e_hi)).
    seed_a_left, seed_d_left : float
        Policy seeds interpolated from the **LEFT** segment at intr_x (for
        downstream use as next‑iteration tail).
    added : bool
        True if two rows were successfully written to `intersections`, else False.

    Numerical method (summary)
    --------------------------
    1) Intersect the infinite lines (LEFT and RIGHT) using the cross‑product form.
       If nearly parallel, use the midpoint of (e_lo, e_hi) for x.
    2) Force x strictly inside (e_lo + eps_sep, e_hi − eps_sep).
    3) Evaluate y from both segments at that x using sign‑preserving slopes,
       take the average, and clip into [min(yL, yR), max(yL, yR)].
    4) Set separation sep = min(eps_sep, 0.25*(e_hi − e_lo)); if sep_cap>0, sep=min(sep, sep_cap).
    5) Emit **two** rows: (intr_x − sep, LEFT policies) and (intr_x + sep, RIGHT policies).
       Interpolate policies along the respective segments.
    """
 
    L_x1, L_y1, L_a1, L_p21, L_d1, L_x2, L_y2, L_a2, L_p22, L_d2 = L
    R_x1, R_y1, R_a1, R_p21, R_d1, R_x2, R_y2, R_a2, R_p22, R_d2 = R
    
    intr_x, intr_y = _force_crossing_inside(
        L_x1, L_y1, L_x2, L_y2,
        R_x1, R_y1, R_x2, R_y2,
        e_lo, e_hi, eps_sep, eps_d, parallel_guard
    )

    if np.isnan(intr_x):
        mid = 0.5 * (e_lo + e_hi)

        denom_L = L_x2 - L_x1
        if np.abs(denom_L) < eps_d:
            denom_L = eps_d if denom_L >= 0.0 else -eps_d
        sL = (L_y2 - L_y1) / denom_L
        yL = L_y1 + sL * (mid - L_x1)

        denom_R = R_x2 - R_x1
        if np.abs(denom_R) < eps_d:
            denom_R = eps_d if denom_R >= 0.0 else -eps_d
        sR = (R_y2 - R_y1) / denom_R
        yR = R_y1 + sR * (mid - R_x1)

        intr_x = mid
        intr_y = 0.5 * (yL + yR)

        print(f"SCAN DEBUG: intr_x is NaN at i={dbg_i}, j={dbg_j}. Falling back to midpoint.")

    interval_length = e_hi - e_lo
    if np.abs(interval_length) < eps_d:
        interval_length = eps_d if interval_length >= 0.0 else -eps_d

    sep = 0.25 * interval_length
    if sep > eps_sep:
        sep = eps_sep
    if sep_cap > 0.0 and sep > sep_cap:
        sep = sep_cap

    n_new = add_intersection_from_pairs_with_sep(
        intersections, n_inter, intr_x, intr_y, sep,
        L, R, eps_d
    )

    if n_new > n_inter:
        denom_L = L_x2 - L_x1
        if np.abs(denom_L) < eps_d:
            denom_L = eps_d if denom_L >= 0.0 else -eps_d
        tL = (intr_x - L_x1) / denom_L
        seed_a = L_a1 + tL * (L_a2 - L_a1)
        seed_d = L_d1 + tL * (L_d2 - L_d1)
        return n_new, intr_x, intr_y, seed_a, seed_d, True
    return n_inter, 0.0, 0.0, 0.0, 0.0, False

# ============== Circular Buffer Utilities ==============

@njit
def circ_put(buf, head, value):
    """Write *value* at *head* position, return new head index."""
    buf[head] = value
    return (head + 1) % buf.size
