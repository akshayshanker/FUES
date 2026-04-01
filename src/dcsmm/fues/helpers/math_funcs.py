import math
from numba import njit
import numpy as np
from dcsmm.fues.constants import EPS_D, EPS_FWD_BACK

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


@njit
def rootsearch_wf(f, a, b, dx, h_prime, z, Ud_prime_a, Ud_prime_h, t):
    """Rootsearch that also returns f values at bracket endpoints.

    Returns (x1, x2, f1, f2) to avoid redundant evaluations in bisection.
    """
    x1 = a
    f1 = f(a, h_prime, z, Ud_prime_a, Ud_prime_h, t)
    x2 = a + dx
    f2 = f(x2, h_prime, z, Ud_prime_a, Ud_prime_h, t)

    while f1 * f2 > 0.0:
        if x1 >= b:
            return np.nan, np.nan, np.nan, np.nan
        x1 = x2
        f1 = f2
        x2 = x1 + dx
        f2 = f(x2, h_prime, z, Ud_prime_a, Ud_prime_h, t)
    return x1, x2, f1, f2


@njit
def bisect_wf(f, x1, x2, f1, f2, h_prime, z, Ud_prime_a, Ud_prime_h, t, xtol=1e-6):
    """Bisection that accepts pre-computed f1, f2 to avoid redundant evals.

    Returns (root, converged) tuple.
    """
    # Check if already at root
    if f1 == 0.0:
        return x1
    if f2 == 0.0:
        return x2

    # Bisection iterations (log2((x2-x1)/xtol) iterations needed)
    while (x2 - x1) > xtol:
        x3 = 0.5 * (x1 + x2)
        f3 = f(x3, h_prime, z, Ud_prime_a, Ud_prime_h, t)
        if f3 == 0.0:
            return x3
        if f2 * f3 < 0.0:
            x1 = x3
            f1 = f3
        else:
            x2 = x3
            f2 = f3
    return 0.5 * (x1 + x2)


@njit
def find_roots_piecewise_linear(resid, grid, max_roots, root_eps=0.0):
    """Find all roots of a piecewise linear function via sign changes.

    Given a residual function evaluated at grid points, finds all roots
    by detecting sign changes and using linear interpolation.

    Parameters
    ----------
    resid : 1D array
        Residual values at each grid point (must be same length as grid)
    grid : 1D array
        Grid points where residual is evaluated
    max_roots : int
        Maximum number of roots to find
    root_eps : float, optional
        If > 0, use coarse sampling at this spacing instead of checking
        every grid point. This makes root finding O(grid_range/root_eps)
        instead of O(n_grid). Default 0.0 means check every point.

    Returns
    -------
    roots : 1D array
        Array of roots (zeros padded with 0.0)
    n_roots : int
        Number of roots found

    Notes
    -----
    When root_eps > 0, we sample at coarse intervals and interpolate
    directly between sample points when a sign change is detected.
    This decouples root-finding cost from grid density.
    """
    n_grid = len(grid)
    roots = np.zeros(max_roots)
    n_roots = 0

    if n_grid < 2:
        return roots, n_roots

    # Determine step size
    grid_range = grid[-1] - grid[0]
    if grid_range <= 0.0:
        return roots, n_roots
    avg_spacing = grid_range / (n_grid - 1)

    if root_eps > 0.0 and root_eps > avg_spacing:
        # Coarse sampling: step based on root_eps spacing
        # Only use if root_eps is larger than grid spacing
        step = int(root_eps / avg_spacing)
    else:
        # Fine sampling: check every grid point
        # Used when root_eps=0 or grid is coarser than root_eps
        step = 1

    # Scan at step intervals
    i = 0
    while i < n_grid - 1 and n_roots < max_roots:
        j = min(i + step, n_grid - 1)

        r_i = resid[i]
        r_j = resid[j]

        # Check for sign change
        if r_i * r_j < 0.0:
            # Linear interpolation for exact root
            x_i = grid[i]
            x_j = grid[j]
            denom = r_j - r_i
            if denom != 0.0:
                root = x_i - r_i * (x_j - x_i) / denom
                roots[n_roots] = root
                n_roots += 1
        elif r_i == 0.0:
            # Exact root at sample point
            roots[n_roots] = grid[i]
            n_roots += 1

        i = j

    # Check last grid point
    if n_roots < max_roots and resid[n_grid - 1] == 0.0:
        roots[n_roots] = grid[n_grid - 1]
        n_roots += 1

    return roots, n_roots


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
    """Interpolate 1D with linear extrapolation.

    When *x* is sorted (ascending), uses a linear walk instead of
    binary search per point, giving O(n + m) instead of O(n log m).

    Parameters
    ----------
    xp : 1D array
        x coordinates of data points (must be increasing)
    yp : 1D array
        y coordinates of data points
    x : 1D array
        points to interpolate
    extrap : bool
        If True, linearly extrapolate beyond xp bounds.

    Returns
    -------
    evals : 1D array
        y values at x
    """
    n_x = len(x)
    n_xp = len(xp)
    evals = np.empty(n_x)

    if n_xp == 0:
        for i in range(n_x):
            evals[i] = 0.0
        return evals

    if n_xp == 1:
        for i in range(n_x):
            evals[i] = yp[0]
        return evals

    x_lo, x_hi = xp[0], xp[-1]

    # Extrapolation slopes
    dx_left = xp[1] - xp[0]
    slope_left = (yp[1] - yp[0]) / dx_left if dx_left != 0.0 else 0.0
    dx_right = xp[-1] - xp[-2]
    slope_right = (yp[-1] - yp[-2]) / dx_right if dx_right != 0.0 else 0.0

    # Detect whether x is sorted (check first few + last)
    x_sorted = True
    check_n = min(n_x, 8)
    for i in range(1, check_n):
        if x[i] < x[i - 1]:
            x_sorted = False
            break
    if x_sorted and n_x > check_n and x[-1] < x[-2]:
        x_sorted = False

    if x_sorted:
        # Linear walk: O(n + m)
        j = 0  # current interval in xp
        for i in range(n_x):
            xi = x[i]
            if xi <= x_lo:
                evals[i] = yp[0] + (xi - x_lo) * slope_left if extrap else yp[0]
            elif xi >= x_hi:
                evals[i] = yp[-1] + (xi - x_hi) * slope_right if extrap else yp[-1]
            else:
                # Walk j forward until xp[j+1] > xi
                while j < n_xp - 2 and xp[j + 1] <= xi:
                    j += 1
                dx = xp[j + 1] - xp[j]
                if dx != 0.0:
                    t = (xi - xp[j]) / dx
                    evals[i] = yp[j] + t * (yp[j + 1] - yp[j])
                else:
                    evals[i] = yp[j]
    else:
        # Fallback: binary search per point
        for i in range(n_x):
            xi = x[i]
            if xi <= x_lo:
                evals[i] = yp[0] + (xi - x_lo) * slope_left if extrap else yp[0]
            elif xi >= x_hi:
                evals[i] = yp[-1] + (xi - x_hi) * slope_right if extrap else yp[-1]
            else:
                lo, hi = 0, n_xp - 1
                while hi - lo > 1:
                    mid = (lo + hi) >> 1
                    if xp[mid] <= xi:
                        lo = mid
                    else:
                        hi = mid
                dx = xp[hi] - xp[lo]
                if dx != 0.0:
                    t = (xi - xp[lo]) / dx
                    evals[i] = yp[lo] + t * (yp[hi] - yp[lo])
                else:
                    evals[i] = yp[lo]

    return evals


@njit(cache=True)
def interp_as_scalar(xp, yp, x, extrap=False):
    """Function interpolates 1D (scalar version) with linear extrapolation.

    Parameters
    ----------
    xp : 1D array
        x coordinates of data points (must be increasing)
    yp : 1D array
        y coordinates of data points
    x : float64
        scalar point to interpolate

    Returns
    -------
    eval : float64
        y value at x
    """
    n_xp = len(xp)

    if n_xp == 0:
        return 0.0

    if n_xp == 1:
        return float(yp[0])

    x_lo, x_hi = xp[0], xp[-1]

    # Left boundary/extrapolation
    if x <= x_lo:
        if extrap:
            dx = xp[1] - xp[0]
            if dx != 0.0:
                return float(yp[0] + (x - x_lo) * (yp[1] - yp[0]) / dx)
        return float(yp[0])

    # Right boundary/extrapolation
    if x >= x_hi:
        if extrap:
            dx = xp[-1] - xp[-2]
            if dx != 0.0:
                return float(yp[-1] + (x - x_hi) * (yp[-1] - yp[-2]) / dx)
        return float(yp[-1])

    # Binary search for interval: find j such that xp[j] <= x < xp[j+1]
    lo, hi = 0, n_xp - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid

    # Linear interpolation
    dx = xp[hi] - xp[lo]
    if dx != 0.0:
        t = (x - xp[lo]) / dx
        return float(yp[lo] + t * (yp[hi] - yp[lo]))
    return float(yp[lo])


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
            # Use slope from first two points, with zero-division protection
            dx_left = xp[1] - xp[0]
            if abs(dx_left) > 1e-14:
                slope_left = (yp[1] - yp[0]) / dx_left
                evals[mask_left] = yp[0] + slope_left * (x[mask_left] - xp[0])
            else:
                # If points are too close, use constant extrapolation
                evals[mask_left] = yp[0]
        
        # Right extrapolation  
        mask_right = x > xp[-1]
        if np.any(mask_right):
            # Use slope from last two points, with zero-division protection
            dx_right = xp[-1] - xp[-2]
            if abs(dx_right) > 1e-14:
                slope_right = (yp[-1] - yp[-2]) / dx_right
                evals[mask_right] = yp[-1] + slope_right * (x[mask_right] - xp[-1])
            else:
                # If points are too close, use constant extrapolation
                evals[mask_right] = yp[-1]
    
    return evals


@njit(cache=True)
def interp_clean_single(xp, yp, x_query, extrap=True):
    """
    Interpolate a single x_query point - fully compiled version.
    Used for batch operations where we need to interpolate at different
    source arrays for each target point.
    """
    n = len(xp)
    
    if n == 0:
        return 0.0
    if n == 1:
        return yp[0]
    
    # Handle left extrapolation
    if x_query <= xp[0]:
        if extrap and n >= 2:
            dx = xp[1] - xp[0]
            if abs(dx) > 1e-14:
                slope = (yp[1] - yp[0]) / dx
                return yp[0] + slope * (x_query - xp[0])
        return yp[0]
    
    # Handle right extrapolation
    if x_query >= xp[-1]:
        if extrap and n >= 2:
            dx = xp[-1] - xp[-2]
            if abs(dx) > 1e-14:
                slope = (yp[-1] - yp[-2]) / dx
                return yp[-1] + slope * (x_query - xp[-1])
        return yp[-1]
    
    # Binary search for interval
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= x_query:
            lo = mid
        else:
            hi = mid
    
    # Linear interpolation
    t = (x_query - xp[lo]) / (xp[hi] - xp[lo])
    return yp[lo] + t * (yp[hi] - yp[lo])


@njit(cache=True)
def interp_to_grid_compiled(m_refined, q_refined, c_refined, w_grid, 
                            Q_out, policy_out, i_h, i_y):
    """
    Interpolate refined EGM solution onto w_grid for a single (i_h, i_y) pair.
    Writes directly to pre-allocated output arrays.
    
    This is the compiled inner loop - called from Python outer loop.
    
    Parameters
    ----------
    m_refined : 1D array
        Refined endogenous grid
    q_refined : 1D array
        Refined Q values
    c_refined : 1D array
        Refined consumption values
    w_grid : 1D array
        Target wealth grid
    Q_out : 3D array (n_W, n_H, n_Y)
        Output array for Q values (modified in place)
    policy_out : 3D array (n_W, n_H, n_Y)
        Output array for policy (modified in place)
    i_h : int
        Housing index
    i_y : int
        Income index
    """
    n_W = len(w_grid)
    n_m = len(m_refined)
    
    if n_m == 0:
        return
    if n_m == 1:
        for iw in range(n_W):
            Q_out[iw, i_h, i_y] = q_refined[0]
            policy_out[iw, i_h, i_y] = c_refined[0]
        return
    
    # Pre-compute slopes for extrapolation
    dx_left = m_refined[1] - m_refined[0]
    dx_right = m_refined[-1] - m_refined[-2]
    
    if abs(dx_left) > 1e-14:
        slope_q_left = (q_refined[1] - q_refined[0]) / dx_left
        slope_c_left = (c_refined[1] - c_refined[0]) / dx_left
    else:
        slope_q_left = 0.0
        slope_c_left = 0.0
        
    if abs(dx_right) > 1e-14:
        slope_q_right = (q_refined[-1] - q_refined[-2]) / dx_right
        slope_c_right = (c_refined[-1] - c_refined[-2]) / dx_right
    else:
        slope_q_right = 0.0
        slope_c_right = 0.0
    
    m_min = m_refined[0]
    m_max = m_refined[-1]
    
    for iw in range(n_W):
        w = w_grid[iw]
        
        # Left extrapolation
        if w <= m_min:
            Q_out[iw, i_h, i_y] = q_refined[0] + slope_q_left * (w - m_min)
            policy_out[iw, i_h, i_y] = c_refined[0] + slope_c_left * (w - m_min)
            continue
        
        # Right extrapolation
        if w >= m_max:
            Q_out[iw, i_h, i_y] = q_refined[-1] + slope_q_right * (w - m_max)
            policy_out[iw, i_h, i_y] = c_refined[-1] + slope_c_right * (w - m_max)
            continue
        
        # Binary search for interval
        lo, hi = 0, n_m - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if m_refined[mid] <= w:
                lo = mid
            else:
                hi = mid
        
        # Linear interpolation
        t = (w - m_refined[lo]) / (m_refined[hi] - m_refined[lo])
        Q_out[iw, i_h, i_y] = q_refined[lo] + t * (q_refined[hi] - q_refined[lo])
        policy_out[iw, i_h, i_y] = c_refined[lo] + t * (c_refined[hi] - c_refined[lo])


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
    e_lo, e_hi, eps, eps_d, parallel_guard
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
def add_single_intersection_right(
    intersections, n_inter, intr_x, intr_y, sep,
    L, R, eps_d
):
    """
    Add a single intersection point on the right side.
    Uses RIGHT branch policies at intr_x + sep.
    L and R format: (x1, y1, a1, p21, d1, x2, y2, a2, p22, d2)
    """
    L_x1, L_y1, L_a1, L_p21, L_d1, L_x2, L_y2, L_a2, L_p22, L_d2 = L
    R_x1, R_y1, R_a1, R_p21, R_d1, R_x2, R_y2, R_a2, R_p22, R_d2 = R
    
    if not np.isnan(intr_x) and n_inter < intersections.shape[0]:
        # Add only the right intersection point
        intersections[n_inter, 0] = intr_x + sep
        intersections[n_inter, 1] = intr_y
        
        # Use RIGHT branch policies
        denom_R = R_x2 - R_x1
        if np.abs(denom_R) < eps_d:
            denom_R = eps_d if denom_R >= 0.0 else -eps_d
        tR = (intr_x + sep - R_x1) / denom_R
        intersections[n_inter, 2] = R_a1 + tR * (R_a2 - R_a1)
        intersections[n_inter, 3] = R_p21 + tR * (R_p22 - R_p21)
        intersections[n_inter, 4] = R_d1 + tR * (R_d2 - R_d1)
        
        return n_inter + 1
    return n_inter

@njit
def add_intersection_from_pairs_with_sep(
    intersections, n_inter, intr_x, intr_y, sep,
    L, R, eps_d
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
    dbg_i, dbg_j, single_intersection=False
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
    single_intersection : bool, optional
        If True, add only one intersection point (on the right) instead of two.
        Default is False for backward compatibility.

    Returns
    -------
    n_inter_new : int
        Updated number of filled rows in `intersections` (n_inter, n_inter+1, or n_inter+2).
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
    5) Emit rows based on single_intersection flag:
       - If False: **two** rows: (intr_x − sep, LEFT policies) and (intr_x + sep, RIGHT policies)
       - If True: **one** row: (intr_x + sep, RIGHT policies)
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

        # NaN intersection fallback — silent in production
        pass

    interval_length = e_hi - e_lo
    if np.abs(interval_length) < eps_d:
        interval_length = eps_d if interval_length >= 0.0 else -eps_d

    sep = 0.25 * interval_length
    if sep > eps_sep:
        sep = eps_sep
    if sep_cap > 0.0 and sep > sep_cap:
        sep = sep_cap

    if single_intersection:
        n_new = add_single_intersection_right(
            intersections, n_inter, intr_x, intr_y, sep,
            L, R, eps_d
        )
    else:
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

@njit(cache=True)
def circ_put(buf, head, value):
    """Write *value* at *head* position, return new head index."""
    buf[head] = value
    return (head + 1) % buf.size


@njit(cache=True)
def _merge_sorted_with_few(
    main_e, main_v, main_p1, main_p2, main_d,
    few_e, few_v, few_p1, few_p2, few_d,
):
    """Merge sorted main arrays with a small secondary set via linear merge.

    Parameters
    ----------
    main_e, main_v, main_p1, main_p2, main_d : 1d arrays
        Sorted main (kept) arrays.
    few_e, few_v, few_p1, few_p2, few_d : 1d arrays
        Small unsorted intersection arrays (typically 2-6 points).

    Returns
    -------
    out_e, out_v, out_p1, out_p2, out_d : 1d arrays
        Merged sorted arrays.
    is_inter : 1d bool array
        True for elements originating from the few (intersection) arrays.
    """
    K = main_e.size
    J = few_e.size
    N = K + J

    out_e = np.empty(N)
    out_v = np.empty(N)
    out_p1 = np.empty(N)
    out_p2 = np.empty(N)
    out_d = np.empty(N)
    is_inter = np.zeros(N, dtype=np.bool_)

    # Sort the few (intersection) arrays — typically J <= 6, argsort is fine
    si = np.argsort(few_e)
    f_e = few_e[si]
    f_v = few_v[si]
    f_p1 = few_p1[si]
    f_p2 = few_p2[si]
    f_d = few_d[si]

    # Linear merge
    i = 0
    j = 0
    k = 0
    while i < K and j < J:
        if main_e[i] <= f_e[j]:
            out_e[k] = main_e[i]
            out_v[k] = main_v[i]
            out_p1[k] = main_p1[i]
            out_p2[k] = main_p2[i]
            out_d[k] = main_d[i]
            i += 1
        else:
            out_e[k] = f_e[j]
            out_v[k] = f_v[j]
            out_p1[k] = f_p1[j]
            out_p2[k] = f_p2[j]
            out_d[k] = f_d[j]
            is_inter[k] = True
            j += 1
        k += 1

    while i < K:
        out_e[k] = main_e[i]
        out_v[k] = main_v[i]
        out_p1[k] = main_p1[i]
        out_p2[k] = main_p2[i]
        out_d[k] = main_d[i]
        i += 1
        k += 1

    while j < J:
        out_e[k] = f_e[j]
        out_v[k] = f_v[j]
        out_p1[k] = f_p1[j]
        out_p2[k] = f_p2[j]
        out_d[k] = f_d[j]
        is_inter[k] = True
        j += 1
        k += 1

    return out_e, out_v, out_p1, out_p2, out_d, is_inter


# ============== Post-Interpolation Jump Correction ==============

@njit(cache=True)
def calculate_gradient_1d(data, x):
    """Calculate gradients for 1D data."""
    n = len(data)
    gradients = np.zeros(n)
    for i in range(1, n):
        dx = x[i] - x[i - 1]
        if dx != 0:
            gradients[i] = (data[i] - data[i - 1]) / dx
        else:
            gradients[i] = 0.0
    return gradients


@njit(cache=True)
def correct_jumps1d_arr(data, x, gradient_jump_threshold, v_arr, d_arr, a_arr):
    """
    Removes jumps in 1D interpolated data using explicit arrays.

    Parameters
    ----------
    data : np.ndarray
        The 1D array of interpolated values to correct.
    x : np.ndarray
        The x-coordinates corresponding to data.
    gradient_jump_threshold : float
        Threshold for detecting jumps based on gradient magnitude.
    v_arr : np.ndarray
        Value function array to correct alongside data.
    d_arr : np.ndarray
        Durable choice array to correct alongside data.
    a_arr : np.ndarray
        Asset choice array to correct alongside data.

    Returns
    -------
    corrected_data : np.ndarray
        The corrected data array.
    corrected_v : np.ndarray
        The corrected value function array.
    corrected_d : np.ndarray
        The corrected durable choice array.
    corrected_a : np.ndarray
        The corrected asset choice array.
    """
    n = len(data)
    corrected_data = np.copy(data)
    corrected_v = np.copy(v_arr)
    corrected_d = np.copy(d_arr)
    corrected_a = np.copy(a_arr)

    gradients = calculate_gradient_1d(data, x)

    for i in range(1, n - 1):
        left_jump = np.abs(gradients[i]) > gradient_jump_threshold
        right_jump = np.abs(gradients[i + 1]) > gradient_jump_threshold

        if left_jump and right_jump:
            # Interpolate from neighbors
            corrected_data[i] = 0.5 * (data[i - 1] + data[i + 1])
            corrected_v[i] = 0.5 * (v_arr[i - 1] + v_arr[i + 1])
            corrected_d[i] = 0.5 * (d_arr[i - 1] + d_arr[i + 1])
            corrected_a[i] = 0.5 * (a_arr[i - 1] + a_arr[i + 1])

    return corrected_data, corrected_v, corrected_d, corrected_a


# ============== Fused Multi-Array Interpolation ==============


@njit(cache=True)
def interp_as_3(xp, yp1, yp2, yp3, x):
    """Interpolate three y-arrays on the same (xp, x) grids in one walk.

    Both xp and x must be sorted ascending. Avoids tripling the walk
    cost when interpolating multiple policies on the same grid.

    Returns
    -------
    out1, out2, out3 : 1D arrays
    """
    n_x = len(x)
    n_xp = len(xp)
    out1 = np.empty(n_x)
    out2 = np.empty(n_x)
    out3 = np.empty(n_x)

    if n_xp == 0:
        for i in range(n_x):
            out1[i] = 0.0
            out2[i] = 0.0
            out3[i] = 0.0
        return out1, out2, out3

    if n_xp == 1:
        for i in range(n_x):
            out1[i] = yp1[0]
            out2[i] = yp2[0]
            out3[i] = yp3[0]
        return out1, out2, out3

    x_lo, x_hi = xp[0], xp[-1]

    j = 0
    for i in range(n_x):
        xi = x[i]
        if xi <= x_lo:
            out1[i] = yp1[0]
            out2[i] = yp2[0]
            out3[i] = yp3[0]
        elif xi >= x_hi:
            out1[i] = yp1[-1]
            out2[i] = yp2[-1]
            out3[i] = yp3[-1]
        else:
            while j < n_xp - 2 and xp[j + 1] <= xi:
                j += 1
            dx = xp[j + 1] - xp[j]
            if dx != 0.0:
                t = (xi - xp[j]) / dx
                out1[i] = yp1[j] + t * (yp1[j + 1] - yp1[j])
                out2[i] = yp2[j] + t * (yp2[j + 1] - yp2[j])
                out3[i] = yp3[j] + t * (yp3[j + 1] - yp3[j])
            else:
                out1[i] = yp1[j]
                out2[i] = yp2[j]
                out3[i] = yp3[j]

    return out1, out2, out3


@njit(cache=True)
def interp_as_2(xp, yp1, yp2, x):
    """Interpolate two y-arrays on the same (xp, x) grids in one walk.

    Both xp and x must be sorted ascending.

    Returns
    -------
    out1, out2 : 1D arrays
    """
    n_x = len(x)
    n_xp = len(xp)
    out1 = np.empty(n_x)
    out2 = np.empty(n_x)

    if n_xp == 0:
        for i in range(n_x):
            out1[i] = 0.0
            out2[i] = 0.0
        return out1, out2

    if n_xp == 1:
        for i in range(n_x):
            out1[i] = yp1[0]
            out2[i] = yp2[0]
        return out1, out2

    x_lo, x_hi = xp[0], xp[-1]

    j = 0
    for i in range(n_x):
        xi = x[i]
        if xi <= x_lo:
            out1[i] = yp1[0]
            out2[i] = yp2[0]
        elif xi >= x_hi:
            out1[i] = yp1[-1]
            out2[i] = yp2[-1]
        else:
            while j < n_xp - 2 and xp[j + 1] <= xi:
                j += 1
            dx = xp[j + 1] - xp[j]
            if dx != 0.0:
                t = (xi - xp[j]) / dx
                out1[i] = yp1[j] + t * (yp1[j + 1] - yp1[j])
                out2[i] = yp2[j] + t * (yp2[j + 1] - yp2[j])
            else:
                out1[i] = yp1[j]
                out2[i] = yp2[j]

    return out1, out2


# ============== Scan Helper Utilities ==============


@njit(inline="always")
def check_same_seg(e_grid, kappa_hat, idx1, idx2, m_bar,
                   eps_d=EPS_D, eps_fwd_back=EPS_FWD_BACK):
    """Check if two points are on the same segment (no jump in policy)."""
    if idx1 < 0 or idx2 < 0 or idx1 >= len(e_grid) or idx2 >= len(e_grid):
        return False
    de = max(eps_d, e_grid[idx2] - e_grid[idx1])
    if de < eps_d * 10:
        return False
    g_a = np.abs((kappa_hat[idx2] - kappa_hat[idx1]) / de)
    return g_a < m_bar and de < eps_fwd_back


@njit(inline="always")
def find_safe_extrapolation_point(e_grid, a_prime, base_idx, N, m_bar,
                                  forward=True, eps_d=EPS_D,
                                  eps_fwd_back=EPS_FWD_BACK):
    """Find a safe point for extrapolation on the same segment as base_idx.

    Returns the index of a point that doesn't jump from base_idx,
    or base_idx if none found.
    """
    if forward:
        for offset in range(1, min(4, N - base_idx)):
            test_idx = base_idx + offset
            if test_idx < N and check_same_seg(
                e_grid, a_prime, base_idx, test_idx, m_bar, eps_d, eps_fwd_back,
            ):
                return test_idx
    else:
        for offset in range(1, min(4, base_idx + 1)):
            test_idx = base_idx - offset
            if test_idx >= 0 and check_same_seg(
                e_grid, a_prime, test_idx, base_idx, m_bar, eps_d, eps_fwd_back,
            ):
                return test_idx
    return base_idx


@njit(inline="always")
def make_pair_from_indices_or_fallback(e, v, a, p2, d,
                                       lo_idx, hi_idx, fb_lo, fb_hi, N):
    """Build a segment pair from two indices, falling back if either is -1."""
    fb_lo = max(0, min(fb_lo, N - 1))
    fb_hi = max(0, min(fb_hi, N - 1))

    if lo_idx != -1 and hi_idx != -1:
        return (e[lo_idx], v[lo_idx], a[lo_idx], p2[lo_idx], d[lo_idx],
                e[hi_idx], v[hi_idx], a[hi_idx], p2[hi_idx], d[hi_idx])
    else:
        return (e[fb_lo], v[fb_lo], a[fb_lo], p2[fb_lo], d[fb_lo],
                e[fb_hi], v[fb_hi], a[fb_hi], p2[fb_hi], d[fb_hi])


@njit(cache=True)
def postclean_double_jump_mask(e_grid, a_prime, m_bar, skip_mask, eps_d=EPS_D):
    """Remove isolated points where both neighbours are policy jumps.

    Keep[i] == False iff BOTH neighbors of i are policy jumps > m_bar.
    First and last points are always kept.
    """
    N = e_grid.size
    keep = np.ones(N, dtype=np.bool_)
    if N <= 2:
        return keep

    for i in range(1, N - 1):
        deL = e_grid[i] - e_grid[i - 1]
        deR = e_grid[i + 1] - e_grid[i]
        if np.abs(deL) < eps_d:
            deL = eps_d if deL >= 0.0 else -eps_d
        if np.abs(deR) < eps_d:
            deR = eps_d if deR >= 0.0 else -eps_d

        gL = np.abs((a_prime[i] - a_prime[i - 1]) / deL)
        gR = np.abs((a_prime[i + 1] - a_prime[i]) / deR)

        if (gL > m_bar) and (gR > m_bar):
            keep[i] = False

    return keep


@njit(cache=True)
def calculate_gradient_1d(data, x):
    """Finite-difference gradient of data with respect to x.

    Returns an array of length len(data)+1 representing the slope
    between consecutive points (element i is the slope from i-1 to i).
    Element 0 is set to 0.
    """
    n = len(data)
    grad = np.empty(n + 1)
    grad[0] = 0.0
    for i in range(1, n):
        dx = x[i] - x[i - 1]
        if dx == 0.0:
            grad[i] = 0.0
        else:
            grad[i] = (data[i] - data[i - 1]) / dx
    grad[n] = 0.0
    return grad


@njit(cache=True)
def correct_jumps1d_arr(data, x, gradient_jump_threshold,
                        v_arr, d_arr, a_arr):
    """Remove isolated jumps in 1D interpolated policy arrays (two-pass).

    A point is flagged when both the left and right gradients exceed
    *gradient_jump_threshold*.  Pass 1 corrects using the original
    neighbours; pass 2 recomputes gradients on the corrected arrays
    and catches clusters that pass 1 could not fix.

    Parameters
    ----------
    data : np.ndarray
        1D array of interpolated values to correct.
    x : np.ndarray
        x-coordinates corresponding to *data*.
    gradient_jump_threshold : float
        Threshold for detecting jumps.
    v_arr, d_arr, a_arr : np.ndarray
        Companion arrays (value, durable choice, asset choice) corrected
        in lockstep with *data*.

    Returns
    -------
    corrected_data, corrected_v, corrected_d, corrected_a : np.ndarray
    """
    n = len(data)
    corrected_data = np.copy(data)
    corrected_v = np.copy(v_arr)
    corrected_d = np.copy(d_arr)
    corrected_a = np.copy(a_arr)

    # --- Pass 1: correct using original neighbours ---
    gradients = calculate_gradient_1d(data, x)
    for i in range(1, n - 1):
        if (np.abs(gradients[i]) > gradient_jump_threshold
                and np.abs(gradients[i + 1]) > gradient_jump_threshold):
            corrected_data[i] = 0.5 * (data[i - 1] + data[i + 1])
            corrected_v[i] = 0.5 * (v_arr[i - 1] + v_arr[i + 1])
            corrected_d[i] = 0.5 * (d_arr[i - 1] + d_arr[i + 1])
            corrected_a[i] = 0.5 * (a_arr[i - 1] + a_arr[i + 1])

    # --- Pass 2: recompute gradients on corrected arrays ---
    gradients2 = calculate_gradient_1d(corrected_data, x)
    for i in range(1, n - 1):
        if (np.abs(gradients2[i]) > gradient_jump_threshold
                and np.abs(gradients2[i + 1]) > gradient_jump_threshold):
            corrected_data[i] = 0.5 * (corrected_data[i - 1]
                                       + corrected_data[i + 1])
            corrected_v[i] = 0.5 * (corrected_v[i - 1]
                                     + corrected_v[i + 1])
            corrected_d[i] = 0.5 * (corrected_d[i - 1]
                                     + corrected_d[i + 1])
            corrected_a[i] = 0.5 * (corrected_a[i - 1]
                                     + corrected_a[i + 1])

    return corrected_data, corrected_v, corrected_d, corrected_a


@njit(cache=True)
def interp_as_4(xp, yp1, yp2, yp3, yp4, x, extrap=False):
    """Fused interpolation of 4 y-arrays on the same xp/x grids.

    Single index walk (O(n+m)) shared across all 4 y-arrays,
    eliminating 3 redundant walks compared to 4 separate interp_as calls.

    Parameters
    ----------
    xp : 1D array  – sorted data x-coordinates
    yp1, yp2, yp3, yp4 : 1D arrays – y data to interpolate
    x  : 1D array  – query points (sorted ascending for linear walk)
    extrap : bool   – linearly extrapolate beyond xp bounds

    Returns
    -------
    out1, out2, out3, out4 : 1D arrays
    """
    n_x = len(x)
    n_xp = len(xp)
    out1 = np.empty(n_x)
    out2 = np.empty(n_x)
    out3 = np.empty(n_x)
    out4 = np.empty(n_x)

    if n_xp == 0:
        for i in range(n_x):
            out1[i] = 0.0
            out2[i] = 0.0
            out3[i] = 0.0
            out4[i] = 0.0
        return out1, out2, out3, out4

    if n_xp == 1:
        for i in range(n_x):
            out1[i] = yp1[0]
            out2[i] = yp2[0]
            out3[i] = yp3[0]
            out4[i] = yp4[0]
        return out1, out2, out3, out4

    x_lo, x_hi = xp[0], xp[-1]

    dx_left = xp[1] - xp[0]
    dx_right = xp[-1] - xp[-2]
    inv_left = 1.0 / dx_left if dx_left != 0.0 else 0.0
    inv_right = 1.0 / dx_right if dx_right != 0.0 else 0.0
    sl1 = (yp1[1] - yp1[0]) * inv_left
    sl2 = (yp2[1] - yp2[0]) * inv_left
    sl3 = (yp3[1] - yp3[0]) * inv_left
    sl4 = (yp4[1] - yp4[0]) * inv_left
    sr1 = (yp1[-1] - yp1[-2]) * inv_right
    sr2 = (yp2[-1] - yp2[-2]) * inv_right
    sr3 = (yp3[-1] - yp3[-2]) * inv_right
    sr4 = (yp4[-1] - yp4[-2]) * inv_right

    # Detect sorted (same heuristic as interp_as)
    x_sorted = True
    check_n = min(n_x, 8)
    for i in range(1, check_n):
        if x[i] < x[i - 1]:
            x_sorted = False
            break
    if x_sorted and n_x > check_n and x[-1] < x[-2]:
        x_sorted = False

    if x_sorted:
        j = 0
        for i in range(n_x):
            xi = x[i]
            if xi <= x_lo:
                if extrap:
                    d = xi - x_lo
                    out1[i] = yp1[0] + d * sl1
                    out2[i] = yp2[0] + d * sl2
                    out3[i] = yp3[0] + d * sl3
                    out4[i] = yp4[0] + d * sl4
                else:
                    out1[i] = yp1[0]
                    out2[i] = yp2[0]
                    out3[i] = yp3[0]
                    out4[i] = yp4[0]
            elif xi >= x_hi:
                if extrap:
                    d = xi - x_hi
                    out1[i] = yp1[-1] + d * sr1
                    out2[i] = yp2[-1] + d * sr2
                    out3[i] = yp3[-1] + d * sr3
                    out4[i] = yp4[-1] + d * sr4
                else:
                    out1[i] = yp1[-1]
                    out2[i] = yp2[-1]
                    out3[i] = yp3[-1]
                    out4[i] = yp4[-1]
            else:
                while j < n_xp - 2 and xp[j + 1] <= xi:
                    j += 1
                dx = xp[j + 1] - xp[j]
                if dx != 0.0:
                    t = (xi - xp[j]) / dx
                    out1[i] = yp1[j] + t * (yp1[j + 1] - yp1[j])
                    out2[i] = yp2[j] + t * (yp2[j + 1] - yp2[j])
                    out3[i] = yp3[j] + t * (yp3[j + 1] - yp3[j])
                    out4[i] = yp4[j] + t * (yp4[j + 1] - yp4[j])
                else:
                    out1[i] = yp1[j]
                    out2[i] = yp2[j]
                    out3[i] = yp3[j]
                    out4[i] = yp4[j]
    else:
        for i in range(n_x):
            xi = x[i]
            if xi <= x_lo:
                if extrap:
                    d = xi - x_lo
                    out1[i] = yp1[0] + d * sl1
                    out2[i] = yp2[0] + d * sl2
                    out3[i] = yp3[0] + d * sl3
                    out4[i] = yp4[0] + d * sl4
                else:
                    out1[i] = yp1[0]
                    out2[i] = yp2[0]
                    out3[i] = yp3[0]
                    out4[i] = yp4[0]
            elif xi >= x_hi:
                if extrap:
                    d = xi - x_hi
                    out1[i] = yp1[-1] + d * sr1
                    out2[i] = yp2[-1] + d * sr2
                    out3[i] = yp3[-1] + d * sr3
                    out4[i] = yp4[-1] + d * sr4
                else:
                    out1[i] = yp1[-1]
                    out2[i] = yp2[-1]
                    out3[i] = yp3[-1]
                    out4[i] = yp4[-1]
            else:
                lo, hi = 0, n_xp - 1
                while hi - lo > 1:
                    mid = (lo + hi) >> 1
                    if xp[mid] <= xi:
                        lo = mid
                    else:
                        hi = mid
                dx = xp[hi] - xp[lo]
                if dx != 0.0:
                    t = (xi - xp[lo]) / dx
                    out1[i] = yp1[lo] + t * (yp1[hi] - yp1[lo])
                    out2[i] = yp2[lo] + t * (yp2[hi] - yp2[lo])
                    out3[i] = yp3[lo] + t * (yp3[hi] - yp3[lo])
                    out4[i] = yp4[lo] + t * (yp4[hi] - yp4[lo])
                else:
                    out1[i] = yp1[lo]
                    out2[i] = yp2[lo]
                    out3[i] = yp3[lo]
                    out4[i] = yp4[lo]

    return out1, out2, out3, out4
