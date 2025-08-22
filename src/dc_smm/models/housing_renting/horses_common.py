import numpy as np
from scipy.interpolate import interp1d
from numba import njit, cuda
from numba.typed import Dict    
from typing import Callable, Literal   # NEW  – remove if unused        # NEW
import time
from functools import lru_cache
import math


@njit
def bellman_obj(a_nxt, w_val, H_val, beta, delta,
                a_grid, V_slice, u_func):
    """Objective function for the Bellman equation maximization.

    Calculates the value of choosing next-period assets `a_nxt`, given
    current wealth `w_val`, housing services `H_val`, and continuation
    value function `V_slice`.

    Args:
        a_nxt (float): Candidate for next-period assets (cntn/post-state)
        w_val (float): Current period wealth (cash-on-hand).
        H_val (float): Current period housing services.
        beta (float): Discount factor.
        delta (float): Present-bias parameter.
        a_grid (np.ndarray): Grid for next-period assets (cntn/post-state)
        V_slice (np.ndarray): Slice of the cntn value function
                              corresponding to cntn H_val and income state.
        u_func (callable): Utility function u(c, H_nxt).

    Returns:
        float: The value of the Bellman equation for the given `a_nxt`.
               Returns -np.inf if consumption is non-positive.
    """
    c = w_val - a_nxt
    if c <= 0.0:
        return -np.inf

    V_nxt = interp_as(a_grid, V_slice, np.array([a_nxt]), extrap=True)[0]

    return u_func(c, H_val) + beta * delta * V_nxt

@njit
def piecewise_gradient_3rd(f, x, m_bar, eps=0.9):
    """
    Compute piecewise gradients using up to 3rd-order finite differences.
    
    This function automatically detects segments by identifying jumps and uses
    the highest-order accurate finite difference scheme available within each segment.
    Enforces 0 < gradient <= 1 (gradient > 1 indicates a jump). Segments are 
    non-overlapping with exclusive boundaries: [i,j), [j,k), [k,l), etc.

    Parameters
    ----------
    f, x : 1-D ndarrays (same length, x strictly increasing)
    m_bar: float   – jump threshold in *slope* space (typically 1.0 for MPC)
    eps   : float  – fallback slope if no positive neighbour exists

    Returns
    -------
    g : 1-D ndarray, slope at each x[i] with 0 < g[i] <= 1
    """
    n = len(x)
    g = np.empty(n)
    
    # Step 1: Identify segment boundaries by detecting jumps (slope > 1 or < 0)
    # Segments are [start, end) - start is inclusive, end is exclusive
    segment_boundaries = np.zeros(n+1, dtype=np.int64)
    segment_boundaries[0] = 0
    n_segments = 1
    
    # Use 1.0 as the upper threshold - slopes > 1 indicate jumps (violate MPC bounds)
    jump_threshold = min(m_bar, 1.0)
    
    for i in range(1, n):
        # Check if derivative would exceed 1 or be negative (indicates jump)
        local_slope = (f[i] - f[i-1]) / (x[i] - x[i-1])
        if local_slope > jump_threshold or local_slope < 0:
            segment_boundaries[n_segments] = i
            n_segments += 1
    segment_boundaries[n_segments] = n
    n_segments += 1
    
    # Step 2: Calculate derivatives within each continuous segment using highest-order scheme
    for seg_idx in range(n_segments - 1):
        start = segment_boundaries[seg_idx]
        end = segment_boundaries[seg_idx + 1]
        seg_len = end - start
        
        if seg_len == 1:
            # Single point segment - use nearest neighbor or fallback
            if seg_idx > 0 and start > 0:
                # Use slope from previous segment's end
                g[start] = (f[start] - f[start-1]) / (x[start] - x[start-1])
            elif seg_idx < n_segments - 2 and end < n:
                # Use slope to next segment's start
                g[start] = (f[end] - f[start]) / (x[end] - x[start])
            else:
                g[start] = eps
            # Ensure positive
            if g[start] <= 0:
                g[start] = eps
            continue
        
        # For multi-point segments, use highest-order scheme possible
        for i in range(start, end):
            # Determine how many points we have available in this segment
            points_left = i - start
            points_right = end - i - 1
            
            # Try 3rd order (5-point Richardson) if we have enough points
            if points_left >= 2 and points_right >= 2:
                # 5-point Richardson: O(h^4) accuracy
                # g = (4*D1 - D2)/3 where D1 uses ±1 points, D2 uses ±2 points
                h1 = x[i+1] - x[i-1]
                D1 = (f[i+1] - f[i-1]) / h1
                
                h2 = x[i+2] - x[i-2]
                D2 = (f[i+2] - f[i-2]) / h2
                
                g[i] = (4.0 * D1 - D2) / 3.0
                # Clip to [0, 1] range
                if g[i] > 1.0:
                    g[i] = 1.0
                elif g[i] < 0:
                    g[i] = eps
                
            # Try 2nd order at segment boundaries (3-point one-sided)
            elif i == start and seg_len >= 3:
                # 3-point forward difference at segment start: O(h^2)
                # f'(x) = (-3f(x) + 4f(x+h) - f(x+2h)) / 2h
                h = x[start+1] - x[start]
                h2 = x[start+2] - x[start]
                # Use actual grid spacing for non-uniform grids
                a0 = -h2 / (h * (h2 - h))
                a1 = h2 / (h * h2)
                a2 = -h / (h2 * (h2 - h))
                g[i] = a0*f[start] + a1*f[start+1] + a2*f[start+2]
                
            elif i == end - 1 and seg_len >= 3:
                # 3-point backward difference at segment end: O(h^2)
                h = x[end-1] - x[end-2]
                h2 = x[end-1] - x[end-3]
                # Use actual grid spacing for non-uniform grids
                a0 = h / (h2 * (h2 - h))
                a1 = -h2 / (h * h2)
                a2 = h2 / (h * (h2 - h))
                g[i] = a0*f[end-3] + a1*f[end-2] + a2*f[end-1]
                
            # Try 2nd order centered (3-point) if we have neighbors
            elif points_left >= 1 and points_right >= 1:
                # Standard central difference: O(h^2)
                g[i] = (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
                
            # Fall back to 1st order at edges
            elif i == start:
                # Forward difference at segment start: O(h)
                if seg_len >= 2:
                    g[i] = (f[start+1] - f[start]) / (x[start+1] - x[start])
                else:
                    g[i] = eps
                    
            elif i == end - 1:
                # Backward difference at segment end: O(h)
                g[i] = (f[i] - f[i-1]) / (x[i] - x[i-1])
                
            else:
                # Should not reach here, but use central difference as fallback
                g[i] = (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
            
            # Enforce slope in (0, 1] range
            if g[i] <= 0 or g[i] > 1.0 or np.isnan(g[i]):
                # Try lower-order schemes
                if i < end - 1:
                    g_forward = (f[i+1] - f[i]) / (x[i+1] - x[i])
                    if 0 < g_forward <= 1.0:
                        g[i] = g_forward
                        continue
                        
                if i > start:
                    g_backward = (f[i] - f[i-1]) / (x[i] - x[i-1])
                    if 0 < g_backward <= 1.0:
                        g[i] = g_backward
                        continue
                
                # Last resort: use fallback or clip
                if g[i] > 1.0:
                    g[i] = 1.0
                elif g[i] <= 0 or np.isnan(g[i]):
                    g[i] = eps
    
    # Step 3: Final pass to ensure all gradients are in (0, 1] range
    for i in range(n):
        if g[i] <= 0 or g[i] > 1.0 or np.isnan(g[i]):
            # Search for nearest valid gradient in same segment
            best_dist = n
            best_g = eps
            
            # Find which segment i belongs to
            my_segment = -1
            for seg_idx in range(n_segments - 1):
                if i >= segment_boundaries[seg_idx] and i < segment_boundaries[seg_idx + 1]:
                    my_segment = seg_idx
                    break
            
            if my_segment >= 0:
                # Search within same segment first
                seg_start = segment_boundaries[my_segment]
                seg_end = segment_boundaries[my_segment + 1]
                for j in range(seg_start, seg_end):
                    if j != i and 0 < g[j] <= 1.0 and not np.isnan(g[j]):
                        dist = abs(i - j)
                        if dist < best_dist:
                            best_dist = dist
                            best_g = g[j]
            
            # Final clipping to ensure bounds
            if best_g > 1.0:
                best_g = 1.0
            elif best_g <= 0:
                best_g = eps
                
            g[i] = best_g
    
    return g

@njit
def piecewise_gradient(f, x, m_bar, eps=0.9):
    """
    Compute piecewise gradients for a function with discontinuities.
    
    This function identifies continuous segments by detecting jumps and 
    calculates robust derivatives within each segment. Segments are 
    non-overlapping with exclusive boundaries: [i,j), [j,k), [k,l), etc.

    Parameters
    ----------
    f : 1-D ndarray
        Function values on a strictly-increasing grid
    x : 1-D ndarray
        Grid points, same length as f
    m_bar : float
        Threshold to flag a jump in derivative space (max allowed |df/dx|)
    eps : float, optional
        Fallback slope if NO positive slope exists (default: 0.9)

    Returns
    -------
    g : 1-D ndarray
        Positive slope at each x[i], computed segment-wise
    """
    n = len(x)
    g = np.empty(n)
    
    # Step 1: Identify segment boundaries by detecting jumps
    # Segments are [start, end) - start is inclusive, end is exclusive
    segment_boundaries = np.zeros(n+1, dtype=np.int64)
    segment_boundaries[0] = 0
    n_segments = 1
    
    for i in range(1, n):
        # Check if derivative would exceed threshold (indicates jump)
        local_slope = np.abs(f[i] - f[i-1]) / (x[i] - x[i-1])
        if local_slope > m_bar:
            # End current segment at i (exclusive), start new segment at i
            segment_boundaries[n_segments] = i
            n_segments += 1
    segment_boundaries[n_segments] = n
    n_segments += 1
    
    # Step 2: Calculate derivatives within each continuous segment
    for seg_idx in range(n_segments - 1):
        start = segment_boundaries[seg_idx]
        end = segment_boundaries[seg_idx + 1]
        seg_len = end - start
        
        if seg_len == 1:
            # Single point segment - use nearest neighbor or fallback
            if seg_idx > 0 and start > 0:
                # Use slope from previous segment's end
                g[start] = (f[start] - f[start-1]) / (x[start] - x[start-1])
            elif seg_idx < n_segments - 2 and end < n:
                # Use slope to next segment's start
                g[start] = (f[end] - f[start]) / (x[end] - x[start])
            else:
                g[start] = eps
            # Ensure positive
            if g[start] <= 0:
                g[start] = eps
            continue
        
        # For multi-point segments, use robust derivative estimation
        for i in range(start, end):
            if i == start:
                # Forward difference at segment start
                if seg_len >= 2:
                    g[i] = (f[start+1] - f[start]) / (x[start+1] - x[start])
                else:
                    g[i] = eps
            elif i == end - 1:
                # Backward difference at segment end
                g[i] = (f[i] - f[i-1]) / (x[i] - x[i-1])
            else:
                # Central difference in segment interior
                # Use wider stencil if available for more stability
                if i - start >= 2 and end - i >= 2:
                    # 5-point stencil if possible
                    g[i] = (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*(x[i+1] - x[i]))
                else:
                    # Standard central difference
                    g[i] = (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
            
            # Enforce positive slope
            if g[i] <= 0:
                # Try one-sided differences
                if i < end - 1:
                    g_forward = (f[i+1] - f[i]) / (x[i+1] - x[i])
                    if g_forward > 0:
                        g[i] = g_forward
                        continue
                if i > start:
                    g_backward = (f[i] - f[i-1]) / (x[i] - x[i-1])
                    if g_backward > 0:
                        g[i] = g_backward
                        continue
                # Last resort: use segment average or fallback
                g[i] = eps
    
    # Step 3: Final pass to ensure all gradients are positive
    for i in range(n):
        if g[i] <= 0 or np.isnan(g[i]):
            # Search for nearest positive gradient
            best_dist = n
            best_g = eps
            for j in range(n):
                if g[j] > 0 and not np.isnan(g[j]):
                    dist = abs(i - j)
                    if dist < best_dist:
                        best_dist = dist
                        best_g = g[j]
            g[i] = best_g
    
    return g

@njit
def piecewise_gradient_with_segments(f, x, segment_boundaries, eps=0.9):
    """
    Compute piecewise gradients using pre-computed segment boundaries.
    
    This is useful when segment boundaries are known from EGM intersection points.
    Segments are non-overlapping with exclusive boundaries: [i,j), [j,k), [k,l), etc.

    Parameters
    ----------
    f : 1-D ndarray
        Function values on a strictly-increasing grid
    x : 1-D ndarray
        Grid points, same length as f
    segment_boundaries : 1-D ndarray
        Indices where segments begin/end (must include 0 and n).
        Segments are [boundaries[i], boundaries[i+1]) - start inclusive, end exclusive.
    eps : float, optional
        Fallback slope if NO positive slope exists (default: 0.9)

    Returns
    -------
    g : 1-D ndarray
        Positive slope at each x[i], computed segment-wise
    """
    n = len(x)
    g = np.empty(n)
    n_segments = len(segment_boundaries)
    
    # Calculate derivatives within each continuous segment
    for seg_idx in range(n_segments - 1):
        start = segment_boundaries[seg_idx]
        end = segment_boundaries[seg_idx + 1]
        seg_len = end - start
        
        if seg_len == 1:
            # Single point segment - use nearest neighbor or fallback
            if seg_idx > 0 and start > 0:
                # Use slope from previous segment's end
                g[start] = (f[start] - f[start-1]) / (x[start] - x[start-1])
            elif seg_idx < n_segments - 2 and end < n:
                # Use slope to next segment's start
                g[start] = (f[end] - f[start]) / (x[end] - x[start])
            else:
                g[start] = eps
            # Ensure positive
            if g[start] <= 0:
                g[start] = eps
            continue
        
        # For multi-point segments, use robust derivative estimation
        for i in range(start, end):
            if i == start:
                # Forward difference at segment start
                if seg_len >= 2:
                    g[i] = (f[start+1] - f[start]) / (x[start+1] - x[start])
                else:
                    g[i] = eps
            elif i == end - 1:
                # Backward difference at segment end
                g[i] = (f[i] - f[i-1]) / (x[i] - x[i-1])
            else:
                # Central difference in segment interior
                g[i] = (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
            
            # Enforce positive slope
            if g[i] <= 0:
                # Try one-sided differences
                if i < end - 1:
                    g_forward = (f[i+1] - f[i]) / (x[i+1] - x[i])
                    if g_forward > 0:
                        g[i] = g_forward
                        continue
                if i > start:
                    g_backward = (f[i] - f[i-1]) / (x[i] - x[i-1])
                    if g_backward > 0:
                        g[i] = g_backward
                        continue
                # Last resort: use fallback
                g[i] = eps
    
    # Final pass to ensure all gradients are positive
    for i in range(n):
        if g[i] <= 0 or np.isnan(g[i]):
            # Use simple fallback for now
            g[i] = eps
    
    return g

def uniqueEG(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return a Boolean mask that keeps the *highest-value* entry for each
    duplicate grid point.

    The original loop searched duplicates with Python `for` – this vectorised
    rewrite is ~30× faster for O(10³) points:

    1.  `np.lexsort((-values, grid))`  sorts by *grid* ascending and *values*
        descending (via the minus sign).
    2.  `np.unique(..., return_index=True)` returns the first occurrence of
        each grid value → already the max-value element because of the sort
        order.
    """

    # 1. indices that would sort by grid ↑, value ↓
    order = np.lexsort((-values, grid))

    # 2. first index (in *order*) of every new grid point
    _, first_idx = np.unique(grid[order], return_index=True)

    # 3. build mask in original order
    mask = np.zeros_like(grid, dtype=bool)
    mask[order[first_idx]] = True
    return mask

def _safe_interp(x, y, bounds_error=False, fill_value=None):
    """Create a 1D interpolation function with safe handling of edge cases.
    
    Parameters
    ----------
    x : ndarray
        x-coordinates for interpolation
    y : ndarray
        y-coordinates (values) for interpolation
    bounds_error : bool, optional
        Whether to raise error for out-of-bounds queries, by default False
    fill_value : float or None, optional
        Fill value for out-of-bounds queries, by default None
        
    Returns
    -------
    function
        Interpolation function
    """
    if len(x) < 2:
        # Not enough points for interpolation, return constant function
        return lambda x_new: np.full_like(np.asarray(x_new), 
                                         float(y[0]) if len(y) > 0 else 0.0)
    
    if fill_value is None:
        # Extrapolate by default
        fill_value = "extrapolate"
        
    # Create interpolation function
    return interp1d(x, y, bounds_error=bounds_error, 
                   fill_value=fill_value)

def F_id(mover):
    """Create an identity operator for a mover that simply passes the data through.
    
    Parameters
    ----------
    mover : Mover
        The mover to create the identity operator for
        
    Returns
    -------
    function
        The identity operator
    """
    def operator(data):
        """Identity operator that returns the data unchanged."""
        return data
    
    return operator

@njit
def interp_as(x_points: np.ndarray,
              y_points: np.ndarray,
              x_query: np.ndarray,
              extrap: bool = False) -> np.ndarray:
    """
    Fast 1‑D interpolation (Numba‑jitted).

    Parameters
    ----------
    x_points : ndarray
        Strictly increasing grid of x‑coordinates.
    y_points : ndarray
        Function values at ``x_points``.
    x_query : ndarray
        Points at which to evaluate the interpolant.
    extrap : bool, default False
        * True  – linear extrapolation beyond the grid.
        * False – constant (flat) extension; no NaNs are returned.

    Returns
    -------
    ndarray
        Interpolated (or extrapolated/extended) values at ``x_query``.
    """
    n_query = x_query.size
    yq = np.empty(n_query)

    x_min, x_max = x_points[0],  x_points[-1]
    y_min, y_max = y_points[0],  y_points[-1]

    # Pre‑compute boundary slopes for linear extrapolation
    if x_points.size >= 2:
        slope_left  = (y_points[1]    - y_min) / (x_points[1]   - x_min)
        slope_right = (y_max          - y_points[-2]) / (x_max - x_points[-2])
    else:                         # degenerate (single point) grid
        slope_left = slope_right = 0.0

    for i in range(n_query):
        x = x_query[i]

        # ---------- Left of the grid ----------
        if x <= x_min:
            if extrap:
                yq[i] = y_min + slope_left * (x - x_min)   # linear
            else:
                yq[i] = y_min                              # constant
            continue

        # ---------- Right of the grid ----------
        if x >= x_max:
            if extrap:
                yq[i] = y_max + slope_right * (x - x_max)  # linear
            else:
                yq[i] = y_max                              # constant
            continue

        # ---------- Inside the grid : binary search ----------
        left, right = 0, x_points.size - 1
        while right - left > 1:
            mid = (left + right) // 2
            if x_points[mid] <= x:
                left = mid
            else:
                right = mid

        x_L, x_R = x_points[left],  x_points[right]
        y_L, y_R = y_points[left], y_points[right]
        t = (x - x_L) / (x_R - x_L)
        yq[i] = y_L + t * (y_R - y_L)

    return yq

@njit
def fast_vectorized_interpolation(values_grid, policies_grid, wealth_grid, valid_mask=None):
    """Fast vectorized interpolation using binary search for multiple points.
    
    This function is designed to replace the loop over valid indices in horses_ownh.py.
    
    Parameters
    ----------
    values_grid : ndarray
        1D grid of x values to interpolate from
    policies_grid : ndarray
        1D grid of y values to interpolate from (corresponsing to values_grid)
    wealth_grid : ndarray
        2D grid of x values to evaluate at
    valid_mask : ndarray, optional
        Boolean mask of points to evaluate, by default None (evaluate all)
    
    Returns
    -------
    tuple
        (values, valid_count) - interpolated values and count of valid points
    """
    n_a, n_h = wealth_grid.shape
    output = np.full_like(wealth_grid, -np.inf)
    valid_count = 0
    
    # Ensure we have a valid mask
    if valid_mask is None:
        valid_mask = np.ones_like(wealth_grid, dtype=np.bool_)
    
    # Get boundary values
    x_min = values_grid[0]
    x_max = values_grid[-1]
    y_min = policies_grid[0]
    y_max = policies_grid[-1]
    
    # Process each point in parallel
    for i in range(n_a):
        for j in range(n_h):
            if valid_mask[i, j]:
                valid_count += 1
                x = wealth_grid[i, j]
                
                # Handle out-of-bounds with extrapolation
                if x <= x_min:
                    output[i, j] = y_min
                    continue
                elif x >= x_max:
                    output[i, j] = y_max
                    continue
                
                # Binary search to find the right segment
                left = 0
                right = len(values_grid) - 1
                
                while right - left > 1:
                    mid = (left + right) // 2
                    if values_grid[mid] <= x:
                        left = mid
                    else:
                        right = mid
                
                # Linear interpolation
                x_left = values_grid[left]
                x_right = values_grid[right]
                y_left = policies_grid[left]
                y_right = policies_grid[right]
                
                # Apply interpolation
                if x_right > x_left:  # Avoid division by zero
                    t = (x - x_left) / (x_right - x_left)
                    output[i, j] = y_left + t * (y_right - y_left)
                else:
                    output[i, j] = y_left
    
    return output, valid_count 


@njit
def _egm_preprocess_core(e_old, vf_old, c_old, a_old,
                         vf_next,              # 1-D, same length as old grids
                         beta, u_func,         # u_func must be @njit-able
                         m_bar,                # jump threshold (not currently used)
                         n_con,                # # constraint nodes
                         n_con_nxt,            # # nodes per jump
                         c_max, h_nxt,         # upper end of [c*, c_max] and housing
                         add_jump_constraints): # whether to add jump constraints
    """
    Returns new (e,vf,c,a) with:
        • n_con borrowing-constraint points, plus
        • (optionally) n_con_nxt points on every jump in endogenous grid,

    all prepended in one shot. No np.concatenate used.
    
    Parameters
    ----------
    e_old, vf_old, c_old, a_old : ndarray
        Original EGM solution arrays
    vf_next : ndarray
        Next period value function (1-D, same length as old grids)
    beta : float
        Discount factor
    u_func : callable
        Utility function (must be @njit-able)
    m_bar : float
        Jump threshold (kept for compatibility, not currently used)
    n_con : int
        Number of borrowing constraint nodes
    n_con_nxt : int
        Number of nodes to add per jump
    c_max : float
        Upper bound on consumption (not currently used)
    h_nxt : float
        Next period housing
    add_jump_constraints : bool
        Whether to add jump constraint points (should be False when delta_pb == 1)
    """

    # ---- 0. Basic sizes and jump detection ----
    n_old = e_old.size
    
    # Initialize jump tracking variables
    n_jump = 0
    n_add = n_con  # Always add borrowing constraint nodes
    
    # Only detect and process jumps if flag is True and n_con_nxt > 0
    if add_jump_constraints and n_con_nxt > 0:
        # Detect jumps in endogenous grid and value function
        e_diff = e_old[1:] - e_old[:-1]
        vf_diff = vf_next[1:] - vf_next[:-1]
        
        # Case 1: Negative jumps in endogenous grid
        jumps_case_1 = e_diff < 0
        j_idx_case_1 = np.where(jumps_case_1)[0]
        n_jump_case_1 = j_idx_case_1.size
        
        # Case 2: Negative jumps in value function
        jumps_case_2 = vf_diff < 0
        j_idx_case_2 = np.where(jumps_case_2)[0]
        n_jump_case_2 = j_idx_case_2.size
        n_jump_case_2= 0    

        # temporry turn off case 2
        #jumps_case_2 = np.zeros_like(jumps_case_2, dtype=np.bool_)
        
        # Total jumps and nodes to add
        n_jump = n_jump_case_1 + n_jump_case_2
        n_add = n_con + n_jump * n_con_nxt
    
    n_total = n_old + n_add

    # ---- 1. Allocate output containers ----
    e_new = np.empty(n_total, dtype=e_old.dtype)
    vf_new = np.empty(n_total, dtype=vf_old.dtype)
    c_new = np.empty(n_total, dtype=c_old.dtype)
    a_new = np.empty(n_total, dtype=a_old.dtype)

    p = 0  # Write pointer into the new arrays
    
    # ---- 2. Borrowing-constraint segment (always first) ----
    min_c = np.min(e_old)
    c_con = np.linspace(1e-100, min_c, n_con)
    e_con = c_con  # m = c at the constraint
    vf_con = u_func(c_con, h_nxt) + beta * vf_next[0]
    a_con = np.empty_like(c_con)
    a_con.fill(a_old[0])  # Borrowing limit

    e_new[p:p+n_con] = e_con
    vf_new[p:p+n_con] = vf_con
    c_new[p:p+n_con] = c_con
    a_new[p:p+n_con] = a_con
    p += n_con

    # ---- 3. Jump segments (only if flag is True) ----
    if add_jump_constraints and n_con_nxt > 0:
        # Process Case 1 jumps (negative jumps in endogenous grid)
        for k in j_idx_case_1:
            a_star = a_old[k+1]
            c_star = c_old[k+1]
            
            # Create consumption segment approaching c_star from below
            lb_c = max(1e-10, c_star - 5)
            c_seg = np.linspace(lb_c, c_star, n_con_nxt).astype(c_old.dtype)
            m_seg = a_star + c_seg
            vf_seg = u_func(c_seg, h_nxt) + beta * vf_next[k+1]

            e_new[p:p+n_con_nxt] = m_seg
            vf_new[p:p+n_con_nxt] = vf_seg
            c_new[p:p+n_con_nxt] = c_seg
            a_new[p:p+n_con_nxt] = a_star
            p += n_con_nxt
        
        """" 
        # Process Case 2 jumps (negative jumps in value function)
        for k in j_idx_case_2:
            a_star = a_old[k]
            c_star = c_old[k]
            
            # Create consumption segment extending from c_star
            c_seg = np.linspace(c_star, c_star + 5, n_con_nxt).astype(c_old.dtype)
            m_seg = a_star + c_seg
            vf_seg = u_func(c_seg, h_nxt) + beta * vf_next[k]

            e_new[p:p+n_con_nxt] = m_seg
            vf_new[p:p+n_con_nxt] = vf_seg
            c_new[p:p+n_con_nxt] = c_seg
            a_new[p:p+n_con_nxt] = a_star
            p += n_con_nxt
        """

    # ---- 4. Copy the original solution after all extras ----
    e_new[n_add:] = e_old
    vf_new[n_add:] = vf_old
    c_new[n_add:] = c_old
    a_new[n_add:] = a_old

    ## return sorted arrays
    sort_idx = np.argsort(e_new)
    e_new = e_new[sort_idx]
    vf_new = vf_new[sort_idx]
    c_new = c_new[sort_idx]
    a_new = a_new[sort_idx]

    return e_new, vf_new, c_new, a_new

def egm_preprocess(egrid, vf, c, a,
                   beta, u_func, vf_next,
                   m_bar,
                   n_con=10,
                   n_con_nxt=0,
                   c_max=None,
                   h_nxt=None,
                   add_jump_constraints=True,  # New parameter
                   **kwargs):
    """
    Wrapper that preprocesses EGM solution by:
      • Adding borrowing constraint points
      • Optionally adding jump constraint points  
      • Removing duplicates with uniqueEG
      • Returning cleaned arrays

    Parameters
    ----------
    egrid : ndarray
        Endogenous grid (cash-on-hand)
    vf : ndarray
        Value function on egrid
    c : ndarray
        Consumption policy on egrid
    a : ndarray
        Asset policy on egrid
    beta : float
        Discount factor
    u_func : callable
        Utility function (must be @njit-able)
    vf_next : ndarray
        Next period value function
    m_bar : float
        Jump threshold (kept for compatibility)
    n_con : int, optional
        Number of borrowing constraint nodes (default: 10)
    n_con_nxt : int, optional
        Number of nodes per jump (default: 0)
    c_max : float, optional
        Upper bound on consumption (default: 1.05 * max(c))
    h_nxt : float, optional
        Next period housing
    add_jump_constraints : bool, optional
        Whether to add jump constraint points (default: True)
        Should be False when delta_pb == 1.0
    **kwargs
        Additional arguments (ignored for compatibility)
    
    Returns
    -------
    tuple
        (e_out, vf_out, c_out, a_out) - Preprocessed and cleaned arrays
    """

    # Choose a default c_max if not specified
    if c_max is None:
        c_max = 1.05 * np.max(c)  # 5% above current max consumption

    # remove points where egrid is negative
    #valid_mask = egrid >= 0
    #egrid = egrid[valid_mask]
    #vf = vf[valid_mask]
    #c = c[valid_mask]
    #a = a[valid_mask]

    # Run the fast core with jump constraint flag
    e_cat, vf_cat, c_cat, a_cat = _egm_preprocess_core(
        egrid, vf, c, a,
        vf_next, beta, u_func,
        m_bar, n_con, n_con_nxt, c_max, h_nxt,
        add_jump_constraints)

    # Remove duplicates based on endogenous grid and value function
    unique_ids = uniqueEG(e_cat, vf_cat)

    e_out = e_cat[unique_ids]
    vf_out = vf_cat[unique_ids]
    c_out = c_cat[unique_ids]
    a_out = a_cat[unique_ids]

    return e_out, vf_out, c_out, a_out


def build_njit_utility(
    expr: str,
    params: Dict[str, float],
    h_placeholder: str = "H_nxt",
    arg1_name: str = "c",
    arg2_name: str = "H",
) -> Callable[[float, float], float]:
    """
    Compile a two-argument utility u(c, H) that is Numba nopython.

    Parameters
    ----------
    expr : str
        Raw expression, e.g. "alpha*np.log(c)+(1-alpha)*np.log(kappa*(H_nxt+iota))"
    params : dict
        Literal parameter values referenced in *expr* (alpha, kappa, …).
    h_placeholder : str, optional
        Token for housing inside *expr* (default "H_nxt").
    arg1_name : str, optional
        The name for the first argument of the compiled function (default "c").
    arg2_name : str, optional
        The name for the second argument of the compiled function (default "H").

    Returns
    -------
    callable
        nopython-compiled function u(c, H) → float
    """

    # 1.  replace the placeholder with the specified run-time variable name
    patched = expr.replace(h_placeholder, arg2_name)

    # 2.  build source code for a pure Python function with dynamic arg names
    func_src = f"def _u({arg1_name}, {arg2_name}):\n    return " + patched

    # 3.  execute in a tiny namespace containing numpy and constants
    ns = {"np": np, **params}
    exec(func_src, ns)          # defines _u in ns
    py_func = ns["_u"]

    # 4.  JIT-compile to nopython; result takes (c, H) positional args
    return njit(py_func)

@lru_cache(maxsize=None)          # one compiled version per (expr, frozenset(params))
def build_njit_utility_cached(expr, params_frozen, h_placeholder="H_nxt"):
    params = dict(params_frozen)  # thaw for substitution
    return build_njit_utility(expr, params, h_placeholder)

def get_u_func(expr_str, param_vals):
    frozen = tuple(sorted(param_vals.items()))          # hashable
    return build_njit_utility_cached(expr_str, frozen)

# ======================================================================
#  GPU Device Functions
# ======================================================================

@cuda.jit(device=True)
def searchsorted_gpu(a, v):
    """
    A GPU-compatible binary search implementation, equivalent to
    np.searchsorted(a, v, side='right').
    """
    lower_bound = 0
    upper_bound = len(a)
    while lower_bound < upper_bound:
        # Use bit shift for faster division by 2
        i = lower_bound + ((upper_bound - lower_bound) >> 1)
        if a[i] < v:
            lower_bound = i + 1
        else:
            upper_bound = i
    return lower_bound

@cuda.jit(device=True)
def interp_gpu(x_new, x_old, y_old):
    """
    A simple linear interpolation function that is compatible with Numba's
    CUDA target.
    """
    i = searchsorted_gpu(x_old, x_new)
    
    # Handle edges
    if i == 0:
        return y_old[0]
    if i >= len(x_old):
        return y_old[-1]

    # Linear interpolation formula
    x0, x1 = x_old[i - 1], x_old[i]
    y0, y1 = y_old[i - 1], y_old[i]
    
    # Avoid division by zero if grid points are not unique
    if x1 == x0:
        return y0
    
    return y0 + (y1 - y0) * (x_new - x0) / (x1 - x0)

@cuda.jit(device=True)
def u_func_gpu_crra(c, H, alpha, kappa, iota):
    """GPU device version of the CRRA utility function."""
    if c <= 0:
        return -1e12
    return (c**(1 - alpha) / (1 - alpha)) * (H**kappa * iota)

@cuda.jit(device=True)
def u_func_gpu_log(c, H, alpha, kappa, iota):
    """GPU device version of the log-utility function."""
    if c <= 0:
        return -1e12
    return alpha * math.log(c) + (1 - alpha) * math.log(kappa * H + iota)

@cuda.jit(device=True)
def bellman_obj_gpu(a_prime, w, H, beta, delta, a_grid, V_next,
                    h_nxt_ind, y_ind,
                    alpha, kappa, iota):
    """
    GPU device version of the Bellman object function (CRRA only).
    """
    c = w - a_prime
    if c <= 0:
        return -1e110

    v_next_slice = V_next[:, h_nxt_ind, y_ind]
    v_interp = interp_gpu(a_prime, a_grid, v_next_slice)
    
    util = u_func_gpu_log(c, H, alpha, kappa, iota)
    
    return util + beta * delta * v_interp

# --- NEW FUNCTIONS FOR EULER ERROR CALCULATION GPU LOG UTILITY ---

@cuda.jit(device=True)
def uc_owner_gpu(c, H, alpha):
    """
    GPU device function for the owner's marginal utility.
    """
    if c <= 0: 
        return 1e112
    return alpha/c

@cuda.jit(device=True)
def uc_renter_gpu(c, S, alpha):
    """
    GPU device function for the renter's marginal utility (H represents S).
    """
    if c <= 0: 
        return 1e112
    return alpha/c
    
@cuda.jit(device=True)
def inv_uc_owner_gpu(lambda_e, H, alpha):
    """
    GPU device function for the owner's inverse marginal utility.
    """
    if lambda_e <= 0: 
        return 1e-12
    return alpha/lambda_e