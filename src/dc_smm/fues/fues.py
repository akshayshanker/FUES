"""Fast Upper-Envelope Scan (FUES) Algorithm Implementation

This module implements the FUES algorithm from Dobrescu & Shanker (2025) for solving 
discrete-continuous dynamic programming problems using Carroll's endogenous grid method.

Key Concepts:
- The upper envelope consists of segments from different "future" choice-specific value functions
- Points on the envelope are identified using local convexity tests
- A single linear scan (O(n) complexity) suffices to find all envelope points

Algorithm Overview:
1. Sort endogenous grid points and associated values/policies
2. Scan through points testing for three cases:
   - Case A: Right-turn with jump → potential drop (requires forward validation)
   - Case B: Value fall → always drop
   - Case C: Left-turn or right-turn without jump → check via backward scan
3. Optionally compute intersection points where value functions cross

Performance optimizations:
- Constants for epsilon values to avoid repeated allocations
- Reciprocals used in hot loop to replace expensive divisions
- Circular buffer for efficient backward scanning
- Pre-allocated arrays for intersection tracking
"""

from numba import njit
import numpy as np

# Constants for better performance
EPS_D = 1e-12 # Epsilon for division protection (updated for numerical stability)
EPS_A = 1e-12  # Epsilon for gradient calculations
EPS_SEP = 1e-100# Epsilon for intersection separation
EPS_fwd_back = (40/2000)*10
PARALLEL_GUARD = 1e-12 # Guard for parallel line detection

TURN_LEFT = 1; TURN_RIGHT = 0
JUMP_YES = 1; JUMP_NO = 0

# ---------------------------------------------------------------------
# Helpers that remain identical ---------------------------------------
# ---------------------------------------------------------------------

@njit(inline="always")
def _between_open(x, lo, hi, eps):
    """Require x strictly inside (lo, hi) with a safety margin eps"""
    if lo > hi:
        lo, hi = hi, lo
    return (x > lo + eps*100) and (x < hi - eps*100)


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
    e_lo, e_hi, eps
):
    """
    Force a crossing point strictly inside (e_lo, e_hi).
    
    IMPORTANT: This function ALWAYS returns a valid intersection point.
    If lines are parallel or intersection is outside interval, it returns
    the midpoint of the interval.
    
    Parameters:
    -----------
    L_x1, L_y1, L_x2, L_y2: Coordinates of left line segment endpoints
    R_x1, R_y1, R_x2, R_y2: Coordinates of right line segment endpoints
    e_lo, e_hi: Interval bounds - must have intersection inside (e_lo, e_hi)
    eps: Safety margin (use EPS_SEP = 1e-5)
    
    Returns:
    --------
    (x, y): Intersection point GUARANTEED inside (e_lo, e_hi)
            Never returns (nan, nan)
    """
    # Step 1: Compute slopes of both lines
    denom_L = max(EPS_D, L_x2 - L_x1)  # EPS_D = 1e-12 from line 30
    denom_R = max(EPS_D, R_x2 - R_x1)
    sL = (L_y2 - L_y1) / denom_L
    sR = (R_y2 - R_y1) / denom_R
    
    # Step 2: Check if lines are parallel
    denom = sL - sR
    
    if np.abs(denom) < PARALLEL_GUARD:  # PARALLEL_GUARD = 1e-12 from line 34
        # Lines are parallel - use midpoint between j and i+1
        # This is the natural crossing point when lines have same slope
        x = 0.5 * (e_lo + e_hi)
    else:
        # Lines intersect - compute analytical intersection
        # We solve: L_y1 + sL*(x - L_x1) = R_y1 + sR*(x - R_x1)
        # Rearranging: x = (R_y1 - L_y1 + sL*L_x1 - sR*R_x1) / (sL - sR)
        num = (R_y1 - L_y1) + sL * L_x1 - sR * R_x1
        x = num / denom
        
        # Step 3: If intersection is outside interval, use midpoint instead of clipping
        if x <= e_lo + eps or x >= e_hi - eps:
            x = 0.5 * (e_lo + e_hi)
    
    # Step 4: Compute y at intersection point
    # If lines actually intersect at x, yL and yR should be equal (or very close)
    # Use average to handle numerical precision issues
    yL = L_y1 + sL * (x - L_x1)
    yR = R_y1 + sR * (x - R_x1)
    y = 0.5 * (yL + yR)  # Use average of both lines at intersection
    
    return (x, y)


@njit
def uniqueEG(egrid, vf):
    egrid_rounded = np.round_(egrid, 10)
    unique_vals = np.unique(egrid_rounded)
    keep = np.full_like(egrid, False, dtype=np.bool_)
    for val in unique_vals:
        if np.isnan(val):
            continue
        idx = np.where(egrid_rounded == val)[0]
        keep[idx[np.argmax(vf[idx])]] = True
    return keep


# ---------------- Circular buffer utilities --------------------------


@njit
def circ_put(buf, head, value):
    """Write *value* at *head* position, return new head index."""
    buf[head] = value
    return (head + 1) % buf.size


# ---------------- Segment intersection (unchanged) -------------------


@njit
def linear_interp(x, x1, x2, y1, y2):
    """Linear interpolation helper."""
    if np.abs(x2 - x1) < EPS_D:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


@njit(inline="always")
def seg_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, EPS2=1):
    """Find intersection point of two line segments.
    
    Returns the intersection point as a tuple (x, y) if segments intersect within their bounds,
    otherwise returns (nan, nan).
    
    Parameters:
    ax1, ay1, ax2, ay2: coordinates of endpoints of first segment
    bx1, by1, bx2, by2: coordinates of endpoints of second segment
    """
    da_x, da_y = ax2 - ax1, ay2 - ay1
    db_x, db_y = bx2 - bx1, by2 - by1
    dp_x, dp_y = ax1 - bx1, ay1 - by1
    dap_x, dap_y = -da_y, da_x
    denom = dap_x * db_x + dap_y * db_y
    if denom == 0.0:
        return (np.nan, np.nan)
    
    # Parameter t for segment b (from b1 to b2)
    t = (dap_x * dp_x + dap_y * dp_y) / denom
    
    # Check if intersection point is within segment b bounds
    if t < 0.0 -EPS2 or t > 1.0 + EPS2:
        return (np.nan, np.nan)
    
    # Calculate parameter s for segment a (from a1 to a2)
    # We need to solve: a1 + s*(a2-a1) = b1 + t*(b2-b1)
    # This gives us s from either x or y coordinate (use the one with larger denominator for stability)
    if abs(da_x) > abs(da_y):
        s = (bx1 + t * db_x - ax1) / da_x
    else:
        s = (by1 + t * db_y - ay1) / da_y
    
    # Check if intersection point is within segment a bounds
    if s < 0.0 -EPS2 or s > 1.0 + EPS2:
        return (np.nan, np.nan)
    
    # Return the intersection point
    return (t * db_x + bx1, t * db_y + by1)


# ---------------- Intersection helpers -------------------


@njit
def add_intersection(
    intersections,
    n_inter,
    intr_x,
    intr_y,
    e_grid,
    a_prime,
    policy_2,
    del_a,
    idx1,
    idx2,
    idx3,
    idx4,
):
    """Add two intersection points to the 2D array - one for each policy branch.

    idx1, idx2: indices for the left branch (old branch)
    idx3, idx4: indices for the right branch (new branch)

    Returns updated n_inter and the interpolated values for the intersection.
    
    Call Sites Summary:
    -------------------
    1. Case A (lines ~732-735): Right-turn jump (when i+1 is kept)
       - Left: j to idx_f (old branch continuing)
       - Right: idx_b to i+1 (new branch jumping in)
    
    2. Case C.1 (lines ~855-858): Left turn with j dropped
       - Left: j to idx_f (old branch being dropped)
       - Right: idx_b to i+1 (new branch taking over)
    
    3. Case C.2 (lines ~917-920): Left turn with j kept
       - Left: j to idx_fwd (old branch continuing)
       - Right: idx_back to i+1 (new branch crossing)
    
    The pattern is consistent: old branch on left (lower e_grid after intersection),
    new branch on right (higher e_grid after intersection).
    """
    if not np.isnan(intr_x) and n_inter + 1 < intersections.shape[0]:
        # Left branch point (slightly before intersection)
        intersections[n_inter, 0] = intr_x - EPS_SEP  # e_grid
        intersections[n_inter, 1] = intr_y            # value

        # Interpolate policies along left branch (idx1 to idx2)
        t_left = (intr_x - e_grid[idx1]) / max(EPS_D, e_grid[idx2] - e_grid[idx1])
        intersections[n_inter, 2] = a_prime[idx1] + t_left * (a_prime[idx2] - a_prime[idx1])      # policy_1
        intersections[n_inter, 3] = policy_2[idx1] + t_left * (policy_2[idx2] - policy_2[idx1])  # policy_2
        intersections[n_inter, 4] = del_a[idx1] + t_left * (del_a[idx2] - del_a[idx1])      # del_a

        # Right branch point (slightly after intersection)
        intersections[n_inter + 1, 0] = intr_x + EPS_SEP  # e_grid
        intersections[n_inter + 1, 1] = intr_y            # value

        # Interpolate policies along right branch (idx3 to idx4)
        t_right = (intr_x - e_grid[idx3]) / max(EPS_D, e_grid[idx4] - e_grid[idx3])
        intersections[n_inter + 1, 2] = a_prime[idx3] + t_right * (a_prime[idx4] - a_prime[idx3])      # policy_1
        intersections[n_inter + 1, 3] = policy_2[idx3] + t_right * (policy_2[idx4] - policy_2[idx3])  # policy_2
        intersections[n_inter + 1, 4] = del_a[idx3] + t_right * (del_a[idx4] - del_a[idx3])      # del_a

        return n_inter + 2, intr_x, intr_y, intersections[n_inter, 2], intersections[n_inter, 4]

    return n_inter, 0.0, 0.0, 0.0, 0.0


@njit(inline="always")
def line_intersect_unbounded(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """Intersection of the infinite lines through (a1,a2) and (b1,b2).
    
    Returns the intersection point as a tuple (x, y).
    Unlike seg_intersect, this doesn't check if the intersection is within segment bounds.
    """
    da_x, da_y = ax2 - ax1, ay2 - ay1
    db_x, db_y = bx2 - bx1, by2 - by1
    denom = da_x * db_y - da_y * db_x
    if np.abs(denom) < PARALLEL_GUARD:
        return (np.nan, np.nan)
    s = ((bx1 - ax1) * db_y - (by1 - ay1) * db_x) / denom
    return (ax1 + s * da_x, ay1 + s * da_y)


@njit(inline="always")
def check_same_branch(e_grid, a_prime, idx1, idx2, m_bar):
    """Check if two points are on the same branch (no jump between them)."""
    if idx1 < 0 or idx2 < 0 or idx1 >= len(e_grid) or idx2 >= len(e_grid):
        return False
    de = max(EPS_D, e_grid[idx2] - e_grid[idx1])
    g_a = np.abs((a_prime[idx2] - a_prime[idx1]) / de)
    return g_a < m_bar and de < EPS_fwd_back

@njit(inline="always")
def find_safe_extrapolation_point(e_grid, a_prime, base_idx, N, m_bar, forward=True):
    """
    Find a safe point for extrapolation that's on the same branch as base_idx.
    Returns the index of a point that doesn't jump from base_idx, or base_idx if none found.
    """
    if forward:
        # Search forward for a point on same branch
        for offset in range(1, min(4, N - base_idx)):
            test_idx = base_idx + offset
            if test_idx < N and check_same_branch(e_grid, a_prime, base_idx, test_idx, m_bar):
                return test_idx
    else:
        # Search backward for a point on same branch
        for offset in range(1, min(4, base_idx + 1)):
            test_idx = base_idx - offset
            if test_idx >= 0 and check_same_branch(e_grid, a_prime, test_idx, base_idx, m_bar):
                return test_idx
    # If no valid point found, return base_idx (will result in flat extrapolation)
    return base_idx

@njit(inline="always")
def make_pair_from_indices_or_fallback(e, v, a, p2, d, lo_idx, hi_idx, fb_lo, fb_hi, N):
    """
    Returns endpoints (x1,y1,a1,p21,d1) and (x2,y2,a2,p22,d2).
    If lo_idx or hi_idx is -1, uses the fallback pair (fb_lo, fb_hi).
    """
    # Bounds checking for fallback indices
    fb_lo = max(0, min(fb_lo, N-1))
    fb_hi = max(0, min(fb_hi, N-1))
    
    if lo_idx != -1 and hi_idx != -1:
        x1, x2 = e[lo_idx], e[hi_idx]
        y1, y2 = v[lo_idx], v[hi_idx]
        a1, a2 = a[lo_idx], a[hi_idx]
        p21, p22 = p2[lo_idx], p2[hi_idx]
        d1, d2 = d[lo_idx], d[hi_idx]
    else:
        x1, x2 = e[fb_lo], e[fb_hi]
        y1, y2 = v[fb_lo], v[fb_hi]
        a1, a2 = a[fb_lo], a[fb_hi]
        p21, p22 = p2[fb_lo], p2[fb_hi]
        d1, d2 = d[fb_lo], d[fb_hi]
    return x1, y1, a1, p21, d1, x2, y2, a2, p22, d2


@njit
def add_intersection_from_pairs(intersections, n_inter, intr_x, intr_y,
                                L_x1, L_y1, L_a1, L_p21, L_d1, L_x2, L_y2, L_a2, L_p22, L_d2,
                                R_x1, R_y1, R_a1, R_p21, R_d1, R_x2, R_y2, R_a2, R_p22, R_d2):
    """Variant of add_intersection that accepts raw endpoints for left/right pairs."""
    if not np.isnan(intr_x) and n_inter + 1 < intersections.shape[0]:
        # left point (slightly before)
        intersections[n_inter, 0] = intr_x - EPS_SEP
        intersections[n_inter, 1] = intr_y
        denom_L = max(EPS_D, L_x2 - L_x1)
        tL = (intr_x - L_x1) / denom_L  # can be <0 or >1 (extrapolation)
        intersections[n_inter, 2] = L_a1  + tL * (L_a2  - L_a1)
        intersections[n_inter, 3] = L_p21 + tL * (L_p22 - L_p21)
        intersections[n_inter, 4] = L_d1  + tL * (L_d2  - L_d1)

        # right point (slightly after)
        intersections[n_inter+1, 0] = intr_x + EPS_SEP
        intersections[n_inter+1, 1] = intr_y
        denom_R = max(EPS_D, R_x2 - R_x1)
        tR = (intr_x - R_x1) / denom_R
        intersections[n_inter+1, 2] = R_a1  + tR * (R_a2  - R_a1)
        intersections[n_inter+1, 3] = R_p21 + tR * (R_p22 - R_p21)
        intersections[n_inter+1, 4] = R_d1  + tR * (R_d2  - R_d1)

        return n_inter + 2
    return n_inter


@njit
def add_intersection_from_pairs_with_sep(
    intersections, n_inter, intr_x, intr_y, sep,
    L_x1, L_y1, L_a1, L_p21, L_d1, L_x2, L_y2, L_a2, L_p22, L_d2,
    R_x1, R_y1, R_a1, R_p21, R_d1, R_x2, R_y2, R_a2, R_p22, R_d2
):
    """
    Add two intersection points with ADAPTIVE separation.
    
    This function adds TWO points:
    1. Left point at (intr_x - sep) with policies from LEFT branch
    2. Right point at (intr_x + sep) with policies from RIGHT branch
    
    Parameters:
    -----------
    intersections: 2D array shape (max_inter, 5) to store intersection data
    n_inter: Current number of intersection points (will add 2 more)
    intr_x, intr_y: The actual intersection coordinates
    sep: ADAPTIVE separation (not fixed EPS_SEP!)
    L_x1, L_y1, L_a1, L_p21, L_d1: Start point of left branch segment
    L_x2, L_y2, L_a2, L_p22, L_d2: End point of left branch segment
    R_x1, R_y1, R_a1, R_p21, R_d1: Start point of right branch segment
    R_x2, R_y2, R_a2, R_p22, R_d2: End point of right branch segment
    
    Returns:
    --------
    Updated n_inter (should be n_inter + 2 if successful)
    """
    if not np.isnan(intr_x) and n_inter + 1 < intersections.shape[0]:
        # Add LEFT point (slightly before intersection)
        intersections[n_inter, 0] = intr_x - sep  # e_grid coordinate
        intersections[n_inter, 1] = intr_y         # value at intersection
        
        # Interpolate policies from LEFT branch at intersection point
        denom_L = max(EPS_D, L_x2 - L_x1)
        tL = (intr_x - L_x1) / denom_L  # Parameter for interpolation
        intersections[n_inter, 2] = L_a1 + tL * (L_a2 - L_a1)    # a_prime
        intersections[n_inter, 3] = L_p21 + tL * (L_p22 - L_p21) # policy_2
        intersections[n_inter, 4] = L_d1 + tL * (L_d2 - L_d1)    # del_a
        
        # Add RIGHT point (slightly after intersection)
        intersections[n_inter+1, 0] = intr_x + sep  # e_grid coordinate
        intersections[n_inter+1, 1] = intr_y         # value at intersection
        
        # Interpolate policies from RIGHT branch at intersection point
        denom_R = max(EPS_D, R_x2 - R_x1)
        tR = (intr_x - R_x1) / denom_R  # Parameter for interpolation
        intersections[n_inter+1, 2] = R_a1 + tR * (R_a2 - R_a1)    # a_prime
        intersections[n_inter+1, 3] = R_p21 + tR * (R_p22 - R_p21) # policy_2
        intersections[n_inter+1, 4] = R_d1 + tR * (R_d2 - R_d1)    # del_a
        
        return n_inter + 2
    return n_inter


@njit
def backward_scan_combined(
    m_buf,
    m_head,
    LB,
    e_grid,
    vf,
    a_prime,
    i,
    j,
    k,
    i_plus_1,
    left_turn,
    g_tilde_a,
    last_turn_left,
    g_1,
    m_bar,
    check_drop=True,
):
    """Backward scan to find previous point on same branch and check optimality.

    This function searches through recently dropped points (stored in circular buffer)
    to find point m that is on the same branch as i+1 (i.e., policy gradient < m_bar).

    In Case C with left turn, it checks if j should be dropped by comparing gradients:
    - If gradient from m to j < gradient from j to i+1, then j is suboptimal

    Parameters
    ----------
    m_buf : array
        Circular buffer storing indices of recently dropped points
    m_head : int
        Current write position in circular buffer
    LB : int
        Buffer size (lookback limit)
    e_grid, vf, a_prime : arrays
        Grid points, values, and policies
    i, j, i_plus_1 : int
        Indices: i (loop counter), j (last kept), i+1 (current point)
    left_turn : bool
        True if g_1 > g_jm1 (convex turn)
    g_tilde_a : float
        Policy gradient between j and i+1
    last_turn_left : bool
        Whether previous iteration was also a left turn
    g_1 : float
        Value gradient from j to i+1
    m_bar : float
        Jump threshold
    check_drop : bool
        If True, check whether to drop j. If False, only find m.

    Returns
    -------
    keep_j : bool
        Whether to keep point j
    m_ind : int
        Index of point m on same branch as i+1 (-1 if not found)
    """
    keep_j = True
    m_ind = -1

    # Search backwards for last same-branch point (most recent to oldest)
    for t in range(LB):
        idx_buf = (m_head - 1 - t) % LB
        m_idx = m_buf[idx_buf]

        # For Case C (check_drop=True), use original conditions
        if check_drop:
            if m_idx != -1:
                de = max(EPS_A, e_grid[i_plus_1] - e_grid[m_idx])
                g_m_a = np.abs((a_prime[i_plus_1] - a_prime[m_idx]) / de)

                d_idx_b_k =  e_grid[k] - e_grid[m_idx]
                g_k_idx_b = np.abs((vf[k] - vf[m_idx]) / d_idx_b_k)
                if g_m_a < m_bar and de < EPS_fwd_back and g_k_idx_b > m_bar:
                    m_ind = m_idx

                    # g_m_vf already computed with de
                    g_m_vf = np.abs((vf[i_plus_1] - vf[m_idx]) / de)
                    if g_1 > g_m_vf:
                        keep_j = False
                    break
        else:
            # For intersection finding (check_drop=False), use original find_backward_same_branch logic
            if m_idx != -1 and m_idx < i_plus_1:
                de = max(EPS_D, e_grid[i_plus_1] - e_grid[m_idx])
                grad_a = np.abs((a_prime[i_plus_1] - a_prime[m_idx]) / de)

                d_idx_b_k =  e_grid[k] - e_grid[m_idx]
                g_k_idx_b = np.abs((vf[k] - vf[m_idx]) / d_idx_b_k)
                if grad_a < m_bar and de < EPS_fwd_back and g_k_idx_b > m_bar:
                    m_ind = m_idx
                    #if left_turn  
                    break

    return keep_j, m_ind


@njit
def find_forward_same_branch(e_grid, a_prime, start_idx, j_idx, N, LB, m_bar):
    """Find the first point in forward scan that's on same branch.

    Returns found flag and index.
    """
    for f in range(min(LB, N - start_idx - 1)):
        if start_idx + 1 + f >= N:
            break
        de = max(EPS_D, e_grid[start_idx + 1 + f] - e_grid[j_idx])
        g_a = np.abs((a_prime[start_idx + 1 + f] - a_prime[j_idx]) / de)
        if g_a < m_bar and de < EPS_fwd_back:
            return True, start_idx + 1 + f
    return False, -1


@njit
def forward_scan_case_a(e_grid, vf, a_prime, i, j, N, LB, m_bar, g_1):
    """Forward scan validation for Case A (right-turn jump).

    When we detect a right-turn jump, point i+1 might be jumping from a dominated
    branch. This function checks if i+1 should be kept by:
    1. Finding a future point f on the same branch as j
    2. Checking if the value gradient from j to i+1 dominates the gradient from i+1 to f

    If g_1 > g_f (gradient j→i+1 > gradient i+1→f), then i+1 lies above the
    extrapolated line from j to f, so we keep it.

    Parameters
    ----------
    e_grid, vf, a_prime : arrays
        Grid points, values, and policies
    i, j : int
        Indices: i (loop counter), j (last kept point)
    N : int
        Total number of grid points
    LB : int
        Lookback/forward buffer size
    m_bar : float
        Jump threshold for same-branch detection
    g_1 : float
        Value gradient from j to i+1

    Returns
    -------
    keep_i1 : bool
        Whether to keep point i+1
    idx_f : int
        Index of forward point on same branch as j (-1 if not found)
    """
    idx_f = -1
    keep_i1 = False
    found_forward_same_branch = False

    for f in range(LB):
        if i + 2 + f >= N:  # CRITICAL: Add bounds check
            break
        de = max(EPS_D, e_grid[i + 2 + f]-e_grid[j])
        #sde_1 = max(EPS_D, e_grid[i + 1] - e_grid[j])
        g_f_a = np.abs((a_prime[j] - a_prime[i + 2 + f]) / de)
        idx_f = i + 2 + f  # Store actual grid index
        de_jump = max(EPS_D, e_grid[idx_f] - e_grid[i + 1])
        g_jump = np.abs((a_prime[i + 1] - a_prime[idx_f]) / de_jump)
        
        if g_f_a < m_bar and de < EPS_fwd_back and g_jump >= m_bar:
            
            found_forward_same_branch = True
            
            # Compute g_f_vf for this point
            de_1 = max(EPS_D, e_grid[i + 2 + f] - e_grid[j])
            g_f_vf_at_idx = (vf[i + 2 + f] - vf[j]) / de_1
            if g_1 > g_f_vf_at_idx:
                # Only keep i+1 if there's also a jump from i+1 to idx_f
                # Check if gradient from i+1 to idx_f exceeds jump threshold
                #
                #if   # There IS a jump from i+1 to idx_f
                keep_i1 = True
                
                break
    
    #if de< 0.05:
    #    keep_i1 = True
    
    if not found_forward_same_branch:
        keep_i1 = True

    #print("keep_i1", keep_i1)

    return keep_i1, idx_f,found_forward_same_branch


# ---------------------------------------------------------------------
# Public wrapper -------------------------------------------------------
# ---------------------------------------------------------------------


#@njit
def FUES(
    e_grid,
    vf,
    policy_1,
    policy_2,
    del_a,
    b=1e-10,
    m_bar=2.0,
    LB=4,
    endog_mbar=False,
    padding_mbar=0.0,
    include_intersections=True,
):
    """Sort input, call scanner, drop NaNs, return cleaned arrays.

    Parameters:
    -----------
    include_intersections : bool, default True
        If True, intersection points where discrete choices switch are included in output.
        If False, returns only the original upper envelope points.
    debug_intersections : bool, default False
        If True, print debug information for each intersection added.
    debug_e_min, debug_e_max : float, default -inf, +inf
        Only print debug info for intersections with e-coordinate in this range.
    debug_v_min, debug_v_max : float, default -inf, +inf
        Only print debug info for intersections with v-coordinate in this range.
    """

    idx = np.argsort(e_grid)
    e_grid = e_grid[idx]
    vf = vf[idx]
    policy_1 = policy_1[idx]
    policy_2 = policy_2[idx]
    del_a = del_a[idx]

    e_grid_out, keep, intersections = _scan(
        e_grid,
        vf,
        policy_1,
        policy_2,
        del_a,
        m_bar,
        LB,
        endog_mbar,
        padding_mbar,
        include_intersections,
        True,  # not_allow_2lefts - default to True
    )

    # Extract kept points using boolean mask
    env_idx = np.flatnonzero(keep)
    e_kept = e_grid_out[env_idx]
    v_kept = vf[env_idx]
    p1_kept = policy_1[env_idx]
    p2_kept = policy_2[env_idx]
    d_kept = del_a[env_idx]

    if include_intersections:
        # If we have intersections, merge them with kept points
        if intersections.shape[0] > 0:
            n_kept = len(e_kept)
            n_inter = intersections.shape[0]
            n_total = n_kept + n_inter

            # Pre-allocate output arrays (faster than concatenate)
            all_e = np.empty(n_total, dtype=e_kept.dtype)
            all_v = np.empty(n_total, dtype=v_kept.dtype)
            all_p1 = np.empty(n_total, dtype=p1_kept.dtype)
            all_p2 = np.empty(n_total, dtype=p2_kept.dtype)
            all_d = np.empty(n_total, dtype=d_kept.dtype)

            # Fill arrays (memory-efficient copy)
            all_e[:n_kept] = e_kept
            all_e[n_kept:] = intersections[:, 0]  # e_grid
            all_v[:n_kept] = v_kept
            all_v[n_kept:] = intersections[:, 1]  # value
            all_p1[:n_kept] = p1_kept
            all_p1[n_kept:] = intersections[:, 2]  # policy_1
            all_p2[:n_kept] = p2_kept
            all_p2[n_kept:] = intersections[:, 3]  # policy_2
            all_d[:n_kept] = d_kept
            all_d[n_kept:] = intersections[:, 4]  # del_a

            # Sort by e_grid to maintain order
            sort_idx = np.argsort(all_e)
            return (
                all_e[sort_idx],
                all_v[sort_idx],
                all_p1[sort_idx],
                all_p2[sort_idx],
                all_d[sort_idx],
            )

    # Return only kept points (original behavior)
    return (e_kept, v_kept, p1_kept, p2_kept, d_kept)


# ---------------------------------------------------------------------
# Non-jitted wrapper for getting intersections separately ---------------
# ---------------------------------------------------------------------


def FUES_sep_intersect(
    e_grid,
    vf,
    policy_1,
    policy_2,
    del_a,
    b=1e-10,
    m_bar=2.0,
    LB=4,
    endog_mbar=False,
    padding_mbar=0.0,
):
    """
    Non-jitted wrapper that returns FUES results and intersection points separately.
    This is intended for plotting purposes only.

    Returns
    -------
    fues_result : tuple
        Standard FUES output (e_grid, vf, policy_1, policy_2, del_a)
    intersections : tuple
        Intersection points (inter_e, inter_v, inter_p1, inter_p2, inter_d)
    """
    # Sort inputs
    idx = np.argsort(e_grid)
    e_grid_sorted = e_grid[idx]
    vf_sorted = vf[idx]
    policy_1_sorted = policy_1[idx]
    policy_2_sorted = policy_2[idx]
    del_a_sorted = del_a[idx]

    # Call scan WITH intersection tracking to get both FUES result and intersections
    e_grid_out, keep, intersections = _scan(
        e_grid_sorted,
        vf_sorted,
        policy_1_sorted,
        policy_2_sorted,
        del_a_sorted,
        m_bar,
        LB,
        endog_mbar,
        padding_mbar,
        True,  # include_intersections
        False,  # not_allow_2lefts - default to True
    )

    # Extract kept points for FUES result using boolean mask
    env_idx = np.flatnonzero(keep)
    fues_result = (
        e_grid_sorted[env_idx],
        vf_sorted[env_idx],
        policy_1_sorted[env_idx],
        policy_2_sorted[env_idx],
        del_a_sorted[env_idx],
    )

    # Convert 2D intersection array to tuple of arrays for backward compatibility
    if intersections.shape[0] > 0:
        inter_tuple = (
            intersections[:, 0].copy(),  # e_grid
            intersections[:, 1].copy(),  # value
            intersections[:, 2].copy(),  # policy_1
            intersections[:, 3].copy(),  # policy_2
            intersections[:, 4].copy(),  # del_a
        )
    else:
        empty = np.zeros(0, dtype=np.float64)
        inter_tuple = (empty, empty, empty, empty, empty)
    
    return fues_result, inter_tuple


# ---------------------------------------------------------------------
# Core scan ------------------------------------------------------------
# ---------------------------------------------------------------------


@njit
def _scan(
    e_grid,
    vf,
    a_prime,
    policy_2,
    del_a,
    m_bar,
    LB,
    endog_mbar,
    padding_mbar,
    include_intersections=True,
    not_allow_2lefts=True,
):
    """Core FUES algorithm: Single-pass scan to identify upper envelope points.

    The algorithm maintains three key indices as it scans:
    - k: "tail" - second-to-last kept point
    - j: "head" - last kept point
    - i+1: current point being evaluated

    For each triplet (k, j, i+1), we compute:
    - g_jm1: value gradient from k to j (slope of previous segment)
    - g_1: value gradient from j to i+1 (slope of current segment)
    - g_tilde_a: policy gradient (for jump detection)

    Parameters
    ----------
    e_grid : array
        Sorted endogenous grid points
    vf : array
        Value function at each grid point (read-only, no longer modified)
    a_prime : array
        Next-period assets (policy function)
    policy_2 : array
        Secondary policy variable
    del_a : array
        Policy gradient
    m_bar : float
        Jump threshold (maximum marginal propensity to save)
    LB : int
        Lookback buffer size for backward/forward scans
    endog_mbar : bool
        Use endogenous jump threshold based on policy gradients
    padding_mbar : float
        Additional padding for endogenous threshold
    include_intersections : bool
        Track intersection points where value functions cross

    Returns
    -------
    e_grid : array
        Original grid (unchanged)
    keep : array
        Boolean mask indicating which points to keep
    intersections : tuple or None
        If include_intersections=True, returns (inter_e, inter_v, inter_p1, inter_p2, inter_d)
        containing intersection points and interpolated policies
    """

    N = e_grid.size
    # Boolean mask to track kept points (instead of vf.copy())
    keep = np.ones(N, dtype=np.bool_)

    # 2D array to track intersection points
    # Column 0: e_grid, 1: value, 2: policy_1, 3: policy_2, 4: del_a
    # Each crossing adds 2 rows, so allocate enough space for worst case
    max_inter = 2 * (N - 1)
    intersections = np.full((max_inter, 5), np.nan)
    n_inter = 0

    # Track if this iteration created an intersection that should be used as k (tail) in next iteration
    use_intersection_as_k = False
    intersection_e = 0.0
    intersection_v = 0.0
    intersection_a = 0.0
    intersection_d = 0.0

    # Track if we just added an intersection in the previous iteration
    added_intersection_last_iter = False

    # Circular buffer for recently dropped indices
    m_buf = np.full(LB, -1)  # -1 denotes empty slot
    m_head = 0  # next write position

    # Index bookkeeping
    j, k = 0, -1  # j is head, k is tail i+1 is lead
    last_turn_left = False
    last_was_jump = False  # Track consecutive jumps to prevent back-to-back jumps
    prev_j = 0  # Track previous j value if we decide k must be reset to previous iteration

    # ==================== MAIN SCAN LOOP ====================
    # Process each lead point i+1 to determine if it lies on the upper envelope.
    # We maintain a growing envelope with points k (tail) and j (head).
    prev_g_tilde_a = m_bar
    for i in range(N - 2):

        if i <= 1:  # first two points always kept
            if i == 0:
                j, k = 0, -1
                prev_j = 0
            else:  # i == 1
                prev_j = j  # j was set in previous iteration
                j, k = i, i - 1
            last_turn_left = False
            last_was_jump = False
            added_intersection_last_iter = False
            continue

        # ============= STEP 1: Compute Gradients =============
        # We need gradients to determine the "turn" direction on the egm- value plane:
        # - Right turn (g_1 < g_jm1): lead value point is concave
        # - Left turn (g_1 > g_jm1): lead value point is convex

        # Use intersection values for k (tail) if we have added intersection in last iteration
        # Consume the flag exactly once
        # BUG FIX: Removed premature reset of flag
        if use_intersection_as_k and include_intersections:
            k_e = intersection_e
            k_v = intersection_v
            k_a = intersection_a
            k_d = intersection_d
            use_intersection_as_k = False  # Consume exactly once
        else:
            k_e = e_grid[k] if k >= 0 else e_grid[0]
            k_v = vf[k] if k >= 0 else vf[0]
            k_a = a_prime[k] if k >= 0 else a_prime[0]
            k_d = del_a[k] if k >= 0 else del_a[0]

        
        # Gradient from tail (k) to head (j) - slope of previous segment
        de_prev = max(EPS_D, e_grid[j] - k_e)
        inv_de_prev = 1.0 / de_prev  # Optimization: multiply is faster than divide
        g_jm1 = (vf[j] - k_v) * inv_de_prev

        # Gradient from head (j) to current point (i+1) - slope of current segment
        de_lead = max(EPS_D, e_grid[i + 1] - e_grid[j])
        inv_de_lead = 1.0 / de_lead
        g_1 = (vf[i + 1] - vf[j]) * inv_de_lead

        # Jump threshold: either fixed (m_bar) or endogenous based on policy gradients
        M_max = max(np.abs(del_a[j]), np.abs(del_a[i + 1])) + padding_mbar
        
        if not endog_mbar:
            #M_max = m_bar
            #M_max = min(prev_g_tilde_a + padding_mbar, m_bar)
            M_max = m_bar
        #if not endog_mbar and last_was_jump:
            


        # Policy gradient for jump detection
        del_pol = a_prime[i + 1] - a_prime[j]
        g_tilde_a = np.abs(del_pol * inv_de_lead)

        # Check for non-monotone policies: if savings rate is decreasing
        del_pol_a = (e_grid[i + 1] - a_prime[i + 1]) - (e_grid[j] - a_prime[j])\
        
        del_pol_2 = policy_2[i + 1] - policy_2[j]

        # ============= STEP 2: Classify Current Situation =============
        # Determine turn direction and jump status
        left_turn_any = g_1 > g_jm1
        jump_now = g_tilde_a > M_max or del_pol_2 < 0 or del_pol_a < 0

        
        
        # Demote any consecutive jump to 'no-jump' this iteration
        #if last_was_jump and jump_now:
        #    jump_now = False
        
        # Derive mutually exclusive cases
        left_turn_jump = left_turn_any and jump_now
        left_turn_no_jump = left_turn_any and (not jump_now)
        right_turn_jump = (not left_turn_any) and jump_now
        right_turn_no_jump = (not left_turn_any) and (not jump_now)

        if right_turn_no_jump:
            prev_g_tilde_a = g_tilde_a

        # Reset intersection tracking flag at start of each iteration
        added_intersection_last_iter = False

        # ============= CASE B: Value Fall =============
        # Drop points that have declining value
        if (vf[i + 1] - vf[j] < 0):
            keep[i + 1] = False
            use_intersection_as_k = False  # Reset flag
            m_head = circ_put(m_buf, m_head, i + 1)
            # Update state flags
            last_turn_left = False  # Value fall is not a geometric turn
            last_was_jump = False  # Value fall is not a jump
            continue

        # ============= CASE A: Right-Turn with Jump =============
        # This indicates a jump to a different discrete choice.
        # The point might be suboptimal (jumping from a dominated branch).
        # We need forward scan to check if this jump is valid.
        """ 
        print = False
        if vf[i + 1] < 8.260 and vf[i + 1]> 8.25 and policy_2[i + 1] < 22 and policy_2[i + 1] > 21:
            print("--------------------------------")
            print(f"vf[i+1] (value at current point): {vf[i + 1]}")
            print(f"g_1 (leading value gradient j->i+1): {g_1}")
            print(f"g_jm1 (previous gradient k->j): {g_jm1}")
            print(f"left_turn_any (g_1 > g_jm1): {left_turn_any}")
            print(f"g_tilde_a (policy gradient): {g_tilde_a}")
            print(f"a_prime[i+1] (next period assets): {policy_2[i + 1]}")
            print(f"right_turn_jump flag: {right_turn_jump}")
            print("--------------------------------")
        """

        if right_turn_jump:
            # Always perform forward scan for correctness
            keep_i1, idx_f, found_forward_same_branch = forward_scan_case_a(
                e_grid, vf, a_prime, i, j, N, LB, M_max, g_1
            )
            #keep_i1 = False
            
            if keep_i1:
                created_intersection = False

                # Find backward point on same branch from i+1
                _, idx_b = backward_scan_combined(
                    m_buf,
                    m_head,
                    LB,
                    e_grid,
                    vf,
                    a_prime,
                    i,
                    j,
                    k,
                    i + 1,
                    False,
                    False,
                    False,
                    0.0,
                    M_max,
                    check_drop=False,
                )
        
                # Case A intersection: Only add intersection if this is a jump iteration
                #test_now  = False
                if include_intersections and not last_was_jump:
                    # Build L (old branch) and R (new branch) with robust fallbacks
                    # L: j → idx_f (forward point on same branch as j), fallback k → j
                    L = make_pair_from_indices_or_fallback(
                        e_grid, vf, a_prime, policy_2, del_a,
                        j, idx_f if idx_f != -1 else -1, k, j, N
                    )
                    # R: idx_b → i+1 (new branch), fallback i+1 → forward point on same branch
                    # If we can't find a backward point, look forward and extrapolate back
                    safe_extrap = find_safe_extrapolation_point(e_grid, a_prime, i+1, N, M_max, forward=True)
                    R = make_pair_from_indices_or_fallback(
                        e_grid, vf, a_prime, policy_2, del_a,
                        idx_b if idx_b != -1 else -1, i+1, i+1, safe_extrap, N
                    )
                    
                    # FORCE a crossing strictly inside (e_j, e_{i+1})
                    intr_x, intr_y = _force_crossing_inside(
                        L[0], L[1], L[5], L[6],  # L_x1, L_y1, L_x2, L_y2
                        R[0], R[1], R[5], R[6],  # R_x1, R_y1, R_x2, R_y2
                        e_grid[j], e_grid[i+1], EPS_SEP
                    )
                    
                    # Fallback: if somehow still nan (shouldn't happen), use midpoint
                    if np.isnan(intr_x):
                        mid = 0.5 * (e_grid[j] + e_grid[i+1])
                        denom_L = max(EPS_D, L[5] - L[0])
                        denom_R = max(EPS_D, R[5] - R[0])
                        sL = (L[6] - L[1]) / denom_L
                        sR = (R[6] - R[1]) / denom_R
                        yL = L[1] + sL * (mid - L[0])
                        yR = R[1] + sR * (mid - R[0])
                        intr_x = mid
                        intr_y = 0.5 * (yL + yR)  # Use average at intersection
                    
                    # ADAPTIVE separation based on interval length
                    interval_length = e_grid[i+1] - e_grid[j]
                    sep = min(EPS_SEP, 0.25 * interval_length)
                    
                    # Write TWO points with adaptive separation
                    n_new = add_intersection_from_pairs_with_sep(
                        intersections, n_inter, intr_x, intr_y, sep,
                        L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8], L[9],
                        R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9]
                    )
                    
                    if n_new > n_inter:
                        n_inter = n_new
                        added_intersection_last_iter = True
                        use_intersection_as_k = True
                        
                        # Seed k values from LEFT branch for next iteration
                        denom_L = max(EPS_D, L[5] - L[0])
                        tL = (intr_x - L[0]) / denom_L
                        intersection_e = intr_x
                        intersection_v = intr_y
                        intersection_a = L[2] + tL * (L[7] - L[2])
                        intersection_d = L[4] + tL * (L[9] - L[4])
                        
                        
                        created_intersection = True

                # Advance indices uniformly
                k = j
                prev_j = j
                j = i + 1
                last_turn_left = True  # Right turn but effectively a left because we kept a jump (see figure in paper )
                #last_was_jump = True
                if not created_intersection:
                    use_intersection_as_k = False  # Reset flag only if no intersection
            if not keep_i1:
                keep[i + 1] = False
                m_head = circ_put(m_buf, m_head, i + 1)
                use_intersection_as_k = False  # Reset flag
                last_turn_left = False  # Right turn but effectively a left because we kept a jump (see figure in paper )
                #j = u
                #last_was_jump = Fa
                # only update last was jump if right jump is kept
            # Update state flags for Case A
            if keep_i1:
                last_was_jump=jump_now
            else:
                last_was_jump=False
            continue

        

        # ============= CASE C: Left Turn (with or without Jump) =============
        # Left turn: potential crossing point, j might be suboptimal
        # Use backward scan to find previous point m on same branch as i+1
        if left_turn_jump or left_turn_no_jump:
            keep_j, m_ind = backward_scan_combined(
                m_buf,
                m_head,
                LB,
                e_grid,
                vf,
                a_prime,
                i,
                j,
                k,
                i + 1,
                left_turn_any,  # Fixed: was undefined 'left_turn'
                g_tilde_a,
                last_turn_left,
                g_1,
                M_max,
                check_drop=True,
            )

            # --- CASE C.1: Left Turn with j Dropped ---
            # The backward scan determined that j is suboptimal (lies below the
            # envelope formed by points m and i+1 on the same branch)
            if not keep_j and left_turn_jump:
                keep[j] = False
                m_head = circ_put(m_buf, m_head, j)  # Add dropped j to circular buffer

                # Compute intersection only on jump iterations
                use_intersection_as_k = False
                created_intersection = False
                added_intersection_last_iter = False
                if include_intersections and not last_was_jump:
                    # IMPORTANT: j is being dropped, so old branch is k → j
                    # No need for forward search since j itself is the endpoint
                    
                    # L (old branch after j dropped): k → j
                    L = make_pair_from_indices_or_fallback(
                        e_grid, vf, a_prime, policy_2, del_a,
                        k, j, k, j, N  # Use same indices for primary and fallback
                    )
                    
                    # R (new branch): m_ind → i+1, fallback i+1 → forward point on same branch
                    # m_ind comes from backward_scan_combined
                    # Note: Intersection is still forced to be in (j, i+1) by _force_crossing_inside
                    safe_extrap = find_safe_extrapolation_point(e_grid, a_prime, i+1, N, M_max, forward=True)
                    R = make_pair_from_indices_or_fallback(
                        e_grid, vf, a_prime, policy_2, del_a,
                        m_ind if m_ind != -1 else -1, i+1, i+1, safe_extrap, N
                    )
                    
                    # FORCE a crossing strictly inside (e_j, e_{i+1})
                    intr_x, intr_y = _force_crossing_inside(
                        L[0], L[1], L[5], L[6],  # L_x1, L_y1, L_x2, L_y2
                        R[0], R[1], R[5], R[6],  # R_x1, R_y1, R_x2, R_y2
                        e_grid[j], e_grid[i+1], EPS_SEP
                    )
                    
                    # Fallback: if somehow still nan, use midpoint
                    if np.isnan(intr_x):
                        mid = 0.5 * (e_grid[j] + e_grid[i+1])
                        denom_L = max(EPS_D, L[5] - L[0])
                        denom_R = max(EPS_D, R[5] - R[0])
                        sL = (L[6] - L[1]) / denom_L
                        sR = (R[6] - R[1]) / denom_R
                        yL = L[1] + sL * (mid - L[0])
                        yR = R[1] + sR * (mid - R[0])
                        intr_x = mid
                        intr_y = 0.5 * (yL + yR)  # Use average at intersection
                    
                    # ADAPTIVE separation
                    interval_length = e_grid[i+1] - e_grid[j]
                    sep = min(EPS_SEP, 0.25 * interval_length)
                    
                    # Write TWO points
                    n_new = add_intersection_from_pairs_with_sep(
                        intersections, n_inter, intr_x, intr_y, sep,
                        L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8], L[9],
                        R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9]
                    )
                    
                    if n_new > n_inter:
                        n_inter = n_new
                        use_intersection_as_k = True
                        
                        # Seed k from LEFT branch
                        denom_L = max(EPS_D, L[5] - L[0])
                        tL = (intr_x - L[0]) / denom_L
                        intersection_e = intr_x
                        intersection_v = intr_y
                        intersection_a = L[2] + tL * (L[7] - L[2])
                        intersection_d = L[4] + tL * (L[9] - L[4])
                        

                        created_intersection = True

                # Advance indices (after dropping j)
                # k stays unchanged (the tail remains the same)
                prev_j = k  # Update prev_j to current tail
                j = i + 1   # Advance j to next point

            # --- CASE C.2: Left Turn but j is Kept ---
            else:
                # "Avoid two lefts" cleanup only for jump iterations
                if  not_allow_2lefts and jump_now and last_was_jump:
                    keep[j] = False
                    #_head = circ_put(m_buf, m_head, j)  # Add dropped j to circular buffer

                    # Remove last intersection to avoid spurious intersections (2 rows)
                    if include_intersections and added_intersection_last_iter and n_inter > 0:
                        n_inter = n_inter - 2
                    
                    j = prev_j

                # Add intersection for left turn case (only on jump iterations)
                
                use_intersection_as_k = False
                if include_intersections and not last_was_jump:
                    # Find forward point on same branch from j
                    found_fwd, idx_fwd = find_forward_same_branch(
                        e_grid, a_prime, j, j, N, LB, m_bar
                    )
                    
                    # Find backward point on same branch from i+1
                    _, idx_back = backward_scan_combined(
                        m_buf, m_head, LB, e_grid, vf, a_prime,
                        i, j, k, i+1, False, False, False, 0.0, M_max,
                        check_drop=False
                    )
                    
                    # L (old branch): j → idx_fwd
                    L = make_pair_from_indices_or_fallback(
                        e_grid, vf, a_prime, policy_2, del_a,
                        j, idx_fwd if found_fwd else -1, k, j, N
                    )
                    
                    # R (new branch): idx_back → i+1, fallback i+1 → forward point on same branch
                    # Note: Intersection is forced to be in (j, i+1) by _force_crossing_inside
                    safe_extrap = find_safe_extrapolation_point(e_grid, a_prime, i+1, N, M_max, forward=True)
                    R = make_pair_from_indices_or_fallback(
                        e_grid, vf, a_prime, policy_2, del_a,
                        idx_back if idx_back != -1 else -1, i+1, i+1, safe_extrap, N
                    )
                    
                    # FORCE a crossing strictly inside (e_j, e_{i+1})
                    intr_x, intr_y = _force_crossing_inside(
                        L[0], L[1], L[5], L[6],
                        R[0], R[1], R[5], R[6],
                        e_grid[j], e_grid[i+1], EPS_SEP
                    )
                    
                    # Fallback to midpoint if needed
                    if np.isnan(intr_x):
                        mid = 0.5 * (e_grid[j] + e_grid[i+1])
                        denom_L = max(EPS_D, L[5] - L[0])
                        denom_R = max(EPS_D, R[5] - R[0])
                        sL = (L[6] - L[1]) / denom_L
                        sR = (R[6] - R[1]) / denom_R
                        yL = L[1] + sL * (mid - L[0])
                        yR = R[1] + sR * (mid - R[0])
                        intr_x = mid
                        intr_y = 0.5 * (yL + yR)  # Use average at intersection
                    
                    # ADAPTIVE separation
                    interval_length = e_grid[i+1] - e_grid[j]
                    sep = min(EPS_SEP, 0.25 * interval_length)
                    
                    # Write TWO points
                    n_new = add_intersection_from_pairs_with_sep(
                        intersections, n_inter, intr_x, intr_y, sep,
                        L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8], L[9],
                        R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9]
                    )
                    
                    if n_new > n_inter:
                        n_inter = n_new
                        use_intersection_as_k = True
                        
                        # Seed k from LEFT branch
                        denom_L = max(EPS_D, L[5] - L[0])
                        tL = (intr_x - L[0]) / denom_L
                        intersection_e = intr_x
                        intersection_v = intr_y
                        intersection_a = L[2] + tL * (L[7] - L[2])
                        intersection_d = L[4] + tL * (L[9] - L[4])
                        
                        
                        created_intersection = True
                
                # Advance indices uniformly
                if  not_allow_2lefts and jump_now and last_was_jump:
                    k = j
                    #prev_j = j
                    j = i + 1
                else:
                    k = j
                    prev_j = j
                    j = i + 1
            
            # Update state flags for Case C
            last_turn_left = True
            last_was_jump = jump_now
            continue

        # ============= CASE R: Right Turn without Jump =============
        # Concave continuation; simply advance indices
        if right_turn_no_jump:
            # Advance indices uniformly
            k = j
            prev_j = j
            j = i + 1
            use_intersection_as_k = False  # Reset flag
            
            # Update state flags for Case R
            last_turn_left = False
            last_was_jump = False
            continue

    # Return intersection results as 2D array slice
    return e_grid, keep, intersections[:n_inter, :]

