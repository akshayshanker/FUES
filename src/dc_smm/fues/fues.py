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
EPS_D = 1e-100  # Epsilon for division protection
EPS_A = 1e-100  # Epsilon for gradient calculations
EPS_SEP = 1e-05 # Epsilon for intersection separation
EPS_fwd_back = 100

# ---------------------------------------------------------------------
# Helpers that remain identical ---------------------------------------
# ---------------------------------------------------------------------


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
    if np.abs(denom) < 1E-14:
        return (np.nan, np.nan)
    s = ((bx1 - ax1) * db_y - (by1 - ay1) * db_x) / denom
    return (ax1 + s * da_x, ay1 + s * da_y)


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
def backward_scan_combined(
    m_buf,
    m_head,
    LB,
    e_grid,
    vf,
    a_prime,
    i,
    j,
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
                if g_m_a < m_bar and de < EPS_fwd_back:
                    m_ind = m_idx
                    if left_turn and not last_turn_left:
                        # g_m_vf already computed with de
                        g_m_vf = (vf[i_plus_1] - vf[m_idx]) / de
                        if g_1 < g_m_vf:
                            keep_j = False
                    break
        else:
            # For intersection finding (check_drop=False), use original find_backward_same_branch logic
            if m_idx != -1 and m_idx < i_plus_1:
                de = max(EPS_D, e_grid[i_plus_1] - e_grid[m_idx])
                grad_a = np.abs((a_prime[i_plus_1] - a_prime[m_idx]) / de)
                if grad_a < m_bar and de < EPS_fwd_back:
                    m_ind = m_idx
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
        
        if g_f_a < m_bar:
            found_forward_same_branch = True
            idx_f = i + 2 + f  # Store actual grid index
            # Compute g_f_vf for this point
            de_1 = max(EPS_D, e_grid[i + 2 + f] - e_grid[j])
            g_f_vf_at_idx = (vf[i + 2 + f] - vf[j]) / de_1
            if g_1 > g_f_vf_at_idx:
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
    include_intersections : bool, default False
        If True, intersection points where discrete choices switch are included in output.
        If False, returns only the original upper envelope points.
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
    max_inter = N // 2
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
    prev_j = 0  # Track previous j value if we decide k must be reset to previous iteration

    # ==================== MAIN SCAN LOOP ====================
    # Process each lead point i+1 to determine if it lies on the upper envelope.
    # We maintain a growing envelope with points k (tail) and j (head).
    for i in range(N - 2):

        if i <= 1:  # first two points always kept
            j, k = i, i - 1
            last_turn_left = False
            added_intersection_last_iter = False
            continue

        # ============= STEP 1: Compute Gradients =============
        # We need gradients to determine the "turn" direction on the egm- value plane:
        # - Right turn (g_1 < g_jm1): lead value point is concave
        # - Left turn (g_1 > g_jm1): lead value point is convex

        # Use intersection values for k (tail) if we have added intersection in last iteration
        if use_intersection_as_k and include_intersections:
            k_e = intersection_e
            k_v = intersection_v
            k_a = intersection_a
            k_d = intersection_d
            use_intersection_as_k = False
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
            M_max = m_bar

        # Policy gradient for jump detection
        del_pol = a_prime[i + 1] - a_prime[j]
        g_tilde_a = np.abs(del_pol * inv_de_lead)

        # Check for non-monotone policies: if savings rate is decreasing
        del_pol_a = (e_grid[i + 1] - a_prime[i + 1]) - (e_grid[j] - a_prime[j])

        # ============= STEP 2: Classify Current Situation =============
        # Determine if we have a right turn with jump or left turn
        right_turn_jump = (g_1 <= g_jm1) and (g_tilde_a > M_max)
        left_turn = g_1 > g_jm1 and (g_tilde_a > M_max)
        right_turn_no_jump = (g_1 <= g_jm1) and (g_tilde_a <= M_max) 

        # Reset intersection tracking flag at start of each iteration
        added_intersection_last_iter = False

        # ============= CASE B: Value Fall or Non-Monotone Policy =============
        # Drop points that have declining value or violate monotonicity
        if (vf[i + 1] - vf[j] < 0 ):
            keep[i + 1] = False
            use_intersection_as_k = False  # Reset flag
            last_turn_left = False
            m_head = circ_put(m_buf, m_head, i + 1)
            continue

        # ============= CASE A: Right-Turn with Jump =============
        # This indicates a jump to a different discrete choice.
        # The point might be suboptimal (jumping from a dominated branch).
        # We need forward scan to check if this jump is valid.
        if right_turn_jump:
            # Always perform forward scan for correctness
            #print("right_turn_jump")
            keep_i1, idx_f, found_forward_same_branch = forward_scan_case_a(
                e_grid, vf, a_prime, i, j, N, LB, m_bar, g_1
            )
            
            if keep_i1 and last_turn_left==False:
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
                    i + 1,
                    False,
                    False,
                    False,
                    0.0,
                    m_bar,
                    check_drop=False,
                )

                # Case A intersection: Always try to add intersection using extrapolation if needed
                if include_intersections:
                    # Build pairs, falling back as needed:
                    # Left/old branch: (j -> idx_f) or fallback (k -> j)
                    L = make_pair_from_indices_or_fallback(
                        e_grid, vf, a_prime, policy_2, del_a,
                        j, idx_f if idx_f != -1 else -1, k, j, N
                    )
                    # Right/new branch: (idx_b -> i+1) or fallback (i+1 -> min(i+2, N-1))
                    R = make_pair_from_indices_or_fallback(
                        e_grid, vf, a_prime, policy_2, del_a,
                        idx_b, i + 1, i + 1, min(i + 2, N - 1), N
                    )

                    # Intersect "infinite" lines on value plane
                    intr_x, intr_y = line_intersect_unbounded(
                        L[0], L[1], L[5], L[6],   # (L_x1,L_y1) -> (L_x2,L_y2)
                        R[0], R[1], R[5], R[6]    # (R_x1,R_y1) -> (R_x2,R_y2)
                    )

                    # Optional: mild guard to avoid absurdly far crossings
                    if not np.isnan(intr_x):
                        # e_min/e_max of the four endpoints, expanded a tad
                        e_min = min(min(L[0], L[5]), min(R[0], R[5])) - 10*EPS_SEP
                        e_max = max(max(L[0], L[5]), max(R[0], R[5])) + 10*EPS_SEP
                        if intr_x >= e_min - 1e-8 and intr_x <= e_max + 1e-8:
                            new_n = add_intersection_from_pairs(
                                intersections, n_inter, intr_x, intr_y,
                                L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8], L[9],
                                R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9]
                            )
                            if new_n > n_inter:
                                n_inter = new_n
                                added_intersection_last_iter = True
                                use_intersection_as_k = True
                                intersection_e = intr_x
                                intersection_v = intr_y
                                # interpolate a_prime, del_a on the left side to seed k
                                denom_L = max(EPS_D, L[5]-L[0])
                                tL = (intr_x - L[0]) / denom_L
                                intersection_a = L[2] + tL * (L[7] - L[2])
                                intersection_d = L[4] + tL * (L[9] - L[4])
                                created_intersection = True

                # Advance indices
                k = j
                prev_j = j
                j = i + 1
                last_turn_left = True
                if not created_intersection:
                    use_intersection_as_k = False  # Reset flag only if no intersection
            else:
                keep[i + 1] = False
                m_head = circ_put(m_buf, m_head, i + 1)
                use_intersection_as_k = False  # Reset flag
                last_turn_left = False
            continue

        

        # ============= CASE C: Left Turn or Right Turn without Jump =============
        # Either:
        # - Left turn (g_1 > g_jm1): potential crossing point, j might be suboptimal
        # - Right turn without jump: normal concave segment continuation
        # Use backward scan to find previous point m on same branch as i+1
        if left_turn:
            keep_j, m_ind = backward_scan_combined(
                m_buf,
                m_head,
                LB,
                e_grid,
                vf,
                a_prime,
                i,
                j,
                i + 1,
                left_turn,
                g_tilde_a,
                last_turn_left,
                g_1,
                m_bar,
                check_drop=True,
            )

            # --- CASE C.1: Left Turn with j Dropped ---
            # The backward scan determined that j is suboptimal (lies below the
            # envelope formed by points m and i+1 on the same branch)
            if not keep_j:
                keep[j] = False
                m_head = circ_put(m_buf, m_head, j)  # Add dropped j to circular buffer

                # Compute intersection only if not a consecutive left turn
                use_intersection_as_k = False
                created_intersection = False
                added_intersection_last_iter = False
                if include_intersections:
                    # Find forward point on same branch from j (old branch)
                    found_fwd, idx_f = find_forward_same_branch(
                        e_grid, a_prime, j, j, N, LB, m_bar
                    )
                    
                    # Find backward point on same branch from i+1 (new branch)
                    # Note: m_ind is already the backward point on the same branch as i+1
                    idx_b = m_ind  # We already have this from the backward scan
                    
                    # Build pairs with fallbacks for extrapolation
                    # Left/old branch: (j -> idx_f) or fallback (k -> j)
                    L = make_pair_from_indices_or_fallback(
                        e_grid, vf, a_prime, policy_2, del_a,
                        j, idx_f if found_fwd else -1, k, j, N
                    )
                    # Right/new branch: (idx_b -> i+1) or fallback (i+1 -> min(i+2, N-1))
                    R = make_pair_from_indices_or_fallback(
                        e_grid, vf, a_prime, policy_2, del_a,
                        idx_b, i + 1, i + 1, min(i + 2, N - 1), N
                    )

                    # Intersect "infinite" lines on value plane
                    intr_x, intr_y = line_intersect_unbounded(
                        L[0], L[1], L[5], L[6],   # (L_x1,L_y1) -> (L_x2,L_y2)
                        R[0], R[1], R[5], R[6]    # (R_x1,R_y1) -> (R_x2,R_y2)
                    )

                    if not np.isnan(intr_x):
                        # e_min/e_max of the four endpoints, expanded a tad
                        e_min = min(L[0], L[5], R[0], R[5]) - 10*EPS_SEP
                        e_max = max(L[0], L[5], R[0], R[5]) + 10*EPS_SEP
                        if intr_x >= e_min - 1e-8 and intr_x <= e_max + 1e-8:
                            n_new = add_intersection_from_pairs(
                                intersections, n_inter, intr_x, intr_y,
                                L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8], L[9],
                                R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9]
                            )
                            if n_new > n_inter:
                                n_inter = n_new
                                added_intersection_last_iter = True
                                use_intersection_as_k = True
                                intersection_e = intr_x
                                intersection_v = intr_y
                                # interpolate a_prime, del_a on the left side to seed k
                                denom_L = max(EPS_D, L[5]-L[0])
                                tL = (intr_x - L[0]) / denom_L
                                intersection_a = L[2] + tL * (L[7] - L[2])
                                intersection_d = L[4] + tL * (L[9] - L[4])
                                j = i + 1  # advance j
                                k = prev_j
                                created_intersection = True

                # Mark this as a left turn after processing
                last_turn_left = True
                if created_intersection == False:
                    j = i + 1  # advance j
                    k = prev_j
                    prev_j = j

            # --- CASE C.2: Left Turn but j is Kept ---
            else:
                if last_turn_left and not_allow_2lefts:
                        keep[j] = False
                        m_head = circ_put(m_buf, m_head, j)  # Add dropped j to circular buffer

                        # Remove last intersection to avoid spurious intersections
                        if include_intersections and added_intersection_last_iter and n_inter > 0:
                            n_inter = n_inter - 2
                        
                        j = prev_j

                    # Add intersection for left turn case
                
                use_intersection_as_k = False
                if include_intersections and  k >= 0:
                        # Find forward point on same branch from j
                        found_fwd, idx_fwd = find_forward_same_branch(
                            e_grid, a_prime, j, j, N, LB, m_bar
                        )

                        # Find backward point on same branch from i+1
                        _, idx_back = backward_scan_combined(
                            m_buf,
                            m_head,
                            LB,
                            e_grid,
                            vf,
                            a_prime,
                            i,
                            j,
                            i + 1,
                            False,
                            False,
                            False,
                            0.0,
                            m_bar,
                            check_drop=False,
                        )
                        found_back = idx_back != -1

                        # Build pairs with fallbacks for extrapolation
                        # Left/old branch: (j -> idx_fwd) or fallback (k -> j)
                        L = make_pair_from_indices_or_fallback(
                            e_grid, vf, a_prime, policy_2, del_a,
                            j, idx_fwd if found_fwd else -1, k, j, N
                        )
                        # Right/new branch: (idx_back -> i+1) or fallback (i+1 -> min(i+2, N-1))
                        R = make_pair_from_indices_or_fallback(
                            e_grid, vf, a_prime, policy_2, del_a,
                            idx_back if found_back else -1, i + 1, i + 1, min(i + 2, N - 1), N
                        )

                        # Intersect "infinite" lines on value plane
                        intr_x, intr_y = line_intersect_unbounded(
                            L[0], L[1], L[5], L[6],   # (L_x1,L_y1) -> (L_x2,L_y2)
                            R[0], R[1], R[5], R[6]    # (R_x1,R_y1) -> (R_x2,R_y2)
                        )

                        if not np.isnan(intr_x):
                            # e_min/e_max of the four endpoints, expanded a tad
                            e_min = min(L[0], L[5], R[0], R[5]) - 10*EPS_SEP
                            e_max = max(L[0], L[5], R[0], R[5]) + 10*EPS_SEP
                            if intr_x >= e_min - 1e-8 and intr_x <= e_max + 1e-8:
                                n_new = add_intersection_from_pairs(
                                    intersections, n_inter, intr_x, intr_y,
                                    L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8], L[9],
                                    R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9]
                                )
                                if n_new > n_inter:
                                    n_inter = n_new
                                    added_intersection_last_iter = True
                                    use_intersection_as_k = True
                                    intersection_e = intr_x
                                    intersection_v = intr_y
                                    # interpolate a_prime, del_a on the left side to seed k
                                    denom_L = max(EPS_D, L[5]-L[0])
                                    tL = (intr_x - L[0]) / denom_L
                                    intersection_a = L[2] + tL * (L[7] - L[2])
                                    intersection_d = L[4] + tL * (L[9] - L[4])
                
                last_turn_left = True
                # Advance indices for next iteration

                if last_turn_left and not_allow_2lefts:
                    prev_j = k
                    k = prev_j
                    #prev_j = j
                    j = i + 1
                    last_turn_left = True
                    #use_intersection_as_k = False  # Reset flag
                else:
                    prev_j = k
                    k = j
                    #prev_j = j
                    j = i + 1
            continue

        # --- CASE C.3: Right Turn without Jump ---
        if right_turn_no_jump:
            last_turn_left = False
            # For right turn without jump, advance j normally
            prev_j = k
            k = j  # Update k before advancing j
            #prev_j = j
            j = i + 1
            use_intersection_as_k = False  # Reset flag only if not left turn
            #added_intersection_last_iter
            continue

    # Return intersection results as 2D array slice
    return e_grid, keep, intersections[:n_inter, :]
