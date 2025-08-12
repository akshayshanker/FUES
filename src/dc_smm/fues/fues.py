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
EPS_D = 1e-50 # Epsilon for division protection (updated for numerical stability)
EPS_SEP = 1e-10 # Epsilon for intersection separation
EPS_fwd_back = 0.5
PARALLEL_GUARD = 1e-12 # Guard for parallel line detection

TURN_LEFT = 1; TURN_RIGHT = 0
JUMP_YES = 1; JUMP_NO = 0

# ---------------------------------------------------------------------
# Helpers that remain identical ---------------------------------------
# ---------------------------------------------------------------------



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

    - Computes infinite-line intersection robustly (vector form, not slopes).
    - Clips x into (e_lo+eps, e_hi-eps).
    - Evaluates both branches at that x and averages.
    - Clips y into the band [min(yL,yR), max(yL,yR)] to avoid tiny overshoots.

    Always returns a valid (x,y) inside (e_lo, e_hi).
    """

    # 0) Order the interval defensively
    lo = e_lo if e_lo <= e_hi else e_hi
    hi = e_hi if e_hi >= e_lo else e_lo

    # 1) Robust infinite-line intersection (parametric / cross-product form)
    dxL = L_x2 - L_x1; dyL = L_y2 - L_y1
    dxR = R_x2 - R_x1; dyR = R_y2 - R_y1
    denom = dxL * dyR - dyL * dxR

    if np.abs(denom) >= parallel_guard:
        # Solve for s in: (L_x1, L_y1) + s*(dxL,dyL) = (R_x1, R_y1) + t*(dxR,dyR)
        s = ((R_x1 - L_x1) * dyR - (R_y1 - L_y1) * dxR) / denom
        x_star = L_x1 + s * dxL
        # We won't trust y_star; we'll recompute y from both branches after clipping x.
    else:
        # Near parallel → no reliable crossing; start from midpoint
        x_star = 0.5 * (lo + hi)

    # 2) Clip x strictly inside (lo, hi)
    x = _clip_open(x_star, lo, hi, eps)

    # 3) Evaluate both branches at the *clipped* x with sign-preserving slopes
    #    (avoid abs() on dx to preserve orientation)
    dxL_safe = dxL if np.abs(dxL) > eps_d else (eps_d if dxL >= 0.0 else -eps_d)
    dxR_safe = dxR if np.abs(dxR) > eps_d else (eps_d if dxR >= 0.0 else -eps_d)
    sL = dyL / dxL_safe
    sR = dyR / dxR_safe
    yL = L_y1 + sL * (x - L_x1)
    yR = R_y1 + sR * (x - R_x1)

    # 4) Average for the crossing value, then clip into the [min, max] band
    y = 0.5 * (yL + yR)
    y_min = yL if yL < yR else yR
    y_max = yR if yR > yL else yL
    if y < y_min:
        y = y_min
    elif y > y_max:
        y = y_max

    return (x, y)


# ---------------- Circular buffer utilities --------------------------


@njit
def circ_put(buf, head, value):
    """Write *value* at *head* position, return new head index."""
    buf[head] = value
    return (head + 1) % buf.size


# ---------------- Intersection helpers -------------------


@njit(inline="always")
def check_same_branch(e_grid, a_prime, idx1, idx2, m_bar, eps_d=EPS_D, eps_fwd_back=EPS_fwd_back):
    """Check if two points are on the same branch (no jump between them)."""
    if idx1 < 0 or idx2 < 0 or idx1 >= len(e_grid) or idx2 >= len(e_grid):
        return False
    de = max(eps_d, e_grid[idx2] - e_grid[idx1])
    g_a = np.abs((a_prime[idx2] - a_prime[idx1]) / de)
    return g_a < m_bar and de < eps_fwd_back

@njit(inline="always")
def find_safe_extrapolation_point(e_grid, a_prime, base_idx, N, m_bar, forward=True, eps_d=EPS_D, eps_fwd_back=EPS_fwd_back):
    """
    Find a safe point for extrapolation that's on the same branch as base_idx.
    Returns the index of a point that doesn't jump from base_idx, or base_idx if none found.
    """
    if forward:
        # Search forward for a point on same branch
        for offset in range(1, min(4, N - base_idx)):
            test_idx = base_idx + offset
            if test_idx < N and check_same_branch(e_grid, a_prime, base_idx, test_idx, m_bar, eps_d, eps_fwd_back):
                return test_idx
    else:
        # Search backward for a point on same branch
        for offset in range(1, min(4, base_idx + 1)):
            test_idx = base_idx - offset
            if test_idx >= 0 and check_same_branch(e_grid, a_prime, test_idx, base_idx, m_bar, eps_d, eps_fwd_back):
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
def add_intersection_from_pairs_with_sep(
    intersections, n_inter, intr_x, intr_y, sep,
    L_x1, L_y1, L_a1, L_p21, L_d1, L_x2, L_y2, L_a2, L_p22, L_d2,
    R_x1, R_y1, R_a1, R_p21, R_d1, R_x2, R_y2, R_a2, R_p22, R_d2,
    eps_d=EPS_D
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
        denom_L = L_x2 - L_x1
        if np.abs(denom_L) < eps_d:
            denom_L = eps_d if denom_L >= 0.0 else -eps_d
        tL = (intr_x - sep - L_x1) / denom_L
        intersections[n_inter, 2] = L_a1 + tL * (L_a2 - L_a1)    # a_prime
        intersections[n_inter, 3] = L_p21 + tL * (L_p22 - L_p21) # policy_2
        intersections[n_inter, 4] = L_d1 + tL * (L_d2 - L_d1)    # del_a

        #if intr_x> L_x1:
       #     intersections[n_inter, 2] = intersections[n_inter, 2] - 1e-10

        
        # Add RIGHT point (slightly after intersection)
        intersections[n_inter+1, 0] = intr_x + sep  # e_grid coordinate
        intersections[n_inter+1, 1] = intr_y         # value at intersection
        
        # Interpolate policies from RIGHT branch at intersection point
        denom_R = R_x2 - R_x1
        if np.abs(denom_R) < eps_d:
            denom_R = eps_d if denom_R >= 0.0 else -eps_d
        tR = (intr_x + sep - R_x1) / denom_R
        intersections[n_inter+1, 2] = R_a1 + tR * (R_a2 - R_a1)    # a_prime
        intersections[n_inter+1, 3] = R_p21 + tR * (R_p22 - R_p21) # policy_2
        intersections[n_inter+1, 4] = R_d1 + tR * (R_d2 - R_d1)    # del_a
        
        return n_inter + 2
    return n_inter


@njit
def backward_scan_combined(
    m_buf, m_head, LB,
    x_dcsn_hat, vlu, kappa,
    i, j, k, i_plus_1,
    left_turn, g_tilde_a, last_turn_left, g_1,
    m_bar,
    check_drop=True,
    eps_d=EPS_D,
):
    """
    Return (keep_j, b) where b is the first deleted index before i+1
    such that:
       gq_j_b > m_bar   (j is on a different branch from b)
       gq_i1_b < m_bar  (i+1 is on the same branch as b)
    and, if check_drop is True, we also test whether p_j lies strictly
    below the chord joining p_b and p_{i+1} on the (vlu, x_dcsn_hat) plane.
    """
    keep_j = True
    b = -1

    # Traverse recently dropped points, most-recent first
    for t in range(LB):
        idx = (m_head - 1 - t) % LB
        cand = m_buf[idx]
        if cand == -1:
            continue

        # Enforce order: b must be before the lead (and usually before the head)
        if cand >= i_plus_1:
            continue
        # Optional but usually desirable:
        # if cand > j:
        #     continue

        # --- Two-branch tests (policy-space) ---
        # gq_i1_b  = |kappa[i+1] - kappa[b]| / |x[i+1] - x[b]|   < m_bar
        # gq_j_b   = |kappa[j]   - kappa[b]| / |x[j]   - x[b]|   > m_bar
        den_ib = max(eps_d, abs(x_dcsn_hat[i_plus_1] - x_dcsn_hat[cand]))
        den_jb = max(eps_d, abs(x_dcsn_hat[j]        - x_dcsn_hat[cand]))
        gq_i1_b = abs(kappa[i_plus_1] - kappa[cand]) / den_ib
        gq_j_b  = abs(kappa[j]        - kappa[cand]) / den_jb

        if not (gq_i1_b < m_bar and gq_j_b > m_bar):
            continue

        # First candidate that passes the two tests is our b
        b = cand

        if check_drop:
            # --- Geometric drop test for j in value space ---
            # Is p_j below the chord from p_b to p_{i+1}?
            # v_on_bi1(x_j) = vlu[b] + (x_j - x_b)/(x_{i+1} - x_b) * (vlu[i+1] - vlu[b])
            w = (x_dcsn_hat[j] - x_dcsn_hat[b]) / den_ib
            v_on_bi1_at_j = vlu[b] + w * (vlu[i_plus_1] - vlu[b])

            # tiny slack to avoid flapping due to roundoff
            if vlu[j] < v_on_bi1_at_j - 1e-14:
                keep_j = False
        break

    return keep_j, b



@njit
def find_forward_same_branch(e_grid, a_prime, start_idx, j_idx, N, LB, m_bar, eps_d=EPS_D, eps_fwd_back=EPS_fwd_back):
    """Find the first point in forward scan that's on same branch.

    Returns found flag and index.
    """
    for f in range(min(LB, N - start_idx - 1)):
        if start_idx + 1 + f >= N:
            break
        de = max(eps_d, e_grid[start_idx + 1 + f] - e_grid[j_idx])
        g_a = np.abs((a_prime[start_idx + 1 + f] - a_prime[j_idx]) / de)
        if g_a < m_bar and de < eps_fwd_back:
            return True, start_idx + 1 + f
    return False, -1


@njit
def forward_scan_case_a(e_grid, vlu, a_prime, i, j, N, LB, m_bar, g_1, eps_d=EPS_D, eps_fwd_back=EPS_fwd_back):
    """Forward scan validation for Case A (right-turn jump).

    When we detect a right-turn jump, point i+1 might be jumping from a dominated
    branch. This function checks if i+1 should be kept by:
    1. Finding a future point f on the same branch as j
    2. Checking if the value gradient from j to i+1 dominates the gradient from i+1 to f

    If g_1 > g_f (gradient j→i+1 > gradient i+1→f), then i+1 lies above the
    extrapolated line from j to f, so we keep it.

    Parameters
    ----------
    e_grid, vlu, a_prime : arrays
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
    idx_f_return = -1
    keep_i1 = False
    found_forward_same_branch = False

    for f in range(LB):
        if i + 2 + f >= N:  # CRITICAL: Add bounds check
            break
        de = max(eps_d, e_grid[i + 2 + f]-e_grid[j])
        #sde_1 = max(eps_d, e_grid[i + 1] - e_grid[j])
        g_f_a = np.abs((a_prime[j] - a_prime[i + 2 + f]) / de)
        idx_f = i + 2 + f  # Store actual grid index
        de_jump = max(eps_d, e_grid[idx_f] - e_grid[i + 1])
        g_jump = np.abs((a_prime[i + 1] - a_prime[idx_f]) / de_jump)
        
        if g_f_a < m_bar and de < eps_fwd_back and g_jump >= m_bar:
            
            found_forward_same_branch = True
            idx_f_return = idx_f
            # Compute g_f_vlu for this point
            de_1 = max(eps_d, e_grid[i + 2 + f] - e_grid[j])
            g_f_vlu_at_idx = (vlu[i + 2 + f] - vlu[j]) / de_1
            if g_1 > g_f_vlu_at_idx:
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

    return keep_i1, idx_f_return,found_forward_same_branch


# ---------------------------------------------------------------------
# Public wrapper -------------------------------------------------------
# ---------------------------------------------------------------------



@njit
def _postclean_double_jump_mask(e_grid, a_prime, m_bar, skip_mask, eps_d=EPS_D):
    """
    Keep[i] == False  iff BOTH neighbors of i are policy jumps > m_bar.
    First and last points are always kept. Points with skip_mask[i]==True
    (e.g., intersection rows) are always kept.

    Parameters
    ----------
    e_grid : 1d array (sorted)
    a_prime: 1d array (policy_1 in your outer API)
    m_bar  : float
    skip_mask : 1d bool array with same length as e_grid
                True -> never drop (e.g., intersection rows)

    Returns
    -------
    keep : 1d bool array
    """
    N = e_grid.size
    keep = np.ones(N, dtype=np.bool_)
    if N <= 2:
        return keep

    # Endpoints: always keep
    keep[0] = True
    keep[N-1] = True

    for i in range(1, N-1):
        #if skip_mask[i]:
        #    continue

        deL = e_grid[i]   - e_grid[i-1]
        deR = e_grid[i+1] - e_grid[i]
        # protect divisions but keep sign (not needed for abs, but consistent)
        if np.abs(deL) < eps_d:
            deL = eps_d if deL >= 0.0 else -eps_d
        if np.abs(deR) < eps_d:
            deR = eps_d if deR >= 0.0 else -eps_d

        gL = np.abs( (a_prime[i]   - a_prime[i-1]) / deL )
        gR = np.abs( (a_prime[i+1] - a_prime[i])   / deR )

        # Drop i only if both sides are true jumps
        if (gL > m_bar) and (gR > m_bar):
            keep[i] = False
            #print(e_grid[i])
            #print(a_prime[i])


    return keep


#@njit
def FUES(
    e_grid, vlu, policy_1, policy_2, del_a,
    b=1e-10, m_bar=1.0, LB=4, endog_mbar=False, padding_mbar=0.0,
    include_intersections=True,
    eps_d=None, eps_sep=None, eps_fwd_back=None, parallel_guard=None,
):
    # Use provided epsilons or fall back to module defaults
    eps_d = eps_d if eps_d is not None else EPS_D
    eps_sep = eps_sep if eps_sep is not None else EPS_SEP
    eps_fwd_back = eps_fwd_back if eps_fwd_back is not None else EPS_fwd_back
    parallel_guard = parallel_guard if parallel_guard is not None else PARALLEL_GUARD
    idx = np.argsort(e_grid)
    e_grid = e_grid[idx]; vlu = vlu[idx]
    policy_1 = policy_1[idx]; policy_2 = policy_2[idx]; del_a = del_a[idx]

    e_out, keep_scan, intersections = _scan(
        e_grid, vlu, policy_1, policy_2, del_a,
        m_bar, LB, endog_mbar, padding_mbar,
        include_intersections, True,
        eps_d, eps_sep, eps_fwd_back, parallel_guard
    )

    env_idx = np.flatnonzero(keep_scan)
    e_kept  = e_out[env_idx]
    v_kept  = vlu[env_idx]
    p1_kept = policy_1[env_idx]
    p2_kept = policy_2[env_idx]
    d_kept  = del_a[env_idx]

    # --- Merge intersections (if any) and sort ---
    if include_intersections and intersections.shape[0] > 0:
        n_kept  = e_kept.size
        n_inter = intersections.shape[0]
        n_total = n_kept + n_inter

        all_e  = np.empty(n_total, dtype=e_kept.dtype)
        all_v  = np.empty(n_total, dtype=v_kept.dtype)
        all_p1 = np.empty(n_total, dtype=p1_kept.dtype)
        all_p2 = np.empty(n_total, dtype=p2_kept.dtype)
        all_d  = np.empty(n_total, dtype=d_kept.dtype)
        is_inter = np.zeros(n_total, dtype=np.bool_)  # track intersection rows

        all_e[:n_kept] = e_kept;       all_e[n_kept:] = intersections[:,0]
        all_v[:n_kept] = v_kept;       all_v[n_kept:] = intersections[:,1]
        all_p1[:n_kept]= p1_kept;      all_p1[n_kept:]= intersections[:,2]
        all_p2[:n_kept]= p2_kept;      all_p2[n_kept:]= intersections[:,3]
        all_d[:n_kept] = d_kept;       all_d[n_kept:] = intersections[:,4]
        is_inter[n_kept:] = True

        sort_idx = np.argsort(all_e)
        all_e  = all_e[sort_idx]
        all_v  = all_v[sort_idx]
        all_p1 = all_p1[sort_idx]
        all_p2 = all_p2[sort_idx]
        all_d  = all_d[sort_idx]
        is_inter = is_inter[sort_idx]

        post_mask = _postclean_double_jump_mask(all_e, all_p2, m_bar, is_inter, eps_d)

        final_mask = post_mask  # or just post_mask if you want only post-clean

        return (all_e[final_mask], all_v[final_mask],
                all_p1[final_mask], all_p2[final_mask], all_d[final_mask])

    # No intersections to merge → still apply the POST-CLEAN on the kept scan
    # Build a skip_mask of all False (no intersections to protect)
    is_inter = np.zeros(e_kept.size, dtype=np.bool_)
    post_mask = _postclean_double_jump_mask(e_kept, p2_kept, m_bar, is_inter, eps_d)

    return (e_kept[post_mask], v_kept[post_mask],
            p1_kept[post_mask], p2_kept[post_mask], d_kept[post_mask])



# ---------------------------------------------------------------------
# Non-jitted wrapper for getting intersections separately ---------------
# ---------------------------------------------------------------------


def FUES_sep_intersect(
    e_grid,
    vlu,
    policy_1,
    policy_2,
    del_a,
    b=1e-10,
    m_bar=2.0,
    LB=4,
    endog_mbar=False,
    padding_mbar=0.0,
    eps_d=None, eps_sep=None, eps_fwd_back=None, parallel_guard=None,
):
    # Use provided epsilons or fall back to module defaults
    eps_d = eps_d if eps_d is not None else EPS_D
    eps_sep = eps_sep if eps_sep is not None else EPS_SEP
    eps_fwd_back = eps_fwd_back if eps_fwd_back is not None else EPS_fwd_back
    parallel_guard = parallel_guard if parallel_guard is not None else PARALLEL_GUARD
    """
    Non-jitted wrapper that returns FUES results and intersection points separately.
    This is intended for plotting purposes only.

    Returns
    -------
    fues_result : tuple
        Standard FUES output (e_grid, vlu, policy_1, policy_2, del_a)
    intersections : tuple
        Intersection points (inter_e, inter_v, inter_p1, inter_p2, inter_d)
    """
    # Sort inputs
    idx = np.argsort(e_grid)
    e_grid_sorted = e_grid[idx]
    vlu_sorted = vlu[idx]
    policy_1_sorted = policy_1[idx]
    policy_2_sorted = policy_2[idx]
    del_a_sorted = del_a[idx]

    # Call scan WITH intersection tracking to get both FUES result and intersections
    e_grid_out, keep, intersections = _scan(
        e_grid_sorted,
        vlu_sorted,
        policy_1_sorted,
        policy_2_sorted,
        del_a_sorted,
        m_bar,
        LB,
        endog_mbar,
        padding_mbar,
        True,  # include_intersections
        False,  # not_allow_2lefts - default to True
        eps_d, eps_sep, eps_fwd_back, parallel_guard
    )

    # Extract kept points for FUES result using boolean mask
    env_idx = np.flatnonzero(keep)
    fues_result = (
        e_grid_sorted[env_idx],
        vlu_sorted[env_idx],
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
    vlu,
    a_prime,
    policy_2,
    del_a,
    m_bar,
    LB,
    endog_mbar,
    padding_mbar,
    include_intersections=True,
    not_allow_2lefts=True,
    eps_d=EPS_D,
    eps_sep=EPS_SEP,
    eps_fwd_back=EPS_fwd_back,
    parallel_guard=PARALLEL_GUARD,
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
    vlu : array
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
    # Boolean mask to track kept points (instead of vlu.copy())
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
        #use_intersection_as_k = False  # Consume exactly once
        if use_intersection_as_k and include_intersections:
            k_e = intersection_e
            k_v = intersection_v
            k_a = intersection_a
            k_d = intersection_d
            use_intersection_as_k = False  # Consume exactly once
        else:
            k_e = e_grid[k] if k >= 0 else e_grid[0]
            k_v = vlu[k] if k >= 0 else vlu[0]
            k_a = a_prime[k] if k >= 0 else a_prime[0]
            k_d = del_a[k] if k >= 0 else del_a[0]

        
        # Gradient from tail (k) to head (j) - slope of previous segment
        de_prev = max(eps_d, e_grid[j] - k_e)
        inv_de_prev = 1.0 / de_prev  # Optimization: multiply is faster than divide
        g_jm1 = (vlu[j] - k_v) * inv_de_prev

        # Gradient from head (j) to current point (i+1) - slope of current segment
        de_lead = max(eps_d, e_grid[i + 1] - e_grid[j])
        inv_de_lead = 1.0 / de_lead
        g_1 = (vlu[i + 1] - vlu[j]) * inv_de_lead

        # Jump threshold: either fixed (m_bar) or endogenous based on policy gradients
        M_max = max(np.abs(del_a[j]), np.abs(del_a[i + 1])) + padding_mbar
        
        if not endog_mbar:
            #M_max = m_bar
            #M_max = min(prev_g_tilde_a + padding_mbar, m_bar)
            M_max = m_bar
        #if not endog_mbar and last_was_jump:
            


        # Policy gradient for jump detection
        del_pol = a_prime[i + 1] - a_prime[j]
        del_pol_2 = policy_2[i + 1] - policy_2[j]
        g_tilde_a = np.abs(del_pol * inv_de_lead)
        g_tilde_a_2 = np.abs(del_pol_2 * inv_de_lead)

        # Check for non-monotone policies: if savings rate is decreasing
        del_pol_a = (e_grid[i + 1] - a_prime[i + 1]) - (e_grid[j] - a_prime[j])
        
        del_pol_2 = policy_2[i + 1] - policy_2[j]

        # ============= STEP 2: Classify Current Situation =============
        # Determine turn direction and jump status
        left_turn_any = g_1 > g_jm1
        jump_now = g_tilde_a > M_max or del_pol_2 < 0 or del_pol_a < 0

        if del_pol_2> eps_d:
            if g_tilde_a_2 > M_max:
                jump_now = True

        
        
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
        if (vlu[i + 1] - vlu[j] < 0):
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
        if vlu[i + 1] < 8.260 and vlu[i + 1]> 8.25 and policy_2[i + 1] < 22 and policy_2[i + 1] > 21:
            print("--------------------------------")
            print(f"vlu[i+1] (value at current point): {vlu[i + 1]}")
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
                e_grid, vlu, a_prime, i, j, N, LB, M_max, g_1, eps_d, eps_fwd_back
            )
            
            #keep_i1 = False
            
            if keep_i1 and not last_was_jump:
                created_intersection = False

                # Find backward point on same branch from i+1
                _, idx_b = backward_scan_combined(
                    m_buf,
                    m_head,
                    LB,
                    e_grid,
                    vlu,
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
                        e_grid, vlu, a_prime, policy_2, del_a,
                        j, idx_f if idx_f != -1 else -1, k, j, N
                    )
                    # R: idx_b → i+1 (new branch), fallback i+1 → forward point on same branch
                    # If we can't find a backward point, look forward and extrapolate back
                    safe_extrap = find_safe_extrapolation_point(e_grid, a_prime, i+1, N, M_max, forward=True)
                    R = make_pair_from_indices_or_fallback(
                        e_grid, vlu, a_prime, policy_2, del_a,
                        idx_b if idx_b != -1 else -1, i+1, i+1, safe_extrap, N
                    )
                    
                    # FORCE a crossing strictly inside (e_j, e_{i+1})
                    intr_x, intr_y = _force_crossing_inside(
                        L[0], L[1], L[5], L[6],  # L_x1, L_y1, L_x2, L_y2
                        R[0], R[1], R[5], R[6],  # R_x1, R_y1, R_x2, R_y2
                        e_grid[j], e_grid[i+1], eps_sep, eps_d, parallel_guard
                    )
                    
                    # Fallback: if somehow still nan (shouldn't happen), use midpoint
                    if np.isnan(intr_x):
                        print(f"SCAN DEBUG: intr_x is NaN at i={i}, j={j}. Falling back to midpoint.")
                        mid = 0.5 * (e_grid[j] + e_grid[i+1])
                        denom_L = max(eps_d, L[5] - L[0])
                        denom_R = max(eps_d, R[5] - R[0])
                        sL = (L[6] - L[1]) / denom_L
                        sR = (R[6] - R[1]) / denom_R
                        yL = L[1] + sL * (mid - L[0])
                        yR = R[1] + sR * (mid - R[0])
                        intr_x = mid
                        intr_y = 0.5 * (yL + yR)  # Use average at intersection
                    
                    # ADAPTIVE separation based on interval length
                    interval_length = e_grid[i+1] - e_grid[j]
                    sep = min(eps_sep, 0.25 * interval_length)
                    
                    # Write TWO points with adaptive separation
                    n_new = add_intersection_from_pairs_with_sep(
                        intersections, n_inter, intr_x, intr_y, sep,
                        L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8], L[9],
                        R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9], eps_d
                    )
                    
                    #add_int = False
                    if n_new > n_inter:
                        n_inter = n_new
                        added_intersection_last_iter = True
                        use_intersection_as_k = True
                        
                        # Seed k values from LEFT branch for next iteration
                        denom_L = max(eps_d, L[5] - L[0])
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
                vlu,
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
                        e_grid, vlu, a_prime, policy_2, del_a,
                        -1, -1, k, j, N  # Use same indices for primary and fallback
                    )
                    
                    # R (new branch): m_ind → i+1, fallback i+1 → forward point on same branch
                    # m_ind comes from backward_scan_combined
                    # Note: Intersection is still forced to be in (j, i+1) by _force_crossing_inside
                    safe_extrap = find_safe_extrapolation_point(e_grid, a_prime, i+1, N, M_max, forward=True)
                    R = make_pair_from_indices_or_fallback(
                        e_grid, vlu, a_prime, policy_2, del_a,
                        m_ind if m_ind != -1 else -1, i+1, i+1, safe_extrap, N
                    )
                    
                    # FORCE a crossing strictly inside (e_j, e_{i+1})
                    intr_x, intr_y = _force_crossing_inside(
                        L[0], L[1], L[5], L[6],  # L_x1, L_y1, L_x2, L_y2
                        R[0], R[1], R[5], R[6],  # R_x1, R_y1, R_x2, R_y2
                        e_grid[j], e_grid[i+1], eps_sep, eps_d, parallel_guard
                    )
                    
                    # Fallback: if somehow still nan, use midpoint
                    if np.isnan(intr_x):
                        print(f"SCAN DEBUG: intr_x is NaN at i={i}, j={j}. Falling back to midpoint.")
                        mid = 0.5 * (e_grid[j] + e_grid[i+1])
                        denom_L = max(eps_d, L[5] - L[0])
                        denom_R = max(eps_d, R[5] - R[0])
                        sL = (L[6] - L[1]) / denom_L
                        sR = (R[6] - R[1]) / denom_R
                        yL = L[1] + sL * (mid - L[0])
                        yR = R[1] + sR * (mid - R[0])
                        intr_x = mid
                        intr_y = 0.5 * (yL + yR)  # Use average at intersection
                    
                    # ADAPTIVE separation
                    interval_length = e_grid[i+1] - e_grid[j]
                    sep = min(eps_sep, 0.25 * interval_length)
                    
                    # Write TWO points
                    n_new = add_intersection_from_pairs_with_sep(
                        intersections, n_inter, intr_x, intr_y, sep,
                        L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8], L[9],
                        R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9], eps_d
                    )
                    
                    if n_new > n_inter:
                        n_inter = n_new
                        use_intersection_as_k = True
                        
                        # Seed k from LEFT branch
                        denom_L = max(eps_d, L[5] - L[0])
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
                        e_grid, a_prime, j, j, N, LB, m_bar, eps_d, eps_fwd_back
                    )
                    
                    # Find backward point on same branch from i+1
                    _, idx_back = backward_scan_combined(
                        m_buf, m_head, LB, e_grid, vlu, a_prime,
                        i, j, k, i+1, False, False, False, 0.0, M_max,
                        check_drop=False, eps_d=eps_d
                    )
                    
                    # L (old branch): j → idx_fwd
                    L = make_pair_from_indices_or_fallback(
                        e_grid, vlu, a_prime, policy_2, del_a,
                        j, idx_fwd if found_fwd else -1, k, j, N
                    )
                    
                    # R (new branch): idx_back → i+1, fallback i+1 → forward point on same branch
                    # Note: Intersection is forced to be in (j, i+1) by _force_crossing_inside
                    safe_extrap = find_safe_extrapolation_point(e_grid, a_prime, i+1, N, M_max, forward=True, eps_d=eps_d, eps_fwd_back=eps_fwd_back)
                    R = make_pair_from_indices_or_fallback(
                        e_grid, vlu, a_prime, policy_2, del_a,
                        idx_back if idx_back != -1 else -1, i+1, i+1, safe_extrap, N
                    )
                    
                    # FORCE a crossing strictly inside (e_j, e_{i+1})
                    intr_x, intr_y = _force_crossing_inside(
                        L[0], L[1], L[5], L[6],
                        R[0], R[1], R[5], R[6],
                        e_grid[j], e_grid[i+1], eps_sep, eps_d, parallel_guard
                    )
                    
                    # Fallback to midpoint if needed
                    if np.isnan(intr_x):
                        print(f"SCAN DEBUG: intr_x is NaN at i={i}, j={j}. Falling back to midpoint.")
                        mid = 0.5 * (e_grid[j] + e_grid[i+1])
                        denom_L = max(eps_d, L[5] - L[0])
                        denom_R = max(eps_d, R[5] - R[0])
                        sL = (L[6] - L[1]) / denom_L
                        sR = (R[6] - R[1]) / denom_R
                        yL = L[1] + sL * (mid - L[0])
                        yR = R[1] + sR * (mid - R[0])
                        intr_x = mid
                        intr_y = 0.5 * (yL + yR)  # Use average at intersection
                    
                    # ADAPTIVE separation
                    interval_length = e_grid[i+1] - e_grid[j]
                    sep = min(eps_sep, 0.25 * interval_length)
                    
                    # Write TWO points
                    n_new = add_intersection_from_pairs_with_sep(
                        intersections, n_inter, intr_x, intr_y, sep,
                        L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8], L[9],
                        R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8], R[9], eps_d
                    )
                    
                    if n_new > n_inter:
                        n_inter = n_new
                        use_intersection_as_k = True
                        
                        # Seed k from LEFT branch
                        denom_L = max(eps_d, L[5] - L[0])
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

