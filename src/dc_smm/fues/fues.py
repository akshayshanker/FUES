"""Fast Upper-Envelope Scan (FUES) Algorithm Implementation

This module implements the FUES algorithm from Dobrescu & Shanker (2025) for solving 
discrete-continuous dynamic programming problems using Carroll's endogenous grid method.

Author: Akshay Shanker, 2025, a.shanker@unsw.edu.au
"""

from numba import njit
import numpy as np
from .helpers.math_funcs import (
    circ_put,
    _forced_intersection_twopoint,
    _force_crossing_inside,
)

# Constants - adjusted for float64 numerical stability
EPS_D = 1e-14  # Machine epsilon for float64 is ~2.2e-16, so 1e-14 is safe
EPS_SEP = 1e-08
EPS_fwd_back = 0.5
PARALLEL_GUARD = 1e-10  # Increased for better parallel line detection
TURN_LEFT = 1
TURN_RIGHT = 0
JUMP_YES = 1
JUMP_NO = 0


@njit(inline="always")
def check_same_seg(e_grid, kappa_hat, idx1, idx2, m_bar, eps_d=EPS_D, eps_fwd_back=EPS_fwd_back):
    """Check if two points are on the same segment (no jump in policy between them)."""
    if idx1 < 0 or idx2 < 0 or idx1 >= len(e_grid) or idx2 >= len(e_grid):
        return False
    de = max(eps_d, e_grid[idx2] - e_grid[idx1]) # should give use positive value since e-grid sorted to be increasing
    # Guard against numerical overflow in gradient computation
    if de < eps_d * 10:  # Very close points
        return False
    g_a = np.abs((kappa_hat[idx2] - kappa_hat[idx1]) / de)
    return g_a < m_bar and de < eps_fwd_back


@njit(inline="always")
def find_safe_extrapolation_point(e_grid, a_prime, base_idx, N, m_bar, forward=True, eps_d=EPS_D, eps_fwd_back=EPS_fwd_back):
    """
    Find a safe point for extrapolation that's on the same segment as base_idx.
    Returns the index of a point that doesn't jump from base_idx, or base_idx if none found.

    Todo
    ----
    - Also include a condition that point jumps from "other" segment that is jumped from or to?
    - Should we remove the min 4 condition?
    """
    if forward:
        # Search forward for a point on same branch
        for offset in range(1, min(4, N - base_idx)):
            test_idx = base_idx + offset
            if test_idx < N and check_same_seg(e_grid, a_prime, base_idx, test_idx, m_bar, eps_d, eps_fwd_back):
                return test_idx
    else:
        # Search backward for a point on same branch
        for offset in range(1, min(4, base_idx + 1)):
            test_idx = base_idx - offset
            if test_idx >= 0 and check_same_seg(e_grid, a_prime, test_idx, base_idx, m_bar, eps_d, eps_fwd_back):
                return test_idx
    # If no valid point found, return base_idx (will result in flat extrapolation)
    return base_idx


@njit(inline="always")
def make_pair_from_indices_or_fallback(e, v, a, p2, d, lo_idx, hi_idx, fb_lo, fb_hi, N):
    """
    If forward and backward scans both return points, then returns endpoints (x1,y1,a1,p21,d1) and (x2,y2,a2,p22,d2).

    Otherwise, returns the fallback pair (fb_lo, fb_hi).
    
    If lo_idx or hi_idx is -1, uses the fallback pair (fb_lo, fb_hi).
    """
    # Bounds checking for fallback indices
    fb_lo = max(0, min(fb_lo, N - 1))
    fb_hi = max(0, min(fb_hi, N - 1))
    
    if lo_idx != -1 and hi_idx != -1:
        return (e[lo_idx], v[lo_idx], a[lo_idx], p2[lo_idx], d[lo_idx],
                e[hi_idx], v[hi_idx], a[hi_idx], p2[hi_idx], d[hi_idx])
    else:
        return (e[fb_lo], v[fb_lo], a[fb_lo], p2[fb_lo], d[fb_lo],
                e[fb_hi], v[fb_hi], a[fb_hi], p2[fb_hi], d[fb_hi])


@njit
def backward_scan_combined(
    m_buf, m_head, LB,
    x_dcsn_hat, vlu, kappa,
    j, i_plus_1,
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

    for t in range(LB):
        idx = (m_head - 1 - t) % LB
        cand = m_buf[idx]
        if cand == -1:
            continue

        if cand >= i_plus_1:
            continue
        den_ib = max(eps_d, abs(x_dcsn_hat[i_plus_1] - x_dcsn_hat[cand]))
        den_jb = max(eps_d, abs(x_dcsn_hat[j] - x_dcsn_hat[cand]))
        gq_i1_b = abs(kappa[i_plus_1] - kappa[cand]) / den_ib
        gq_j_b = abs(kappa[j] - kappa[cand]) / den_jb

        if not (gq_i1_b < m_bar and gq_j_b > m_bar):
            continue

        b = cand

        if check_drop:
            w = (x_dcsn_hat[j] - x_dcsn_hat[b]) / den_ib
            v_on_bi1_at_j = vlu[b] + w * (vlu[i_plus_1] - vlu[b])

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
def check_intersection_within_bounds(e_grid, vlu, j, idx_f, i_plus_1, idx_b, eps_d=EPS_D, parallel_guard=PARALLEL_GUARD):
    """
    Check if the natural intersection of segment [j, idx_f] with segment [i+1, idx_b] 
    lies within the segment boundaries [j, i+1].
    
    Computes the raw intersection without forcing it inside bounds, then checks
    if it naturally falls within the interval.
    
    Parameters
    ----------
    e_grid, vlu : arrays
        Grid points and values
    j, idx_f : int
        Indices for the first segment (j to forward point)
    i_plus_1, idx_b : int  
        Indices for the second segment (i+1 to backward point)
    eps_d : float
        Small epsilon for division protection
        
    Returns
    -------
    bool
        True if intersection is within [e_grid[j], e_grid[i_plus_1]]
    """
    # Check if we have valid indices
    if idx_f == -1 or idx_b == -1:
        return False
    
    # Compute the raw intersection using the same logic as _force_crossing_inside
    # but without the clipping step
    L_x1, L_y1 = e_grid[j], vlu[j]
    L_x2, L_y2 = e_grid[idx_f], vlu[idx_f]
    R_x1, R_y1 = e_grid[i_plus_1], vlu[i_plus_1]
    R_x2, R_y2 = e_grid[idx_b], vlu[idx_b]
    
    dxL = L_x2 - L_x1
    dyL = L_y2 - L_y1
    dxR = R_x2 - R_x1
    dyR = R_y2 - R_y1
    
    # Check for parallel lines
    denom = dxL * dyR - dyL * dxR
    if np.abs(denom) < parallel_guard:
        return False  # Lines are parallel or nearly parallel
    
    # Compute intersection point using parametric form
    # Intersection occurs at parameter s on the left line
    s = ((R_x1 - L_x1) * dyR - (R_y1 - L_y1) * dxR) / denom
    
    # Compute the x-coordinate of the intersection
    x_intersect = L_x1 + s * dxL
    
    # Check if intersection is within the segment boundary [j, i+1]
    # Use a small tolerance for numerical precision
    tol = eps_d
    e_lo = e_grid[j] - tol
    e_hi = e_grid[i_plus_1] + tol
    
    # Return true if intersection is within bounds
    if x_intersect >= e_lo and x_intersect <= e_hi:
        return True
    else:
        return False

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
                keep_i1 = True
                break
    
    if not found_forward_same_branch:
        keep_i1 = True

    return keep_i1, idx_f_return, found_forward_same_branch


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
    keep[N - 1] = True

    for i in range(1, N - 1):
        deL = e_grid[i] - e_grid[i - 1]
        deR = e_grid[i + 1] - e_grid[i]
        # protect divisions but keep sign (not needed for abs, but consistent)
        if np.abs(deL) < eps_d:
            deL = eps_d if deL >= 0.0 else -eps_d
        if np.abs(deR) < eps_d:
            deR = eps_d if deR >= 0.0 else -eps_d

        gL = np.abs((a_prime[i] - a_prime[i - 1]) / deL)
        gR = np.abs((a_prime[i + 1] - a_prime[i]) / deR)

        # Drop i only if both sides are true jumps
        if (gL > m_bar) and (gR > m_bar):
            keep[i] = False

    return keep


def FUES(
    e_grid, vlu, policy_1, policy_2, del_a,
    b=1e-10, m_bar=1.0, LB=4, endog_mbar=False, padding_mbar=0.0,
    include_intersections=True,
    return_intersections_separately=False,
    single_intersection=False,
    no_double_jumps=True,
    disable_jump_checks=False,  # NEW: Control manual overrides for jump checks
    eps_d=None, eps_sep=None, eps_fwd_back=None, parallel_guard=None,
):
    """
    Fast Upper-Envelope Scan (FUES) wrapper.

    Computes the upper envelope of future segment choices over an endogenous grid in a
    single pass and returns the retained points. Optionally, it also creates
    explicit intersection points at discrete-choice switches and can return
    those intersections separately for downstream processing.

    The Numba-compiled scanner (`_scan`) provides O(N) time behavior with a
    small, fixed look-back/forward window. This wrapper prepares inputs,
    invokes the scanner, and post-processes the results (merging or returning
    intersections separately, plus a final post-clean step to avoid spurious
   consecutive jumps).

    Parameters
    ----------
    e_grid : ndarray, shape (N,)
        Endogenous decision grid. The inputs are internally sorted by `e_grid`.
    vlu : ndarray, shape (N,)
        Choice-specific value at each grid point (read-only in the scan).
    policy_1 : ndarray, shape (N,)
        Primary policy aligned with `e_grid` (e.g., consumption).
    policy_2 : ndarray, shape (N,)
        Secondary policy aligned with `e_grid` (e.g., asset policy a'). Used
        in jump classification and as payload in intersections.
    del_a : ndarray, shape (N,)
        Policy-gradient-like series used for endogenous jump thresholds.

    b : float, default 1e-10
        Legacy argument retained for signature stability. Not used.
    m_bar : float, default 1.0
        Jump threshold for same-branch tests. If `endog_mbar=True`, the
        threshold adapts using `del_a` with `padding_mbar`.
    LB : int, default 4
        Look-back/forward buffer length used by backward/forward scans.
    endog_mbar : bool, default False
        If True, uses endogenous jump threshold based on `del_a`.
    padding_mbar : float, default 0.0
        Extra padding added to the endogenous threshold.
    include_intersections : bool, default True
        If True, create forced intersections at kept jumps.
    return_intersections_separately : bool, default False
        If True, return a pair `(fues_result, inter_tuple)` instead of a
        single merged result. See Returns.
    single_intersection : bool, default False
        If True, create only one intersection point (on the right) instead of two.
        This reduces the number of points but may affect envelope smoothness.
    disable_jump_checks : bool, default False
        If True, applies manual overrides to disable jump validity checks:
        - Forces keep_i1=False in right turn cases
        - Forces keep_j=True in left turn cases
        Default is False (checks are enabled, no overrides).
    eps_d : float, optional
        Minimum separation between grid points. Defaults to `EPS_D`.
    eps_sep : float, optional
        Minimum separation used when creating intersections. Defaults to
        `EPS_SEP`.
    eps_fwd_back : float, optional
        Proximity threshold for forward/backward scans. Defaults to
        `EPS_fwd_back`.
    parallel_guard : float, optional
        Tolerance to guard against near-parallel segment geometry when forming
        intersections. Defaults to `PARALLEL_GUARD`.

    Returns
    -------
    tuple
        If `return_intersections_separately` is False:
            (e_kept, v_kept, p1_kept, p2_kept, d_kept)

        If `return_intersections_separately` is True:
            (fues_result, inter_tuple)

            where
              - fues_result = (e_kept, v_kept, p1_kept, p2_kept, d_kept)
              - inter_tuple = (e_inter, v_inter, p1_inter, p2_inter, d_inter)

            Each array in `inter_tuple` contains only intersection rows.

    Notes
    -----
    - Inputs are sorted by `e_grid` prior to scanning. Outputs inherit this
      order; merged outputs are resorted after adding intersections.
    - Intersections are forced to lie strictly within open intervals, using
      `eps_sep` and `parallel_guard` to avoid degenerate intersections.
    - When `single_intersection=True`, only one intersection point is created on the
      right side of the crossing, reducing the total number of points but potentially
      affecting envelope smoothness at discrete choice switches.
    - Both policies are used to detect jumps. Policy 1 used in forward and backward scans.

    """
    # Use provided epsilons or fall back to module defaults
    eps_d = eps_d if eps_d is not None else EPS_D
    eps_sep = eps_sep if eps_sep is not None else EPS_SEP
    eps_fwd_back = eps_fwd_back if eps_fwd_back is not None else EPS_fwd_back
    parallel_guard = parallel_guard if parallel_guard is not None else PARALLEL_GUARD
    
    # Ensure float64 precision for all arrays
    e_grid = np.asarray(e_grid, dtype=np.float64)
    vlu = np.asarray(vlu, dtype=np.float64)
    policy_1 = np.asarray(policy_1, dtype=np.float64)
    policy_2 = np.asarray(policy_2, dtype=np.float64)
    del_a = np.asarray(del_a, dtype=np.float64)
    
    idx = np.argsort(e_grid)
    e_grid = e_grid[idx]
    vlu = vlu[idx]
    policy_1 = policy_1[idx]
    policy_2 = policy_2[idx]
    del_a = del_a[idx]

    e_out, keep_scan, intersections = _scan(
        e_grid, vlu, policy_1, policy_2, del_a,
        m_bar, LB, endog_mbar, padding_mbar,
        include_intersections, no_double_jumps, single_intersection,
        disable_jump_checks,
        eps_d, eps_sep, eps_fwd_back, parallel_guard
    )

    env_idx = np.flatnonzero(keep_scan)
    e_kept = e_out[env_idx]
    v_kept = vlu[env_idx]
    p1_kept = policy_1[env_idx]
    p2_kept = policy_2[env_idx]
    d_kept = del_a[env_idx]

    if include_intersections and intersections.shape[0] > 0:
        if return_intersections_separately:
            inter_tuple = (
                intersections[:, 0].copy(),
                intersections[:, 1].copy(),
                intersections[:, 2].copy(),
                intersections[:, 3].copy(),
                intersections[:, 4].copy(),
            )
            fues_result = (e_kept, v_kept, p1_kept, p2_kept, d_kept)
            return fues_result, inter_tuple
        
        n_kept = e_kept.size
        n_inter = intersections.shape[0]
        n_total = n_kept + n_inter

        all_e = np.empty(n_total, dtype=e_kept.dtype)
        all_v = np.empty(n_total, dtype=v_kept.dtype)
        all_p1 = np.empty(n_total, dtype=p1_kept.dtype)
        all_p2 = np.empty(n_total, dtype=p2_kept.dtype)
        all_d = np.empty(n_total, dtype=d_kept.dtype)
        is_inter = np.zeros(n_total, dtype=np.bool_)

        all_e[:n_kept] = e_kept
        all_e[n_kept:] = intersections[:, 0]
        all_v[:n_kept] = v_kept
        all_v[n_kept:] = intersections[:, 1]
        all_p1[:n_kept] = p1_kept
        all_p1[n_kept:] = intersections[:, 2]
        all_p2[:n_kept] = p2_kept
        all_p2[n_kept:] = intersections[:, 3]
        all_d[:n_kept] = d_kept
        all_d[n_kept:] = intersections[:, 4]
        is_inter[n_kept:] = True

        sort_idx = np.argsort(all_e)
        all_e = all_e[sort_idx]
        all_v = all_v[sort_idx]
        all_p1 = all_p1[sort_idx]
        all_p2 = all_p2[sort_idx]
        all_d = all_d[sort_idx]
        is_inter = is_inter[sort_idx]

        post_mask = _postclean_double_jump_mask(all_e, all_p2, m_bar, is_inter, eps_d)

        final_mask = post_mask

        return (all_e[final_mask], all_v[final_mask],
                all_p1[final_mask], all_p2[final_mask], all_d[final_mask])

    is_inter = np.zeros(e_kept.size, dtype=np.bool_)
    post_mask = _postclean_double_jump_mask(e_kept, p2_kept, m_bar, is_inter, eps_d)

    if return_intersections_separately:
        empty = np.zeros(0, dtype=e_kept.dtype)
        inter_tuple = (empty, empty, empty, empty, empty)
        fues_result = (
            e_kept[post_mask],
            v_kept[post_mask],
            p1_kept[post_mask],
            p2_kept[post_mask],
            d_kept[post_mask],
        )
        return fues_result, inter_tuple

    return (e_kept[post_mask], v_kept[post_mask],
            p1_kept[post_mask], p2_kept[post_mask], d_kept[post_mask])


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
    include_intersections=False,
    not_allow_2lefts=True,
    single_intersection=False,
    disable_jump_checks=False,
    left_turn_no_jump_strict=False,
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
    single_intersection : bool
        If True, create only one intersection point (on the right) instead of two
    disable_jump_checks : bool
        If True, applies manual overrides to disable jump validity checks
    left_turn_no_jump_strict : bool
        If True, left turns without jumps use same logic as left turns with jumps
        (backward scan, intersection creation). Default False uses simple pointer advance.

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
    keep = np.ones(N, dtype=np.bool_)

    # Adjust capacity based on whether we're using single or double intersections
    max_inter = (N - 1) if single_intersection else 2 * (N - 1)
    intersections = np.full((max_inter, 5), np.nan)
    n_inter = 0

    use_intersection_as_k = False
    intersection_e = 0.0
    intersection_v = 0.0

    added_intersection_last_iter = False

    m_buf = np.full(LB, -1)
    m_head = 0

    j, k = 0, -1
    last_was_jump = False
    prev_j = 0


    for i in range(N - 2):

        if i <= 1:
            if i == 0:
                j, k = 0, -1
                prev_j = 0
            else:
                prev_j = j
                j, k = i, i - 1
            last_turn_left = False
            last_was_jump = False
            added_intersection_last_iter = False
            continue

        # Compute gradients
        if use_intersection_as_k and include_intersections:
            k_e = intersection_e
            k_v = intersection_v

            use_intersection_as_k = False
        else:
            k_e = e_grid[k] if k >= 0 else e_grid[0]
            k_v = vlu[k] if k >= 0 else vlu[0]


        
        de_prev = max(eps_d, e_grid[j] - k_e)
        inv_de_prev = 1.0 / de_prev
        g_jm1 = (vlu[j] - k_v) * inv_de_prev

        de_lead = max(eps_d, e_grid[i + 1] - e_grid[j])
        inv_de_lead = 1.0 / de_lead
        g_1 = (vlu[i + 1] - vlu[j]) * inv_de_lead

        M_max = max(np.abs(del_a[j]), np.abs(del_a[i + 1])) + padding_mbar
        
        if not endog_mbar:
            M_max = m_bar
            


        del_pol = a_prime[i + 1] - a_prime[j]
        del_pol_2 = policy_2[i + 1] - policy_2[j]
        g_tilde_a = np.abs(del_pol * inv_de_lead)
        g_tilde_a_2 = np.abs(del_pol_2 * inv_de_lead)

        del_pol_a = (e_grid[i + 1] - a_prime[i + 1]) - (e_grid[j] - a_prime[j])
        
        del_pol_2 = policy_2[i + 1] - policy_2[j]

        # Classify turn direction and jump status
        left_turn_any = g_1 > g_jm1
        jump_now = (g_tilde_a > M_max) #or (g_tilde_a_2 > M_max)
        #jump_now = g_tilde_a > M_max

        #if del_pol_2> eps_d:
        #    if g_tilde_a_2 > M_max:
        #        jump_now = True

        
        
        left_turn_jump = left_turn_any and jump_now
        left_turn_no_jump = left_turn_any and (not jump_now)
        right_turn_jump = (not left_turn_any) and jump_now
        right_turn_no_jump = (not left_turn_any) and (not jump_now)



        added_intersection_last_iter = False

        # Case B: Value fall
        if (vlu[i + 1] - vlu[j] < 0):
            keep[i + 1] = False
            use_intersection_as_k = False
            m_head = circ_put(m_buf, m_head, i + 1)
            last_turn_left = False
            last_was_jump = False
            continue

        # Case A: Right-turn with jump
        if right_turn_jump:
            keep_i1, idx_f, found_forward_same_branch = forward_scan_case_a(
                e_grid, vlu, a_prime, i, j, N, LB, M_max, g_1, eps_d, eps_fwd_back
            )
            # Apply manual override only if disable_jump_checks is True
            if disable_jump_checks:
                keep_i1 = False
            if keep_i1 and not last_was_jump:
                created_intersection = False

                _, idx_b = backward_scan_combined(
                    m_buf,
                    m_head,
                    LB,
                    e_grid,
                    vlu,
                    a_prime,
                    j,
                    i + 1,
                    M_max,
                    check_drop=False,
                    eps_d=eps_d,
                )
                
                # Check if we should create an intersection
                create_intersection = include_intersections and not last_was_jump
                
                # Augmented keep_i1 condition:
                # If found_forward_same_branch is true and both idx_f and idx_b are valid,
                # check if intersection of [j, idx_f] with [i+1, idx_b] is within segment boundary
                if found_forward_same_branch and idx_f != -1 and idx_b != -1:
                    intersection_within = check_intersection_within_bounds(
                        e_grid, vlu, j, idx_f, i + 1, idx_b, eps_d
                    )
                    if not intersection_within:  
                        # If intersection is OUTSIDE bounds:
                        # - Don't keep i+1
                        # - Don't create intersection
                        keep_i1 = False
                        create_intersection = False
        
                # Only create intersection if it's within bounds
                if create_intersection and keep_i1:
                    L = make_pair_from_indices_or_fallback(
                        e_grid, vlu, a_prime, policy_2, del_a,
                        j, idx_f if idx_f != -1 else -1, k, j, N
                    )
                    safe_extrap = find_safe_extrapolation_point(e_grid, a_prime, i+1, N, M_max, forward=True, eps_d=eps_d, eps_fwd_back=eps_fwd_back)
                    R = make_pair_from_indices_or_fallback(
                        e_grid, vlu, a_prime, policy_2, del_a,
                        idx_b if idx_b != -1 else -1, i+1, i+1, safe_extrap, N
                    )

                    n_inter, intersection_e, intersection_v, _, _, added = _forced_intersection_twopoint(
                        intersections, n_inter,
                        e_grid[j], e_grid[i+1], -1.0,   # sep_cap disabled
                        L, R,
                        eps_d, eps_sep, parallel_guard,
                        i, j, single_intersection
                    )

                    if added:
                        added_intersection_last_iter = True
                        use_intersection_as_k = True
                        created_intersection = True

                # Only update k, j, etc. if keep_i1 is still true after augmented check
                if keep_i1:
                    k = j
                    prev_j = j
                    j = i + 1
                    last_turn_left = True
                    if not created_intersection:
                        use_intersection_as_k = False
            if not keep_i1:
                keep[i + 1] = False
                m_head = circ_put(m_buf, m_head, i + 1)
                use_intersection_as_k = False
                last_turn_left = False
            if keep_i1:
                last_was_jump=jump_now
            else:
                last_was_jump=False
            continue

        

        # Case C: Left turn (with jump, or no-jump if strict mode)
        if left_turn_jump or (left_turn_no_jump and left_turn_no_jump_strict):
            keep_j, m_ind = backward_scan_combined(
                m_buf,
                m_head,
                LB,
                e_grid,
                vlu,
                a_prime,
                j,
                i + 1,
                M_max,
                check_drop=True,
                eps_d=eps_d,
            )
            # Apply manual override only if disable_jump_checks is True
            if disable_jump_checks:
                keep_j = True
            # Case C.1: Left turn with j dropped
            if not keep_j and left_turn_jump:
                keep[j] = False
                m_head = circ_put(m_buf, m_head, j)
                use_intersection_as_k = False
                created_intersection = False
                added_intersection_last_iter = False
                if include_intersections and not last_was_jump:
                    L = make_pair_from_indices_or_fallback(
                        e_grid, vlu, a_prime, policy_2, del_a,
                        -1, -1, k, j, N
                    )
                    safe_extrap = find_safe_extrapolation_point(e_grid, a_prime, i+1, N, M_max, forward=True, eps_d=eps_d, eps_fwd_back=eps_fwd_back)
                    R = make_pair_from_indices_or_fallback(
                        e_grid, vlu, a_prime, policy_2, del_a,
                        m_ind if m_ind != -1 else -1, i+1, i+1, safe_extrap, N
                    )

                    n_inter, intersection_e, intersection_v, _, _, added = _forced_intersection_twopoint(
                        intersections, n_inter,
                        e_grid[j], e_grid[i+1], -1.0,
                        L, R,
                        eps_d, eps_sep, parallel_guard,
                        i, j, single_intersection
                    )

                    if added:
                        use_intersection_as_k = True
                        created_intersection = True

                prev_j = k
                j = i + 1

            # Case C.2: Left turn with j kept
            else:
                if  not_allow_2lefts and jump_now and last_was_jump:
                    keep[j] = False
                    if include_intersections and added_intersection_last_iter and n_inter > 0:
                        # Adjust based on whether we're using single or double intersections
                        n_inter = n_inter - 1 if single_intersection else n_inter - 2
                    
                    j = prev_j

                use_intersection_as_k = False
                if include_intersections and not last_was_jump:
                    found_fwd, idx_fwd = find_forward_same_branch(
                        e_grid, a_prime, j, j, N, LB, m_bar, eps_d, eps_fwd_back
                    )
                    
                    _, idx_back = backward_scan_combined(
                        m_buf, m_head, LB, e_grid, vlu, a_prime,
                        j, i+1, M_max,
                        check_drop=False, eps_d=eps_d
                    )
                    
                    L = make_pair_from_indices_or_fallback(
                        e_grid, vlu, a_prime, policy_2, del_a,
                        j, idx_fwd if found_fwd else -1, k, j, N
                    )
                    
                    safe_extrap = find_safe_extrapolation_point(e_grid, a_prime, i+1, N, M_max, forward=True, eps_d=eps_d, eps_fwd_back=eps_fwd_back)
                    R = make_pair_from_indices_or_fallback(
                        e_grid, vlu, a_prime, policy_2, del_a,
                        idx_back if idx_back != -1 else -1, i+1, i+1, safe_extrap, N
                    )
                    
                    n_inter, intersection_e, intersection_v, _, _, added = _forced_intersection_twopoint(
                        intersections, n_inter,
                        e_grid[j], e_grid[i+1], -1.0,
                        L, R,
                        eps_d, eps_sep, parallel_guard,
                        i, j, single_intersection
                    )
                    
                    if added:
                        use_intersection_as_k = True
                        created_intersection = True
                
                if  not_allow_2lefts and jump_now and last_was_jump:
                    k = j
                    j = i + 1
                else:
                    k = j
                    prev_j = j
                    j = i + 1
            
            last_turn_left = True
            last_was_jump = jump_now
            continue

        # Case R: Right turn without jump (or left turn no-jump if not strict)
        if right_turn_no_jump or (left_turn_no_jump and not left_turn_no_jump_strict):
            k = j
            prev_j = j
            j = i + 1
            use_intersection_as_k = False
            last_turn_left = False
            last_was_jump = False
            continue

    return e_grid, keep, intersections[:n_inter, :]

