"""Fast Upper-Envelope Scan (FUES) Algorithm

Implements the FUES algorithm from Dobrescu & Shanker
(2022) for solving discrete-continuous dynamic
programming problems using Carroll's endogenous grid
method.

Author: Akshay Shanker, 2025, a.shanker@unsw.edu.au
"""

from numba import njit
import numpy as np
from .helpers.math_funcs import (
    circ_put,
    _forced_intersection_twopoint,
    _force_crossing_inside,
    _merge_sorted_with_few,
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
def check_same_seg(
        x_dcsn_hat,
        kappa_hat,
        idx1,
        idx2,
        m_bar,
        eps_d=EPS_D,
        eps_fwd_back=EPS_fwd_back):
    """Check if two points are on the same segment."""
    if idx1 < 0 or idx2 < 0 or idx1 >= len(
            x_dcsn_hat) or idx2 >= len(x_dcsn_hat):
        return False
    # should give use positive value since e-grid sorted to be increasing
    de = max(eps_d, x_dcsn_hat[idx2] - x_dcsn_hat[idx1])
    # Guard against numerical overflow in gradient computation
    if de < eps_d * 10:  # Very close points
        return False
    g_a = np.abs((kappa_hat[idx2] - kappa_hat[idx1]) / de)
    return g_a < m_bar and de < eps_fwd_back


@njit(inline="always")
def find_safe_extrapolation_point(
        x_dcsn_hat,
        kappa_hat,
        base_idx,
        N,
        m_bar,
        forward=True,
        eps_d=EPS_D,
        eps_fwd_back=EPS_fwd_back):
    """Find a safe extrapolation point on same segment.

    Returns the index of a point that doesn't jump
    from base_idx, or base_idx if none found.

    Todo
    ----
    - Also include a condition that point jumps
      from "other" segment?
    - Should we remove the min 4 condition?
    """
    if forward:
        # Search forward for a point on same branch
        for offset in range(1, min(4, N - base_idx)):
            test_idx = base_idx + offset
            if test_idx < N and check_same_seg(
                    x_dcsn_hat,
                    kappa_hat,
                    base_idx,
                    test_idx,
                    m_bar,
                    eps_d,
                    eps_fwd_back):
                return test_idx
    else:
        # Search backward for a point on same branch
        for offset in range(1, min(4, base_idx + 1)):
            test_idx = base_idx - offset
            if test_idx >= 0 and check_same_seg(
                    x_dcsn_hat,
                    kappa_hat,
                    test_idx,
                    base_idx,
                    m_bar,
                    eps_d,
                    eps_fwd_back):
                return test_idx
    # If no valid point found, return base_idx (will result in flat
    # extrapolation)
    return base_idx


@njit(inline="always")
def make_pair_from_indices_or_fallback(
        e, v, a, p2, d, lo_idx, hi_idx, fb_lo, fb_hi, N):
    """Return endpoints from forward/backward scans.

    If both scans return points, returns
    (x1,y1,a1,p21,d1) and (x2,y2,a2,p22,d2).
    Otherwise returns the fallback pair.
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


@njit(cache=True)
def backward_scan_combined(
    m_buf, m_head, LB,
    x_dcsn_hat, v_hat, kappa_hat,
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
    below the chord joining p_b and p_{i+1} on the (v_hat, x_dcsn_hat) plane.
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
        gq_i1_b = abs(kappa_hat[i_plus_1] - kappa_hat[cand]) / den_ib
        gq_j_b = abs(kappa_hat[j] - kappa_hat[cand]) / den_jb

        if not (gq_i1_b < m_bar and gq_j_b > m_bar):
            continue

        b = cand

        if check_drop:
            w = (x_dcsn_hat[j] - x_dcsn_hat[b]) / den_ib
            v_on_bi1_at_j = v_hat[b] + w * (v_hat[i_plus_1] - v_hat[b])

            if v_hat[j] < v_on_bi1_at_j - 1e-14:
                keep_j = False
        break

    return keep_j, b


@njit(cache=True)
def find_forward_same_branch(
        x_dcsn_hat,
        kappa_hat,
        start_idx,
        j_idx,
        N,
        LB,
        m_bar,
        eps_d=EPS_D,
        eps_fwd_back=EPS_fwd_back):
    """Find the first point in forward scan that's on same branch.

    Returns found flag and index.
    """
    for f in range(min(LB, N - start_idx - 1)):
        if start_idx + 1 + f >= N:
            break
        de = max(eps_d, x_dcsn_hat[start_idx + 1 + f] - x_dcsn_hat[j_idx])
        g_a = np.abs((kappa_hat[start_idx + 1 + f] - kappa_hat[j_idx]) / de)
        if g_a < m_bar and de < eps_fwd_back:
            return True, start_idx + 1 + f
    return False, -1


@njit(cache=True)
def check_intersection_within_bounds(
        x_dcsn_hat,
        v_hat,
        j,
        idx_f,
        i_plus_1,
        idx_b,
        eps_d=EPS_D,
        parallel_guard=PARALLEL_GUARD):
    """Check if intersection of two segments is in bounds.

    Tests whether the natural intersection of segment
    [j, idx_f] with segment [i+1, idx_b] lies within
    [x_dcsn_hat[j], x_dcsn_hat[i_plus_1]].

    Parameters
    ----------
    x_dcsn_hat, v_hat : arrays
        Grid points and values.
    j, idx_f : int
        Indices for the first segment.
    i_plus_1, idx_b : int
        Indices for the second segment.
    eps_d : float
        Small epsilon for division protection.

    Returns
    -------
    bool
        True if intersection is within bounds.
    """
    # Check if we have valid indices
    if idx_f == -1 or idx_b == -1:
        return False

    # Raw intersection (same logic as
    # _force_crossing_inside, without clipping)
    L_x1, L_y1 = x_dcsn_hat[j], v_hat[j]
    L_x2, L_y2 = x_dcsn_hat[idx_f], v_hat[idx_f]
    R_x1, R_y1 = x_dcsn_hat[i_plus_1], v_hat[i_plus_1]
    R_x2, R_y2 = x_dcsn_hat[idx_b], v_hat[idx_b]

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
    e_lo = x_dcsn_hat[j] - tol
    e_hi = x_dcsn_hat[i_plus_1] + tol

    # Return true if intersection is within bounds
    if x_intersect >= e_lo and x_intersect <= e_hi:
        return True
    else:
        return False


@njit(cache=True)
def forward_scan_case_a(
        x_dcsn_hat,
        v_hat,
        kappa_hat,
        i,
        j,
        N,
        LB,
        m_bar,
        g_1,
        eps_d=EPS_D,
        eps_fwd_back=EPS_fwd_back):
    """Forward scan for Case A (right-turn jump).

    Checks if i+1 should be kept by finding a future
    point f on the same branch as j and comparing
    value gradients. If g_1 > g_f, i+1 lies above the
    extrapolated line from j to f, so we keep it.

    Parameters
    ----------
    x_dcsn_hat, v_hat, kappa_hat : arrays
        Grid points, values, and policies.
    i, j : int
        Loop counter and last kept point.
    N : int
        Total number of grid points.
    LB : int
        Lookback/forward buffer size.
    m_bar : float
        Jump threshold for same-branch detection.
    g_1 : float
        Value gradient from j to i+1.

    Returns
    -------
    keep_i1 : bool
        Whether to keep point i+1.
    idx_f : int
        Forward point on same branch (-1 if none).
    """
    idx_f = -1
    idx_f_return = -1
    keep_i1 = False
    found_forward_same_branch = False

    for f in range(LB):
        if i + 2 + f >= N:  # CRITICAL: Add bounds check
            break
        de = max(eps_d, x_dcsn_hat[i + 2 + f] - x_dcsn_hat[j])
        # sde_1 = max(eps_d, x_dcsn_hat[i + 1] - x_dcsn_hat[j])
        g_f_a = np.abs((kappa_hat[j] - kappa_hat[i + 2 + f]) / de)
        idx_f = i + 2 + f  # Store actual grid index
        de_jump = max(eps_d, x_dcsn_hat[idx_f] - x_dcsn_hat[i + 1])
        g_jump = np.abs((kappa_hat[i + 1] - kappa_hat[idx_f]) / de_jump)

        if g_f_a < m_bar and de < eps_fwd_back and g_jump >= m_bar:

            found_forward_same_branch = True
            idx_f_return = idx_f
            # Compute g_f_v_hat for this point (de already equals max(eps_d,
            # x_dcsn_hat[i+2+f]-x_dcsn_hat[j]))
            g_f_v_at_idx = (v_hat[i + 2 + f] - v_hat[j]) / de
            if g_1 > g_f_v_at_idx:
                keep_i1 = True
                break

    if not found_forward_same_branch:
        keep_i1 = True

    return keep_i1, idx_f_return, found_forward_same_branch


@njit(cache=True)
def _postclean_double_jump_mask(
        x_dcsn_hat,
        x_cntn_hat,
        m_bar,
        skip_mask,
        eps_d=EPS_D):
    """
    Keep[i] == False  iff BOTH neighbors of i are policy jumps > m_bar.
    First and last points are always kept. Points with skip_mask[i]==True
    (e.g., intersection rows) are always kept.

    Parameters
    ----------
    x_dcsn_hat : 1d array (sorted)
        Endogenous decision grid.
    x_cntn_hat : 1d array
        Continuation/exogenous grid (used for jump detection).
    m_bar  : float
    skip_mask : 1d bool array with same length as x_dcsn_hat
                True -> never drop (e.g., intersection rows)

    Returns
    -------
    keep : 1d bool array
    """
    N = x_dcsn_hat.size
    keep = np.ones(N, dtype=np.bool_)
    if N <= 2:
        return keep

    # Endpoints: always keep
    keep[0] = True
    keep[N - 1] = True

    for i in range(1, N - 1):
        deL = x_dcsn_hat[i] - x_dcsn_hat[i - 1]
        deR = x_dcsn_hat[i + 1] - x_dcsn_hat[i]
        # protect divisions but keep sign (not needed for abs, but consistent)
        if np.abs(deL) < eps_d:
            deL = eps_d if deL >= 0.0 else -eps_d
        if np.abs(deR) < eps_d:
            deR = eps_d if deR >= 0.0 else -eps_d

        gL = np.abs((x_cntn_hat[i] - x_cntn_hat[i - 1]) / deL)
        gR = np.abs((x_cntn_hat[i + 1] - x_cntn_hat[i]) / deR)

        # Drop i only if both sides are true jumps
        if (gL > m_bar) and (gR > m_bar):
            keep[i] = False

    return keep


def FUES(
    x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat, del_a=None,
    m_bar=1.0, LB=4, endog_mbar=False, padding_mbar=0.0,
    include_intersections=True,
    return_intersections_separately=False,
    single_intersection=False,
    no_double_jumps=True,
    disable_jump_checks=False,
    assume_sorted=False,
    eps_d=None, eps_sep=None, eps_fwd_back=None, parallel_guard=None,
):
    """Fast Upper-Envelope Scan (FUES) wrapper.

    Computes the upper envelope in a single pass
    and returns retained points.  Optionally creates
    intersection points at discrete-choice switches.

    Parameters
    ----------
    x_dcsn_hat : ndarray (N,)
        Unrefined endogenous decision grid.
    v_hat : ndarray (N,)
        Unrefined value correspondence.
    kappa_hat : ndarray (N,)
        Unrefined primary control (e.g. consumption).
    x_cntn_hat : ndarray (N,)
        Continuation / exogenous grid (e.g. next-period
        assets). Used for jump classification.
    del_a : ndarray (N,), optional
        Policy-gradient series for endogenous thresholds.
        Required when ``endog_mbar=True``.
    m_bar : float, default 1.0
        Jump threshold for same-branch tests.
    LB : int, default 4
        Look-back/forward buffer length.
    endog_mbar : bool, default False
        Use endogenous jump threshold from `del_a`.
    padding_mbar : float, default 0.0
        Extra padding for endogenous threshold.
    include_intersections : bool, default True
        Create forced intersections at kept jumps.
    return_intersections_separately : bool
        If True, return ``(fues_result, inter_tuple)``.
    single_intersection : bool, default False
        Create only one intersection per crossing.
    disable_jump_checks : bool, default False
        Override jump validity checks.
    eps_d : float, optional
        Minimum grid-point separation.
    eps_sep : float, optional
        Minimum separation for intersections.
    eps_fwd_back : float, optional
        Proximity threshold for fwd/bwd scans.
    parallel_guard : float, optional
        Guard against near-parallel segments.

    Returns
    -------
    tuple
        ``(x_dcsn_ref, v_ref, kappa_ref,
        x_cntn_ref, del_a_ref)``
        or ``(fues_result, inter_tuple)`` when
        ``return_intersections_separately=True``.
    """
    # Use provided epsilons or fall back to module defaults
    if eps_d is None:
        eps_d = EPS_D
    if eps_sep is None:
        eps_sep = EPS_SEP
    if eps_fwd_back is None:
        eps_fwd_back = EPS_fwd_back
    if parallel_guard is None:
        parallel_guard = PARALLEL_GUARD

    # Ensure float64 precision for all arrays
    x_dcsn_hat = np.asarray(x_dcsn_hat, dtype=np.float64)
    v_hat = np.asarray(v_hat, dtype=np.float64)
    kappa_hat = np.asarray(kappa_hat, dtype=np.float64)
    x_cntn_hat = np.asarray(x_cntn_hat, dtype=np.float64)

    if del_a is None:
        if endog_mbar:
            raise ValueError("del_a is required when endog_mbar=True")
        del_a = np.zeros_like(x_dcsn_hat)
    else:
        del_a = np.asarray(del_a, dtype=np.float64)

    if not assume_sorted and not np.all(np.diff(x_dcsn_hat) >= 0):
        idx = np.argsort(x_dcsn_hat)
        x_dcsn_hat = x_dcsn_hat[idx]
        v_hat = v_hat[idx]
        kappa_hat = kappa_hat[idx]
        x_cntn_hat = x_cntn_hat[idx]
        del_a = del_a[idx]

    e_out, keep_scan, intersections = _scan(
        x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat, del_a,
        m_bar, LB, endog_mbar, padding_mbar,
        include_intersections, no_double_jumps, single_intersection,
        disable_jump_checks,
        eps_d, eps_sep, eps_fwd_back, parallel_guard
    )

    env_idx = np.flatnonzero(keep_scan)
    x_dcsn_ref = e_out[env_idx]
    v_ref = v_hat[env_idx]
    kappa_ref = kappa_hat[env_idx]
    x_cntn_ref = x_cntn_hat[env_idx]
    del_a_ref = del_a[env_idx]

    if include_intersections and intersections.shape[0] > 0:
        if return_intersections_separately:
            inter_tuple = (
                intersections[:, 0].copy(),
                intersections[:, 1].copy(),
                intersections[:, 2].copy(),
                intersections[:, 3].copy(),
                intersections[:, 4].copy(),
            )
            fues_result = (
                x_dcsn_ref,
                v_ref,
                kappa_ref,
                x_cntn_ref,
                del_a_ref)
            return fues_result, inter_tuple

        (all_e, all_v, all_p1,
         all_p2, all_d, is_inter) = _merge_sorted_with_few(
            x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, del_a_ref,
            intersections[:, 0], intersections[:, 1], intersections[:, 2],
            intersections[:, 3], intersections[:, 4],
        )

        post_mask = _postclean_double_jump_mask(
            all_e, all_p2, m_bar, is_inter, eps_d)

        return (all_e[post_mask], all_v[post_mask],
                all_p1[post_mask], all_p2[post_mask], all_d[post_mask])

    is_inter = np.zeros(x_dcsn_ref.size, dtype=np.bool_)
    post_mask = _postclean_double_jump_mask(
        x_dcsn_ref, x_cntn_ref, m_bar, is_inter, eps_d)

    if return_intersections_separately:
        empty = np.zeros(0, dtype=x_dcsn_ref.dtype)
        inter_tuple = (empty, empty, empty, empty, empty)
        fues_result = (
            x_dcsn_ref[post_mask],
            v_ref[post_mask],
            kappa_ref[post_mask],
            x_cntn_ref[post_mask],
            del_a_ref[post_mask],
        )
        return fues_result, inter_tuple

    return (x_dcsn_ref[post_mask], v_ref[post_mask],
            kappa_ref[post_mask], x_cntn_ref[post_mask], del_a_ref[post_mask])


@njit(cache=True)
def _scan(
    x_dcsn_hat,
    v_hat,
    kappa_hat,
    x_cntn_hat,
    del_a,
    m_bar,
    LB,
    endog_mbar,
    padding_mbar,
    include_intersections=False,
    not_allow_2lefts=True,
    single_intersection=False,
    disable_jump_checks=False,
    eps_d=EPS_D,
    eps_sep=EPS_SEP,
    eps_fwd_back=EPS_fwd_back,
    parallel_guard=PARALLEL_GUARD,
):
    """Core FUES scan: single-pass upper envelope.

    Maintains indices k (tail), j (head), i+1 (current)
    and classifies each triplet by secant turn direction
    and policy jump status.

    Parameters
    ----------
    x_dcsn_hat : array
        Sorted endogenous decision grid.
    v_hat : array
        Value correspondence (read-only).
    kappa_hat : array
        Primary control (e.g. consumption).
    x_cntn_hat : array
        Continuation grid (e.g. next-period assets).
    del_a : array
        Policy gradient (for endogenous m_bar).
    m_bar : float
        Jump threshold.
    LB : int
        Lookback buffer size.
    endog_mbar : bool
        Use endogenous jump threshold.
    padding_mbar : float
        Padding for endogenous threshold.
    include_intersections : bool
        Track intersection points.
    single_intersection : bool
        One intersection per crossing.
    disable_jump_checks : bool
        Override jump validity checks.

    Returns
    -------
    x_dcsn_hat : array
        Original grid (unchanged).
    keep : bool array
        Mask of points to keep.
    intersections : ndarray
        Intersection rows (n_inter, 5).
    """

    N = x_dcsn_hat.size
    keep = np.ones(N, dtype=np.bool_)

    # Adjust capacity based on whether we're using single or double
    # intersections
    max_inter = (N - 1) if single_intersection else 2 * (N - 1)
    intersections = np.empty((max_inter, 5))
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

    # Pre-compute abs(del_a) for endogenous m_bar (avoids per-iteration
    # np.abs)
    abs_del_a = np.abs(del_a) if endog_mbar else del_a

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
            k_e = x_dcsn_hat[k] if k >= 0 else x_dcsn_hat[0]
            k_v = v_hat[k] if k >= 0 else v_hat[0]

        de_prev = max(eps_d, x_dcsn_hat[j] - k_e)
        inv_de_prev = 1.0 / de_prev
        g_jm1 = (v_hat[j] - k_v) * inv_de_prev

        de_lead = max(eps_d, x_dcsn_hat[i + 1] - x_dcsn_hat[j])
        inv_de_lead = 1.0 / de_lead
        g_1 = (v_hat[i + 1] - v_hat[j]) * inv_de_lead

        if endog_mbar:
            M_max = max(abs_del_a[j], abs_del_a[i + 1]) + padding_mbar
        else:
            M_max = m_bar

        del_pol = kappa_hat[i + 1] - kappa_hat[j]
        g_tilde_a = np.abs(del_pol * inv_de_lead)

        # Classify turn direction and jump status
        left_turn_any = g_1 > g_jm1
        jump_now = g_tilde_a > M_max

        left_turn_jump = left_turn_any and jump_now
        right_turn_jump = (not left_turn_any) and jump_now

        added_intersection_last_iter = False

        # Case B: Value fall
        if (v_hat[i + 1] - v_hat[j] < 0):
            keep[i + 1] = False
            use_intersection_as_k = False
            m_head = circ_put(m_buf, m_head, i + 1)
            last_turn_left = False
            last_was_jump = False
            continue

        # Case A: Right-turn with jump
        if right_turn_jump:
            (keep_i1, idx_f,
             found_forward_same_branch) = forward_scan_case_a(
                x_dcsn_hat, v_hat, kappa_hat,
                i, j, N, LB, M_max, g_1,
                eps_d, eps_fwd_back)
            # Apply manual override only if disable_jump_checks is True
            if disable_jump_checks:
                keep_i1 = False
            if keep_i1 and not last_was_jump:
                created_intersection = False

                _, idx_b = backward_scan_combined(
                    m_buf,
                    m_head,
                    LB,
                    x_dcsn_hat,
                    v_hat,
                    kappa_hat,
                    j,
                    i + 1,
                    M_max,
                    check_drop=False,
                    eps_d=eps_d,
                )

                # Check if we should create an intersection
                create_intersection = (
                    include_intersections and not last_was_jump)

                # Augmented keep_i1: if forward same
                # branch found and both idx valid, check
                # intersection is within segment boundary
                if found_forward_same_branch and idx_f != -1 and idx_b != -1:
                    intersection_within = check_intersection_within_bounds(
                        x_dcsn_hat, v_hat, j, idx_f, i + 1, idx_b, eps_d
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
                        x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat, del_a,
                        j, idx_f if idx_f != -1 else -1, k, j, N
                    )
                    safe_extrap = find_safe_extrapolation_point(
                        x_dcsn_hat,
                        kappa_hat,
                        i + 1,
                        N,
                        M_max,
                        forward=True,
                        eps_d=eps_d,
                        eps_fwd_back=eps_fwd_back)
                    R = make_pair_from_indices_or_fallback(
                        x_dcsn_hat,
                        v_hat,
                        kappa_hat,
                        x_cntn_hat,
                        del_a,
                        idx_b if idx_b != -1 else -1,
                        i + 1,
                        i + 1,
                        safe_extrap,
                        N)

                    (n_inter, intersection_e,
                     intersection_v, _, _, added
                     ) = _forced_intersection_twopoint(
                        intersections, n_inter,
                        x_dcsn_hat[j],
                        x_dcsn_hat[i + 1],
                        -1.0,  # sep_cap disabled
                        L, R,
                        eps_d, eps_sep,
                        parallel_guard,
                        i, j, single_intersection,
                    )

                    if added:
                        added_intersection_last_iter = True
                        use_intersection_as_k = True
                        created_intersection = True

                # Only update k, j, etc. if keep_i1 is still true after
                # augmented check
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
                last_was_jump = jump_now
            else:
                last_was_jump = False
            continue

        # Case C: Left turn
        if left_turn_jump:
            keep_j, m_ind = backward_scan_combined(
                m_buf,
                m_head,
                LB,
                x_dcsn_hat,
                v_hat,
                kappa_hat,
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
                        x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat, del_a,
                        -1, -1, k, j, N
                    )
                    safe_extrap = find_safe_extrapolation_point(
                        x_dcsn_hat,
                        kappa_hat,
                        i + 1,
                        N,
                        M_max,
                        forward=True,
                        eps_d=eps_d,
                        eps_fwd_back=eps_fwd_back)
                    R = make_pair_from_indices_or_fallback(
                        x_dcsn_hat,
                        v_hat,
                        kappa_hat,
                        x_cntn_hat,
                        del_a,
                        m_ind if m_ind != -1 else -1,
                        i + 1,
                        i + 1,
                        safe_extrap,
                        N)

                    (n_inter, intersection_e,
                     intersection_v, _, _, added
                     ) = _forced_intersection_twopoint(
                        intersections, n_inter,
                        x_dcsn_hat[j],
                        x_dcsn_hat[i + 1],
                        -1.0,
                        L, R,
                        eps_d, eps_sep,
                        parallel_guard,
                        i, j, single_intersection,
                    )

                    if added:
                        use_intersection_as_k = True
                        created_intersection = True

                prev_j = k
                j = i + 1

            # Case C.2: Left turn with j kept
            else:
                if not_allow_2lefts and jump_now and last_was_jump:
                    keep[j] = False
                    if (include_intersections
                            and added_intersection_last_iter
                            and n_inter > 0):
                        # Adjust based on whether we're using single or double
                        # intersections
                        if single_intersection:
                            n_inter -= 1
                        else:
                            n_inter -= 2

                    j = prev_j

                use_intersection_as_k = False
                if include_intersections and not last_was_jump:
                    found_fwd, idx_fwd = (
                        find_forward_same_branch(
                            x_dcsn_hat, kappa_hat,
                            j, j, N, LB, m_bar,
                            eps_d, eps_fwd_back))

                    _, idx_back = backward_scan_combined(
                        m_buf, m_head, LB, x_dcsn_hat, v_hat, kappa_hat,
                        j, i + 1, M_max,
                        check_drop=False, eps_d=eps_d
                    )

                    L = make_pair_from_indices_or_fallback(
                        x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat, del_a,
                        j, idx_fwd if found_fwd else -1, k, j, N
                    )

                    safe_extrap = find_safe_extrapolation_point(
                        x_dcsn_hat,
                        kappa_hat,
                        i + 1,
                        N,
                        M_max,
                        forward=True,
                        eps_d=eps_d,
                        eps_fwd_back=eps_fwd_back)
                    R = make_pair_from_indices_or_fallback(
                        x_dcsn_hat,
                        v_hat,
                        kappa_hat,
                        x_cntn_hat,
                        del_a,
                        idx_back if idx_back != -1 else -1,
                        i + 1,
                        i + 1,
                        safe_extrap,
                        N)

                    (n_inter, intersection_e,
                     intersection_v, _, _, added
                     ) = _forced_intersection_twopoint(
                        intersections, n_inter,
                        x_dcsn_hat[j],
                        x_dcsn_hat[i + 1],
                        -1.0,
                        L, R,
                        eps_d, eps_sep,
                        parallel_guard,
                        i, j, single_intersection,
                    )

                    if added:
                        use_intersection_as_k = True
                        created_intersection = True

                if not_allow_2lefts and jump_now and last_was_jump:
                    k = j
                    j = i + 1
                else:
                    k = j
                    prev_j = j
                    j = i + 1

            last_turn_left = True
            last_was_jump = jump_now
            continue

        # Case R: No jump (right or left turn without policy jump)
        if not jump_now:
            k = j
            prev_j = j
            j = i + 1
            use_intersection_as_k = False
            last_turn_left = False
            last_was_jump = False
            continue

    return x_dcsn_hat, keep, intersections[:n_inter, :]
