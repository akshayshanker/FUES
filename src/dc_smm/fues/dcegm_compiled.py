"""
DCEGM with numba-compiled interpolation helpers.

This is an optimized version that compiles the interpolation and filtering
loops. However, testing showed the original HARK-based version is actually
faster due to HARK's already-optimized implementation.

This file is kept for reference and potential future optimization work.
"""

import numpy as np
from numba import njit, prange

from HARK.dcegm import calc_nondecreasing_segments, upper_envelope


@njit(cache=True)
def _interp_njit(xp, fp, x):
    """Simple linear interpolation - numba compiled."""
    n = len(x)
    result = np.empty(n)
    for i in range(n):
        xi = x[i]
        # Binary search for interval
        if xi <= xp[0]:
            result[i] = fp[0]
        elif xi >= xp[-1]:
            result[i] = fp[-1]
        else:
            # Find interval
            lo, hi = 0, len(xp) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if xp[mid] <= xi:
                    lo = mid
                else:
                    hi = mid
            # Linear interpolation
            t = (xi - xp[lo]) / (xp[hi] - xp[lo])
            result[i] = fp[lo] + t * (fp[hi] - fp[lo])
    return result


@njit(cache=True)
def _searchsorted_njit(a, v):
    """Binary search - numba compiled."""
    n = len(v)
    result = np.empty(n, dtype=np.int64)
    m = len(a)
    for i in range(n):
        vi = v[i]
        lo, hi = 0, m
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < vi:
                lo = mid + 1
            else:
                hi = mid
        result[i] = lo
    return result


@njit(cache=True)
def _fill_envelopes(m_upper, inds_upper, n_segments,
                    seg_starts, seg_ends,
                    c_all, a_all, v_all, d_all, m_all):
    """
    Fill c1_env, a1_env, v1_env, d1_env in one compiled pass.
    
    seg_starts[k], seg_ends[k] give the slice indices for segment k
    in the flat arrays c_all, a_all, etc.
    """
    n = len(m_upper)
    c1_env = np.empty(n)
    a1_env = np.empty(n)
    v1_env = np.empty(n)
    d1_env = np.empty(n)
    
    # Initialize with nan
    for i in range(n):
        c1_env[i] = np.nan
        a1_env[i] = np.nan
        v1_env[i] = np.nan
        d1_env[i] = np.nan
    
    # Process each segment
    for k in range(n_segments):
        s0 = seg_starts[k]
        s1 = seg_ends[k]
        
        # Extract segment arrays
        m_seg = m_all[s0:s1]
        c_seg = c_all[s0:s1]
        a_seg = a_all[s0:s1]
        v_seg = v_all[s0:s1]
        d_seg = d_all[s0:s1]
        
        # Find points belonging to this segment
        # Count first
        count = 0
        for i in range(n):
            if inds_upper[i] == k:
                count += 1
        
        if count == 0:
            continue
            
        # Collect m_upper values for this segment
        m_k = np.empty(count)
        idx_k = np.empty(count, dtype=np.int64)
        j = 0
        for i in range(n):
            if inds_upper[i] == k:
                m_k[j] = m_upper[i]
                idx_k[j] = i
                j += 1
        
        # Interpolate all 4 arrays at once
        a_interp = _interp_njit(m_seg, a_seg, m_k)
        v_interp = _interp_njit(m_seg, v_seg, m_k)
        d_interp = _interp_njit(m_seg, d_seg, m_k)
        
        # For c, use searchsorted (nearest-neighbor style from original)
        c_idx = _searchsorted_njit(m_seg, m_k)
        # Clamp indices
        for j in range(count):
            if c_idx[j] >= len(c_seg):
                c_idx[j] = len(c_seg) - 1
        
        # Write back
        for j in range(count):
            i = idx_k[j]
            c1_env[i] = c_seg[c_idx[j]]
            a1_env[i] = a_interp[j]
            v1_env[i] = v_interp[j]
            d1_env[i] = d_interp[j]
    
    return c1_env, a1_env, v1_env, d1_env


@njit(cache=True)
def _filter_by_membership(a1_env, a_prime, m_upper, c1_env, v1_env, d1_env):
    """
    Filter to keep only points where a1_env value exists in a_prime.
    Replaces np.in1d which isn't supported in numba.
    """
    n = len(a1_env)
    m = len(a_prime)
    
    # Create set of a_prime values for O(1) lookup
    # Since we can't use sets in numba, use sorted array + binary search
    a_prime_sorted = np.sort(a_prime)
    
    # First pass: count valid
    count = 0
    for i in range(n):
        val = a1_env[i]
        if np.isnan(val):
            continue
        # Binary search in a_prime_sorted
        lo, hi = 0, m
        found = False
        while lo < hi:
            mid = (lo + hi) // 2
            if abs(a_prime_sorted[mid] - val) < 1e-12:
                found = True
                break
            elif a_prime_sorted[mid] < val:
                lo = mid + 1
            else:
                hi = mid
        if found:
            count += 1
    
    # Allocate output
    a1_env2 = np.empty(count)
    m_upper2 = np.empty(count)
    c_env2 = np.empty(count)
    v_env2 = np.empty(count)
    d_env2 = np.empty(count)
    
    # Second pass: fill
    j = 0
    for i in range(n):
        val = a1_env[i]
        if np.isnan(val):
            continue
        lo, hi = 0, m
        found = False
        while lo < hi:
            mid = (lo + hi) // 2
            if abs(a_prime_sorted[mid] - val) < 1e-12:
                found = True
                break
            elif a_prime_sorted[mid] < val:
                lo = mid + 1
            else:
                hi = mid
        if found:
            a1_env2[j] = a1_env[i]
            m_upper2[j] = m_upper[i]
            c_env2[j] = c1_env[i]
            v_env2[j] = v1_env[i]
            d_env2[j] = d1_env[i]
            j += 1
    
    return a1_env2, m_upper2, c_env2, v_env2, d_env2


def dcegm_compiled(c, dela, vf, a_prime, x):
    """
    DCEGM upper envelope algorithm with numba-compiled helpers.
    
    Uses HARK for segment detection and upper envelope computation,
    but numba-compiled helpers for the interpolation and filtering steps.
    
    Note: Testing showed the original HARK-based version (dcegm.py) is 
    actually faster. This version is kept for reference.
    """
    # Step 1: Detect non-decreasing segments (HARK - can't compile this)
    start, end = calc_nondecreasing_segments(x, vf)
    n_segments = len(start)
    
    if n_segments == 0:
        # No valid segments - return empty arrays
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty
    
    # Step 2: Build segment data as flat arrays (instead of lists)
    # Pre-compute total size
    total_pts = sum(end[j] - start[j] + 1 for j in range(n_segments))
    
    # Allocate flat arrays
    c_all = np.empty(total_pts)
    a_all = np.empty(total_pts)
    m_all = np.empty(total_pts)
    v_all = np.empty(total_pts)
    d_all = np.empty(total_pts)
    seg_starts = np.empty(n_segments, dtype=np.int64)
    seg_ends = np.empty(n_segments, dtype=np.int64)
    
    # Also build segments list for HARK (still needed for upper_envelope)
    segments = []
    pos = 0
    for j in range(n_segments):
        s, e = start[j], end[j] + 1
        seg_len = e - start[j]
        seg_starts[j] = pos
        seg_ends[j] = pos + seg_len
        
        c_all[pos:pos+seg_len] = c[s:e]
        a_all[pos:pos+seg_len] = a_prime[s:e]
        m_all[pos:pos+seg_len] = x[s:e]
        v_all[pos:pos+seg_len] = vf[s:e]
        d_all[pos:pos+seg_len] = dela[s:e]
        
        segments.append([x[s:e], vf[s:e]])
        pos += seg_len
    
    # Step 3: Compute upper envelope (HARK - can't compile this)
    m_upper, v_upper, inds_upper = upper_envelope(segments, calc_crossings=False)
    
    # Convert inds_upper to int64 for numba
    inds_upper = inds_upper.astype(np.int64)
    
    # Step 4: Fill envelopes using compiled helper (FAST)
    c1_env, a1_env, v1_env, d1_env = _fill_envelopes(
        m_upper, inds_upper, n_segments,
        seg_starts, seg_ends,
        c_all, a_all, v_all, d_all, m_all
    )
    
    # Step 5: Filter to original grid points using compiled helper (FAST)
    a1_env2, m_upper2, c_env2, v_env2, d_env2 = _filter_by_membership(
        a1_env, a_prime, m_upper, c1_env, v1_env, d1_env
    )
    
    return a1_env2, m_upper2, c_env2, v_env2, d_env2













