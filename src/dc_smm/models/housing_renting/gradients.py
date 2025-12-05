"""
Gradient computation functions for piecewise-linear functions with jumps.

This module provides robust gradient estimation for consumption/policy functions
that may have discontinuities (jumps) due to discrete choice behavior.

Author: Akshay Shanker, 2025
"""

import numpy as np
from numba import njit


# ============================================================
# Helper Functions
# ============================================================

@njit
def _median_njit(arr):
    """Compute median of a 1D array (Numba-compatible)."""
    n = len(arr)
    if n == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    if n % 2 == 1:
        return sorted_arr[n // 2]
    else:
        return 0.5 * (sorted_arr[n // 2 - 1] + sorted_arr[n // 2])


@njit
def _get_neighbor_slope(f, x, idx, n, boundaries, n_seg, seg_idx, eps):
    """Get slope from neighboring segment for single-point segments."""
    # Try previous segment
    if seg_idx > 0 and idx > 0:
        prev_start = boundaries[seg_idx - 1]
        prev_end = boundaries[seg_idx]
        if prev_end - prev_start >= 2:
            dx = x[idx] - x[idx - 1]
            if dx > 1e-15:
                slope = (f[idx] - f[idx - 1]) / dx
                if 0 < slope <= 1.0:
                    return slope
    
    # Try next segment
    if seg_idx < n_seg - 2 and idx < n - 1:
        next_start = boundaries[seg_idx + 1]
        next_end = boundaries[seg_idx + 2]
        if next_end - next_start >= 2:
            dx = x[idx + 1] - x[idx]
            if dx > 1e-15:
                slope = (f[idx + 1] - f[idx]) / dx
                if 0 < slope <= 1.0:
                    return slope
    
    return eps


@njit
def _forward_diff(f, x, i, end, fallback):
    """Forward difference with fallback."""
    if i + 1 < end:
        dx = x[i + 1] - x[i]
        if dx > 1e-15:
            slope = (f[i + 1] - f[i]) / dx
            if 0 < slope <= 1.0:
                return slope
    return fallback


@njit
def _backward_diff(f, x, i, start, fallback):
    """Backward difference with fallback."""
    if i > start:
        dx = x[i] - x[i - 1]
        if dx > 1e-15:
            slope = (f[i] - f[i - 1]) / dx
            if 0 < slope <= 1.0:
                return slope
    return fallback


@njit
def _central_diff_3pt(f, x, i, fallback):
    """3-point central difference."""
    dx = x[i + 1] - x[i - 1]
    if dx > 1e-15:
        slope = (f[i + 1] - f[i - 1]) / dx
        if 0 < slope <= 1.0:
            return slope
    return fallback


@njit
def _richardson_5pt(f, x, i, fallback):
    """5-point Richardson extrapolation for O(h^4) accuracy."""
    h1 = x[i + 1] - x[i - 1]
    h2 = x[i + 2] - x[i - 2]
    
    if h1 > 1e-15 and h2 > 1e-15:
        D1 = (f[i + 1] - f[i - 1]) / h1
        D2 = (f[i + 2] - f[i - 2]) / h2
        slope = (4.0 * D1 - D2) / 3.0
        if 0 < slope <= 1.0:
            return slope
    return fallback


@njit
def _pchip_slope(f, x, i, seg_start, seg_end):
    """
    PCHIP-style slope that preserves monotonicity and respects MPC bounds.
    
    Uses weighted harmonic mean of left/right slopes for interior points.
    This guarantees the gradient stays within the range of neighboring slopes,
    preserving monotonicity and preventing overshoots.
    
    For MPC: result is clamped to (0, 1].
    """
    # Boundary cases: one-sided differences
    if i == seg_start:
        if i + 1 < seg_end:
            h = x[i + 1] - x[i]
            if h > 1e-15:
                slope = (f[i + 1] - f[i]) / h
                return max(1e-10, min(1.0, slope))
        return 1.0  # Default MPC at constraint
    
    if i == seg_end - 1:
        h = x[i] - x[i - 1]
        if h > 1e-15:
            slope = (f[i] - f[i - 1]) / h
            return max(1e-10, min(1.0, slope))
        return 1.0
    
    # Interior point: weighted harmonic mean
    h_left = x[i] - x[i - 1]
    h_right = x[i + 1] - x[i]
    
    if h_left < 1e-15 or h_right < 1e-15:
        return 1.0
    
    s_left = (f[i] - f[i - 1]) / h_left
    s_right = (f[i + 1] - f[i]) / h_right
    
    # Clamp individual slopes to valid MPC range
    s_left = max(1e-10, min(1.0, s_left))
    s_right = max(1e-10, min(1.0, s_right))
    
    # Both slopes positive (which they should be for MPC in (0,1])
    # Use weighted harmonic mean (Fritsch-Carlson formula)
    w1 = 2.0 * h_right + h_left
    w2 = h_right + 2.0 * h_left
    
    # Harmonic mean: avoids overshoots, preserves monotonicity
    slope = (w1 + w2) / (w1 / s_left + w2 / s_right)
    
    return max(1e-10, min(1.0, slope))


# ============================================================
# Main Gradient Functions
# ============================================================

@njit
def piecewise_gradient_pchip(f, x, m_bar, eps=0.9):
    """
    PCHIP-style monotone gradient for piecewise functions.
    
    Uses shape-preserving interpolation principles to compute gradients
    that respect MPC bounds (0, 1] and preserve monotonicity within segments.
    
    Key properties:
    - Gradient at each point is bounded by neighboring finite differences
    - No overshoots or undershoots at extrema
    - Naturally handles the MPC ∈ (0, 1] constraint
    - Smooth within segments, discontinuous only at detected jumps
    
    Parameters
    ----------
    f : 1-D ndarray
        Function values (e.g., consumption policy)
    x : 1-D ndarray
        Grid points (e.g., wealth grid), strictly increasing
    m_bar : float
        Jump threshold for segment detection
    eps : float, optional
        Fallback slope (default: 0.9)
    
    Returns
    -------
    g : 1-D ndarray
        Gradient at each point, guaranteed in (0, 1]
    """
    n = len(x)
    g = np.empty(n)
    
    if n == 0:
        return g
    if n == 1:
        g[0] = eps
        return g
    
    # ---- Step 1: Detect segment boundaries ----
    segment_boundaries = np.zeros(n + 1, dtype=np.int64)
    segment_boundaries[0] = 0
    n_segments = 1
    
    jump_threshold = min(m_bar, 1.0)
    
    for i in range(1, n):
        dx = x[i] - x[i - 1]
        if dx > 1e-15:
            local_slope = (f[i] - f[i - 1]) / dx
            # Jump if slope exceeds threshold, is negative, or very small (near-zero denominator issues)
            if local_slope > jump_threshold or local_slope < 0:
                segment_boundaries[n_segments] = i
                n_segments += 1
    
    segment_boundaries[n_segments] = n
    n_segments += 1
    
    # ---- Step 2: Compute PCHIP slopes within each segment ----
    for seg_idx in range(n_segments - 1):
        start = segment_boundaries[seg_idx]
        end = segment_boundaries[seg_idx + 1]
        seg_len = end - start
        
        if seg_len == 1:
            # Single-point segment: use neighbor slope or fallback
            g[start] = eps
            # Try to get slope from neighboring segment
            if seg_idx > 0 and start > 0:
                dx = x[start] - x[start - 1]
                if dx > 1e-15:
                    slope = (f[start] - f[start - 1]) / dx
                    if 0 < slope <= 1.0:
                        g[start] = slope
            continue
        
        if seg_len == 2:
            # Two-point segment: simple slope for both
            dx = x[end - 1] - x[start]
            if dx > 1e-15:
                slope = (f[end - 1] - f[start]) / dx
                slope = max(1e-10, min(1.0, slope))
                g[start] = slope
                g[start + 1] = slope
            else:
                g[start] = eps
                g[start + 1] = eps
            continue
        
        # Multi-point segment: use PCHIP slopes
        for i in range(start, end):
            g[i] = _pchip_slope(f, x, i, start, end)
    
    return g


@njit
def _smooth_within_segment(g, start, end, smoothing_window=3):
    """Apply local smoothing within a segment using moving average."""
    seg_len = end - start
    if seg_len < smoothing_window + 2:
        return  # Too short to smooth
    
    # Only smooth interior points (preserve boundaries)
    half_win = smoothing_window // 2
    g_smoothed = np.empty(seg_len)
    
    for i in range(seg_len):
        if i < half_win or i >= seg_len - half_win:
            # Keep boundary values unchanged
            g_smoothed[i] = g[start + i]
        else:
            # Moving average
            total = 0.0
            count = 0
            for j in range(-half_win, half_win + 1):
                val = g[start + i + j]
                if 0 < val <= 1.0:
                    total += val
                    count += 1
            if count > 0:
                g_smoothed[i] = total / count
            else:
                g_smoothed[i] = g[start + i]
    
    # Write back
    for i in range(seg_len):
        if 0 < g_smoothed[i] <= 1.0:
            g[start + i] = g_smoothed[i]


@njit
def piecewise_gradient_robust(f, x, m_bar, eps=0.9, min_seg_len=4, guard_distance=2, 
                               smooth_segments=True, smoothing_window=3):
    """
    Robust piecewise gradient computation for functions with jumps.
    
    Improvements over piecewise_gradient_3rd:
    1. Two-pass jump detection (value + curvature based)
    2. Weighted blending near segment boundaries (smooth transition)
    3. Segment-local smoothing via median fallback
    4. Adaptive fallback based on segment statistics
    5. Optional post-hoc smoothing within segments
    
    Parameters
    ----------
    f : 1-D ndarray
        Function values on a strictly-increasing grid
    x : 1-D ndarray
        Grid points, same length as f
    m_bar : float
        Max slope threshold for jump detection (typically 1.0 for MPC)
    eps : float, optional
        Fallback slope if none valid (default: 0.9)
    min_seg_len : int, optional
        Minimum segment length for higher-order schemes (default: 4)
    guard_distance : int, optional
        Distance from segment boundary where weighted blending starts (default: 2)
        Points within guard_distance of boundaries use weighted blend of
        one-sided and central differences
    smooth_segments : bool, optional
        If True, apply local smoothing within each segment (default: True)
    smoothing_window : int, optional
        Window size for local smoothing (default: 3, must be odd)
    
    Returns
    -------
    g : 1-D ndarray
        Slope at each x[i], with 0 < g[i] <= 1
    """
    n = len(x)
    g = np.empty(n)
    
    if n == 0:
        return g
    if n == 1:
        g[0] = eps
        return g
    
    # ============================================================
    # Pass 1: Robust Jump Detection
    # ============================================================
    # Use multiple criteria: slope threshold + sign change + curvature
    
    segment_boundaries = np.zeros(n + 1, dtype=np.int64)
    segment_boundaries[0] = 0
    n_segments = 1
    
    # Compute local slopes
    local_slopes = np.empty(n - 1)
    for i in range(n - 1):
        dx = x[i + 1] - x[i]
        if dx > 1e-15:
            local_slopes[i] = (f[i + 1] - f[i]) / dx
        else:
            local_slopes[i] = 0.0
    
    # Use 1.0 as upper threshold - slopes > 1 indicate jumps (violate MPC bounds)
    jump_threshold = min(m_bar, 1.0)
    
    # Detect jumps using multiple criteria
    for i in range(1, n):
        is_jump = False
        
        # Criterion 1: Slope exceeds threshold or is negative
        if local_slopes[i - 1] > jump_threshold or local_slopes[i - 1] < 0:
            is_jump = True
        
        # Criterion 2: Slope sign change with large magnitude
        if i < n - 1 and not is_jump:
            if local_slopes[i - 1] * local_slopes[i] < 0:  # Sign change
                if abs(local_slopes[i - 1]) > 0.5 * jump_threshold or abs(local_slopes[i]) > 0.5 * jump_threshold:
                    is_jump = True
        
        # Criterion 3: Second derivative spike (curvature discontinuity)
        if i >= 2 and i < n - 1 and not is_jump:
            h1 = x[i] - x[i - 1]
            h2 = x[i + 1] - x[i]
            if h1 > 1e-15 and h2 > 1e-15:
                d2f = 2 * ((f[i + 1] - f[i]) / h2 - (f[i] - f[i - 1]) / h1) / (h1 + h2)
                # Large curvature relative to function scale
                f_scale = max(1e-10, abs(f[i]))
                if abs(d2f) * (h1 + h2) > 2.0 * f_scale:
                    is_jump = True
        
        if is_jump:
            segment_boundaries[n_segments] = i
            n_segments += 1
    
    segment_boundaries[n_segments] = n
    n_segments += 1
    
    # ============================================================
    # Pass 2: Compute Gradients Within Segments
    # ============================================================
    
    for seg_idx in range(n_segments - 1):
        start = segment_boundaries[seg_idx]
        end = segment_boundaries[seg_idx + 1]
        seg_len = end - start
        
        if seg_len == 1:
            # Single point: use neighboring segment slope or fallback
            g[start] = _get_neighbor_slope(f, x, start, n, segment_boundaries, 
                                           n_segments, seg_idx, eps)
            continue
        
        if seg_len == 2:
            # Two points: simple forward/backward difference
            dx = x[end - 1] - x[start]
            if dx > 1e-15:
                slope = (f[end - 1] - f[start]) / dx
                if 0 < slope <= 1.0:
                    g[start] = slope
                    g[start + 1] = slope
                else:
                    g[start] = eps
                    g[start + 1] = eps
            else:
                g[start] = eps
                g[start + 1] = eps
            continue
        
        # Compute segment statistics for adaptive fallback
        seg_slopes = np.empty(seg_len - 1)
        for i in range(seg_len - 1):
            idx = start + i
            dx = x[idx + 1] - x[idx]
            if dx > 1e-15:
                seg_slopes[i] = (f[idx + 1] - f[idx]) / dx
            else:
                seg_slopes[i] = eps
        
        # Segment median slope as robust fallback
        seg_median = _median_njit(seg_slopes)
        if seg_median <= 0 or seg_median > 1.0:
            seg_median = eps
        
        # Compute gradients for each point in segment with weighted blending
        for i in range(start, end):
            rel_pos = i - start  # Position within segment
            dist_to_start = rel_pos
            dist_to_end = end - 1 - i
            min_dist = min(dist_to_start, dist_to_end)
            
            # Compute one-sided gradient (always safe near boundaries)
            if dist_to_start <= dist_to_end:
                # Closer to start: prefer forward difference
                g_onesided = _forward_diff(f, x, i, end, seg_median)
            else:
                # Closer to end: prefer backward difference
                g_onesided = _backward_diff(f, x, i, start, seg_median)
            
            # Compute central/higher-order gradient (better in interior)
            if seg_len >= min_seg_len and dist_to_start >= 2 and dist_to_end >= 2:
                g_central = _richardson_5pt(f, x, i, seg_median)
            elif dist_to_start >= 1 and dist_to_end >= 1:
                g_central = _central_diff_3pt(f, x, i, seg_median)
            else:
                g_central = g_onesided  # No central available
            
            # Weighted blending: weight = 0 at boundary, 1 at guard_distance
            if min_dist >= guard_distance:
                # Fully in interior: use central/higher-order
                g[i] = g_central
            elif guard_distance > 0:
                # Blend between one-sided and central
                weight = float(min_dist) / float(guard_distance)
                g[i] = weight * g_central + (1.0 - weight) * g_onesided
            else:
                # guard_distance = 0: always use one-sided
                g[i] = g_onesided
            
            # Enforce (0, 1] constraint
            if g[i] <= 0 or g[i] > 1.0 or np.isnan(g[i]):
                g[i] = seg_median
    
    # Optional: Apply local smoothing within each segment
    if smooth_segments and smoothing_window >= 3:
        for seg_idx in range(n_segments - 1):
            start = segment_boundaries[seg_idx]
            end = segment_boundaries[seg_idx + 1]
            _smooth_within_segment(g, start, end, smoothing_window)
    
    return g


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
        dx = x[i] - x[i-1]
        if dx > 1e-15:
            local_slope = (f[i] - f[i-1]) / dx
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
                dx = x[start] - x[start-1]
                if dx > 1e-15:
                    g[start] = (f[start] - f[start-1]) / dx
                else:
                    g[start] = eps
            elif seg_idx < n_segments - 2 and end < n:
                # Use slope to next segment's start
                dx = x[end] - x[start]
                if dx > 1e-15:
                    g[start] = (f[end] - f[start]) / dx
                else:
                    g[start] = eps
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
                h2 = x[i+2] - x[i-2]
                
                if h1 > 1e-15 and h2 > 1e-15:
                    D1 = (f[i+1] - f[i-1]) / h1
                    D2 = (f[i+2] - f[i-2]) / h2
                    g[i] = (4.0 * D1 - D2) / 3.0
                else:
                    g[i] = eps
                
                # Clip to [0, 1] range
                if g[i] > 1.0:
                    g[i] = 1.0
                elif g[i] < 0:
                    g[i] = eps
                
            # Try 2nd order at segment boundaries (3-point one-sided)
            elif i == start and seg_len >= 3:
                # 3-point forward difference at segment start: O(h^2)
                h = x[start+1] - x[start]
                h2 = x[start+2] - x[start]
                if h > 1e-15 and h2 > h + 1e-15:
                    a0 = -h2 / (h * (h2 - h))
                    a1 = h2 / (h * h2)
                    a2 = -h / (h2 * (h2 - h))
                    g[i] = a0*f[start] + a1*f[start+1] + a2*f[start+2]
                else:
                    g[i] = eps
                
            elif i == end - 1 and seg_len >= 3:
                # 3-point backward difference at segment end: O(h^2)
                h = x[end-1] - x[end-2]
                h2 = x[end-1] - x[end-3]
                if h > 1e-15 and h2 > h + 1e-15:
                    a0 = h / (h2 * (h2 - h))
                    a1 = -h2 / (h * h2)
                    a2 = h2 / (h * (h2 - h))
                    g[i] = a0*f[end-3] + a1*f[end-2] + a2*f[end-1]
                else:
                    g[i] = eps
                
            # Try 2nd order centered (3-point) if we have neighbors
            elif points_left >= 1 and points_right >= 1:
                # Standard central difference: O(h^2)
                dx = x[i+1] - x[i-1]
                if dx > 1e-15:
                    g[i] = (f[i+1] - f[i-1]) / dx
                else:
                    g[i] = eps
                
            # Fall back to 1st order at edges
            elif i == start:
                # Forward difference at segment start: O(h)
                if seg_len >= 2:
                    dx = x[start+1] - x[start]
                    if dx > 1e-15:
                        g[i] = (f[start+1] - f[start]) / dx
                    else:
                        g[i] = eps
                else:
                    g[i] = eps
                    
            elif i == end - 1:
                # Backward difference at segment end: O(h)
                dx = x[i] - x[i-1]
                if dx > 1e-15:
                    g[i] = (f[i] - f[i-1]) / dx
                else:
                    g[i] = eps
                
            else:
                # Should not reach here, but use central difference as fallback
                dx = x[i+1] - x[i-1]
                if dx > 1e-15:
                    g[i] = (f[i+1] - f[i-1]) / dx
                else:
                    g[i] = eps
            
            # Enforce slope in (0, 1] range
            if g[i] <= 0 or g[i] > 1.0 or np.isnan(g[i]):
                # Try lower-order schemes
                if i < end - 1:
                    dx = x[i+1] - x[i]
                    if dx > 1e-15:
                        g_forward = (f[i+1] - f[i]) / dx
                        if 0 < g_forward <= 1.0:
                            g[i] = g_forward
                            continue
                        
                if i > start:
                    dx = x[i] - x[i-1]
                    if dx > 1e-15:
                        g_backward = (f[i] - f[i-1]) / dx
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
        dx = x[i] - x[i-1]
        if dx > 1e-15:
            local_slope = np.abs(f[i] - f[i-1]) / dx
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
                dx = x[start] - x[start-1]
                if dx > 1e-15:
                    g[start] = (f[start] - f[start-1]) / dx
                else:
                    g[start] = eps
            elif seg_idx < n_segments - 2 and end < n:
                # Use slope to next segment's start
                dx = x[end] - x[start]
                if dx > 1e-15:
                    g[start] = (f[end] - f[start]) / dx
                else:
                    g[start] = eps
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
                    dx = x[start+1] - x[start]
                    if dx > 1e-15:
                        g[i] = (f[start+1] - f[start]) / dx
                    else:
                        g[i] = eps
                else:
                    g[i] = eps
            elif i == end - 1:
                # Backward difference at segment end
                dx = x[i] - x[i-1]
                if dx > 1e-15:
                    g[i] = (f[i] - f[i-1]) / dx
                else:
                    g[i] = eps
            else:
                # Central difference in segment interior
                # Use wider stencil if available for more stability
                if i - start >= 2 and end - i >= 2:
                    # 5-point stencil if possible
                    dx = x[i+1] - x[i]
                    if dx > 1e-15:
                        g[i] = (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*dx)
                    else:
                        g[i] = eps
                else:
                    # Standard central difference
                    dx = x[i+1] - x[i-1]
                    if dx > 1e-15:
                        g[i] = (f[i+1] - f[i-1]) / dx
                    else:
                        g[i] = eps
            
            # Enforce positive slope
            if g[i] <= 0:
                # Try one-sided differences
                if i < end - 1:
                    dx = x[i+1] - x[i]
                    if dx > 1e-15:
                        g_forward = (f[i+1] - f[i]) / dx
                        if g_forward > 0:
                            g[i] = g_forward
                            continue
                if i > start:
                    dx = x[i] - x[i-1]
                    if dx > 1e-15:
                        g_backward = (f[i] - f[i-1]) / dx
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
                dx = x[start] - x[start-1]
                if dx > 1e-15:
                    g[start] = (f[start] - f[start-1]) / dx
                else:
                    g[start] = eps
            elif seg_idx < n_segments - 2 and end < n:
                # Use slope to next segment's start
                dx = x[end] - x[start]
                if dx > 1e-15:
                    g[start] = (f[end] - f[start]) / dx
                else:
                    g[start] = eps
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
                    dx = x[start+1] - x[start]
                    if dx > 1e-15:
                        g[i] = (f[start+1] - f[start]) / dx
                    else:
                        g[i] = eps
                else:
                    g[i] = eps
            elif i == end - 1:
                # Backward difference at segment end
                dx = x[i] - x[i-1]
                if dx > 1e-15:
                    g[i] = (f[i] - f[i-1]) / dx
                else:
                    g[i] = eps
            else:
                # Central difference in segment interior
                dx = x[i+1] - x[i-1]
                if dx > 1e-15:
                    g[i] = (f[i+1] - f[i-1]) / dx
                else:
                    g[i] = eps
            
            # Enforce positive slope
            if g[i] <= 0:
                # Try one-sided differences
                if i < end - 1:
                    dx = x[i+1] - x[i]
                    if dx > 1e-15:
                        g_forward = (f[i+1] - f[i]) / dx
                        if g_forward > 0:
                            g[i] = g_forward
                            continue
                if i > start:
                    dx = x[i] - x[i-1]
                    if dx > 1e-15:
                        g_backward = (f[i] - f[i-1]) / dx
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


# ============================================================
# Dispatcher Function - Select Gradient Method by Name
# ============================================================

def get_gradient_function(method="robust"):
    """
    Return the appropriate gradient function based on method name.
    
    Parameters
    ----------
    method : str
        Gradient method to use. Options:
        - "basic": Simple piecewise gradient with jump detection
        - "3rd_order": Higher-order finite differences (O(h^4) accuracy)
        - "robust": Multi-criteria jump detection + weighted blending
        - "pchip": PCHIP-style monotone gradients (recommended for MPC)
    
    Returns
    -------
    callable
        The gradient function matching the requested method
    
    Examples
    --------
    >>> grad_func = get_gradient_function("pchip")
    >>> mpc = grad_func(policy, grid, m_bar=1.0)
    """
    method_lower = method.lower().strip()
    
    if method_lower in ("basic", "simple"):
        return piecewise_gradient
    elif method_lower in ("3rd_order", "3rd", "third_order"):
        return piecewise_gradient_3rd
    elif method_lower in ("robust", "weighted"):
        return piecewise_gradient_robust
    elif method_lower in ("pchip", "monotone", "shape_preserving"):
        return piecewise_gradient_pchip
    else:
        # Default to pchip if unknown (best for MPC)
        import warnings
        warnings.warn(f"Unknown gradient method '{method}', defaulting to 'pchip'")
        return piecewise_gradient_pchip


def compute_gradient(f, x, m_bar, method="pchip", eps=0.9, guard_distance=2,
                     smooth_segments=True, smoothing_window=3):
    """
    Compute piecewise gradient using the specified method.
    
    This is a convenience function that dispatches to the appropriate
    gradient function based on the method name.
    
    Parameters
    ----------
    f : 1-D ndarray
        Function values on a strictly-increasing grid
    x : 1-D ndarray
        Grid points, same length as f
    m_bar : float
        Max slope threshold for jump detection
    method : str, optional
        Gradient method: "basic", "3rd_order", "robust", or "pchip" (default)
    eps : float, optional
        Fallback slope if none valid (default: 0.9)
    guard_distance : int, optional
        For "robust" method: distance for weighted blending (default: 2)
    smooth_segments : bool, optional
        For "robust" method: apply local smoothing within segments (default: True)
    smoothing_window : int, optional
        For "robust" method: window size for smoothing (default: 3)
    
    Returns
    -------
    g : 1-D ndarray
        Gradient at each point, in (0, 1] for MPC
    """
    method_lower = method.lower()
    
    if method_lower in ("pchip", "monotone", "shape_preserving"):
        return piecewise_gradient_pchip(f, x, m_bar, eps=eps)
    elif method_lower in ("robust", "weighted"):
        return piecewise_gradient_robust(f, x, m_bar, eps=eps, guard_distance=guard_distance,
                                         smooth_segments=smooth_segments, 
                                         smoothing_window=smoothing_window)
    elif method_lower in ("3rd_order", "3rd", "third_order"):
        return piecewise_gradient_3rd(f, x, m_bar, eps=eps)
    else:
        return piecewise_gradient(f, x, m_bar, eps=eps)
