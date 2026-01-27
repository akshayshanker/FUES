import numpy as np
from scipy.interpolate import interp1d
from numba import njit
from numba.typed import Dict    
from typing import Callable, Literal   # NEW  – remove if unused        # NEW
import time
from functools import lru_cache
import math
import warnings

# Conditional CUDA import - only load if available
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    cuda = None
    CUDA_AVAILABLE = False

# Import gradient functions from dedicated module (backward compatibility re-export)
from dc_smm.models.housing_renting.gradients import (
    piecewise_gradient,
    piecewise_gradient_3rd,
    piecewise_gradient_with_segments,
    piecewise_gradient_robust,
    compute_gradient,
    get_gradient_function,
)


def _ensure_1d_contig(x, dtype=None):
    """Ensure array is 1D and C-contiguous."""
    a = np.asarray(x)
    if a.ndim > 1:
        a = np.squeeze(a)
    a = np.ravel(a)
    if dtype is not None:
        a = a.astype(dtype, copy=False)
    return np.ascontiguousarray(a)

def _grid_spacing_eps(e):
    """Compute adaptive tolerance based on grid spacing."""
    e = np.ravel(np.asarray(e))
    es = np.sort(e)
    d = np.diff(es)
    pos = d[d > 0]
    h = np.median(pos) if pos.size else max(1.0, np.max(np.abs(es)) if es.size else 1.0)
    scale = max(1.0, np.max(np.abs(es)) if es.size else 1.0)
    # Conservative: scale by machine eps and small fraction of typical spacing
    return max(10*np.finfo(es.dtype if es.dtype.kind in "fc" else np.float64).eps * scale, 0.03*h)

def uniqueEG_numeric(
    e, v, c, a,
    *,
    eps_e=None, tol_v=1e-12, tol_c=1e-12, tol_a=1e-12,
    tie_policy="nearest_prev",
    warn_on_policy_conflict=True,
    logger=None
):
    """
    Robust deduplication of EGM grid points.
    
    Only removes points when both grid AND values are near-duplicates.
    Warns about policy conflicts at equal values.
    When resolving ties, uses policy closest to previous cluster for continuity.
    """
    # Normalize inputs
    e = _ensure_1d_contig(e, dtype=np.float64)
    v = _ensure_1d_contig(v, dtype=np.float64)
    c = _ensure_1d_contig(c, dtype=np.float64)
    a = _ensure_1d_contig(a, dtype=np.float64)
    n = e.size
    
    if not (v.size == c.size == a.size == n):
        raise ValueError(f"Shape mismatch e:{e.shape} v:{v.shape} c:{c.shape} a:{a.shape}")
    
    # Stable sort by e
    idx = np.argsort(e, kind="mergesort")
    e, v, c, a = e[idx], v[idx], c[idx], a[idx]
    
    if eps_e is None:
        eps_e = _grid_spacing_eps(e)
    
    info = dict(
        n_rows_in=n, n_rows_out=None, eps_e_used=float(eps_e),
        tol_v=float(tol_v), tol_c=float(tol_c), tol_a=float(tol_a),
        n_clusters=0, n_ties=0, n_policy_conflicts=0
    )
    
    keep_mask = np.zeros(n, dtype=bool)
    
    # Track previous cluster's policy for continuity
    prev_c = None
    prev_a = None
    
    # Process clusters
    i = 0
    while i < n:
        j = i
        # Build cluster: e[k]-e[i] <= eps_e
        while j + 1 < n and e[j + 1] - e[i] <= eps_e:
            j += 1
        info["n_clusters"] += 1
        
        if i == j:
            # Single point cluster
            keep_mask[i] = True
            prev_c = c[i]
            prev_a = a[i]
            i = j + 1
            continue
        
        # Multi-point cluster
        v_block = v[i:j+1]
        c_block = c[i:j+1]
        a_block = a[i:j+1]
        
        vmax = np.max(v_block)
        winners = np.where(np.abs(v_block - vmax) <= tol_v)[0]
        
        if winners.size <= 1:
            # No value tie; keep ALL rows
            keep_mask[i:j+1] = True
            # Update previous policy to last kept point
            prev_c = c[j]
            prev_a = a[j]
            i = j + 1
            continue
        
        info["n_ties"] += 1
        # Value tie; check policy spread among winners
        w_c = c_block[winners]
        w_a = a_block[winners]
        spread_c = np.max(w_c) - np.min(w_c)
        spread_a = np.max(w_a) - np.min(w_a)
        
        if (spread_c <= tol_c) and (spread_a <= tol_a):
            # Exact duplicate policies; keep first winner only
            keep_mask[i:j+1] = True
            dup_idxs = winners[1:] + i
            keep_mask[dup_idxs] = False
            # Update previous policy
            kept_idx = winners[0] + i
            prev_c = c[kept_idx]
            prev_a = a[kept_idx]
            i = j + 1
            continue
        
        # Policy conflict at equal value
        info["n_policy_conflicts"] += 1
        if warn_on_policy_conflict:
            # Build warning message
            base_msg = (f"uniqueEG_numeric: value tie with conflicting policies near e≈{float(np.mean(e[i:j+1])):.6g} "
                       f"(eps_e={eps_e:.3g}). winners={winners.size}, v≈{float(vmax):.6g}±{float(tol_v):.1e}; "
                       f"c∈[{float(np.min(w_c)):.6g},{float(np.max(w_c)):.6g}], "
                       f"a'∈[{float(np.min(w_a)):.6g},{float(np.max(w_a)):.6g}]. ")
            
            # Add policy-specific info
            if tie_policy == "nearest_prev" and prev_c is not None:
                msg = base_msg + f"tie_policy={tie_policy} (prev: c={prev_c:.4g}, a'={prev_a:.4g})."
            else:
                msg = base_msg + f"tie_policy={tie_policy}."
            if logger is not None:
                try: 
                    logger.warning(msg)
                except Exception: 
                    pass
            warnings.warn(msg, category=UserWarning)
        
        # Choose canonical winner
        if tie_policy == "min_a":
            k_rel = winners[np.argmin(w_a)]
        elif tie_policy == "max_a":
            k_rel = winners[np.argmax(w_a)]
        elif tie_policy == "first":
            k_rel = winners[0]
        elif tie_policy == "medoid":
            cbar = float(np.mean(w_c))
            abar = float(np.mean(w_a))
            d2 = (w_c - cbar)**2 + (w_a - abar)**2
            k_rel = winners[np.argmin(d2)]
        else:  # "nearest_prev" default - choose closest to previous cluster
            if prev_c is not None and prev_a is not None:
                # Choose winner closest to previous cluster's policy
                d2 = (w_c - prev_c)**2 + (w_a - prev_a)**2
                k_rel = winners[np.argmin(d2)]
            else:
                # No previous cluster, fall back to medoid
                cbar = float(np.mean(w_c))
                abar = float(np.mean(w_a))
                d2 = (w_c - cbar)**2 + (w_a - abar)**2
                k_rel = winners[np.argmin(d2)]
        
        # Keep all non-winners; among winners keep only chosen one
        keep_mask[i:j+1] = True
        drop_rel = winners[winners != k_rel]
        keep_mask[i + drop_rel] = False
        
        # Update previous policy with the chosen winner
        kept_idx = k_rel + i
        prev_c = c[kept_idx]
        prev_a = a[kept_idx]
        
        i = j + 1
    
    # Finalize
    e_out = e[keep_mask]
    v_out = v[keep_mask]
    c_out = c[keep_mask]
    a_out = a[keep_mask]
    info["n_rows_out"] = int(e_out.size)
    
    # Verify invariants
    assert e_out.ndim == v_out.ndim == c_out.ndim == a_out.ndim == 1
    assert e_out.size == v_out.size == c_out.size == a_out.size
    # np.testing.assert_allclose(e_out, a_out + c_out, rtol=0, atol=max(tol_c, tol_a))
    
    return e_out, v_out, c_out, a_out, info


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

# Note: piecewise_gradient, piecewise_gradient_3rd, piecewise_gradient_with_segments, 
# and piecewise_gradient_robust are now imported from gradients.py at the top of this file
# for backward compatibility

@njit(cache=True)
def _uniqueEG_core(grid: np.ndarray, values: np.ndarray, tol: float) -> np.ndarray:
    """Numba-compiled core for uniqueEG.
    
    Uses argsort on grid, then manually finds groups and keeps highest value.
    """
    n = len(grid)
    if n == 0:
        return np.zeros(0, dtype=np.bool_)
    
    # Sort by grid ascending
    order = np.argsort(grid)
    
    # For each group of close grid points, keep the one with highest value
    mask = np.zeros(n, dtype=np.bool_)
    
    # Track best index in current group
    group_start = 0
    best_idx = order[0]
    best_val = values[order[0]]
    
    for i in range(1, n):
        idx = order[i]
        if grid[idx] - grid[order[group_start]] > tol:
            # New group - mark best from previous group
            mask[best_idx] = True
            group_start = i
            best_idx = idx
            best_val = values[idx]
        else:
            # Same group - check if this has higher value
            if values[idx] > best_val:
                best_idx = idx
                best_val = values[idx]
    
    # Don't forget the last group
    mask[best_idx] = True
    
    return mask


def uniqueEG(grid: np.ndarray, values: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Return a Boolean mask that keeps the *highest-value* entry for each
    duplicate or near-duplicate grid point.

    Uses compiled core for performance.
    
    Parameters
    ----------
    grid : ndarray
        Grid points to check for duplicates
    values : ndarray
        Values associated with each grid point
    tol : float, optional
        Tolerance for considering points as duplicates (default: 1e-10)
    """
    return _uniqueEG_core(grid, values, tol)


@njit
def _calculate_gradient_1d(data, x):
    """
    Calculate gradient of 1D data array with respect to x.
    
    Parameters
    ----------
    data : ndarray
        1D data values
    x : ndarray  
        1D x-coordinates
        
    Returns
    -------
    ndarray
        Gradient array (same length as input, with boundary handling)
    """
    n = len(data)
    grad = np.zeros(n)
    
    for i in range(1, n):
        dx = x[i] - x[i - 1]
        if abs(dx) > 1e-15:
            grad[i] = (data[i] - data[i - 1]) / dx
    
    # Assume continuous gradient at start
    if n > 1:
        grad[0] = grad[1]
    
    return grad


@njit
def correct_jumps_1d_with_mask(policy, value, x, gradient_jump_threshold, min_segment_length=3):
    """
    Correct short segments and isolated points by extrapolating from good segments.
    
    Segments are defined by DECREASING policy (gradient <= 0).
    Within a segment, policy should be increasing. Short segments are
    replaced by extrapolation from the last good (long enough) segment.
    
    Parameters
    ----------
    policy : ndarray
        1D policy array (consumption)
    value : ndarray  
        1D value array (Q_dcsn)
    x : ndarray
        1D grid (wealth)
    gradient_jump_threshold : float
        Not used for segment detection (kept for API compatibility)
    min_segment_length : int
        Minimum segment length. Shorter segments are extrapolated.
        
    Returns
    -------
    tuple
        (corrected_policy, corrected_value)
    """
    n = len(policy)
    if n < 4:
        return policy.copy(), value.copy()
    
    corrected_policy = policy.copy()
    corrected_value = value.copy()
    
    # Compute gradients
    gradients = _calculate_gradient_1d(policy, x)
    
    # --- Pass 1: Find segment boundaries (where policy decreases) ---
    segment_starts = np.zeros(n, dtype=np.int64)
    segment_starts[0] = 0
    n_segments = 1
    
    for i in range(1, n - 1):
        if gradients[i] <= 0:  # Policy decreasing = new segment
            segment_starts[n_segments] = i
            n_segments += 1
    segment_starts[n_segments] = n
    n_segments += 1
    
    # --- Pass 2: Extrapolate short segments from last good segment ---
    for seg_idx in range(1, n_segments - 2):
        start = segment_starts[seg_idx]
        end = segment_starts[seg_idx + 1]
        seg_len = end - start
        
        if seg_len < min_segment_length:
            # Find last good segment (searching backwards)
            good_end = -1
            for prev_idx in range(seg_idx - 1, -1, -1):
                prev_start = segment_starts[prev_idx]
                prev_end = segment_starts[prev_idx + 1]
                if prev_end - prev_start >= min_segment_length:
                    good_end = prev_end
                    break
            
            if good_end >= 4:
                # Richardson extrapolation from last 4 points
                h1 = x[good_end - 1] - x[good_end - 2]
                h2 = x[good_end - 1] - x[good_end - 4]
                
                if h1 > 1e-15 and h2 > 1e-15:
                    D1_pol = (corrected_policy[good_end - 1] - corrected_policy[good_end - 2]) / h1
                    D1_val = (corrected_value[good_end - 1] - corrected_value[good_end - 2]) / h1
                    D2_pol = (corrected_policy[good_end - 1] - corrected_policy[good_end - 4]) / h2
                    D2_val = (corrected_value[good_end - 1] - corrected_value[good_end - 4]) / h2
                    
                    slope_policy = (4.0 * D1_pol - D2_pol) / 3.0
                    slope_value = (4.0 * D1_val - D2_val) / 3.0
                    
                    # Clamp policy slope to (0, 1]
                    slope_policy = max(0.1, min(1.0, slope_policy))
                    
                    # Extrapolate all points in short segment
                    x_anchor = x[good_end - 1]
                    pol_anchor = corrected_policy[good_end - 1]
                    val_anchor = corrected_value[good_end - 1]
                    
                    for i in range(start, end):
                        corrected_policy[i] = pol_anchor + slope_policy * (x[i] - x_anchor)
                        corrected_value[i] = val_anchor + slope_value * (x[i] - x_anchor)
    
    # --- Pass 3: Fix NaN values ---
    for i in range(2, n - 2):
        if np.isnan(corrected_policy[i]) and i >= 3:
            dx = x[i - 2] - x[i - 3]
            if abs(dx) > 1e-15:
                slope_policy = (corrected_policy[i - 2] - corrected_policy[i - 3]) / dx
                slope_value = (corrected_value[i - 2] - corrected_value[i - 3]) / dx
                corrected_policy[i] = corrected_policy[i - 2] + slope_policy * (x[i] - x[i - 2])
                corrected_value[i] = corrected_value[i - 2] + slope_value * (x[i] - x[i - 2])
    
    return corrected_policy, corrected_value


@njit
def correct_jumps_policy_and_value(policy_slice, value_slice, w_grid, m_bar, min_segment_length=3):
    """
    Correct isolated jumps in policy and value function by extrapolation.
    
    Jump detection is based on POLICY gradient only. When a jump is detected
    in the policy, BOTH policy and value are corrected at those same points.
    
    Parameters
    ----------
    policy_slice : ndarray
        1D policy array (e.g., consumption policy for fixed (h, y))
    value_slice : ndarray
        1D value array (e.g., Q_dcsn for fixed (h, y))
    w_grid : ndarray
        Wealth grid
    m_bar : float
        Jump threshold (policy gradient > m_bar indicates a jump)
    min_segment_length : int
        Minimum segment length (shorter segments are corrected)
        
    Returns
    -------
    tuple
        (corrected_policy, corrected_value) - Both arrays corrected at same points
    """
    return correct_jumps_1d_with_mask(policy_slice, value_slice, w_grid, m_bar, min_segment_length)


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
def _add_tax_constraint_segments_from_base_grid(
    e_old, vf_old, c_old, a_old,
    vf_next, beta, u_func, h_nxt,
    a_base, c_base,
    vf_base,               # 1D array of base (pre-preprocess) values aligned with a_base
    tax_node_indices,      # 1D array of indices into a_base/c_base/vf_next/vf_base
    tax_node_c_lb_pct,     # 1D array of lb percentages
    tax_node_c_ub_pct,     # 1D array of ub percentages
    tax_node_is_lhs,       # 1D array: 1 if LHS node (enter bracket), 0 if RHS node (exit bracket)
    n_points_per_node=10,
    lambda_next=None,     # Optional: scaled continuation marginal utility on base grid
    uc_func=None,         # Optional: marginal utility function uc(c, h)
    override_KT_conditions=False,
):
    """
    Add constraint segments at specified tax bracket nodes, using indices into
    the *base* continuation asset grid (a_base).

    Why this exists
    --------------
    `tax_node_indices` are computed as indices into the model's continuation
    asset grid (e.g., `a_nxt_grid`). However, EGM preprocessing may prepend
    borrowing/jump constraint points and may optionally sort the resulting
    arrays. In that case, indices into the base grid no longer correspond to
    positions in the preprocessed arrays.

    This helper avoids index misalignment by:
    - Looking up (a_star, c_star) at node indices in (a_base, c_base)
    - Appending the generated constraint points to the already-preprocessed
      arrays (e_old, vf_old, c_old, a_old)

    KT / FOC filtering
    ------------------
    If `lambda_next` and `uc_func` are provided, we apply the same Kuhn–Tucker
    style filter used for pb != 1 jump segments:
        uc(c) >= lambda_next[k]
    where `lambda_next` is already fully scaled (e.g., beta*delta*Rfree*lambda).
    """
    n_nodes = len(tax_node_indices)
    if n_nodes == 0:
        return e_old, vf_old, c_old, a_old

    n_old = len(e_old)
    max_add = n_nodes * n_points_per_node

    # Pre-allocate
    e_new = np.empty(n_old + max_add, dtype=e_old.dtype)
    vf_new = np.empty(n_old + max_add, dtype=vf_old.dtype)
    c_new = np.empty(n_old + max_add, dtype=c_old.dtype)
    a_new = np.empty(n_old + max_add, dtype=a_old.dtype)

    p = 0
    n_base = len(a_base)

    for node_idx in range(n_nodes):
        k = tax_node_indices[node_idx]
        c_lb_pct = tax_node_c_lb_pct[node_idx]
        c_ub_pct = tax_node_c_ub_pct[node_idx]
        is_lhs = tax_node_is_lhs[node_idx]

        if k < 0 or k >= n_base:
            continue

        # Monotonicity guard for tax nodes:
        # If this is an LHS node (entering the *next* bracket/segment),
        # only add constraints if the base value function is locally increasing
        # to the right: vf_base[k+1] - vf_base[k] > 0.
        if is_lhs == 1:
            if k >= (len(vf_base) - 1):
                continue
            if (vf_base[k + 1] - vf_base[k]) <= 0: #k+1 is the first piint on the new segm. 
                continue

        a_star = a_base[k]
        c_star = c_base[k]

        if c_star < 1e-8:
            continue

        lb_c = max(1e-8, c_star * (1.0 - c_lb_pct))
        ub_c = c_star * (1.0 + c_ub_pct)

        for i in range(n_points_per_node):
            c_pt = lb_c + (ub_c - lb_c) * i / max(1, n_points_per_node - 1)

            # Optional KT/FOC filtering (side-specific):
            # - LHS node (entering next bracket): keep if uc(c) >= lambda_next[k]
            # - RHS node (exiting current bracket): keep if uc(c) <= lambda_next[k]
            if lambda_next is not None and uc_func is not None and (not override_KT_conditions):
                mu_c = uc_func(c_pt, h_nxt)
                if is_lhs == 1:
                    if mu_c < lambda_next[k]:
                        continue
                else:
                    if mu_c > lambda_next[k]:
                        continue

            m_pt = a_star + c_pt
            vf_pt = u_func(c_pt, h_nxt) + beta * vf_next[k]

            e_new[p] = m_pt
            vf_new[p] = vf_pt
            c_new[p] = c_pt
            a_new[p] = a_star
            p += 1

    # Append original arrays
    for i in range(n_old):
        e_new[p + i] = e_old[i]
        vf_new[p + i] = vf_old[i]
        c_new[p + i] = c_old[i]
        a_new[p + i] = a_old[i]

    actual_size = p + n_old
    return e_new[:actual_size], vf_new[:actual_size], c_new[:actual_size], a_new[:actual_size]


@njit
def _egm_preprocess_core(e_old, vf_old, c_old, a_old,
                         vf_next,              # 1-D, same length as old grids
                         beta, u_func,         # u_func must be @njit-able
                         m_bar,                # jump threshold (not currently used)
                         n_con,                # # constraint nodes
                         n_con_nxt,            # # nodes per jump (fallback if mean_e_diff not available)
                         c_max, h_nxt,         # upper end of [c*, c_max] and housing
                         add_jump_constraints, # whether to add jump constraints
                         lambda_next=None,     # Marginal utility values (already scaled by beta*delta*Rfree)
                         uc_func=None,         # Marginal utility function
                         override_KT_conditions=False,  # Override FOC filtering when True
                         jump_extend="after",  # Which segments to extend at jumps
                         grad_c_threshold=1.04,  # Threshold for grad_c jump detection
                         c_star_lb_pct=0.10,  # Lower bound = c_star * (1 - lb_pct), e.g., 0.10 = 10%
                         c_star_ub_pct=0.10,  # Upper bound = c_star * (1 + ub_pct), e.g., 0.10 = 10%
                         use_mean_spacing=True,  # Use mean e_diff for point spacing at jumps
                         sort_output=True):    # Sort output by endogenous grid (True for FUES, False for DCEGM)
    """
    Returns new (e,vf,c,a) with:
        • n_con borrowing-constraint points, plus
        • (optionally) constraint points at jumps that satisfy FOC

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
        Target number of nodes to add per jump (actual may be less due to FOC filtering)
    c_max : float
        Upper bound on consumption (not currently used)
    h_nxt : float
        Next period housing
    add_jump_constraints : bool
        Whether to add jump constraint points (should be False when delta_pb == 1)
    lambda_next : ndarray, optional
        Marginal utility values (already fully scaled: beta*delta*lambda*Rfree)
    uc_func : callable, optional
        Marginal utility function uc(c, h)
    override_KT_conditions : bool, optional
        If True, disables FOC filtering at jumps (keeps all points).
        Default is False (FOC filtering enabled).
    jump_extend : str, optional
        Which segments to extend at jumps. Options:
        - "before": only extend segment before jump (using point k)
        - "after": only extend segment after jump (using point k+1)
        - "both": extend both segments (default)
    grad_c_threshold : float, optional
        Threshold for consumption gradient jump detection.
        Jumps detected when |dc/da| > grad_c_threshold. Default: 1.04
    c_star_lb_pct : float, optional
        Percentage of c_star to subtract for lower bound of constraint segment.
        Lower bound = max(1e-8, c_star * (1 - c_star_lb_pct)). Default: 0.10 (10%)
    c_star_ub_pct : float, optional
        Percentage of c_star to add for upper bound of constraint segment.
        Upper bound = c_star * (1 + c_star_ub_pct). Default: 0.10 (10%)
    use_mean_spacing : bool, optional
        If True (default), determines the number of constraint points at each jump
        based on mean spacing of input grid (mean_e_diff). Points are spaced at
        approximately mean_e_diff intervals. If False, uses n_con_nxt as fixed count.

    Notes
    -----
    When lambda_next and uc_func are provided, FOC checks are performed
    unless override_KT_conditions is True.
    The comparison is direct since lambda_next is already fully scaled.
    Points are added based on corrected constraint conditions
    """

    # ---- 0. Basic sizes and jump detection ----
    n_old = e_old.size
    
    # Initialize jump tracking variables
    n_jump = 0
    n_add = n_con  # Always add borrowing constraint nodes
    
    # Compute mean spacing of input grid (used for spacing constraint points)
    e_diff_all = e_old[1:] - e_old[:-1]
    
    
    # Pre-allocate arrays for storing n_points per jump segment
    # Will be populated if use_mean_spacing is True
    n_points_after = None
    n_points_before = None
    allow_after = None
    
    # Only detect and process jumps if flag is True and n_con_nxt > 0
    if add_jump_constraints and n_con_nxt > 0:
        # Optimized jump detection - compute differences once
        e_diff = e_diff_all  # Reuse already computed
        vf_diff = vf_next[1:] - vf_next[:-1]
        c_diff = c_old[1:] - c_old[:-1]
        a_diff = a_old[1:] - a_old[:-1]
        q_diff = vf_old[1:] - vf_old[:-1]  # local monotonicity of value on (e_old) grid

        mean_e_diff = np.mean(np.abs(a_diff))

        # Optimized gradient computation - avoid redundant operations
        # Create mask once and reuse
        valid_diff_mask = np.abs(a_diff) > 1e-15

        # Pre-allocate gradient arrays
        grad_e = np.zeros_like(a_diff)
        grad_c = np.zeros_like(a_diff)

        # Compute gradients only where valid
        if np.any(valid_diff_mask):
            grad_e[valid_diff_mask] = np.abs(e_diff[valid_diff_mask] / a_diff[valid_diff_mask])
            grad_c[valid_diff_mask] = np.abs(c_diff[valid_diff_mask] / a_diff[valid_diff_mask])

        # Case 1: Negative jumps in endogenous grid OR gradient of c exceeds threshold
        # Additional condition: q_diff > 0 (value function must be increasing over the jump)
        jumps_case_1 = ((e_diff < 0) | (grad_c > grad_c_threshold)) & (q_diff > 0)
        j_idx_case_1 = np.where(jumps_case_1)[0]
        n_jump_case_1 = j_idx_case_1.size
        
        # Total jumps
        n_jump = n_jump_case_1
        
        # Calculate number of points to add per jump segment
        min_following_seg_len = 4  # Require at least this many points after the jump
        if use_mean_spacing and mean_e_diff > 0 and n_jump_case_1 > 0:
            # Pre-compute n_points for each jump segment based on mean spacing
            n_points_after = np.zeros(n_jump_case_1, dtype=np.int64)
            n_points_before = np.zeros(n_jump_case_1, dtype=np.int64)
            allow_after = np.ones(n_jump_case_1, dtype=np.bool_)
            allow_before = np.ones(n_jump_case_1, dtype=np.bool_)
            
            for idx, k in enumerate(j_idx_case_1):
                # Determine available length of the following segment before the next jump
                next_jump_idx = j_idx_case_1[idx + 1] if idx + 1 < n_jump_case_1 else n_old - 1
                after_len = (next_jump_idx - (k + 1)) + 1  # inclusive length
                if after_len < min_following_seg_len:
                    allow_after[idx] = False

                # Require the value function to be locally increasing only on the
                # *next* (right) segment where we add the LHS constraint segment.
                # The RHS constraint segment of the previous (left) segment should
                # still be added even if vf is locally decreasing to the left.
                #
                # "after" segment uses point k+1, so require vf_old[k+1] - vf_old[k] > 0
                if q_diff[k] <= 0:
                    allow_after[idx] = False
                
                # "after" segment bounds (using k+1)
                if jump_extend in ("after", "both") and allow_after[idx]:
                    c_star = c_old[k+1]
                    lb_c = max(1e-8, c_star - c_star_lb_pct)
                    ub_c = c_star * (1.0 + c_star_ub_pct) 
                    interval_width = ub_c - lb_c
                    n_pts = max(2, int(np.ceil(interval_width / mean_e_diff)))
                    n_points_after[idx] = n_pts
                
                # "before" segment bounds (using k)
                if jump_extend in ("before", "both") and allow_before[idx]:
                    c_star2 = c_old[k]
                    lb_c2 = max(1e-8, c_star2 * (1.0 - c_star_lb_pct))
                    ub_c2 = c_star2 * (1.0 + c_star_ub_pct)
                    interval_width2 = ub_c2 - lb_c2
                    n_pts2 = max(2, int(np.ceil(interval_width2 / mean_e_diff)))
                    n_points_before[idx] = n_pts2
            
            # Total points to add from jumps
            total_jump_points = int(np.sum(n_points_after) + np.sum(n_points_before))
            n_add = n_con + total_jump_points
        else:
            allow_after = np.ones(n_jump_case_1, dtype=np.bool_)
            allow_before = np.ones(n_jump_case_1, dtype=np.bool_)

            # Same monotonicity guard in the fixed-count case:
            # apply only to the "after" (next-segment LHS) constraint segment.
            for idx, k in enumerate(j_idx_case_1):
                if q_diff[k] <= 0:
                    allow_after[idx] = False

            # Fall back to fixed n_con_nxt per segment
            if jump_extend == "both":
                segments_per_jump = 2
            else:
                segments_per_jump = 1
            n_add = n_con + n_jump_case_1 * n_con_nxt * segments_per_jump
    
    n_total = n_old + n_add

    # ---- 1. Allocate output containers ----
    e_new = np.empty(n_total, dtype=e_old.dtype)
    vf_new = np.empty(n_total, dtype=vf_old.dtype)
    c_new = np.empty(n_total, dtype=c_old.dtype)
    a_new = np.empty(n_total, dtype=a_old.dtype)

    p = 0  # Write pointer into the new arrays
    
    # ---- 2. Borrowing-constraint segment (always first) ----
    min_c = np.min(e_old)
    # Use safer lower bound for numerical stability with float64
    c_con = np.linspace(1e-10, min_c, n_con)  # Changed from 1e-100 to 1e-10
    e_con = c_con + a_old[0] # m = c at the constraint
    vf_con = u_func(c_con, h_nxt) + beta * vf_next[0]
    # More efficient: use np.full instead of empty + fill
    a_con = np.full(n_con, a_old[0], dtype=c_con.dtype)  # Borrowing limit

    e_new[p:p+n_con] = e_con
    vf_new[p:p+n_con] = vf_con
    c_new[p:p+n_con] = c_con
    a_new[p:p+n_con] = a_con
    p += n_con

    # ---- 3. Jump segments (only if flag is True) ----
    if add_jump_constraints and n_con_nxt > 0:
        # Process Case 1 jumps (negative jumps in endogenous grid)
        # Segments added depend on jump_extend setting:
        #   "after": only add segment using k+1 (point after the jump)
        #   "before": only add segment using k (point before the jump)
        #   "both": add both segments
        for idx, k in enumerate(j_idx_case_1):
            # Determine number of points for this segment
            # Use pre-computed counts if using mean spacing, else fall back to n_con_nxt
            if n_points_after is not None:
                n_pts_after = int(n_points_after[idx])
            else:
                n_pts_after = n_con_nxt
            
            if n_points_before is not None:
                n_pts_before = int(n_points_before[idx])
            else:
                n_pts_before = n_con_nxt
            
            # Skip adding "after" segment if the following segment is too short
            if not allow_after[idx]:
                n_pts_after = 0
            if not allow_before[idx]:
                n_pts_before = 0
            
            # First segment: using k+1 (point after the jump)
            # Only add if jump_extend is "after" or "both"
            if jump_extend in ("after", "both") and n_pts_after > 0:
                a_star = a_old[k+1]
                c_star = c_old[k+1]

                # Create consumption segment around c_star at k+1
                # Use percentage-based bounds proportional to c_star
                lb_c = max(1e-8, c_star - c_star_lb_pct)
                ub_c = c_star * (1.0 + c_star_ub_pct)
                c_seg = np.linspace(lb_c, ub_c, n_pts_after).astype(np.float64)

                # FOC check: Filter points based on first-order condition (vectorized)
                if lambda_next is not None and uc_func is not None:
                    # Vectorized evaluation of marginal utility for all points
                    valid_mask = np.ones(n_pts_after, dtype=np.bool_)

                    # Compute all marginal utilities at once (vectorized)
                    # This is much faster than the loop
                    mu_c_vec = np.zeros(n_pts_after)
                    for i in range(n_pts_after):
                        mu_c_vec[i] = uc_func(c_seg[i], h_nxt)

                    # Vectorized comparison
                    valid_mask = (mu_c_vec >= lambda_next[k+1])

                    # Override FOC filtering if requested (for testing)
                    if override_KT_conditions:
                        valid_mask = np.ones(n_pts_after, dtype=np.bool_)
                    c_seg_valid = c_seg[valid_mask]
                    n_valid = c_seg_valid.size
                else:
                    # No FOC check, keep all points
                    c_seg_valid = c_seg
                    n_valid = n_pts_after

                # Add valid points
                if n_valid > 0:
                    m_seg = a_star + c_seg_valid
                    vf_seg = u_func(c_seg_valid, h_nxt) + beta * vf_next[k+1]

                    e_new[p:p+n_valid] = m_seg
                    vf_new[p:p+n_valid] = vf_seg
                    c_new[p:p+n_valid] = c_seg_valid
                    a_new[p:p+n_valid] = a_star
                    p += n_valid

            # Second segment: using k (point before the jump)
            # Only add if jump_extend is "before" or "both"
            if jump_extend in ("before", "both"):
                a_star2 = a_old[k]
                c_star2 = c_old[k]

                # Create consumption segment around c_star at k
                # Use percentage-based bounds proportional to c_star
                lb_c2 = max(1e-8, c_star2 * (1.0 - c_star_lb_pct))
                ub_c2 = c_star2 * (1.0 + c_star_ub_pct)
                c_seg2 = np.linspace(lb_c2, ub_c2, n_pts_before).astype(np.float64)

                # FOC check: Filter points based on first-order condition (vectorized)
                if lambda_next is not None and uc_func is not None:
                    # Vectorized evaluation of marginal utility for all points
                    valid_mask2 = np.ones(n_pts_before, dtype=np.bool_)

                    # Compute all marginal utilities at once (vectorized)
                    mu_c2_vec = np.zeros(n_pts_before)
                    for i in range(n_pts_before):
                        mu_c2_vec[i] = uc_func(c_seg2[i], h_nxt)

                    # Vectorized comparison
                    valid_mask2 = (mu_c2_vec <= lambda_next[k] + 1e-1)

                    # Override FOC filtering if requested (for testing)
                    if override_KT_conditions:
                        valid_mask2 = np.ones(n_pts_before, dtype=np.bool_)
                    c_seg2_valid = c_seg2[valid_mask2]
                    n_valid2 = c_seg2_valid.size
                else:
                    # No FOC check, keep all points
                    c_seg2_valid = c_seg2
                    n_valid2 = n_pts_before

                # Add valid points
                if n_valid2 > 0:
                    m_seg2 = a_star2 + c_seg2_valid
                    vf_seg2 = u_func(c_seg2_valid, h_nxt) + beta * vf_next[k]

                    e_new[p:p+n_valid2] = m_seg2
                    vf_new[p:p+n_valid2] = vf_seg2
                    c_new[p:p+n_valid2] = c_seg2_valid
                    a_new[p:p+n_valid2] = a_star2
                    p += n_valid2
        

    # ---- 4. Copy the original solution after all extras ----
    # Note: p now contains the actual number of added points (may be < n_add due to FOC filtering)
    # Copy original solution starting at position p
    n_orig_to_copy = n_old
    e_new[p:p+n_orig_to_copy] = e_old
    vf_new[p:p+n_orig_to_copy] = vf_old
    c_new[p:p+n_orig_to_copy] = c_old
    a_new[p:p+n_orig_to_copy] = a_old

    # Trim arrays to actual size (p + n_old)
    actual_size = p + n_orig_to_copy
    e_new = e_new[:actual_size]
    vf_new = vf_new[:actual_size]
    c_new = c_new[:actual_size]
    a_new = a_new[:actual_size]

    # Sort output arrays only for FUES (by endogenous grid m)
    # For DCEGM (sort_output=False): keep original order with constraint points prepended
    # The prepended constraint points have a_nxt = a_old[0] (minimum), and
    # original EGM points are already in exogenous grid order (increasing a_nxt)
    if sort_output:
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
                   jump_extend="both",  # NEW: Which segments to extend at jumps
                   sort_output=True,    # Sort output by endogenous grid (True for FUES, False for DCEGM)
                   tax_constraint_nodes=None,  # NEW: Tax bracket constraint nodes
                   n_points_per_node=10,  # NEW: Configurable constraint points per tax node
                   **kwargs):
    """
    Wrapper that preprocesses EGM solution by:
      • Adding borrowing constraint points
      • Optionally adding jump constraint points (for pb != 1)
      • Optionally adding tax bracket constraint points (for use_taxes=True, pb=1)
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
    jump_extend : str, optional
        Which segments to extend at jumps. Options:
        - "before": only extend segment before jump (using point k)
        - "after": only extend segment after jump (using point k+1)
        - "both": extend both segments (default)
    tax_constraint_nodes : list of dict, optional
        List of tax bracket constraint specifications. Each dict contains:
        - 'a_idx': index in a grid for the bracket boundary
        - 'c_lb_pct', 'c_ub_pct': consumption bounds as percentages
        Only used when use_taxes=True and pb=1. Default: None
    **kwargs
        Additional arguments passed to _egm_preprocess_core:
        - lambda_next: Marginal utility values (already scaled)
        - uc_func: Marginal utility function
        - override_KT_conditions: Override FOC filtering
        - grad_c_threshold: Threshold for grad_c jump detection (default: 1.04)
        - c_star_lb_pct: Percentage for c_star lower bound (default: 0.10 = 10%)
        - c_star_ub_pct: Percentage for c_star upper bound (default: 0.10 = 10%)
    
    Returns
    -------
    tuple
        (e_out, vf_out, c_out, a_out) - Preprocessed and cleaned arrays
    """

    # Choose a default c_max if not specified
    if c_max is None:
        c_max = 1.05 * np.max(c)  # 5% above current max consumption

    # Ensure float64 precision for all input arrays
    egrid = np.asarray(egrid, dtype=np.float64)
    vf = np.asarray(vf, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    vf_next = np.asarray(vf_next, dtype=np.float64)

    # remove points where egrid is negative
    #valid_mask = egrid >= 0
    #egrid = egrid[valid_mask]
    #vf = vf[valid_mask]
    #c = c[valid_mask]
    #a = a[valid_mask]

    # Run the fast core with jump constraint flag and FOC checking
    # NOTE: When tax constraints are used (pb=1), we disable jump constraints
    effective_add_jump = add_jump_constraints and (tax_constraint_nodes is None or len(tax_constraint_nodes) == 0)

    e_cat, vf_cat, c_cat, a_cat = _egm_preprocess_core(
        egrid, vf, c, a,
        vf_next, beta, u_func,
        m_bar, n_con, n_con_nxt, c_max, h_nxt,
        effective_add_jump,  # Disable jump constraints when using tax constraints
        lambda_next=kwargs.get('lambda_next'),  # Pass through lambda (already scaled)
        uc_func=kwargs.get('uc_func'),          # Pass through marginal utility
        override_KT_conditions=kwargs.get('override_KT_conditions', False),  # Pass override setting
        jump_extend=jump_extend,  # Pass through jump extend option
        grad_c_threshold=kwargs.get('grad_c_threshold', 1.04),  # Pass through grad_c threshold
        c_star_lb_pct=kwargs.get('c_star_lb_pct', 0.10),  # Pass through c_star lb percentage
        c_star_ub_pct=kwargs.get('c_star_ub_pct', 0.10),  # Pass through c_star ub percentage
        use_mean_spacing=kwargs.get('use_mean_spacing', True),  # Use mean e_diff for point spacing
        sort_output=sort_output)  # Sort by endogenous grid (True for FUES, False for DCEGM)

    # Add tax constraint segments if provided (for use_taxes=True, pb=1)
    if tax_constraint_nodes is not None and len(tax_constraint_nodes) > 0:
        # Convert list of dicts to arrays for njit function
        tax_node_indices = np.array([n['a_idx'] for n in tax_constraint_nodes], dtype=np.int64)
        tax_node_c_lb_pct = np.array([n['c_lb_pct'] for n in tax_constraint_nodes], dtype=np.float64)
        tax_node_c_ub_pct = np.array([n['c_ub_pct'] for n in tax_constraint_nodes], dtype=np.float64)
        tax_node_is_lhs = np.array([1 if n.get('side') == 'lhs' else 0 for n in tax_constraint_nodes], dtype=np.int8)

        # IMPORTANT: `a_idx` is defined on the base `a` grid (e.g. a_nxt_grid),
        # not on the post-preprocessed arrays. Use the base-grid aware helper.
        e_cat, vf_cat, c_cat, a_cat = _add_tax_constraint_segments_from_base_grid(
            e_cat, vf_cat, c_cat, a_cat,
            vf_next, beta, u_func, h_nxt,
            a, c, vf,
            tax_node_indices, tax_node_c_lb_pct, tax_node_c_ub_pct, tax_node_is_lhs,
            n_points_per_node=n_points_per_node,
            lambda_next=kwargs.get('lambda_next'),
            uc_func=kwargs.get('uc_func'),
            override_KT_conditions=kwargs.get('override_KT_conditions', False),
        )

        # Re-sort after adding tax constraints if needed
        if sort_output:
            sort_idx = np.argsort(e_cat)
            e_cat = e_cat[sort_idx]
            vf_cat = vf_cat[sort_idx]
            c_cat = c_cat[sort_idx]
            a_cat = a_cat[sort_idx]

    # Remove duplicates based on endogenous grid and value function
    # Use a tolerance appropriate for float64 precision and the scale of the problem
    tol = 1e-10 if not add_jump_constraints else 1e-8  # More tolerance when delta != 1
    unique_ids = uniqueEG(e_cat, vf_cat, tol=tol)

    e_out = e_cat[unique_ids]
    vf_out = vf_cat[unique_ids]
    c_out = c_cat[unique_ids]
    a_out = a_cat[unique_ids]

    return e_out, vf_out, c_out, a_out


@njit
def uc_test(c, h):
    """
    Simple test marginal utility function for FOC verification.

    Parameters
    ----------
    c : float
        Consumption
    h : float
        Housing

    Returns
    -------
    float
        Marginal utility of consumption

    Notes
    -----
    This is a simple CRRA marginal utility for testing FOC checks.
    Can be imported and passed directly to egm_preprocess via uc_func parameter.
    """
    gamma = 2.0  # Risk aversion parameter
    alpha = 0.77  # Consumption share in Cobb-Douglas
    return alpha/c


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
#  GPU Device Functions (only defined when CUDA is available)
# ======================================================================

if CUDA_AVAILABLE and cuda is not None:
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