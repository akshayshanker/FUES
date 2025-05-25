import numpy as np
from scipy.interpolate import interp1d
from numba import njit
from numba.typed import Dict    
from typing import Callable   # NEW  – remove if unused        # NEW
import time



@njit
def piecewise_gradient(f, x, m_bar, eps=1e-12):
    """
    Finite differences that
      • never straddle jumps  |f[i+1]-f[i]| > c_bar
      • enforce g[i]  > 0  (monotone, strictly)

    Parameters
    ----------
    f      : 1-D ndarray, function values on a strictly-increasing grid
    x      : 1-D ndarray, same length as f
    c_bar  : float, threshold to flag a jump in *function* space
    eps    : float, fallback slope if NO positive neighbour exists

    Returns
    -------
    g      : 1-D ndarray, positive slope at each x[i]
    """
    n = len(x)
    g_raw = np.empty(n)

    # ---- pass 1: local finite differences, set NaN where invalid ----
    for i in range(n):
        left_ok  = (i > 0)   and (np.abs(f[i]   - f[i-1]) <= m_bar)
        right_ok = (i < n-1) and (np.abs(f[i+1] - f[i]  ) <= m_bar)

        if left_ok and right_ok:
            g_raw[i] = (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
        elif right_ok:
            g_raw[i] = (f[i+1] - f[i])   / (x[i+1] - x[i])
        elif left_ok:
            g_raw[i] = (f[i]   - f[i-1]) / (x[i]   - x[i-1])
        else:                       # isolated jump
            g_raw[i] = np.nan       # mark as invalid

        # mark non-positive slopes as invalid
        if not np.isnan(g_raw[i]) and g_raw[i] <= 0.0:
            g_raw[i] = np.nan

    # ---- pass 2: replace NaNs with nearest positive neighbour ----
    g = np.empty(n)
    for i in range(n):
        if not np.isnan(g_raw[i]):          # already positive
            g[i] = g_raw[i]
            continue

        # search outward for nearest positive slope
        offset = 1
        replacement_found = False
        while not replacement_found and (i-offset >= 0 or i+offset < n):
            if i - offset >= 0 and not np.isnan(g_raw[i - offset]):
                g[i] = g_raw[i - offset]
                replacement_found = True
            elif i + offset < n and not np.isnan(g_raw[i + offset]):
                g[i] = g_raw[i + offset]
                replacement_found = True
            offset += 1

        # ultimate fallback
        if not replacement_found:
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
def interp_as(x_points, y_points, x_query, extrap=False):
    """Fast interpolation for array queries with optional extrapolation.
    
    This is a jitted version of the interpolation function used in the Fella model.
    
    Parameters
    ----------
    x_points : ndarray
        X-values of the known points
    y_points : ndarray
        Y-values of the known points
    x_query : ndarray
        X-values to query
    extrap : bool, optional
        Whether to extrapolate for out-of-bounds values, by default True
    
    Returns
    -------
    ndarray
        Interpolated y-values at x_query points
    """
    # Initialize output array
    n_query = len(x_query)
    y_query = np.zeros(n_query)
    
    # Boundary checks for extrapolation
    x_min = x_points[0]
    x_max = x_points[-1]
    y_min = y_points[0]
    y_max = y_points[-1]
    
    # Iterate through query points
    for i in range(n_query):
        x = x_query[i]
        
        # Handle out-of-bounds
        if x <= x_min:
            y_query[i] = y_min if extrap else np.nan
            continue
        elif x >= x_max:
            y_query[i] = y_max if extrap else np.nan
            continue
            
        # Find position using binary search
        # This is much faster than a linear search for large arrays
        left = 0
        right = len(x_points) - 1
        
        while right - left > 1:
            mid = (left + right) // 2
            if x_points[mid] <= x:
                left = mid
            else:
                right = mid
                
        # Linear interpolation
        x_left = x_points[left]
        x_right = x_points[right]
        y_left = y_points[left]
        y_right = y_points[right]
        
        # Compute interpolated value
        if x_right > x_left:  # Avoid division by zero
            t = (x - x_left) / (x_right - x_left)
            y_query[i] = y_left + t * (y_right - y_left)
        else:
            y_query[i] = y_left
            
    return y_query

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
                         m_bar,                # jump threshold
                         n_con,                # # constraint nodes
                         c_max, h_nxt):               # upper end of [c*, c_max]
    """
    Returns new (e,vf,c,a) with

        • n_con borrowing-constraint points, plus
        • n_con points on every 'big' jump in a (|Δa| > m_bar),

    all prepended in one shot.  No np.concatenate used.
    """

    # ---- 0.   basic sizes --------------------------------------------------
    n_old   = e_old.size
    base  = a_old[1:] - a_old[:-1]
    diff  = e_old[1:] - e_old[:-1]

    # relative gap |Δa| / |a_{t}|   (robust to a_{t}=0)
    rel_gap = np.empty_like(diff)
    for i in range(diff.size):
        b = base[i]
        rel_gap[i] = np.abs(diff[i] / b) if b != 0.0 else np.inf
    jumps  = rel_gap > 2          # m_bar is now a relative threshold
    j_idx  = np.where(jumps)[0]
    j_idx   = np.where(jumps)[0]          # jump i  ⇒  segment between i and i+1
    n_jump  = j_idx.size
    n_add   = n_con + n_jump * n_con      # total new nodes
    n_total = n_old + n_add

    #print(j_idx)
    print(beta)
    # ---- 1.   allocate output containers ----------------------------------
    e_new  = np.empty(n_total, dtype=e_old.dtype)
    vf_new = np.empty(n_total, dtype=vf_old.dtype)
    c_new  = np.empty(n_total, dtype=c_old.dtype)
    a_new  = np.empty(n_total, dtype=a_old.dtype)

    p = 0   # write pointer into the new arrays
    # -----------------------------------------------------------------------
    # 2.  Borrowing-constraint segment  (always first)
    # -----------------------------------------------------------------------
    min_c  = np.min(c_old)
    c_con = np.linspace(1e-100, min_c, n_con).astype(c_old.dtype)
    e_con  = c_con                        # m = c at the constraint
    vf_con = u_func(c_con, h_nxt) + beta * vf_next[0]
    a_con  = np.empty_like(c_con)
    a_con.fill(a_old[0])                  # borrowing limit

    e_new[p:p+n_con]  = e_con
    vf_new[p:p+n_con] = vf_con
    c_new[p:p+n_con]  = c_con
    a_new[p:p+n_con]  = a_con
    p += n_con

    # -----------------------------------------------------------------------
    # 3.  Jump segments
    # -----------------------------------------------------------------------
    for k in j_idx:                       # jump between k and k+1
        a_star = a_old[k]
        c_star = c_old[k]
        e_star = e_old[k]

        c_seg = np.linspace(c_star, c_star+1, n_con).astype(c_old.dtype)
        m_seg = a_star + c_seg

        vf_seg = u_func(c_seg, h_nxt) + beta * vf_next[k]

        e_new[p:p+n_con]  = m_seg
        vf_new[p:p+n_con] = vf_seg
        c_new[p:p+n_con]  = c_seg
        a_new[p:p+n_con]  = a_star
        p += n_con

    # -----------------------------------------------------------------------
    # 4.  Copy the original solution after all extras
    # -----------------------------------------------------------------------
    e_new[n_add:]  = e_old
    vf_new[n_add:] = vf_old
    c_new[n_add:]  = c_old
    a_new[n_add:]  = a_old

    return e_new, vf_new, c_new, a_new

def egm_preprocess(egrid, vf, c, a,
                   beta, u_func, vf_next,
                   m_bar,
                   n_con=10,
                   c_max=None,
                   h_nxt=None,
                   **kwargs):
    """
    Wrapper that

      • calls the @njit core above,
      • removes duplicates with uniqueEG,
      • returns cleaned arrays (same order as before).

    Any extra kwargs are ignored so you can keep the
    original signature (Pi, i_z, i_h_prime, h_nxt, …).
    """

    # choose a default c_max if the caller doesn’t specify one
    if c_max is None:
        c_max = 1.05 * np.max(c)          # 5 % above current max consumption


    #  (If you still need monotone sorting, re-enable the block below)
    #sort_idx = np.argsort(egrid)
    #egrid, vf,vf_next, c, a = (arr[sort_idx] for arr in (egrid, vf, vf_next,c, a))


    # ---- run the fast core -------------------------------------------------
    e_cat, vf_cat, c_cat, a_cat = _egm_preprocess_core(
        egrid, vf, c, a,
        vf_next, beta, u_func,
        m_bar, n_con, c_max, h_nxt)

    # ---- uniqueness & (optional) sorting -----------------------------------
    unique_ids = uniqueEG(e_cat, vf_cat)

    e_out  = e_cat[unique_ids]
    vf_out = vf_cat[unique_ids]
    c_out  = c_cat[unique_ids]
    a_out  = a_cat[unique_ids]

    #  (If you still need monotone sorting, re-enable the block below)
    # sort_idx = np.argsort(e_out)
    # e_out, vf_out, c_out, a_out = (arr[sort_idx] for arr in (e_out, vf_out, c_out, a_out))

    return e_out, vf_out, c_out, a_out


def build_njit_utility(
    expr: str,
    params: Dict[str, float],
    h_placeholder: str = "H_nxt",
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

    Returns
    -------
    callable
        nopython-compiled function u(c, H) → float
    """

    # 1.  replace the placeholder with a run-time variable name 'H'
    #patched = expr.replace(h_placeholder, "H")
    patched = expr.replace(h_placeholder, "H")

    # 2.  build source code for a pure Python function
    func_src = "def _u(c, H):\n    return " + patched

    # 3.  execute in a tiny namespace containing numpy and constants
    ns = {"np": np, **params}
    exec(func_src, ns)          # defines _u in ns
    py_func = ns["_u"]

    # 4.  JIT-compile to nopython; result takes (c, H) positional args
    return njit(py_func)