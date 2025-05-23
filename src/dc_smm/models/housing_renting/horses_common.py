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


def egm_preprocess(egrid, vf, c, a, beta, u_func, vf_next, Pi=None, i_z=None, i_h_prime=None, n_con=10, h_nxt=None):
    """
    Preprocess endogenous grid and associated values following the approach in fella.py.
    
    This function:
    1. Adds constraint points at the borrowing constraint with small consumption values
    2. Concatenates them with the main EGM solution
    3. Ensures uniqueness in the grid
    
    Parameters
    ----------
    egrid : ndarray
        Endogenous grid (cash-on-hand)
    vf : ndarray
        Value function values
    c : ndarray
        Consumption policy values
    a : ndarray
        Asset policy values
    beta : float
        Discount factor
    u_func : callable
        Utility function that takes consumption and housing as arguments
    vf_next : ndarray or float
        Value function for next period or continuation value at constraint
    Pi : ndarray, optional
        Transition matrix for income shocks, for computing expectations
    i_z : int, optional
        Current income state index
    i_h_prime : int, optional
        Housing choice state index
    n_con : int, optional
        Number of constraint points to add, default 10
    h_nxt : float, optional
        Housing value for current iteration
    
    Returns
    -------
    tuple
        (egrid_cleaned, vf_cleaned, c_cleaned, a_cleaned)
    """
    # Find minimum consumption in current solution
    min_c_val = np.min(c) 
    c_array = np.linspace(1e-100, min_c_val, n_con)
    e_array = c_array  # For constraint points, c = m (no savings)
    
    # Generate utility values for constraint points
    # Simple approach: just calculate utility directly for constraint points
    # We're using the constraint points at the borrowing limit
    #vf_array = np.zeros(n_con)
    #for i in range(n_con):
        # Use the provided housing value (could come from the outer loop in horses_c.py)
        # Default to 0.0 only if no h_nxt is provided
    h_val = 0.0 if h_nxt is None else h_nxt
    
    vf_array = u_func(**{"c": c_array, "H_nxt": h_val}) + beta * vf_next[0]
    
    # Asset policy at constraint is minimum asset value
    b_array = np.zeros(n_con)
    b_array.fill(a[0])  # Using first value of asset grid as borrowing constraint

    # Pre-allocate once and write by slice → ~2× faster than four separate
    # np.concatenate calls and avoids the temporary tuple objects.
    n_old = egrid.size
    n_new = n_con + n_old

    egrid_concat = np.empty(n_new, dtype=egrid.dtype)
    egrid_concat[:n_con] = e_array
    egrid_concat[n_con:] = egrid

    vf_concat = np.empty(n_new, dtype=vf.dtype)
    vf_concat[:n_con] = vf_array
    vf_concat[n_con:] = vf

    c_concat = np.empty(n_new, dtype=c.dtype)
    c_concat[:n_con] = c_array
    c_concat[n_con:] = c

    a_concat = np.empty(n_new, dtype=a.dtype)
    a_concat[:n_con] = b_array
    a_concat[n_con:] = a

    # Ensure uniqueness in grid
    uniqueIds = uniqueEG(egrid_concat, vf_concat)
    egrid_unique = egrid_concat[uniqueIds]
    vf_unique = vf_concat[uniqueIds]
    c_unique = c_concat[uniqueIds]
    a_unique = a_concat[uniqueIds]
    
    # Sort by grid values to ensure monotonicity
    #sort_indices = np.argsort(egrid_unique)
    #egrid_unique = egrid_unique[sort_indices]
    #vf_unique = vf_unique[sort_indices]
    #c_unique = c_unique[sort_indices] 
    #a_unique = a_unique[sort_indices]
    
    return egrid_unique, vf_unique, c_unique, a_unique


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