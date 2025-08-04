import numpy as np
from scipy.interpolate import interp1d
from numba import njit, cuda
from numba.typed import Dict    
from typing import Callable, Literal   # NEW  – remove if unused        # NEW
import time
from functools import lru_cache
import math


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

@njit
def piecewise_gradient_3rd(f, x, m_bar, eps=0.9):
    """
    Third-order finite differences that
      • never straddle jumps  |Δf/Δx| > m_bar
      • enforce strictly-positive slopes

    Parameters
    ----------
    f, x : 1-D ndarrays (same length, x strictly increasing)
    m_bar: float   – jump threshold in *slope* space
    eps   : float  – fallback slope if no positive neighbour exists

    Returns
    -------
    g : 1-D ndarray, positive slope at each x[i]
    """
    n = len(x)
    g_raw = np.empty(n)

    def smooth(i, j):
        """True if segment [i,j] is smooth (no jump)."""
        df = f[j] - f[i]
        dx = x[j] - x[i]
        return np.abs(df / dx) <= m_bar

    # ---- pass 1 : compute local derivatives or set NaN -------------
    for i in range(n):
        # indices we *might* use for the 5-point stencil
        i_m2, i_m1, i_p1, i_p2 = i - 2, i - 1, i + 1, i + 2

        # check which neighbours are inside the array and smooth
        have_m2 = (i_m2 >= 0)   and smooth(i_m2, i_m1)
        have_m1 = (i_m1 >= 0)   and smooth(i_m1, i)
        have_p1 = (i_p1 < n)    and smooth(i, i_p1)
        have_p2 = (i_p2 < n)    and smooth(i_p1, i_p2)

        if have_m2 and have_m1 and have_p1 and have_p2:
            # ---- 5-point Richardson (O(h^3)) -----------------------
            h1  = x[i_p1] - x[i_m1]
            D1  = (f[i_p1] - f[i_m1]) / h1

            h2  = x[i_p2] - x[i_m2]
            D2  = (f[i_p2] - f[i_m2]) / h2

            g_raw[i] = (4.0 * D1 - D2) / 3.0
        elif have_m1 and have_p1:
            # ---- 3-point centred (O(h^2)) --------------------------
            g_raw[i] = (f[i_p1] - f[i_m1]) / (x[i_p1] - x[i_m1])
        elif have_p1:
            # ---- 2-point forward  (O(h)) ---------------------------
            g_raw[i] = (f[i_p1] - f[i]) / (x[i_p1] - x[i])
        elif have_m1:
            # ---- 2-point backward (O(h)) ---------------------------
            g_raw[i] = (f[i] - f[i_m1]) / (x[i] - x[i_m1])
        else:
            g_raw[i] = np.nan

        # mark non-positive slopes as invalid
        if not np.isnan(g_raw[i]) and g_raw[i] <= 0.0:
            g_raw[i] = np.nan

    # ---- pass 2 : fill NaNs with nearest positive neighbour --------
    g = np.empty(n)
    for i in range(n):
        if not np.isnan(g_raw[i]):
            g[i] = g_raw[i]
            continue

        # search outward
        offset = 1
        found  = False
        while not found and (i - offset >= 0 or i + offset < n):
            if i - offset >= 0 and not np.isnan(g_raw[i - offset]):
                g[i] = g_raw[i - offset]
                found = True
            elif i + offset < n and not np.isnan(g_raw[i + offset]):
                g[i] = g_raw[i + offset]
                found = True
            offset += 1

        if not found:          # ultimate fallback
            g[i] = eps

    return g

@njit
def piecewise_gradient(f, x, m_bar, eps=0.9):
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
        left_ok  = (i > 0)   and (np.abs(f[i]   - f[i-1])/(x[i]   - x[i-1]) <= m_bar)
        right_ok = (i < n-1) and (np.abs(f[i+1] - f[i]  )/(x[i+1] - x[i]) <= m_bar)

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
def _egm_preprocess_core(e_old, vf_old, c_old, a_old,
                         vf_next,              # 1-D, same length as old grids
                         beta, u_func,         # u_func must be @njit-able
                         m_bar,                # jump threshold
                         n_con,                # # constraint nodes
                         n_con_nxt,
                         c_max, h_nxt):               # upper end of [c*, c_max]
    """
    Returns new (e,vf,c,a) with

        • n_con borrowing-constraint points, plus
        • n_con points on every 'big' jump in a (|Δa| > m_bar),

    all prepended in one shot.  No np.concatenate used.
    """

    # ---- 0.   basic sizes --------------------------------------------------
    n_old   = e_old.size
    #diff  = a_old[1:] - a_old[:-1]
    #base  = e_old[1:] - e_old[:-1]
    diff = e_old[1:] - e_old[:-1]
    base = a_old[1:] - a_old[:-1]

    del_vf =vf_next[1:] - vf_next[:-1]

    # relative gap |Δa| / |a_{t}|   (robust to a_{t}=0)

    if n_con_nxt>0:
        rel_gap = np.empty_like(diff)
        for i in range(diff.size):
            b = base[i]
            rel_gap[i] = np.abs(diff[i] / b) if b != 0.0 else np.inf
    
        #jumps  = rel_gap > m_bar          # m_bar is now a relative threshold
        jumps  = diff<0 
        #del_vf_bool = del_vf<0
        #jumps = jumps*del_vf_bool
        j_idx  = np.where(jumps)[0]
        j_idx   = np.where(jumps)[0]          # jump i  ⇒  segment between i and i+1
        n_jump  = j_idx.size
        n_add   = n_con + n_jump * n_con_nxt      # total new nodes
        n_total = n_old + n_add
    else:
        n_add = n_con
        n_total = n_old + n_add
        n_jump = 0

    #print(j_idx)
    #print(beta)
    # ---- 1.   allocate output containers ----------------------------------
    e_new  = np.empty(n_total, dtype=e_old.dtype)
    vf_new = np.empty(n_total, dtype=vf_old.dtype)
    c_new  = np.empty(n_total, dtype=c_old.dtype)
    a_new  = np.empty(n_total, dtype=a_old.dtype)

    p = 0   # write pointer into the new arrays
    # -----------------------------------------------------------------------
    # 2.  Borrowing-constraint segment  (always first)
    # -----------------------------------------------------------------------
    min_c  = np.min(e_old)  
    c_con = np.linspace(1e-100, min_c, n_con)
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

    if n_con_nxt>0:
        for k in j_idx:
            a_star = a_old[k+1]
            c_star = c_old[k+1]
            e_star = e_old[k+1]
            
            lb_c = max(1e-10,c_star-10)

            c_seg = np.linspace(lb_c, c_star, n_con_nxt).astype(c_old.dtype)
            #a_seg = np.linspace(a_star, a_star+2, n_con).astype(a_old.dtype)
            m_seg = a_star + c_seg

            vf_seg = u_func(c_seg, h_nxt) + beta * vf_next[k+1]

            e_new[p:p+n_con]  = m_seg
            vf_new[p:p+n_con] = vf_seg
            c_new[p:p+n_con]  = c_seg
            a_new[p:p+n_con]  = a_star
            p += n_con
            

        # for 

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
                   n_con_nxt = 0,
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

    # choose a default c_max if the caller doesn't specify one
    if c_max is None:
        c_max = 1.05 * np.max(c)          # 5 % above current max consumption


    #  (If you still need monotone sorting, re-enable the block below)
    #sort_idx = np.argsort(egrid)
    #egrid, vf,vf_next, c, a = (arr[sort_idx] for arr in (egrid, vf, vf_next,c, a))


    # ---- run the fast core -------------------------------------------------
    e_cat, vf_cat, c_cat, a_cat = _egm_preprocess_core(
        egrid, vf, c, a,
        vf_next, beta, u_func,
        m_bar, n_con,n_con_nxt, c_max, h_nxt)

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
#  GPU Device Functions
# ======================================================================

@cuda.jit(device=True)
def searchsorted_gpu(a, v):
    """
    A GPU-compatible binary search implementation, equivalent to
    np.searchsorted(a, v, side='right').
    """
    lower_bound = 0
    upper_bound = len(a)
    while lower_bound < upper_bound:
        i = lower_bound + (upper_bound - lower_bound) // 2
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