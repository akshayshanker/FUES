import numpy as np
from scipy.interpolate import interp1d
from numba import njit, prange
import time

def uniqueEG(grid, values):
    """
    Find unique indices in the endogenous grid to ensure strict monotonicity.
    
    This approach mimics the uniqueEG function from fella.py implementation,
    but uses numpy for simplicity.
    
    Parameters
    ----------
    grid : ndarray
        Endogenous grid (cash-on-hand)
    values : ndarray
        Values associated with grid points (e.g., value function)
    
    Returns
    -------
    ndarray
        Boolean array of unique indices
    """
    # Sort indices by grid values
    sort_indices = np.argsort(grid)
    grid_sorted = grid[sort_indices]
    values_sorted = values[sort_indices]
    
    # Find where grid values are strictly increasing
    # (include first point always)
    n = len(grid)
    unique_mask = np.ones(n, dtype=bool)
    
    for i in range(1, n):
        # If grid point is the same as previous but value is lower, mark as non-unique
        if grid_sorted[i] == grid_sorted[i-1] and values_sorted[i] <= values_sorted[i-1]:
            unique_mask[sort_indices[i]] = False
    
    return unique_mask

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
def interp_as(x_points, y_points, x_query, extrap=True):
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
    for i in prange(n_a):
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
def housing_choice_solver(resource_grid, h_grid, h_next_grid, w_grid, value_grids, lambda_grids, tau, min_wealth):
    """
    Jitted function to solve the housing choice problem following Fella implementation.
    
    Parameters
    ----------
    resource_grid : 2D array (n_a, n_h)
        Resources available at each (a,h) combination
    h_grid : 1D array
        Housing grid (current period)
    h_next_grid : 1D array
        Next-period housing choice grid
    w_grid : 1D array
        Wealth grid for interpolation
    value_grids : list of 1D arrays
        List of value function grids for each housing choice option
    lambda_grids : list of 1D arrays
        List of lambda function grids for each housing choice option
    tau : float
        Transaction cost parameter
    min_wealth : float
        Minimum wealth value
    
    Returns
    -------
    tuple
        (best_values, best_lambdas, best_indices) arrays
    """
    # Get dimensions
    n_a, n_h = resource_grid.shape
    n_h_next = len(h_next_grid)
    
    # Output arrays
    best_values = np.full((n_a, n_h), -np.inf)
    best_lambdas = np.zeros((n_a, n_h))
    best_indices = np.zeros((n_a, n_h), dtype=np.int32)
    
    # Loop over all states
    for i_a in range(n_a):
        for i_h in range(n_h):
            # Current resource and housing values
            resource = resource_grid[i_a, i_h]
            h_current = h_grid[i_h]
            
            # Try each housing choice option
            for i_h_next in range(n_h_next):
                h_next = h_next_grid[i_h_next]
                
                # Calculate transaction cost
                chi = 0 if h_current == h_next else 1
                trans_cost = chi * tau * np.abs(h_next)
                
                # Calculate wealth
                w_dscn_val = resource + chi * (h_current - h_next) - trans_cost
                
                # Skip if wealth is below minimum
                if w_dscn_val < min_wealth:
                    continue
                
                # Get value and lambda through interpolation
                # Using the value grid for this housing choice
                v_vals = value_grids[i_h_next]
                l_vals = lambda_grids[i_h_next]
                
                # Linear interpolation
                if w_dscn_val <= w_grid[0]:
                    value = v_vals[0]
                    lambda_val = l_vals[0]
                elif w_dscn_val >= w_grid[-1]:
                    value = v_vals[-1]
                    lambda_val = l_vals[-1]
                else:
                    # Find position using binary search
                    left = 0
                    right = len(w_grid) - 1
                    
                    while right - left > 1:
                        mid = (left + right) // 2
                        if w_grid[mid] <= w_dscn_val:
                            left = mid
                        else:
                            right = mid
                    
                    # Interpolate
                    x_left = w_grid[left]
                    x_right = w_grid[right]
                    v_left = v_vals[left]
                    v_right = v_vals[right]
                    
                    l_left = l_vals[left]
                    l_right = l_vals[right]
                    
                    t = (w_dscn_val - x_left) / (x_right - x_left)
                    value = v_left + t * (v_right - v_left)
                    lambda_val = l_left + t * (l_right - l_left)
                
                # Update best choice if this is better
                if value > best_values[i_a, i_h]:
                    best_values[i_a, i_h] = value
                    best_lambdas[i_a, i_h] = lambda_val
                    best_indices[i_a, i_h] = i_h_next
    
    return best_values, best_lambdas, best_indices 

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
    min_c_val = np.min(c) + 1e-1
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

    # Concatenate constraint points with existing solution
    egrid_concat = np.concatenate((e_array, egrid))
    vf_concat = np.concatenate((vf_array, vf))
    c_concat = np.concatenate((c_array, c))
    a_concat = np.concatenate((b_array, a))

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