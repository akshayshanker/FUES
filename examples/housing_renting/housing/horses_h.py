import numpy as np
from scipy.interpolate import interp1d
from numba import njit
from .horses_common import _safe_interp, housing_choice_solver, interp_as

def F_shocks_dcsn_to_arvl(mover):
    """Create operator for shock integration in backward step.
    
    This integrates over income shocks for a given state.
    
    Parameters
    ----------
    mover : Mover
        The dcsn_to_arvl mover object with self-contained model
        
    Returns
    -------
    callable
        The operator function that transforms dcsn perch data to arvl perch data
    """
    # Extract income shock grid and transition matrix
    model = mover.model
    shock_info = model.num.shocks.income_shock
    Pi = shock_info.process.transition_matrix
    
    def operator(perch_data):
        """Transform decision data into arrival data by integrating over income shock.
        
        Parameters
        ----------
        perch_data : dict
            Decision perch data with value function and marginal value
            
        Returns
        -------
        dict
            Arrival perch data with integrated value function and marginal value
        """
        vlu_dcsn = perch_data["vlu"]
        lambda_dcsn = perch_data["lambda"]
        
        # Integrate over income states using einsum (matrix multiplication)
        vlu_arvl = np.einsum('ahj,ij->ahi', vlu_dcsn, Pi)
        lambda_arvl = np.einsum('ahj,ij->ahi', lambda_dcsn, Pi)
        
        return {
            "vlu": vlu_arvl,
            "lambda": lambda_arvl
        }
    
    return operator

@njit
def renter_housing_choice_solver(w_grid, S_grid, y_grid, w_rent_grid, vlu_cntn, lambda_cntn, Pr, shock_grid):
    """
    Jitted function to solve the renter housing choice problem.
    
    Parameters
    ----------
    w_grid : 1D array
        Wealth grid for decision
    S_grid : 1D array
        Rental housing service grid
    y_grid : 1D array
        Income shock grid indices
    w_rent_grid : 1D array
        Post-rental wealth grid for interpolation
    vlu_cntn : 3D array (n_w_rent, n_S, n_y)
        Value function grid for continuation
    lambda_cntn : 3D array (n_w_rent, n_S, n_y)
        Marginal value function grid for continuation
    Pr : float
        Rental price
    shock_grid : 1D array
        Income shock values
    
    Returns
    -------
    tuple
        (vlu_dcsn, lambda_dcsn, S_policy) arrays
    """
    # Get dimensions
    n_w = len(w_grid)
    n_S = len(S_grid)
    n_y = len(y_grid)
    
    # Initialize output arrays
    vlu_dcsn = np.zeros((n_w, n_y))
    lambda_dcsn = np.zeros((n_w, n_y))
    S_policy = np.zeros((n_w, n_y), dtype=np.int32)
    
    # Solve for each income state
    for i_y in range(n_y):
        y_val = shock_grid[i_y]
        
        # For each wealth level
        for i_w in range(n_w):
            w_dcsn_val = w_grid[i_w]
            
            # Initialize best values
            best_value = -np.inf
            best_lambda = 0.0
            best_S_idx = 0
            
            # Try each rental service level
            for i_S in range(n_S):
                S_val = S_grid[i_S]
                
                # Calculate post-rental wealth
                w_cntn_val = w_dcsn_val - Pr * S_val + y_val
                
                # Skip if not enough money for this rental level
                if w_cntn_val < w_rent_grid[0]:
                    continue
                
                # Linear interpolation for value and lambda
                # Handle boundary cases first
                if w_cntn_val <= w_rent_grid[0]:
                    vlu = vlu_cntn[0, i_S, i_y]
                    lambda_val = lambda_cntn[0, i_S, i_y]
                elif w_cntn_val >= w_rent_grid[-1]:
                    vlu = vlu_cntn[-1, i_S, i_y]
                    lambda_val = lambda_cntn[-1, i_S, i_y]
                else:
                    # Find position using binary search
                    left = 0
                    right = len(w_rent_grid) - 1
                    
                    while right - left > 1:
                        mid = (left + right) // 2
                        if w_rent_grid[mid] <= w_cntn_val:
                            left = mid
                        else:
                            right = mid
                    
                    # Linear interpolation
                    x_left = w_rent_grid[left]
                    x_right = w_rent_grid[right]
                    v_left = vlu_cntn[left, i_S, i_y]
                    v_right = vlu_cntn[right, i_S, i_y]
                    
                    l_left = lambda_cntn[left, i_S, i_y]
                    l_right = lambda_cntn[right, i_S, i_y]
                    
                    t = (w_cntn_val - x_left) / (x_right - x_left)
                    vlu = v_left + t * (v_right - v_left)
                    lambda_val = l_left + t * (l_right - l_left)
                
                # Update best choice if this is better
                if vlu > best_value:
                    best_value = vlu
                    best_lambda = lambda_val
                    best_S_idx = i_S
            
            # Store results
            vlu_dcsn[i_w, i_y] = best_value
            lambda_dcsn[i_w, i_y] = best_lambda
            S_policy[i_w, i_y] = best_S_idx
    
    return vlu_dcsn, lambda_dcsn, S_policy

def F_h_cntn_to_dcsn(mover):
    """Create operator for housing choice (for both owners and renters).
    
    Implements discrete housing choice through vectorized enumeration.
    Detects whether it's being called for owner or renter based on stage name.
    
    Parameters
    ----------
    mover : Mover
        The cntn_to_dcsn mover with self-contained model
        
    Returns
    -------
    callable
        The operator function that transforms cntn perch data to dcsn perch data
    """
    # Extract model
    model = mover.model
    # Get income shock information
    shock_info = model.num.shocks.income_shock
    shock_grid = shock_info.process.values
    
    # Determine if this is owner or renter housing stage
    # Check mover.stage_name which should be prefixed with the stage type
    is_renter = "RNTH" in mover.stage_name
    
    # Get grid data using grid proxies
    if is_renter:
        # Renter housing choice grids
        w_grid = model.num.state_space.dcsn.grids.w
        y_grid = model.num.state_space.dcsn.grids.y
        w_rent_grid = model.num.state_space.cntn.grids.w_rent  # Cash-on-hand after rental choice
        S_grid = model.num.state_space.cntn.grids.S  # Rental services grid
        
        # Get parameters
        params = model.param
        Pr = params.Pr  # Rental price per unit
    else:
        # Owner housing choice grids
        a_grid = model.num.state_space.dcsn.grids.a
        H_grid = model.num.state_space.dcsn.grids.H
        y_grid = model.num.state_space.dcsn.grids.y
        w_grid = model.num.state_space.cntn.grids.w_own   # Cash-on-hand grid from source perch
        H_nxt_grid = model.num.state_space.cntn.grids.H_nxt  # Housing choice grid
        
        # Get parameters
        params = model.param
        tau = params.phi
        r = params.r
        Pr = params.Pr
    
    def operator(perch_data):
        vlu_cntn = perch_data["vlu"] 
        lambda_cntn = perch_data["lambda"]
        
        if is_renter:
            # Use numba-optimized renter housing choice solver
            vlu_dcsn, lambda_dcsn, S_policy = renter_housing_choice_solver(
                w_grid, S_grid, y_grid, w_rent_grid, 
                vlu_cntn, lambda_cntn, Pr, shock_grid
            )
            
            return {
                "vlu": vlu_dcsn,
                "lambda": lambda_dcsn,
                "S_policy": S_policy
            }
            
        else:
            # Original owner housing choice implementation
            n_a = len(a_grid)
            n_H = len(H_grid)
            n_y = len(y_grid)
            n_H_nxt = len(H_nxt_grid)
            
            # Store the raw values for each housing choice and income level
            value_grids = []
            lambda_grids = []
            
            for i_h_nxt in range(n_H_nxt):
                value_grids.append(w_grid)
                lambda_grids.append(w_grid)
                
            # Initialize output arrays
            vlu_dcsn = np.zeros((n_a, n_H, n_y))
            lambda_dcsn = np.zeros((n_a, n_H, n_y))
            H_policy = np.zeros((n_a, n_H, n_y), dtype=int)
            
            # Create resources matrix: (a,h) -> resources
            a_mesh, H_mesh = np.meshgrid(a_grid, H_grid, indexing='ij')
            
            # Solve for each income state
            #$print(y_grid)
            for i_y, y_val in enumerate(shock_grid):
                # Calculate resources for all asset-housing combinations
                resources_liquid = (1 + r) * a_mesh + y_val
                
                # Prepare value grids for current income state
                curr_value_grids = []
                curr_lambda_grids = []
                #print(y_grid)
                for i_h_nxt in range(n_H_nxt):
                    curr_value_grids.append(vlu_cntn[:, i_h_nxt, i_y])
                    curr_lambda_grids.append(lambda_cntn[:, i_h_nxt, i_y])
                
                # Call jitted solver using Fella-style approach
                best_values, best_lambdas, best_indices = housing_choice_solver(
                    resources_liquid, H_grid, H_nxt_grid, 
                    w_grid, curr_value_grids, curr_lambda_grids, 
                    tau, w_grid[0]
                )
                
                # Store results for this income state
                vlu_dcsn[:, :, i_y] = best_values
                lambda_dcsn[:, :, i_y] = best_lambdas
                H_policy[:, :, i_y] = best_indices
            
            return {
                "vlu": vlu_dcsn,
                "lambda": lambda_dcsn,
                "H_policy": H_policy
            }
    
    return operator 