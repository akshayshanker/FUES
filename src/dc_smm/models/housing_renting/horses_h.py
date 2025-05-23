import numpy as np
from numba import njit
from dc_smm.models.housing_renting.horses_common import interp_as

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
def housing_choice_solver_owner(resource_grid, h_grid, h_next_grid, w_grid, Q_cntn, v_cntn, lambda_cntn, tau, min_wealth):
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
    """
    # Get dimensions
    n_a, n_h = resource_grid.shape
    n_h_next = len(h_next_grid)
    
    # Output arrays
    best_Q = np.full((n_a, n_h), -np.inf)
    best_lambda = np.zeros((n_a, n_h))
    best_indices = np.zeros((n_a, n_h), dtype=np.int32)
    best_v = np.zeros((n_a, n_h))
    
    # Loop over all states
    for i_a in range(n_a):
        for i_h in range(n_h):
            # Current resource and housing values
            resource = resource_grid[i_a, i_h]
            h_current = h_grid[i_h]
            
            best_lambda[i_a, i_h] = 0
            best_v[i_a, i_h]      = -np.inf
            best_indices[i_a, i_h]= 0

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
                #v_vals = Q_cntn[:,i_h_next]
                #l_vals = lambda_grids[:,i_h_next]

                maximand = interp_as(w_grid, Q_cntn[:,i_h_next], np.array([w_dscn_val]))[0]

                # Update best choice if this is better
                if maximand > best_Q[i_a, i_h]:
                    best_Q[i_a, i_h] = maximand
                    best_lambda[i_a, i_h] = interp_as(w_grid, lambda_cntn[:,i_h_next], np.array([w_dscn_val]))[0]
                    best_v[i_a, i_h] = interp_as(w_grid, v_cntn[:,i_h_next], np.array([w_dscn_val]))[0]
                    best_indices[i_a, i_h] = i_h_next
    
    return best_Q, best_v, best_lambda, best_indices 

@njit
def housing_choice_solver_renter(w_grid, S_grid, y_grid, w_rent_grid, q_cntn, vlu_cntn, lambda_cntn, Pr, shock_grid):
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
        Post-decision wealth grid for interpolation
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
        (q_dcsn, vlu_dcsn, lambda_dcsn, S_policy) arrays
    """
    # Get dimensions
    n_w = len(w_grid)
    n_S = len(S_grid)
    n_y = len(y_grid)
    
    # Initialize output arrays
    vlu_dcsn = np.zeros((n_w, n_y))
    q_dcsn = np.zeros((n_w, n_y))
    lambda_dcsn = np.zeros((n_w, n_y))
    S_policy = np.zeros((n_w, n_y), dtype=np.int32)
    
    # Solve for each income state
    for i_y in range(n_y):
        y_val = shock_grid[i_y]
        
        # For each wealth level
        for i_w in range(n_w):
            w_dcsn_val = w_grid[i_w]
            
            # Initialize best values
            best_q = -np.inf
            best_lambda = 0
            best_S_idx = 0 
            best_v = -np.inf
            
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
                maximand = interp_as(w_rent_grid, q_cntn[:, i_S, i_y], np.array([w_cntn_val]))[0]
                
                # Update best choice if this is better
                if maximand > best_q:
                    best_q = maximand
                    best_lambda = interp_as(w_rent_grid, lambda_cntn[:, i_S, i_y], np.array([w_cntn_val]))[0]
                    best_v = interp_as(w_rent_grid, vlu_cntn[:, i_S, i_y], np.array([w_cntn_val]))[0]
                    best_S_idx = i_S
            
            # Store results
            q_dcsn[i_w, i_y] = best_q
            lambda_dcsn[i_w, i_y] = best_lambda
            S_policy[i_w, i_y] = best_S_idx
            vlu_dcsn[i_w, i_y] = best_v
    
    return q_dcsn, vlu_dcsn, lambda_dcsn, S_policy

def F_h_cntn_to_dcsn_owner(mover):
    """Create operator for housing choice (for both owners and renters).
    
    Implements discrete housing choice through vectorized enumeration.
    Detects whether it's being called for owner or renter based on stage name.
    Maximises present-biased payoff `Q_dcsn`; forwards lifetime `vlu`.
    
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
        Q_cntn = perch_data["Q"]   # objective
        

        # Original owner housing choice implementation
        n_a = len(a_grid)
        n_H = len(H_grid)
        n_y = len(y_grid)
        n_H_nxt = len(H_nxt_grid)
        
        # Initialize output arrays
        vlu_dcsn = np.zeros((n_a, n_H, n_y))
        Q_dcsn = np.zeros((n_a, n_H, n_y))
        lambda_dcsn = np.zeros((n_a, n_H, n_y))
        H_policy = np.zeros((n_a, n_H, n_y), dtype=int)
        
        # Create resources matrix: (a,h) -> resources
        a_mesh, H_mesh = np.meshgrid(a_grid, H_grid, indexing='ij')
        
        # Solve for each income state
        for i_y, y_val in enumerate(shock_grid):
            # Calculate resources for all asset-housing combinations
            resources_liquid = (1 + r) * a_mesh + y_val
            
            Q_dcsn_sl, v_dcsn_sl, lambda_dcsn_sl, idx_sl = housing_choice_solver_owner(
                resources_liquid, H_grid, H_nxt_grid, 
                w_grid, Q_cntn[:,:,i_y],vlu_cntn[:,:,i_y], lambda_cntn[:,:,i_y], 
                tau, w_grid[0]
            )
            
            # Store results for this income state
            lambda_dcsn[:, :, i_y] = lambda_dcsn_sl
            H_policy[:, :, i_y] = idx_sl
            Q_dcsn[:, :, i_y] = Q_dcsn_sl
            vlu_dcsn[:, :, i_y] = v_dcsn_sl
            
        return {
            "Q": Q_dcsn,
            "vlu": vlu_dcsn,
            "lambda": lambda_dcsn,
            "H_policy": H_policy
        }
    
    return operator

def F_h_cntn_to_dcsn_renter(mover):
    """Create operator for housing choice (for both owners and renters).
    
    Implements discrete housing choice through vectorized enumeration.
    Detects whether it's being called for owner or renter based on stage name.
    Maximises present-biased payoff `Q_dcsn`; forwards lifetime `vlu`.
    
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
    # Renter housing choice grids
    w_grid = model.num.state_space.dcsn.grids.w
    y_grid = model.num.state_space.dcsn.grids.y
    w_rent_grid = model.num.state_space.cntn.grids.w_rent  # Cash-on-hand after rental choice
    S_grid = model.num.state_space.cntn.grids.S  # Rental services grid
    
    # Get parameters
    params = model.param
    Pr = params.Pr  # Rental price per unit

    
    def operator(perch_data):
        vlu_cntn = perch_data["vlu"] 
        lambda_cntn = perch_data["lambda"]
        Q_cntn = perch_data["Q"]   # objective
        
        # Use numba-optimized renter housing choice solver
        Q_dcsn, vlu_dcsn, lambda_dcsn, S_policy = housing_choice_solver_renter(
            w_grid, S_grid, y_grid, w_rent_grid, 
            Q_cntn, vlu_cntn, lambda_cntn, Pr, shock_grid
        )
        
        return {
            "Q": Q_dcsn,
            "vlu": vlu_dcsn,
            "lambda": lambda_dcsn,
            "S_policy": S_policy
        }
        

    
    return operator 