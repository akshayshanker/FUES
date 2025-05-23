import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

def F_t_cntn_to_dcsn(mover):
    """Create operator for tenure choice (TENU) continuation to decision.
    
    Implements the max operation between owner and renter paths:
    V_v(a, H, y) = max{V_e^{rent}((1+r)a+H, y), V_e^{own}(a, y, H)}
    
    Maximises present-biased payoff `Q`; forwards lifetime `vlu`.
    
    Parameters
    ----------
    mover : Mover
        The cntn_to_dcsn mover with self-contained model
        
    Returns
    -------
    callable
        The operator function that transforms continuation perch data to decision perch data
    """
    # Extract model
    model = mover.model
    
    # Get parameters
    r = model.param.r  # Interest rate
    
    # Get grids from model
    a_grid = model.num.state_space.dcsn.grids.a
    H_grid = model.num.state_space.dcsn.grids.H
    y_grid = model.num.state_space.dcsn.grids.y
    
    # Create meshgrid for vectorized operations
    n_a = len(a_grid)
    n_H = len(H_grid)
    n_y = len(y_grid)
    
    # Get rent grid for interpolation
    w_cntn_rent_grid = model.num.state_space.cntn_rent.grids.w
    
    # Precompute the transformed states for renter path
    # Create meshgrid of (a, H) for broadcasting
    a_mesh, H_mesh = np.meshgrid(a_grid, H_grid, indexing='ij')
    
    # Calculate liquid wealth for rent path: (1+r)a + H for all combinations
    w_rent_mesh = (1 + r) * a_mesh + H_mesh
    
    def operator(perch_data):
        """Transform continuation data into decision data using vectorized operations
        
        Parameters
        ----------
        perch_data : dict
            Dictionary with 'from_owner' and 'from_renter' keys containing the 
            continuation values from both paths
            
        Returns
        -------
        dict
            Decision perch data with value function, marginal value, and tenure policy
        """
        # Extract continuation values for owner and renter paths
        # Use the same keys as in whisperer.py
        own_data = perch_data["from_owner"]
        rent_data = perch_data["from_renter"]
        
        # Get values for the owner path (already on correct grid)
        vlu_own = own_data["vlu"]     # Shape: (n_a, n_H, n_y)
        lambda_own = own_data["lambda"]  # Shape: (n_a, n_H, n_y)
        Qlu_own = own_data["Q"]      # NEW
        
        # Get rent path data
        vlu_rent_raw = rent_data["vlu"]       # Shape: (n_w, n_y)
        lambda_rent_raw = rent_data["lambda"]  # Shape: (n_w, n_y)
        Qlu_rent_raw = rent_data["Q"]   # NEW
        
        # Initialize result arrays
        vlu_dcsn = np.zeros((n_a, n_H, n_y))
        lambda_dcsn = np.zeros((n_a, n_H, n_y))
        policy_tenu = np.zeros((n_a, n_H, n_y))
        
        # Process each income state separately (vectorizing over a and H)
        for i_y in range(n_y):
            # Create interpolation functions for this income state
            vlu_rent_interp = interp1d(
                w_cntn_rent_grid, vlu_rent_raw[:, i_y], 
                bounds_error=False, fill_value="extrapolate"
            )
            lambda_rent_interp = interp1d(
                w_cntn_rent_grid, lambda_rent_raw[:, i_y], 
                bounds_error=False, fill_value="extrapolate"
            )
            Qlu_rent_interp = interp1d(
                w_cntn_rent_grid, Qlu_rent_raw[:, i_y],
                bounds_error=False, fill_value="extrapolate"
            )
            
            # Vectorized interpolation for all (a, H) points at this income level
            # Reshape w_rent_mesh to 1D for interpolation
            w_rent_flat = w_rent_mesh.flatten()
            
            # Apply interpolation
            vlu_rent_flat = vlu_rent_interp(w_rent_flat)
            lambda_rent_flat = lambda_rent_interp(w_rent_flat)
            Qlu_rent_flat = Qlu_rent_interp(w_rent_flat)
            
            # Reshape back to (n_a, n_H)
            vlu_rent_2d = vlu_rent_flat.reshape(n_a, n_H)
            lambda_rent_2d = lambda_rent_flat.reshape(n_a, n_H)
            Qlu_rent_2d = Qlu_rent_flat.reshape(n_a, n_H)
            
            # Extract owner values for this income state
            vlu_own_2d = vlu_own[:, :, i_y]
            lambda_own_2d = lambda_own[:, :, i_y]
            Qlu_own_2d = Qlu_own[:, :, i_y]
            
            # Compare values and create masks for owner and renter paths
            own_is_better = Qlu_own_2d > Qlu_rent_2d   # present-biased rule
            
            # Use masks to set policy
            policy_tenu[:, :, i_y] = own_is_better.astype(float)
            
            # Use masks to select values
            vlu_dcsn[:, :, i_y] = np.where(own_is_better, vlu_own_2d, vlu_rent_2d)
            lambda_dcsn[:, :, i_y] = np.where(own_is_better, lambda_own_2d, lambda_rent_2d)
        
        return {
            "vlu": vlu_dcsn,
            "lambda": lambda_dcsn,
            "tenure_policy": policy_tenu
        }
    
    return operator