"""
Wrapper functions for different upper envelope implementations.

This module provides helper functions to build and reuse different
upper envelope implementations, focusing on avoiding redundant compilations.
"""

import numpy as np
from typing import Callable, Any

# Attempt to import the consav upperenvelope module
try:
    from consav import upperenvelope
    consav_available = True
except ImportError:
    consav_available = False
    print("ConSav package not available. Install with: pip install consav")


def build_consav_envelope(u_func: Callable) -> Callable:
    """
    Build a ConSav upper envelope function that can be reused.
    
    This function compiles the ConSav envelope once via upperenvelope.create(u_func)
    and returns that callable, avoiding repeated compilation overhead.
    
    The ConSav upper envelope takes an endogenous grid (m_vec) that may be non-monotonic
    and produces a consumption policy and value function on a specified target grid (grid_m).
    
    Important distinction:
    - m_vec: The endogenous grid of cash-on-hand values generated during EGM.
             This grid may NOT be monotonic, which is why upper envelope is needed.
    - grid_m: The target grid where we want the consumption policy and value function.
              This is typically a fixed, evenly spaced grid that we interpolate onto.
    
    Parameters
    ----------
    u_func : callable
        Utility function that accepts consumption as its first argument
        
    Returns
    -------
    callable
        The compiled ConSav upper envelope function ready to use.
        The returned function has the signature:
        envelope(grid_a, m_vec, c_vec, v_vec, grid_m, c_ast_vec, v_ast_vec, *args)
        
        Where:
        - grid_a: Asset grid
        - m_vec: Cash-on-hand values from EGM (potentially non-monotonic)
        - c_vec: Consumption values from EGM
        - v_vec: Value function values from EGM 
        - grid_m: Target cash-on-hand grid for interpolation
        - c_ast_vec: Output array for optimal consumption on grid_m
        - v_ast_vec: Output array for optimal values on grid_m
        - *args: Additional arguments passed to the utility function
        
    Raises
    ------
    ImportError
        If the consav package is not installed
    """
    if not consav_available:
        raise ImportError(
            "ConSav package not available. Install with: pip install consav"
        )
    
    # Create the upper envelope function with the provided utility function
    envelope = upperenvelope.create(u_func)
    
    return envelope 