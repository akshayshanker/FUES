"""
Endogenous Grid Method Upper Envelope Module

This module centralizes the EGM upper-envelope logic and 
adds native Consav support with a consistent interface.
"""

import numpy as np
import time
from typing import Dict, Tuple, Callable, Optional, Any
from numba.typed import Dict
from FUES.math_funcs import interp_as, correct_jumps1d


# Try to import optional packages for upper envelope methods
try:
    from .fues import fues.kernels as fues_algorithm
    fues_available = True
except ImportError:
    fues_available = False
    fues_algorithm = None

try:
    from FUES.DCEGM import dcegm
    dcegm_available = True
except ImportError:
    dcegm_available = False
    dcegm = None

try:
    from FUES.RFC_simple import rfc
    rfc_available = True
except ImportError:
    rfc_available = False
    rfc = None

try:
    from ..dynx_runner.consav_ue import upper
    consav_available = True
except ImportError:
    consav_available = False

try:
    from consav import upperenvelope
except ImportError as e:      # let import fail loudly somewhere else if needed
    raise ImportError("Consav package not found; install via "
                      "`pip install consav`") from e



def EGM_UE(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    v_nxt_raw: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    w_grid: Optional[np.ndarray],
    uc_func_partial: Callable,
    u_func: Callable,
    ue_method: str = "FUES",
    m_bar: float = 1.2,
    lb: int = 3,
    rfc_radius: float = 0.75,
    rfc_n_iter: int = 20,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute the upper envelope for the Endogenous Grid Method (EGM).
    
    Parameters
    ----------
    x_dcsn_hat : np.ndarray
        Endogenous grid (m) values - decision state value as a function of the
        continuous state variables
    qf_hat : np.ndarray
        Value function values on the endogenous grid
    c : np.ndarray
        Consumption values on the endogenous grid
    a : np.ndarray
        Asset values on the endogenous grid
    w_grid : np.ndarray
        Target wealth grid for interpolation
    uc_func_partial : Callable
        Marginal utility function
    u_func : Callable
        Utility function (required for ConSav method)
    ue_method : str, optional
        Upper envelope method, by default "FUES"
    m_bar : float, optional
        Upper envelope parameter for FUES, by default 1.2
    lb : int, optional
        Look-back parameter for FUES, by default 3
    rfc_radius : float, optional
        Radius parameter for RFC, by default 0.75
    rfc_n_iter : int, optional
        Number of iterations for RFC, by default 20

    
    Returns
    -------
    Tuple[Dict, Dict, Dict]
        Three dictionaries:
        - refined: contains refined grid values (may be empty)
        - raw: contains raw grid values (may be empty)
        - interpolated: contains interpolated values on the target grid
    """
    if w_grid is None:
        raise ValueError("w_grid must be provided for interpolation")
    
    # Initialize empty result dictionaries
    refined = {}
    raw = {}
    interpolated = {
        "m": np.zeros_like(w_grid),
        "c": np.zeros_like(w_grid),
        "v": np.zeros_like(w_grid),
        "a": np.zeros_like(w_grid),
        "lambda": np.zeros_like(w_grid),
        "ue_time": 0.0
    }
    
    # Start timing
    start_time = time.time()
    
    # Upper envelope algorithm selection
    ue_method = ue_method.upper()
    
    if ue_method == "CONSAV":
        if not consav_available:
            raise ImportError("ConSav package not available. Install with: python3 -m pip install consav")
        
        # Use ConSav's upper envelope
        m_new, v_new, c_new, a_new, lambda_new = _consav_upper_envelope(a,
            x_dcsn_hat, c, v_nxt_raw, w_grid,uc_func_partial, u_func["func"], False, u_func["args"]
        )
        
        # Fill the interpolated dictionary
        interpolated["m"] = m_new
        interpolated["c"] = c_new
        interpolated["v"] = v_new
        interpolated["a"] = a_new
        interpolated["lambda"] = lambda_new
        
    elif ue_method == "FUES":
        if not fues_available:
            raise ImportError("FUES not available")
        
        # Convert lb to int to avoid numba type errors
        # Handle the case where lb could be a reference list
        if isinstance(lb, list):
            lb_int = 3  # Default fallback if lb is a list
        else:
            lb_int = int(lb)
            
        # Call FUES with original parameter order
        m_refined, v_refined, c_refined, a_refined, lambda_refined = fues_algorithm(
            x_dcsn_hat, qf_hat, c, a, a, m_bar=m_bar, LB=lb_int
        )
        
        # Fill the refined dictionary
        refined = {
            "m": m_refined,
            "v": v_refined,
            "c": c_refined,
            "a": a_refined,
            "lambda": uc_func_partial(c_refined)  # Calculate lambda from consumption
        }
        
        # Fill the raw dictionary
        raw = {
            "m": x_dcsn_hat,
            "v": qf_hat,
            "c": c,
            "a": a
        }
        interpolated_numba = Dict()
        # Interpolate onto target grid if we have sufficient refined points
        if len(m_refined) >= 2:
            interpolated_numba["m"] = w_grid
            interpolated_numba["c"] = interp_as(m_refined, c_refined, w_grid, extrap=True)
            interpolated_numba["v"] = interp_as(m_refined, v_refined, w_grid, extrap=True)
            interpolated_numba["a"] = interp_as(m_refined, a_refined, w_grid, extrap=True)
            #interpolated["lambda"] = uc_func_partial(interpolated["c"])

        #interpolated["c"], interpolated_new = correct_jumps1d(
        #        interpolated_numba["c"], w_grid, 1.3, interpolated_numba
        #    )
        interpolated["c"] = interpolated_numba["c"]
        interpolated["v"] = interpolated_numba["v"]
        interpolated["a"] = interpolated_numba["a"]
        
        interpolated["lambda"] = uc_func_partial(interpolated["c"])

        
    elif ue_method == "DCEGM":
        if not dcegm_available:
            raise ImportError("DCEGM not available")
        
        # Call DCEGM with original parameter order
        a_refined, m_refined, c_refined, v_refined, _ = dcegm(c, c, qf_hat, a, x_dcsn_hat)
        
        # Calculate lambda (marginal utility) values
        lambda_refined = uc_func_partial(c_refined)
        
        # Fill the refined dictionary
        refined = {
            "m": m_refined,
            "v": v_refined,
            "c": c_refined,
            "a": a_refined,
            "lambda": lambda_refined
        }
        
        # Fill the raw dictionary
        raw = {
            "m": x_dcsn_hat,
            "v": qf_hat,
            "c": c,
            "a": a
        }
        
        # Interpolate onto target grid if we have sufficient refined points
        if len(m_refined) >= 2:
            interpolated["m"] = w_grid
            interpolated["c"] = interp_as(m_refined, c_refined, w_grid)
            interpolated["v"] = interp_as(m_refined, v_refined, w_grid)
            interpolated["a"] = interp_as(m_refined, a_refined, w_grid)
            interpolated["lambda"] = interp_as(m_refined, lambda_refined, w_grid)
    
    elif ue_method == "RFC":
        if not rfc_available:
            raise ImportError("RFC not available")
        
        # Call RFC with original parameter order and format
        lambda_egm = uc_func_partial(c)
        xr = np.array([x_dcsn_hat]).T
        vfr = np.array([qf_hat]).T
        gradr = np.array([lambda_egm]).T
        pr = np.array([a]).T
        
        # Call RFC with original parameter structure
        sub_points, _, _ = rfc(xr, gradr, vfr, pr, m_bar, rfc_radius, rfc_n_iter)
        
        # Process results with mask as in original code
        mask = np.ones(len(x_dcsn_hat), dtype=bool)
        if len(sub_points) > 0:
            mask[sub_points] = False
            
        # Extract refined values using mask
        m_refined = x_dcsn_hat[mask]
        v_refined = qf_hat[mask]
        c_refined = c[mask]
        a_refined = a[mask]
        lambda_refined = lambda_egm[mask]
        
        # Fill the refined dictionary
        refined = {
            "m": m_refined,
            "v": v_refined,
            "c": c_refined,
            "a": a_refined,
            "lambda": lambda_refined
        }
        
        # Fill the raw dictionary
        raw = {
            "m": x_dcsn_hat,
            "v": qf_hat,
            "c": c,
            "a": a
        }
        
        # Interpolate onto target grid if we have sufficient refined points
        if len(m_refined) >= 2:
            interpolated["m"] = w_grid
            interpolated["c"] = _interp_on_grid(m_refined, c_refined, w_grid)
            interpolated["v"] = _interp_on_grid(m_refined, v_refined, w_grid)
            interpolated["a"] = _interp_on_grid(m_refined, a_refined, w_grid)
            interpolated["lambda"] = _interp_on_grid(m_refined, lambda_refined, w_grid)
    
    else:  # Default to SIMPLE
        # Use simple monotonicity-enforcing method
        m_refined, v_refined, c_refined, a_refined, lambda_refined = _simple_upper_envelope(
            x_dcsn_hat, qf_hat, c, a, uc_func_partial
        )
        
        # Fill the refined dictionary
        refined = {
            "m": m_refined,
            "v": v_refined,
            "c": c_refined,
            "a": a_refined,
            "lambda": lambda_refined
        }
        
        # Fill the raw dictionary
        raw = {
            "m": x_dcsn_hat,
            "v": qf_hat,
            "c": c,
            "a": a
        }
        
        # Interpolate onto target grid if we have sufficient refined points
        if len(m_refined) >= 2:
            interpolated["m"] = w_grid
            interpolated["c"] = _interp_on_grid(m_refined, c_refined, w_grid)
            interpolated["v"] = _interp_on_grid(m_refined, v_refined, w_grid)
            interpolated["a"] = _interp_on_grid(m_refined, a_refined, w_grid)
            interpolated["lambda"] = _interp_on_grid(m_refined, lambda_refined, w_grid)
    
    # Record elapsed time
    interpolated["ue_time"] = time.time() - start_time
    
    return refined, raw, interpolated


def _consav_upper_envelope_custom(grid_a,
    m_raw:    np.ndarray,          # endogenous m-grid from EGM (unsorted allowed)
    c_raw:    np.ndarray,          # consumption on m_raw
    v_raw:    np.ndarray,          # value function on m_raw
    w_grid:   np.ndarray,          # target cash-on-hand grid to evaluate on
    uc_func_partial: Callable[[np.ndarray], np.ndarray],  # λ(c) for this H-slice
    u_func:   Callable[[np.ndarray], np.ndarray],         # period utility  u(c)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised Consav upper-envelope helper.

    Returns
    -------
    (m_out, v_out, c_out, a_out, lambda_out)
      m_out == w_grid  (passed through)
      other arrays have same shape as w_grid
    """

 

    # 3.  allocate outputs
    c_out = np.empty_like(w_grid)
    v_out = np.empty_like(w_grid)

    # 4.  call Consav kernel (fully vectorised)
    #     Signature: env(grid_a, m_raw, c_raw, v_raw, m_eval, c_out, v_out, rho)
    #upper(grid_a, m_raw, c_raw, v_raw, w_grid,c_out, v_out)

    # 5.  assets and marginal utility on the evaluation grid
    a_out      = np.maximum(w_grid - c_out, 0.0)
    #print(a_out)
    lambda_out = uc_func_partial(c_out)          # recomputed for safety

    return w_grid, v_out, c_out, a_out,lambda_out

_CONSAV_CACHE: Dict[int, Callable] = {}

def _consav_upper_envelope(
    grid_a: np.ndarray,          # Na  (assets used to build EGM)
    m_vec:  np.ndarray,          # Na  (cash-on-hand after EGM step)
    c_vec:  np.ndarray,          # Na  (consumption at m_vec)
    v_vec:  np.ndarray,          # Na  (already includes period utility!)
    grid_m: np.ndarray,          # Nm  (common m-grid for interpolation)
    uc_func_partial: Callable[[np.ndarray], np.ndarray],  # λ(c)
    u_func: Callable[[float], float],  # njitted one-arg utility
    use_inv_w: bool = False,  # set to True only if v_vec = −1 / inv_w
    *u_args
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run ConSav's upper-envelope kernel and return (m, v, c, a, λ).

    Notes
    -----
    * `u_func` **must be nopython-compiled** (`@njit`) and take **one scalar
      or 1-D array argument `c`**.  Any constants (e.g. `H_val`) must be
      hard-wired beforehand.
    * `v_vec` should already contain period utility; set `use_inv_w=True`
      only if you follow the Consav convention with inverse-w.
    """
    # -------- 1. fetch / compile cached kernel ---------------------------
    key = (u_func.py_func.__code__.co_code, use_inv_w)
    env = _CONSAV_CACHE.get(key)
    if env is None:
        env = upperenvelope.create(u_func, use_inv_w)
        _CONSAV_CACHE[key] = env

    # -------- 2. allocate outputs ---------------------------------------
    c_ast = np.empty_like(grid_m)
    v_ast = np.empty_like(grid_m)

    # -------- 3. run the envelope kernel --------------------------------
    env(grid_a, m_vec, c_vec, v_vec, grid_m, c_ast, v_ast, *u_args)

    # -------- 4. derive assets and λ ------------------------------------
    a_ast = np.maximum(grid_m - c_ast, 0.0)
    lam   = uc_func_partial(c_ast)

    return grid_m, v_ast, c_ast, a_ast, lam




def _simple_upper_envelope(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    uc_func_partial: Callable
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a simple monotonicity-enforcing upper envelope.
    
    Parameters
    ----------
    x_dcsn_hat : np.ndarray
        Endogenous grid (m) values
    qf_hat : np.ndarray
        Value function values on the endogenous grid
    c : np.ndarray
        Consumption values on the endogenous grid
    a : np.ndarray
        Asset values on the endogenous grid
    uc_func_partial : Callable
        Marginal utility function
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Refined m, v, c, a, lambda arrays
    """
    # Sort inputs by x_dcsn_hat if not already sorted
    if not np.all(np.diff(x_dcsn_hat) > 0):
        idx = np.argsort(x_dcsn_hat)
        x_dcsn_hat = x_dcsn_hat[idx]
        qf_hat = qf_hat[idx]
        c = c[idx]
        a = a[idx]
    
    # Check for non-monotonicity in consumption
    if np.all(np.diff(c) >= 0):
        # Already monotonic, just return inputs
        lambda_vals = uc_func_partial(c)
        return x_dcsn_hat, qf_hat, c, a, lambda_vals
    
    # Find strictly increasing segments
    dc = np.diff(c)
    mask = np.append(True, dc > 0)
    
    # Keep only strictly increasing segments
    m_refined = x_dcsn_hat[mask]
    v_refined = qf_hat[mask]
    c_refined = c[mask]
    a_refined = a[mask]
    
    # Calculate marginal utility
    lambda_refined = uc_func_partial(c_refined)
    
    return m_refined, v_refined, c_refined, a_refined, lambda_refined 