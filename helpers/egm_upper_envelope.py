"""
Endogenous Grid Method Upper Envelope Module

This module centralizes the EGM upper-envelope logic and 
adds native Consav support with a consistent interface.
"""

import numpy as np
import time
from typing import Dict, Tuple, Callable, Optional, Any

from helpers.ue import get_engine, fill_interpolated
from helpers import ue as _ue_mod
from FUES.math_funcs import interp_as  # only used by legacy helpers below (if kept)


# Try to import optional packages for upper envelope methods
try:
    from ..housing_renting.fues import FUES as fues_algorithm
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
except ImportError:
    upperenvelope = None



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
    """Universal entry point for all upper-envelope algorithms.

    This is now a *thin* façade: it forwards to a concrete engine
    registered in ``helpers.ue`` and performs common bookkeeping
    (raw dict assembly, interpolation onto ``w_grid``, timing).
    The public signature is kept intact for backward compatibility.
    """

    if w_grid is None:
        raise ValueError("w_grid must be provided for interpolation")

    # -------- raw (always reported) -----------------------------------
    raw = {"m": x_dcsn_hat, "v": qf_hat, "c": c, "a": a}

    # -------- select engine ------------------------------------------
    engine = get_engine(ue_method)
    if engine is None:
        raise ValueError(
            f"Unknown UE method '{ue_method}'. Available: {', '.join(_ue_mod.available())}"
        )

    # -------- run -----------------------------------------------------
    t0 = time.time()

    refined = engine(
        x_dcsn_hat,
        qf_hat,
        c,
        a,
        v_raw=v_nxt_raw,  # consumed only by CONSAV engine
        w_grid=w_grid,
        uc_func_partial=uc_func_partial,
        u_func=u_func,
        m_bar=m_bar,
        lb=lb,
        rfc_radius=rfc_radius,
        rfc_n_iter=rfc_n_iter,
    )

    ue_time = time.time() - t0

    # -------- interpolation -----------------------------------------
    interpolated = fill_interpolated(refined, w_grid, uc_func_partial)
    interpolated["ue_time"] = ue_time

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