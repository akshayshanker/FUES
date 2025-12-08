####################
# helpers/ue.py (new file)
####################
"""Unified Upper-Envelope (UE) engine registry for the FUES project.

This module factors out the logic that was previously hard-wired in
`helpers/egm_upper_envelope.EGM_UE` into three reusable components:

1.  A *registry* (`register`, `get_engine`) through which concrete UE
    engines are made discoverable.
2.  A small set of *engine wrappers* (FUES, DCEGM, RFC, SIMPLE).  Each
    wrapper cleans the EGM output for one particular algorithm and
    returns a *refined* dict with keys ``m, v, c, a, lambda``.
3.  `fill_interpolated` — common interpolation + λ computation that
    maps a refined grid onto a target evaluation grid.

The goal is to let `EGM_UE` shrink to a thin façade while making it
trivial to plug in new engines (just add a ``@register("NAME")``
wrapper below).
"""
from __future__ import annotations

from typing import Callable, Dict, Any, Protocol, Optional, Tuple
import time
from numba import njit
from consav import upperenvelope 

import numpy as np

# Algorithm imports with error handling
try:
    from dc_smm.fues.fues_v0dev import FUES as fues_v0dev
except ImportError:
    fues_v0dev = None

try:
    from dc_smm.fues.dcegm import dcegm
except ImportError:
    dcegm = None

try:
    from dc_smm.fues.rfc_simple import rfc
except ImportError:
    rfc = None

try:
    from dc_smm.fues.fues import FUES as fues_current
    from dc_smm.fues.helpers import correct_jumps1d
except ImportError:
    fues_current = None
    correct_jumps1d = None

# Common post-processing helpers import
from dc_smm.fues.helpers import interp_as

# ---------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------
_REGISTRY: Dict[str, Callable[..., Dict[str, np.ndarray]]] = {}


def register(name: str):
    """Decorator: ``@register("FUES")`` registers the wrapped engine."""

    def decorator(fn: Callable[..., Dict[str, np.ndarray]]):
        _REGISTRY[name.upper()] = fn
        return fn

    return decorator


def get_engine(name: str | None):
    """Return engine callable or *None* if not registered."""
    if name is None:
        return None
    return _REGISTRY.get(name.upper())


def available() -> list[str]:
    """Names of all registered engines."""
    return list(_REGISTRY)


# ---------------------------------------------------------------------
#  Common post-processing helpers
# ---------------------------------------------------------------------


def EGM_UE(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    v_cntn_hat: np.ndarray,
    kappa_hat: np.ndarray,
    X_cntn: np.ndarray,
    X_dcsn: Optional[np.ndarray],
    uc_func_partial: Callable,
    u_func: Callable,
    ue_method: str = "FUES",
    m_bar: float = 1.0,
    lb: int = 4,
    rfc_radius: float = 0.75,
    rfc_n_iter: int = 20,
    interpolate: bool = False,
    include_intersections: bool = True,
    ue_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Universal entry point for all upper-envelope algorithms.

    This is now a *thin* façade: it forwards to a concrete engine
    registered in ``helpers.ue`` and performs common bookkeeping
    (raw dict assembly, interpolation onto ``X_dcsn``, timing).
    The public signature is kept intact for backward compatibility.
    
    Parameters
    ----------
    ue_kwargs : dict, optional
        Method-specific keyword arguments passed directly to the engine.
        For FUES, this can include: endog_mbar, padding_mbar, single_intersection,
        no_double_jumps, disable_jump_checks, eps_d, eps_sep, etc.
    """

    if X_dcsn is None:
        raise ValueError("X_dcsn must be provided for interpolation")

    # -------- raw (always reported) -----------------------------------
    raw = {"x_dcsn_hat": x_dcsn_hat, "qf_hat": qf_hat, "kappa_hat": kappa_hat, "X_cntn": X_cntn}

    # -------- select engine ------------------------------------------
    engine = get_engine(ue_method)
    if engine is None:
        raise ValueError(
            f"Unknown UE method '{ue_method}'. Available: {', '.join(available())}"
        )

    # -------- run -----------------------------------------------------
    t0 = time.time()
    
    # Merge method-specific kwargs
    engine_kwargs = {
        "x_dcsn_hat": x_dcsn_hat,
        "qf_hat": qf_hat,
        "kappa_hat": kappa_hat,
        "X_cntn": X_cntn,
        "v_cntn_hat": v_cntn_hat,
        "X_dcsn": X_dcsn,
        "uc_func_partial": uc_func_partial,
        "u_func": u_func,
        "m_bar": m_bar,
        "lb": lb,
        "rfc_radius": rfc_radius,
        "rfc_n_iter": rfc_n_iter,
        "include_intersections": include_intersections,
    }
    if ue_kwargs:
        engine_kwargs.update(ue_kwargs)

    refined = engine(**engine_kwargs)

    ue_time = time.time() - t0

    # -------- interpolation -----------------------------------------
    if interpolate:
        interpolated = fill_interpolated(refined, X_dcsn, uc_func_partial)
    else:
        interpolated = {}


    interpolated["ue_time"] = ue_time
    refined["ue_time"] = ue_time

    return refined, raw, interpolated



def fill_interpolated(
    refined: Dict[str, np.ndarray],
    X_dcsn: np.ndarray,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, np.ndarray]:
    """Interpolate (m,v,kappa,X_cntn) onto ``X_dcsn`` and recompute λ.

    Returns a dict with keys ``m, v, kappa, X_cntn, lambda`` (mirroring *refined*)
    plus does *not* time anything.
    """

    if refined is None or len(refined.get("m", [])) < 2:
        # Not enough points – return zeros so caller can decide what to do
        out = {k: np.zeros_like(X_dcsn) for k in ("m", "kappa", "v", "X_cntn", "lambda")}
        out["m"] = X_dcsn
        return out

    m_ref = refined["m"]
    out = {
        "m": X_dcsn,
        "kappa": interp_as(m_ref, refined["kappa"], X_dcsn, extrap=True),
        "v": interp_as(m_ref, refined["v"], X_dcsn, extrap=True),
        "X_cntn": interp_as(m_ref, refined["X_cntn"], X_dcsn, extrap=True),
    }
    out["lambda"] = uc_func_partial(out["kappa"])
    return out


# ---------------------------------------------------------------------
#  Engine implementations
# ---------------------------------------------------------------------


@register("FUES_V0DEV")
def _fues_v0dev_engine(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    kappa_hat: np.ndarray,
    X_cntn: np.ndarray,
    *,
    v_cntn: Optional[np.ndarray] = None,
    X_dcsn: Optional[np.ndarray] = None,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    m_bar: float = 1.0,
    lb: int = 4,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Wrapper around original FUES (v0dev) implementation.

    Accepts the minimal subset of keyword args required by the FUES
    signature; surplus kwargs are ignored so `EGM_UE` can forward its
    entire **kwargs without filtering.
    """

    if fues_v0dev is None:
        raise ImportError("FUES v0dev algorithm not importable")

    # Guard against lb being a list (edge-case seen in original code)
    lb_int = int(lb[0]) if isinstance(lb, (list, tuple)) else int(lb)

    x_dcsn_ref, qf_ref, kappa_ref, x_cntn_ref, _ = fues_v0dev(
        x_dcsn_hat, qf_hat, kappa_hat, X_cntn, X_cntn, m_bar=m_bar, LB=lb_int
    )

    return {
        "x_dcsn_ref": x_dcsn_ref,
        "v_dcsn_ref": qf_ref,
        "kappa_ref": kappa_ref,
        "x_cntn_ref": x_cntn_ref,
        "lambda_ref": uc_func_partial(kappa_ref),
    }



@register("DCEGM")
def _dcegm_engine(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    kappa_hat: np.ndarray,
    X_cntn: np.ndarray,
    *,
    v_cntn: Optional[np.ndarray] = None,
    X_dcsn: Optional[np.ndarray] = None,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Wrapper around Iskhakov-et-al DCEGM."""

    if dcegm is None:
        raise ImportError("DCEGM algorithm not importable")

    # Sort by X_cntn (next-period assets) to preserve original EGM iteration order
    # DCEGM detects backward-bending segments in x_dcsn_hat (endogenous grid)
    if not np.all(np.diff(X_cntn) >= 0):
        idx = np.argsort(X_cntn)
        x_dcsn_hat = x_dcsn_hat[idx]
        qf_hat = qf_hat[idx]
        kappa_hat = kappa_hat[idx]
        X_cntn = X_cntn[idx]

    x_cntn_ref, x_dcsn_ref, kappa_ref, qf_ref, _ = dcegm(kappa_hat, kappa_hat, qf_hat, X_cntn, x_dcsn_hat)

    return {
        "x_dcsn_ref": x_dcsn_ref,
        "v_dcsn_ref": qf_ref,
        "kappa_ref": kappa_ref,
        "x_cntn_ref": x_cntn_ref,
        "lambda_ref": uc_func_partial(kappa_ref),
    }


@register("RFC")
def _rfc_engine(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    kappa_hat: np.ndarray,
    X_cntn: np.ndarray,
    *,
    v_cntn: Optional[np.ndarray] = None,
    X_dcsn: Optional[np.ndarray] = None,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    m_bar: float = 1.0,
    rfc_radius: float = 0.75,
    rfc_n_iter: int = 20,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Fast RFC wrapper (1-D case)."""

    if rfc is None:
        raise ImportError("RFC algorithm not importable")

    lambda_egm = uc_func_partial(kappa_hat)

    xr = np.array([x_dcsn_hat]).T
    qfr = np.array([qf_hat]).T
    gradr = np.array([lambda_egm]).T
    pr = np.array([X_cntn]).T

    sub_points, _, _ = rfc(xr, gradr, qfr, pr, m_bar, rfc_radius, rfc_n_iter)

    mask = np.ones(len(x_dcsn_hat), dtype=bool)
    if len(sub_points):  # noqa: WPS505 (explicit)
        mask[sub_points] = False

    return {
        "x_dcsn_ref": x_dcsn_hat[mask],
        "v_dcsn_ref": qf_hat[mask],
        "kappa_ref": kappa_hat[mask],
        "x_cntn_ref": X_cntn[mask],
        "lambda_ref": lambda_egm[mask],
    }




@register("FUES")
def _fues_engine(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    kappa_hat: np.ndarray,
    X_cntn: np.ndarray,
    *,
    v_cntn: Optional[np.ndarray] = None,
    X_dcsn: Optional[np.ndarray] = None,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    m_bar: float = 1.0,
    lb: int = 4,
    include_intersections: bool = True,
    # FUES-specific kwargs (forwarded from ue_kwargs)
    endog_mbar: bool = False,
    padding_mbar: float = 0.0,
    single_intersection: bool = False,
    no_double_jumps: bool = True,
    disable_jump_checks: bool = False,
    return_intersections_separately: bool = False,
    left_turn_no_jump_strict: bool = False,
    use_post_state_jump_test: bool = False,
    detect_decreasing_policy: bool = False,
    post_clean_double_jumps: bool = True,
    post_clean_passes: int = 2,
    jump_check_tol: float = 0.0,
    eps_d: Optional[float] = None,
    eps_sep: Optional[float] = None,
    eps_fwd_back: Optional[float] = None,
    parallel_guard: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Wrapper around current FUES implementation.

    The optimized production version with pre-allocated scratch buffers,
    true circular buffer, and improved left turn interpolation.
    Uses the same interface as the original FUES implementation.
    
    FUES-specific parameters (via ue_kwargs):
        endog_mbar: Use endogenous m_bar based on policy gradients
        padding_mbar: Padding for m_bar calculation
        single_intersection: Only compute single intersection per jump
        no_double_jumps: Filter out consecutive jumps
        disable_jump_checks: Disable manual jump check overrides
        left_turn_no_jump_strict: Treat left turns without jumps same as left turns with jumps
        use_post_state_jump_test: Also use post-state gradient for jump detection
        detect_decreasing_policy: Treat decreasing policy (del_pol < 0) as a jump (independent of other options)
        post_clean_double_jumps: Post-process to remove points with double jumps on both sides
        post_clean_passes: Number of cleaning passes for double-jump removal (default: 2)
        jump_check_tol: Tolerance for value gradient check in forward scan (default: 0.0)
        eps_d, eps_sep, eps_fwd_back, parallel_guard: Numerical tolerances
    """

    if fues_current is None or correct_jumps1d is None:
        raise ImportError("FUES algorithm not importable")

    # Guard against lb being a list (edge-case seen in original code)
    lb_int = int(lb[0]) if isinstance(lb, (list, tuple)) else int(lb)

    # Build kwargs for fues_current, only include non-None values
    fues_kwargs = {
        "m_bar": m_bar,
        "LB": lb_int,
        "include_intersections": include_intersections,
        "endog_mbar": endog_mbar,
        "padding_mbar": padding_mbar,
        "single_intersection": single_intersection,
        "no_double_jumps": no_double_jumps,
        "disable_jump_checks": disable_jump_checks,
        "return_intersections_separately": return_intersections_separately,
        "left_turn_no_jump_strict": left_turn_no_jump_strict,
        "use_post_state_jump_test": use_post_state_jump_test,
        "detect_decreasing_policy": detect_decreasing_policy,
        "post_clean_double_jumps": post_clean_double_jumps,
        "post_clean_passes": post_clean_passes,
        "jump_check_tol": jump_check_tol,
    }
    # Add optional numerical tolerances if specified
    if eps_d is not None:
        fues_kwargs["eps_d"] = eps_d
    if eps_sep is not None:
        fues_kwargs["eps_sep"] = eps_sep
    if eps_fwd_back is not None:
        fues_kwargs["eps_fwd_back"] = eps_fwd_back
    if parallel_guard is not None:
        fues_kwargs["parallel_guard"] = parallel_guard

    x_dcsn_ref, qf_ref, kappa_ref, x_cntn_ref, _ = fues_current(
        x_dcsn_hat, qf_hat, kappa_hat, X_cntn, X_cntn,
        **fues_kwargs
    )

    return {
        "x_dcsn_ref": x_dcsn_ref,
        "v_dcsn_ref": qf_ref,
        "kappa_ref": kappa_ref,
        "x_cntn_ref": x_cntn_ref,
        "lambda_ref": uc_func_partial(kappa_ref),
    }


# ------------------------------------------------------------------
#  SIMPLE monotonicity-enforcing fallback
# ------------------------------------------------------------------


def _simple_upper_envelope(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    kappa_hat: np.ndarray,
    X_cntn: np.ndarray,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, np.ndarray]:
    """Very light monotonic-filter (identical to old implementation)."""

    # sort by x if needed
    if not np.all(np.diff(x_dcsn_hat) > 0):
        idx = np.argsort(x_dcsn_hat)
        x_dcsn_hat, qf_hat, kappa, X_cntn = (
            x_dcsn_hat[idx],
            qf_hat[idx],
            kappa[idx],
            X_cntn[idx],
        )

    # already monotone?
    if np.all(np.diff(kappa) >= 0):
        lam = uc_func_partial(kappa)
        return {
            "m": x_dcsn_hat,
            "v": qf_hat,
            "kappa_ref": kappa_hat,
            "X_cntn": X_cntn,
            "lambda": lam,
        }

    # keep only strictly increasing c segments
    mask = np.append(True, np.diff(kappa) > 0)
    lam = uc_func_partial(kappa[mask])
    return {
        "x_dcsn_ref": x_dcsn_hat[mask],
        "v_dcsn_ref": qf_hat[mask],
        "kappa_ref": kappa_hat[mask],
        "x_cntn_ref": X_cntn[mask],
        "lambda_ref": lam,
    }


@register("SIMPLE")
def _simple_engine(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    kappa: np.ndarray,
    X_cntn: np.ndarray,
    *,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    return _simple_upper_envelope(x_dcsn_hat, qf_hat, kappa, X_cntn, uc_func_partial)


@register("CONSAV")
def _consav_engine(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    kappa_hat: np.ndarray,
    X_cntn: np.ndarray,
    v_cntn_hat: np.ndarray,
    X_dcsn: np.ndarray,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    u_func: Dict[str, Any],
    use_inv_w: bool = False,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Vectorised Consav upper-envelope.

    Parameters
    ----------
    x_dcsn_hat, qf_hat, kappa, X_cntn : raw EGM outputs
    X_dcsn : evaluation grid (Nm, strictly increasing)
    uc_func_partial : marginal utility λ(kappa)
    u_func : dict with keys ``func`` (njitted utility) and ``args``.
    use_inv_w : see Consav documentation
    """

    # cache compiled kernel to avoid recompilation cost
    key = (u_func["func"].py_func.__code__.co_code, use_inv_w)
    if "_CONSAV_CACHE" not in globals():
        globals()["_CONSAV_CACHE"] = {}
        globals()["_CONSAV_CACHE_STATS"] = {"hits": 0, "misses": 0}
    
    env_cache = globals()["_CONSAV_CACHE"]
    cache_stats = globals()["_CONSAV_CACHE_STATS"]
    env = env_cache.get(key)
    if env is None:
        cache_stats["misses"] += 1
        # print(f"[CONSAV] Cache MISS #{cache_stats['misses']} - compiling new upper envelope")
        env = upperenvelope.create(u_func["func"], use_inv_w)
        env_cache[key] = env
    else:
        cache_stats["hits"] += 1
        # Uncomment to debug: print(f"[CONSAV] Cache HIT #{cache_stats['hits']}")

    # allocate outputs
    kappa_pol = np.empty_like(X_dcsn)
    v_dcsn = np.empty_like(X_dcsn)

    # Convert the dictionary of arguments into a tuple of values.
    # Handle both dict and single value cases for backward compatibility
    if isinstance(u_func["args"], dict):
        args_as_tuple = tuple(u_func["args"].values())
    else:
        # Single value case (backward compatibility)
        args_as_tuple = (u_func["args"],)

    # Call the compiled Numba kernel, unpacking the arguments correctly.
    env(X_cntn, x_dcsn_hat, kappa_hat, v_cntn_hat, X_dcsn, kappa_pol, v_dcsn, *args_as_tuple)

    X_cntn_pol = np.maximum(X_dcsn - kappa_pol, 0.0)
    lambda_dcsn = uc_func_partial(kappa_pol)

    return {"x_dcsn_ref": X_dcsn,
            "v_dcsn_ref": v_dcsn,
            "kappa_ref": kappa_pol,
            "x_cntn_ref": X_cntn_pol,
            "lambda_ref": lambda_dcsn}


# ------------------------------------------------------------------
#  End of file
# ------------------------------------------------------------------