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

from typing import Callable, Dict, Any, Protocol
import time

import numpy as np

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
from FUES.math_funcs import interp_as  # noqa: E402  (after np import)


def fill_interpolated(
    refined: Dict[str, np.ndarray],
    w_grid: np.ndarray,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, np.ndarray]:
    """Interpolate (m,v,c,a) onto ``w_grid`` and recompute λ.

    Returns a dict with keys ``m, v, c, a, lambda`` (mirroring *refined*)
    plus does *not* time anything.
    """

    if refined is None or len(refined.get("m", [])) < 2:
        # Not enough points – return zeros so caller can decide what to do
        out = {k: np.zeros_like(w_grid) for k in ("m", "c", "v", "a", "lambda")}
        out["m"] = w_grid
        return out

    m_ref = refined["m"]
    out = {
        "m": w_grid,
        "c": interp_as(m_ref, refined["c"], w_grid, extrap=True),
        "v": interp_as(m_ref, refined["v"], w_grid, extrap=True),
        "a": interp_as(m_ref, refined["a"], w_grid, extrap=True),
    }
    out["lambda"] = uc_func_partial(out["c"])
    return out


# ---------------------------------------------------------------------
#  Engine implementations
# ---------------------------------------------------------------------


@register("FUES")
def _fues_engine(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    *,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    m_bar: float = 1.2,
    lb: int = 3,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Wrapper around `FUES.FUES`.

    Accepts the minimal subset of keyword args required by the FUES
    signature; surplus kwargs are ignored so `EGM_UE` can forward its
    entire **kwargs without filtering.
    """

    try:
        from FUES.FUES import FUES as fues_alg  # noqa: WPS433  (runtime import)
    except ImportError as err:
        raise ImportError("FUES algorithm not importable") from err

    # Guard against lb being a list (edge-case seen in original code)
    lb_int = int(lb[0]) if isinstance(lb, (list, tuple)) else int(lb)

    m_ref, v_ref, c_ref, a_ref, _ = fues_alg(
        x_dcsn_hat, qf_hat, c, a, a, m_bar=m_bar, LB=lb_int
    )

    return {
        "m": m_ref,
        "v": v_ref,
        "c": c_ref,
        "a": a_ref,
        "lambda": uc_func_partial(c_ref),
    }


@register("DCEGM")
def _dcegm_engine(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    *,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Wrapper around Iskhakov-et-al DCEGM."""

    try:
        from FUES.DCEGM import dcegm  # noqa: WPS433
    except ImportError as err:
        raise ImportError("DCEGM algorithm not importable") from err

    a_ref, m_ref, c_ref, v_ref, _ = dcegm(c, c, qf_hat, a, x_dcsn_hat)

    return {
        "m": m_ref,
        "v": v_ref,
        "c": c_ref,
        "a": a_ref,
        "lambda": uc_func_partial(c_ref),
    }


@register("RFC")
def _rfc_engine(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    *,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    m_bar: float = 1.2,
    rfc_radius: float = 0.75,
    rfc_n_iter: int = 20,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Fast RFC wrapper (1-D case)."""

    try:
        from FUES.RFC_simple import rfc  # noqa: WPS433
    except ImportError as err:
        raise ImportError("RFC algorithm not importable") from err

    lambda_egm = uc_func_partial(c)

    xr = np.array([x_dcsn_hat]).T
    vfr = np.array([qf_hat]).T
    gradr = np.array([lambda_egm]).T
    pr = np.array([a]).T

    sub_points, _, _ = rfc(xr, gradr, vfr, pr, m_bar, rfc_radius, rfc_n_iter)

    mask = np.ones(len(x_dcsn_hat), dtype=bool)
    if len(sub_points):  # noqa: WPS505 (explicit)
        mask[sub_points] = False

    return {
        "m": x_dcsn_hat[mask],
        "v": qf_hat[mask],
        "c": c[mask],
        "a": a[mask],
        "lambda": lambda_egm[mask],
    }


# ------------------------------------------------------------------
#  SIMPLE monotonicity-enforcing fallback
# ------------------------------------------------------------------


def _simple_upper_envelope(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, np.ndarray]:
    """Very light monotonic-filter (identical to old implementation)."""

    # sort by x if needed
    if not np.all(np.diff(x_dcsn_hat) > 0):
        idx = np.argsort(x_dcsn_hat)
        x_dcsn_hat, qf_hat, c, a = (
            x_dcsn_hat[idx],
            qf_hat[idx],
            c[idx],
            a[idx],
        )

    # already monotone?
    if np.all(np.diff(c) >= 0):
        lam = uc_func_partial(c)
        return {
            "m": x_dcsn_hat,
            "v": qf_hat,
            "c": c,
            "a": a,
            "lambda": lam,
        }

    # keep only strictly increasing c segments
    mask = np.append(True, np.diff(c) > 0)
    lam = uc_func_partial(c[mask])
    return {
        "m": x_dcsn_hat[mask],
        "v": qf_hat[mask],
        "c": c[mask],
        "a": a[mask],
        "lambda": lam,
    }


@register("SIMPLE")
def _simple_engine(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    *,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    return _simple_upper_envelope(x_dcsn_hat, qf_hat, c, a, uc_func_partial)


@register("CONSAV")
def _consav_engine(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    c: np.ndarray,
    a: np.ndarray,
    *,
    w_grid: np.ndarray,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    u_func: Dict[str, Any],
    use_inv_w: bool = False,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Vectorised Consav upper-envelope.

    Parameters
    ----------
    x_dcsn_hat, qf_hat, c, a : raw EGM outputs
    w_grid : evaluation grid (Nm, strictly increasing)
    uc_func_partial : marginal utility λ(c)
    u_func : dict with keys ``func`` (njitted utility) and ``args``.
    use_inv_w : see Consav documentation
    """
    try:
        from consav import upperenvelope  # noqa: WPS433
    except ImportError as err:
        raise ImportError("Consav not installed; `pip install consav`. ") from err

    # cache compiled kernel to avoid recompilation cost
    key = (u_func["func"].py_func.__code__.co_code, use_inv_w)
    if "_CONSAV_CACHE" not in globals():
        globals()["_CONSAV_CACHE"] = {}
    env_cache = globals()["_CONSAV_CACHE"]
    env = env_cache.get(key)
    if env is None:
        env = upperenvelope.create(u_func["func"], use_inv_w)
        env_cache[key] = env

    # allocate outputs
    c_ast = np.empty_like(w_grid)
    v_ast = np.empty_like(w_grid)

    env(a, x_dcsn_hat, c, qf_hat, w_grid, c_ast, v_ast, *u_func["args"])

    a_ast = np.maximum(w_grid - c_ast, 0.0)
    lam = uc_func_partial(c_ast)

    return {"m": w_grid, "v": v_ast, "c": c_ast, "a": a_ast, "lambda": lam}


# ------------------------------------------------------------------
#  End of file
# ------------------------------------------------------------------ 