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
    m_bar: float = 1.2,
    lb: int = 3,
    rfc_radius: float = 0.75,
    rfc_n_iter: int = 20,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Universal entry point for all upper-envelope algorithms.

    This is now a *thin* façade: it forwards to a concrete engine
    registered in ``helpers.ue`` and performs common bookkeeping
    (raw dict assembly, interpolation onto ``X_dcsn``, timing).
    The public signature is kept intact for backward compatibility.
    """

    if X_dcsn is None:
        raise ValueError("X_dcsn must be provided for interpolation")

    # -------- raw (always reported) -----------------------------------
    raw = {"x_dcsn_hat": x_dcsn_hat, "qf_hat": qf_hat, "kappa_hat": kappa_hat, "X_cntn": X_cntn}

    # -------- select engine ------------------------------------------
    engine = get_engine(ue_method)
    if engine is None:
        raise ValueError(
            f"Unknown UE method '{ue_method}'. Available: {', '.join(_ue_mod.available())}"
        )

    # -------- run -----------------------------------------------------
    t0 = time.time()

    refined = engine(
        x_dcsn_hat=x_dcsn_hat,
        qf_hat=qf_hat,
        kappa_hat=kappa_hat,
        X_cntn=X_cntn,
        v_cntn_hat=v_cntn_hat,
        X_dcsn=X_dcsn,
        uc_func_partial=uc_func_partial,
        u_func=u_func,
        m_bar=m_bar,
        lb=lb,
        rfc_radius=rfc_radius,
        rfc_n_iter=rfc_n_iter,
    )

    ue_time = time.time() - t0

    # -------- interpolation -----------------------------------------
    interpolated = fill_interpolated(refined, X_dcsn, uc_func_partial)
    interpolated["ue_time"] = ue_time

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

    x_dcsn_ref, qf_ref, kappa_ref, x_cntn_ref, _ = fues_alg(
        x_dcsn_hat, qf_hat, kappa_hat, X_cntn, X_cntn, m_bar=m_bar, LB=lb_int
    )

    return {
        "x_dcsn_ref": x_dcsn_ref,
        "v_dcsn_ref": qf_ref,
        "kappa_ref": kappa_ref,
        "x_cntn_ref": x_cntn_ref,
        "lambda": uc_func_partial(kappa_ref),
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

    try:
        from FUES.DCEGM import dcegm  # noqa: WPS433
    except ImportError as err:
        raise ImportError("DCEGM algorithm not importable") from err

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
    kappa_pol = np.empty_like(X_dcsn)
    v_dcsn = np.empty_like(X_dcsn)

    env(X_cntn, x_dcsn_hat, kappa_hat, v_cntn_hat, X_dcsn, kappa_pol, v_dcsn, *u_func["args"])

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