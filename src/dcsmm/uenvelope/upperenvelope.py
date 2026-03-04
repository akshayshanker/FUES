"""Upper-envelope engine registry for FUES.

Factors the logic previously hard-wired in the EGM solver into:

1.  A *registry* (`register`, `get_engine`) through which concrete UE
    engines are made discoverable.
2.  Engine wrappers (FUES, DCEGM, RFC, SIMPLE, CONSAV).  Each wrapper
    normalises one algorithm's output into a common dict with keys
    ``x_dcsn_ref, v_dcsn_ref, kappa_ref, x_cntn_ref, lambda_ref``.
3.  `fill_interpolated` — common interpolation + lambda computation that
    maps a refined grid onto a target evaluation grid.
"""
from __future__ import annotations

from typing import Callable, Dict, Any, Optional
import time

# Load consav.upperenvelope directly from file, bypassing consav/__init__.py
# which has a hard (undeclared) dependency on EconModel.
try:
    import importlib.util as _ilu, os as _os
    _consav_spec = _ilu.find_spec("consav")
    if _consav_spec is None or _consav_spec.submodule_search_locations is None:
        raise ImportError
    _ue_path = _os.path.join(_consav_spec.submodule_search_locations[0], "upperenvelope.py")
    _ue_spec = _ilu.spec_from_file_location("consav.upperenvelope", _ue_path)
    _consav_ue = _ilu.module_from_spec(_ue_spec)
    _ue_spec.loader.exec_module(_consav_ue)
except (ImportError, FileNotFoundError, AttributeError):
    _consav_ue = None

import numpy as np

# Algorithm imports with error handling
try:
    from dcsmm.fues.fues_v0dev import FUES as fues_v0dev
except ImportError:
    fues_v0dev = None

try:
    from dcsmm.fues.dcegm import dcegm
except ImportError:
    dcegm = None

try:
    from dcsmm.fues.rfc_simple import rfc
except ImportError:
    rfc = None

try:
    from dcsmm.fues.fues import FUES as fues_current
except ImportError:
    fues_current = None

# Common post-processing helpers import
from dcsmm.fues.helpers import interp_as

# ---------------------------------------------------------------------
# Registry
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
#  Facade
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
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Universal entry point for all upper-envelope algorithms.

    Dispatches to a registered engine, times execution, and optionally
    interpolates the refined result onto ``X_dcsn``.
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

    # -------- build kwargs: ue_kwargs first, explicit params override --
    all_kwargs = dict(ue_kwargs or {})
    all_kwargs.update(
        m_bar=m_bar,
        lb=lb,
        rfc_radius=rfc_radius,
        rfc_n_iter=rfc_n_iter,
        include_intersections=include_intersections,
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
        **all_kwargs,
    )

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
    """Interpolate refined result onto ``X_dcsn`` and recompute lambda.

    Returns dict with keys ``x_dcsn_ref, v_dcsn_ref, kappa_ref,
    x_cntn_ref, lambda_ref``.
    """

    if refined is None or len(refined.get("x_dcsn_ref", [])) < 2:
        out = {k: np.zeros_like(X_dcsn)
               for k in ("x_dcsn_ref", "kappa_ref", "v_dcsn_ref", "x_cntn_ref", "lambda_ref")}
        out["x_dcsn_ref"] = X_dcsn
        return out

    m_ref = refined["x_dcsn_ref"]
    out = {
        "x_dcsn_ref": X_dcsn,
        "kappa_ref": interp_as(m_ref, refined["kappa_ref"], X_dcsn, extrap=True),
        "v_dcsn_ref": interp_as(m_ref, refined["v_dcsn_ref"], X_dcsn, extrap=True),
        "x_cntn_ref": interp_as(m_ref, refined["x_cntn_ref"], X_dcsn, extrap=True),
    }
    out["lambda_ref"] = uc_func_partial(out["kappa_ref"])
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
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    m_bar: float = 1.0,
    lb: int = 4,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Wrapper around original FUES (v0dev) implementation."""

    if fues_v0dev is None:
        raise ImportError("FUES v0dev algorithm not importable")

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
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Wrapper around Iskhakov-et-al DCEGM."""

    if dcegm is None:
        raise ImportError("DCEGM algorithm not importable")

    # NOTE: Do NOT sort arrays before passing to DCEGM.
    # DCEGM's calc_nondecreasing_segments expects the original EGM iteration order
    # (sorted by exogenous grid a'). Pre-sorting by X_cntn or x_dcsn_hat breaks
    # segment detection and creates spurious multiple segments.

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
    if len(sub_points):
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
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    m_bar: float = 1.0,
    lb: int = 4,
    include_intersections: bool = True,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Wrapper around current FUES implementation.

    Additional FUES-specific kwargs (passed via ue_kwargs) include:
        endog_mbar, padding_mbar, single_intersection, no_double_jumps,
        disable_jump_checks, eps_d, eps_sep, eps_fwd_back, parallel_guard
    """

    if fues_current is None:
        raise ImportError("FUES algorithm not importable")

    lb_int = int(lb[0]) if isinstance(lb, (list, tuple)) else int(lb)

    fues_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ('endog_mbar', 'padding_mbar', 'single_intersection',
                 'no_double_jumps', 'disable_jump_checks',
                 'return_intersections_separately', 'assume_sorted',
                 'eps_d', 'eps_sep', 'eps_fwd_back', 'parallel_guard')
    }

    x_dcsn_ref, qf_ref, kappa_ref, x_cntn_ref, _ = fues_current(
        x_dcsn_hat, qf_hat, kappa_hat, X_cntn, X_cntn,
        m_bar=m_bar, LB=lb_int, include_intersections=include_intersections,
        **fues_kwargs
    )

    return {
        "x_dcsn_ref": x_dcsn_ref,
        "v_dcsn_ref": qf_ref,
        "kappa_ref": kappa_ref,
        "x_cntn_ref": x_cntn_ref,
        "lambda_ref": uc_func_partial(kappa_ref),
    }


@register("SIMPLE")
def _simple_engine(
    x_dcsn_hat: np.ndarray,
    qf_hat: np.ndarray,
    kappa_hat: np.ndarray,
    X_cntn: np.ndarray,
    *,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Light monotonic-filter fallback."""

    # sort by x if needed
    if not np.all(np.diff(x_dcsn_hat) > 0):
        idx = np.argsort(x_dcsn_hat)
        x_dcsn_hat = x_dcsn_hat[idx]
        qf_hat = qf_hat[idx]
        kappa_hat = kappa_hat[idx]
        X_cntn = X_cntn[idx]

    # keep only strictly increasing c segments
    mask = np.append(True, np.diff(kappa_hat) > 0)

    return {
        "x_dcsn_ref": x_dcsn_hat[mask],
        "v_dcsn_ref": qf_hat[mask],
        "kappa_ref": kappa_hat[mask],
        "x_cntn_ref": X_cntn[mask],
        "lambda_ref": uc_func_partial(kappa_hat[mask]),
    }


# ------------------------------------------------------------------
#  CONSAV engine
# ------------------------------------------------------------------

_CONSAV_CACHE: Dict[tuple, Any] = {}


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
    x_dcsn_hat, qf_hat, kappa_hat, X_cntn : raw EGM outputs
    X_dcsn : evaluation grid (Nm, strictly increasing)
    uc_func_partial : marginal utility lambda(kappa)
    u_func : dict with keys ``func`` (njitted utility) and ``args``.
    use_inv_w : see Consav documentation
    """

    key = (u_func["func"].py_func.__code__.co_code, use_inv_w)
    env = _CONSAV_CACHE.get(key)
    if env is None:
        env = _consav_ue.create(u_func["func"], use_inv_w)
        _CONSAV_CACHE[key] = env

    kappa_pol = np.empty_like(X_dcsn)
    v_dcsn = np.empty_like(X_dcsn)

    if isinstance(u_func["args"], dict):
        args_as_tuple = tuple(u_func["args"].values())
    else:
        args_as_tuple = (u_func["args"],)

    env(X_cntn, x_dcsn_hat, kappa_hat, v_cntn_hat, X_dcsn, kappa_pol, v_dcsn, *args_as_tuple)

    X_cntn_pol = np.maximum(X_dcsn - kappa_pol, 0.0)
    lambda_dcsn = uc_func_partial(kappa_pol)

    return {
        "x_dcsn_ref": X_dcsn,
        "v_dcsn_ref": v_dcsn,
        "kappa_ref": kappa_pol,
        "x_cntn_ref": X_cntn_pol,
        "lambda_ref": lambda_dcsn,
    }
