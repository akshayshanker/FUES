"""Upper-envelope engine registry for FUES.

Factors the logic previously hard-wired in the EGM solver into:

1.  A *registry* (`register`, `get_engine`) through which concrete UE
    engines are made discoverable.
2.  Engine wrappers (FUES, DCEGM, RFC, SIMPLE, CONSAV).  Each wrapper
    normalises one algorithm's output into a common dict with keys
    ``x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, lambda_ref``.
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

try:
    from dcsmm.fues.fues_v0_1dev import FUES as fues_v0_1dev
except ImportError:
    fues_v0_1dev = None

try:
    from dcsmm.fues.fues_v0_2dev import FUES as fues_v0_2dev
except ImportError:
    fues_v0_2dev = None

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
    v_hat: np.ndarray,
    v_cntn_hat: np.ndarray,
    kappa_hat: np.ndarray,
    x_cntn_hat: np.ndarray,
    X_dcsn: Optional[np.ndarray],
    uc_func_partial: Callable,
    u_func: Callable,
    method_switch: str | None = None,
    m_bar: float = 1.0,
    lb: int = 4,
    rfc_radius: float = 0.75,
    rfc_n_iter: int = 20,
    interpolate: bool = False,
    include_intersections: bool = True,
    ue_kwargs: Optional[Dict[str, Any]] = None,
    ue_method: str | None = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Universal entry point for all upper-envelope algorithms.

    Takes the unrefined EGM correspondence, dispatches to a registered
    upper-envelope engine, times the execution, and optionally
    interpolates the refined result onto the target decision grid
    ``X_dcsn``.

    Parameters
    ----------
    x_dcsn_hat : ndarray, shape (N,)
        Unrefined endogenous decision grid produced by the EGM
        inversion step (``hat{x}``).
    v_hat : ndarray, shape (N,)
        Unrefined value correspondence aligned with ``x_dcsn_hat``
        (``hat{v}``).
    v_cntn_hat : ndarray, shape (N,)
        Continuation value at each point on the exogenous grid.
        Used by engines that need the raw continuation value
        (e.g., CONSAV).
    kappa_hat : ndarray, shape (N,)
        Unrefined primary control aligned with ``x_dcsn_hat``
        (e.g., consumption ``hat{c}``).
    x_cntn_hat : ndarray, shape (N,)
        Unrefined continuation / exogenous grid aligned with
        ``x_dcsn_hat`` (e.g., next-period assets ``hat{x}'``).
    X_dcsn : ndarray, shape (M,)
        Target decision grid onto which refined policies are
        interpolated.  Must be strictly increasing.
    uc_func_partial : callable
        Marginal-utility function ``u'(c)`` used to compute
        ``lambda_ref`` from the refined consumption policy.
    u_func : callable or dict
        Utility function (or ``{"func": njit_u, "args": ...}``
        dict for engines like CONSAV that need it directly).
    method_switch : str, optional
        Name of the registered upper-envelope engine to use. Defaults
        to ``"FUES"`` when both ``method_switch`` and ``ue_method`` are
        omitted. Available engines: FUES, FUES_V0DEV, FUES_V0_1DEV,
        FUES_V0_2DEV, DCEGM, RFC, SIMPLE, CONSAV.
    ue_method : str, optional
        Deprecated alias for ``method_switch``; do not pass both.
    m_bar : float, default 1.0
        Jump-detection threshold passed to FUES / RFC engines.
    lb : int, default 4
        Look-back / look-forward buffer length for FUES engines.
    rfc_radius : float, default 0.75
        Radius parameter for the RFC engine.
    rfc_n_iter : int, default 20
        Iteration count for the RFC engine.
    interpolate : bool, default False
        If True, interpolate the refined result onto ``X_dcsn``
        and return the interpolated dict in the third element.
    include_intersections : bool, default True
        If True (and the engine supports it), create explicit
        intersection points at discrete-choice switches.
    ue_kwargs : dict, optional
        Additional keyword arguments forwarded to the engine
        (e.g., ``endog_mbar``, ``padding_mbar``,
        ``single_intersection``).

    Returns
    -------
    refined : dict
        Upper-envelope output with keys:

        - ``x_dcsn_ref`` — refined decision grid
        - ``v_dcsn_ref`` — refined value on the decision grid
        - ``kappa_ref``  — refined primary control
        - ``x_cntn_ref`` — refined continuation grid
        - ``lambda_ref`` — marginal utility at ``kappa_ref``
        - ``ue_time``    — wall-clock seconds for the UE step

    raw : dict
        Unrefined inputs passed through for diagnostics:
        ``{x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat}``.

    interpolated : dict
        If ``interpolate=True``, same keys as *refined* but
        evaluated on ``X_dcsn``.  Always contains ``ue_time``.
    """

    if X_dcsn is None:
        raise ValueError("X_dcsn must be provided for interpolation")

    if method_switch is not None and ue_method is not None and method_switch != ue_method:
        raise ValueError("Pass only one of method_switch, ue_method")
    algo = method_switch if method_switch is not None else ue_method
    if algo is None:
        algo = "FUES"

    # -------- raw (always reported) -----------------------------------
    raw = {"x_dcsn_hat": x_dcsn_hat, "v_hat": v_hat, "kappa_hat": kappa_hat, "x_cntn_hat": x_cntn_hat}

    # -------- select engine ------------------------------------------
    engine = get_engine(algo)
    if engine is None:
        raise ValueError(
            f"Unknown UE method '{algo}'. Available: {', '.join(available())}"
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
        v_hat=v_hat,
        kappa_hat=kappa_hat,
        x_cntn_hat=x_cntn_hat,
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
    v_hat: np.ndarray,
    kappa_hat: np.ndarray,
    x_cntn_hat: np.ndarray,
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

    x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, _ = fues_v0dev(
        x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat, x_cntn_hat, m_bar=m_bar, LB=lb_int
    )

    return {
        "x_dcsn_ref": x_dcsn_ref,
        "v_dcsn_ref": v_ref,
        "kappa_ref": kappa_ref,
        "x_cntn_ref": x_cntn_ref,
        "lambda_ref": uc_func_partial(kappa_ref),
    }


@register("DCEGM")
@register("MSS")
def _dcegm_engine(
    x_dcsn_hat: np.ndarray,
    v_hat: np.ndarray,
    kappa_hat: np.ndarray,
    x_cntn_hat: np.ndarray,
    *,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Wrapper around Iskhakov-et-al DCEGM."""

    if dcegm is None:
        raise ImportError("DCEGM algorithm not importable")

    # NOTE: Do NOT sort arrays before passing to DCEGM.
    # DCEGM's calc_nondecreasing_segments expects the original EGM iteration order
    # (sorted by exogenous grid a'). Pre-sorting by x_cntn_hat or x_dcsn_hat breaks
    # segment detection and creates spurious multiple segments.

    x_cntn_ref, x_dcsn_ref, kappa_ref, v_ref, _ = dcegm(kappa_hat, kappa_hat, v_hat, x_cntn_hat, x_dcsn_hat)

    return {
        "x_dcsn_ref": x_dcsn_ref,
        "v_dcsn_ref": v_ref,
        "kappa_ref": kappa_ref,
        "x_cntn_ref": x_cntn_ref,
        "lambda_ref": uc_func_partial(kappa_ref),
    }


@register("RFC")
def _rfc_engine(
    x_dcsn_hat: np.ndarray,
    v_hat: np.ndarray,
    kappa_hat: np.ndarray,
    x_cntn_hat: np.ndarray,
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
    qfr = np.array([v_hat]).T
    gradr = np.array([lambda_egm]).T
    pr = np.array([x_cntn_hat]).T

    sub_points, _, _ = rfc(xr, gradr, qfr, pr, m_bar, rfc_radius, rfc_n_iter)

    mask = np.ones(len(x_dcsn_hat), dtype=bool)
    if len(sub_points):
        mask[sub_points] = False

    return {
        "x_dcsn_ref": x_dcsn_hat[mask],
        "v_dcsn_ref": v_hat[mask],
        "kappa_ref": kappa_hat[mask],
        "x_cntn_ref": x_cntn_hat[mask],
        "lambda_ref": lambda_egm[mask],
    }


@register("FUES")
def _fues_engine(
    x_dcsn_hat: np.ndarray,
    v_hat: np.ndarray,
    kappa_hat: np.ndarray,
    x_cntn_hat: np.ndarray,
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

    x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, _ = fues_current(
        x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat,
        m_bar=m_bar, LB=lb_int, include_intersections=include_intersections,
        **fues_kwargs
    )

    return {
        "x_dcsn_ref": x_dcsn_ref,
        "v_dcsn_ref": v_ref,
        "kappa_ref": kappa_ref,
        "x_cntn_ref": x_cntn_ref,
        "lambda_ref": uc_func_partial(kappa_ref),
    }


@register("FUES_V0_1DEV")
def _fues_v0_1dev_engine(
    x_dcsn_hat: np.ndarray,
    v_hat: np.ndarray,
    kappa_hat: np.ndarray,
    x_cntn_hat: np.ndarray,
    *,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    m_bar: float = 1.0,
    lb: int = 4,
    include_intersections: bool = True,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Wrapper around FUES v0.1dev (release-prep baseline)."""

    if fues_v0_1dev is None:
        raise ImportError("FUES v0.1dev not importable")

    lb_int = int(lb[0]) if isinstance(lb, (list, tuple)) else int(lb)

    fues_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ('endog_mbar', 'padding_mbar', 'single_intersection',
                 'no_double_jumps', 'disable_jump_checks',
                 'return_intersections_separately',
                 'eps_d', 'eps_sep', 'eps_fwd_back', 'parallel_guard')
    }

    x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, _ = fues_v0_1dev(
        x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat, x_cntn_hat,
        m_bar=m_bar, LB=lb_int, include_intersections=include_intersections,
        **fues_kwargs
    )

    return {
        "x_dcsn_ref": x_dcsn_ref,
        "v_dcsn_ref": v_ref,
        "kappa_ref": kappa_ref,
        "x_cntn_ref": x_cntn_ref,
        "lambda_ref": uc_func_partial(kappa_ref),
    }


@register("FUES_V0_2DEV")
def _fues_v0_2dev_engine(
    x_dcsn_hat: np.ndarray,
    v_hat: np.ndarray,
    kappa_hat: np.ndarray,
    x_cntn_hat: np.ndarray,
    *,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    m_bar: float = 1.0,
    lb: int = 4,
    include_intersections: bool = True,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Wrapper around FUES v0.2dev (fues-experiments optimizations)."""

    if fues_v0_2dev is None:
        raise ImportError("FUES v0.2dev not importable")

    lb_int = int(lb[0]) if isinstance(lb, (list, tuple)) else int(lb)

    fues_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ('endog_mbar', 'padding_mbar', 'single_intersection',
                 'no_double_jumps', 'disable_jump_checks',
                 'return_intersections_separately', 'assume_sorted',
                 'eps_d', 'eps_sep', 'eps_fwd_back', 'parallel_guard')
    }

    x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, _ = fues_v0_2dev(
        x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat,
        m_bar=m_bar, LB=lb_int, include_intersections=include_intersections,
        **fues_kwargs
    )

    return {
        "x_dcsn_ref": x_dcsn_ref,
        "v_dcsn_ref": v_ref,
        "kappa_ref": kappa_ref,
        "x_cntn_ref": x_cntn_ref,
        "lambda_ref": uc_func_partial(kappa_ref),
    }


@register("SIMPLE")
def _simple_engine(
    x_dcsn_hat: np.ndarray,
    v_hat: np.ndarray,
    kappa_hat: np.ndarray,
    x_cntn_hat: np.ndarray,
    *,
    uc_func_partial: Callable[[np.ndarray], np.ndarray],
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Light monotonic-filter fallback."""

    # sort by x if needed
    if not np.all(np.diff(x_dcsn_hat) > 0):
        idx = np.argsort(x_dcsn_hat)
        x_dcsn_hat = x_dcsn_hat[idx]
        v_hat = v_hat[idx]
        kappa_hat = kappa_hat[idx]
        x_cntn_hat = x_cntn_hat[idx]

    # keep only strictly increasing c segments
    mask = np.append(True, np.diff(kappa_hat) > 0)

    return {
        "x_dcsn_ref": x_dcsn_hat[mask],
        "v_dcsn_ref": v_hat[mask],
        "kappa_ref": kappa_hat[mask],
        "x_cntn_ref": x_cntn_hat[mask],
        "lambda_ref": uc_func_partial(kappa_hat[mask]),
    }


# ------------------------------------------------------------------
#  CONSAV engine
# ------------------------------------------------------------------

_CONSAV_CACHE: Dict[tuple, Any] = {}


@register("CONSAV")
def _consav_engine(
    x_dcsn_hat: np.ndarray,
    v_hat: np.ndarray,
    kappa_hat: np.ndarray,
    x_cntn_hat: np.ndarray,
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
    x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat : raw EGM outputs
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

    env(x_cntn_hat, x_dcsn_hat, kappa_hat, v_cntn_hat, X_dcsn, kappa_pol, v_dcsn, *args_as_tuple)

    x_cntn_pol = np.maximum(X_dcsn - kappa_pol, 0.0)
    lambda_dcsn = uc_func_partial(kappa_pol)

    return {
        "x_dcsn_ref": X_dcsn,
        "v_dcsn_ref": v_dcsn,
        "kappa_ref": kappa_pol,
        "x_cntn_ref": x_cntn_pol,
        "lambda_ref": lambda_dcsn,
    }
