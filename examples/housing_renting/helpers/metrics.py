"""
Deviation metrics for comparing model policies against reference solutions.
"""

from __future__ import annotations
from typing import Any, Callable, Literal, Optional

import numpy as np
from dynx.runner.circuit_runner import CircuitRunner
from dynx.runner.reference_utils import load_reference_model

# ─────────────────────────────────── helpers ────────────────────────────────────
def _extract_policy(
    model: Any,
    stage: str = "OWNC",
    sol_attr: str = "policy",
    key: str = "c",
    *,
    perch_grid_key: str = "dcsn",
    cont_grid_key: str = "w",
    period_idx: int | str | None = 0,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return (policy_array, grid_array) extracted from *model*.

    Both values are ``None`` if they cannot be located.
    """

    # 1. Locate the period ───────────────────────────────────────────────
    if not (hasattr(model, "periods_list") and model.periods_list):
        return None, None
    periods = model.periods_list

    if period_idx is None:
        # search until we find a period containing *stage*
        period_obj = next(
            (p for p in periods if hasattr(p, "get_stage") and
             _safe_call(lambda: p.get_stage(stage))),   # type: ignore[arg-type]
            None,
        )
    else:
        if period_idx == "first":
            period_idx = 0
        try:
            period_obj = (model.get_period(period_idx)        # type: ignore[attr-defined]
                          if hasattr(model, "get_period")
                          else periods[period_idx])
        except (IndexError, KeyError):
            return None, None

    if period_obj is None or not hasattr(period_obj, "get_stage"):
        return None, None

    # 2. Locate the stage and perch ──────────────────────────────────────
    try:
        stage_obj = period_obj.get_stage(stage)
    except (AttributeError, KeyError):
        return None, None

    perch_obj = getattr(stage_obj, perch_grid_key, None)
    if perch_obj is None:
        return None, None

    # 3. Extract the policy  ─────────────────────────────────────────────
    pol = None
    if hasattr(perch_obj, "sol"):
        sol_obj = perch_obj.sol
        pol = _pull_from_solution(sol_obj, sol_attr, key)
    if pol is None:                      # last‑ditch: array on perch itself
        try:
            pol = np.asarray(getattr(perch_obj, key))
        except AttributeError:
            pol = None

    # 4. Extract the grid   ──────────────────────────────────────────────
    cont_grid = None
    if hasattr(perch_obj, "grid"):
        gproxy = perch_obj.grid
        if hasattr(gproxy, cont_grid_key):
            cont_grid = np.asarray(getattr(gproxy, cont_grid_key))

    return pol, cont_grid


def _pull_from_solution(sol_obj: Any, sol_attr: str, key: str) -> Optional[np.ndarray]:
    """Helper: pull ndarray from *sol_obj* or return None."""
    if sol_obj is None:
        return None
    if hasattr(sol_obj, sol_attr):
        pol_obj = getattr(sol_obj, sol_attr)
    elif isinstance(sol_obj, dict) and sol_attr in sol_obj:
        pol_obj = sol_obj[sol_attr]
    else:
        return None

    if key in ("", None):
        return pol_obj if isinstance(pol_obj, np.ndarray) else None
    if hasattr(pol_obj, key):
        return np.asarray(getattr(pol_obj, key))
    if isinstance(pol_obj, dict) and key in pol_obj:
        return np.asarray(pol_obj[key])
    return None


def _safe_call(fn):
    """Call fn() and return result, or None on *any* exception."""
    try:
        return fn()
    except Exception:
        return None


# ─────────────────────────────── metric factory ────────────────────────────────
def make_policy_dev_metric(
    policy_attr: str,
    norm: Literal["L2", "Linf"],
    *,
    stage: str = "OWNC",
    sol_attr: str = "policy",
    perch_grid_key: str = "dcsn",
    cont_grid_key: str = "w",
    interp_axis: Optional[int] = None,
) -> Callable[[Any, CircuitRunner, np.ndarray], float]:
    """
    Build a deviation metric with optional explicit interpolation axis.

    * If *interp_axis* is ``None`` (default), the single axis whose lengths
      differ is used automatically.
    * No fall‑backs: if grids are missing or shapes mismatch in >1 dimension,
      the function returns ``np.nan``.
    """

    def metric(
        model: Any, *, _runner: Optional[CircuitRunner] = None, _x: Optional[np.ndarray] = None
    ) -> float:
        if _runner is None or _x is None:
            return np.nan

        ref_model = load_reference_model(_runner, _x)
        if ref_model is None:
            return np.nan

        pol,   g_mod = _extract_policy(
            model,
            stage, sol_attr, policy_attr,
            perch_grid_key=perch_grid_key,
            cont_grid_key=cont_grid_key,
        )
        refp, g_ref = _extract_policy(
            ref_model,
            stage, sol_attr, policy_attr,
            perch_grid_key=perch_grid_key,
            cont_grid_key=cont_grid_key,
        )
        if pol is None or refp is None or g_mod is None or g_ref is None:
            return np.nan

        # ── interpolation step ─────────────────────────────────────────
        if pol.shape != refp.shape:
            if len(pol.shape) != len(refp.shape):
                return np.nan

            # determine axis
            if interp_axis is not None:
                ax = interp_axis
                # ensure this is the ONLY mismatching axis
                other_axes_equal = all(
                    a == b if i != ax else True
                    for i, (a, b) in enumerate(zip(pol.shape, refp.shape))
                )
                if not other_axes_equal or pol.shape[ax] == refp.shape[ax]:
                    return np.nan
            else:
                diff_axes = [i for i, (a, b) in enumerate(zip(pol.shape, refp.shape)) if a != b]
                if len(diff_axes) != 1:
                    return np.nan
                ax = diff_axes[0]

            if g_mod.size != pol.shape[ax] or g_ref.size != refp.shape[ax]:
                return np.nan  # grid lengths and array lengths inconsistent

            # reshape‑and‑interp using np.interp
            ref_swapped = np.moveaxis(refp, ax, -1)        # (..., n_old)
            lead = ref_swapped.reshape(-1, g_ref.size)     # (m, n_old)
            out  = np.empty((lead.shape[0], g_mod.size), dtype=refp.dtype)
            for i, row in enumerate(lead):
                out[i] = np.interp(g_mod, g_ref, row)
            refp = np.moveaxis(out.reshape(*ref_swapped.shape[:-1], g_mod.size), -1, ax)

            if pol.shape != refp.shape:
                return np.nan

        # ── compute deviation ──────────────────────────────────────────
        diff = pol - refp
        ord_ = 2 if norm == "L2" else np.inf
        return float(np.linalg.norm(diff.ravel(), ord=ord_))

    metric.__name__ = f"dev_{policy_attr}_{norm}"
    metric.__doc__  = (
        f"{norm} deviation of '{policy_attr}' policy from reference (1-D interp on axis "
        f'"{interp_axis if interp_axis is not None else "auto"}")'
    )
    return metric


# ─────────────────────────── concrete convenience metrics ──────────────────────
dev_c_L2   = make_policy_dev_metric("c",   "L2")
dev_c_Linf = make_policy_dev_metric("c",   "Linf")
dev_a_L2   = make_policy_dev_metric("a",   "L2")
dev_a_Linf = make_policy_dev_metric("a",   "Linf")
dev_v_L2   = make_policy_dev_metric("v",   "L2",  sol_attr="value")
dev_v_Linf = make_policy_dev_metric("v",   "Linf", sol_attr="value")
dev_pol_L2   = make_policy_dev_metric("pol", "L2")
dev_pol_Linf = make_policy_dev_metric("pol", "Linf")
