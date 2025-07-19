"""
Deviation metrics for comparing model policies against reference solutions.
"""

from __future__ import annotations
from typing import Any, Callable, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
import itertools
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
        print("Extracting policy from model")
        pol,   g_mod = _extract_policy(
            model,
            stage, sol_attr, policy_attr,
            perch_grid_key=perch_grid_key,
            cont_grid_key=cont_grid_key,
        )
        print("Extracting policy from baseline")
        refp, g_ref = _extract_policy(
            ref_model,
            stage, sol_attr, policy_attr,
            perch_grid_key=perch_grid_key,
            cont_grid_key=cont_grid_key,
        )
        print("Policy extracted from model")
        if pol is None or refp is None or g_mod is None or g_ref is None:
            print("Policy not extracted from model or baseline")
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
                    print("Grid lengths and array lengths inconsistent1")
                    return np.nan
            else:
                diff_axes = [i for i, (a, b) in enumerate(zip(pol.shape, refp.shape)) if a != b]
                if len(diff_axes) != 1:
                    return np.nan
                ax = diff_axes[0]

            if g_mod.size != pol.shape[ax] or g_ref.size != refp.shape[ax]:
                print("Grid lengths and array lengths inconsistent -- axes used is ", ax)
                print("Model grid length: ", g_mod.size)
                print("Model array length: ", pol.shape[ax])
                print("Baseline grid length: ", g_ref.size)
                print("Baseline array length: ", refp.shape[ax])
                return np.nan  # grid lengths and array lengths inconsistent

            # reshape‑and‑interp using np.interp
            ref_swapped = np.moveaxis(refp, ax, -1)        # (..., n_old)
            lead = ref_swapped.reshape(-1, g_ref.size)     # (m, n_old)
            out  = np.empty((lead.shape[0], g_mod.size), dtype=refp.dtype)
            for i, row in enumerate(lead):
                out[i] = np.interp(g_mod, g_ref, row)
            refp = np.moveaxis(out.reshape(*ref_swapped.shape[:-1], g_mod.size), -1, ax)

            if pol.shape != refp.shape:
                print("Policy shapes inconsistent")
                return np.nan

        # ── compute deviation ──────────────────────────────────────────
        diff = pol - refp
        ord_ = 2 if norm == "L2" else np.inf
        return float(np.linalg.norm(diff.ravel(), ord=ord_))/(len(pol)**(1/ord_))

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


# ─────────────────────────── plotting comparison factory ──────────────────────
def plot_comparison_factory(
    decision_variable: str, 
    dim_labels: dict, 
    plot_axis_label: str, 
    slice_config: dict = None,
    stage: str = "OWNC",
    sol_attr: str = None,
    period_idx: int = 0
):
    """
    A factory that creates a configurable metric function for plotting comparisons.

    Args:
        decision_variable (str): The model attribute to plot (e.g., 'v', 'c').
        dim_labels (dict): Maps every dimension index to a label (e.g., {0: 'k_idx', 1: 'a_idx'}).
        plot_axis_label (str): The label for the plot's x-axis (e.g., 'a_idx').
        slice_config (dict, optional): Maps a dimension label to a list of specific
                                       indices to plot. E.g., {'k_idx': [5, 15]}.
                                       If None, a plot for every index is generated.
        stage (str): The stage to extract data from (default: "OWNC").
        sol_attr (str): The solution attribute ('policy' or 'value'). If None, 
                        automatically determined based on decision_variable.
        period_idx (int): The period index to extract data from (default: 0).
    """
    def _plotter(model, _runner, _x):
        if not hasattr(_runner, 'ref_model_for_plotting'):
            return np.nan

        baseline_model = _runner.ref_model_for_plotting
        img_dir = _runner.output_root / "images"
        img_dir.mkdir(exist_ok=True)
        params_dict = _runner.unpack(_x)
        method_name = params_dict.get("master.methods.upper_envelope", "unknown_method")

        try:
            # Extract data using the existing _extract_policy function for consistency
            # Determine the correct sol_attr if not provided
            _sol_attr = sol_attr if sol_attr is not None else ("policy" if decision_variable in ("c", "a", "h", "pol") else "value")
            
            fast_data, fast_grid = _extract_policy(
                model, 
                key=decision_variable, 
                sol_attr=_sol_attr,
                stage=stage,
                period_idx=period_idx
            )
            baseline_data, baseline_grid = _extract_policy(
                baseline_model, 
                key=decision_variable, 
                sol_attr=_sol_attr,
                stage=stage,
                period_idx=period_idx
            )
            
            if fast_data is None or baseline_data is None:
                print(f"[warn] Could not extract {decision_variable} data for plotting")
                return np.nan
            
            # Handle choice dimension by taking max if it exists
            if fast_data.ndim > 2 and str(dim_labels.get(fast_data.ndim - 1, '')).lower() == 'choice':
                fast_data = fast_data.max(axis=-1)
                baseline_data = baseline_data.max(axis=-1)
            
            diff_data = fast_data - baseline_data

            # Find the plot axis index
            plot_axis_index = -1
            slice_dims = {}
            for index, label in dim_labels.items():
                if label == plot_axis_label:
                    plot_axis_index = index
                elif label.lower() != 'choice' and index < diff_data.ndim:
                    slice_dims[index] = label

            if plot_axis_index == -1:
                raise ValueError(f"plot_axis_label '{plot_axis_label}' not found in dim_labels.")

            slice_indices = sorted(slice_dims.keys())
            
            # Determine which slices to plot
            if slice_config:
                slice_ranges = []
                for i in slice_indices:
                    dim_label = dim_labels[i]
                    if dim_label in slice_config:
                        slice_ranges.append(slice_config[dim_label])
                    else:
                        slice_ranges.append(range(diff_data.shape[i]))
            else:
                slice_ranges = [range(diff_data.shape[i]) for i in slice_indices]

            # Generate plots for each slice combination
            for slice_vals in itertools.product(*slice_ranges):
                slicer = [slice(None)] * diff_data.ndim
                for i, val in enumerate(slice_vals):
                    slicer[slice_indices[i]] = val
                
                plot_data = diff_data[tuple(slicer)]

                label_parts = [f"{slice_dims[idx]}={val}" for idx, val in zip(slice_indices, slice_vals)]
                title = f"{decision_variable.title()} Diff: {method_name} vs. Baseline\n({', '.join(label_parts)})"
                filename = f"{decision_variable}_diff_{method_name}_{'_'.join(label_parts)}.png"

                fig, ax = plt.subplots()
                ax.set_title(title)
                ax.set_xlabel(plot_axis_label)
                ax.set_ylabel("Difference")
                ax.plot(plot_data)
                ax.grid(True)
                
                plt.savefig(img_dir / filename)
                plt.close(fig)

        except Exception as err:
            import traceback
            print(f"[warn] Plotting for '{decision_variable}' failed on '{method_name}': {err}\n{traceback.format_exc()}")

        return np.nan

    _plotter.__name__ = f"plot_{decision_variable}_comparison"
    _plotter.__doc__ = (
        f"Plot comparison of '{decision_variable}' between fast method and baseline reference.\n"
        f"Extracts data from stage '{stage}' using sol_attr '{sol_attr or "auto"}' at period {period_idx}."
    )
    return _plotter
