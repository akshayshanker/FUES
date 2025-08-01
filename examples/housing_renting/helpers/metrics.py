"""
Deviation metrics for comparing model policies against reference solutions.
"""

from __future__ import annotations
from typing import Any, Callable, Literal, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for speed
import matplotlib.pyplot as plt
import itertools
import gc
from contextlib import contextmanager
from pathlib import Path
from scipy import interpolate
from dynx.runner.circuit_runner import CircuitRunner
from dynx.runner.reference_utils import load_reference_model
try:
    from dynx.runner.metric_requirements import get_metric_requirements
except ImportError:
    # Fallback if metric_requirements module not available
    def get_metric_requirements(metric_names):
        return None, None
try:
    from dynx.runner.reference_cache import get_cached_reference_model
except ImportError:
    # Fallback to regular loading if cache not available
    get_cached_reference_model = load_reference_model

# Import memory utilities for granular logging
try:
    from .memory_utils import log_memory_usage
except ImportError:
    try:
        from memory_utils import log_memory_usage
    except ImportError:
        # Fallback if memory_utils not available
        def log_memory_usage(label="", verbose=True):
            import psutil
            import os
            if verbose:
                process = psutil.Process(os.getpid())
                mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
                print(f"[Memory {label}]: {mem_gb:.2f} GB")


def _safe_array_view(arr, *args, **kwargs):
    """
    Create array view when possible, fall back to copy if needed.
    
    This helper function attempts to create a view of the array with the given
    operations, but falls back to copying if the operation would require it.
    """
    try:
        # Try to create a view first
        if hasattr(arr, 'view'):
            return arr.view(*args, **kwargs)
        else:
            return arr
    except (ValueError, AttributeError):
        # Fall back to copy if view is not possible
        return np.array(arr, *args, **kwargs)


def _optimize_array_layout(arr):
    """
    Ensure array is C-contiguous for optimal memory access patterns.
    
    Returns a view if already C-contiguous, otherwise returns a copy.
    """
    if arr.flags['C_CONTIGUOUS']:
        return arr  # Already optimal, return view
    else:
        return np.ascontiguousarray(arr)  # Make C-contiguous copy


def fast_interp_1d_vectorized(x_new, x_old, y_old_2d):
    """
    Vectorized 1D interpolation for 2D arrays.
    
    Interpolates each row of y_old_2d from x_old to x_new grid.
    Uses scipy's interp1d which is much faster than np.interp in loops.
    """
    # Create interpolator - this is the expensive operation
    f = interpolate.interp1d(x_old, y_old_2d, axis=1, 
                            kind='linear', bounds_error=False, 
                            fill_value='extrapolate', assume_sorted=True)
    # Apply to new grid - this is fast
    return f(x_new)

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


# ─────────────────────────────── memory management ────────────────────────────────
@contextmanager
def managed_model_load(runner, x, metric_name=None):
    """
    Context manager for safe model loading with automatic cleanup.
    
    Args:
        runner: CircuitRunner instance
        x: Parameter vector
        metric_name: Optional name of the metric for selective loading
    """
    model = None
    try:
        # Use superset caching - the cache will load periods 0 and 1 with all required stages
        # This ensures all metrics share the same cached model
        if metric_name:
            print(f"  [Cache] Requesting baseline for {metric_name} (using superset cache: periods 0-1)")
        
        model = get_cached_reference_model(runner, x, metric_requirements=None)
        yield model
    finally:
        if model is not None:
            # Clean up model internals if it has cleanup methods
            if hasattr(model, 'cleanup'):
                model.cleanup()
            del model
            gc.collect()


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
    Build a deviation metric that computes L2 or L∞ norm differences between policy functions
    from fast solution methods and high-density baseline solutions.

    This function creates metrics that ensure accurate like-for-like comparisons by handling
    different grid sizes through interpolation. When comparing solutions with different
    discretizations (e.g., 100 vs 1000 grid points), the metric automatically interpolates
    the reference solution onto the test solution's grid.

    Interpolation Logic:
        1. Extracts policy functions and their corresponding grids from both models
        2. Identifies the axis where grid sizes differ (typically the state-space dimension)
        3. Interpolates the reference (baseline) policy onto the test method's grid using np.interp
        4. Computes element-wise differences on the common grid
        5. Calculates the requested norm (L2 or L∞) of the difference vector
        6. Normalizes by the appropriate power of the array length for comparable metrics

    This interpolation ensures that:
        - Comparisons are made at equivalent economic states
        - Metrics are independent of the specific grid discretization chosen
        - Results are consistent between plotting and numerical error metrics

    Args:
        policy_attr (str): Name of the policy attribute to compare (e.g., 'c', 'a', 'v').
        norm (Literal["L2", "Linf"]): Type of norm to compute ("L2" for Euclidean, "Linf" for max).
        stage (str): Stage name to extract data from (default: "OWNC").
        sol_attr (str): Solution attribute type ('policy' or 'value', default: "policy").
        perch_grid_key (str): Key for the perch object (default: "dcsn").
        cont_grid_key (str): Key for the continuous grid (default: "w").
        interp_axis (Optional[int]): Explicit axis for interpolation. If None, automatically
                                     detected as the single axis where lengths differ.

    Returns:
        Callable: A metric function that accepts (model, _runner, _x) and returns the
                  computed deviation as a float. Returns np.nan if extraction or 
                  interpolation fails.

    Example:
        >>> c_l2_metric = make_policy_dev_metric("c", "L2")
        >>> # Use with CircuitRunner: metric_fns={"dev_c_L2": c_l2_metric}
    """

    def metric(
        model: Any, *, _runner: Optional[CircuitRunner] = None, _x: Optional[np.ndarray] = None
    ) -> float:
        if _runner is None or _x is None:
            return np.nan

        # Initialize interpolation variables at function level for cleanup access
        ref_swapped = None
        lead = None
        out = None

        # Determine metric name based on parameters for selective loading
        metric_name = None
        if policy_attr == "c" and norm == "L2":
            metric_name = "dev_c_L2"
        elif policy_attr == "c" and norm == "Linf":
            metric_name = "dev_c_Linf"
        elif policy_attr == "v" and norm == "L2":
            metric_name = "dev_v_L2"
        
        # Use context manager for safe reference model loading
        try:
            with managed_model_load(_runner, _x, metric_name=metric_name) as ref_model:
                if ref_model is None:
                    return np.nan
                
                log_memory_usage("before policy extraction", verbose=False)
                print("Extracting policy from model")
                pol, g_mod = _extract_policy(
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
                log_memory_usage("after policy extraction", verbose=False)
                if pol is None or refp is None or g_mod is None or g_ref is None:
                    print("Policy not extracted from model or baseline")
                    return np.nan

                # ── interpolation step ─────────────────────────────────────────
                try:
                    if pol.shape != refp.shape:
                        log_memory_usage("before interpolation", verbose=False)
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

                        # Use vectorized interpolation for much better performance
                        ref_swapped = np.moveaxis(refp, ax, -1)        # (..., n_old)
                        ref_swapped = _optimize_array_layout(ref_swapped)  # Ensure C-contiguous for optimal access
                        lead = ref_swapped.reshape(-1, g_ref.size, order='C')     # (m, n_old), ensure C-contiguous
                        
                        # Vectorized interpolation - much faster than loop
                        out = fast_interp_1d_vectorized(g_mod, g_ref, lead)
                        
                        refp = np.moveaxis(out.reshape(*ref_swapped.shape[:-1], g_mod.size), -1, ax)
                        log_memory_usage("after interpolation", verbose=False)

                        if pol.shape != refp.shape:
                            print("Policy shapes inconsistent")
                            return np.nan

                    # ── compute deviation ──────────────────────────────────────────
                    diff = pol - refp
                    ord_ = 2 if norm == "L2" else np.inf
                    result = float(np.linalg.norm(diff.ravel(), ord=ord_)) / (len(pol) ** (1 / ord_))
                    
                    # Clean up large arrays before return
                    del diff, pol, refp, g_mod, g_ref
                    # The interpolation arrays (ref_swapped, lead, out) will be cleaned up
                    # automatically by Python's garbage collector
                    gc.collect()
                    log_memory_usage("after metric cleanup", verbose=False)
                    
                    return result
                    
                except Exception:
                    # Re-raise the exception - cleanup will happen automatically
                    raise
                        
        except Exception as e:
            print(f"Error in metric computation: {e}")
            gc.collect()
            return np.nan

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
    A factory that creates a configurable metric function for plotting comparisons between
    fast solution methods and high-density baseline solutions.

    This function generates professional error plots that show the difference between policy
    or value functions computed by different solution methods. The plots use proper interpolation
    to ensure accurate like-for-like comparisons on common grids, following the same methodology
    as the L2 deviation metrics.

    Interpolation and Grid Handling:
        When fast methods and baseline use different grid sizes (e.g., 100 vs 1000 points),
        the function automatically:
        1. Extracts both policy functions and their corresponding grids
        2. Identifies which dimension differs between the two solutions
        3. Interpolates the baseline policy onto the fast method's grid using np.interp
        4. Calculates differences on the common grid for accurate error measurement
        5. Plots against actual economic quantities (wealth, housing) not just indices

    This ensures that comparison plots are scientifically accurate and consistent with
    how L2 deviation metrics are calculated.

    Args:
        decision_variable (str): The model attribute to plot (e.g., 'vlu', 'c', 'a').
        dim_labels (dict): Maps dimension indices to labels (e.g., {0: 'w_idx', 1: 'h_idx'}).
        plot_axis_label (str): The label for the plot's x-axis (e.g., 'w_idx' for wealth).
        slice_config (dict, optional): Maps dimension labels to lists of specific indices
                                       to plot. E.g., {'h_idx': [5, 10, 15]}. If None, 
                                       plots are generated for all indices.
        stage (str): The stage to extract data from (default: "OWNC").
        sol_attr (str): The solution attribute ('policy' or 'value'). If None, 
                        automatically determined based on decision_variable.
        period_idx (int): The period index to extract data from (default: 0).

    Returns:
        function: A metric function that can be used with CircuitRunner. The function
                  generates professional error plots saved in the bundle directory
                  structure, showing differences with proper statistics and styling.

    Example:
        >>> plot_metric = plot_comparison_factory(
        ...     decision_variable='c',
        ...     dim_labels={0: 'w_idx', 1: 'h_idx', 2: 'y_idx'},
        ...     plot_axis_label='w_idx',
        ...     slice_config={'h_idx': [5, 10, 15]}
        ... )
        >>> # Use with CircuitRunner metric_fns
    """
    def _plotter(model, _runner, _x):
        # Initialize interpolation variables at function level for cleanup access
        ref_swapped = None
        lead = None
        out = None
        
        # Determine metric name based on decision variable for selective loading
        metric_name = None
        if decision_variable == 'c':
            metric_name = 'plot_c_comparison'
        elif decision_variable in ('vlu', 'v'):
            metric_name = 'plot_v_comparison'
        
        # Use context manager for safe baseline model loading
        try:
            with managed_model_load(_runner, _x, metric_name=metric_name) as baseline_model:
                if baseline_model is None:
                    return np.nan
                
                # Use bundle path if available, otherwise fall back to output_root/images
                bundle_path = _runner._bundle_path(_x)
                if bundle_path and bundle_path.exists():
                    img_dir = bundle_path / "images"
                elif _runner.output_root:
                    img_dir = _runner.output_root / "images"
                else:
                    # Fallback if no output paths configured
                    img_dir = Path("images")
                
                img_dir.mkdir(parents=True, exist_ok=True)
                params_dict = _runner.unpack(_x)
                method_name = params_dict.get("master.methods.upper_envelope", "unknown_method")

                # Extract data using the existing _extract_policy function for consistency
                # Determine the correct sol_attr if not provided
                _sol_attr = sol_attr if sol_attr is not None else ("policy" if decision_variable in ("c", "a", "h", "pol") else "value")
                
                log_memory_usage("before plotting data extraction", verbose=False)
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
                log_memory_usage("after plotting data extraction", verbose=False)
                
                if fast_data is None or baseline_data is None or fast_grid is None or baseline_grid is None:
                    print(f"[warn] Could not extract {decision_variable} data for plotting")
                    return np.nan
                
                try:
                    # Handle choice dimension by taking max if it exists
                    if fast_data.ndim > 2 and str(dim_labels.get(fast_data.ndim - 1, '')).lower() == 'choice':
                        fast_data = _optimize_array_layout(fast_data.max(axis=-1))
                        baseline_data = _optimize_array_layout(baseline_data.max(axis=-1))
                    
                    # ── Interpolation step (same as L2 metric) ─────────────────────────────────────
                    if fast_data.shape != baseline_data.shape:
                        if len(fast_data.shape) != len(baseline_data.shape):
                            print(f"[warn] Incompatible array dimensions for plotting")
                            return np.nan

                        # Find the axis that differs (should correspond to our plot axis)
                        diff_axes = [i for i, (a, b) in enumerate(zip(fast_data.shape, baseline_data.shape)) if a != b]
                        if len(diff_axes) != 1:
                            print(f"[warn] Multiple differing axes, cannot interpolate for plotting")
                            return np.nan
                        
                        interp_axis = diff_axes[0]
                        
                        # Check grid lengths match array dimensions
                        if fast_grid.size != fast_data.shape[interp_axis] or baseline_grid.size != baseline_data.shape[interp_axis]:
                            print(f"[warn] Grid lengths don't match array dimensions for plotting")
                            return np.nan

                        # Use vectorized interpolation for much better performance
                        ref_swapped = np.moveaxis(baseline_data, interp_axis, -1)        # (..., n_old)
                        ref_swapped = _optimize_array_layout(ref_swapped)  # Ensure C-contiguous for optimal access
                        lead = ref_swapped.reshape(-1, baseline_grid.size, order='C')    # (m, n_old), ensure C-contiguous
                        
                        # Vectorized interpolation - much faster than loop
                        out = fast_interp_1d_vectorized(fast_grid, baseline_grid, lead)
                        
                        baseline_data = np.moveaxis(out.reshape(*ref_swapped.shape[:-1], fast_grid.size), -1, interp_axis)

                    # Now both arrays should have the same shape
                    if fast_data.shape != baseline_data.shape:
                        print(f"[warn] Arrays still have different shapes after interpolation")
                        return np.nan
                    
                    # Calculate difference on common grid
                    diff_data = fast_data - baseline_data
                    
                    # Clean up intermediate arrays after interpolation
                    if ref_swapped is not None:
                        del ref_swapped
                    if lead is not None:
                        del lead
                    if out is not None:
                        del out
                    
                    # Use the fast method grid for plotting (since we interpolated onto it)
                    plot_axis_grid = fast_grid

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
                        fig = None
                        try:
                            slicer = [slice(None)] * diff_data.ndim
                            for i, val in enumerate(slice_vals):
                                slicer[slice_indices[i]] = val
                            
                            plot_data = diff_data[tuple(slicer)]

                            label_parts = [f"{slice_dims[idx]}={val}" for idx, val in zip(slice_indices, slice_vals)]
                            title = f"{decision_variable.title()} Error: {method_name} vs. Baseline"
                            subtitle = f"({', '.join(label_parts)})"
                            filename = f"{decision_variable}_error_{method_name}_{'_'.join(label_parts)}.png"

                            # Create professional-looking error plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Determine x-axis values
                            if plot_axis_grid is not None and len(plot_axis_grid) == len(plot_data):
                                x_values = plot_axis_grid
                                x_label = plot_axis_label.replace('_idx', '').replace('_', ' ').title()
                            else:
                                x_values = np.arange(len(plot_data))
                                x_label = f"{plot_axis_label.replace('_idx', '')} Index"
                            
                            # Plot with professional styling
                            ax.plot(x_values, plot_data, 'o-', linewidth=1.5, markersize=4, 
                                   color='#d62728', alpha=0.8, label=f'{method_name} - Baseline')
                            
                            # Add zero line for reference
                            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
                            
                            # Styling
                            ax.set_xlabel(x_label, fontsize=12)
                            ax.set_ylabel(f"{decision_variable.title()} Error", fontsize=12)
                            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                            ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, 
                                   ha='center', va='top', fontsize=10, style='italic')
                            
                            # Grid and spines
                            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_linewidth(0.5)
                            ax.spines['bottom'].set_linewidth(0.5)
                            
                            # Improve tick formatting
                            ax.tick_params(axis='both', which='major', labelsize=10)
                            
                            # Add some statistics as text
                            abs_error = np.abs(plot_data)
                            max_error = np.max(abs_error)
                            mean_error = np.mean(abs_error)
                            stats_text = f'Max |Error|: {max_error:.4f}\nMean |Error|: {mean_error:.4f}'
                            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                                   fontsize=9, verticalalignment='top',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
                            
                            # Tight layout and save with reduced DPI for speed
                            plt.tight_layout()
                            plt.savefig(img_dir / filename, dpi=100, bbox_inches='tight', 
                                       facecolor='white', edgecolor='none')

                            print(f"Comparison plot saved to {img_dir / filename}")
                        
                        finally:
                            # Ensure matplotlib figure is always closed
                            if fig is not None:
                                plt.close(fig)
                    
                    # Clean up data arrays after plotting
                    del fast_data, baseline_data, fast_grid, baseline_grid, diff_data
                    # The interpolation arrays (ref_swapped, lead, out) will be cleaned up
                    # automatically by Python's garbage collector when the function exits
                    gc.collect()
                    log_memory_usage("after plotting cleanup", verbose=False)

                except Exception as inner_err:
                    # Don't try to clean up here - let the outer handler do it
                    # This avoids Python's variable scoping issues in nested handlers
                    raise

        except Exception as err:
            import traceback
            print(f"[warn] Plotting for '{decision_variable}' failed on '{method_name}': {err}\n{traceback.format_exc()}")
            gc.collect()

        return np.nan

    _plotter.__name__ = f"plot_{decision_variable}_comparison"
    _plotter.__doc__ = (
        f"Plot comparison of '{decision_variable}' between fast method and baseline reference.\n"
        f"Extracts data from stage '{stage}' using sol_attr '{sol_attr or "auto"}' at period {period_idx}.\n\n"
        f"Automatically interpolates baseline policy onto fast method grid for accurate comparison.\n"
        f"Generates professional error plots with statistics and proper economic axis labels.\n"
        f"Saves plots in bundle directory structure for organized results."
    )
    return _plotter
