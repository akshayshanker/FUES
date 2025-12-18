"""
CSV export functionality for plot data from solver_runner.

This module provides drop-in replacements for plotting functions that export
data to CSV files instead of generating matplotlib plots. This allows running
on compute clusters without graphics dependencies, then generating interactive
plots locally.

The CSV files contain all the data needed to reconstruct plots, including:
- Grid points (x-axis values)
- Policy/value function data (y-axis values)  
- Method names and parameters
- Metadata for plot configuration

No additional dependencies required - uses only standard library CSV module.
"""

import csv
import os
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def csv_generate_plots(model, method, image_dir, plot_period=0, bounds=None,
                       save_dir=None, load_dir=None, plot_all_H_idx=False, 
                       y_idx_list=None, egm_bounds=None, skip_egm_plots=False):
    """
    Drop-in replacement for generate_plots that exports EGM data to CSV.
    
    This function has the exact same signature as generate_plots from helpers.plots
    but exports the EGM grid data to CSV files instead of creating matplotlib plots.
    
    Parameters match generate_plots exactly for seamless replacement.
    
    Parameters
    ----------
    skip_egm_plots : bool, optional
        If True, skip EGM CSV exports (only export policy data). Default is False.
    """
    # Convert image_dir to appropriate csv directories
    image_dir = Path(image_dir)
    
    # Check if we're in the new organized structure (images or timestamped images directory)
    # We're already in bundles/hash/METHOD/images_TIMESTAMP/, don't create another method dir
    if (image_dir.name.startswith("images") and "bundles" in str(image_dir)):
        # Create separate directories for different CSV types directly in image_dir
        egm_csv_dir = ensure_dir(image_dir / "egm_csv")
        policy_csv_dir = ensure_dir(image_dir / "policy_csv")
        # Use egm_csv_dir as the main csv_dir for EGM data
        csv_dir = egm_csv_dir
    else:
        # Fallback for backward compatibility
        csv_dir = ensure_dir(image_dir / "csv_egm_data")
    
    # Get the solution
    first_period = model.get_period(plot_period)
    ownc_stage = first_period.get_stage("OWNC")
    
    if not ownc_stage or not ownc_stage.dcsn or not ownc_stage.dcsn.sol:
        print(f"Warning: No solution found for {method} at period {plot_period}")
        return
    
    sol = ownc_stage.dcsn.sol
    grid = ownc_stage.dcsn.grid
    
    # Determine H_indices based on configuration
    if plot_all_H_idx and grid.H_nxt is not None:
        H_indices = list(range(len(grid.H_nxt)))
    elif grid.H_nxt is not None:
        H_indices = [0, len(grid.H_nxt) // 2, len(grid.H_nxt) - 1]
    else:
        H_indices = [0, 1, 2]  # Default if grid not available
    
    # Default y_idx_list - use all income states if None
    if y_idx_list is None:
        # Try to get number of income states from solution shape
        if hasattr(sol, 'policy') and hasattr(sol.policy, 'c') and sol.policy.c is not None:
            n_y = sol.policy.c.shape[2] if len(sol.policy.c.shape) >= 3 else 1
            y_idx_list = list(range(n_y))
        else:
            y_idx_list = [0]  # Fallback to first state only
    
    # Extract EGM data if available (skip if skip_egm_plots is True)
    if not skip_egm_plots:
        print(f"[DEBUG CSV Export] Checking for EGM data in solution object...")
        print(f"[DEBUG CSV Export] sol has EGM attr: {hasattr(sol, 'EGM')}")
        if hasattr(sol, '_arr'):
            print(f"[DEBUG CSV Export] sol._arr type: {type(sol._arr)}")
        if hasattr(sol, '_jit'):
            print(f"[DEBUG CSV Export] sol has _jit attr: True")
            if hasattr(sol._jit, 'EGM'):
                print(f"[DEBUG CSV Export] sol._jit has EGM attr: True")
        
        if hasattr(sol, 'EGM'):
            egm_data = sol.EGM
            print(f"[DEBUG CSV Export] Found EGM data in sol.EGM")
        elif hasattr(sol, '_arr') and isinstance(sol._arr, dict) and 'EGM' in sol._arr:
            egm_data = sol._arr['EGM']
            print(f"[DEBUG CSV Export] Found EGM data in sol._arr['EGM']")
        elif hasattr(sol, '_jit') and hasattr(sol._jit, 'EGM'):
            egm_data = sol._jit.EGM
            print(f"[DEBUG CSV Export] Found EGM data in sol._jit.EGM")
        else:
            egm_data = None
            print(f"[DEBUG CSV Export] No EGM data found")
        
        if egm_data:
            # Export EGM grid data
            for y_idx in y_idx_list:
                for H_idx in H_indices:
                    grid_key = f"{y_idx}-{H_idx}"
                    
                    # Check if egm_data is a dict or has dict-like attributes
                    if isinstance(egm_data, dict):
                        unrefined_data = egm_data.get('unrefined', {})
                        refined_data = egm_data.get('refined', {})
                    else:
                        # Handle SimpleNamespace or other object types
                        unrefined_data = getattr(egm_data, 'unrefined', {})
                        refined_data = getattr(egm_data, 'refined', {})
                    
                    # Export unrefined data
                    if unrefined_data:
                        export_egm_grid_csv(
                            csv_dir / f"egm_unrefined_y{y_idx}_h{H_idx}.csv",
                            unrefined_data, grid_key, H_idx, y_idx,
                            grid.H_nxt[H_idx] if grid.H_nxt is not None else H_idx
                        )
                    
                    # Export refined data
                    if refined_data:
                        export_egm_grid_csv(
                            csv_dir / f"egm_refined_y{y_idx}_h{H_idx}.csv",
                            refined_data, grid_key, H_idx, y_idx,
                            grid.H_nxt[H_idx] if grid.H_nxt is not None else H_idx
                        )
    else:
        print(f"[CSV Export] Skipping EGM CSV exports for {method} (--skip-egm-plots enabled)")
    
    # Export policy function data to appropriate directory
    if image_dir.name == "images" and "bundles" in str(image_dir):
        export_policy_data_csv(policy_csv_dir, sol, grid, H_indices, y_idx_list)
    else:
        export_policy_data_csv(csv_dir, sol, grid, H_indices, y_idx_list)
    
    # Save metadata
    metadata = {
        'method': method,
        'period': plot_period,
        'H_indices': H_indices,
        'y_idx_list': y_idx_list,
        'H_grid': grid.H_nxt.tolist() if grid.H_nxt is not None else None,
        'w_grid': grid.w.tolist() if grid.w is not None else None
    }
    
    with open(csv_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if image_dir.name.startswith("images") and "bundles" in str(image_dir):
        print(f"[CSV Export] Saved data for {method}:")
        if not skip_egm_plots:
            print(f"  - EGM data: {egm_csv_dir}")
        print(f"  - Policy data: {policy_csv_dir}")
        if 'egm_csv_dir' in locals():
            egm_count = len(list(egm_csv_dir.glob('*.csv'))) if not skip_egm_plots else 0
        else:
            egm_count = 0
        if 'policy_csv_dir' in locals():
            policy_count = len(list(policy_csv_dir.glob('*.csv')))
        else:
            policy_count = 0
        total_csv = egm_count + policy_count
        print(f"[CSV Export] Total files: metadata.json + {total_csv} CSV files")
    else:
        print(f"[CSV Export] Saved data for {method} to: {csv_dir}")
        print(f"[CSV Export] Files: metadata.json + {len(list(csv_dir.glob('*.csv')))} CSV files")


def export_egm_grid_csv(filepath, data_dict, grid_key, H_idx, y_idx, h_value):
    """Export a single EGM grid to CSV."""
    prefixed_key = f"e_{grid_key}"
    
    # Handle both dict and object attribute access
    if isinstance(data_dict, dict):
        if prefixed_key not in data_dict:
            return
        e_grid = data_dict[prefixed_key]
        vf = data_dict.get(f"Q_{grid_key}")
        c = data_dict.get(f"c_{grid_key}")
        a = data_dict.get(f"a_{grid_key}")
    else:
        # Handle SimpleNamespace or other objects
        if not hasattr(data_dict, prefixed_key):
            return
        e_grid = getattr(data_dict, prefixed_key)
        vf = getattr(data_dict, f"Q_{grid_key}", None)
        c = getattr(data_dict, f"c_{grid_key}", None)
        a = getattr(data_dict, f"a_{grid_key}", None)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['e', 'value', 'consumption', 'assets', 'H_idx', 'y_idx', 'h_value'])
        
        for i in range(len(e_grid)):
            row = [
                e_grid[i] if e_grid is not None else '',
                vf[i] if vf is not None and i < len(vf) else '',
                c[i] if c is not None and i < len(c) else '',
                a[i] if a is not None and i < len(a) else '',
                H_idx,
                y_idx,
                h_value
            ]
            writer.writerow(row)


def export_policy_data_csv(csv_dir, sol, grid, H_indices, y_idx_list):
    """Export policy function data to CSV."""
    
    # Export consumption policy
    if hasattr(sol, 'policy') and hasattr(sol.policy, 'c'):
        for y_idx in y_idx_list:
            data_rows = []
            for w_idx, w in enumerate(grid.w):
                for H_idx in H_indices:
                    h_val = grid.H_nxt[H_idx] if grid.H_nxt is not None else H_idx
                    c_val = sol.policy.c[w_idx, H_idx, y_idx]
                    data_rows.append([w, h_val, H_idx, y_idx, c_val])
            
            with open(csv_dir / f'policy_c_y{y_idx}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['w', 'h_value', 'H_idx', 'y_idx', 'consumption'])
                writer.writerows(data_rows)
    
    # Export value function
    if hasattr(sol, 'vlu'):
        for y_idx in y_idx_list:
            data_rows = []
            for w_idx, w in enumerate(grid.w):
                for H_idx in H_indices:
                    h_val = grid.H_nxt[H_idx] if grid.H_nxt is not None else H_idx
                    v_val = sol.vlu[w_idx, H_idx, y_idx]
                    data_rows.append([w, h_val, H_idx, y_idx, v_val])
            
            with open(csv_dir / f'value_y{y_idx}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['w', 'h_value', 'H_idx', 'y_idx', 'value'])
                writer.writerows(data_rows)


def csv_plot_comparison_factory(
    decision_variable: str,
    dim_labels: dict,
    plot_axis_label: str,
    slice_config: dict = None,
    stage: str = "OWNC",
    sol_attr: str = None,
    period_idx: int = 0
):
    """
    Drop-in replacement for plot_comparison_factory that exports to CSV.
    
    This factory creates a metric function that exports comparison data to CSV
    instead of creating matplotlib plots. The CSV files can then be used to
    generate interactive plots locally.
    
    Parameters match plot_comparison_factory exactly for seamless replacement.
    """
    
    def _csv_exporter(model, _runner, _x):
        """Export comparison data to CSV instead of plotting."""
        
        # Import here to avoid circular dependencies
        from .metrics import _extract_policy, managed_model_load
        
        # Determine metric name
        metric_name = None
        if decision_variable == 'c':
            metric_name = 'plot_c_comparison'
        elif decision_variable in ('vlu', 'v'):
            metric_name = 'plot_v_comparison'
        
        try:
            # Load baseline model
            with managed_model_load(_runner, _x, metric_name=metric_name) as baseline_model:
                if baseline_model is None:
                    return np.nan
                
                # Set up output directory
                bundle_path = _runner._bundle_path(_x)
                if bundle_path and bundle_path.exists():
                    csv_dir = ensure_dir(bundle_path / "csv_comparisons")
                elif _runner.output_root:
                    csv_dir = ensure_dir(_runner.output_root / "csv_comparisons")
                else:
                    csv_dir = ensure_dir(Path("csv_comparisons"))
                
                params_dict = _runner.unpack(_x)
                method_name = params_dict.get("master.methods.upper_envelope", "unknown_method")
                
                # Determine sol_attr
                _sol_attr = sol_attr if sol_attr is not None else (
                    "policy" if decision_variable in ("c", "a", "h", "pol") else "value"
                )
                
                # Extract data from both models
                fast_data, fast_grid = _extract_policy(
                    model,
                    key=decision_variable,
                    sol_attr=_sol_attr,
                    stage=stage,
                    period_idx=period_idx
                )
                
                ref_data, ref_grid = _extract_policy(
                    baseline_model,
                    key=decision_variable,
                    sol_attr=_sol_attr,
                    stage=stage,
                    period_idx=period_idx
                )
                
                if fast_data is None or ref_data is None:
                    return np.nan
                
                # Export comparison data
                comparison_file = csv_dir / f"{method_name}_{decision_variable}_comparison.csv"
                
                with open(comparison_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    header = ['grid_idx', 'grid_value', 'fast_value', 'baseline_value', 
                             'difference', 'method', 'decision_var']
                    
                    # Add dimension info
                    for dim_idx, dim_label in dim_labels.items():
                        header.append(f'{dim_label}_idx')
                    
                    writer.writerow(header)
                    
                    # Determine which dimensions to iterate over
                    plot_axis_idx = next((k for k, v in dim_labels.items() if v == plot_axis_label), 0)
                    
                    # Get slices to plot
                    if slice_config and plot_axis_label in slice_config:
                        slices_to_plot = slice_config[plot_axis_label]
                    else:
                        slices_to_plot = range(fast_data.shape[plot_axis_idx])
                    
                    # Export data for each slice
                    for slice_idx in slices_to_plot:
                        # Extract slice of data
                        if plot_axis_idx == 0:
                            fast_slice = fast_data[:, slice_idx] if fast_data.ndim > 1 else fast_data
                            ref_slice = ref_data[:, slice_idx] if ref_data.ndim > 1 else ref_data
                        elif plot_axis_idx == 1:
                            fast_slice = fast_data[slice_idx, :] if fast_data.ndim > 1 else fast_data
                            ref_slice = ref_data[slice_idx, :] if ref_data.ndim > 1 else ref_data
                        else:
                            fast_slice = fast_data
                            ref_slice = ref_data
                        
                        # Interpolate if grids differ
                        if len(fast_grid) != len(ref_grid):
                            ref_slice = np.interp(fast_grid, ref_grid, ref_slice)
                        
                        # Write data rows
                        for i, grid_val in enumerate(fast_grid):
                            row = [
                                i,
                                grid_val,
                                fast_slice[i] if i < len(fast_slice) else np.nan,
                                ref_slice[i] if i < len(ref_slice) else np.nan,
                                fast_slice[i] - ref_slice[i] if i < len(fast_slice) and i < len(ref_slice) else np.nan,
                                method_name,
                                decision_variable
                            ]
                            
                            # Add dimension indices
                            for dim_idx in sorted(dim_labels.keys()):
                                if dim_idx == plot_axis_idx:
                                    row.append(slice_idx)
                                else:
                                    row.append(0)  # Default to 0 for non-plot dimensions
                            
                            writer.writerow(row)
                
                print(f"Exported {method_name} {decision_variable} comparison to {comparison_file}")
                
                # Return a success indicator instead of np.nan
                return 0.0
                
        except Exception as e:
            print(f"Error exporting CSV comparison for {method_name}: {e}")
            return np.nan
    
    return _csv_exporter


# Alias for backward compatibility
export_egm_data_to_csv = csv_generate_plots