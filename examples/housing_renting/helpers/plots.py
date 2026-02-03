"""Plotting utilities for housing renting model analysis.

This module provides functions to generate various plots for analyzing
the housing renting model solutions, including:
- EGM (Endogenous Grid Method) grid plots
- Policy function plots
- Value and Q function comparisons

Main Functions
--------------
generate_plots : Generate all plots for a given solution method
plot_egm_grids : Plot refined vs unrefined EGM grids
plot_dcsn_policy : Plot policy functions (consumption, housing, tenure)
plot_compare_value_Q : Compare value/Q functions across methods
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TABLEAU_COLORS
import seaborn as sns

# Optional dependency: dynx is only required for full model plotting.
# Keep this module importable for lightweight utilities (e.g., tax-table plotting).
try:
    from dynx.stagecraft.solmaker import Solution  # type: ignore
except Exception:  # pragma: no cover
    class Solution:  # type: ignore
        pass

def generate_plots(model, method, image_dir, plot_period=0, bounds=None,
                  save_dir=None, load_dir=None, plot_all_H_idx=False,
                  y_idx_list=None, egm_bounds=None, skip_egm_plots=False,
                  policy_config=None):
    """
    Generate both EGM grid plots and policy function plots for a model using a specific method.

    Parameters
    ----------
    model : ModelCircuit or None
        Solved model circuit, or None if loading from disk
    method : str
        Upper envelope method used (FUES, DCEGM, etc.)
    image_dir : str
        Directory to save the output images
    plot_period : int
        Period to plot policy functions for
    bounds : dict
        Dictionary containing bounds for policy function plots
    save_dir : str, optional
        If provided and model is not None, save Solution objects to this directory
    load_dir : str, optional
        If provided and model is None, load Solution objects from this directory
    plot_all_H_idx : bool, optional
        If True, plot EGM grids for all H_idx values. If False (default),
        plot only low, middle, and high housing values.
    y_idx_list : list of int, optional
        List of income indices to plot for EGM grids. If None, defaults to [0].
    egm_bounds : dict, optional
        Dictionary specifying bounds for EGM plots. Keys can be:
        - 'value': (xmin, xmax, ymin, ymax) for value function panel
        - 'assets': (xmin, xmax, ymin, ymax) for assets panel
        - 'he_space': (xmin, xmax, ymin, ymax) for h-e space plot
        Can also specify per y_idx: 'value_y0', 'assets_y1', etc.
    skip_egm_plots : bool, optional
        If True, skip generating EGM plots (saves time with many points). Default False.
    policy_config : dict, optional
        Configuration for policy plots. Keys:
        - 'zoom_windows': list of (x_min, x_max) tuples for asset axis bounds
        - 'y_idx_list': list of income indices to plot
        - 'H_idx_list': list of housing indices to plot
        If None, uses default behavior (full range, y_idx=0, auto H selection).

    Notes
    -----
    If both save_dir and load_dir are None, behavior is unchanged (plots from live model).
    If model is None, load_dir must be provided to load saved solutions.
    """
    # We're already in the method-specific directory, don't create another one
    # The path is: bundles/hash/METHOD/images_TIMESTAMP/
    method_dir = image_dir  # Use image_dir directly, no subdirectory

    if load_dir is not None:
        method_dir_load = os.path.join(load_dir, method)
    else:
        method_dir_load = None

    if save_dir is not None:
        method_dir_save = os.path.join(save_dir, method)
    else:
        method_dir_save = None

    # Create directories for different plot types directly in image_dir
    egm_dir = os.path.join(image_dir, "egm_plots")
    policy_dir = os.path.join(image_dir, "policy_plots")
    os.makedirs(egm_dir, exist_ok=True)
    os.makedirs(policy_dir, exist_ok=True)
    
    # Get the solution either from model or disk
    if model is None:
        if method_dir_load is None:
            raise ValueError("If model is None, load_dir must be provided")
        # Load solution from disk
        fname = f"{method_dir_load}/OWNC_dcsn_period{plot_period}"  # Remove .npz extension
        if not os.path.exists(f"{fname}.npz"):
            raise FileNotFoundError(f"Solution file not found: {fname}.npz")
        ownc_sol = Solution.load(fname)
        print(f"Loaded Solution ← {fname}.npz")
        first_period = None
        ownc_stage = None
    else:
        # Get solution from live model
        first_period = model.get_period(plot_period)
        ownc_stage = first_period.get_stage("OWNC")
        ownc_sol = ownc_stage.dcsn.sol
        
        # Save solution if requested
        if method_dir_save is not None:
            os.makedirs(method_dir_save, exist_ok=True)
            fname = f"{method_dir_save}/OWNC_dcsn_period{plot_period}"  # Remove .npz extension
            ownc_sol.save(fname)
            print(f"Saved Solution → {fname}.npz")
    
    # Generate EGM grid plots (unless skipped)
    if not skip_egm_plots:
        #print(f"\nGenerating EGM grid plots for {method}...")

        # Get the grid dimensions from the solution
        H_grid = ownc_stage.dcsn.grid.H_nxt if ownc_stage else None  # TODO: Store grid in Solution

        # Select housing values to plot
        if plot_all_H_idx and H_grid is not None:
            # Plot all H_idx values - useful for detailed analysis
            H_indices = list(range(len(H_grid)))
            print(f"Plotting EGM grids for all {len(H_grid)} housing values")
        elif H_grid is not None:
            # Select 3 housing values spread across the grid (default behavior)
            H_indices = [0, len(H_grid) // 2, len(H_grid) - 1]  # Low, middle, and high housing values
        else:
            H_indices = [0, 1, 2]  # Default indices if grid not available

        # Default to first income state if not specified
        # Note: For large income grids (16+ states), consider passing explicit list to avoid too many plots
        if y_idx_list is None:
            y_idx_list = [0]  # Default to first state only for EGM plots (can be slow with many states)

        # Plot EGM grid for selected housing and income values
        for y_idx in y_idx_list:
            for H_idx in H_indices:
                # Standard plot with default bounds
                plot_egm_grids(first_period, H_idx, y_idx, method, egm_dir,
                              egm_bounds, sol_override=ownc_sol)

                # Check for variant bounds (_wide, _zoom, _zoom2, _zoom3, _zoom4, etc.) and generate additional plots
                if egm_bounds:
                    for suffix in ["_wide", "_zoom", "_zoom2", "_zoom3", "_zoom4"]:
                        variant_value_key = f"value_y{y_idx}_h{H_idx}{suffix}"
                        variant_assets_key = f"assets_y{y_idx}_h{H_idx}{suffix}"
                        if variant_value_key in egm_bounds or variant_assets_key in egm_bounds:
                            # Create modified bounds dict with variant values mapped to standard keys
                            variant_bounds = egm_bounds.copy()
                            if variant_value_key in egm_bounds:
                                variant_bounds[f"value_y{y_idx}_h{H_idx}"] = egm_bounds[variant_value_key]
                            if variant_assets_key in egm_bounds:
                                variant_bounds[f"assets_y{y_idx}_h{H_idx}"] = egm_bounds[variant_assets_key]
                            # Also check for he_space variant
                            variant_he_key = f"he_space_y{y_idx}_h{H_idx}{suffix}"
                            if variant_he_key in egm_bounds:
                                variant_bounds[f"he_space_y{y_idx}_h{H_idx}"] = egm_bounds[variant_he_key]
                            # Use subfolder for zoom4 plots
                            if suffix == "_zoom4":
                                zoom4_dir = os.path.join(egm_dir, "zoom4")
                                os.makedirs(zoom4_dir, exist_ok=True)
                                plot_egm_grids(first_period, H_idx, y_idx, method, zoom4_dir,
                                              variant_bounds, sol_override=ownc_sol, filename_suffix=suffix)
                            else:
                                plot_egm_grids(first_period, H_idx, y_idx, method, egm_dir,
                                              variant_bounds, sol_override=ownc_sol, filename_suffix=suffix)

        #print(f"EGM grid plots for {method} saved to {egm_dir}")
    else:
        print(f"Skipping EGM grid plots for {method} (--skip-egm-plots enabled)")
    
    # Generate policy function plots
    #print(f"\nGenerating policy function plots for {method}...")
    
    # Plot policy functions
    plot_dcsn_policy(first_period if model else None, policy_dir, bounds,
                     sol_override=ownc_sol, policy_config=policy_config)
    #print(f"Policy function plots for {method} saved to {policy_dir}")

def plot_dcsn_policy(first_period, image_dir, bounds=None, sol_override=None, policy_config=None):
    """Plot policy functions for the renting model, matching fella plot style.
    
    Parameters
    ----------
    first_period : Period or None
        Model period to extract solutions from.
    image_dir : str
        Directory to save output images.
    bounds : dict, optional
        Legacy bounds parameter (unused, kept for backward compatibility).
    sol_override : dict, optional
        Solution dict to use instead of extracting from period.
    policy_config : dict, optional
        Configuration for policy plots:
        - 'zoom_windows': list of (x_min, x_max) tuples for asset axis
        - 'y_idx_list': list of income indices to plot
        - 'H_idx_list': list of housing indices to plot
        If None, uses default (full range, y_idx=0, auto H selection).
    """
    
    # Ensure output directory exists
    os.makedirs(image_dir, exist_ok=True)

    # Parse policy_config with defaults
    if policy_config is None:
        policy_config = {}
    
    zoom_windows = policy_config.get('zoom_windows', [(None, None)])
    y_idx_list = policy_config.get('y_idx_list', [0])
    H_idx_list = policy_config.get('H_idx_list', None)  # None = auto-select

    # Set seaborn style
    sns.set(style="white", rc={"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})
    
    # Extended color cycle for more H values
    color_cycle = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # Get stages
    if first_period is not None:
        tenu_stage = first_period.get_stage("TENU")
        ownh_stage = first_period.get_stage("OWNH") 
        ownc_stage = first_period.get_stage("OWNC")
        rnth_stage = first_period.get_stage("RNTH")
        rntc_stage = first_period.get_stage("RNTC")
    else:
        if sol_override is None:
            raise ValueError("Either first_period or sol_override must be provided")
        class DummyStage:
            def __init__(self):
                self.dcsn = type('obj', (object,), {'sol': None, 'grid': None})()
                self.cntn = type('obj', (object,), {'grid': None})()
        tenu_stage = ownh_stage = ownc_stage = rnth_stage = rntc_stage = DummyStage()
    
    ownc_sol = sol_override if sol_override is not None else (
        ownc_stage.dcsn.sol if ownc_stage.dcsn.sol is not None else None
    )
    
    # Helper to determine H_indices based on config and grid
    def get_H_indices(H_grid_or_H_nxt_grid, use_H_idx_list=True):
        """Return H indices to plot, respecting H_idx_list if set."""
        n_H = len(H_grid_or_H_nxt_grid)
        if use_H_idx_list and H_idx_list is not None:
            # Filter to valid indices
            return [i for i in H_idx_list if i < n_H]
        else:
            # Default: low, mid, high
            if n_H >= 3:
                return [0, n_H // 2, n_H - 1]
            else:
                return list(range(n_H))
    
    # Helper to format zoom window for filename
    def zoom_suffix(x_min, x_max):
        if x_min is None and x_max is None:
            return ""  # Full range, no suffix
        elif x_min is None:
            return f"_x0-{int(x_max)}"
        elif x_max is None:
            return f"_x{int(x_min)}-end"
        else:
            return f"_x{int(x_min)}-{int(x_max)}"
    
    # Helper to apply xlim
    def apply_xlim(ax, x_min, x_max):
        if x_min is not None or x_max is not None:
            ax.set_xlim(left=x_min, right=x_max)
    
    # Loop over income indices and zoom windows
    for y_idx in y_idx_list:
        for x_min, x_max in zoom_windows:
            suffix = f"_y{y_idx}" + zoom_suffix(x_min, x_max)
            
            # --- 1. Consumption policies plot ---
            fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6))
            for ax in ax1:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.grid(True)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            
            if ownc_sol and ownc_stage.dcsn.grid:
                w_grid = ownc_stage.dcsn.grid.w
                H_nxt_grid = ownc_stage.dcsn.grid.H_nxt
                H_indices = get_H_indices(H_nxt_grid)
                    
                for i, H_idx in enumerate(H_indices):
                    h_val = H_nxt_grid[H_idx]
                    color = color_cycle[i % len(color_cycle)]
                    owner_consumption = _get_sol_field(ownc_sol, "policy", "c")[:, H_idx, y_idx]
                    ax1[0].plot(w_grid, owner_consumption, color=color, linestyle='-', label=f"H={h_val:.2f}")
                
                ax1[0].set_title(f"Owner Consumption (y_idx={y_idx})")
                ax1[0].set_ylabel("Consumption (c)")
                ax1[0].set_xlabel("Cash-on-Hand (w)")
                ax1[0].legend(frameon=False, prop={'size': 10})
                apply_xlim(ax1[0], x_min, x_max)
            
            if rntc_stage.dcsn.sol and rntc_stage.dcsn.grid:
                w_grid_rent = rntc_stage.dcsn.grid.w
                S_grid = rntc_stage.dcsn.grid.H_nxt
                S_indices = get_H_indices(S_grid, use_H_idx_list=False)  # Use default for S
                
                for i, S_idx in enumerate(S_indices):
                    s_val = S_grid[S_idx]
                    color = color_cycle[i % len(color_cycle)]
                    renter_consumption = _get_sol_field(rntc_stage.dcsn.sol, "policy", "c")[:, S_idx, y_idx]
                    ax1[1].plot(w_grid_rent, renter_consumption, color=color, linestyle='-', label=f"S={s_val:.2f}")
                
                ax1[1].set_title(f"Renter Consumption (y_idx={y_idx})")
                ax1[1].set_xlabel("Cash-on-Hand (w)")
                ax1[1].legend(frameon=False, prop={'size': 10})
                apply_xlim(ax1[1], x_min, x_max)

            fig1.tight_layout()
            fig1.savefig(os.path.join(image_dir, f"consumption_policies{suffix}.png"))
            plt.close(fig1)

            # --- 2. Housing choice policies plot ---
            fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
            for ax in ax2:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.grid(True)

            if ownh_stage.dcsn.sol and ownh_stage.dcsn.grid:
                a_grid = ownh_stage.dcsn.grid.a
                H_grid = ownh_stage.dcsn.grid.H
                H_indices = get_H_indices(H_grid)
                
                H_policy_data = _get_sol_field(ownh_stage.dcsn.sol, "policy", "H")
                if H_policy_data is not None:
                    H_nxt_grid = ownh_stage.cntn.grid.H_nxt
                    for i, H_idx in enumerate(H_indices):
                        h_val = H_grid[H_idx]
                        color = color_cycle[i % len(color_cycle)]
                        H_policy_idx = H_policy_data[:, H_idx, y_idx].astype(int)
                        H_policy = H_nxt_grid[H_policy_idx]
                        ax2[0].step(a_grid, H_policy, color=color, where='post', label=f"H={h_val:.2f}")
                    
                    ax2[0].set_title(f"Owner Housing Policy (y_idx={y_idx})")
                    ax2[0].set_xlabel("Assets (a)")
                    ax2[0].set_ylabel("Housing Choice (H')")
                    ax2[0].legend(frameon=False, prop={'size': 10})
                    apply_xlim(ax2[0], x_min, x_max)
            
            if rnth_stage.dcsn.sol and rnth_stage.dcsn.grid:
                w_grid_rent = rnth_stage.dcsn.grid.w
                S_policy_data = _get_sol_field(rnth_stage.dcsn.sol, "policy", "S")
                if S_policy_data is not None:
                    S_grid = rnth_stage.cntn.grid.S
                    S_policy_idx = S_policy_data[:, y_idx].astype(int)
                    S_policy = S_grid[S_policy_idx]
                    ax2[1].step(w_grid_rent, S_policy, where='post')
                    ax2[1].set_title(f"Renter Service Choice (y_idx={y_idx})")
                    ax2[1].set_xlabel("Cash-on-Hand (w)")
                    ax2[1].set_ylabel("Rental Services (S)")
                    apply_xlim(ax2[1], x_min, x_max)
            
            fig2.tight_layout()
            fig2.savefig(os.path.join(image_dir, f"housing_policies{suffix}.png"))
            plt.close(fig2)

            # --- 3. Tenure choice plot ---
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.spines['right'].set_visible(False)
            ax3.spines['top'].set_visible(False)
            ax3.grid(True)

            if tenu_stage.dcsn.sol and tenu_stage.dcsn.grid:
                tenure_policy_data = _get_sol_field(tenu_stage.dcsn.sol, "policy", "tenure")
                if tenure_policy_data is not None:
                    a_grid = tenu_stage.dcsn.grid.a
                    H_grid = tenu_stage.dcsn.grid.H
                    H_indices = get_H_indices(H_grid)
                    
                    for i, H_idx in enumerate(H_indices):
                        h_val = H_grid[H_idx]
                        color = color_cycle[i % len(color_cycle)]
                        tenure_policy = tenure_policy_data[:, H_idx, y_idx]
                        ax3.step(a_grid, tenure_policy, color=color, where='post', linewidth=2, label=f"H={h_val:.2f}")
                    
                    ax3.set_title(f"Tenure Choice (y_idx={y_idx})")
                    ax3.set_xlabel("Assets (a)")
                    ax3.set_ylabel("Tenure Choice")
                    ax3.set_yticks([0, 1])
                    ax3.set_yticklabels(["Rent", "Own"])
                    ax3.legend(frameon=False, prop={'size': 10})
                    apply_xlim(ax3, x_min, x_max)

            fig3.tight_layout()
            fig3.savefig(os.path.join(image_dir, f"tenure_policy{suffix}.png"))
            plt.close(fig3)

            # --- 4. Value and Q Function Plots ---
            fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 6))
            for ax in [ax4, ax5]:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.grid(True)

            if ownc_sol and ownc_stage.dcsn.grid:
                w_grid = ownc_stage.dcsn.grid.w
                H_nxt_grid = ownc_stage.dcsn.grid.H_nxt
                H_indices = get_H_indices(H_nxt_grid)

                vlu_data = _get_sol_field(ownc_sol, "vlu")
                q_data = _get_sol_field(ownc_sol, "Q")

                for i, H_idx in enumerate(H_indices):
                    h_val = H_nxt_grid[H_idx]
                    color = color_cycle[i % len(color_cycle)]
                    label = f"H={h_val:.2f}"
                    if vlu_data is not None:
                        ax4.plot(w_grid, vlu_data[:, H_idx, y_idx], color=color, label=label)
                    if q_data is not None:
                        ax5.plot(w_grid, q_data[:, H_idx, y_idx], color=color, label=label)

                apply_xlim(ax4, x_min, x_max)
                apply_xlim(ax5, x_min, x_max)

            ax4.set_title(f"Owner Value Function (y_idx={y_idx})")
            ax4.set_xlabel("Cash-on-Hand (w)")
            ax4.set_ylabel("Value")
            ax4.legend(frameon=False, prop={'size': 10})

            ax5.set_title(f"Owner Q-Function (y_idx={y_idx})")
            ax5.set_xlabel("Cash-on-Hand (w)")
            ax5.legend(frameon=False, prop={'size': 10})
            
            fig4.tight_layout()
            fig4.savefig(os.path.join(image_dir, f"value_q_functions{suffix}.png"))
            plt.close(fig4)

            # --- 5. Savings Plot ---
            fig5, ax6 = plt.subplots(figsize=(10, 6))
            ax6.spines['right'].set_visible(False)
            ax6.spines['top'].set_visible(False)
            ax6.grid(True)

            if ownc_sol and ownc_stage.dcsn.grid:
                H_nxt_grid = ownc_stage.dcsn.grid.H_nxt
                H_indices = get_H_indices(H_nxt_grid)

                for i, H_idx in enumerate(H_indices):
                    h_val = H_nxt_grid[H_idx]
                    color = color_cycle[i % len(color_cycle)]
                    w_grid = ownc_stage.dcsn.grid.w
                    consumption = _get_sol_field(ownc_sol, "policy", "c")[:, H_idx, y_idx]
                    savings = w_grid - consumption
                    ax6.plot(w_grid, savings, color=color, label=f"H={h_val:.2f}")

                ax6.set_title(f"Owner Savings (y_idx={y_idx})")
                ax6.set_xlabel("Cash-on-Hand (w)")
                ax6.set_ylabel("Savings (w-c)")
                ax6.legend(frameon=False, prop={'size': 10})
                apply_xlim(ax6, x_min, x_max)

            fig5.tight_layout()
            fig5.savefig(os.path.join(image_dir, f"savings{suffix}.png"))
            plt.close(fig5)
    
    plt.close('all')

# Add debugging information to plot_egm_grids function
def plot_egm_grids(period, H_idx, y_idx, method, image_dir, bounds=None, sol_override=None,
                   filename_suffix=""):
    """
    Plot endogenous grids before and after upper envelope refinement in the style of example_egm_plot.py.

    Parameters
    ----------
    period : Period
        The model period containing the stages to plot
    H_idx : int
        Housing grid index to plot
    y_idx : int
        Income grid index to plot
    method : str
        Upper envelope method used (FUES, DCEGM, etc.)
    image_dir : str
        Directory to save the output images
    bounds : dict
        Dictionary containing bounds for plots. Keys can be:
        - 'value': (xmin, xmax, ymin, ymax) for value function panel
        - 'assets': (xmin, xmax, ymin, ymax) for assets panel
        - 'he_space': (xmin, xmax, ymin, ymax) for h-e space plot
        - 'value_y{idx}': bounds specific to income index
        - 'assets_y{idx}': bounds specific to income index
        - 'he_space_y{idx}': bounds specific to income index
        - 'value_h{idx}': bounds specific to housing index
        - 'assets_h{idx}': bounds specific to housing index
        - 'he_space_h{idx}': bounds specific to housing index
        - 'value_y{y_idx}_h{h_idx}': bounds specific to both indices
        - 'assets_y{y_idx}_h{h_idx}': bounds specific to both indices
        - 'he_space_y{y_idx}_h{h_idx}': bounds specific to both indices
    sol_override : dict, optional
        If provided, use this solution instead of the one from the period
    filename_suffix : str, optional
        Suffix to append to filename (e.g., "_wide" for alternate bounds)
    """
    # Imports are at module level
    
    # Ensure output directory exists (handles race conditions in MPI)
    os.makedirs(image_dir, exist_ok=True)
    
    if bounds is None:
        bounds = {}

    # Get the owner consumption stage from the period
    if period is not None:
        ownc_stage = period.get_stage("OWNC")
    else:
        # If no period, we need sol_override
        if sol_override is None:
            print(f"[DEBUG] {method}: No period and no sol_override provided")
            return
        ownc_stage = None
    
    # Check if EGM grids are available
    egm_data = _get_sol_field(sol_override or ownc_stage.dcsn.sol, "EGM")
    if egm_data is None or not egm_data:
        print(f"[DEBUG] {method}: No EGM grid data available")

        return
    

    
    # Extract grid data
    grid_key = f"{y_idx}-{H_idx}"
    
    # Get the EGM data structure
    egm_struct = _get_sol_field(sol_override or ownc_stage.dcsn.sol, "EGM")
    if egm_struct is None:
        print(f"[DEBUG] {method}: No EGM structure found")
        return
    
    # Check if we have the unrefined data with the correct prefixed key
    unrefined_dict = egm_struct.get("unrefined", {}) if isinstance(egm_struct, dict) else getattr(egm_struct, 'unrefined', {})
    prefixed_e_key = f"e_{grid_key}"
    
    if prefixed_e_key not in unrefined_dict:
        print(f"[DEBUG] {method}: Grid key {prefixed_e_key} not found in EGM data.")
        if unrefined_dict:
            print(f"[DEBUG] {method}: No unrefined EGM data found")
        return
    
    # Check if we have valid data - use prefixed keys
    e_grid_unrefined = unrefined_dict.get(f"e_{grid_key}")
    if e_grid_unrefined is None or len(e_grid_unrefined) == 0:
        print(f"[DEBUG] {method}: Empty or missing unrefined EGM grid for key e_{grid_key}")
        return
    
    print(f"[DEBUG] {method}: Found valid EGM data for {grid_key}, grid size: {len(e_grid_unrefined)}")
    
    # Get other unrefined data with prefixed keys
    vf_unrefined = unrefined_dict.get(f"Q_{grid_key}")
    c_unrefined = unrefined_dict.get(f"c_{grid_key}")
    a_unrefined = unrefined_dict.get(f"a_{grid_key}")
    
    if any(x is None for x in [vf_unrefined, c_unrefined, a_unrefined]):
        print(f"[DEBUG] {method}: Missing some unrefined data components")
        return
    
    # Skip if we don't have grid information
    if ownc_stage is None:
        print(f"[DEBUG] {method}: No ownc_stage available, cannot determine H_nxt value")
        return
    
    h_unrefined = np.ones_like(e_grid_unrefined) * ownc_stage.dcsn.grid.H_nxt[H_idx]
    
    # Refined grids - get with prefixed keys
    refined_dict = egm_struct.get("refined", {}) if isinstance(egm_struct, dict) else getattr(egm_struct, 'refined', {})
    e_grid_refined = refined_dict.get(f"e_{grid_key}")
    vf_refined = refined_dict.get(f"Q_{grid_key}")
    c_refined = refined_dict.get(f"c_{grid_key}")
    a_refined = refined_dict.get(f"a_{grid_key}")
    
    if any(x is None for x in [e_grid_refined, vf_refined, c_refined, a_refined]):
        print(f"[DEBUG] {method}: Missing some refined data components")
        return
    h_refined = np.ones_like(e_grid_refined) * ownc_stage.dcsn.grid.H_nxt[H_idx]
    
    # Get housing and income values for labels
    h_value = ownc_stage.dcsn.grid.H_nxt[H_idx]
    y_value = y_idx  # This is the index, not the actual value
    
    print(f"[DEBUG] {method}: Creating plots for H={h_value:.2f}, y_idx={y_value}")
    
    # Set up seaborn style - matching durables plots
    sns.set(style="white", rc={"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})

    # Create a two-panel figure: value function and housing choice
    fig, ax = plt.subplots(1, 2)
    
    # Create a divergent color palette
    palette = sns.color_palette("cubehelix", 3)
    
    # Plot value function
    ax[0].scatter(
        e_grid_unrefined[1:],
        vf_unrefined[1:],
        s=20,
        facecolors='none',
        edgecolors='r',
        label='EGM points'
    )

    ax[0].scatter(
        e_grid_refined[1:],
        vf_refined[1:],
        color='blue',
        s=15,
        marker='x',
        linewidth=0.75,
        label=f'{method} optimal points'
    )

    # Apply custom bounds for value panel
    # Check bounds in order of specificity: y+h specific, then h-specific, then y-specific, then general
    value_yh_key = f"value_y{y_idx}_h{H_idx}"
    value_h_key = f"value_h{H_idx}"
    value_y_key = f"value_y{y_idx}"
    
    if value_yh_key in bounds:
        xmin, xmax, ymin, ymax = bounds[value_yh_key]
    elif value_h_key in bounds:
        xmin, xmax, ymin, ymax = bounds[value_h_key]
    elif value_y_key in bounds:
        xmin, xmax, ymin, ymax = bounds[value_y_key]
    elif "value" in bounds:
        xmin, xmax, ymin, ymax = bounds["value"]
    elif "egm_value" in bounds:  # Legacy support
        xmin, xmax, ymin, ymax = bounds["egm_value"]
    else:
        xmin = xmax = ymin = ymax = None
    
    if xmin is not None or xmax is not None:
        ax[0].set_xlim([xmin, xmax])

    # Auto-fit y-axis for value panel based on data within x-range
    if ymin is None and ymax is None and (xmin is not None or xmax is not None):
        # Filter data within x-range and compute y bounds
        x_lo = xmin if xmin is not None else -np.inf
        x_hi = xmax if xmax is not None else np.inf
        mask = (e_grid_refined >= x_lo) & (e_grid_refined <= x_hi)
        if np.any(mask):
            vf_in_range = vf_refined[mask]
            data_ymin = np.min(vf_in_range)
            data_ymax = np.max(vf_in_range)
            # Add small margin (5%)
            margin = 0.05 * (data_ymax - data_ymin) if data_ymax > data_ymin else 0.1
            ax[0].set_ylim([data_ymin - margin, data_ymax + margin])
    elif ymin is not None or ymax is not None:
        ax[0].set_ylim([ymin, ymax])

    # Formatting for value function plot
    ax[0].set_ylabel('Value', fontsize=11)
    # Show all spines as subtle border (matching durables style)
    for spine in ax[0].spines.values():
        spine.set_visible(True)
        spine.set_color('0.65')
        spine.set_linewidth(0.8)
    ax[0].legend(frameon=False, prop={'size': 10})
    ax[0].tick_params(labelsize=9)
    ax[0].grid(True)

    # Get kink points for later use (on assets panel y-axis)
    kink_points = bounds.get("kink_points", [])
    
    # Special tick formatting for value_h14 plot
    if H_idx == 14:
        # For H_idx=14, set ticks every 0.0001 on y-axis
        if ymin is not None and ymax is not None:
            # Create ticks every 0.0001 within the bounds
            y_ticks = np.arange(ymin, ymax + 0.0001, 0.0001)
            ax[0].yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
            ax[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        else:
            # Fallback to default if bounds not specified
            ax[0].yaxis.set_major_locator(mticker.MaxNLocator(6))
            ax[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    else:
        # Use MaxNLocator to set a fixed number of ticks for other plots
        ax[0].yaxis.set_major_locator(mticker.MaxNLocator(6))
        ax[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    
    ax[0].xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    
    # Plot housing choice
    ax[1].scatter(
        e_grid_unrefined,
        a_unrefined,
        s=20,
        facecolor='none',
        edgecolor='r',
        label='EGM points'
    )

    ax[1].scatter(
        e_grid_refined,
        a_refined,
        s=20,
        color='blue',
        marker='x',
        linewidth=0.75,
        label=f'{method} optimal points'
    )
    
    # Apply custom bounds for housing choice panel
    # Check bounds in order of specificity: y+h specific, then h-specific, then y-specific, then general
    assets_yh_key = f"assets_y{y_idx}_h{H_idx}"
    assets_h_key = f"assets_h{H_idx}"
    assets_y_key = f"assets_y{y_idx}"
    
    if assets_yh_key in bounds:
        xmin, xmax, ymin, ymax = bounds[assets_yh_key]
    elif assets_h_key in bounds:
        xmin, xmax, ymin, ymax = bounds[assets_h_key]
    elif assets_y_key in bounds:
        xmin, xmax, ymin, ymax = bounds[assets_y_key]
    elif "assets" in bounds:
        xmin, xmax, ymin, ymax = bounds["assets"]
    elif "egm_assets" in bounds:  # Legacy support
        xmin, xmax, ymin, ymax = bounds["egm_assets"]
    else:
        xmin = xmax = ymin = ymax = None
    
    if xmin is not None or xmax is not None:
        ax[1].set_xlim([xmin, xmax])
    if ymin is not None or ymax is not None:
        ax[1].set_ylim([ymin, ymax])
    
    # Formatting for housing choice plot
    ax[1].set_ylabel(r'Liquid savings at time $t+1$', fontsize=11)
    ax[1].tick_params(labelsize=9)
    # Show all spines as subtle border (matching durables style)
    for spine in ax[1].spines.values():
        spine.set_visible(True)
        spine.set_color('0.65')
        spine.set_linewidth(0.8)
    ax[1].grid(True)

    # Draw horizontal lines at kink points (tax bracket boundaries on t+1 savings)
    # Only draw lines within the visible y-range
    # Get current y-limits for filtering
    assets_ylim = ax[1].get_ylim()
    y_lo, y_hi = assets_ylim[0], assets_ylim[1]

    # First line gets label for legend, rest are unlabeled
    tax_line_drawn = False
    for kp in kink_points:
        # Only draw if kink point is strictly within visible range
        if y_lo < kp < y_hi:
            if not tax_line_drawn:
                ax[1].axhline(y=kp, color='black', linestyle='--', linewidth=0.8,
                             label='Tax brackets')
                tax_line_drawn = True
            else:
                ax[1].axhline(y=kp, color='black', linestyle='--', linewidth=0.8)

    # Legend after tax bracket lines so they appear in it
    # Place at upper right to avoid overlap with tax bracket horizontal lines
    ax[1].legend(frameon=False, prop={'size': 10}, loc='upper right')

    # Special tick formatting for value_h14 plot
    if H_idx == 14:
        # For H_idx=14, set ticks every 0.1 on y-axis for assets
        if ymin is not None and ymax is not None:
            # Create ticks every 0.1 within the bounds
            y_ticks = np.arange(ymin, ymax + 0.1, 0.1)
            ax[1].yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
            ax[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        else:
            # Fallback to default if bounds not specified
            ax[1].yaxis.set_major_locator(mticker.MaxNLocator(6))
            ax[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    else:
        # Use MaxNLocator to set a fixed number of ticks for other plots
        ax[1].yaxis.set_major_locator(mticker.MaxNLocator(6))
        ax[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    ax[1].xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax[1].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # No title - matching durables style

    # Shared x-axis label (matching durables style)
    fig.supxlabel(r'Total wealth at time $t$', fontsize=11)

    # Tight layout
    fig.tight_layout()

    # Save figure
    period_idx = period.time_index if period else 0
    filename = f"egm_grid_H{H_idx}_y{y_idx}_period{period_idx}_{method}{filename_suffix}.png"
    fig.savefig(os.path.join(image_dir, filename))
    #print(f"EGM grid plot saved to {os.path.join(image_dir, filename)}")
    
    # Close figure
    plt.close(fig)
    
    # Now create an additional plot showing endogenous grids in the h-e space (like the third plot in example_egm_plot.py)
    # Match durables single-panel style: no explicit figsize, font.size=11
    fig2, ax2 = plt.subplots(1, 1)

    # Set style - matching durables single-panel plot
    sns.set(style="white", rc={"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})

    # Plot endogenous grids
    ax2.scatter(
        a_unrefined[1:],
        e_grid_unrefined[1:],
        s=20,
        facecolor='none',
        edgecolor='r',
        label='EGM points'
    )

    ax2.scatter(
        a_refined[1:],
        e_grid_refined[1:],
        color='blue',
        s=15,
        marker='x',
        linewidth=0.75,
        label=f'{method} optimal points'
    )

    # Formatting
    ax2.set_xlabel(r'Exogenous grid of housing assets at time t+1', fontsize=11)
    ax2.set_ylabel(r'Endogenous grid of total wealth at time $t$', fontsize=11)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.legend(frameon=False, prop={'size': 11})
    ax2.grid(True)

    # Apply custom bounds for h-e space plot
    # Check bounds in order of specificity: y+h specific, then h-specific, then y-specific, then general
    he_yh_key = f"he_space_y{y_idx}_h{H_idx}"
    he_h_key = f"he_space_h{H_idx}"
    he_y_key = f"he_space_y{y_idx}"

    if he_yh_key in bounds:
        xmin, xmax, ymin, ymax = bounds[he_yh_key]
    elif he_h_key in bounds:
        xmin, xmax, ymin, ymax = bounds[he_h_key]
    elif he_y_key in bounds:
        xmin, xmax, ymin, ymax = bounds[he_y_key]
    elif "he_space" in bounds:
        xmin, xmax, ymin, ymax = bounds["he_space"]
    else:
        xmin = xmax = ymin = ymax = None

    if xmin is not None or xmax is not None:
        ax2.set_xlim([xmin, xmax])
    if ymin is not None or ymax is not None:
        ax2.set_ylim([ymin, ymax])

    # Use locators before formatters to ensure fixed number of ticks
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # No title - matching durables single-panel style

    # Save figure
    period_idx = period.time_index if period else 0
    filename2 = f"egm_grid_he_space_H{H_idx}_y{y_idx}_period{period_idx}_{method}{filename_suffix}.png"
    fig2.savefig(os.path.join(image_dir, filename2))

    # Close figure
    plt.close(fig2)

    # Create full-range h-e space plot (0-20 x-axis) with tax bracket lines
    fig3, ax3 = plt.subplots(1, 1)

    # Set style - matching durables single-panel plot
    sns.set(style="white", rc={"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})

    # Plot endogenous grids
    ax3.scatter(
        a_unrefined[1:],
        e_grid_unrefined[1:],
        s=20,
        facecolor='none',
        edgecolor='r',
        label='EGM points'
    )

    ax3.scatter(
        a_refined[1:],
        e_grid_refined[1:],
        color='blue',
        s=15,
        marker='x',
        linewidth=0.75,
        label=f'{method} optimal points'
    )

    # Draw vertical dotted lines at tax bracket boundaries (on x-axis = exogenous grid)
    tax_line_drawn = False
    for kp in kink_points:
        if kp <= 20:  # Only draw within x-axis range
            if not tax_line_drawn:
                ax3.axvline(x=kp, color='gray', linestyle=':', linewidth=1.0, alpha=0.8,
                           label='Tax brackets')
                tax_line_drawn = True
            else:
                ax3.axvline(x=kp, color='gray', linestyle=':', linewidth=1.0, alpha=0.8)

    # Formatting
    ax3.set_xlabel(r'Exogenous grid of liquid savings at time $t+1$', fontsize=11)
    ax3.set_ylabel(r'Endogenous grid of total wealth at time $t$', fontsize=11)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.legend(frameon=False, prop={'size': 11})
    ax3.grid(True)

    # Fixed bounds: x-axis 0-20, y-axis auto
    ax3.set_xlim([0, 20])

    # Use locators before formatters
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax3.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # No title - matching durables single-panel style

    # Save figure
    filename3 = f"egm_grid_he_space_full_H{H_idx}_y{y_idx}_period{period_idx}_{method}{filename_suffix}.png"
    fig3.savefig(os.path.join(image_dir, filename3))

    # Close figure
    plt.close(fig3)

    # Create zoomed e-a space plot (x-axis = total wealth 4-5) with tax bracket lines
    # Note: axes swapped vs h-e space - x=wealth at t, y=savings at t+1
    fig4, ax4 = plt.subplots(1, 1)

    # Set style - matching durables single-panel plot
    sns.set(style="white", rc={"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})

    # Plot endogenous grids (x = e_grid = wealth at t, y = a = savings at t+1)
    ax4.scatter(
        e_grid_unrefined[1:],
        a_unrefined[1:],
        s=20,
        facecolor='none',
        edgecolor='r',
        label='EGM points'
    )

    ax4.scatter(
        e_grid_refined[1:],
        a_refined[1:],
        color='blue',
        s=15,
        marker='x',
        linewidth=0.75,
        label=f'{method} optimal points'
    )

    # Draw horizontal dotted lines at tax bracket boundaries (on y-axis = savings)
    tax_line_drawn = False
    for kp in kink_points:
        # Show all kink points that might be visible in y-range
        if not tax_line_drawn:
            ax4.axhline(y=kp, color='gray', linestyle=':', linewidth=1.0, alpha=0.8,
                       label='Tax brackets')
            tax_line_drawn = True
        else:
            ax4.axhline(y=kp, color='gray', linestyle=':', linewidth=1.0, alpha=0.8)

    # Formatting
    ax4.set_xlabel(r'Total wealth at time $t$', fontsize=11)
    ax4.set_ylabel(r'Liquid savings at time $t+1$', fontsize=11)
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.legend(frameon=False, prop={'size': 11})
    ax4.grid(True)

    # Fixed x-bounds 4-5 (total wealth at t), y-axis auto-fit
    ax4.set_xlim([4, 5])

    # Use locators before formatters
    ax4.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax4.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax4.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # No title - matching durables single-panel style

    # Save figure
    filename4 = f"egm_grid_he_space_zoom45_H{H_idx}_y{y_idx}_period{period_idx}_{method}{filename_suffix}.png"
    fig4.savefig(os.path.join(image_dir, filename4))

    # Close figure
    plt.close(fig4)

    # =========================================================================
    # NEW: Two-panel plot with exogenous grid (a_nxt) on x-axis
    # LHS: X = a_nxt, Y = Q (value = u + beta*vf_nxt)
    # RHS: X = a_nxt, Y = m (endogenous grid)
    # =========================================================================
    fig5, ax5 = plt.subplots(1, 2, figsize=(10, 4))

    # Set style
    sns.set(style="white", rc={"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})

    # --- LHS: Value (Q) vs a_nxt ---
    ax5[0].scatter(
        a_unrefined[1:],
        vf_unrefined[1:],
        s=20,
        facecolors='none',
        edgecolors='r',
        label='EGM points'
    )
    ax5[0].scatter(
        a_refined[1:],
        vf_refined[1:],
        color='blue',
        s=15,
        marker='x',
        linewidth=0.75,
        label=f'{method} optimal points'
    )

    # Draw vertical lines at tax bracket boundaries
    tax_line_drawn = False
    for kp in kink_points:
        if 8 <= kp <= 10:  # Only within x-axis range
            if not tax_line_drawn:
                ax5[0].axvline(x=kp, color='black', linestyle='--', linewidth=0.8,
                              label='Tax brackets')
                tax_line_drawn = True
            else:
                ax5[0].axvline(x=kp, color='black', linestyle='--', linewidth=0.8)

    ax5[0].set_ylabel('Value', fontsize=11)
    ax5[0].set_xlim([8, 10])

    # Auto-fit y-axis to REFINED points only within x-range [8, 10]
    # (unrefined points may have outliers from constraint segments)
    mask_ref = (a_refined[1:] >= 8) & (a_refined[1:] <= 10)
    vf_in_range = vf_refined[1:][mask_ref]
    if len(vf_in_range) > 0:
        y_min, y_max = vf_in_range.min(), vf_in_range.max()
        y_margin = (y_max - y_min) * 0.05  # 5% margin
        ax5[0].set_ylim([y_min - y_margin, y_max + y_margin])

    ax5[0].legend(frameon=False, prop={'size': 10})
    ax5[0].grid(True)
    for spine in ax5[0].spines.values():
        spine.set_visible(True)
        spine.set_color('0.65')
        spine.set_linewidth(0.8)
    ax5[0].xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax5[0].yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax5[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax5[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # --- RHS: Endogenous grid (m) vs a_nxt ---
    ax5[1].scatter(
        a_unrefined[1:],
        e_grid_unrefined[1:],
        s=20,
        facecolors='none',
        edgecolors='r',
        label='EGM points'
    )
    ax5[1].scatter(
        a_refined[1:],
        e_grid_refined[1:],
        color='blue',
        s=15,
        marker='x',
        linewidth=0.75,
        label=f'{method} optimal points'
    )

    # Draw vertical lines at tax bracket boundaries
    tax_line_drawn = False
    for kp in kink_points:
        if 8 <= kp <= 10:  # Only within x-axis range
            if not tax_line_drawn:
                ax5[1].axvline(x=kp, color='black', linestyle='--', linewidth=0.8,
                              label='Tax brackets')
                tax_line_drawn = True
            else:
                ax5[1].axvline(x=kp, color='black', linestyle='--', linewidth=0.8)

    ax5[1].set_ylabel(r'Endogenous grid (total wealth at $t$)', fontsize=11)
    ax5[1].set_xlim([8, 10])
    ax5[1].set_ylim([8, 11])
    ax5[1].legend(frameon=False, prop={'size': 10})
    ax5[1].grid(True)
    for spine in ax5[1].spines.values():
        spine.set_visible(True)
        spine.set_color('0.65')
        spine.set_linewidth(0.8)
    ax5[1].xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax5[1].yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax5[1].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax5[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Shared x-axis label
    fig5.supxlabel(r'Exogenous grid (savings at $t+1$)', fontsize=11)

    fig5.tight_layout()

    # Save to same directory as other plots from this function call
    filename5 = f"egm_exog_space_H{H_idx}_y{y_idx}_period{period_idx}_{method}{filename_suffix}.png"
    fig5.savefig(os.path.join(image_dir, filename5))

    plt.close(fig5)

def plot_compare_value_Q(model_list, methods, image_dir, plot_period=0, bounds=None):
    """Create comparison plots of value and Q functions across several methods.

    Parameters
    ----------
    model_list : list[ModelCircuit]
        List of solved model circuits – one per method.
    methods : list[str]
        Names of methods in the same order as `model_list` (e.g. ["FUES", "VFI", "DCEGM"]).
    image_dir : str
        Directory in which to save the output figures.
    plot_period : int, optional
        Model period that should be visualised (default: 0).
    bounds : dict | None, optional
        Optional axis‐bounds in the same spirit as the `BOUNDS` dict in ``solve_single_model.py``.
        Keys accepted: ``value`` and ``Q`` – each mapping to (xmin, xmax, ymin, ymax).
    """
    # Imports are at module level

    if bounds is None:
        bounds = {}

    # One colour per housing value, one marker per method
    colour_cycle = list(TABLEAU_COLORS.values())[:3]
    marker_cycle = ["o", "s", "^", "D", "x", "*"]

    # Use the first model to extract the housing grid
    base_period = model_list[0].get_period(plot_period)
    ownc_base = base_period.get_stage("OWNC")
    H_grid = ownc_base.dcsn.grid.H_nxt
    y_idx = 0  # consumption plots already fix this to 0

    # Representative housing indices (low / mid / high)
    if len(H_grid) >= 5:
        H_indices = [0, len(H_grid)//2, len(H_grid)-1]
    else:
        H_indices = list(range(len(H_grid)))

    # --------------------------------------------------
    #   1. VALUE FUNCTION COMPARISON
    # --------------------------------------------------
    fig_val, ax_val = plt.subplots(figsize=(10, 6))

    for h_pos, H_idx in enumerate(H_indices):
        colour = colour_cycle[h_pos % len(colour_cycle)]
        for m_pos, (method, model) in enumerate(zip(methods, model_list)):
            period = model.get_period(plot_period)
            ownc = period.get_stage("OWNC")
            if _get_sol_field(ownc.dcsn.sol, "vlu") is None:
                continue  # skip silently if value function not available
            w_grid = ownc.dcsn.grid.w
            v_func = _get_sol_field(ownc.dcsn.sol, "vlu")[:, H_idx, y_idx]
            label = f"{method} (H={H_grid[H_idx]:.2f})"
            ax_val.plot(
                w_grid,
                v_func,
                color=colour,
                marker=marker_cycle[m_pos % len(marker_cycle)],
                markevery=max(1, len(w_grid)//15),
                linewidth=1,
                label=label
            )

    ax_val.set_xlabel("Cash-on-Hand (w)")
    ax_val.set_ylabel("Value Function (V)")
    ax_val.set_title("Value Function – method comparison")
    ax_val.grid(True)
    if "value" in bounds:
        xmin, xmax, ymin, ymax = bounds["value"]
        if xmin is not None or xmax is not None:
            ax_val.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            ax_val.set_ylim(bottom=ymin, top=ymax)
    ax_val.legend(fontsize=8, ncol=2)

    fig_val.tight_layout()
    fig_val.savefig(os.path.join(image_dir, "compare_value_functions.png"))

    # --------------------------------------------------
    #   2. Q FUNCTION COMPARISON
    # --------------------------------------------------
    fig_q, ax_q = plt.subplots(figsize=(10, 6))

    for h_pos, H_idx in enumerate(H_indices):
        colour = colour_cycle[h_pos % len(colour_cycle)]
        for m_pos, (method, model) in enumerate(zip(methods, model_list)):
            period = model.get_period(plot_period)
            ownc = period.get_stage("OWNC")
            if _get_sol_field(ownc.dcsn.sol, "Q") is None:
                continue
            w_grid = ownc.dcsn.grid.w
            q_func = _get_sol_field(ownc.dcsn.sol, "Q")[:, H_idx, y_idx]
            label = f"{method} (H={H_grid[H_idx]:.2f})"
            ax_q.plot(
                w_grid,
                q_func,
                color=colour,
                marker=marker_cycle[m_pos % len(marker_cycle)],
                markevery=max(1, len(w_grid)//15),
                linewidth=1,
                label=label
            )

    ax_q.set_xlabel("Cash-on-Hand (w)")
    ax_q.set_ylabel("Q Function (Q)")
    ax_q.set_title("Q Function – method comparison")
    ax_q.grid(True)
    if "Q" in bounds:
        xmin, xmax, ymin, ymax = bounds["Q"]
        if xmin is not None or xmax is not None:
            ax_q.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            ax_q.set_ylim(bottom=ymin, top=ymax)
    ax_q.legend(fontsize=8, ncol=2)

    fig_q.tight_layout()
    fig_q.savefig(os.path.join(image_dir, "compare_Q_functions.png"))

    # --------------------------------------------------
    #   3. VALUE FUNCTION DIFFERENCE PLOT
    # --------------------------------------------------
    if len(model_list) > 1:
        baseline_method = methods[0]
        fig_diff, ax_diff = plt.subplots(figsize=(10, 6))

        for h_pos, H_idx in enumerate(H_indices):
            colour = colour_cycle[h_pos % len(colour_cycle)]
            # Baseline value function
            v_base = _get_sol_field(model_list[0].get_period(plot_period).get_stage("OWNC").dcsn.sol, "Q")[:, H_idx, y_idx]

            for m_pos, (method, model) in enumerate(zip(methods[1:], model_list[1:]), start=1):
                v_other = _get_sol_field(model.get_period(plot_period).get_stage("OWNC").dcsn.sol, "Q")[:, H_idx, y_idx]
                diff_raw = v_other - v_base
                # Exclude large positive deviations > 1 for cleaner visualisation
                import numpy as np
                diff = np.where(np.abs(diff_raw) > 0.55, np.nan, diff_raw)
                label = f"{method}-{baseline_method} (H={H_grid[H_idx]:.2f})"

                ax_diff.plot(
                    w_grid,
                    diff,
                    color=colour,
                    marker=marker_cycle[m_pos % len(marker_cycle)],
                    markevery=max(1, len(w_grid)//15),
                    linewidth=1,
                    label=label
                )

        # zero reference line
        ax_diff.axhline(0, color="black", linestyle="--", linewidth=0.8)

        ax_diff.set_xlabel("Cash-on-Hand (w)")
        ax_diff.set_ylabel(f"Δ Q Function (method − {baseline_method})")
        ax_diff.set_title(f"Q Function Difference: method minus {baseline_method}")
        ax_diff.grid(True, linestyle=":", linewidth=0.5)

        if "value_diff" in bounds:
            xmin, xmax, ymin, ymax = bounds["value_diff"]
            if xmin is not None or xmax is not None:
                ax_diff.set_xlim(left=xmin, right=xmax)
            if ymin is not None or ymax is not None:
                ax_diff.set_ylim(bottom=ymin, top=ymax)

        ax_diff.legend(fontsize=8, ncol=2)
        fig_diff.tight_layout()
        fig_diff.savefig(os.path.join(image_dir, "compare_Q_function_differences.png"))

    plt.close("all")


def plot_compare_consumption_policy(model_fues, model_vfi, image_dir, plot_period=0,
                                    H_indices=None, y_idx=0, bounds=None,
                                    jump_threshold=0.5):
    """Create side-by-side consumption policy plots comparing FUES and VFI.

    Similar to durables plot_pols function - shows two methods side by side
    with jump removal for clean visualization.

    Parameters
    ----------
    model_fues : ModelCircuit
        Solved model using FUES method
    model_vfi : ModelCircuit
        Solved model using VFI method (baseline)
    image_dir : str
        Directory to save the output images
    plot_period : int
        Period index to plot (default: 0)
    H_indices : list[int] or None
        Housing indices to plot. If None, uses [0, mid, last].
    y_idx : int
        Income index to use (default: 0)
    bounds : dict or None
        Optional axis bounds with keys 'consumption' -> (xmin, xmax, ymin, ymax)
    jump_threshold : float
        Threshold for detecting jumps in policy function (default: 0.5)
    """
    sns.set(style="white", rc={"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})

    # Extract periods and stages
    period_fues = model_fues.get_period(plot_period)
    period_vfi = model_vfi.get_period(plot_period)

    ownc_fues = period_fues.get_stage("OWNC")
    ownc_vfi = period_vfi.get_stage("OWNC")

    # Get grids
    w_grid = ownc_fues.dcsn.grid.w
    H_grid = ownc_fues.dcsn.grid.H_nxt

    # Determine H indices to plot
    if H_indices is None:
        if len(H_grid) >= 5:
            H_indices = [0, len(H_grid)//2, len(H_grid)-1]
        else:
            H_indices = list(range(len(H_grid)))

    # Extract consumption policies
    c_fues = _get_sol_field(ownc_fues.dcsn.sol, "policy", "c")
    c_vfi = _get_sol_field(ownc_vfi.dcsn.sol, "policy", "c")

    if c_fues is None or c_vfi is None:
        print("[warn] Could not extract consumption policy for comparison plot")
        return

    # Color palette for different H values
    colors = ['blue', 'red', 'green']

    # Create two-panel figure: FUES on left, VFI on right
    fig, ax = plt.subplots(1, 2)

    for col_idx, H_idx in enumerate(H_indices):
        color = colors[col_idx % len(colors)]
        H_val = H_grid[H_idx]
        label = f"H = {H_val:.2f}"

        # Extract 1D consumption policy for this H, y
        c_fues_1d = c_fues[:, H_idx, y_idx]
        c_vfi_1d = c_vfi[:, H_idx, y_idx]

        # Detect jumps and insert NaN to break the line (like durables plot_pols)
        # FUES
        pos_fues = np.where(np.abs(np.diff(c_fues_1d)) > jump_threshold)[0] + 1
        c_fues_plot = np.insert(c_fues_1d, pos_fues, np.nan)
        w_fues_plot = np.insert(w_grid, pos_fues, np.nan)

        # VFI
        pos_vfi = np.where(np.abs(np.diff(c_vfi_1d)) > jump_threshold)[0] + 1
        c_vfi_plot = np.insert(c_vfi_1d, pos_vfi, np.nan)
        w_vfi_plot = np.insert(w_grid, pos_vfi, np.nan)

        # Plot FUES (left panel)
        ax[0].plot(w_fues_plot, c_fues_plot, color=color, label=label, linewidth=0.75)

        # Plot VFI (right panel)
        ax[1].plot(w_vfi_plot, c_vfi_plot, color=color, label=label, linewidth=0.75)

    # Format left panel (FUES)
    ax[0].set_ylabel(r'Consumption at time $t$', fontsize=11)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].tick_params(labelsize=9)
    ax[0].grid(True)
    ax[0].legend(frameon=False, prop={'size': 10})
    ax[0].set_title('FUES', fontsize=11)

    # Format right panel (VFI)
    ax[1].set_ylabel(r'Consumption at time $t$', fontsize=11)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].tick_params(labelsize=9)
    ax[1].grid(True)
    ax[1].legend(frameon=False, prop={'size': 10})
    ax[1].set_title('VFI', fontsize=11)

    # Apply bounds if provided
    if bounds and "consumption" in bounds:
        xmin, xmax, ymin, ymax = bounds["consumption"]
        if xmin is not None or xmax is not None:
            ax[0].set_xlim([xmin, xmax])
            ax[1].set_xlim([xmin, xmax])
        if ymin is not None or ymax is not None:
            ax[0].set_ylim([ymin, ymax])
            ax[1].set_ylim([ymin, ymax])

    # Tick formatters
    ax[0].yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax[0].xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    ax[1].yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax[1].xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    # Shared x-axis label
    fig.supxlabel(r'Total wealth at time $t$', fontsize=11)

    fig.tight_layout()

    # Save figure
    filename = f"compare_consumption_FUES_VFI_period{plot_period}_y{y_idx}.png"
    fig.savefig(os.path.join(image_dir, filename))
    print(f"Consumption comparison plot saved to {os.path.join(image_dir, filename)}")

    plt.close(fig)


def _get_sol_field(sol, *keys):
    """Safely get a nested attribute from a Solution object."""
    try:
        for key in keys:
            if isinstance(sol, dict):
                sol = sol.get(key)
            else:
                # Handle special case for EGM data which uses prefixed keys
                if hasattr(sol, 'EGM') and key == "EGM":
                    sol = sol.EGM
                elif hasattr(sol, key):
                    sol = getattr(sol, key)
                else:
                    # Handle EGM keys with prefixes (e.g., "unrefined.e" -> EGM.unrefined["e_..."])
                    if "." in key:
                        parts = key.split(".", 1)
                        if len(parts) == 2:
                            base_key, field = parts
                            if hasattr(sol, base_key):
                                base_obj = getattr(sol, base_key)
                                if isinstance(base_obj, dict):
                                    # Look for keys that start with the field prefix
                                    prefix = f"{field}_"
                                    matching_keys = {k[len(prefix):]: v for k, v in base_obj.items() 
                                                   if k.startswith(prefix)}
                                    if matching_keys:
                                        sol = matching_keys
                                    else:
                                        return None
                                else:
                                    return None
                            else:
                                return None
                        else:
                            return None
                    else:
                        return None
            if sol is None:
                return None
        return sol
    except (AttributeError, KeyError):
        return None


def _style_axis_spines(ax, *, sides=("left", "bottom"), color="0.65", linewidth=0.8) -> None:
    """Lighten axis spines (paper-style)."""
    for side in sides:
        try:
            ax.spines[side].set_color(color)
            ax.spines[side].set_linewidth(linewidth)
        except Exception:
            pass
    try:
        ax.tick_params(axis="both", which="both", color=color)
    except Exception:
        pass


def plot_tax_table(
    tax_table: dict,
    *,
    a_min: float = 0.0,
    a_max: float | None = None,
    plot_tau_y_pct: bool = True,
    figsize=(6.5, 3.0),
    n_bg: int = 1500,
    show_jump_lines: bool = False,
    save_path: str | None = None,
):
    """Plot a piecewise-linear asset tax schedule from a dict.

    This is a lightweight plotting helper intended for paper figures.
    It does **not** build/compile models — it just visualizes the schedule.

    Expected schema (typically `master.parameters.tax_table`):
    - tax_table["brackets"] is a list of dicts with keys: a0, a1, B, tau_a

    Interpretation:
      T(a) = B + tau_a * (a - a0)  for a in [a0, a1)
    Jumps are represented by discontinuities in B at bracket boundaries.

    If `plot_tau_y_pct=True`, we plot the marginal rate as an income-tax-equivalent
    percent on the asset return, using tau_y(a) = tau_a(a) / r_tax.
    """
    if tax_table is None:
        raise ValueError("tax_table is None")

    # For paper figures we want consistent fonts (matches examples/durables).
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["font.family"] = "DejaVu Sans"

    r_tax = None
    if plot_tau_y_pct:
        try:
            r_tax = float((tax_table or {}).get("r_tax", None))
        except Exception:
            r_tax = None
        if r_tax is None or (not np.isfinite(r_tax)) or r_tax <= 0:
            raise ValueError(
                "plot_tau_y_pct=True requires a finite positive `tax_table['r_tax']` "
                f"(got r_tax={r_tax})."
            )

    brackets = list((tax_table or {}).get("brackets", []))
    if len(brackets) == 0:
        raise ValueError("tax_table['brackets'] is empty")

    # Parse + sort brackets
    parsed = []
    for b in brackets:
        if b is None:
            continue
        parsed.append(
            {
                "a0": float(b["a0"]),
                "a1": float(b["a1"]),
                "B": float(b["B"]),
                "tau_a": float(b["tau_a"]),
            }
        )
    parsed.sort(key=lambda d: d["a0"])

    if a_max is None:
        finite_ends = [b["a1"] for b in parsed if np.isfinite(b["a1"])]
        a_max = max(finite_ends) if finite_ends else (parsed[-1]["a0"] * 1.05)

    a_min = float(a_min)
    a_max = float(a_max)
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max <= a_min:
        raise ValueError(f"Invalid plot range: a_min={a_min}, a_max={a_max}")

    # Build contiguous segments over [a_min, a_max]
    segments = []
    for b in parsed:
        seg_start = max(a_min, b["a0"])
        seg_end = min(a_max, b["a1"]) if np.isfinite(b["a1"]) else a_max
        if seg_end <= seg_start:
            continue
        segments.append((seg_start, seg_end, b["tau_a"], b["a0"], b["B"]))
    if len(segments) == 0:
        raise ValueError("No tax brackets overlap the requested plot range.")

    # Dense evaluation for background + total tax line
    a_grid = np.linspace(a_min, a_max, int(n_bg))
    T_grid = np.full_like(a_grid, np.nan, dtype=float)
    tau_grid = np.full_like(a_grid, np.nan, dtype=float)

    for b in parsed:
        a0, a1, B, tau = b["a0"], b["a1"], b["B"], b["tau_a"]
        if np.isfinite(a1):
            mask = (a_grid >= a0) & (a_grid < a1)
        else:
            mask = a_grid >= a0
        if not np.any(mask):
            continue
        tau_grid[mask] = tau
        T_grid[mask] = B + tau * (a_grid[mask] - a0)

    # Detect discontinuous level jumps
    jumps = []
    for i in range(len(parsed) - 1):
        left = parsed[i]
        right = parsed[i + 1]
        a_jump = right["a0"]
        if a_jump < a_min or a_jump > a_max:
            continue
        T_left = left["B"] + left["tau_a"] * (a_jump - left["a0"])
        jump_size = right["B"] - T_left
        if abs(jump_size) > 1e-10:
            jumps.append(
                {
                    "a": a_jump,
                    "jump": float(jump_size),
                    "tau_before": float(left["tau_a"]),
                    "tau_after": float(right["tau_a"]),
                }
            )

    # Step path for marginal tax
    x_step = [segments[0][0]]
    y_step = [segments[0][2]]
    for i in range(len(segments) - 1):
        x_step.append(segments[i][1])
        y_step.append(segments[i + 1][2])
    x_step.append(segments[-1][1])
    y_step.append(segments[-1][2])

    # Convert marginal tax to an income-tax-equivalent rate on the return:
    # y = r_tax * a  =>  dT/dy = (dT/da) / r_tax.
    if plot_tau_y_pct:
        y_step_plot = [(tau / r_tax) * 100.0 for tau in y_step]
        y_mark_conv = lambda tau: (tau / r_tax) * 100.0  # noqa: E731
        y_label = "Marginal capital income tax (%)"
        y_fmt = FormatStrFormatter("%.1f")
    else:
        y_step_plot = list(y_step)
        y_mark_conv = lambda tau: tau  # noqa: E731
        y_label = r"Marginal asset tax $\tau_a(a)$"
        y_fmt = FormatStrFormatter("%.3f")

    # Total tax polyline (with vertical segments at jumps)
    x_tax = []
    y_tax = []
    for i, (seg_start, seg_end, tau, a0_ref, B_ref) in enumerate(segments):
        if len(x_tax) == 0:
            x_tax.append(seg_start)
            y_tax.append(B_ref + tau * (seg_start - a0_ref))
        x_tax.append(seg_end)
        y_tax.append(B_ref + tau * (seg_end - a0_ref))

        if i < len(segments) - 1:
            a_boundary = segments[i + 1][0]
            if abs(a_boundary - seg_end) < 1e-10:
                for j in jumps:
                    if abs(j["a"] - a_boundary) < 1e-10:
                        T_left = y_tax[-1]
                        T_right = T_left + j["jump"]
                        x_tax.extend([a_boundary, a_boundary])
                        y_tax.extend([T_left, T_right])
                        break

    # Paper style: match the paper plots (small ticks, DejaVu Sans).
    sns.set(style="white", rc={"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})
    plt.rcParams.update(
        {"axes.grid": True, "grid.color": "black", "grid.alpha": "0.25", "grid.linestyle": "--"}
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()

    # Total tax (secondary axis)
    ax2.plot(x_tax, y_tax, color="C0", linewidth=1.4, zorder=2)
    ax2.grid(False)  # No grid lines from secondary axis

    # Marginal tax (primary axis)
    ax.step(x_step, y_step_plot, where="post", color="black", linewidth=2.0, zorder=3)

    # Jump markers on marginal tax
    for j in jumps:
        y_mark = max(y_mark_conv(j["tau_before"]), y_mark_conv(j["tau_after"]))
        ax.scatter(
            [j["a"]],
            [y_mark],
            s=42,
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )
        if show_jump_lines:
            ax.axvline(j["a"], color="black", alpha=0.18, linewidth=0.9, zorder=1)

    ax.set_xlim(a_min, a_max)
    ax.set_ylim(bottom=0.0, top=max(y_step_plot) * 1.15)
    ax2.set_ylim(bottom=0.0, top=max(y_tax) * 1.05)

    ax.set_xlabel(r"Assets $a$", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax2.set_ylabel("Total tax", fontsize=11)

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(y_fmt)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    ax.grid(True, alpha=0.25)

    # Lighten axis spines (avoid bold black axes in the paper)
    _style_axis_spines(ax, sides=("left", "bottom"))
    _style_axis_spines(ax2, sides=("right",), color="0.65", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=9)
    ax2.tick_params(axis="y", labelsize=9)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax, ax2


def plot_asset_tax_schedule_from_yaml(
    master_yaml_path,
    schedule_key: str = "tax_table",
    a_min: float = 0.0,
    a_max=None,
    figsize=(6.5, 3.0),
    n_bg: int = 1500,
    show_jump_lines: bool = False,
    save_path=None,
):
    """Plot marginal and total asset tax implied by a YAML schedule.

    Produces a "paper-ready" figure with:
    - A step line for marginal asset tax (left axis)
    - A line for total tax paid (right axis)
    - Markers at discontinuous level jumps in total tax

    Parameters
    ----------
    master_yaml_path : str | pathlib.Path
        Path to the model master YAML (e.g. config_HR/.../master.yml).
    schedule_key : str
        Key under YAML["parameters"] that stores the asset tax schedule.
    a_min, a_max : float
        Asset range to plot. If a_max is None, we use max(settings.a_max, max finite a1 in schedule).
    figsize : tuple
        Figure size in inches.
    n_bg : int
        Resolution for the background gradient / total tax line evaluation.
    show_jump_lines : bool
        If True, draw faint vertical lines at jump points.
    save_path : str | None
        If provided, save the figure (png/pdf/etc.) to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        Primary axis (marginal tax).
    ax2 : matplotlib.axes.Axes
        Secondary axis (total tax).
    """
    import yaml
    with open(master_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    params = (cfg or {}).get("parameters", {})
    if schedule_key not in params:
        # Backward compatibility: older configs used a more specific key.
        for k in ("tax_table", "asset_tax_schedule_au_2015_16"):
            if k in params:
                schedule_key = k
                break
        else:
            raise KeyError(
                f"Missing tax schedule in `parameters` (tried `{schedule_key}`, "
                f"`tax_table`, `asset_tax_schedule_au_2015_16`) in {master_yaml_path}"
            )

    schedule = params[schedule_key] or {}
    if a_max is None:
        a_max_cfg = (cfg or {}).get("settings", {}).get("a_max", None)
        finite_ends = []
        for b in (schedule or {}).get("brackets", []):
            if b is None:
                continue
            try:
                a1 = float(b.get("a1", np.inf))
            except Exception:
                continue
            if np.isfinite(a1):
                finite_ends.append(a1)
        a_max_sched = max(finite_ends) if finite_ends else None
        candidates = [x for x in [a_max_cfg, a_max_sched] if x is not None]
        a_max = max(candidates) if candidates else None
    return plot_tax_table(
        schedule,
        a_min=a_min,
        a_max=a_max,
        figsize=figsize,
        n_bg=n_bg,
        show_jump_lines=show_jump_lines,
        save_path=save_path,
    )

