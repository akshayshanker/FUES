import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from dynx.stagecraft.solmaker import Solution

def generate_plots(model, method, image_dir, plot_period=0, bounds=None,
                  save_dir=None, load_dir=None):
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
        Dictionary containing bounds for plots
    save_dir : str, optional
        If provided and model is not None, save Solution objects to this directory
    load_dir : str, optional
        If provided and model is None, load Solution objects from this directory
        
    Notes
    -----
    If both save_dir and load_dir are None, behavior is unchanged (plots from live model).
    If model is None, load_dir must be provided to load saved solutions.
    """
    # Import plotting functions
    import os
    
    # Base directory for this method
    method_dir = os.path.join(image_dir, method)

    if load_dir is not None:
        method_dir_load = os.path.join(load_dir, method)
    else:
        method_dir_load = None

    if save_dir is not None:
        method_dir_save = os.path.join(save_dir, method)
    else:
        method_dir_save = None
    
    # Create directories for different plot types
    egm_dir = os.path.join(method_dir, "egm_plots")
    policy_dir = os.path.join(method_dir, "policy_plots")
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
    else:
        # Get solution from live model
        first_period = model.get_period(plot_period)
        ownc_stage = first_period.get_stage("OWNC")
        ownc_sol = ownc_stage.dcsn.sol
        
        # Save solution if requested
        if method_dir_save is not None:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{method_dir_save}/OWNC_dcsn_period{plot_period}"  # Remove .npz extension
            ownc_sol.save(fname)
            print(f"Saved Solution → {fname}.npz")
    
    # Generate EGM grid plots
    #print(f"\nGenerating EGM grid plots for {method}...")
    
    # Get the grid dimensions from the solution
    H_grid = ownc_stage.dcsn.grid.H_nxt if model else None  # TODO: Store grid in Solution
    
    # Select 3 housing values spread across the grid
    if H_grid is not None:
        H_indices = [0, len(H_grid) // 2, len(H_grid) - 1]  # Low, middle, and high housing values
    else:
        H_indices = [0, 1, 2]  # Default indices if grid not available
    y_idx = 0  # First income state
    
    # Plot EGM grid for three different housing values
    for H_idx in H_indices:
        plot_egm_grids(first_period if model else None, H_idx, y_idx, method, egm_dir, bounds, sol_override=ownc_sol)
    
    #print(f"EGM grid plots for {method} saved to {egm_dir}")
    
    # Generate policy function plots
    #print(f"\nGenerating policy function plots for {method}...")
    
    # Plot policy functions
    plot_dcsn_policy(first_period if model else None, policy_dir, bounds, sol_override=ownc_sol)
    #print(f"Policy function plots for {method} saved to {policy_dir}")

def plot_dcsn_policy(first_period, image_dir, bounds=None, sol_override=None):
    """Plot policy functions for the renting model.
    
    Parameters
    ----------
    first_period : Period
        Period object containing stage data
    image_dir : str
        Directory to save the output images
    bounds : dict
        Dictionary containing bounds for plots
    sol_override : dict, optional
        If provided, use this solution instead of the one from the first_period
    """
    # We'll create three plots:
    # 1. Consumption policies for owners and renters
    # 2. Housing choice policies
    # 3. Tenure choice policies
    
    import matplotlib.pyplot as plt
    import os
    import matplotlib.ticker as mticker
    from matplotlib.colors import TABLEAU_COLORS

    if bounds is None:
        bounds = {}

    # Use a color cycle from Tableau colors
    color_cycle = list(TABLEAU_COLORS.values())[:3]  # Get first three colors

    # Get stages - only get them if first_period is provided
    if first_period is not None:
        tenu_stage = first_period.get_stage("TENU")
        ownh_stage = first_period.get_stage("OWNH") 
        ownc_stage = first_period.get_stage("OWNC")
        rnth_stage = first_period.get_stage("RNTH")
        rntc_stage = first_period.get_stage("RNTC")
    else:
        # If no first_period, we must have sol_override
        if sol_override is None:
            raise ValueError("Either first_period or sol_override must be provided")
        # Create dummy stages with None values - we'll only use sol_override
        class DummyStage:
            def __init__(self):
                self.dcsn = type('obj', (object,), {'sol': None, 'grid': None})()
                self.cntn = type('obj', (object,), {'grid': None})()
        tenu_stage = ownh_stage = ownc_stage = rnth_stage = rntc_stage = DummyStage()
    
    # 1. Consumption policies plot
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6))
    
    # Owner consumption policy - plot for different housing values
    # Use sol_override only for OWNC stage
    ownc_sol = sol_override if sol_override is not None else (ownc_stage.dcsn.sol if ownc_stage.dcsn.sol is not None else None)
    
    if ownc_sol is not None and ownc_stage.dcsn.grid is not None:
        w_grid = ownc_stage.dcsn.grid.w
        H_nxt_grid = ownc_stage.dcsn.grid.H_nxt
        y_idx = 0  # Middle income state
        
        # Get indices for low, medium, high housing values
        if len(H_nxt_grid) >= 5:
            H_indices = [0, len(H_nxt_grid)//2, len(H_nxt_grid)-1]  # Low, medium, high
        else:
            H_indices = list(range(len(H_nxt_grid)))  # Use all available indices
            
        # Plot consumption policy for different housing values
        for i, H_idx in enumerate(H_indices):
            h_val = H_nxt_grid[H_idx]
            color = color_cycle[i % len(color_cycle)]
            owner_consumption = _get_sol_field(ownc_sol, "policy", "c")[:, H_idx, y_idx]
            ax1[0].plot(w_grid, owner_consumption, color=color, linestyle='-', 
                       label=f"Housing={h_val:.2f}")
        
        # Add 45-degree line for reference
        ax1[0].plot(w_grid, w_grid, 'k--', label="45-degree Line")
        ax1[0].set_title("Owner Consumption Policy by Housing Value")
        ax1[0].set_xlabel("Cash-on-Hand (w)")
        ax1[0].set_ylabel("Consumption (c)")
        ax1[0].legend()
        ax1[0].grid(True)
    
    # Renter consumption policy - plot for different rental service values
    # Don't use sol_override for RNTC stage
    if rntc_stage.dcsn.sol is not None and rntc_stage.dcsn.grid is not None:
        w_grid_rent = rntc_stage.dcsn.grid.w
        S_grid = rntc_stage.dcsn.grid.H_nxt
        
        # Get indices for low, medium, high rental service values
        if len(S_grid) >= 5:
            S_indices = [0, len(S_grid)//2, len(S_grid)-1]  # Low, medium, high
        else:
            S_indices = list(range(len(S_grid)))  # Use all available indices
        
        # Plot consumption policy for different rental service values
        for i, S_idx in enumerate(S_indices):
            s_val = S_grid[S_idx]
            color = color_cycle[i % len(color_cycle)]
            renter_consumption = _get_sol_field(rntc_stage.dcsn.sol, "policy", "c")[:, S_idx, y_idx]
            ax1[1].plot(w_grid_rent, renter_consumption, color=color, linestyle='-', 
                       label=f"Services={s_val:.2f}")
        
        # Add 45-degree line for reference
        ax1[1].plot(w_grid_rent, w_grid_rent, 'k--', label="45-degree Line")
        ax1[1].set_title("Renter Consumption Policy by Service Level")
        ax1[1].set_xlabel("Cash-on-Hand (w)")
        ax1[1].set_ylabel("Consumption (c)")
        ax1[1].legend()
        ax1[1].grid(True)

    # Allow user-specified axis limits via bounds dict
    # keys: 'cons_owner', 'cons_renter' each → (xmin,xmax,ymin,ymax)
    if "cons_owner" in bounds:
        xmin,xmax,ymin,ymax = bounds["cons_owner"]
        ax1[0].set_xlim([xmin,xmax]); ax1[0].set_ylim([ymin,ymax])
    if "cons_renter" in bounds:
        xmin,xmax,ymin,ymax = bounds["cons_renter"]
        ax1[1].set_xlim([xmin,xmax]); ax1[1].set_ylim([ymin,ymax])
    
    plt.tight_layout()
    fig1.savefig(os.path.join(image_dir, "consumption_policies.png"))
    #print(f"Consumption policies plot saved to {os.path.join(image_dir, 'consumption_policies.png')}")
    
    # 2. Housing choice policies plot
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
    
    # Owner housing policy - plot for different housing values
    # Don't use sol_override for OWNH stage
    if ownh_stage.dcsn.sol is not None and ownh_stage.dcsn.grid is not None:
        a_grid = ownh_stage.dcsn.grid.a
        H_grid = ownh_stage.dcsn.grid.H
        
        # Get indices for low, medium, high current housing values
        if len(H_grid) >= 5:
            H_indices = [0, len(H_grid)//2, len(H_grid)-1]  # Low, medium, high
        else:
            H_indices = list(range(len(H_grid)))  # Use all available indices
        
        if _get_sol_field(ownh_stage.dcsn.sol, "H_policy") is not None:
            # Plot housing policy for different current housing values
            for i, H_idx in enumerate(H_indices):
                h_val = H_grid[H_idx]
                color = color_cycle[i % len(color_cycle)]
                
                H_policy_idx = _get_sol_field(ownh_stage.dcsn.sol, "H_policy")[:, H_idx, y_idx].astype(int)
                H_nxt_grid = ownh_stage.cntn.grid.H_nxt
                H_policy = H_nxt_grid[H_policy_idx]
            
                ax2[0].step(a_grid, H_policy, color=color, where='post', 
                            label=f"Current H={h_val:.2f}")
                
            ax2[0].set_title("Owner Housing Policy by Current Housing")
            ax2[0].set_xlabel("Assets (a)")
            ax2[0].set_ylabel("Housing Choice (H)")
            ax2[0].legend()
            ax2[0].grid(True)
    
    # Renter housing (services) policy - plot for different income levels
    # Don't use sol_override for RNTH stage
    if rnth_stage.dcsn.sol is not None and rnth_stage.dcsn.grid is not None:
        w_grid_rent = rnth_stage.dcsn.grid.w
        y_grid = rnth_stage.dcsn.grid.y
        
        # Get indices for different income levels
        if len(y_grid) >= 3:
            y_indices = [0, len(y_grid)//2, len(y_grid)-1]  # Low, medium, high
        else:
            y_indices = list(range(len(y_grid)))  # Use all available indices
        
        if _get_sol_field(rnth_stage.dcsn.sol, "S_policy") is not None:
            # Plot rental service policy for different income levels
            for i, y_idx in enumerate(y_indices):
                y_val = y_grid[y_idx]
                color = color_cycle[i % len(color_cycle)]
                
                S_policy_idx = _get_sol_field(rnth_stage.dcsn.sol, "S_policy")[:, y_idx].astype(int)
                S_grid = rnth_stage.cntn.grid.S
                S_policy = S_grid[S_policy_idx]
                
                ax2[1].step(w_grid_rent, S_policy, color=color, where='post', 
                           label=f"Income={y_val:.2f}")
                
            ax2[1].set_title("Renter Housing Services by Income")
            ax2[1].set_xlabel("Cash-on-Hand (w)")
            ax2[1].set_ylabel("Rental Services (S)")
            ax2[1].legend()
            ax2[1].grid(True)
    
    # Owner housing limits
    if "owner_housing" in bounds:
        xmin,xmax,ymin,ymax = bounds["owner_housing"]
        ax2[0].set_xlim([xmin,xmax]); ax2[0].set_ylim([ymin,ymax])
    
    plt.tight_layout()
    fig2.savefig(os.path.join(image_dir, "housing_policies.png"))
    #print(f"Housing policies plot saved to {os.path.join(image_dir, 'housing_policies.png')}")
    
    # 3. Tenure choice plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Extract tenure choice from TENU stage
    # Don't use sol_override for TENU stage
    if tenu_stage.dcsn.sol is not None and tenu_stage.dcsn.grid is not None:
        if _get_sol_field(tenu_stage.dcsn.sol, "tenure_policy") is not None:
            a_grid = tenu_stage.dcsn.grid.a
            H_grid = tenu_stage.dcsn.grid.H
            
            # Get indices for different housing values
            if len(H_grid) >= 3:
                H_indices = [0, len(H_grid)//2, len(H_grid)-1]  # Low, medium, high
            else:
                H_indices = list(range(len(H_grid)))  # Use all available indices
            
            # Plot tenure choice for different housing values
            for i, H_idx in enumerate(H_indices):
                h_val = H_grid[H_idx]
                color = color_cycle[i % len(color_cycle)]
                
                tenure_policy = _get_sol_field(tenu_stage.dcsn.sol, "tenure_policy")[:, H_idx, y_idx]
                
                # Plot tenure choices as steps (0 = rent, 1 = own)
                ax3.step(a_grid, tenure_policy, color=color, where='post', linewidth=2,
                        label=f"Housing={h_val:.2f}")
            
            ax3.set_title("Tenure Choice Policy by Housing Value (0 = Rent, 1 = Own)")
            ax3.set_xlabel("Assets (a)")
            ax3.set_ylabel("Tenure Choice")
            
            # Set fixed ticks for y-axis with proper labels
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(["Rent", "Own"])
            
            ax3.legend()
            ax3.grid(True)
    
    # Tenure plot limits
    if "tenure" in bounds:
        xmin,xmax,ymin,ymax = bounds["tenure"]
        ax3.set_xlim([xmin,xmax]); ax3.set_ylim([ymin,ymax])
    
    plt.tight_layout()
    fig3.savefig(os.path.join(image_dir, "tenure_policy.png"))
    #print(f"Tenure policy plot saved to {os.path.join(image_dir, 'tenure_policy.png')}")

    # 4. Plot value function
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    # Extract value function from OWNC stage
    # Use sol_override for OWNC stage
    if ownc_sol is not None:
        if _get_sol_field(ownc_sol, "vlu") is not None and ownc_stage.dcsn.grid is not None:
            a_grid = ownc_stage.dcsn.grid.w
            H_grid = ownc_stage.dcsn.grid.H_nxt  # Use H_nxt_grid from OWNC stage
        
            # Plot value function for different housing values
            for i, H_idx in enumerate(H_indices):
                h_val = H_grid[H_idx]
                color = color_cycle[i % len(color_cycle)]
                
                value_function = _get_sol_field(ownc_sol, "vlu")[:, H_idx, y_idx]

                # plot
                ax4.plot(a_grid, value_function, color=color, label=f"Housing={h_val:.2f}")
                
            ax4.set_title("Owner Value Function by Housing Value")
            ax4.set_xlabel("Assets (a)")
            ax4.set_ylabel("Value Function (V)")
            ax4.legend()
            ax4.grid(True)

    #5. Plot Q func
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    # Extract Q function from OWNC stage
    # Use sol_override for OWNC stage
    if ownc_sol is not None:
        if _get_sol_field(ownc_sol, "Q") is not None and ownc_stage.dcsn.grid is not None:
            a_grid = ownc_stage.dcsn.grid.w
            H_grid = ownc_stage.dcsn.grid.H_nxt  # Use H_nxt_grid from OWNC stage
            
            # Plot Q function for different housing values
            for i, H_idx in enumerate(H_indices):
                h_val = H_grid[H_idx]
                color = color_cycle[i % len(color_cycle)]
                
                Q_function = _get_sol_field(ownc_sol, "Q")[:, H_idx, y_idx]
                
                # plot
                ax5.plot(a_grid, Q_function, color=color, label=f"Housing={h_val:.2f}")
                
            ax5.set_title("Owner Q Function by Housing Value")
            ax5.set_xlabel("Assets (a)")
            ax5.set_ylabel("Q Function (Q)")
            ax5.legend()
            ax5.grid(True)

    fig4.savefig(os.path.join(image_dir, "value_function.png"))
    fig5.savefig(os.path.join(image_dir, "Q_function.png"))

    # Plot savings
    fig6, ax6 = plt.subplots(figsize=(10, 6))

    # Use sol_override for OWNC stage
    if ownc_sol is not None and ownc_stage.dcsn.grid is not None:
        # No "if 'savings' …" guard here – we compute it ourselves
        for i, H_idx in enumerate(H_indices):
            h_val = H_grid[H_idx]
            color = color_cycle[i % len(color_cycle)]

            w_grid = ownc_stage.dcsn.grid.w          # cash-on-hand grid
            consumption = _get_sol_field(ownc_sol, "policy", "c")[:, H_idx, y_idx]
            savings = w_grid - consumption           # simple definition

            ax6.plot(w_grid, savings,
                    color=color,
                    label=f"Housing={h_val:.2f}")

        ax6.set_title("Owner Savings")
        ax6.set_xlabel("Cash-on-Hand (w)")
        ax6.set_ylabel("Savings (w – c)")
        ax6.legend()
        ax6.grid(True)

    fig6.savefig(os.path.join(image_dir, "savings.png"))
    
    # Close all figures
    plt.close('all')

def plot_endogenous_grids(first_period, image_dir):
    """Plot refined vs. unrefined endogenous grids for consumption stages.
    
    This function visualizes the endogenous grids before and after applying the 
    upper envelope method, helping to understand how the method refines the grids.
    
    Parameters
    ----------
    first_period : dict
        Dictionary containing the stages for the first period
    image_dir : str
        Directory to save the output images
    """
    # Get the owner and renter consumption stages
    ownc_stage = first_period.get_stage("OWNC")
    rntc_stage = first_period.get_stage("RNTC")
    
    # Check if EGM grids are available in the stages
    if _get_sol_field(ownc_stage.dcsn.sol, "EGM") is None or _get_sol_field(rntc_stage.dcsn.sol, "EGM") is None:
        print("EGM grid data not available. Please run the model with EGM solution method.")
        return
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import matplotlib.ticker as mticker
    from matplotlib.colors import TABLEAU_COLORS
    
    # Use a color cycle from Tableau colors
    color_cycle = list(TABLEAU_COLORS.values())[:3]  # Get first three colors
    
    # Set seaborn style
    sns.set(style="white", rc={"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})
    
    # 1. Plot unrefined vs refined grids for owner consumption
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6))
    
    # Set common styling for both subplots
    for ax in axes1:
        ax.set_xlabel('Cash-on-Hand (w)', fontsize=11)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True)
        # Use a proper formatter instead of manually setting tick labels
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    
    # Left plot: Value function
    axes1[0].set_ylabel('Value Function', fontsize=11)
    axes1[0].set_title("OWNC: Value Function Grids", fontsize=11)
    
    # Right plot: Consumption policy
    axes1[1].set_ylabel('Consumption (c)', fontsize=11)
    axes1[1].set_title("OWNC: Consumption Policy Grids", fontsize=11)
    
    # Income index for plotting (middle income level)
    y_idx = 1
    
    # Plot for a few representative housing values
    H_nxt_grid = ownc_stage.dcsn.grid.H_nxt
    if len(H_nxt_grid) >= 5:
        H_indices = [0, len(H_nxt_grid)//2, len(H_nxt_grid)-1]  # Low, medium, high
    else:
        H_indices = list(range(len(H_nxt_grid)))  # Use all available indices
    
    # Plot owner consumption grids
    for i, H_idx in enumerate(H_indices):
        h_val = H_nxt_grid[H_idx]
        color = color_cycle[i % len(color_cycle)]
        grid_key = f"{y_idx}-{H_idx}"
        
        # Check if this grid key exists
        unrefined_e = _get_sol_field(ownc_stage.dcsn.sol, "EGM", "unrefined.e")
        if grid_key in unrefined_e:
            # Unrefined grids (empty circles)
            axes1[0].scatter(
                            _get_sol_field(ownc_stage.dcsn.sol, "EGM", "unrefined.e")[grid_key],
            _get_sol_field(ownc_stage.dcsn.sol, "EGM", "unrefined.Q")[grid_key],
                edgecolors=color, facecolors='none', s=30, label=f"Unrefined H={h_val:.2f}"
            )
            axes1[1].scatter(
                            _get_sol_field(ownc_stage.dcsn.sol, "EGM", "unrefined.e")[grid_key],
            _get_sol_field(ownc_stage.dcsn.sol, "EGM", "unrefined.c")[grid_key],
                color=color, label=f"Unrefined H={h_val:.2f}"
            )
            
            # Refined grids (filled markers)
            axes1[0].plot(
                            _get_sol_field(ownc_stage.dcsn.sol, "EGM", "refined.e")[grid_key],
            _get_sol_field(ownc_stage.dcsn.sol, "EGM", "refined.Q")[grid_key],
                color='black', label=f"Refined H={h_val:.2f}"
            )
            axes1[1].plot(
                            _get_sol_field(ownc_stage.dcsn.sol, "EGM", "refined.e")[grid_key],
            _get_sol_field(ownc_stage.dcsn.sol, "EGM", "refined.c")[grid_key],
                color='black', label=f"Refined H={h_val:.2f}"
            )
    
    # Add legend to the first subplot only
    handles, labels = axes1[0].get_legend_handles_labels()
    
    # Filter to show only one set of labels per housing value
    filtered_handles = []
    filtered_labels = []
    seen_h_values = set()
    
    for h, l in zip(handles, labels):
        if "Unrefined" in l:
            h_value = l.split("=")[1]
            if h_value not in seen_h_values:
                seen_h_values.add(h_value)
                filtered_handles.append(h)
                filtered_labels.append(l)
        elif "Refined" in l and l.split("=")[1] in seen_h_values:
            filtered_handles.append(h)
            filtered_labels.append(l)
    
    axes1[0].legend(filtered_handles, filtered_labels, loc='best', frameon=False, prop={'size': 9})
    
    plt.tight_layout()
    fig1.savefig(os.path.join(image_dir, "ownc_endogenous_grids.png"))
    #print(f"Owner consumption endogenous grids plot saved to {os.path.join(image_dir, 'ownc_endogenous_grids.png')}")
    
    # 2. Plot unrefined vs refined grids for renter consumption
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
    
    # Set common styling for both subplots
    for ax in axes2:
        ax.set_xlabel('Cash-on-Hand (w)', fontsize=11)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True)
        # Use a proper formatter instead of manually setting tick labels
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    
    # Left plot: Value function
    axes2[0].set_ylabel('Value Function', fontsize=11)
    axes2[0].set_title("RNTC: Value Function Grids", fontsize=11)
    
    # Right plot: Consumption policy
    axes2[1].set_ylabel('Consumption (c)', fontsize=11)
    axes2[1].set_title("RNTC: Consumption Policy Grids", fontsize=11)
    
    # Income index for plotting (middle income level)
    y_idx = 1
    
    # Plot for a few representative service values
    S_grid = rntc_stage.dcsn.grid.H_nxt  # Note: H_nxt is used for rental services
    if len(S_grid) >= 5:
        S_indices = [0, len(S_grid)//2, len(S_grid)-1]  # Low, medium, high
    else:
        S_indices = list(range(len(S_grid)))  # Use all available indices
    
    # Plot renter consumption grids
    for i, S_idx in enumerate(S_indices):
        s_val = S_grid[S_idx]
        color = color_cycle[i % len(color_cycle)]
        grid_key = f"{y_idx}-{S_idx}"
        
        # Check if this grid key exists
        if grid_key in rntc_stage.dcsn.sol["EGM"]["unrefined"]["e"]:
            # Unrefined grids (empty circles)
            axes2[0].scatter(
                rntc_stage.dcsn.sol["EGM"]["unrefined"]["e"][grid_key],
                rntc_stage.dcsn.sol["EGM"]["unrefined"]["Q"][grid_key],
                edgecolors=color, facecolors='none', s=30, label=f"Unrefined S={s_val:.2f}"
            )
            axes2[1].scatter(
                rntc_stage.dcsn.sol["EGM"]["unrefined"]["e"][grid_key],
                rntc_stage.dcsn.sol["EGM"]["unrefined"]["c"][grid_key],
                edgecolors=color, facecolors='none', s=30, label=f"Unrefined S={s_val:.2f}"
            )
            
            # Refined grids (filled markers)
            axes2[0].scatter(
                rntc_stage.dcsn.sol["EGM"]["refined"]["e"][grid_key],
                rntc_stage.dcsn.sol["EGM"]["refined"]["Q"][grid_key],
                color=color, marker='x', s=30, label=f"Refined S={s_val:.2f}"
            )
            axes2[1].scatter(
                rntc_stage.dcsn.sol["EGM"]["refined"]["e"][grid_key],
                rntc_stage.dcsn.sol["EGM"]["refined"]["c"][grid_key],
                color=color, marker='x', s=30, label=f"Refined S={s_val:.2f}"
            )
    
    # Add legend to the first subplot only
    handles, labels = axes2[0].get_legend_handles_labels()
    
    # Filter to show only one set of labels per service value
    filtered_handles = []
    filtered_labels = []
    seen_s_values = set()
    
    for h, l in zip(handles, labels):
        if "Unrefined" in l:
            s_value = l.split("=")[1]
            if s_value not in seen_s_values:
                seen_s_values.add(s_value)
                filtered_handles.append(h)
                filtered_labels.append(l)
        elif "Refined" in l and l.split("=")[1] in seen_s_values:
            filtered_handles.append(h)
            filtered_labels.append(l)
    
    axes2[0].legend(filtered_handles, filtered_labels, loc='best', frameon=False, prop={'size': 9})
    
    plt.tight_layout()
    fig2.savefig(os.path.join(image_dir, "rntc_endogenous_grids.png"))
    #print(f"Renter consumption endogenous grids plot saved to {os.path.join(image_dir, 'rntc_endogenous_grids.png')}")
    
    # Close all figures
    plt.close('all') 

def plot_egm_grids(period, H_idx, y_idx, method, image_dir, bounds=None, sol_override=None):
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
        Dictionary containing bounds for plots
    sol_override : dict, optional
        If provided, use this solution instead of the one from the period
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import matplotlib.ticker as mticker
    from matplotlib.colors import TABLEAU_COLORS
    import os
    
    if bounds is None:
        bounds = {}

    # Get the owner consumption stage from the period
    if period is not None:
        ownc_stage = period.get_stage("OWNC")
    else:
        # If no period, we need sol_override
        if sol_override is None:
            # No data to plot
            return
        ownc_stage = None
    
    # Check if EGM grids are available
    egm_data = _get_sol_field(sol_override if sol_override else (ownc_stage.dcsn.sol if ownc_stage else None), "EGM")
    if egm_data is None or not egm_data:
        #print(f"EGM grid data not available for period {period.time_index}. Skipping plot.")
        return
    
    # Extract grid data
    grid_key = f"{y_idx}-{H_idx}"
    
    unrefined_e = _get_sol_field(sol_override or ownc_stage.dcsn.sol, "EGM", "unrefined.e")
    if unrefined_e is None or grid_key not in unrefined_e:
        #print(f"Grid key {grid_key} not found in EGM data. Available keys: {list(unrefined_e.keys()) if unrefined_e else []}")
        return
    
    # Unrefined grids
    e_grid_unrefined = _get_sol_field(sol_override or ownc_stage.dcsn.sol, "EGM", "unrefined.e")[grid_key]
    vf_unrefined = _get_sol_field(sol_override or ownc_stage.dcsn.sol, "EGM", "unrefined.Q")[grid_key]
    c_unrefined = _get_sol_field(sol_override or ownc_stage.dcsn.sol, "EGM", "unrefined.c")[grid_key]
    a_unrefined = _get_sol_field(sol_override or ownc_stage.dcsn.sol, "EGM", "unrefined.a")[grid_key]
    
    # Skip if we don't have grid information
    if ownc_stage is None:
        # Can't plot without grid information
        return
    
    h_unrefined = np.ones_like(e_grid_unrefined) * ownc_stage.dcsn.grid.H_nxt[H_idx]
    
    # Refined grids
    e_grid_refined = _get_sol_field(sol_override or ownc_stage.dcsn.sol, "EGM", "refined.e")[grid_key]
    vf_refined = _get_sol_field(sol_override or ownc_stage.dcsn.sol, "EGM", "refined.Q")[grid_key]
    c_refined = _get_sol_field(sol_override or ownc_stage.dcsn.sol, "EGM", "refined.c")[grid_key]
    a_refined = _get_sol_field(sol_override or ownc_stage.dcsn.sol, "EGM", "refined.a")[grid_key]
    h_refined = np.ones_like(e_grid_refined) * ownc_stage.dcsn.grid.H_nxt[H_idx]
    
    # Get housing and income values for labels
    h_value = ownc_stage.dcsn.grid.H_nxt[H_idx]
    y_value = y_idx  # This is the index, not the actual value
    
    # Set up seaborn style similar to example_egm_plot.py
    sns.set(style="white", rc={"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})
    
    # Create a two-panel figure: value function and housing choice
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
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
    
    ax[0].plot(
        e_grid_refined[1:],
        vf_refined[1:],
        color=palette[2],
        linewidth=1,
        label='Value function'
    )
    
    # Apply custom bounds for value panel
    if "egm_value" in bounds:
        xmin,xmax,ymin,ymax = bounds["egm_value"]
        ax[0].set_xlim([xmin,xmax]); ax[0].set_ylim([ymin,ymax])
    
    # Formatting for value function plot
    ax[0].set_ylabel('Value', fontsize=11)
    ax[0].set_xlabel(r'Total wealth at time $t$', fontsize=11)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].legend(frameon=False, prop={'size': 10})
    ax[0].grid(True)
    
    # Use MaxNLocator to set a fixed number of ticks
    ax[0].yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax[0].xaxis.set_major_locator(mticker.MaxNLocator(6))
    
    # Set formatters after setting locators
    ax[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
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
    if "egm_assets" in bounds:
        xmin,xmax,ymin,ymax = bounds["egm_assets"]
        ax[1].set_xlim([xmin,xmax]); ax[1].set_ylim([ymin,ymax])
    
    # Formatting for housing choice plot
    ax[1].set_ylabel(r'Housing assets at time $t+1$', fontsize=11)
    ax[1].set_xlabel(r'Total wealth at time $t$', fontsize=11)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].grid(True)
    
    # Use MaxNLocator to set a fixed number of ticks
    ax[1].yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax[1].xaxis.set_major_locator(mticker.MaxNLocator(6))
    
    # Set formatters after setting locators
    ax[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax[1].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    
    # Add title with method, housing value, and income index
    fig.suptitle(f"{method} Upper Envelope: H={h_value:.2f}, Income Index={y_value}", fontsize=12)
    
    # Tight layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    # Save figure
    filename = f"egm_grid_H{H_idx}_y{y_idx}_period{period.time_index}_{method}.png"
    fig.savefig(os.path.join(image_dir, filename))
    #print(f"EGM grid plot saved to {os.path.join(image_dir, filename)}")
    
    # Close figure
    plt.close(fig)
    
    # Now create an additional plot showing endogenous grids in the h-e space (like the third plot in example_egm_plot.py)
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Set style
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
    
    # Use locators before formatters to ensure fixed number of ticks
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    
    # Add title with method, housing value, and income index
    ax2.set_title(f"{method} Upper Envelope: H={h_value:.2f}, Income Index={y_value}", fontsize=12)
    
    # Save figure
    filename2 = f"egm_grid_he_space_H{H_idx}_y{y_idx}_period{period.time_index}_{method}.png"
    fig2.savefig(os.path.join(image_dir, filename2))
    #print(f"EGM grid h-e space plot saved to {os.path.join(image_dir, filename2)}")
    
    # Close figure
    plt.close(fig2)

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
    import os
    import matplotlib.pyplot as plt
    from matplotlib.colors import TABLEAU_COLORS

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

def _get_sol_field(sol, field, subfield=None):
    """Helper to access solution fields from either dict or Solution object.
    
    Parameters
    ----------
    sol : dict or Solution
        Solution object
    field : str
        Main field name (e.g., 'policy', 'vlu', 'EGM')
    subfield : str, optional
        Subfield name for nested access (e.g., 'c' for policy.c)
        
    Returns
    -------
    array or dict
        The requested field data
    """
    if isinstance(sol, Solution):
        if field == "policy":
            return sol.policy[subfield] if subfield else dict(sol.policy)
        elif field == "EGM":
            if subfield:
                # Handle nested EGM access like EGM.refined.e
                parts = subfield.split('.')
                if len(parts) == 2:  # e.g., "refined.e"
                    layer, grid_type = parts
                    # Return dict of all keys for this grid type
                    result = {}
                    layer_dict = getattr(sol.EGM, layer)
                    prefix = f"{grid_type}_"
                    for key, val in layer_dict.items():
                        if key.startswith(prefix):
                            grid_key = key[len(prefix):]  # Remove prefix
                            result[grid_key] = val
                    return result
                else:
                    result = sol.EGM
                    for part in parts:
                        result = getattr(result, part) if hasattr(result, part) else result[part]
                    return result
            else:
                # Return full dict representation with reconstructed structure
                egm_dict = {}
                for layer in ["unrefined", "refined", "interpolated"]:
                    layer_data = {}
                    for grid_type in ["e", "Q", "c", "a", "lambda"]:
                        grid_data = {}
                        prefix = f"{grid_type}_"
                        layer_dict = getattr(sol.EGM, layer)
                        for key, val in layer_dict.items():
                            if key.startswith(prefix):
                                grid_key = key[len(prefix):]
                                grid_data[grid_key] = val
                        if grid_data:
                            layer_data[grid_type] = grid_data
                    egm_dict[layer] = layer_data
                return egm_dict
        elif field == "timing":
            return dict(sol.timing)
        elif field in ["vlu", "Q", "lambda", "phi"]:
            return getattr(sol, field.replace("lambda", "lambda_"))
        elif field == "H_policy":
            return sol.policy["H"]
        elif field == "S_policy":
            return sol.policy["S"]
        elif field == "tenure_policy":
            return sol.policy["tenure"]
        else:
            return sol[field] if hasattr(sol, '__getitem__') else getattr(sol, field)
    else:
        # Legacy dict access
        if subfield and field in sol and isinstance(sol[field], dict):
            return sol[field][subfield]
        else:
            return sol.get(field)

