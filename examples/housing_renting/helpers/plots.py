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
    """Plot policy functions for the renting model, matching fella plot style."""
    
    import matplotlib.pyplot as plt
    import os
    import matplotlib.ticker as mticker
    from matplotlib.ticker import FormatStrFormatter
    import seaborn as sns

    if bounds is None:
        bounds = {}

    # Set seaborn style to match plot_pols_fella
    sns.set(style="white", rc={"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})
    color_cycle = ['blue', 'red', 'green']  # Use distinct colors like fella plot

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
    
    # --- 1. Consumption policies plot ---
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6))
    for ax in ax1:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ownc_sol = sol_override if sol_override is not None else (ownc_stage.dcsn.sol if ownc_stage.dcsn.sol is not None else None)
    y_idx = 0 # Assume middle income state for simplicity
    
    if ownc_sol and ownc_stage.dcsn.grid:
        w_grid = ownc_stage.dcsn.grid.w
        H_nxt_grid = ownc_stage.dcsn.grid.H_nxt
        H_indices = [0, len(H_nxt_grid)//2, len(H_nxt_grid)-1] if len(H_nxt_grid) >= 3 else list(range(len(H_nxt_grid)))
            
        for i, H_idx in enumerate(H_indices):
            h_val = H_nxt_grid[H_idx]
            color = color_cycle[i % len(color_cycle)]
            owner_consumption = _get_sol_field(ownc_sol, "policy", "c")[:, H_idx, y_idx]
            ax1[0].plot(w_grid, owner_consumption, color=color, linestyle='-', label=f"H={h_val:.2f}")
        
        ax1[0].set_title("Owner Consumption")
        ax1[0].set_ylabel("Consumption (c)")
        ax1[0].set_xlabel("Cash-on-Hand (w)")
        ax1[0].legend(frameon=False, prop={'size': 10})
    
    if rntc_stage.dcsn.sol and rntc_stage.dcsn.grid:
        w_grid_rent = rntc_stage.dcsn.grid.w
        S_grid = rntc_stage.dcsn.grid.H_nxt
        S_indices = [0, len(S_grid)//2, len(S_grid)-1] if len(S_grid) >= 3 else list(range(len(S_grid)))
        
        for i, S_idx in enumerate(S_indices):
            s_val = S_grid[S_idx]
            color = color_cycle[i % len(color_cycle)]
            renter_consumption = _get_sol_field(rntc_stage.dcsn.sol, "policy", "c")[:, S_idx, y_idx]
            ax1[1].plot(w_grid_rent, renter_consumption, color=color, linestyle='-', label=f"S={s_val:.2f}")
        
        ax1[1].set_title("Renter Consumption")
        ax1[1].set_xlabel("Cash-on-Hand (w)")
        ax1[1].legend(frameon=False, prop={'size': 10})

    fig1.tight_layout()
    fig1.savefig(os.path.join(image_dir, "consumption_policies.png"))

    # --- 2. Housing choice policies plot ---
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
    for ax in ax2:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True)

    if ownh_stage.dcsn.sol and ownh_stage.dcsn.grid:
        a_grid = ownh_stage.dcsn.grid.a
        H_grid = ownh_stage.dcsn.grid.H
        H_indices = [0, len(H_grid)//2, len(H_grid)-1] if len(H_grid) >= 3 else list(range(len(H_grid)))
        
        H_policy_data = _get_sol_field(ownh_stage.dcsn.sol, "policy", "H")
        if H_policy_data is not None:
            H_nxt_grid = ownh_stage.cntn.grid.H_nxt
            for i, H_idx in enumerate(H_indices):
                h_val = H_grid[H_idx]
                color = color_cycle[i % len(color_cycle)]
                H_policy_idx = H_policy_data[:, H_idx, y_idx].astype(int)
                H_policy = H_nxt_grid[H_policy_idx]
                ax2[0].step(a_grid, H_policy, color=color, where='post', label=f"Current H={h_val:.2f}")
            
            ax2[0].set_title("Owner Housing Policy")
            ax2[0].set_xlabel("Assets (a)")
            ax2[0].set_ylabel("Housing Choice (H')")
            ax2[0].legend(frameon=False, prop={'size': 10})
    
    if rnth_stage.dcsn.sol and rnth_stage.dcsn.grid:
        w_grid_rent = rnth_stage.dcsn.grid.w
        S_policy_data = _get_sol_field(rnth_stage.dcsn.sol, "policy", "S")
        if S_policy_data is not None:
            S_grid = rnth_stage.cntn.grid.S
            S_policy_idx = S_policy_data[:, y_idx].astype(int)
            S_policy = S_grid[S_policy_idx]
            ax2[1].step(w_grid_rent, S_policy, where='post')
            ax2[1].set_title("Renter Service Choice")
            ax2[1].set_xlabel("Cash-on-Hand (w)")
            ax2[1].set_ylabel("Rental Services (S)")
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(image_dir, "housing_policies.png"))

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
            H_indices = [0, len(H_grid)//2, len(H_grid)-1] if len(H_grid) >= 3 else list(range(len(H_grid)))
            
            for i, H_idx in enumerate(H_indices):
                h_val = H_grid[H_idx]
                color = color_cycle[i % len(color_cycle)]
                tenure_policy = tenure_policy_data[:, H_idx, y_idx]
                ax3.step(a_grid, tenure_policy, color=color, where='post', linewidth=2, label=f"H={h_val:.2f}")
            
            ax3.set_title("Tenure Choice (0=Rent, 1=Own)")
            ax3.set_xlabel("Assets (a)")
            ax3.set_ylabel("Tenure Choice")
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(["Rent", "Own"])
            ax3.legend(frameon=False, prop={'size': 10})

    fig3.tight_layout()
    fig3.savefig(os.path.join(image_dir, "tenure_policy.png"))

    # --- 4. Value and Q Function Plots ---
    fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 6))
    for ax in [ax4, ax5]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True)

    if ownc_sol and ownc_stage.dcsn.grid:
        w_grid = ownc_stage.dcsn.grid.w
        H_nxt_grid = ownc_stage.dcsn.grid.H_nxt
        H_indices = [0, len(H_nxt_grid)//2, len(H_nxt_grid)-1] if len(H_nxt_grid) >= 3 else list(range(len(H_nxt_grid)))

        vlu_data = _get_sol_field(ownc_sol, "vlu")
        q_data = _get_sol_field(ownc_sol, "Q")

        for i, H_idx in enumerate(H_indices):
            h_val = H_nxt_grid[H_idx]
            color = color_cycle[i % len(color_cycle)]
            label=f"H={h_val:.2f}"
            if vlu_data is not None:
                ax4.plot(w_grid, vlu_data[:, H_idx, y_idx], color=color, label=label)
            if q_data is not None:
                ax5.plot(w_grid, q_data[:, H_idx, y_idx], color=color, label=label)

    ax4.set_title("Owner Value Function")
    ax4.set_xlabel("Cash-on-Hand (w)")
    ax4.set_ylabel("Value")
    ax4.legend(frameon=False, prop={'size': 10})

    ax5.set_title("Owner Q-Function")
    ax5.set_xlabel("Cash-on-Hand (w)")
    ax5.legend(frameon=False, prop={'size': 10})
    
    fig4.tight_layout()
    fig4.savefig(os.path.join(image_dir, "value_q_functions.png"))

    # --- 5. Savings Plot ---
    fig5, ax6 = plt.subplots(figsize=(10, 6))
    ax6.spines['right'].set_visible(False)
    ax6.spines['top'].set_visible(False)
    ax6.grid(True)

    if ownc_sol and ownc_stage.dcsn.grid:
        H_nxt_grid = ownc_stage.dcsn.grid.H_nxt
        H_indices = [0, len(H_nxt_grid)//2, len(H_nxt_grid)-1] if len(H_nxt_grid) >= 3 else list(range(len(H_nxt_grid)))

        for i, H_idx in enumerate(H_indices):
            h_val = H_nxt_grid[H_idx]
            color = color_cycle[i % len(color_cycle)]
            w_grid = ownc_stage.dcsn.grid.w
            consumption = _get_sol_field(ownc_sol, "policy", "c")[:, H_idx, y_idx]
            savings = w_grid - consumption
            ax6.plot(w_grid, savings, color=color, label=f"H={h_val:.2f}")

        ax6.set_title("Owner Savings (w-c)")
        ax6.set_xlabel("Cash-on-Hand (w)")
        ax6.set_ylabel("Savings")
        ax6.legend(frameon=False, prop={'size': 10})

    fig5.tight_layout()
    fig5.savefig(os.path.join(image_dir, "savings.png"))
    
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

# Add debugging information to plot_egm_grids function
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

