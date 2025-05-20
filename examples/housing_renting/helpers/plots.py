def generate_plots(model, method, image_dir):
    """
    Generate both EGM grid plots and policy function plots for a model using a specific method.
    
    Parameters
    ----------
    model : ModelCircuit
        Solved model circuit
    method : str
        Upper envelope method used (FUES, DCEGM, etc.)
    image_dir : str
        Directory to save the output images
    """
    # Import plotting functions
    import os
    
    # Base directory for this method
    method_dir = os.path.join(image_dir, method)
    
    # Create directories for different plot types
    egm_dir = os.path.join(method_dir, "egm_plots")
    policy_dir = os.path.join(method_dir, "policy_plots")
    os.makedirs(egm_dir, exist_ok=True)
    os.makedirs(policy_dir, exist_ok=True)
    
    # Generate EGM grid plots
    #print(f"\nGenerating EGM grid plots for {method}...")
    
    # Get the first period for EGM plots
    first_period = model.get_period(0)
    
    # For the owner consumption stage
    ownc_stage = first_period.get_stage("OWNC")
    
    # Generate plots for a specific housing and income state
    H_grid = ownc_stage.dcsn.grid.H_nxt
    
    # Select 3 housing values spread across the grid
    H_indices = [0, len(H_grid) // 2, len(H_grid) - 1]  # Low, middle, and high housing values
    y_idx = 0  # First income state
    
    # Plot EGM grid for three different housing values
    for H_idx in H_indices:
        plot_egm_grids(first_period, H_idx, y_idx, method, egm_dir)
    
    #print(f"EGM grid plots for {method} saved to {egm_dir}")
    
    # Generate policy function plots
    #print(f"\nGenerating policy function plots for {method}...")
    
    # Always use period 0 as requested
    period_to_plot = model.get_period(0)
    #print(f"Plotting period 0 policies for {method}...")
    
    # Plot policy functions
    plot_dcsn_policy(period_to_plot, policy_dir)
    #print(f"Policy function plots for {method} saved to {policy_dir}")

def plot_dcsn_policy(first_period, image_dir):
    """Plot policy functions for the renting model.
    
    Parameters
    ----------
    first_period : Period
        Period object containing stage data
    image_dir : str
        Directory to save the output images
    """
    # We'll create three plots:
    # 1. Consumption policies for owners and renters
    # 2. Housing choice policies
    # 3. Tenure choice policies
    
    import matplotlib.pyplot as plt
    import os
    import matplotlib.ticker as mticker
    from matplotlib.colors import TABLEAU_COLORS

    # Use a color cycle from Tableau colors
    color_cycle = list(TABLEAU_COLORS.values())[:3]  # Get first three colors

    tenu_stage = first_period.get_stage("TENU")
    ownh_stage = first_period.get_stage("OWNH") 
    ownc_stage = first_period.get_stage("OWNC")
    rnth_stage = first_period.get_stage("RNTH")
    rntc_stage = first_period.get_stage("RNTC")
    
    # 1. Consumption policies plot
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6))
    
    # Owner consumption policy - plot for different housing values
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
        owner_consumption = ownc_stage.dcsn.sol["policy"][:, H_idx, y_idx]
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
        renter_consumption = rntc_stage.dcsn.sol["policy"][:, S_idx, y_idx]
        ax1[1].plot(w_grid_rent, renter_consumption, color=color, linestyle='-', 
                   label=f"Services={s_val:.2f}")
    
    # Add 45-degree line for reference
    ax1[1].plot(w_grid_rent, w_grid_rent, 'k--', label="45-degree Line")
    ax1[1].set_title("Renter Consumption Policy by Service Level")
    ax1[1].set_xlabel("Cash-on-Hand (w)")
    ax1[1].set_ylabel("Consumption (c)")
    ax1[1].legend()
    ax1[1].grid(True)

    ax1[0].set_xlim([0, 15])
    ax1[1].set_xlim([0, 15])
    ax1[0].set_ylim([0, 10])
    ax1[1].set_ylim([0, 10])
    
    plt.tight_layout()
    fig1.savefig(os.path.join(image_dir, "consumption_policies.png"))
    #print(f"Consumption policies plot saved to {os.path.join(image_dir, 'consumption_policies.png')}")
    
    # 2. Housing choice policies plot
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
    
    # Owner housing policy - plot for different housing values
    a_grid = ownh_stage.dcsn.grid.a
    H_grid = ownh_stage.dcsn.grid.H
    
    # Get indices for low, medium, high current housing values
    if len(H_grid) >= 5:
        H_indices = [0, len(H_grid)//2, len(H_grid)-1]  # Low, medium, high
    else:
        H_indices = list(range(len(H_grid)))  # Use all available indices
    
    if "H_policy" in ownh_stage.dcsn.sol:
        # Plot housing policy for different current housing values
        for i, H_idx in enumerate(H_indices):
            h_val = H_grid[H_idx]
            color = color_cycle[i % len(color_cycle)]
            
            H_policy_idx = ownh_stage.dcsn.sol["H_policy"][:, H_idx, y_idx]
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
    w_grid_rent = rnth_stage.dcsn.grid.w
    y_grid = rnth_stage.dcsn.grid.y
    
    # Get indices for different income levels
    if len(y_grid) >= 3:
        y_indices = [0, len(y_grid)//2, len(y_grid)-1]  # Low, medium, high
    else:
        y_indices = list(range(len(y_grid)))  # Use all available indices
    
    if "S_policy" in rnth_stage.dcsn.sol:
        # Plot rental service policy for different income levels
        for i, y_idx in enumerate(y_indices):
            y_val = y_grid[y_idx]
            color = color_cycle[i % len(color_cycle)]
            
            S_policy_idx = rnth_stage.dcsn.sol["S_policy"][:, y_idx]
            S_grid = rnth_stage.cntn.grid.S
            S_policy = S_grid[S_policy_idx]
            
            ax2[1].step(w_grid_rent, S_policy, color=color, where='post', 
                       label=f"Income={y_val:.2f}")
            
        ax2[1].set_title("Renter Housing Services by Income")
        ax2[1].set_xlabel("Cash-on-Hand (w)")
        ax2[1].set_ylabel("Rental Services (S)")
        ax2[1].legend()
        ax2[1].grid(True)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(image_dir, "housing_policies.png"))
    #print(f"Housing policies plot saved to {os.path.join(image_dir, 'housing_policies.png')}")
    
    # 3. Tenure choice plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Extract tenure choice from TENU stage
    if "tenure_policy" in tenu_stage.dcsn.sol:
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
            
            tenure_policy = tenu_stage.dcsn.sol["tenure_policy"][:, H_idx, y_idx]
            
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
    
    plt.tight_layout()
    fig3.savefig(os.path.join(image_dir, "tenure_policy.png"))
    #print(f"Tenure policy plot saved to {os.path.join(image_dir, 'tenure_policy.png')}")
    
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
    if "EGM" not in ownc_stage.dcsn.sol or "EGM" not in rntc_stage.dcsn.sol:
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
        if grid_key in ownc_stage.dcsn.sol["EGM"]["unrefined"]["e"]:
            # Unrefined grids (empty circles)
            axes1[0].scatter(
                ownc_stage.dcsn.sol["EGM"]["unrefined"]["e"][grid_key],
                ownc_stage.dcsn.sol["EGM"]["unrefined"]["v"][grid_key],
                edgecolors=color, facecolors='none', s=30, label=f"Unrefined H={h_val:.2f}"
            )
            axes1[1].scatter(
                ownc_stage.dcsn.sol["EGM"]["unrefined"]["e"][grid_key],
                ownc_stage.dcsn.sol["EGM"]["unrefined"]["c"][grid_key],
                color=color, label=f"Unrefined H={h_val:.2f}"
            )
            
            # Refined grids (filled markers)
            axes1[0].plot(
                ownc_stage.dcsn.sol["EGM"]["refined"]["e"][grid_key],
                ownc_stage.dcsn.sol["EGM"]["refined"]["v"][grid_key],
                color='black', label=f"Refined H={h_val:.2f}"
            )
            axes1[1].plot(
                ownc_stage.dcsn.sol["EGM"]["refined"]["e"][grid_key],
                ownc_stage.dcsn.sol["EGM"]["refined"]["c"][grid_key],
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
                rntc_stage.dcsn.sol["EGM"]["unrefined"]["v"][grid_key],
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
                rntc_stage.dcsn.sol["EGM"]["refined"]["v"][grid_key],
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

def plot_egm_grids(period, H_idx, y_idx, method, image_dir):
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
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import matplotlib.ticker as mticker
    from matplotlib.colors import TABLEAU_COLORS
    import os
    
    # Get the owner consumption stage from the period
    ownc_stage = period.get_stage("OWNC")
    
    # Check if EGM grids are available
    if "EGM" not in ownc_stage.dcsn.sol:
        #print(f"EGM grid data not available for period {period.time_index}. Skipping plot.")
        return
    
    # Extract grid data
    grid_key = f"{y_idx}-{H_idx}"
    
    if grid_key not in ownc_stage.dcsn.sol["EGM"]["unrefined"]["e"]:
        #print(f"Grid key {grid_key} not found in EGM data. Available keys: {list(ownc_stage.dcsn.sol['EGM']['unrefined']['e'].keys())}")
        return
    
    # Unrefined grids
    e_grid_unrefined = ownc_stage.dcsn.sol["EGM"]["unrefined"]["e"][grid_key]
    vf_unrefined = ownc_stage.dcsn.sol["EGM"]["unrefined"]["v"][grid_key]
    c_unrefined = ownc_stage.dcsn.sol["EGM"]["unrefined"]["c"][grid_key]
    a_unrefined = ownc_stage.dcsn.sol["EGM"]["unrefined"]["a"][grid_key]
    h_unrefined = np.ones_like(e_grid_unrefined) * ownc_stage.dcsn.grid.H_nxt[H_idx]
    
    # Refined grids
    e_grid_refined = ownc_stage.dcsn.sol["EGM"]["refined"]["e"][grid_key]
    vf_refined = ownc_stage.dcsn.sol["EGM"]["refined"]["v"][grid_key]
    c_refined = ownc_stage.dcsn.sol["EGM"]["refined"]["c"][grid_key]
    a_refined = ownc_stage.dcsn.sol["EGM"]["refined"]["a"][grid_key]
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

