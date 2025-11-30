"""
Example: How to zoom in on specific regions for the highest housing index
"""

from helpers.plots import generate_plots

# Assuming you have 15 housing values (H_idx from 0 to 14)
# The highest H_idx would be 14

# Method 1: If you know the exact H_idx value
egm_bounds = {
    # For highest housing index (e.g., H_idx=14)
    'value_h14': (0, 5, 0, 3),      # x: 0-5, y: 0-3 for value function
    'assets_h14': (0, 5, None, None), # x: 0-5 for liquid savings (RHS panel)
}

# Generate plots only for the highest housing value
generate_plots(
    model=model,
    method="FUES",
    image_dir="output/plots",
    egm_bounds=egm_bounds
)


# Method 2: If you want to plot multiple H values but zoom only on the highest
egm_bounds = {
    # Default bounds for other H values
    'value': (0, 100, -10, 10),
    'assets': (0, 50, 0, 20),
    
    # Specific zoom for highest H_idx=14
    'value_h14': (0, 5, 0, 3),
    'assets_h14': (0, 5, None, None),
}

generate_plots(
    model=model,
    method="FUES",
    image_dir="output/plots",
    plot_all_H_idx=False,  # This will plot low, middle, high H values
    egm_bounds=egm_bounds
)


# Method 3: If you want even more specific control (e.g., different zoom for different income levels)
egm_bounds = {
    # Zoom settings for highest housing at all income levels
    'value_h14': (0, 5, 0, 3),
    'assets_h14': (0, 5, None, None),
    
    # Or even more specific: highest housing at specific income
    'value_y0_h14': (0, 5, 0, 2.5),   # Slightly different for low income
    'value_y2_h14': (0, 5, 0, 3.5),   # Slightly different for high income
}

generate_plots(
    model=model,
    method="FUES",
    image_dir="output/plots",
    y_idx_list=[0, 1, 2],  # Plot all income levels
    egm_bounds=egm_bounds
)


# Method 4: If you don't know the exact H_idx count, find it first
if model is not None:
    first_period = model.get_period(0)
    ownc_stage = first_period.get_stage("OWNC")
    H_grid = ownc_stage.dcsn.grid.H_nxt
    highest_H_idx = len(H_grid) - 1
    
    # Create bounds dynamically
    egm_bounds = {
        f'value_h{highest_H_idx}': (0, 5, 0, 3),
        f'assets_h{highest_H_idx}': (0, 5, None, None),
    }
    
    print(f"Setting zoom bounds for H_idx={highest_H_idx}")
    
    generate_plots(
        model=model,
        method="FUES",
        image_dir="output/plots",
        egm_bounds=egm_bounds
    )