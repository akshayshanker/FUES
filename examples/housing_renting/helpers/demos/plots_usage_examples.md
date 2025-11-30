# Plots.py Usage Examples

## New Features Added

### 1. Plot Multiple Income Indices (y_idx)
You can now plot EGM grids for multiple income levels by providing a list of indices:

```python
from helpers.plots import generate_plots

# Plot for all income levels (0, 1, 2)
generate_plots(model, method="FUES", image_dir="output/plots", 
               y_idx_list=[0, 1, 2])

# Plot for specific income levels
generate_plots(model, method="FUES", image_dir="output/plots", 
               y_idx_list=[0, 2])  # Only low and high income
```

### 2. Plot All Housing Values
Plot EGM grids for all H_idx values instead of just low/middle/high:

```python
# Plot all housing values
generate_plots(model, method="FUES", image_dir="output/plots", 
               plot_all_H_idx=True)

# Combine with multiple y_idx
generate_plots(model, method="FUES", image_dir="output/plots", 
               plot_all_H_idx=True, y_idx_list=[0, 1, 2])
```

### 3. Custom Axis Bounds for EGM Plots

The bounds system uses a hierarchy of specificity. More specific bounds override more general ones:
1. **Most specific**: Both y and h indices specified (e.g., `'value_y0_h5'`)
2. **Housing-specific**: Only h index specified (e.g., `'value_h5'`)
3. **Income-specific**: Only y index specified (e.g., `'value_y0'`)
4. **Global**: No indices specified (e.g., `'value'`)

#### Global bounds (apply to all plots of that type):
```python
egm_bounds = {
    'value': (0, 100, -10, 10),      # (xmin, xmax, ymin, ymax) for value function
    'assets': (0, 50, 0, 20),        # bounds for assets panel
    'he_space': (0, 20, 0, 100)      # bounds for h-e space plot
}

generate_plots(model, method="FUES", image_dir="output/plots", 
               egm_bounds=egm_bounds)
```

#### Income-specific bounds:
```python
egm_bounds = {
    'value_y0': (0, 50, -5, 5),      # bounds for y_idx=0 value plot
    'value_y1': (0, 100, -10, 10),   # bounds for y_idx=1 value plot
    'assets_y0': (0, 30, 0, 15),     # bounds for y_idx=0 assets plot
    'he_space_y2': (0, 25, 0, 120)   # bounds for y_idx=2 h-e space
}

generate_plots(model, method="FUES", image_dir="output/plots", 
               y_idx_list=[0, 1, 2], egm_bounds=egm_bounds)
```

#### Housing-specific bounds:
```python
egm_bounds = {
    'value_h0': (0, 30, -3, 3),      # bounds for H_idx=0 (low housing)
    'value_h10': (0, 200, -20, 20),  # bounds for H_idx=10 (high housing)
    'assets_h0': (0, 20, 0, 10),     # bounds for H_idx=0 assets
    'he_space_h5': (0, 50, 0, 150)   # bounds for H_idx=5 h-e space
}

generate_plots(model, method="FUES", image_dir="output/plots", 
               plot_all_H_idx=True, egm_bounds=egm_bounds)
```

#### Combined y and h specific bounds:
```python
egm_bounds = {
    # Global default
    'value': (0, 100, -10, 10),
    
    # Income-specific
    'value_y0': (0, 50, -5, 5),      # Low income: smaller range
    
    # Housing-specific  
    'value_h10': (0, 200, -20, 20),  # High housing: larger range
    
    # Both indices specific (highest priority)
    'value_y0_h0': (0, 20, -2, 2),   # Low income + low housing: very small
    'value_y2_h10': (0, 300, -30, 30), # High income + high housing: very large
    'assets_y1_h5': (0, 75, 0, 25),  # Middle income + middle housing
}

generate_plots(model, method="FUES", image_dir="output/plots", 
               y_idx_list=[0, 1, 2], plot_all_H_idx=True,
               egm_bounds=egm_bounds)
```

#### Mix global and specific bounds:
```python
egm_bounds = {
    # Global defaults
    'value': (0, 100, -10, 10),      
    'assets': (0, 50, 0, 20),        
    
    # Override for specific cases
    'value_y0': (0, 50, -5, 5),      # All H values at y=0 use this
    'value_h0': (0, 30, None, None),  # All y values at H=0 use this x-range
    'value_y0_h0': (0, 20, -2, 2),    # y=0, H=0 uses this (most specific)
}
```

### 4. Complete Example
```python
# Generate comprehensive plots with custom settings
generate_plots(
    model=model,
    method="FUES",
    image_dir="output/plots",
    plot_period=0,
    plot_all_H_idx=False,  # Use representative H values
    y_idx_list=[0, 1, 2],  # Plot all income levels
    egm_bounds={
        # Global bounds
        'value': (0, 100, None, None),    # Only set x-axis limits
        'assets': (None, None, 0, 30),    # Only set y-axis limits
        # Income-specific overrides
        'value_y0': (0, 50, -5, 5),       # Full bounds for low income
        'he_space_y2': (0, 20, 0, 100),   # Full bounds for high income h-e space
    },
    bounds={  # Policy function plot bounds (existing parameter)
        'consumption': (0, 50, 0, 10),
        'housing': (0, 50, 0, 20)
    }
)
```

## Notes
- Use `None` for any bound you don't want to set (e.g., `(None, 100, None, None)` sets only xmax)
- Income-specific bounds override global bounds when both are specified
- Legacy `egm_value` and `egm_assets` keys are still supported for backward compatibility
- Default behavior (no parameters) remains unchanged for backward compatibility