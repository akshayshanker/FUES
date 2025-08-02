# Plotting Comparison System for Housing-Renting Model

This document describes how to use the flexible plotting comparison system implemented for the housing-renting model in the FUES framework.

## Overview

The plotting system automatically generates 2D plots comparing policy and value functions of different "fast" solution methods against a high-definition baseline model. The system is configurable and integrated into the main `solve_runner.py` workflow.

## Key Features

- **Automatic comparison plots**: Generates difference plots between fast methods (FUES, CONSAV, etc.) and the baseline (VFI_HDGRID)
- **Configurable state-space slices**: Choose which indices of the state space to plot
- **Multiple variables**: Can plot both consumption policies (`c`) and value functions (`v`)
- **Clean integration**: Seamlessly integrated into the existing CircuitRunner workflow

## Usage

### Basic Usage

To enable plotting, simply add the `--plots` flag to your `solve_runner.py` command:

```bash
python3 -m examples.housing_renting.solve_runner \
    --periods 3 \
    --ue-method VFI_HDGRID,FUES2DEV,CONSAV \
    --plots \
    --output-root solutions/HR \
    --bundle-prefix HR \
    --vfi-ngrid 1000 \
    --HD-points 1000 \
    --grid-points 500 \
    --recompute-baseline \
    --fresh-fast
```

### Configuration

The plotting system is configured in `solve_runner.py` with the following parameters:

#### Dimension Labels
```python
asset_dims = {
    0: 'k_idx',    # Physical capital (housing)
    1: 'a_idx',    # Liquid assets  
    2: 'choice'    # The decision/choice axis
}
```

#### Slice Configuration
```python
plots_of_interest = {
    'k_idx': [5, 10, 15]  # Generate plots only for these k indices
}
```

#### Available Metrics
- `plot_c_comparison`: Consumption policy comparison plots
- `plot_v_comparison`: Value function comparison plots

### Output

When plotting is enabled, the system generates:

1. **Baseline plots**: Using the existing `generate_plots()` function
2. **Comparison plots**: Difference plots for each fast method vs. baseline

Plots are saved to: `{output_root}/images/`

#### Example Output Files
- `c_diff_FUES2DEV_k_idx=5.png`: Consumption difference for FUES2DEV vs baseline at k_idx=5
- `c_diff_FUES2DEV_k_idx=10.png`: Consumption difference for FUES2DEV vs baseline at k_idx=10
- `v_diff_CONSAV_k_idx=5.png`: Value function difference for CONSAV vs baseline at k_idx=5

## Customization

### Adding New Variables to Plot

To plot additional variables (e.g., savings policy `a`), add new metrics to the configuration:

```python
if args.plots:
    metric_fns.update({
        "plot_a_comparison": plot_comparison_factory(
            decision_variable='a',
            dim_labels=asset_dims,
            plot_axis_label='a_idx',
            slice_config=plots_of_interest
        ),
    })
```

### Changing Plot Configurations

Modify the `plots_of_interest` dictionary to change which state-space slices are plotted:

```python
plots_of_interest = {
    'k_idx': [0, 5, 10, 15, 20],  # More k indices
    'a_idx': [10, 20]             # Also slice on a_idx
}
```

### Different Plot Axis

Change the `plot_axis_label` to plot along a different dimension:

```python
"plot_c_comparison": plot_comparison_factory(
    decision_variable='c',
    dim_labels=asset_dims,
    plot_axis_label='k_idx',  # Plot along k instead of a
    slice_config={'a_idx': [10, 20]}
),
```

## Testing

A test script is provided to verify the functionality:

```bash
cd FUES/examples/housing_renting
python test_plotting.py
```

This runs a small example and generates test plots in `test_solutions/HR_plot_test/images/`.

## Implementation Details

### Architecture

1. **Factory Pattern**: `plot_comparison_factory()` creates configurable plotting metrics
2. **Integration**: Plotting metrics are added to CircuitRunner's `metric_fns`
3. **Data Extraction**: Uses existing `_extract_policy()` function for robustness
4. **Memory Management**: Baseline model is stored temporarily and cleaned up after use

### Key Components

- `helpers/metrics.py`: Contains `plot_comparison_factory()` and plotting logic
- `solve_runner.py`: Configuration and integration with CircuitRunner
- Policy extraction via `_extract_policy()` ensures compatibility with existing codebase

### Memory Considerations

- Baseline model is stored only temporarily for plotting comparisons
- Automatic cleanup prevents memory leaks
- Individual method models are processed and deleted immediately after plotting

## Troubleshooting

### Common Issues

1. **"Could not extract data"**: Check that the model has the expected policy/value structure
2. **"Grid lengths inconsistent"**: Verify dimension labels match actual model structure  
3. **Empty plots**: Check that slice indices are within valid ranges for the model

### Debug Mode

Enable verbose output to see detailed information:

```bash
python3 -m examples.housing_renting.solve_runner --verbose --plots ...
```

### Checking Model Structure

If you need to verify the model structure, you can inspect it programmatically:

```python
# In solve_runner.py, after model creation:
if args.verbose:
    print(f"Model structure: {type(model)}")
    # Add more debugging info as needed
``` 