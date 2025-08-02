# Interactive Scripts for Gadi

This directory contains interactive scripts for running FUES housing renting model computations in Gadi interactive sessions.

## Setup

1. **Start an interactive session on Gadi:**
   ```bash
   qsub -I -P tp66 -q expresssr -l ncpus=1,mem=100GB,walltime=2:00:00 -l storage=scratch/tp66
   ```

2. **Navigate to the scripts directory:**
   ```bash
   cd $HOME/dev/fues.dev/FUES/scripts/int
   ```

## Available Scripts

### `circuit_run_HR_single.sh` - Single Core Interactive Runner

Perfect for development, testing, and quick runs without job submission overhead.

#### Basic Usage

**Interactive mode (recommended for first-time users):**
```bash
./circuit_run_HR_single.sh
```
This will:
- Set up the environment automatically
- Present configuration options
- Ask for confirmation before running
- Show live output with full logging

**Quick start with default settings:**
```bash
./circuit_run_HR_single.sh --config HIGH_RES_SETTINGS_E
```

#### Advanced Usage Examples

**Fast Euler error computation only (no baseline loading):**
```bash
./circuit_run_HR_single.sh --metrics euler_error --config HIGH_RES_SETTINGS_E
```
⚡ **Performance**: Skips expensive baseline loading, runs in ~30 seconds

**L2 comparison with debug tracing:**
```bash
./circuit_run_HR_single.sh --metrics dev_c_L2 --trace --config HIGH_RES_SETTINGS_E
```

**Generate plots only:**
```bash
./circuit_run_HR_single.sh --metrics plots --config HIGH_RES_SETTINGS_E
```

**Full comparison (all metrics):**
```bash
./circuit_run_HR_single.sh --metrics all --config HIGH_RES_SETTINGS_E
```

**Clear cache and run with tracing:**
```bash
./circuit_run_HR_single.sh --clear-cache --trace --config HIGH_RES_SETTINGS_E
```

#### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config NAME` | Run specific configuration | `--config HIGH_RES_SETTINGS_E` |
| `--metrics LIST` | Select metrics to compute | `--metrics euler_error,dev_c_L2` |
| `--no-plots` | Skip plot generation | `--no-plots` |
| `--trace` | Enable debug tracing | `--trace` |
| `--clear-cache` | Clear Numba cache first | `--clear-cache` |
| `--help` | Show help message | `--help` |

#### Available Metrics

| Metric | Description | Baseline Required |
|--------|-------------|-------------------|
| `euler_error` | Euler equation errors | ❌ No |
| `dev_c_L2` | L2 deviation of consumption | ✅ Yes |
| `plots` | Error comparison plots | ✅ Yes |
| `all` | All metrics (default) | ✅ Yes |

#### Performance Tips

1. **For development/testing**: Use `--metrics euler_error` for fastest execution
2. **For debugging**: Add `--trace` to see detailed execution steps
3. **For clean runs**: Use `--clear-cache` if you encounter LLVM errors
4. **For memory issues**: Monitor memory usage with trace output

#### Common Workflows

**Development workflow:**
```bash
# 1. Quick test with minimal metrics
./circuit_run_HR_single.sh --metrics euler_error --config HIGH_RES_SETTINGS_E

# 2. If successful, run full comparison
./circuit_run_HR_single.sh --metrics all --config HIGH_RES_SETTINGS_E
```

**Debugging workflow:**
```bash
# Clear cache and run with full tracing
./circuit_run_HR_single.sh --clear-cache --trace --metrics euler_error --config HIGH_RES_SETTINGS_E
```

**Analysis workflow:**
```bash
# Generate specific comparisons
./circuit_run_HR_single.sh --metrics dev_c_L2,plots --config HIGH_RES_SETTINGS_E
```

## Output Structure

```
/scratch/tp66/$USER/FUES/solutions/housing_renting/test_0.1_interactive/
├── bundles/           # Solver output bundles
├── images/           # Generated plots (if --plots enabled)
├── comparison_table.tex  # LaTeX comparison table
└── design_matrix.csv    # Parameter combinations tested

logs/test_0.1_interactive/
└── run.log           # Complete execution log
```

## Troubleshooting

### Common Issues

**"Baseline bundle not found"**
- Run a GPU job first to compute the baseline
- Or use `--metrics euler_error` to skip baseline requirement

**"LLVM ERROR"**
- Run with `--clear-cache` to clear Numba compilation cache

**"Variable scoping error"**
- This indicates a code bug, report to developers

**"Failed to load Python module"**
- Ensure you're running on Gadi compute nodes
- Check that modules are available: `module avail python3`

### Memory Monitoring

The script includes built-in memory monitoring when `--trace` is enabled:
```
Memory usage at start: 0.40 GB
Memory usage after baseline computation: 4.55 GB
Memory usage at end: 6.57 GB
```

### Getting Help

```bash
./circuit_run_HR_single.sh --help
```

## Integration with New Features

This script takes advantage of recent improvements:

1. **Configurable Metrics**: Use `--metrics` to select only needed computations
2. **Memory Management**: Built-in memory optimization and monitoring
3. **Smart Baseline Loading**: Only loads baseline when comparison metrics are needed
4. **Enhanced Error Handling**: Clear error messages with helpful hints

## Performance Comparison

| Metrics | Baseline Loading | Typical Runtime | Memory Peak |
|---------|------------------|-----------------|-------------|
| `euler_error` | ❌ No | ~30 seconds | ~1 GB |
| `dev_c_L2` | ✅ Yes | ~2 minutes | ~5 GB |
| `plots` | ✅ Yes | ~3 minutes | ~6 GB |
| `all` | ✅ Yes | ~3-4 minutes | ~7 GB | 