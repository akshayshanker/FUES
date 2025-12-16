# Housing-Renting Experiments

## Quick Start

```bash
# 1. Run GPU baseline sweep first (creates VFI_HDGRID_GPU baselines)
qsub run_sweep_vfi_gpu.pbs

# 2. Then run fast methods sweep (compares against baselines)
qsub run_sweep_noPB.pbs
```

## Experiment Set Design

All experiment configuration is centralized in YAML files under `experiment_sets/`. PBS scripts read these files directly.

### Experiment Set Files

| File | Purpose |
|------|---------|
| `sweep_VFI_GPU.yml` | GPU baseline sweep (VFI_HDGRID_GPU at high resolution) |
| `sweep_noPB.yml` | Fast methods sweep (FUES, CONSAV, DCEGM, VFI) |
| `sweep_noPB_small.yml` | Small test sweep for debugging |

### YAML Structure

```yaml
# Experiment identification
config_id: "test_0.1"           # Config version identifier
trial_id: "paper-sweep"         # Trial name (shared for baseline matching)
output_dir: "/scratch/tp66/${USER}/FUES/solutions/housing_renting"

# Reference method for baseline comparison (fast methods only)
ref_method: VFI_HDGRID_GPU

# Parameter paths for bundle hashing (3-element vector)
param_paths:
  - master.methods.upper_envelope    # Method
  - master.settings.a_points         # Grid points
  - master.settings.H_points         # Housing grid points

# Sweep dimensions
sweep:
  methods:
    - FUES
    - CONSAV
  grid_sizes:
    - 4000
    - 6000
  H_sizes:
    - 7
    - 10
    - 15

# Fixed parameters (not swept)
fixed:
  delta_pb: 1
  periods: 20
  vfi_ngrid: 50000  # Only for VFI methods

# Override params when building ref_params for baseline lookup
ref_params_override:
  grid_sizes: 200000  # Compare fast methods to high-res VFI baseline
```

## Output Directory Structure

```
/scratch/tp66/$USER/FUES/solutions/housing_renting/
└── test_0.1-paper-sweep/               ← config_id-trial_id
    ├── bundles/
    │   ├── VFI_HDGRID_GPU_200000_7/    ← method_gridsize_Hpoints
    │   │   ├── OWNC.pkl
    │   │   └── images_20251211_120000/
    │   ├── VFI_HDGRID_GPU_200000_10/
    │   ├── FUES_4000_7/
    │   └── CONSAV_4000_7/
    ├── sweep_results.csv               ← All metrics
    └── design_matrix.csv               ← Parameter combinations
```

## How Baseline Comparison Works

### The Problem
Fast methods (FUES at 4000 points) need to compare against VFI baseline (at 200000 points).
Different grid sizes = different bundle hashes.

### The Solution
Use `ref_method` and `ref_params_override` in the YAML:

```yaml
# In sweep_noPB.yml
ref_method: VFI_HDGRID_GPU
ref_params_override:
  grid_sizes: 200000
```

For a sweep config `(FUES, 4000, H=7)`, the runner builds:
```
ref_params = ['VFI_HDGRID_GPU', 200000, 7]
```

This matches the VFI baseline bundle with the **same H_points**.

### Matching trial_id
Both `sweep_VFI_GPU.yml` and `sweep_noPB.yml` must use the **same `trial_id`** so they write to the same output directory:

```yaml
# sweep_VFI_GPU.yml
trial_id: "paper-sweep"

# sweep_noPB.yml
trial_id: "paper-sweep"  # Same!
```

## PBS Scripts

PBS scripts read all configuration from the YAML files:

| Script | Experiment Set | Purpose |
|--------|----------------|---------|
| `run_sweep_vfi_gpu.pbs` | `sweep_VFI_GPU.yml` | GPU baseline sweep |
| `run_sweep_noPB.pbs` | `sweep_noPB.yml` | Fast methods sweep |
| `run_sweep_noPB_test.sh` | `sweep_noPB_small.yml` | Quick test |

**What PBS scripts read from YAML:**
- `config_id` → experiment version
- `trial_id` → trial identifier
- `output_dir` → base output path

**What solve_runner.py reads from YAML:**
- All sweep parameters (methods, grid_sizes, H_sizes)
- Fixed parameters (periods, vfi_ngrid, delta_pb)
- ref_method and ref_params_override for baseline matching

## Parameter Definitions

### Sweep Parameters (Create different bundles)

| Parameter | Description |
|-----------|-------------|
| `methods` | Upper envelope methods to sweep |
| `grid_sizes` | Asset grid points (a_points = w_points) |
| `H_sizes` | Housing/services grid points (H_points = S_points) |

### Fixed Parameters (Same for all configs in sweep)

| Parameter | Description |
|-----------|-------------|
| `periods` | Number of lifecycle periods |
| `vfi_ngrid` | VFI brute-force search grid density |
| `delta_pb` | Price bound delta (1 = full range) |

## Common Workflows

### 1. Run Complete Paper Experiment

```bash
# Step 1: Compute GPU baselines
qsub run_sweep_vfi_gpu.pbs

# Step 2: Wait for completion, then run fast methods
qsub run_sweep_noPB.pbs
```

### 2. Change Experiment Parameters

Edit the YAML file directly:
```yaml
# experiment_sets/sweep_noPB.yml
sweep:
  grid_sizes:
    - 1000
    - 2000
    - 4000  # Add more grid sizes
```

### 3. Run a Test Sweep

```bash
qsub run_sweep_noPB_test.sh
# Uses sweep_noPB_small.yml with fewer configs
```

### 4. Check Results

```bash
cat /scratch/tp66/$USER/FUES/solutions/housing_renting/test_0.1-paper-sweep/sweep_results.csv
```

## Files Reference

| File | Purpose |
|------|---------|
| `experiment_sets/*.yml` | Experiment configuration |
| `run_sweep_*.pbs` | PBS job scripts |
| `logs/` | Job output logs |
| `../../examples/housing_renting/solve_runner.py` | Main solver |
| `../../examples/housing_renting/helpers/execution_settings.py` | Config loading |

## Metrics

Available metrics (passed via `--metrics` flag):

| Metric | Needs Baseline | Description |
|--------|----------------|-------------|
| `euler_error` | No | Euler equation error |
| `dev_c_L2` | Yes | L2 deviation from baseline consumption |
| `plot_c_comparison` | Yes | Consumption comparison plots |
| `plot_v_comparison` | Yes | Value function comparison plots |

```bash
--metrics "euler_error"           # Fast, no baseline needed
--metrics "euler_error,dev_c_L2"  # With baseline comparison
```
