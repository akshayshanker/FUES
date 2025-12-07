# Housing-Renting Experiments

## Quick Start

```bash
# 1. Edit configs/pbs_run_presets.sh to set your parameters
# 2. Edit run_housing_single_core.sh to select which config to run
# 3. Submit:
qsub run_housing_single_core.sh
```

## How Experiments Are Organized

```
/scratch/tp66/$USER/FUES/solutions/housing_renting/
└── test_0.1_gpu_test/              ← VERSION_TAG + TRIAL_ID
    ├── bundles/
    │   ├── abc123/                 ← Hash from hd_points (baseline)
    │   │   └── VFI_HDGRID_GPU/
    │   └── def456/                 ← Hash from grid_points (fast methods)
    │       ├── FUES/
    │       └── CONSAV/
    ├── raw_metrics.csv             ← All metrics
    └── comparison_table.tex        ← LaTeX table
```

Note: If `hd_points = grid_points`, all methods share the same hash folder.

## The Two Things You Change

### 1. `configs/pbs_run_presets.sh` - Define parameter sets

```bash
declare -A MY_EXPERIMENT
MY_EXPERIMENT[periods]=5
MY_EXPERIMENT[vfi_ngrid]=1E4
MY_EXPERIMENT[hd_points]=6000
MY_EXPERIMENT[grid_points]=2000
MY_EXPERIMENT[delta_pb]=1
MY_EXPERIMENT[version_suffix]="test_0.1"
```

## Parameter Definitions

### Model Settings (Hardcoded - not in bundle hash)

| Parameter | Description |
|-----------|-------------|
| `periods` | Number of time periods in the lifecycle model (e.g., 5 = 5 decision periods) |
| `vfi_ngrid` | Grid density for VFI solver. Higher = more accurate baseline but slower. Typical: 1E4-1E6 |

### Grid Points (Affect bundle hash)

| Parameter | Used By | Description |
|-----------|---------|-------------|
| `hd_points` | Baseline (VFI_HDGRID) | High-density grid for the "ground truth" solution. Typical: 6000-20000 |
| `grid_points` | Fast methods (FUES, CONSAV) | Grid density for EGM-based methods. Typical: 2000-6000 |

**Note:** `hd_points` and `grid_points` set three config values each: `a_points`, `a_nxt_points`, `w_points` (all set to the same value).

### Economic Parameters (Affect bundle hash)

| Parameter | Description |
|-----------|-------------|
| `delta_pb` | Price bound delta for housing prices. 1.0 = full range, <1.0 = tighter bounds |

### Naming

| Parameter | Description |
|-----------|-------------|
| `version_suffix` | Becomes `VERSION_TAG` in output paths. Use for config versioning (e.g., "test_0.1", "prod_v2") |

### 2. `run_housing_single_core.sh` - Select what to run

```bash
CONFIG_TO_RUN=("MY_EXPERIMENT")    # Which config from pbs_run_presets.sh
TRIAL_ID="Nov2025"                 # ← Your experimental trial name
```

## Naming Convention

| Name | What It Is | Example |
|------|------------|---------|
| `VERSION_TAG` | Config version (from `version_suffix`) | `test_0.1` |
| `TRIAL_ID` | Your trial/iteration name | `Nov2025`, `v2`, `gpu_test` |
| `{hash}` | Auto-generated from grid params | `a1b2c3d4` |

**Output directory:** `{VERSION_TAG}_{TRIAL_ID}` → `test_0.1_Nov2025`

## What Creates Different Bundles?

The bundle hash is computed from `[points, points, points, delta_pb]` where `points` depends on method type:

| Method Type | Points Used | Example |
|-------------|-------------|---------|
| Baseline (VFI_HDGRID, VFI_HDGRID_GPU) | `hd_points` | 6000 |
| Fast methods (FUES, CONSAV, DCEGM) | `grid_points` | 2000 |

**If `hd_points ≠ grid_points`:** Baseline and fast methods have DIFFERENT hashes:
```
bundles/
├── abc123/              ← hash(6000, 6000, 6000, 1.0) 
│   └── VFI_HDGRID_GPU/
└── def456/              ← hash(2000, 2000, 2000, 1.0)
    ├── FUES/
    └── CONSAV/
```

**If `hd_points = grid_points`:** All methods share the same hash folder.

### How Baseline Comparison Works

When fast methods compute comparison metrics (e.g., `dev_c_L2`), they need to load the baseline. This works correctly even with different hashes:

```
bundles/
├── abc123/              ← Baseline hash (hd_points=6000)
│   └── VFI_HDGRID_GPU/  ← Baseline loaded from HERE
└── def456/              ← Fast method hash (grid_points=2000)
    └── FUES/            ← Running here, compares against abc123
```

**How it works:** The runner stores `ref_params` (with `hd_points`) separately. When loading the baseline for comparison, it uses `ref_params` to find the correct bundle, not the current method's params.

**Key point:** As long as you use the same `hd_points` and `delta_pb`, the correct baseline is always found.

### Comparing Fast Methods to a Different Baseline Grid

If you want to compare FUES (at 6000 points) against a pre-computed VFI_HDGRID baseline (at 20000 points):

```bash
--ue-method FUES \
--baseline-method VFI_HDGRID \
--HD-points 20000 \
--grid-points 6000
```

Even though you're NOT running VFI_HDGRID, you must set:
- `--baseline-method VFI_HDGRID` so the baseline hash uses `hd_points`
- `--HD-points 20000` to match the pre-computed baseline's grid size

This ensures the runner looks for the baseline at the correct hash: `hash(20000, 20000, 20000, delta_pb)`.

## ⚠️ Hardcoded Settings (Not in Hash)

These settings **do NOT create different bundles** - changing them overwrites previous results:

| Setting | CLI Flag | Current Default |
|---------|----------|-----------------|
| Periods | `--periods` | 5 |
| VFI grid | `--vfi-ngrid` | 10000 |

**If you need different periods/vfi-ngrid:** Use a different `TRIAL_ID` to avoid overwriting.

## Metrics

```bash
--metrics "euler_error"           # Just Euler error (fast, no baseline needed)
--metrics "euler_error,dev_c_L2"  # Euler + deviation (needs baseline)
--metrics "all"                   # Everything (default)
```

## Files

| File | Purpose |
|------|---------|
| `configs/pbs_run_presets.sh` | Parameter configurations (grid sizes, periods) |
| `experiment_sets/default.yml` | Defines `param_paths` for bundle hashing |
| `run_housing_single_core.sh` | PBS job script |
| `../../examples/housing_renting/solve_runner.py` | Main solver |

## Experiment Sets

Experiment sets define which parameters go into the bundle hash. Located in `experiment_sets/`.

**To use a custom experiment set:**
```bash
python3 -m examples.housing_renting.solve_runner --experiment-set my_experiment ...
```

**To create a new experiment set:**
1. Copy `experiment_sets/default.yml` to `experiment_sets/my_experiment.yml`
2. Modify `param_paths` as needed
3. Use `--experiment-set my_experiment`

**Example `default.yml`:**
```yaml
param_paths:
  - master.methods.upper_envelope    # Method (excluded from hash)
  - master.settings.a_points         # Grid points
  - master.settings.a_nxt_points     
  - master.settings.w_points         
  - master.parameters.delta_pb       # Price bound delta
```

## Common Workflows

### Run a new experiment
1. Add new config in `configs/pbs_run_presets.sh`
2. Set `CONFIG_TO_RUN=("NEW_CONFIG")` in script
3. Set unique `TRIAL_ID`
4. Submit

### Iterate on same config
Just change `TRIAL_ID` → results go to new directory

### Compare methods
Same config runs all methods (FUES, CONSAV, etc.) in separate subfolders under same hash

### Check results
```bash
cat /scratch/tp66/$USER/FUES/solutions/housing_renting/test_0.1_Nov2025/raw_metrics.csv
```
