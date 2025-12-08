#!/bin/bash
# ======================================================================
#  Retirement Model Timing Experiments
#  Runs FUES vs DC-EGM comparison for Ishkakov et al (2017) model
# ======================================================================
#PBS -l ncpus=1
#PBS -l mem=5GB
#PBS -q expresssr
#PBS -P tp66
#PBS -l walltime=1:00:00
#PBS -l storage=scratch/tp66
#PBS -l wd
#PBS -o logs/
#PBS -e logs/

set -euo pipefail

# ======================================================================
#  USER SETTINGS - Modify these as needed
# ======================================================================

# Parameter file (in params/ folder)
PARAMS_FILE="params/baseline.yml"   # Options: baseline.yml, high_beta.yml, low_delta.yml, long_horizon.yml

# Baseline model settings
GRID_SIZE=2000              # Baseline grid size for plots (overrides params file)
PLOT_AGE=16                 # Age to plot EGM grids
OUTPUT_DIR=""               # Leave empty for default (results/retirement)

# Timing sweep settings
RUN_TIMINGS=true           # Run full timing comparison (slow)
SWEEP_GRIDS="1000,2000,3000,6000,10000"   # Grid sizes for sweep
SWEEP_DELTAS="0.25,0.5,1,2"              # Delta values for sweep
SWEEP_RUNS=3                            # Number of runs per config (best of n)

# ======================================================================
#  ENVIRONMENT SETUP
# ======================================================================

# Detect if running on PBS or locally
if [[ -n "${PBS_JOBFS:-}" ]]; then
    module load python3/3.12.1
    export NUMBA_CACHE_DIR=$PBS_JOBFS
    export VENV_ROOT=/scratch/tp66/$USER/venvs
    source "$VENV_ROOT/fues02-py3121/bin/activate"
else
    export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
fi

export NUMBA_NUM_THREADS=1

# ======================================================================
#  PATH SETUP
# ======================================================================

# Handle PBS vs local execution
if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    # PBS job: use submission directory
    SCRIPT_DIR="$PBS_O_WORKDIR/experiments/retirement"
    REPO_ROOT="$PBS_O_WORKDIR"
else
    # Local execution
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/results/retirement"
fi

# ======================================================================
#  RUN EXPERIMENT
# ======================================================================

echo "========================================================"
echo "  Retirement Model Experiment"
echo "  Params: $PARAMS_FILE"
echo "  Grid size: $GRID_SIZE"
echo "  Plot age: $PLOT_AGE"
echo "  Run timings: $RUN_TIMINGS"
if [[ "$RUN_TIMINGS" == "true" ]]; then
echo "    Sweep grids: $SWEEP_GRIDS"
echo "    Sweep deltas: $SWEEP_DELTAS"
echo "    Sweep runs: $SWEEP_RUNS"
fi
echo "  Output: $OUTPUT_DIR"
echo "========================================================"

cd "$REPO_ROOT"

# Build command
CMD="python3 $SCRIPT_DIR/run_experiment.py"
CMD="$CMD --params $PARAMS_FILE"
CMD="$CMD --grid-size $GRID_SIZE"
CMD="$CMD --plot-age $PLOT_AGE"
CMD="$CMD --output-dir $OUTPUT_DIR"

if [[ "$RUN_TIMINGS" == "true" ]]; then
    CMD="$CMD --run-timings"
    CMD="$CMD --sweep-grids $SWEEP_GRIDS"
    CMD="$CMD --sweep-deltas $SWEEP_DELTAS"
    CMD="$CMD --sweep-runs $SWEEP_RUNS"
fi

# Run
$CMD

echo "========================================================"
echo "  Experiment complete"
echo "  Plots saved to: $OUTPUT_DIR/plots"
echo "========================================================"
