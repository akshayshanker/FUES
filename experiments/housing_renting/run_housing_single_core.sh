#!/bin/bash
#PBS -N fues-single-core
#PBS -P tp66
#PBS -q expresssr
#PBS -l ncpus=1,mem=30GB,walltime=04:00:00
#PBS -l storage=scratch/tp66
#PBS -l wd
#PBS -j oe
#PBS -r y
#PBS -o logs/
# Note: Create logs/ folder before submitting: mkdir -p experiments/housing_renting/logs

# ======================================================================
#  Single Core Job - Load Baseline, Compute Fast Methods Only
#  Updated for Gadi hugemem queue
# ======================================================================

set -euo pipefail

# --- Path Setup (handle PBS vs local) ---
# PBS_O_WORKDIR is the directory where qsub was invoked (experiments/housing_renting/)
if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    SCRIPT_DIR="$PBS_O_WORKDIR"
    REPO_ROOT="$(cd "$PBS_O_WORKDIR/../.." && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

# --- Source the Configuration Library ---
source "$SCRIPT_DIR/configs/pbs_run_presets.sh"

# --- Define the Sequence of Configurations to Run ---
CONFIG_TO_RUN=("STD_RES_SETTINGS_4")


# --- Environment Setup ---
module purge
module load python3/3.12.1
module load openmpi/4.1.5  # Required for mpi4py even in single-core mode
# Use public venv
VENV_PUBLIC="${VENV_PUBLIC:-/scratch/tp66/$USER/venvs/fues_public}"
source "$VENV_PUBLIC/bin/activate"

export FUES_HOME="$REPO_ROOT"
# Add both repo root (for examples.*) and src/ (for dc_smm.*) to PYTHONPATH
export PYTHONPATH="$FUES_HOME:$FUES_HOME/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$FUES_HOME"

# --- Single Core Configuration ---
# No MPI settings needed for single core
export NUMBA_CACHE_DIR=/scratch/tp66/$USER/numba_cache

# Always clear Numba cache to ensure fresh compilation with latest code
# Use || true to avoid failure if another job is using the cache
echo "Clearing Numba cache at $NUMBA_CACHE_DIR..."
rm -rf "$NUMBA_CACHE_DIR" 2>/dev/null || true
mkdir -p "$NUMBA_CACHE_DIR"
export NUMBA_NUM_THREADS=1

# Hide GPUs from Numba to prevent CUDA initialization errors
export CUDA_VISIBLE_DEVICES=""
export NUMBA_CUDA_LOG_LEVEL=WARNING
export NUMBA_DISABLE_CUDA=1

# --- Suppress errors and warnings ---
export PYTHONPATH=$PWD:$PYTHONPATH
export MAKEMOD_QUIET=true           # Only errors
export PERIOD_QUIET=true            # Only errors
export SHOCKS_QUIET=true

# --- Pre-run checks ---
echo "Checking environment..."
echo "Python: $(which python3)"
echo "NUMBA_CACHE_DIR: $NUMBA_CACHE_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Test import of critical packages
python3 -c "import numba; import quantecon; print('Numba version:', numba.__version__); print('Quantecon loaded successfully')" || {
    echo "ERROR: Failed to import required packages" >&2
    exit 1
}

# --- Loop Through and Execute Each Configuration ---
for CONFIG_NAME in "${CONFIG_TO_RUN[@]}"; do
    
    declare -n CONFIG_REF=$CONFIG_NAME

    echo "========================================================"
    echo "Running Configuration: ${CONFIG_NAME} (Single Core)"
    echo "Periods: ${CONFIG_REF[periods]}"
    echo "VFI Grid: ${CONFIG_REF[vfi_ngrid]}"
    echo "HD Points: ${CONFIG_REF[hd_points]}"
    echo "Grid Points: ${CONFIG_REF[grid_points]}"
    echo "========================================================"

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    VERSION_TAG="${CONFIG_REF[version_suffix]}"
    TRIAL_ID="single_core"
    RUN_ID="${VERSION_TAG}_${TIMESTAMP}_${TRIAL_ID}"
    
    # All logs go to experiments/housing_renting/logs/ (same as PBS -o)
    LOG_DIR="$SCRIPT_DIR/logs"
    OUTPUT_DIR="/scratch/tp66/$USER/FUES/solutions/housing_renting/${VERSION_TAG}_${TRIAL_ID}"
    mkdir -p "$LOG_DIR"

    echo "Starting single-core run for ${CONFIG_NAME} at $(date)"
    echo "Output will be saved to: $OUTPUT_DIR"
    echo "Logs will be saved to: $LOG_DIR"
    echo "NOTE: Baseline (VFI_HDGRID_GPU) will be loaded from existing bundles via --include-baseline"
    echo "NOTE: euler_error metric doesn't require baseline comparison (runs independently)"
    echo "NOTE: EGM plots ENABLED via --plots and --trace flags"

    python3 -m examples.housing_renting.solve_runner \
      --periods "${CONFIG_REF[periods]}" \
      --ue-method "FUES, DCEGM,CONSAV" \
      --output-root "$OUTPUT_DIR" \
      --config-id "${VERSION_TAG}" \
      --RUN-ID "${VERSION_TAG}_${TIMESTAMP}" \
      --vfi-ngrid "${CONFIG_REF[vfi_ngrid]}" \
      --HD-points "${CONFIG_REF[hd_points]}" \
      --grid-points "${CONFIG_REF[grid_points]}" \
      --delta-pb "${CONFIG_REF[delta_pb]}" \
      --baseline-method "VFI_HDGRID_GPU" \
      --metrics "euler_error" \
      --fresh-fast \
      --csv-export \
      --plots \
      --trace \
      2> >(tee "${LOG_DIR}/run_${TIMESTAMP}.err") \
      1> >(tee "${LOG_DIR}/run_${TIMESTAMP}.log")

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Run for ${CONFIG_NAME} failed with exit code: $EXIT_CODE" >&2
        echo "Check error log at: ${LOG_DIR}/run_${TIMESTAMP}.err" >&2
        
        # Check for common errors in the log
        if grep -q "LLVM ERROR" "${LOG_DIR}/run_${TIMESTAMP}.err"; then
            echo "HINT: LLVM error detected. Try running with --clear-cache flag" >&2
        fi
        if grep -q "baseline.*not found" "${LOG_DIR}/run_${TIMESTAMP}.err"; then
            echo "HINT: Baseline bundle not found. Run MPI job first to compute baseline" >&2
        fi
        
        # Exit on first failure for single core jobs
        exit $EXIT_CODE
    else
        echo "Run for ${CONFIG_NAME} completed successfully."
        echo "Results saved to: $OUTPUT_DIR"
    fi
    
done

echo "All single-core configurations processed."
exit 0
