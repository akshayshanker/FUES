#!/bin/bash
#PBS -N fues-single-core
#PBS -P tp66
#PBS -q hugemem
#PBS -l ncpus=1,mem=1470GB,walltime=04:00:00
#PBS -l storage=scratch/tp66
#PBS -l wd
#PBS -j oe
#PBS -r y

# ======================================================================
#  Single Core Job - Load Baseline, Compute Fast Methods Only
#  Updated for Gadi hugemem queue
# ======================================================================

set -euo pipefail

# --- Source the Configuration Library ---
source ../lib/job_configs.sh

# --- Define the Sequence of Configurations to Run ---
CONFIG_TO_RUN=("HIGH_RES_SETTINGS_C")


# --- Environment Setup ---
module purge
module load python3/3.12.1
export VENV_ROOT=/scratch/tp66/$USER/venvs
source "$VENV_ROOT/fues02-py3121/bin/activate"

export FUES_HOME=$HOME/dev/fues.dev/FUES
export PYTHONPATH="$FUES_HOME${PYTHONPATH:+:$PYTHONPATH}"
cd "$FUES_HOME"

# --- Single Core Configuration ---
# No MPI settings needed for single core
export NUMBA_CACHE_DIR=/scratch/tp66/$USER/numba_cache

# Optional: Clear cache if needed
if [[ "${1:-}" == "--clear-cache" ]]; then
    echo "Clearing Numba cache at $NUMBA_CACHE_DIR..."
    rm -rf $NUMBA_CACHE_DIR
    shift
fi

mkdir -p $NUMBA_CACHE_DIR
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
    # TRIAL_ID can be set as environment variable or default to empty
    TRIAL_ID="gpu_test"
    
    # Build paths based on whether TRIAL_ID is set
    if [[ -n "$TRIAL_ID" ]]; then
        RUN_ID="${VERSION_TAG}_${TIMESTAMP}_${TRIAL_ID}"
        LOG_DIR="logs/${VERSION_TAG}_${TRIAL_ID}"
        OUTPUT_DIR="/scratch/tp66/$USER/FUES/solutions/housing_renting/${VERSION_TAG}_${TRIAL_ID}"
    else
        RUN_ID="${VERSION_TAG}_${TIMESTAMP}"
        LOG_DIR="logs/${VERSION_TAG}"
        OUTPUT_DIR="/scratch/tp66/$USER/FUES/solutions/housing_renting/${VERSION_TAG}"
    fi
    mkdir -p "$LOG_DIR"

    echo "Starting single-core run for ${CONFIG_NAME} at $(date)"
    echo "Output will be saved to: $OUTPUT_DIR"
    echo "Logs will be saved to: $LOG_DIR"
    echo "NOTE: Baseline will be loaded from existing bundles, not recomputed"
    echo "NOTE: Using selective loading for Euler error - loading only periods 0,1"

    # Selective loading: Euler error only needs:
    # - Period 0: OWNC stage (for current consumption)
    # - Period 1: All stages (OWNC, TENU, OWNH, RNTH, RNTC for next period policies)
    # This reduces loading from 75 to 18 pickle files (76% reduction)
    python3 -m examples.housing_renting.solve_runner \
      --periods "${CONFIG_REF[periods]}" \
      --ue-method "FUES, CONSAV,VFI_HDGRID" \
      --output-root "$OUTPUT_DIR" \
      --bundle-prefix "${VERSION_TAG}" \
      --RUN-ID "${VERSION_TAG}_${TIMESTAMP}" \
      --vfi-ngrid "${CONFIG_REF[vfi_ngrid]}" \
      --HD-points "${CONFIG_REF[hd_points]}" \
      --grid-points "${CONFIG_REF[grid_points]}" \
      --baseline-method "VFI_HDGRID" \
      --metrics "euler_error, plot_c_comparison" \
      --include-baseline
      --fresh-fast \
      --plots \
      --trace \
      --low-memory \
      --load-periods "0,1" \
      --load-stages '{"0": ["OWNC"], "1": null}' \
      2> >(tee "${LOG_DIR}/run.err") \
      1> >(tee "${LOG_DIR}/run.log")

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Run for ${CONFIG_NAME} failed with exit code: $EXIT_CODE" >&2
        echo "Check error log at: ${LOG_DIR}/run.err" >&2
        
        # Check for common errors in the log
        if grep -q "LLVM ERROR" "${LOG_DIR}/run.err"; then
            echo "HINT: LLVM error detected. Try running with --clear-cache flag" >&2
        fi
        if grep -q "baseline.*not found" "${LOG_DIR}/run.err"; then
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