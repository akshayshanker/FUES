set -euo pipefail

# --- Source the Configuration Library ---
source ../lib/job_configs.sh

# --- Define the Sequence of Configurations to Run ---
CONFIG_TO_RUN=(
    "STD_RES_SETTINGS"
    # "DEBUG_SETTINGS"
)

# --- Environment Setup ---
module purge
module load python3/3.12.1 openmpi/4.1.7
export VENV_ROOT=/scratch/tp66/$USER/venvs
source "$VENV_ROOT/fues02-py3121/bin/activate"

export FUES_HOME=$HOME/dev/fues.dev/FUES
export PYTHONPATH="$FUES_HOME${PYTHONPATH:+:$PYTHONPATH}"
cd "$FUES_HOME"

# --- MPI and Numba Configuration ---
export OMPI_MCA_btl_base_warn_component_unused=0

# Numba settings - use persistent cache to avoid recompilation issues
export NUMBA_CACHE_DIR=/scratch/tp66/$USER/numba_cache

# Optional: Clear cache if passed as argument
if [[ "${1:-}" == "--clear-cache" ]]; then
    echo "Clearing Numba cache at $NUMBA_CACHE_DIR..."
    rm -rf $NUMBA_CACHE_DIR
    shift  # Remove the argument
fi

mkdir -p $NUMBA_CACHE_DIR
# export NUMBA_DISABLE_CACHE=1  # Commented out - allow caching to prevent LLVM errors
export NUMBA_NUM_THREADS=1

# Hide GPUs from Numba to prevent CUDA initialization errors
export CUDA_VISIBLE_DEVICES=""
export NUMBA_CUDA_LOG_LEVEL=WARNING
export NUMBA_DISABLE_CUDA=1

# ---  errors 

export PYTHONPATH=$PWD:$PYTHONPATH
export MAKEMOD_QUIET=true           # Only errors
export PERIOD_QUIET=true            # Only errors
export SHOCKS_QUIET=true

# Suppress non-fatal MPI warnings
export OMPI_MCA_btl_base_warn_component_unused=0
export OMPI_MCA_mca_base_component_show_load_errors=0
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_mpi_warn_on_fork=0
export OMPI_MCA_orte_base_help_aggregate=1
export OMPI_MCA_mpi_show_handle_leaks=0
export OMPI_MCA_btl_openib_warn_default_gid_prefix=0
export OMPI_MCA_btl_openib_warn_no_device_params_found=0
export OMPI_MCA_coll_ml_priority=0
export OMPI_MCA_coll_hcoll_enable=0

# Note: Numba settings already configured above
# Using shared cache directory instead of process-specific to avoid recompilation

# --- Pre-run checks ---
echo "Checking environment..."
echo "Python: $(which python3)"
echo "MPI: $(which mpiexec)"
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
    echo "Running Configuration: ${CONFIG_NAME}"
    echo "Periods: ${CONFIG_REF[periods]}"
    echo "VFI Grid: ${CONFIG_REF[vfi_ngrid]}"
    echo "HD Points: ${CONFIG_REF[hd_points]}"
    echo "Grid Points: ${CONFIG_REF[grid_points]}"
    echo "========================================================"

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    VERSION_TAG="${CONFIG_REF[version_suffix]}"
    LOG_DIR="logs/${VERSION_TAG}"
    mkdir -p "$LOG_DIR"
    
    SOLUTION_ROOT="/scratch/tp66/$USER/FUES/solutions/housing_renting/"
    OUTPUT_DIR="$SOLUTION_ROOT/${VERSION_TAG}"

    echo "Starting MPI run for ${CONFIG_NAME} at $(date)"
    echo "Output will be saved to: $OUTPUT_DIR"
    echo "Logs will be saved to: $LOG_DIR"

    mpiexec -np 45 python3 -m examples.housing_renting.solve_runner \
      --periods "${CONFIG_REF[periods]}" \
      --ue-method "VFI_HDGRID,FUES2DEV" \
      --output-root "$OUTPUT_DIR" \
      --bundle-prefix "${VERSION_TAG}" \
      --RUN-ID "${VERSION_TAG}_${TIMESTAMP}" \
      --vfi-ngrid "${CONFIG_REF[vfi_ngrid]}" \
      --HD-points "${CONFIG_REF[hd_points]}" \
      --grid-points "${CONFIG_REF[grid_points]}" \
      --recompute-baseline \
      --fresh-fast \
      --delta-pb "${CONFIG_REF[delta_pb]}" \
      --mpi \
      --plots \
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
        if grep -q "CUDA_ERROR" "${LOG_DIR}/run.err"; then
            echo "HINT: CUDA errors detected (these can be ignored on CPU nodes)" >&2
        fi
        
        # Decide if you want to exit immediately or continue with next config
        exit $EXIT_CODE  # Exit on first failure
    else
        echo "Run for ${CONFIG_NAME} completed successfully."
        echo "Results saved to: $OUTPUT_DIR"
    fi
    
done

echo "All configurations processed."
exit 0 