#!/bin/bash
# ======================================================================
#  Interactive GPU Job Submission Script
# ======================================================================
#
# This script is designed for interactive runs on a Gadi GPU node.
# To use it, first request an interactive GPU node, for example:
#
#   qsub -I -P tp66 -q gpuvolta -l ncpus=12,mem=383GB,ngpus=1,walltime=01:00:00,storage=gdata/tp66+scratch/tp66
#
# Once the job starts, you can run this script directly from the shell.
#

set -euo pipefail

# --- 1. Source the Configuration Library ---
source ../lib/job_configs.sh

# --- 2. Define the Sequence of Configurations to Run ---
CONFIG_TO_RUN=(
    "HIGH_RES_SETTINGS_A"
)

# --- 3. Environment Setup ---
module purge
module load python3/3.12.1 
module load cuda/12.8.0

export VENV_ROOT=/scratch/tp66/$USER/venvs
source "$VENV_ROOT/fues02-py3121/bin/activate"

export FUES_HOME=$HOME/dev/fues.dev/FUES
export PYTHONPATH="$FUES_HOME${PYTHONPATH:+:$PYTHONPATH}"
cd "$FUES_HOME"

# --- 4. Numba Configuration ---
export NUMBA_DISABLE_CACHE=1
export NUMBA_NUM_THREADS=1

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


# --- 5. Main Execution Loop ---
for CONFIG_NAME in "${CONFIG_TO_RUN[@]}"; do

    declare -n CONFIG_REF=$CONFIG_NAME

    echo "========================================================"
    echo "Running GPU Profiling for: ${CONFIG_NAME}"
    echo "========================================================"

    # --- Logging and Paths ---
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    VERSION_TAG="${CONFIG_REF[version_suffix]}"
    # TRIAL_ID can be set as environment variable or default to empty
    TRIAL_ID="${TRIAL_ID:-}"
    
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
    
    #PROFILE_OUTPUT_FILE="${FUES_HOME}/profile_${RUN_ID}.pstats"

    echo "Starting GPU profiling run for ${CONFIG_NAME} at $(date)"
    echo "Output will be saved to: $OUTPUT_DIR"
    echo "Logs will be saved to: $LOG_DIR"
    #echo "Profile data will be saved to: $PROFILE_OUTPUT_FILE"

    # --- Execution with cProfile ---
    # The python3 command is now wrapped with 'cProfile' to generate a performance profile.
    python3 -m examples.housing_renting.solve_runner \
      --periods "${CONFIG_REF[periods]}" \
      --ue-method "VFI_HDGRID_GPU,FUES2DEV" \
      --output-root "$OUTPUT_DIR" \
      --bundle-prefix "$VERSION_TAG" \
      --vfi-ngrid "${CONFIG_REF[vfi_ngrid]}" \
      --HD-points "${CONFIG_REF[hd_points]}" \
      --grid-points "${CONFIG_REF[grid_points]}" \
      --recompute-baseline \
      --fresh-fast \
      --precompile \
      --delta-pb "${CONFIG_REF[delta_pb]}" \
      --plots \
      --gpu \
      2> >(tee "${LOG_DIR}/run.err") \
      1> >(tee "${LOG_DIR}/run.log")

    echo "GPU profiling run for ${CONFIG_NAME} completed successfully."

done

echo "All GPU configurations processed."
exit 0