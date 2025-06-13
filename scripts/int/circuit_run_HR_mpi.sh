module load openmpi/4.1.7
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
export OMPI_MCA_btl_base_warn_component_unused=0
export OMPI_MCA_mca_base_component_show_load_errors=0
export OMPI_MCA_coll_ml_priority=0
export OMPI_MCA_coll_hcoll_enable=0

# Numba settings for MPI safety
export NUMBA_DISABLE_CACHE=1        # Disable caching to avoid MPI conflicts
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$  # Use process-specific cache dir
export NUMBA_NUM_THREADS=1          # Prevent thread conflicts

cd $HOME/dev/fues.dev/FUES

# Clear Numba cache to avoid MPI cache corruption
echo "Clearing Numba cache..."
python3 -c "import numba; numba.config.CACHE_DIR.clear()" 2>/dev/null || true
find ~/.numba_cache -name "*.nbc" -delete 2>/dev/null || true
find ~/.numba_cache -name "*.nbi" -delete 2>/dev/null || true

# Create logs directory if it doesn't exist
mkdir -p logs

# Set log file names with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/HR_mpi_run_${TIMESTAMP}.log"
ERROR_FILE="logs/HR_mpi_errors_${TIMESTAMP}.log"

echo "Starting MPI run at $(date)"
echo "Logs will be saved to: $LOG_FILE"
echo "Errors will be saved to: $ERROR_FILE"

# ------------ MPI RUN ------------
mpiexec -np 45 \
        python3 -m examples.housing_renting.solve_runner \
          --periods 3 \
          --ue-method ALL \
          --output-root /scratch/tp66/$USER/FUES/solutions/HR_test_v4 \
          --bundle-prefix HR_test_v4 \
          --vfi-ngrid 100 \
          --HD-points 600 \
          --grid-points 500 \
          --recompute-baseline \
          --fresh-fast \
          --mpi \
          --plots \
        2> >(grep -v "LOG_CAT_ML\|basesmuma\|ml_discover_hierarchy" | tee -a "$ERROR_FILE" >&2) \
        1> >(tee -a "$LOG_FILE")

# Capture exit code
EXIT_CODE=$?

echo "MPI run completed at $(date) with exit code: $EXIT_CODE"
echo "Check logs at: $LOG_FILE"
echo "Check errors at: $ERROR_FILE"

exit $EXIT_CODE