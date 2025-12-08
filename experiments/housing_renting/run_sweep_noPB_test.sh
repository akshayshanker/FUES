#!/bin/bash
#PBS -N sweep-noPB-test
#PBS -P tp66
#PBS -q express
#PBS -l ncpus=12,mem=96GB,walltime=02:00:00
#PBS -l storage=scratch/tp66
#PBS -l wd
#PBS -j oe
#PBS -r y
#PBS -o logs/

# ======================================================================
#  MPI Sweep Test - Small configuration for testing sweep mode
#  Trial ID: paper-v1-test
#  12 configs (3 methods × 2 grids × 2 H_sizes) on 12 MPI ranks
# ======================================================================

set -euo pipefail

# --- Path Setup ---
if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    SCRIPT_DIR="$PBS_O_WORKDIR"
    REPO_ROOT="$(cd "$PBS_O_WORKDIR/../.." && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

# --- Environment Setup ---
module purge
module load python3/3.12.1
module load openmpi/4.1.5

export VENV_ROOT=/scratch/tp66/$USER/venvs
source "$VENV_ROOT/fues02-py3121/bin/activate"

export FUES_HOME="$REPO_ROOT"
export PYTHONPATH="$FUES_HOME/src:$FUES_HOME${PYTHONPATH:+:$PYTHONPATH}"
cd "$FUES_HOME"

# --- MPI Configuration ---
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/scratch/tp66/$USER/numba_cache

# Suppress OpenMPI warnings
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll=^hcoll
export PMIX_MCA_gds=hash

echo "Clearing Numba cache at $NUMBA_CACHE_DIR..."
rm -rf "$NUMBA_CACHE_DIR"
mkdir -p "$NUMBA_CACHE_DIR"

# Hide GPUs
export CUDA_VISIBLE_DEVICES=""
export NUMBA_DISABLE_CUDA=1

# Suppress verbose output
export MAKEMOD_QUIET=true
export PERIOD_QUIET=true
export SHOCKS_QUIET=true

# --- Pre-run checks ---
echo "Checking environment..."
echo "Python: $(which python3)"
echo "MPI ranks: ${PBS_NCPUS:-12}"
echo "NUMBA_CACHE_DIR: $NUMBA_CACHE_DIR"

python3 -c "import numba; import quantecon; print('Numba version:', numba.__version__); print('Quantecon loaded successfully')" || {
    echo "ERROR: Failed to import required packages" >&2
    exit 1
}

# --- Sweep Configuration ---
CONFIG_ID="test_0.1"
TRIAL_ID="paper-v1-test"
EXPERIMENT_SET="sweep_noPB_small"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TRIAL_ID}_${TIMESTAMP}"
LOG_DIR="$SCRIPT_DIR/logs"
OUTPUT_DIR="/scratch/tp66/$USER/FUES/solutions/housing_renting/${CONFIG_ID}-${TRIAL_ID}"
EXPERIMENT_YAML="$SCRIPT_DIR/experiment_sets/${EXPERIMENT_SET}.yml"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Read periods from experiment set YAML
PERIODS=$(python3 -c "import yaml; print(yaml.safe_load(open('$EXPERIMENT_YAML'))['fixed']['periods'])")
echo "Periods from YAML: $PERIODS"

echo "========================================================"
echo "Parameter Sweep Test: ${EXPERIMENT_SET}"
echo "Config ID: ${CONFIG_ID}"
echo "Trial ID: ${TRIAL_ID}"
echo "Periods: ${PERIODS}"
echo "Output: ${OUTPUT_DIR}"
echo "Mode: MPI (${PBS_NCPUS:-12} ranks)"
echo "========================================================"

# Run sweep with MPI
mpirun -np ${PBS_NCPUS:-12} python3 -m examples.housing_renting.solve_runner \
    --sweep \
    --experiment-set "${EXPERIMENT_SET}" \
    --config-id "${CONFIG_ID}" \
    --output-root "$OUTPUT_DIR" \
    --RUN-ID "$RUN_ID" \
    --periods "$PERIODS" \
    --mpi \
    --metrics "euler_error" \
    --fresh-fast \
    --trace \
    2> >(tee "${LOG_DIR}/sweep_test_${TIMESTAMP}.err") \
    1> >(tee "${LOG_DIR}/sweep_test_${TIMESTAMP}.log")

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Sweep test failed with exit code: $EXIT_CODE" >&2
    exit $EXIT_CODE
else
    echo "Sweep test completed successfully."
    echo "Results saved to: $OUTPUT_DIR"
    
    # Generate paper tables from results (in config-id/trial-id subfolder)
    echo "========================================================"
    echo "Generating paper tables..."
    echo "========================================================"
    
    TABLES_OUTPUT="$REPO_ROOT/results/housing_renting/${CONFIG_ID}-${TRIAL_ID}"
    mkdir -p "$TABLES_OUTPUT"
    
    python3 "$REPO_ROOT/examples/housing_renting/helpers/generate_paper_tables.py" \
        --results "$OUTPUT_DIR/sweep_results.csv" \
        --output "$TABLES_OUTPUT" \
        2>&1 | tee "${LOG_DIR}/tables_test_${TIMESTAMP}.log"
    
    echo "Paper tables saved to: $TABLES_OUTPUT"
fi

exit 0
