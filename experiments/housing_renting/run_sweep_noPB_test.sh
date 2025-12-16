#!/bin/bash
#PBS -N sweep-noPB-test
#PBS -P tp66
#PBS -q hugemem
#PBS -l ncpus=48,mem=1470GB,walltime=10:00:00
#PBS -l storage=scratch/tp66
#PBS -l wd
#PBS -j oe
#PBS -r y
#PBS -o logs/

# ======================================================================
#  MPI Sweep Test - One config per MPI rank
#  36 configs (4 methods × 3 grids × 3 H_sizes) on 36 MPI ranks
#  Each rank gets its own CPU (36 ranks on 48 CPUs)
#  
#  Hugemem: 1 node, 48 CPUs, 1470 GB RAM
#  36 ranks × ~40GB each = plenty of headroom
#  48h walltime for single hugemem node
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

VENV_PUBLIC="${VENV_PUBLIC:-/scratch/tp66/$USER/venvs/fues_public}"
source "$VENV_PUBLIC/bin/activate"

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
rm -rf "$NUMBA_CACHE_DIR" 2>/dev/null || true
mkdir -p "$NUMBA_CACHE_DIR"

# Hide GPUs
export CUDA_VISIBLE_DEVICES=""
export NUMBA_DISABLE_CUDA=1

# Suppress verbose output
export MAKEMOD_QUIET=true
export PERIOD_QUIET=true
export SHOCKS_QUIET=true

# DCEGM segment debugging (set to 1 to enable)
# export DCEGM_VERBOSE=1

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
EXPERIMENT_SET="sweep_noPB_small"
EXPERIMENT_SET_FILE="$SCRIPT_DIR/experiment_sets/${EXPERIMENT_SET}.yml"

# Read config from YAML (falls back to defaults if not found)
CONFIG_ID=$(python3 -c "import yaml; d=yaml.safe_load(open('$EXPERIMENT_SET_FILE')); print(d.get('config_id', 'test_0.1'))")
TRIAL_ID=$(python3 -c "import yaml; d=yaml.safe_load(open('$EXPERIMENT_SET_FILE')); print(d.get('trial_id', 'sweep-noPB-small'))")
OUTPUT_BASE=$(python3 -c "import yaml,os; d=yaml.safe_load(open('$EXPERIMENT_SET_FILE')); print(os.path.expandvars(d.get('output_dir', '/scratch/tp66/\$USER/FUES/solutions/housing_renting')))")

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TRIAL_ID}_${TIMESTAMP}"
LOG_DIR="$SCRIPT_DIR/logs"
OUTPUT_DIR="${OUTPUT_BASE}/${CONFIG_ID}-${TRIAL_ID}"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

MPI_RANKS=36
echo "========================================================"
echo "Parameter Sweep Test: ${EXPERIMENT_SET}"
echo "Config ID: ${CONFIG_ID}"
echo "Trial ID: ${TRIAL_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "Mode: MPI (${MPI_RANKS} ranks on 48 CPUs = 1 CPU per model)"
echo "Queue: hugemem (1 node, 1470GB RAM, ~40GB per rank)"
echo "Note: periods/vfi_ngrid read from experiment set YAML"
echo "========================================================"

# Run sweep with MPI - one config per rank, each loads its own baseline
# Note: periods/vfi_ngrid come from experiment set YAML fixed section
# For megamem: use 36 ranks (36 configs) but with more memory per rank
mpirun -np ${MPI_RANKS} python3 -m examples.housing_renting.solve_runner \
    --sweep \
    --experiment-set "${EXPERIMENT_SET}" \
    --config-id "${CONFIG_ID}" \
    --output-root "$OUTPUT_DIR" \
    --RUN-ID "$RUN_ID" \
    --mpi \
    --metrics "euler_error" \
    --plots \
    --skip-egm-plots \
    --fresh-fast \
    --low-memory \
    --skip-bundle-save \
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
    
    # Copy sweep_results.csv to results folder for persistence and re-tabulation
    echo "Copying sweep_results.csv to results folder..."
    cp "$OUTPUT_DIR/sweep_results.csv" "$TABLES_OUTPUT/sweep_results.csv"
    echo "Copied: $TABLES_OUTPUT/sweep_results.csv"
    
    # Generate tables from the local copy
    python3 "$REPO_ROOT/examples/housing_renting/tabulate_results.py" \
        --trial "${CONFIG_ID}-${TRIAL_ID}" \
        --experiment-set "${EXPERIMENT_SET}" \
        2>&1 | tee "${LOG_DIR}/tables_test_${TIMESTAMP}.log"
    
    echo "Paper tables saved to: $TABLES_OUTPUT"
fi

exit 0
