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
#PBS -j oe
#PBS -o /dev/null
#PBS -e /dev/null

set -euo pipefail

# ======================================================================
#  USER SETTINGS - Modify these as needed
# ======================================================================

# Parameter file (in params/ folder)
PARAMS_FILE="params/sigma05.yml"   # Options: baseline.yml, high_beta.yml, low_delta.yml, long_horizon.yml

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

# Run id for logs + output folder
RUN_ID="${RUN_ID:-retirement_$(date +%Y%m%d_%H%M%S)}"

# Prefer scratch on Gadi; fall back to /tmp for local runs
SCRATCH_ROOT="/scratch/tp66/${USER}"
if [[ -d "${SCRATCH_ROOT}" ]]; then
    BASE_OUT="${SCRATCH_ROOT}/FUES"
else
    BASE_OUT="${TMPDIR:-/tmp}/FUES"
fi

# Logs go to BASE_OUT (avoid home on Gadi)
LOG_DIR="${LOG_DIR:-${BASE_OUT}/logs/retirement}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
ERR_FILE="${LOG_DIR}/${RUN_ID}.err"

# Mirror stdout/stderr into scratch logs
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${ERR_FILE}" >&2)

# Detect if running on PBS or locally
if [[ -n "${PBS_JOBFS:-}" ]]; then
    module purge
    module load python3/3.12.1

    # Activate dcsmm venv on scratch
    DCSMM_VENV="${DCSMM_VENV:-/scratch/tp66/${USER}/venvs/dcsmm}"
    source "${DCSMM_VENV}/bin/activate"

    # Keep Numba cache on scratch (PBS jobfs is usually too small)
    export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/scratch/tp66/${USER}/numba_cache}"
else
    export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${BASE_OUT}/numba_cache}"
fi

export NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${BASE_OUT}/mplconfig}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

# ======================================================================
#  PATH SETUP
# ======================================================================

# Handle PBS vs local execution
if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    # PBS job: user may submit from repo root OR from experiments/retirement/
    if [[ -f "${PBS_O_WORKDIR}/pyproject.toml" ]]; then
        REPO_ROOT="${PBS_O_WORKDIR}"
    elif [[ -f "${PBS_O_WORKDIR}/../pyproject.toml" ]]; then
        REPO_ROOT="$(cd "${PBS_O_WORKDIR}/.." && pwd)"
    elif [[ -f "${PBS_O_WORKDIR}/../../pyproject.toml" ]]; then
        REPO_ROOT="$(cd "${PBS_O_WORKDIR}/../.." && pwd)"
    else
        REPO_ROOT="${PBS_O_WORKDIR}"
    fi
else
    # Local execution
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

EXAMPLE_DIR="${REPO_ROOT}/examples/retirement"

if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ -d "${SCRATCH_ROOT}" ]]; then
        OUTPUT_DIR="${SCRATCH_ROOT}/FUES/solutions/retirement/${RUN_ID}"
    else
        OUTPUT_DIR="${REPO_ROOT}/results/retirement/${RUN_ID}"
    fi
fi

# Ensure outputs directory exists
mkdir -p "${OUTPUT_DIR}"

# Make repo + src importable (fixes `import dcsmm` without pip install)
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

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
echo "  Log: $LOG_FILE"
echo "  Err: $ERR_FILE"
echo "========================================================"

cd "$REPO_ROOT"

# Build command
CMD="python3 $EXAMPLE_DIR/run_experiment.py"
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
