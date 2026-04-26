#!/bin/bash
# ======================================================================
#  Retirement Model Timing Experiments
#  Runs FUES vs DC-EGM comparison for Ishkakov et al (2017) model
# ======================================================================
#PBS -l ncpus=240
#PBS -l mem=960GB
#PBS -q normal
#PBS -P tp66
#PBS -l walltime=1:00:00
#PBS -l storage=scratch/tp66+gdata/tp66
#PBS -l wd
#PBS -j oe
#PBS -o /g/data/tp66/logs/retirement/
#PBS -e /g/data/tp66/logs/retirement/

set -euo pipefail

# ======================================================================
#  USER SETTINGS - Modify these as needed
# ======================================================================

# Parameter file (in params/ folder)
PARAMS_FILE="params/baseline.yml"   # Options: baseline.yml, high_beta.yml, low_delta.yml, long_horizon.yml

# Baseline model settings
GRID_SIZE=2000              # Baseline grid size for plots (overrides params file)
PLOT_AGE=16                 # Age to plot EGM grids

# Timing sweep (RunSpec v2: YAML ranges, no run_timings / sweep_grids extras)
RUN_TIMINGS=true             # true = full (grid×delta×method) sweep + tables + plots
SWEEP_RUNS=3                 # kikku best-of-n per test row
LATEX_GRIDS="1000,2000,3000,6000,10000"  # LaTeX table subset; md gets all
RANGES_DIR="experiments/retirement"      # @files resolved from REPO root (see cd below)

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
LOG_DIR="${LOG_DIR:-/g/data/tp66/logs/retirement}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
ERR_FILE="${LOG_DIR}/${RUN_ID}.err"

# Mirror stdout/stderr into scratch logs
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${ERR_FILE}" >&2)

# Detect if running on PBS or locally
if [[ -n "${PBS_JOBFS:-}" ]]; then
    module purge
    module load python3/3.12.1

    # Activate fues venv
    FUES_VENV="${FUES_VENV:-$HOME/venvs/fues}"
    source "${FUES_VENV}/bin/activate"

    # Use PBS_JOBFS (fast local SSD on compute node)
    export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${PBS_JOBFS}}"
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

# Match the durables PBS convention: single-line hardcoded scratch path.
# Override via `OUTPUT_DIR=/some/path qsub ...` for local runs.
# make_run_dir creates YYYY-MM-DD/NNN/ inside OUTPUT_DIR automatically.
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/tp66/$USER/FUES/retirement}"

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
    echo "    Full timing sweep: params/settings/methods ranges (YAML @files)"
    echo "    Sweep runs: $SWEEP_RUNS"
fi
echo "  Output: $OUTPUT_DIR"
echo "  Log: $LOG_FILE"
echo "  Err: $ERR_FILE"
echo "========================================================"

cd "$REPO_ROOT"

# Resolve params file relative to experiments/retirement/
TIMINGS_SCRIPT_DIR="${REPO_ROOT}/experiments/retirement"
if [[ ! -f "${PARAMS_FILE}" ]]; then
    PARAMS_FILE="${TIMINGS_SCRIPT_DIR}/${PARAMS_FILE}"
fi

# MPI setup (sweep distributes grid points across ranks)
module load openmpi/4.1.5
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll=^hcoll
export PMIX_MCA_gds=hash

# Build command (kikku v2: Cartesian product of @file ranges, single sweep in run.py)
CMD="mpiexec -n $PBS_NCPUS python3 -u -m mpi4py -m examples.retirement.run"
CMD="$CMD --override-file $PARAMS_FILE"
CMD="$CMD --settings-override plot_age=$PLOT_AGE"
CMD="$CMD --output-dir $OUTPUT_DIR"

if [[ "$RUN_TIMINGS" == "true" ]]; then
    CMD="$CMD --sweep"
    CMD="$CMD --params-range @${RANGES_DIR}/timing_deltas.yaml"
    CMD="$CMD --settings-range @${RANGES_DIR}/timing_grids.yaml"
    CMD="$CMD --methods-range @${RANGES_DIR}/timing_methods.yaml"
    CMD="$CMD --latex-grids=$LATEX_GRIDS"
    CMD="$CMD --sweep-runs $SWEEP_RUNS"
else
    # Single-baseline 4-UE run with plot grid; no multi-axis ranges
    CMD="$CMD --settings-override grid_size=$GRID_SIZE"
fi

# Run
$CMD

echo "========================================================"
echo "  Experiment complete"
echo "  Plots saved to: $OUTPUT_DIR/plots"
echo "========================================================"
