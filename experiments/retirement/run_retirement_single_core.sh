#!/bin/bash
#PBS -N fues-retirement-sc
#PBS -P tp66
#PBS -q expresssr
#PBS -l ncpus=1,mem=5GB,walltime=01:00:00
#PBS -l storage=scratch/tp66
#PBS -j oe
#PBS -o /g/data/tp66/logs/retirement/
#PBS -e /g/data/tp66/logs/retirement/
#
# Single-core retirement model runner (NO sweep) that saves ALL outputs to scratch.
#
# - Runs `examples/retirement/run.py` (baseline solve + plots)
# - Uses the dcsmm venv (defaults to /scratch/tp66/$USER/venvs/dcsmm)
# - Ensures repo + src are on PYTHONPATH for `dcsmm` imports
#
# Usage:
#   qsub experiments/retirement/run_retirement_single_core.sh
#
# Optional overrides (via env vars):
#   PARAMS_FILE="params/baseline.yml"
#   GRID_SIZE=3000
#   PLOT_AGE=17
#   OUTPUT_BASE="/scratch/tp66/$USER/FUES/solutions/retirement"
#   DCSMM_VENV="/scratch/tp66/$USER/venvs/dcsmm"
#

set -euo pipefail

# -----------------------------------------------------------------------------
# Resolve repo root - FUES project is at fixed location
# -----------------------------------------------------------------------------
REPO_ROOT="/home/141/as3442/dev/fues.dev/FUES"
EXAMPLE_DIR="${REPO_ROOT}/examples/retirement"

echo "REPO_ROOT: ${REPO_ROOT}"

# -----------------------------------------------------------------------------
# User-configurable inputs (defaults)
# -----------------------------------------------------------------------------
PARAMS_FILE="${PARAMS_FILE:-params/baseline.yml}"
GRID_SIZE="${GRID_SIZE:-3000}"
PLOT_AGE="${PLOT_AGE:-17}"

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/tp66/${USER}/FUES/retirement}"
# make_run_dir creates YYYY-MM-DD/NNN/ inside OUTPUT_DIR automatically

LOG_DIR="/scratch/tp66/${USER}/FUES/logs/retirement"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

LOG_FILE="${LOG_DIR}/retirement_${RUN_ID}.log"
ERR_FILE="${LOG_DIR}/retirement_${RUN_ID}.err"

echo "========================================================" | tee "${LOG_FILE}"
echo "Retirement model (single-core, no sweep)" | tee -a "${LOG_FILE}"
echo "Repo: ${REPO_ROOT}" | tee -a "${LOG_FILE}"
echo "Params: ${PARAMS_FILE}" | tee -a "${LOG_FILE}"
echo "Grid size: ${GRID_SIZE}" | tee -a "${LOG_FILE}"
echo "Plot age: ${PLOT_AGE}" | tee -a "${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Err: ${ERR_FILE}" | tee -a "${LOG_FILE}"
echo "========================================================" | tee -a "${LOG_FILE}"

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
module load python3/3.12.1

DCSMM_VENV="${DCSMM_VENV:-/scratch/tp66/${USER}/venvs/dcsmm}"
if [[ ! -f "${DCSMM_VENV}/bin/activate" ]]; then
  {
    echo "ERROR: dcsmm venv not found at: ${DCSMM_VENV}"
    echo "Run: bash scripts/setup_venv.sh"
  } | tee -a "${LOG_FILE}" >&2
  exit 1
fi
source "${DCSMM_VENV}/bin/activate"

# Numba cache on PBS_JOBFS (fast local SSD, fresh each job)
export NUMBA_CACHE_DIR=$PBS_JOBFS
export NUMBA_NUM_THREADS=1

# Avoid writing any __pycache__ into the repo on $HOME
export PYTHONDONTWRITEBYTECODE=1

# Ensure headless plotting
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/scratch/tp66/${USER}/mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

# Make repo + src importable (fixes `import dcsmm` without pip install)
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

{
  echo ""
  echo "--- Python environment ---"
  which python3
  python3 -c "import sys; print('python:', sys.executable); print('sys.path[0]:', sys.path[0])"
  echo ""
  echo "--- Import check (dcsmm + plotting deps) ---"
  python3 -c "import dcsmm; from dcsmm.fues import FUES; import HARK; print('dcsmm:', dcsmm.__file__)"
  echo ""
  echo "--- Running retirement baseline + plots ---"
} | tee -a "${LOG_FILE}"

# Resolve params file relative to experiments/retirement/
if [[ ! -f "${PARAMS_FILE}" ]]; then
    PARAMS_FILE="${REPO_ROOT}/experiments/retirement/${PARAMS_FILE}"
fi

python3 -m examples.retirement.run \
  --override-file "${PARAMS_FILE}" \
  --setting-override grid_size="${GRID_SIZE}" \
  --setting-override plot_age="${PLOT_AGE}" \
  --output-dir "${OUTPUT_DIR}" \
  1> >(tee -a "${LOG_FILE}") \
  2> >(tee -a "${ERR_FILE}" >&2)

echo "" | tee -a "${LOG_FILE}"
echo "========================================================" | tee -a "${LOG_FILE}"
echo "Done." | tee -a "${LOG_FILE}"
echo "Outputs (plots/tables) saved to: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "========================================================" | tee -a "${LOG_FILE}"
