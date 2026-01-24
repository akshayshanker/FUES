#!/bin/bash
#PBS -N fues-retirement-sc
#PBS -P tp66
#PBS -q expresssr
#PBS -l ncpus=1,mem=10GB,walltime=01:00:00
#PBS -l storage=scratch/tp66
#PBS -j oe
#PBS -o /dev/null
#PBS -e /dev/null
#
# Single-core retirement model runner (NO sweep) that saves ALL outputs to scratch.
#
# - Runs `experiments/retirement/run_experiment.py` (baseline solve + plots)
# - Uses the public venv (defaults to /scratch/tp66/$USER/venvs/fues_public)
# - Fixes `ModuleNotFoundError: dc_smm` by ensuring repo + src are on PYTHONPATH
#
# Usage:
#   qsub experiments/retirement/run_retirement_single_core.sh
#
# Optional overrides (via env vars):
#   PARAMS_FILE="params/baseline.yml"
#   GRID_SIZE=3000
#   PLOT_AGE=17
#   OUTPUT_BASE="/scratch/tp66/$USER/FUES/solutions/retirement"
#   VENV_PUBLIC="/scratch/tp66/$USER/venvs/fues_public"
#

set -euo pipefail

# -----------------------------------------------------------------------------
# Resolve repo root - FUES project is at fixed location
# -----------------------------------------------------------------------------
REPO_ROOT="/home/141/as3442/dev/fues.dev/FUES"
SCRIPT_DIR="${REPO_ROOT}/experiments/retirement"

echo "REPO_ROOT: ${REPO_ROOT}"

# -----------------------------------------------------------------------------
# User-configurable inputs (defaults)
# -----------------------------------------------------------------------------
PARAMS_FILE="${PARAMS_FILE:-params/baseline.yml}"
GRID_SIZE="${GRID_SIZE:-3000}"
PLOT_AGE="${PLOT_AGE:-17}"

OUTPUT_BASE="${OUTPUT_BASE:-/scratch/tp66/${USER}/FUES/solutions/retirement}"
RUN_ID="${RUN_ID:-baseline_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_ID}"

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

VENV_PUBLIC="${VENV_PUBLIC:-/scratch/tp66/${USER}/venvs/fues_public}"
if [[ ! -f "${VENV_PUBLIC}/bin/activate" ]]; then
  {
    echo "ERROR: Public venv not found at: ${VENV_PUBLIC}"
    echo "Set VENV_PUBLIC to the correct path, or create it on scratch."
  } | tee -a "${LOG_FILE}" >&2
  exit 1
fi
source "${VENV_PUBLIC}/bin/activate"

# Numba cache on PBS_JOBFS (fast local SSD, fresh each job)
export NUMBA_CACHE_DIR=$PBS_JOBFS
export NUMBA_NUM_THREADS=1

# Avoid writing any __pycache__ into the repo on $HOME
export PYTHONDONTWRITEBYTECODE=1

# Ensure headless plotting
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/scratch/tp66/${USER}/mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

# Make repo + src importable (fixes `import dc_smm` without pip install)
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

{
  echo ""
  echo "--- Python environment ---"
  which python3
  python3 -c "import sys; print('python:', sys.executable); print('sys.path[0]:', sys.path[0])"
  echo ""
  echo "--- Import check (dc_smm + plotting deps) ---"
  python3 -c "import dc_smm; import dc_smm.models.retirement.retirement as r; import HARK; print('dc_smm:', dc_smm.__file__)"
  echo ""
  echo "--- Running retirement baseline + plots ---"
} | tee -a "${LOG_FILE}"

python3 "${SCRIPT_DIR}/run_experiment.py" \
  --params "${PARAMS_FILE}" \
  --grid-size "${GRID_SIZE}" \
  --plot-age "${PLOT_AGE}" \
  --output-dir "${OUTPUT_DIR}" \
  1> >(tee -a "${LOG_FILE}") \
  2> >(tee -a "${ERR_FILE}" >&2)

echo "" | tee -a "${LOG_FILE}"
echo "========================================================" | tee -a "${LOG_FILE}"
echo "Done." | tee -a "${LOG_FILE}"
echo "Outputs (plots/tables) saved to: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "========================================================" | tee -a "${LOG_FILE}"

