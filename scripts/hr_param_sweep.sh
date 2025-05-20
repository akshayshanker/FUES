#!/usr/bin/env bash
# Single-core sweep for the housing-renting model
# ------------------------------------------------------------------
set -euo pipefail

# ------------------------------------------------------------------
# Resolve project root (= one directory above this script)
# ------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Load Python (no MPI module needed)
module load python3/3.12.1 2>/dev/null || true

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PY_SCRIPT="${ROOT_DIR}/experiments/housing_renting/param_sweep.py"
OUT_DIR="${ROOT_DIR}/results"
OUT_FILE="${OUT_DIR}/single_core_sweep.csv"
mkdir -p "${OUT_DIR}"

# ---- parameter grid -------------------------------------------------
BETA_MIN=0.91
BETA_MAX=0.99
BETA_N=2

GAMMA_MIN=1.5
GAMMA_MAX=6.0
GAMMA_N=2

UE_METHODS="FUES,CONSAV"          # edit as needed

# (optional) echo how many runs we expect
IFS=',' read -r -a METHOD_ARR <<< "$UE_METHODS"
METHODS_N=${#METHOD_ARR[@]}
TOTAL_RUNS=$(( METHODS_N * BETA_N * GAMMA_N ))
echo "Will run ${TOTAL_RUNS} parameter points on a single process."

# ------------------------------------------------------------------
# Launch (serial)
# ------------------------------------------------------------------
python3 "$PY_SCRIPT" \
    --param master.parameters.beta="${BETA_MIN}:${BETA_MAX}:${BETA_N}" \
    --param master.parameters.gamma="${GAMMA_MIN}:${GAMMA_MAX}:${GAMMA_N}" \
    --ue-methods "$UE_METHODS" \
    --output "$OUT_FILE" \
    "$@"

echo "Detailed results written to ${OUT_FILE}"
