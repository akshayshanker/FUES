#!/usr/bin/env bash
# Single-core (non-MPI) demo sweep for the housing-renting model
# ------------------------------------------------------------------
set -euo pipefail

# ------------------------------------------------------------------
# Resolve project root (= one directory above this script)
# ------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Add repo root to PYTHONPATH so "examples.…" imports work everywhere
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# (Optional) load the Python module on NCI – comment out if you activate
# your own venv.  Remove or change the version if needed.
module load python3/3.12.1 2>/dev/null || true

# ------------------------------------------------------------------
# Path to the Python driver & output location
# ------------------------------------------------------------------
PY_SCRIPT="${ROOT_DIR}/experiments/housing_renting/param_sweep.py"
OUT_DIR="${ROOT_DIR}/results"
OUT_FILE="${OUT_DIR}/single_core.csv"
mkdir -p "${OUT_DIR}"

# ------------------------------------------------------------------
# Launch the sweep (no --use-mpi ⇒ single process)
# Extra CLI flags handed to this wrapper are forwarded.
# ------------------------------------------------------------------
python3 "${PY_SCRIPT}" \
    --param master.parameters.beta=0.95:0.99:2 \
    --param master.parameters.gamma=2.5:5.0:2 \
    --ue-method FUES \
    --output "${OUT_FILE}" \
    "$@" 