#!/usr/bin/env bash
# MPI demo sweep for the housing-renting model
# (requires mpi4py and OpenMPI; run inside a PBS job with mpiprocs = N_PROCS)
# ------------------------------------------------------------------
set -euo pipefail

# ------------------------------------------------------------------
# Resolve project root (= one directory above this script)
# ------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Make repo imports (examples.…) work everywhere
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Load Python with mpi4py (adjust version if needed)
module load python3/3.12.1 2>/dev/null || true
module load openmpi/5.0.5

# ------------------------------------------------------------------
# Paths and sweep configuration
# ------------------------------------------------------------------
PY_SCRIPT="${ROOT_DIR}/experiments/housing_renting/param_sweep.py"
OUT_DIR="${ROOT_DIR}/results"
OUT_FILE="${OUT_DIR}/mpi_sweep.csv"
mkdir -p "${OUT_DIR}"

# ---- parameter grid -------------------------------------------------
BETA_MIN=0.91
BETA_MAX=0.99
BETA_N=4

GAMMA_MIN=1.5
GAMMA_MAX=6.0
GAMMA_N=3


UE_METHODS="FUES,CONSAV,RFC,DCEGM"        # <— edit here
IFS=',' read -r -a METHOD_ARR <<< "$UE_METHODS"
METHODS_N=${#METHOD_ARR[@]}


# Total draws = product of grid sizes × methods
N_PROCS=$(( BETA_N * GAMMA_N * METHODS_N ))   # → 4

# ------------------------------------------------------------------
# Launch the sweep with MPI
# ------------------------------------------------------------------
mpiexec -n "$N_PROCS" python3 "$PY_SCRIPT"             \
    --param master.parameters.beta="${BETA_MIN}:${BETA_MAX}:${BETA_N}" \
    --param master.parameters.gamma="${GAMMA_MIN}:${GAMMA_MAX}:${GAMMA_N}" \
    --ue-methods "$UE_METHODS" \
    --use-mpi \
    --n-procs "$N_PROCS" \
    --output "$OUT_FILE"

