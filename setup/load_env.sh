#!/bin/bash
# Activate the dcsmm venv and set runtime environment.
# Works on both Gadi and local.
#
# Usage: source setup/load_env.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -d "/scratch/tp66" ]]; then
    module load python3/3.12.1
    source /scratch/tp66/${USER}/venvs/dcsmm/bin/activate

    # Numba: use fast local SSD for cache if on a compute node, else scratch
    export NUMBA_CACHE_DIR="${PBS_JOBFS:-/scratch/tp66/${USER}/numba_cache}"
    export NUMBA_NUM_THREADS=1

    # Avoid writing __pycache__ to home
    export PYTHONDONTWRITEBYTECODE=1

    # Headless plotting
    export MPLBACKEND=Agg
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/scratch/tp66/${USER}/mplconfig}"
    mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" 2>/dev/null
else
    source "${REPO_ROOT}/.venv/bin/activate"
    export NUMBA_NUM_THREADS=1
fi

cd "${REPO_ROOT}"

echo "Environment ready:"
echo "  Python: $(which python3)"
echo "  Venv:   ${VIRTUAL_ENV}"
echo "  Repo:   $(pwd)"
echo "  NUMBA_CACHE_DIR: ${NUMBA_CACHE_DIR:-default}"
echo "  NUMBA_NUM_THREADS: ${NUMBA_NUM_THREADS}"
