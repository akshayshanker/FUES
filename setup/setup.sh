#!/bin/bash
# ==========================================================================
#  setup/setup.sh — one script for install + activate + env.
#
#  Usage (always `source`, not `bash` — venv activation must persist):
#
#    source setup/setup.sh              # install-if-missing, then activate + set env
#    source setup/setup.sh --update     # git pull + reinstall dcsmm/kikku, then activate
#
#  Full rebuild:
#    rm -rf $HOME/venvs/fues   # (or .venv on local)
#    source setup/setup.sh
#
#  Detects Gadi (presence of /scratch/tp66) vs local:
#    Gadi  → venv at $HOME/venvs/fues (NFS — needed for concurrent MPI reads)
#    Local → venv at $REPO_ROOT/.venv
#
#  One install profile: pip install -e ".[examples]". Covers durables,
#  retirement, notebooks, Gadi sweeps, estimation flow. HARK + ConSav are
#  in the core deps (EGM_UE benchmarks need them).
# ==========================================================================

# Refuse to run when executed as `bash setup.sh` — the venv activation
# and env exports would vanish with the subshell. Must be `source`d.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: setup.sh must be sourced, not executed directly." >&2
    echo "" >&2
    echo "  You ran:    bash setup/setup.sh" >&2
    echo "  Should be:  source setup/setup.sh" >&2
    echo "" >&2
    echo "Running with bash puts the venv activation in a subshell that" >&2
    echo "exits immediately, leaving your parent shell unchanged." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

UPDATE=0
for arg in "$@"; do
    case "$arg" in
        --update) UPDATE=1 ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            return 0 2>/dev/null || exit 0
            ;;
    esac
done

if [[ -d "/scratch/tp66" ]]; then
    IS_GADI=1
    VENV_DIR="${FUES_VENV:-$HOME/venvs/fues}"
    module purge 2>/dev/null || true
    module load python3/3.12.1
    module load openmpi/4.1.5
else
    IS_GADI=0
    VENV_DIR="${REPO_ROOT}/.venv"
fi

# ---- Install (only if venv is missing) ---------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[setup] Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip --quiet

    if [[ "$IS_GADI" -eq 1 ]]; then
        echo "[setup] Gadi: scientific stack (numpy/numba/scipy/matplotlib)"
        pip install "numpy>=1.26,<2.0" "numba>=0.59" "scipy>=1.12,<2.0" matplotlib --quiet
        pip install EconModel consav --quiet
    fi

    echo "[setup] Installing dcsmm[examples] (editable)"
    pip install -e ".[examples]" --quiet

    echo "[setup] Installing bright-forest dolo fork @ phase1.1_0.1"
    pip install lark multipledispatch --quiet
    pip install --no-deps "dolang @ git+https://github.com/bright-forest/dolang.py.git@phase1.1_0.1" --quiet
    pip install --no-deps "dolo @ git+https://github.com/bright-forest/dolo.git@phase1.1_0.1" --quiet

    if [[ "$IS_GADI" -eq 1 ]]; then
        echo "[setup] Building mpi4py from source against loaded OpenMPI"
        pip install --no-binary :all: mpi4py --quiet
    fi

    echo "[setup] Verifying..."
    python3 -c "from dcsmm.fues import FUES; print('  OK: dcsmm.fues')"
    python3 -c "from HARK.interpolation import LinearInterp; from HARK.dcegm import upper_envelope; print('  OK: HARK (EGM_UE benchmark)')"
    python3 -c "import consav; print('  OK: consav')"
    python3 -c "from kikku.run.estimate import estimate; print('  OK: kikku.run.estimate')"
    python3 -c "from dolo.compiler.spec_factory import load, make; print('  OK: dolo.compiler.spec_factory')"
    python3 -c "import inspect; from kikku.run.sweep import sweep; sig = inspect.signature(sweep); assert 'comm' in sig.parameters and 'on_error' in sig.parameters; print('  OK: kikku.run.sweep exposes comm + on_error')"
    if [[ "$IS_GADI" -eq 1 ]]; then
        python3 -c "from mpi4py import MPI; print(f'  OK: mpi4py ({MPI.Get_library_version().splitlines()[0][:60]})')"
    fi
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# ---- --update: pull + reinstall (no-deps; pinned scientific stack) -----
if [[ "$UPDATE" -eq 1 ]]; then
    echo "[setup] git pull"
    git pull
    echo "[setup] Reinstalling dcsmm + kikku (--no-deps)"
    pip install -e ".[examples]" --no-deps --quiet
    pip install --force-reinstall --no-deps \
        "kikku[estimation] @ git+https://github.com/bright-forest/kikku.git" --quiet
fi

# ---- Runtime environment ------------------------------------------------
export NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
if [[ "$IS_GADI" -eq 1 ]]; then
    export NUMBA_CACHE_DIR="${PBS_JOBFS:-/scratch/tp66/${USER}/numba_cache}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/scratch/tp66/${USER}/mplconfig}"
    mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" 2>/dev/null || true
fi

echo "[setup] ready"
echo "  python: $(which python3)"
echo "  venv:   $VIRTUAL_ENV"
echo "  repo:   $(pwd)"
echo "  HEAD:   $(git log --oneline -1 2>/dev/null || echo 'no git')"
