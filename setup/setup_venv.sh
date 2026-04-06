#!/bin/bash
# ==========================================================================
#  Setup script for dcsmm + kikku.
#
#  On Gadi: creates a venv on /home/ (NFS) — NOT /scratch/ (Lustre).
#    /home/ NFS handles concurrent MPI reads at scale (520+ ranks).
#    /scratch/ Lustre causes BrokenPipeError under concurrent import load.
#    NO --system-site-packages — full isolation from /apps/ packages.
#
#  Locally: creates a .venv in the repo root.
#
#  Usage on Gadi (login node):
#    cd /home/141/as3442/dev/fues.dev/FUES
#    bash setup/setup_venv.sh
#
#  Usage locally:
#    cd /path/to/FUES
#    bash setup/setup_venv.sh
#
# ==========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -d "/scratch/tp66" ]]; then
    # ===================== GADI =====================
    echo "Detected NCI Gadi — creating venv on /home/ (NFS)"

    module purge
    module load python3/3.12.1
    module load openmpi/4.1.5
    python3 --version

    # Venv on /home/ — NFS handles concurrent reads, fully isolated
    VENV_DIR="$HOME/venvs/fues"

    echo ""
    echo "=== Step 1: Create clean venv at ${VENV_DIR} ==="
    if [[ -d "${VENV_DIR}" ]]; then
        echo "Removing existing venv..."
        rm -rf "${VENV_DIR}"
    fi
    # NO --system-site-packages — avoids numpy/matplotlib version conflicts
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip --quiet

    echo ""
    echo "=== Step 2: Core numerical stack ==="
    pip install "numpy>=1.26,<2.0" --quiet
    pip install "numba>=0.59" --quiet
    pip install "scipy>=1.12,<2.0" --quiet
    pip install matplotlib --quiet
    pip install EconModel consav --quiet

    echo ""
    echo "=== Step 3: Install dcsmm (editable, durables-est profile) ==="
    cd "$REPO_ROOT"
    # Core deps include consav and quantecon (used in horses code).
    # durables-est adds kikku[estimation] + pyyaml.
    pip install -e ".[durables-est]" --quiet

    echo ""
    echo "=== Step 4: Install dolang + dolo ==="
    pip install lark multipledispatch --quiet
    pip install --no-deps "dolang @ git+https://github.com/bright-forest/dolang.py.git@phase1.1_0.1" --quiet
    pip install --no-deps "dolo @ git+https://github.com/bright-forest/dolo.git@phase1.1_0.1" --quiet

    echo ""
    echo "=== Step 5: Build mpi4py from source ==="
    pip install --no-binary :all: mpi4py --quiet

    echo ""
    echo "=== Step 6: Verify ==="
    python3 -c "import numpy; print(f'numpy {numpy.__version__}')"
    python3 -c "import numba; print(f'numba {numba.__version__}')"
    python3 -c "import scipy; print(f'scipy {scipy.__version__}')"
    python3 -c "import matplotlib; print(f'matplotlib {matplotlib.__version__}')"
    python3 -c "from dcsmm.fues import FUES; print('OK: dcsmm.fues')"
    python3 -c "from kikku.run.estimate import estimate; print('OK: kikku.run.estimate')"
    python3 -c "from kikku.dynx import load_syntax; print('OK: kikku.dynx')"
    python3 -c "import dolo; print('OK: dolo')"
    python3 -c "from mpi4py import MPI; print(f'OK: mpi4py')"

    echo ""
    echo "=== Done (Gadi) ==="
    echo "Venv:     ${VENV_DIR}"
    echo "Activate: source ${VENV_DIR}/bin/activate"

else
    # ===================== LOCAL =====================
    echo "Detected local environment"
    VENV_DIR="${REPO_ROOT}/.venv"
    python3 --version

    echo ""
    echo "=== Step 1: Create venv at ${VENV_DIR} ==="
    if [[ -d "${VENV_DIR}" ]]; then
        echo "Removing existing venv..."
        rm -rf "${VENV_DIR}"
    fi
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip --quiet

    echo ""
    echo "=== Step 2: Install everything ==="
    cd "$REPO_ROOT"
    pip install -e ".[examples]" --quiet
    pip install "kikku[estimation] @ git+https://github.com/bright-forest/kikku.git" --quiet
    pip install lark multipledispatch --quiet
    pip install --no-deps "dolang @ git+https://github.com/bright-forest/dolang.py.git@phase1.1_0.1" --quiet
    pip install --no-deps "dolo @ git+https://github.com/bright-forest/dolo.git@phase1.1_0.1" --quiet

    echo ""
    echo "=== Step 3: Verify ==="
    python3 -c "import numpy; print(f'numpy {numpy.__version__}')"
    python3 -c "import numba; print(f'numba {numba.__version__}')"
    python3 -c "from dcsmm.fues import FUES; print('OK: dcsmm.fues')"
    python3 -c "from kikku.run.estimate import estimate; print('OK: kikku.run.estimate')"
    python3 -c "import dolo; print('OK: dolo')"

    echo ""
    echo "=== Done (local) ==="
    echo "Activate: source ${VENV_DIR}/bin/activate"
fi
