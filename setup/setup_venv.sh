#!/bin/bash
# ==========================================================================
#  Setup script for dcsmm + kikku.
#
#  On Gadi: installs into ~/.local/ (user site-packages on /home/ NFS).
#    This avoids Lustre BrokenPipeError at scale (520+ MPI ranks).
#    /home/ NFS handles concurrent reads; /scratch/ Lustre does not.
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
    echo "Detected NCI Gadi — installing to ~/.local/ (NFS, not Lustre)"

    module purge
    module load python3/3.12.1
    module load openmpi/4.1.5
    python3 --version

    echo ""
    echo "=== Step 1: Core numerical stack ==="
    pip install --user "numpy>=1.26,<2.0" --quiet
    pip install --user "numba>=0.59" --quiet
    pip install --user "scipy>=1.12,<2.0" --quiet

    echo ""
    echo "=== Step 2: Install dcsmm (editable) ==="
    cd "$REPO_ROOT"
    pip install --user -e ".[examples]" --quiet

    echo ""
    echo "=== Step 3: Install kikku from GitHub ==="
    pip install --user "kikku[estimation] @ git+https://github.com/bright-forest/kikku.git" --quiet

    echo ""
    echo "=== Step 4: Install dolang + dolo ==="
    pip install --user lark multipledispatch --quiet
    pip install --user --no-deps "dolang @ git+https://github.com/bright-forest/dolang.py.git@phase1.1_0.1" --quiet
    pip install --user --no-deps "dolo @ git+https://github.com/bright-forest/dolo.git@phase1.1_0.1" --quiet

    echo ""
    echo "=== Step 5: Build mpi4py from source ==="
    pip install --user --no-binary :all: mpi4py --quiet

    echo ""
    echo "=== Step 6: Verify ==="
    python3 -c "import numpy; print(f'numpy {numpy.__version__}')"
    python3 -c "import numba; print(f'numba {numba.__version__}')"
    python3 -c "import scipy; print(f'scipy {scipy.__version__}')"
    python3 -c "from dcsmm.fues import FUES; print('OK: dcsmm.fues')"
    python3 -c "from kikku.run.estimate import estimate; print('OK: kikku.run.estimate')"
    python3 -c "from kikku.dynx import load_syntax; print('OK: kikku.dynx')"
    python3 -c "import dolo; print('OK: dolo')"
    python3 -c "from mpi4py import MPI; print(f'OK: mpi4py')"

    echo ""
    echo "=== Done (Gadi) ==="
    echo "Packages installed to: ~/.local/lib/python3.12/site-packages/"
    echo "No venv activation needed — just: module load python3/3.12.1"

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
