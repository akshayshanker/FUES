#!/bin/bash
# ==========================================================================
#  Setup script for dcsmm + kikku.
#
#  Creates a virtual environment with all dependencies for solving,
#  simulating, and estimating DDSL models.
#
#  Works on both NCI Gadi and a local laptop/desktop.
#
#  Usage on Gadi (login node or interactive PBS):
#
#    cd /home/141/as3442/dev/fues.dev/FUES
#    bash setup/setup_venv.sh
#
#  Usage on laptop:
#
#    cd /path/to/FUES
#    bash setup/setup_venv.sh
#
# ==========================================================================
set -euo pipefail

# Detect repo root (parent of setup/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Detect environment: Gadi (has /scratch/tp66) vs local
if [[ -d "/scratch/tp66" ]]; then
    echo "Detected NCI Gadi environment"
    VENV_DIR="/scratch/tp66/${USER}/venvs/fues"
    module purge
    module load python3/3.11.0
    module load openmpi/4.1.5
else
    echo "Detected local environment"
    VENV_DIR="${REPO_ROOT}/.venv"
fi

python3 --version

echo ""
echo "=== Step 1: Create venv at ${VENV_DIR} ==="
if [[ -d "${VENV_DIR}" ]]; then
    echo "Removing existing venv..."
    rm -rf "${VENV_DIR}"
fi
# Do NOT use --system-site-packages (avoids numpy/numba conflicts from system packages)
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet

echo ""
echo "=== Step 2: Core numerical stack (pinned for numba compatibility) ==="
pip install "numpy>=1.26,<2.0" --quiet
pip install "numba>=0.59" --quiet
pip install "scipy>=1.12" --quiet

echo ""
echo "=== Step 3: Install dcsmm (editable) + all dependencies ==="
cd "${REPO_ROOT}"
pip install -e ".[examples]" --quiet

echo ""
echo "=== Step 4: Install kikku from GitHub (with estimation extras) ==="
# NOT editable (-e) — editable puts source on /scratch/ (Lustre) which causes
# BrokenPipeError at scale (520+ MPI ranks all reading .py files simultaneously).
# Non-editable installs compiled .pyc into site-packages/ which handles concurrent reads.
pip install "kikku[estimation] @ git+https://github.com/bright-forest/kikku.git" --quiet

echo ""
echo "=== Step 5: Install dolang + dolo (bright-forest phase1.1_0.1) ==="
pip install lark multipledispatch --quiet
pip install --no-deps "dolang @ git+https://github.com/bright-forest/dolang.py.git@phase1.1_0.1" --quiet
pip install --no-deps "dolo @ git+https://github.com/bright-forest/dolo.git@phase1.1_0.1" --quiet

echo ""
echo "=== Step 6: MPI (Gadi only — build from source against system OpenMPI) ==="
if [[ -d "/scratch/tp66" ]]; then
    pip install --no-binary :all: mpi4py --quiet
    echo "mpi4py built from source"
else
    echo "Skipping mpi4py (local dev — install manually if needed)"
fi

echo ""
echo "=== Step 7: Verify ==="
python3 -c "import numpy; print(f'numpy {numpy.__version__}')"
python3 -c "import numba; print(f'numba {numba.__version__}')"
python3 -c "import scipy; print(f'scipy {scipy.__version__}')"
python3 -c "from dcsmm.fues import FUES; print('OK: dcsmm.fues.FUES')"
python3 -c "from dcsmm.uenvelope import EGM_UE; print('OK: dcsmm.uenvelope.EGM_UE')"
python3 -c "from kikku.run.estimate import estimate; print('OK: kikku.run.estimate')"
python3 -c "from kikku.run.moments import make_moment_fn; print('OK: kikku.run.moments')"
python3 -c "from kikku.dynx import load_syntax; print('OK: kikku.dynx')"
python3 -c "import dolo; print('OK: dolo')"
python3 -c "import yaml; print('OK: pyyaml')"

if [[ -d "/scratch/tp66" ]]; then
    python3 -c "from mpi4py import MPI; print(f'OK: mpi4py (ranks={MPI.COMM_WORLD.Get_size()})')"
fi

echo ""
pip list 2>/dev/null | grep -i -E "dcsmm|kikku|numba|numpy|scipy|mpi4py|dolo|pandas|pyyaml"

echo ""
echo "=== Done ==="
echo "Venv:     ${VENV_DIR}"
echo "Activate: source ${VENV_DIR}/bin/activate"
