#!/bin/bash
# ==========================================================================
#  Setup script for dcsmm.
#
#  For anyone who clones the FUES repo and wants an editable install
#  of dcsmm — run this once to create a virtual environment with all
#  dependencies. Code changes in src/dcsmm/ take effect immediately.
#
#  Works on both NCI Gadi and a local laptop/desktop.
#
#  REPO_ROOT  Root of the cloned FUES repository (auto-detected).
#
#  VENV_DIR   Virtual environment directory. Contains Python, pip, and
#             all installed packages. Activate it before running code.
#             - On Gadi: /scratch/tp66/$USER/venvs/dcsmm
#             - Locally: .venv inside the repo
#
#  Usage on Gadi (inside an interactive PBS session):
#
#    qsub -I -q expresssr -P tp66 -l ncpus=1,mem=8GB,walltime=01:00:00,storage=scratch/tp66,wd
#    cd /home/141/as3442/dev/fues.dev/FUES
#    bash scripts/setup_venv.sh
#
#  Usage on laptop:
#
#    cd /path/to/FUES
#    bash scripts/setup_venv.sh
#
# ==========================================================================
set -euo pipefail

# Detect repo root (parent of scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Detect environment: Gadi (has /scratch/tp66) vs local
if [[ -d "/scratch/tp66" ]]; then
    echo "Detected NCI Gadi environment"
    VENV_DIR="/scratch/tp66/${USER}/venvs/dcsmm"
    module load python3/3.12.1
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
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip

echo ""
echo "=== Step 2: Install dcsmm (editable) + all dependencies ==="
cd "${REPO_ROOT}"
pip install -e ".[examples]"

echo ""
echo "=== Step 3: Install dolang + dolo (bright-forest phase1.1_0.1, no-deps) ==="
pip install lark multipledispatch  # dolang/dolo deps not pulled by dcsmm
pip install --no-deps "dolang @ git+https://github.com/bright-forest/dolang.py.git@phase1.1_0.1"
pip install --no-deps "dolo @ git+https://github.com/bright-forest/dolo.git@phase1.1_0.1"

echo ""
echo "=== Step 4: Verify ==="
python3 -c "from dcsmm.fues import FUES; print('OK: dcsmm.fues.FUES')"
python3 -c "from dcsmm.uenvelope import EGM_UE; print('OK: dcsmm.uenvelope.EGM_UE')"
python3 -c "import HARK; print('OK: HARK (econ-ark)')"
python3 -c "from dcsmm.uenvelope.upperenvelope import _consav_ue; assert _consav_ue is not None; print('OK: consav.upperenvelope (via dcsmm)')"
python3 -c "import yaml; print('OK: pyyaml')"
python3 -c "import matplotlib; print('OK: matplotlib')"
python3 -c "from kikku.period_graphs import period_to_graph; print('OK: kikku')"
python3 -c "import dolo; print('OK: dolo')"
pip list | grep -i -E "dcsmm|numba|numpy|scipy|HARK|consav|interpolation|dill|matplotlib|seaborn|pyyaml|dolo"

echo ""
echo "=== Done ==="
echo "Venv:     ${VENV_DIR}"
echo "Activate: source ${VENV_DIR}/bin/activate"
