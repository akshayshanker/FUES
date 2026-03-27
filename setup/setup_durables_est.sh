#!/bin/bash
# ==========================================================================
#  Lean venv for durables estimation on Gadi.
#
#  No HARK, no consav, no sympy, no matplotlib.
#  Just: dcsmm (FUES) + kikku (moments/CE) + dolo (calibration) + mpi4py.
#
#  Usage:
#    cd /home/141/as3442/dev/fues.dev/FUES
#    bash setup/setup_durables_est.sh
#
# ==========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module purge
module load python3/3.12.1
module load openmpi/4.1.5
python3 --version

VENV_DIR="$HOME/venvs/fues"

echo ""
echo "=== Step 1: Create clean venv at ${VENV_DIR} ==="
if [[ -d "${VENV_DIR}" ]]; then
    echo "Removing existing venv..."
    rm -rf "${VENV_DIR}"
fi
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet

echo ""
echo "=== Step 2: Core stack (pinned for numba compat) ==="
pip install "numpy>=1.26,<2.0" --quiet
pip install "numba>=0.59" --quiet
pip install "scipy>=1.12,<2.0" --quiet

echo ""
echo "=== Step 3: Install dcsmm + kikku (durables-est profile) ==="
cd "$REPO_ROOT"
pip install -e ".[durables-est]" --quiet

echo ""
echo "=== Step 4: Install dolang + dolo (lazy init — no sympy at import) ==="
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
python3 -c "from dcsmm.fues import FUES; print('OK: dcsmm.fues')"
python3 -c "from kikku.run.estimate import estimate; print('OK: kikku.run.estimate')"
python3 -c "from kikku.run.moments import make_moment_fn; print('OK: kikku.run.moments')"
python3 -c "from dolo.compiler.calibration import calibrate; print('OK: dolo.compiler.calibration')"
python3 -c "from mpi4py import MPI; print('OK: mpi4py')"

# Confirm no heavy imports
python3 -c "
import sys
before = set(sys.modules.keys())
from dolo.compiler.calibration import calibrate
after = set(sys.modules.keys())
n = len(after - before)
has_sympy = any('sympy' in m for m in (after-before))
print(f'dolo.compiler.calibration imports {n} modules, sympy={has_sympy}')
"

echo ""
echo "=== Done ==="
echo "Venv:     ${VENV_DIR}"
echo "Activate: source ${VENV_DIR}/bin/activate"
echo "Profile:  durables-est (lean — no HARK/sympy/matplotlib)"
