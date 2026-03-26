#!/bin/bash
# Setup venv for estimation on Gadi.
# Run this ONCE on a login node before submitting PBS jobs.
#
# Usage:
#   cd /home/141/as3442/dev/fues.dev/FUES
#   bash experiments/durables/estimation/setup_venv.sh

set -euo pipefail

module purge
module load python3/3.11.0
module load openmpi/4.1.5

VENV=/scratch/tp66/$USER/venvs/fues

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV..."
    python3 -m venv --system-site-packages "$VENV"
fi

source "$VENV/bin/activate"

echo "Installing FUES (dcsmm)..."
pip install -e . --quiet

echo "Installing kikku from GitHub..."
pip install -e "git+https://github.com/bright-forest/kikku.git#egg=kikku[estimation]" --quiet

echo "Building mpi4py from source..."
pip install --no-binary :all: mpi4py --quiet

echo ""
echo "Venv ready at: $VENV"
echo "Python: $(which python3)"
echo "Packages:"
pip list 2>/dev/null | grep -E "dcsmm|kikku|mpi4py|numba|numpy"
