#!/bin/bash
# ==========================================================================
#  Quick update: pull latest code, reinstall kikku, activate venv.
#
#  Run on Gadi before submitting estimation jobs:
#    cd ~/dev/fues.dev/FUES
#    source setup/update_and_activate.sh
#
#  What it does:
#    1. Activates the fues venv
#    2. Pulls latest FUES code
#    3. Reinstalls kikku from GitHub (picks up CE/moment fixes)
#    4. Reinstalls dcsmm in editable mode (picks up estimate.py changes)
#    5. Prints versions and confirms
#
#  Note: use `source` not `bash` — the venv activation must persist
#  in your shell for subsequent qsub commands.
# ==========================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Venv path — override with: VENV=~/venvs/other source setup/update_and_activate.sh
VENV="${VENV:-$HOME/venvs/fues}"

module purge
module load python3/3.12.1
module load openmpi/4.1.5

echo ""
echo "=== Activating venv ==="
source "$VENV/bin/activate"
echo "  Python: $(python3 --version)"
echo "  Venv:   $VIRTUAL_ENV"

echo ""
echo "=== Pulling latest FUES ==="
git pull origin durables-ddsl-phase2
echo "  HEAD: $(git log --oneline -1)"

echo ""
echo "=== Reinstalling kikku (from GitHub main) ==="
pip install --force-reinstall --no-deps \
    "kikku @ git+https://github.com/bright-forest/kikku.git" --quiet
python3 -c "import kikku; print(f'  kikku installed from: {kikku.__file__}')"

echo ""
echo "=== Reinstalling dcsmm (editable, examples profile) ==="
pip install -e ".[examples]" --quiet --no-deps
python3 -c "import dcsmm; print(f'  dcsmm installed from: {dcsmm.__file__}')"

echo ""
echo "=== Verify ==="
python3 -c "from kikku.run.estimate import estimate; print('  OK: kikku.run.estimate')"
python3 -c "from kikku.run.moments import make_moment_fn; print('  OK: kikku.run.moments')"
python3 -c "from HARK.interpolation import LinearInterp; print('  OK: HARK (retirement plots)')"
python3 -c "from examples.durables.run import main; print('  OK: examples.durables.run')"
python3 -c "from examples.retirement.run import main; print('  OK: examples.retirement.run')"

echo ""
echo "=== Ready ==="
echo "  Repo: $REPO_ROOT"
echo "  HEAD: $(git log --oneline -1)"
echo ""
echo "  Submit jobs with:"
echo "    qsub experiments/durables/estimation/run_estimation.pbs"
echo "    qsub experiments/durables/estimation/run_large_egm.pbs"
echo "    etc."
