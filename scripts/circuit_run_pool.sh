#!/bin/bash
# ---- PBS resources -------------------------------------------------
#PBS -l ncpus=48,mem=192GB,walltime=05:00:00
#PBS -l storage=scratch/tp66
#PBS -q normal
#PBS -P tp66
#PBS -l wd
#PBS -j oe
#---------------------------------------------------------------------

set -eu   # abort on first error

# -------- RUN-SPECIFIC SETTINGS -------------------------------------
BUNDLE_PREFIX="HR_test_v5_pool"   # <- change me for each experiment
VFI_NGRID=1E3                   # <- actual n_grid (# of a′ points)

# job name shows up in qstat / nqstat
# (safe: max 15 chars on Gadi, so trim if needed)
#PBS -N "${BUNDLE_PREFIX}"

OUT_ROOT="/scratch/tp66/${USER}/FUES/solutions/${BUNDLE_PREFIX}_${VFI_NGRID}"

echo "Bundle prefix :  ${BUNDLE_PREFIX}"
echo "VFI grid size :  ${VFI_NGRID}"
echo "Output root   :  ${OUT_ROOT}"
echo "-----------------------------------------------------------------"

# -------- Modules + venv --------------------------------------------
module purge
module load python3/3.12.1

VENV_ROOT=/scratch/tp66/${USER}/venvs
source "${VENV_ROOT}/fues02-py3121/bin/activate"

# -------- Code location & PYTHONPATH --------------------------------
export FUES_HOME=$HOME/dev/fues.dev/FUES
export PYTHONPATH="$FUES_HOME${PYTHONPATH:+:$PYTHONPATH}"
cd "$FUES_HOME"

# -------- Thread hygiene -------------------------------------------
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1   # instant log flushing

echo "Running on $(hostname)  |  $(date)"
echo "Using up to ${PBS_NCPUS:-48} pool workers"
echo "───────────────────────────────────────────────────────────────"

# -------- Launch ----------------------------------------------------
python3 -m examples.housing_renting.circuit_runner_solving \
        --periods 5 \
        --output-root  "${OUT_ROOT}" \
        --bundle-prefix "${BUNDLE_PREFIX}" \
        --vfi-ngrid "${VFI_NGRID}"
