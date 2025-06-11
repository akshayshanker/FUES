#!/bin/bash
#PBS -N fues-run
#PBS -P tp66
#PBS -q normal
#PBS -l ncpus=48,mem=192GB,walltime=05:00:00
#PBS -l storage=scratch/tp66
#PBS -l wd
#PBS -j oe          

set -eu  # fail hard on errors
module purge
module load python3/3.12.1

export VENV_ROOT=/scratch/tp66/$USER/venvs
source "$VENV_ROOT/fues02-py3121/bin/activate"

export FUES_HOME=$HOME/dev/fues.dev/FUES
export PYTHONPATH="$FUES_HOME${PYTHONPATH:+:$PYTHONPATH}"   # ← fixed line
cd "$FUES_HOME"

export OMP_NUM_THREADS=${PBS_NCPUS:-48}
export NUMBA_NUM_THREADS=$OMP_NUM_THREADS
export NUMBA_THREADING_LAYER=omp
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# stop every math library from also spawning threads
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# (optional) make Python prints appear immediately
export PYTHONUNBUFFERED=1

echo "Numba threads: $NUMBA_NUM_THREADS  |  MKL threads: $MKL_NUM_THREADS"
echo "Running on $(hostname)   $(date)"
echo "────────────────────────────────────────────────────────"

python3 -m examples.housing_renting.circuit_runner_solving \
       --periods 5 \
       --output-root /scratch/tp66/$USER/FUES/solutions/HR_test_v3 \
       --bundle-prefix HR_test_v3 \
       --vfi-ngrid-index 1
