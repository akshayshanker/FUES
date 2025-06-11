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


module load openmpi/4.1.7          # sets MPI headers/libs first


export FUES_HOME=$HOME/dev/fues.dev/FUES
export PYTHONPATH="$FUES_HOME${PYTHONPATH:+:$PYTHONPATH}"   # ← fixed line
cd "$FUES_HOME"

export PYTHONPATH=$PWD:$PYTHONPATH
export MAKEMOD_QUIET=true           # Only errors
export PERIOD_QUIET=true            # Only errors
export SHOCKS_QUIET=true

cd $HOME/dev/fues.dev/FUES
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1


# ------------ MPI RUN ------------
mpiexec -np 45 \
        python3 -m examples.housing_renting.solve_runner \
          --periods 3 \
          --ue-method ALL \
          --output-root /scratch/tp66/$USER/FUES/solutions/HR_test_v3 \
          --bundle-prefix HR \
          --vfi-ngrid 1E4 \
          --HD-points 1000 \
          --grid-points 4000 \
          --recompute-baseline \
          --fresh-fast \
          --mpi \
          --plots
