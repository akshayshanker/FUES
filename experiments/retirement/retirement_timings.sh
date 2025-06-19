#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=25GB
#PBS -q expresssr
#PBS -P tp66
#PBS -l walltime=10:00:00
#PBS -l storage=scratch/pv33+gdata/pv33
#PBS -l wd

module load python3/3.12.1

# INSTEAD, ADD THIS LINE to enable caching on the node's fast local disk:
export NUMBA_CACHE_DIR=$PBS_JOBFS

export NUMBA_NUM_THREADS=1

export VENV_ROOT=/scratch/tp66/$USER/venvs
source "$VENV_ROOT/fues02-py3121/bin/activate"



cd /home/141/as3442/dev/fues.dev/FUES/examples/retirement

python3 retirement_plot.py