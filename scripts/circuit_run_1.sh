#!/bin/bash
#PBS -N fues-run
#PBS -P tp66
#PBS -q normal
#PBS -l ncpus=48
#PBS -l mem=192GB           
#PBS -l walltime=05:00:00
#PBS -l storage=scratch/tp66
#PBS -l wd
#PBS -j oe

module purge
module load python3/3.12.1

export VENV_ROOT=/scratch/tp66/$USER/venvs
source "$VENV_ROOT/fues02-py3121/bin/activate"

export FUES_HOME=$HOME/dev/fues.dev/FUES
export PYTHONPATH="$FUES_HOME:$PYTHONPATH"
cd "$FUES_HOME"

python -m examples.housing_renting.circuit_runner_solving --periods 5 --output-root /scratch/tp66/as3442/FUES/solutions/HR_test_v2 --bundle-prefix HR_test_v2 --vfi-ngrid-index 1
