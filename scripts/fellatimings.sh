#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=25GB
#PBS -q expresssr
#PBS -P tp66
#PBS -l walltime=10:00:00
#PBS -l storage=scratch/pv33+gdata/pv33
#PBS -l wd

module load python3/3.12.1

python3 fella_plot.py