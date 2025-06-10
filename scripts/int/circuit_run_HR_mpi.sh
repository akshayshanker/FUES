

module load openmpi/4.1.7
export PYTHONPATH=$PWD:$PYTHONPATH
export MAKEMOD_QUIET=true           # Only errors
export PERIOD_QUIET=true            # Only errors

cd $HOME/dev/fues.dev/FUES



# ------------ MPI RUN ------------
mpiexec -np 45 \
        python3 -m examples.housing_renting.circuit_runner_solving \
          --periods 3 \
          --ue-method ALL \
          --output-root /scratch/tp66/$USER/FUES/solutions/HR_test_v3 \
          --bundle-prefix HR \
          --vfi-ngrid 30000 \
          --recompute-baseline \
          --fresh-fast \
          --mpi \
          --plots