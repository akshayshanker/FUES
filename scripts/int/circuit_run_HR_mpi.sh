

module load openmpi/4.1.7
export PYTHONPATH=$PWD:$PYTHONPATH
export MAKEMOD_QUIET=true           # Only errors
export PERIOD_QUIET=true            # Only errors
export SHOCKS_QUIET=true

cd $HOME/dev/fues.dev/FUES



# ------------ MPI RUN ------------
mpiexec -np 45 \
        python3 -m examples.housing_renting.solve_runner \
          --periods 3 \
          --ue-method ALL \
          --output-root /scratch/tp66/$USER/FUES/solutions/HR_test_v3 \
          --bundle-prefix HR \
          --vfi-ngrid 1000 \
          --HD-points 10000 \
          --grid-points 4000 \
          --recompute-baseline \
          --fresh-fast \
          --mpi \
          --plots