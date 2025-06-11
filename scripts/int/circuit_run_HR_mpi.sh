

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
          --output-root /scratch/tp66/$USER/FUES/solutions/HR_test_v4 \
          --bundle-prefix HR_test_v4 \
          --vfi-ngrid 100 \
          --HD-points 650 \
          --grid-points 500 \
          --recompute-baseline \
          --fresh-fast \
          --mpi \
          --plots