#!/usr/bin/env bash
#
# Wrapper around `param_sweep.py` that makes it easy to run
#  • locally  (no MPI)
#  • or on an interactive cluster node with MPI.
#
# Examples
# -----------------------------------------------------------------------------
# Local, single process:
#   ./run_param_sweep.sh --output results.csv
#
# Interactive node with 16 ranks & an extra parameter sweep:
#   mpiexec -n 16 ./run_param_sweep.sh \
#       --use-mpi --n-procs 16 \
#       --param master.parameters.beta=0.9:0.99:7 \
#       --param master.parameters.gamma=1.0:4.0:9 \
#       --param master.parameters.theta=0.1,0.2,0.3 \
#       --ue-method CONSAV --ue-method FUES \
#       --output sweep.csv
# -----------------------------------------------------------------------------

# Absolute path to this script and to the Python driver
SCRIPT_DIR="~/dev/fues.dev/FUES/experiments/housing_renting"
PY_SCRIPT="${SCRIPT_DIR}/../dev/fues.dev/FUES/experiments/housing_renting/param_sweep.py"

# Forward every CLI argument to the Python script
python3 "$PY_SCRIPT" "$@" 