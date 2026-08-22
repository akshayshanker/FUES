#!/bin/bash
# Submit gamma_c sweep jobs for females (EGM + NEGM)
# Run from repo root: bash benchmarks/durables/estimation/submit_sweep_gamma_c_females.sh

SCRIPT_DIR="benchmarks/durables/estimation"

echo "Submitting gamma_c sweep jobs (females)..."
echo ""

qsub "$SCRIPT_DIR/run_selfgen_sweep_gamma_c_egm.pbs"
qsub "$SCRIPT_DIR/run_selfgen_sweep_gamma_c_negm.pbs"

echo ""
echo "Submitted 2 jobs. Check with: qstat -u \$USER"
