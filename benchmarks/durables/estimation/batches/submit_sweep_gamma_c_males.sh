#!/bin/bash
# Submit gamma_c sweep jobs for males (EGM + NEGM)
# Run from repo root: bash benchmarks/durables/estimation/submit_sweep_gamma_c_males.sh

SCRIPT_DIR="benchmarks/durables/estimation"

echo "Submitting gamma_c sweep jobs (males)..."
echo ""

qsub "$SCRIPT_DIR/run_selfgen_sweep_gamma_c_egm_males.pbs"
qsub "$SCRIPT_DIR/run_selfgen_sweep_gamma_c_negm_males.pbs"

echo ""
echo "Submitted 2 jobs. Check with: qstat -u \$USER"
