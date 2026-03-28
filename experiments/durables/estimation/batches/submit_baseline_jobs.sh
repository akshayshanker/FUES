#!/bin/bash
# Submit all baseline precomputed estimation jobs (females + males, EGM + NEGM)
# Run from repo root: bash experiments/durables/estimation/submit_baseline_jobs.sh

SCRIPT_DIR="experiments/durables/estimation"

echo "Submitting baseline estimation jobs..."
echo ""

qsub "$SCRIPT_DIR/run_large_egm.pbs"
qsub "$SCRIPT_DIR/run_large_negm.pbs"
qsub "$SCRIPT_DIR/run_xlarge_egm.pbs"
qsub "$SCRIPT_DIR/run_xlarge_negm.pbs"

echo ""
echo "Submitted 6 jobs. Check with: qstat -u \$USER"
