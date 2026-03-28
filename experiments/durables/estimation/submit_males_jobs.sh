#!/bin/bash
# Submit male baseline estimation jobs (EGM + NEGM)
# Run from repo root: bash experiments/durables/estimation/submit_males_jobs.sh

SCRIPT_DIR="experiments/durables/estimation"

echo "Submitting male baseline estimation jobs..."
echo ""

qsub "$SCRIPT_DIR/run_large_egm_males.pbs"
qsub "$SCRIPT_DIR/run_large_negm_males.pbs"

echo ""
echo "Submitted 2 jobs. Check with: qstat -u \$USER"
