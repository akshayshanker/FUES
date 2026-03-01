#!/bin/bash
# Launch an interactive PBS session on Gadi expresssr queue (single core).
# Once inside, run: source setup/load_env.sh
#
# Usage: bash setup/interactive_session.sh

echo "Launching interactive session on expresssr (1 CPU, 8GB, 1 hour)..."
echo "Once inside, run:  source setup/load_env.sh"
echo ""

qsub -I -q expresssr -P tp66 -l ncpus=1,mem=5GB,walltime=01:00:00,storage=scratch/tp66,wd
