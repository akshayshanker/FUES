#!/bin/bash
# Quick setup for estimation venv on Gadi.
# Just calls the main setup script.
#
# Usage:
#   cd /home/141/as3442/dev/fues.dev/FUES
#   bash experiments/durables/estimation/setup_venv.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$REPO_ROOT/setup/setup_venv.sh"
