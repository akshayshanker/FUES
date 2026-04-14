#!/bin/bash
# Quick setup for estimation venv on Gadi.
# Thin wrapper — use `source setup/setup.sh` from the repo root instead.
#
# Usage (from repo root, `source` so venv activation persists):
#   source setup/setup.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
echo "NOTE: prefer 'source setup/setup.sh' from $REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/setup/setup.sh"
