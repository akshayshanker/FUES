#!/usr/bin/env bash
set -euo pipefail

# Move housing_renting logs from the repo to scratch.
# Source: experiments/housing_renting/logs
# Destination: /scratch/tp66/$USER/FUES/solutions/housing_renting/logs

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../experiments/housing_renting/logs" && pwd)"
DEST_DIR="/scratch/tp66/${USER}/FUES/solutions/housing_renting/logs"

echo "Source:      $SRC_DIR"
echo "Destination: $DEST_DIR"

mkdir -p "$DEST_DIR"

shopt -s nullglob
files=("$SRC_DIR"/*)
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No files to move."
  exit 0
fi

echo "Moving ${#files[@]} files..."
mv "${files[@]}" "$DEST_DIR/"
echo "Done."
