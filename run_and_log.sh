#!/bin/bash

# Simple wrapper to run circuit_run_HR_single and save output to a timestamped log file

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/interactive_runs"
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Starting run at $(date)"
echo "Log will be saved to: $LOG_FILE"
echo "========================================"
echo

# Run the script and capture all output
./scripts/int/circuit_run_HR_single.sh "$@" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo
echo "========================================"
echo "Run completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "Log saved to: $LOG_FILE"
echo "========================================"

# Also create a symlink to the latest log for easy access
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/latest.log"
echo "Latest log symlinked to: ${LOG_DIR}/latest.log"

exit $EXIT_CODE