#!/bin/bash

# Helper script to submit DGX A100 GPU jobs with specific configurations
# Usage: ./submit_dgxa100_config.sh CONFIG1 [CONFIG2 ...]
# Example: ./submit_dgxa100_config.sh HIGH_RES_SETTINGS_K
# Example: ./submit_dgxa100_config.sh HIGH_RES_SETTINGS_C HIGH_RES_SETTINGS_D HIGH_RES_SETTINGS_E

if [[ $# -eq 0 ]]; then
    echo "Error: No configurations specified."
    echo "Usage: $0 CONFIG1 [CONFIG2 ...]"
    echo ""
    echo "Available configurations:"
    echo "  HIGH_RES_SETTINGS_A through HIGH_RES_SETTINGS_N"
    echo "  STD_RES_SETTINGS"
    echo ""
    echo "Example: $0 HIGH_RES_SETTINGS_K"
    echo "Example: $0 HIGH_RES_SETTINGS_C HIGH_RES_SETTINGS_D HIGH_RES_SETTINGS_E"
    exit 1
fi

echo "Submitting DGX A100 jobs for ${#} configuration(s): $*"
echo ""

# Loop through all provided configurations and submit each as a separate job
for CONFIG in "$@"; do
    echo "Submitting job for configuration: $CONFIG"
    
    # Submit with the configuration as an environment variable
    JOB_ID=$(qsub -v FUES_CONFIG="$CONFIG" run_housing_dgxa100_single.pbs)
    
    echo "  Job submitted: $JOB_ID"
    echo ""
done

echo "All jobs submitted successfully."
echo "Total jobs: ${#}"