#!/bin/bash

# Helper script to submit MULTIPLE GPU jobs - one per configuration
# Usage: ./submit_multi_housing_gpu.sh CONFIG1 [CONFIG2 ...]
# Example: ./submit_multi_housing_gpu.sh HIGH_RES_SETTINGS_K
# Example: ./submit_multi_housing_gpu.sh HIGH_RES_SETTINGS_A HIGH_RES_SETTINGS_B

if [[ $# -eq 0 ]]; then
    echo "Error: No configurations specified."
    echo "Usage: $0 CONFIG1 [CONFIG2 ...]"
    echo ""
    echo "Available configurations:"
    echo "  STD_RES_SETTINGS"
    echo "  STD_RES_SETTINGS_PB"
    echo "  HIGH_RES_SETTINGS_A"
    echo "  HIGH_RES_SETTINGS_A_PB"
    echo "  HIGH_RES_SETTINGS_B"
    echo "  HIGH_RES_SETTINGS_C"
    echo "  HIGH_RES_SETTINGS_D"
    echo "  HIGH_RES_SETTINGS_K"
    echo ""
    echo "Example: $0 HIGH_RES_SETTINGS_K"
    echo "Example: $0 HIGH_RES_SETTINGS_A HIGH_RES_SETTINGS_B"
    echo ""
    echo "This will submit SEPARATE jobs for each configuration."
    exit 1
fi

echo "Submitting ${#@} separate jobs..."
echo ""

# Submit a separate job for each configuration
for config in "$@"; do
    echo "Submitting job for: $config"
    
    # Create a temporary PBS script for this specific configuration
    TMPFILE=$(mktemp /tmp/gpu_job_${config}_XXXXXX.pbs)
    
    # Create the CONFIG_TO_RUN with just this one configuration
    CONFIG_ARRAY="CONFIG_TO_RUN=(\"$config\")"
    
    # Read the original PBS script and replace the CONFIG_TO_RUN section
    awk -v configs="$CONFIG_ARRAY" -v config_name="$config" '
        /^#PBS -N / {
            print "#PBS -N fues-gpu-" config_name
            next
        }
        /^# --- 2\. Define the Sequence of Configurations to Run ---$/ {
            print
            getline  # Skip the comment or check line
            if ($0 ~ /^# Check if FUES_CONFIGS/) {
                # Skip the entire if-else block
                while (getline && $0 !~ /^fi$/) {}
                getline  # Skip the fi line
            } else if ($0 ~ /^CONFIG_TO_RUN=/) {
                # Skip the old CONFIG_TO_RUN definition
                while (getline && $0 !~ /^\)$/) {}
            }
            print configs
            print ""
            next
        }
        { print }
    ' run_housing_gpu.pbs > "$TMPFILE"
    
    # Submit the job and capture the job ID
    JOB_ID=$(qsub "$TMPFILE")
    echo "  Job submitted: $JOB_ID"
    echo "  Temporary PBS script: $TMPFILE"
    
    # Optional: remove temporary file after submission
    # rm -f "$TMPFILE"
    
    echo ""
done

echo "All jobs submitted successfully!"
echo ""
echo "Check job status with: qstat -u $USER"