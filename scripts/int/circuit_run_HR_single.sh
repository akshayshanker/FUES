#!/bin/bash

# ======================================================================
#  Interactive Single Core Runner - Housing Renting Model
# ======================================================================
#
# Usage: ./circuit_run_HR_single.sh [options]
#
# This script runs the housing renting model in single-core mode for
# interactive sessions on Gadi. It's designed for development, testing,
# and quick runs without PBS job submission.
#
# Options:
#   --clear-cache     Clear Numba cache before running
#   --config NAME     Run specific configuration (default: interactive selection)
#   --metrics LIST    Comma-separated metrics to compute (default: all)
#   --no-plots        Skip plot generation
#   --trace           Enable debug tracing
#   --help           Show this help message
#
# ======================================================================

set -euo pipefail

# --- Script Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(dirname "$SCRIPT_DIR")"

# --- Default Settings ---
CLEAR_CACHE=false
CONFIG_NAME="HIGH_RES_SETTINGS_A_PB"
METRICS="euler_error"
ENABLE_PLOTS=true
ENABLE_TRACE=false
SHOW_HELP=false

# --- Parse Command Line Arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --clear-cache)
            CLEAR_CACHE=true
            shift
            ;;
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --no-plots)
            ENABLE_PLOTS=false
            shift
            ;;
        --trace)
            ENABLE_TRACE=true
            shift
            ;;
        --help)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

# --- Show Help ---
if [[ "$SHOW_HELP" == true ]]; then
    echo "Interactive Single Core Runner - Housing Renting Model"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --clear-cache     Clear Numba cache before running"
    echo "  --config NAME     Run specific configuration (HIGH_RES_SETTINGS_E, etc.)"
    echo "  --metrics LIST    Metrics to compute (euler_error, dev_c_L2, plots, all)"
    echo "  --no-plots        Skip plot generation"
    echo "  --trace           Enable debug tracing"
    echo "  --help           Show this help message"
    echo
    echo "Examples:"
    echo "  $0                                    # Interactive mode, all metrics"
    echo "  $0 --config HIGH_RES_SETTINGS_E      # Run specific config"
    echo "  $0 --metrics euler_error             # Only Euler errors (fast)"
    echo "  $0 --metrics dev_c_L2,plots --trace  # L2 + plots with debug"
    echo
    exit 0
fi

# --- Source the Configuration Library ---
if [[ -f "$SCRIPTS_ROOT/lib/job_configs.sh" ]]; then
    source "$SCRIPTS_ROOT/lib/job_configs.sh"
else
    echo "ERROR: Configuration library not found at $SCRIPTS_ROOT/lib/job_configs.sh" >&2
    exit 1
fi

# --- Environment Setup ---
echo "Setting up environment for interactive single-core run..."

# Check if we're in an interactive session on Gadi
if [[ -z "${PBS_JOBID:-}" ]] && [[ "${HOSTNAME:-}" == *gadi* ]]; then
    echo "Detected Gadi interactive session"
else
    echo "Warning: This script is designed for Gadi interactive sessions"
fi

# Module loading
echo "Loading Python module..."
module purge 2>/dev/null || true
module load python3/3.12.1 2>/dev/null || {
    echo "ERROR: Failed to load Python module. Are you on Gadi?" >&2
    exit 1
}

# Virtual environment
export VENV_ROOT=/scratch/tp66/$USER/venvs
if [[ -d "$VENV_ROOT/fues02-py3121" ]]; then
    source "$VENV_ROOT/fues02-py3121/bin/activate"
    echo "Activated virtual environment: fues02-py3121"
else
    echo "ERROR: Virtual environment not found at $VENV_ROOT/fues02-py3121" >&2
    echo "Please set up the virtual environment first" >&2
    exit 1
fi

# FUES setup
export FUES_HOME=${FUES_HOME:-$HOME/dev/fues.dev/FUES}
if [[ ! -d "$FUES_HOME" ]]; then
    echo "ERROR: FUES_HOME directory not found: $FUES_HOME" >&2
    exit 1
fi

export PYTHONPATH="$FUES_HOME${PYTHONPATH:+:$PYTHONPATH}"
cd "$FUES_HOME"

# --- Numba Configuration ---
export NUMBA_CACHE_DIR=/scratch/tp66/$USER/numba_cache

if [[ "$CLEAR_CACHE" == true ]]; then
    echo "Clearing Numba cache at $NUMBA_CACHE_DIR..."
    rm -rf "$NUMBA_CACHE_DIR"
fi

mkdir -p "$NUMBA_CACHE_DIR"
export NUMBA_NUM_THREADS=1

# Hide GPUs from Numba to prevent CUDA initialization errors in single-core mode
export CUDA_VISIBLE_DEVICES=""
export NUMBA_CUDA_LOG_LEVEL=WARNING
export NUMBA_DISABLE_CUDA=1

# --- Suppress warnings for cleaner output ---
export MAKEMOD_QUIET=true
export PERIOD_QUIET=true
export SHOCKS_QUIET=true

# --- Pre-run Validation ---
echo "Validating environment..."
echo "Python: $(which python3)"
echo "FUES_HOME: $FUES_HOME"
echo "NUMBA_CACHE_DIR: $NUMBA_CACHE_DIR"

# Test critical imports
python3 -c "
import numba
import quantecon
print('Numba version:', numba.__version__)
print('Quantecon loaded successfully')
" || {
    echo "ERROR: Failed to import required packages" >&2
    exit 1
}

# --- Configuration Selection ---
if [[ -z "$CONFIG_NAME" ]]; then
    echo
    echo "Available configurations:"
    echo "1. HIGH_RES_SETTINGS_E  - High resolution settings"
    echo "2. Enter custom config name"
    echo
    read -p "Select configuration (1-2): " choice
    
    case $choice in
        1)
            CONFIG_NAME="HIGH_RES_SETTINGS_E"
            ;;
        2)
            read -p "Enter configuration name: " CONFIG_NAME
            ;;
        *)
            echo "Invalid choice. Using HIGH_RES_SETTINGS_E"
            CONFIG_NAME="HIGH_RES_SETTINGS_E"
            ;;
    esac
fi

# Validate configuration exists
if ! declare -p "$CONFIG_NAME" &>/dev/null; then
    echo "ERROR: Configuration '$CONFIG_NAME' not found in job_configs.sh" >&2
    exit 1
fi

declare -n CONFIG_REF=$CONFIG_NAME

# --- Interactive Settings Confirmation ---
echo
echo "========================================================"
echo "Configuration: $CONFIG_NAME"
echo "Periods: ${CONFIG_REF[periods]}"
echo "VFI Grid: ${CONFIG_REF[vfi_ngrid]}"
echo "HD Points: ${CONFIG_REF[hd_points]}"
echo "Grid Points: ${CONFIG_REF[grid_points]}"
echo "Metrics: $METRICS"
echo "Plots: $([ "$ENABLE_PLOTS" == true ] && echo "enabled" || echo "disabled")"
echo "Trace: $([ "$ENABLE_TRACE" == true ] && echo "enabled" || echo "disabled")"
echo "========================================================"

# Ask for confirmation unless running non-interactively
if [[ -t 0 ]]; then  # Check if stdin is a terminal
    read -p "Continue with this configuration? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted by user"
        exit 0
    fi
fi

# --- Run Configuration ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
VERSION_TAG="${CONFIG_REF[version_suffix]}"
TRIAL_ID="interactive"

RUN_ID="${VERSION_TAG}_${TIMESTAMP}_${TRIAL_ID}"
LOG_DIR="logs/${VERSION_TAG}_${TRIAL_ID}"
OUTPUT_DIR="/scratch/tp66/$USER/FUES/solutions/housing_renting/${VERSION_TAG}_${TRIAL_ID}"

mkdir -p "$LOG_DIR"

echo
echo "Starting interactive single-core run at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"

# --- Build Command Line Arguments ---
ARGS=(
    --periods "${CONFIG_REF[periods]}"
    --ue-method "FUES"
    --output-root "$OUTPUT_DIR"
    --config-id "$VERSION_TAG"
    --RUN-ID "$RUN_ID"
    --vfi-ngrid "${CONFIG_REF[vfi_ngrid]}"
    --HD-points "${CONFIG_REF[hd_points]}"
    --grid-points "${CONFIG_REF[grid_points]}"
    --delta-pb "${CONFIG_REF[delta_pb]}"
    --baseline-method "CONSAV"
    --fresh-fast
    --csv-export 
    --metrics "$METRICS"
    --trace
    --precompile 
    --verbose
    --csv-export
    --plots
)

if [[ "$ENABLE_PLOTS" == true ]]; then
    ARGS+=(--plots)
fi

if [[ "$ENABLE_TRACE" == true ]]; then
    ARGS+=(--trace)
fi

# --- Execute the Run ---
echo "Executing: python3 -m examples.housing_renting.solve_runner ${ARGS[*]}"
echo

# Run with live output and logging
python3 -m examples.housing_renting.solve_runner "${ARGS[@]}" 2>&1 | tee "${LOG_DIR}/run.log"

EXIT_CODE=${PIPESTATUS[0]}

# --- Post-run Summary ---
echo
echo "========================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Run completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Logs saved to: $LOG_DIR"
    
    # Show quick summary if available
    if [[ -f "$OUTPUT_DIR/comparison_table.tex" ]]; then
        echo
        echo "LaTeX table generated: $OUTPUT_DIR/comparison_table.tex"
    fi
    
    if [[ -d "$OUTPUT_DIR/images" ]]; then
        PLOT_COUNT=$(find "$OUTPUT_DIR/images" -name "*.png" 2>/dev/null | wc -l)
        if [[ $PLOT_COUNT -gt 0 ]]; then
            echo "Generated $PLOT_COUNT plot files in: $OUTPUT_DIR/images"
        fi
    fi
    
else
    echo "❌ Run failed with exit code: $EXIT_CODE"
    echo "Check error log: ${LOG_DIR}/run.log"
    
    # Check for common issues
    if grep -q "cannot access local variable" "${LOG_DIR}/run.log"; then
        echo "Hint: Variable scoping error detected. This may be a code bug."
    fi
    if grep -q "baseline.*not found" "${LOG_DIR}/run.log"; then
        echo "Hint: Baseline bundle not found. Run GPU job first to compute baseline."
    fi
    if grep -q "LLVM ERROR" "${LOG_DIR}/run.log"; then
        echo "Hint: LLVM error detected. Try running with --clear-cache."
    fi
    if grep -q "Memory usage.*GB" "${LOG_DIR}/run.log"; then
        echo "Hint: Check memory usage patterns in the log for optimization opportunities."
    fi
fi

echo "========================================================"
exit $EXIT_CODE 