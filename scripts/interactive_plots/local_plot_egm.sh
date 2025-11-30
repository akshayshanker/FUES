#!/bin/bash

# ======================================================================
#  Local EGM Interactive Plot Generator
# ======================================================================
#
# This script downloads CSV data from Gadi and generates interactive
# Plotly dashboards for EGM plots locally.
#
# Usage: ./local_plot_egm.sh [options]
#
# Options:
#   --download        Download fresh CSV data from Gadi
#   --no-download     Skip download, use existing local data
#   --gadi-path PATH  Custom path on Gadi (default: auto-detect latest)
#   --methods LIST    Comma-separated list of methods (default: FUES,CONSAV)
#   --output DIR      Output directory for HTML plots (default: ~/Desktop/fues_plots)
#   --help           Show this help message
#
# Examples:
#   ./local_plot_egm.sh --download                    # Download and plot
#   ./local_plot_egm.sh --no-download                 # Use existing CSVs
#   ./local_plot_egm.sh --methods FUES                # Only plot FUES
#
# ======================================================================

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUES_HOME="$(dirname "$SCRIPT_DIR")"

# Use mounted Gadi path directly
# This should point to where the CSV data actually is
MOUNTED_SCRATCH_DIR="/Users/akshayshanker/gadi/scratch/tp66/as3442/FUES/solutions/housing_renting/test_0.1_gpu_test/bundles/a1f054de/FUES/images/egm_csv"
LOCAL_CSV_DIR="${HOME}/Desktop/fues_csv_data"  # Local directory for working with the data

# Default settings
DOWNLOAD=true
GADI_USER="${USER}"
GADI_HOST="gadi.nci.org.au"
GADI_BASE_PATH="/scratch/tp66/${GADI_USER}/FUES/solutions/housing_renting"
OUTPUT_DIR="${HOME}/Desktop/fues_plots"
METHODS="FUES"
CUSTOM_GADI_PATH=""
SHOW_HELP=false

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --download)
            DOWNLOAD=true
            shift
            ;;
        --no-download)
            DOWNLOAD=false
            shift
            ;;
        --gadi-path)
            CUSTOM_GADI_PATH="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
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
    head -n 26 "$0" | tail -n 24 | sed 's/^# //'
    exit 0
fi

# --- Color Output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
echo_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# --- Header ---
echo
echo "======================================================================"
echo "                 Local EGM Interactive Plot Generator                 "
echo "======================================================================"
echo

# --- Check Python Dependencies ---
echo_info "Checking Python dependencies..."

if ! python3 -c "import plotly" 2>/dev/null; then
    echo_error "Plotly not installed. Installing..."
    pip install plotly || {
        echo_error "Failed to install plotly. Please install manually: pip install plotly"
        exit 1
    }
fi

if ! python3 -c "import pandas" 2>/dev/null; then
    echo_error "Pandas not installed. Installing..."
    pip install pandas || {
        echo_error "Failed to install pandas. Please install manually: pip install pandas"
        exit 1
    }
fi

echo_success "Python dependencies verified"

# --- Download CSV Data from Gadi ---
if [[ "$DOWNLOAD" == true ]]; then
    echo
    echo_info "Downloading CSV data from Gadi..."
    
    # Determine Gadi path
    if [[ -n "$CUSTOM_GADI_PATH" ]]; then
        GADI_CSV_PATH="$CUSTOM_GADI_PATH"
    else
        # Try to find the most recent interactive run
        echo_info "Looking for most recent data on Gadi..."
        
        # SSH to Gadi and find the latest directory
        LATEST_DIR=$(ssh "${GADI_USER}@${GADI_HOST}" "
            cd ${GADI_BASE_PATH} 2>/dev/null && \
            ls -dt *_interactive/csv_egm_data 2>/dev/null | head -1
        " 2>/dev/null || echo "")
        
        if [[ -z "$LATEST_DIR" ]]; then
            echo_warn "Could not find CSV data on Gadi automatically"
            echo "Please specify the path with --gadi-path"
            echo "Example: --gadi-path /scratch/tp66/${GADI_USER}/FUES/solutions/housing_renting/test_0.1_interactive/csv_egm_data"
            exit 1
        fi
        
        GADI_CSV_PATH="${GADI_BASE_PATH}/${LATEST_DIR}"
    fi
    
    echo_info "Gadi path: $GADI_CSV_PATH"
    echo_info "Local path: $LOCAL_CSV_DIR"
    
    # Create local directory
    mkdir -p "$LOCAL_CSV_DIR"
    
    # Download with rsync
    echo_info "Downloading files..."
    rsync -avz --progress \
        "${GADI_USER}@${GADI_HOST}:${GADI_CSV_PATH}/" \
        "${LOCAL_CSV_DIR}/" || {
        echo_error "Failed to download CSV files from Gadi"
        echo "Please check your connection and path"
        exit 1
    }
    
    echo_success "CSV files downloaded successfully"
    
    # Show what was downloaded
    echo
    echo_info "Downloaded methods:"
    for method_dir in "$LOCAL_CSV_DIR"/*; do
        if [[ -d "$method_dir" ]]; then
            method_name=$(basename "$method_dir")
            csv_count=$(find "$method_dir" -name "*.csv" 2>/dev/null | wc -l)
            echo "  - $method_name: $csv_count CSV files"
        fi
    done
else
    echo_info "Using mounted scratch directory instead of downloading"
    
    # Check if mounted scratch directory exists
    if [[ ! -d "$MOUNTED_SCRATCH_DIR" ]]; then
        echo_error "Mounted scratch directory not found: $MOUNTED_SCRATCH_DIR"
        echo "Please check your mount or specify correct path"
        exit 1
    fi
    
    # Sync from mounted scratch to local directory
    echo_info "Syncing from mounted scratch to local directory..."
    mkdir -p "$LOCAL_CSV_DIR"
    
    # Create FUES subdirectory structure expected by the plotting script
    mkdir -p "$LOCAL_CSV_DIR/FUES"
    
    # Copy files from mounted scratch
    echo_info "Copying files from: $MOUNTED_SCRATCH_DIR"
    echo_info "Copying files to: $LOCAL_CSV_DIR/FUES/"
    
    cp -r "$MOUNTED_SCRATCH_DIR"/* "$LOCAL_CSV_DIR/FUES/" 2>/dev/null || {
        echo_warn "Some files may not have copied. Continuing..."
    }
    
    # Show what was copied
    echo_info "Files copied:"
    ls -la "$LOCAL_CSV_DIR/FUES/" | head -10
fi

# --- Generate Interactive Plots ---
echo
echo_info "Generating interactive plots..."
echo_info "Methods: $METHODS"
echo_info "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the plotting script
PLOT_SCRIPT="${FUES_HOME}/examples/housing_renting/interactive_plots_from_csv.py"

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo_error "Plotting script not found: $PLOT_SCRIPT"
    echo "Please ensure you're running from the FUES directory"
    exit 1
fi

python3 "$PLOT_SCRIPT" \
    --csv-dir "$LOCAL_CSV_DIR" \
    --methods "$METHODS" \
    --output-dir "$OUTPUT_DIR" || {
    echo_error "Failed to generate plots"
    exit 1
}

echo
echo_success "Interactive plots generated successfully!"

# --- Summary ---
echo
echo "======================================================================"
echo "                              COMPLETE                                "
echo "======================================================================"
echo
echo "📊 Interactive dashboards saved to: $OUTPUT_DIR"
echo
echo "Files created:"
for html_file in "$OUTPUT_DIR"/*.html; do
    if [[ -f "$html_file" ]]; then
        echo "  - $(basename "$html_file")"
    fi
done
echo
echo "To view the plots:"
echo "  1. Open the HTML files in your browser"
echo "  2. Use mouse to zoom (click and drag)"
echo "  3. Double-click to reset view"
echo "  4. Hover for exact values"
echo "  5. Click legend items to toggle visibility"
echo
echo "To rerun without downloading:"
echo "  $0 --no-download"
echo
echo "======================================================================"

# Open in browser if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo
    read -p "Open dashboards in browser? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for html_file in "$OUTPUT_DIR"/*.html; do
            if [[ -f "$html_file" ]]; then
                open "$html_file"
                break  # Just open the first one
            fi
        done
    fi
fi