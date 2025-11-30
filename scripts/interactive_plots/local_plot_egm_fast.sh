#!/bin/bash

# ======================================================================
#  Optimized Local EGM Interactive Plot Generator
# ======================================================================
#
# This script uses the optimized plotting script with WebGL and 
# intelligent downsampling for better performance with large datasets.
#
# Usage: ./local_plot_egm_fast.sh [options]
#
# Options:
#   --download        Download fresh CSV data from Gadi
#   --no-download     Skip download, use existing local data
#   --gadi-path PATH  Custom path on Gadi (default: auto-detect latest)
#   --methods LIST    Comma-separated list of methods (default: FUES)
#   --output DIR      Output directory for HTML plots (default: ~/Desktop/fues_plots_fast)
#   --max-points N    Max points per plot (default: 0=unlimited, use 5000 for faster load)
#   --downsample TYPE Downsampling: uniform|adaptive|lttb (default: adaptive)
#   --no-webgl        Disable WebGL acceleration
#   --help           Show this help message
#
# Examples:
#   ./local_plot_egm_fast.sh --no-download                 # Fast plot with existing data
#   ./local_plot_egm_fast.sh --max-points 10000            # More detail
#   ./local_plot_egm_fast.sh --downsample lttb             # Best quality sampling
#
# ======================================================================

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUES_HOME="$(dirname "$SCRIPT_DIR")"

# Use mounted Gadi path directly
MOUNTED_SCRATCH_DIR="/Users/akshayshanker/gadi/scratch/tp66/as3442/FUES/solutions/housing_renting/test_0.1_gpu_test/bundles/a1f054de/FUES/images/egm_csv"
LOCAL_CSV_DIR="${HOME}/Desktop/fues_csv_data"

# Default settings
DOWNLOAD=true
GADI_USER="${USER}"
GADI_HOST="gadi.nci.org.au"
GADI_BASE_PATH="/scratch/tp66/${GADI_USER}/FUES/solutions/housing_renting"
OUTPUT_DIR="${HOME}/Desktop/fues_plots_fast"
METHODS="FUES"
CUSTOM_GADI_PATH=""
MAX_POINTS=0  # 0 means unlimited - keep all points
DOWNSAMPLE="adaptive"
USE_WEBGL="--use-webgl"
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
        --max-points)
            MAX_POINTS="$2"
            shift 2
            ;;
        --downsample)
            DOWNSAMPLE="$2"
            shift 2
            ;;
        --no-webgl)
            USE_WEBGL="--no-webgl"
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
    head -n 28 "$0" | tail -n 26 | sed 's/^# //'
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
echo "           Optimized Local EGM Interactive Plot Generator            "
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

if ! python3 -c "import numpy" 2>/dev/null; then
    echo_error "NumPy not installed. Installing..."
    pip install numpy || {
        echo_error "Failed to install numpy. Please install manually: pip install numpy"
        exit 1
    }
fi

echo_success "Python dependencies verified"

# --- Handle CSV Data ---
if [[ "$DOWNLOAD" == true ]]; then
    echo
    echo_info "Downloading CSV data from Gadi..."
    
    # Determine Gadi path
    if [[ -n "$CUSTOM_GADI_PATH" ]]; then
        GADI_CSV_PATH="$CUSTOM_GADI_PATH"
    else
        echo_info "Looking for most recent data on Gadi..."
        
        LATEST_DIR=$(ssh "${GADI_USER}@${GADI_HOST}" "
            cd ${GADI_BASE_PATH} 2>/dev/null && \
            ls -dt *_interactive/csv_egm_data 2>/dev/null | head -1
        " 2>/dev/null || echo "")
        
        if [[ -z "$LATEST_DIR" ]]; then
            echo_warn "Could not find CSV data on Gadi automatically"
            echo "Please specify the path with --gadi-path"
            exit 1
        fi
        
        GADI_CSV_PATH="${GADI_BASE_PATH}/${LATEST_DIR}"
    fi
    
    echo_info "Gadi path: $GADI_CSV_PATH"
    echo_info "Local path: $LOCAL_CSV_DIR"
    
    mkdir -p "$LOCAL_CSV_DIR"
    
    echo_info "Downloading files..."
    rsync -avz --progress \
        "${GADI_USER}@${GADI_HOST}:${GADI_CSV_PATH}/" \
        "${LOCAL_CSV_DIR}/" || {
        echo_error "Failed to download CSV files from Gadi"
        exit 1
    }
    
    echo_success "CSV files downloaded successfully"
else
    echo_info "Using mounted scratch directory"
    
    if [[ ! -d "$MOUNTED_SCRATCH_DIR" ]]; then
        echo_error "Mounted scratch directory not found: $MOUNTED_SCRATCH_DIR"
        exit 1
    fi
    
    echo_info "Syncing from mounted scratch to local directory..."
    mkdir -p "$LOCAL_CSV_DIR/FUES"
    
    cp -r "$MOUNTED_SCRATCH_DIR"/* "$LOCAL_CSV_DIR/FUES/" 2>/dev/null || {
        echo_warn "Some files may not have copied. Continuing..."
    }
fi

# --- Count Data Points ---
echo
echo_info "Analyzing data size..."

TOTAL_POINTS=0
for csv_file in "$LOCAL_CSV_DIR"/FUES/*.csv; do
    if [[ -f "$csv_file" ]]; then
        POINTS=$(wc -l < "$csv_file")
        TOTAL_POINTS=$((TOTAL_POINTS + POINTS))
    fi
done

echo_info "Total data points across all files: $TOTAL_POINTS"

if [[ $TOTAL_POINTS -gt 100000 ]]; then
    echo_warn "Large dataset detected. Optimization will be applied."
    if [[ $MAX_POINTS -eq 0 ]]; then
        echo_warn "Consider using --max-points to limit points for better performance"
    fi
fi

# --- Generate Optimized Interactive Plots ---
echo
echo_info "Generating optimized interactive plots..."
echo_info "Settings:"
echo "  - Methods: $METHODS"
echo "  - Max points: ${MAX_POINTS} (0=unlimited)"
echo "  - Downsampling: $DOWNSAMPLE"
echo "  - WebGL: $([ "$USE_WEBGL" = "--use-webgl" ] && echo "enabled" || echo "disabled")"
echo "  - Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Use the optimized plotting script
PLOT_SCRIPT="${FUES_HOME}/examples/housing_renting/interactive_plots_from_csv_optimized.py"

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo_error "Optimized plotting script not found: $PLOT_SCRIPT"
    exit 1
fi

# Build python command
PYTHON_CMD="python3 $PLOT_SCRIPT"
PYTHON_CMD="$PYTHON_CMD --csv-dir $LOCAL_CSV_DIR"
PYTHON_CMD="$PYTHON_CMD --methods $METHODS"
PYTHON_CMD="$PYTHON_CMD --output-dir $OUTPUT_DIR"
PYTHON_CMD="$PYTHON_CMD --max-points $MAX_POINTS"
PYTHON_CMD="$PYTHON_CMD --downsample $DOWNSAMPLE"

if [[ "$USE_WEBGL" == "--no-webgl" ]]; then
    PYTHON_CMD="$PYTHON_CMD --no-webgl"
fi

# Run the plotting script
eval $PYTHON_CMD || {
    echo_error "Failed to generate plots"
    exit 1
}

echo
echo_success "Optimized interactive plots generated successfully!"

# --- Performance Report ---
echo
echo "======================================================================"
echo "                      PERFORMANCE OPTIMIZATIONS                       "
echo "======================================================================"
echo
echo "✅ WebGL Rendering: $([ "$USE_WEBGL" = "--use-webgl" ] && echo "Enabled for hardware acceleration" || echo "Disabled")"
echo "✅ Downsampling: $DOWNSAMPLE method"
if [[ $MAX_POINTS -gt 0 ]]; then
    echo "✅ Point Limit: $MAX_POINTS points per plot"
else
    echo "⚠️  Point Limit: Unlimited (may be slow for large datasets)"
fi
echo
echo "Performance Tips:"
echo "  • For smoother interaction, use --max-points 2000"
echo "  • For best quality, use --downsample lttb"
echo "  • For fastest loading, use --downsample uniform"
echo "  • Enable WebGL for best performance (default)"
echo

# --- Summary ---
echo "======================================================================"
echo "                              COMPLETE                                "
echo "======================================================================"
echo
echo "📊 Optimized dashboards saved to: $OUTPUT_DIR"
echo

# Check if index.html was created
if [[ -f "$OUTPUT_DIR/index.html" ]]; then
    echo "📋 Overview page: $OUTPUT_DIR/index.html"
    echo
fi

echo "Files created:"
for html_file in "$OUTPUT_DIR"/*.html "$OUTPUT_DIR"/*/*.html; do
    if [[ -f "$html_file" ]]; then
        echo "  - $(basename "$(dirname "$html_file")")/$(basename "$html_file")"
    fi
done | head -20

echo
echo "To view the plots:"
echo "  1. Open index.html for an overview of all plots"
echo "  2. Click individual plots to view them"
echo "  3. Use Pan mode (default) for smooth navigation"
echo "  4. Box/Lasso select to zoom into regions"
echo "  5. Double-click to reset view"
echo
echo "To rerun with different settings:"
echo "  $0 --no-download --max-points 2000   # Faster"
echo "  $0 --no-download --max-points 0      # Full detail"
echo
echo "======================================================================"

# Open in browser if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo
    read -p "Open overview in browser? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [[ -f "$OUTPUT_DIR/index.html" ]]; then
            open "$OUTPUT_DIR/index.html"
        else
            # Open first HTML file found
            for html_file in "$OUTPUT_DIR"/*.html "$OUTPUT_DIR"/*/*.html; do
                if [[ -f "$html_file" ]]; then
                    open "$html_file"
                    break
                fi
            done
        fi
    fi
fi