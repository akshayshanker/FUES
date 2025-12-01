#!/bin/bash
# Generate interactive plots from existing CSV data or run solver with CSV export

# Default paths (adjust as needed)
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/as3442}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_DIR/outputs}"
CSV_DIR="${CSV_DIR:-$OUTPUT_DIR/csv_egm_data}"
PLOT_DIR="${PLOT_DIR:-$OUTPUT_DIR/interactive_plots}"

# Check if CSV data already exists
if [ -d "$CSV_DIR" ] && [ "$(ls -A $CSV_DIR)" ]; then
    echo "Found existing CSV data in $CSV_DIR"
    echo "Generating interactive plots..."
else
    echo "No CSV data found. Run solver with --csv-export first:"
    echo "  python solve_runner.py --csv-export --periods 3 --ue-method ALL --output-dir $OUTPUT_DIR"
    exit 1
fi

# Generate interactive plots
cd /home/141/as3442/dev/fues.dev/FUES/examples/housing_renting
python interactive_plots_from_csv.py \
    --csv-dir "$CSV_DIR" \
    --output-dir "$PLOT_DIR"

echo ""
echo "Interactive plots generated in: $PLOT_DIR"
echo ""
echo "To view plots:"
echo "  1. From mounted scratch on local machine:"
echo "     open $PLOT_DIR/FUES/egm_y0_h0.html"
echo ""
echo "  2. Or copy to local machine:"
echo "     rsync -av gadi:$PLOT_DIR/ ./local_plots/"
echo ""

# List generated HTML files
echo "Generated plot files:"
find "$PLOT_DIR" -name "*.html" -type f | head -20