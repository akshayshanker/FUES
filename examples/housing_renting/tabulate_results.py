#!/usr/bin/env python3
"""
Standalone script to generate paper tables from sweep results.

This reads sweep_results.csv from the results folder (not scratch) and
regenerates the LaTeX and Markdown tables. Use this to re-format tables
without re-running the sweep.

Usage:
    # From FUES repo root:
    python examples/housing_renting/tabulate_results.py --trial test_0.1-paper-sweep-2-small
    
    # With custom experiment set (for reading periods):
    python examples/housing_renting/tabulate_results.py \
        --trial test_0.1-paper-sweep-2-small \
        --experiment-set sweep_noPB_small
    
    # Override periods manually:
    python examples/housing_renting/tabulate_results.py \
        --trial test_0.1-paper-sweep-2-small \
        --periods 5
"""

import argparse
import sys
from pathlib import Path

# IMPORTANT: do NOT import `examples.housing_renting` as a package here.
# Its `__init__.py` pulls in the full solver stack (dcsmm, numba.cuda, mpi4py),
# which is unnecessary for table generation and can fail on login nodes.
SCRIPT_DIR = Path(__file__).resolve().parent  # .../examples/housing_renting
# Repo root is two levels up: .../FUES
REPO_ROOT = SCRIPT_DIR.parent.parent

# Make `helpers.*` importable without executing `examples/housing_renting/__init__.py`.
sys.path.insert(0, str(SCRIPT_DIR))

from helpers.generate_paper_tables import (
    load_sweep_results,
    generate_accuracy_latex,
    generate_timing_latex,
    generate_accuracy_markdown,
    generate_timing_markdown,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper tables from sweep results in the results folder"
    )
    parser.add_argument(
        "--trial", 
        type=str, 
        required=True,
        help="Trial ID (subfolder name in results/housing_renting/)"
    )
    parser.add_argument(
        "--periods", 
        type=int, 
        default=None,
        help="Number of periods T (reads from experiment YAML if not specified)"
    )
    parser.add_argument(
        "--experiment-set", 
        type=str, 
        default=None,
        help="Experiment set name to read periods from (e.g., sweep_noPB_small)"
    )
    args = parser.parse_args()
    
    # Paths
    results_dir = REPO_ROOT / "results" / "housing_renting" / args.trial
    csv_path = results_dir / "sweep_results.csv"
    
    if not csv_path.exists():
        print(f"ERROR: Results CSV not found: {csv_path}")
        print(f"Make sure the sweep has been run and CSV copied to results folder.")
        sys.exit(1)
    
    print(f"Loading results from: {csv_path}")
    
    # Load data
    df = load_sweep_results(csv_path)
    
    # Print summary
    print(f"\nLoaded {len(df)} configurations:")
    print(f"  Methods: {sorted(df['method'].unique())}")
    print(f"  Grid sizes: {sorted(df['grid'].unique())}")
    print(f"  H sizes: {sorted(df['H'].unique())}")
    
    # Determine periods (T) - from argument, experiment YAML, or default
    periods = args.periods
    if periods is None and args.experiment_set:
        # Try to read from experiment YAML
        try:
            import yaml
            exp_set_path = (
                REPO_ROOT / "experiments" / "housing_renting" / 
                "experiment_sets" / f"{args.experiment_set}.yml"
            )
            if exp_set_path.exists():
                with open(exp_set_path) as f:
                    exp_config = yaml.safe_load(f)
                periods = exp_config.get("fixed", {}).get("periods", 5)
                print(f"  Read periods={periods} from {args.experiment_set}.yml")
        except Exception as e:
            print(f"  Warning: Could not read experiment set: {e}")
            periods = 5
    # If experiment set wasn't found/read, fall back safely
    if periods is None:
        periods = 5  # Default fallback
    
    # Model parameters string with proper LaTeX math formatting
    model_params = (
        rf"Model parameters: $r=0.06$, $\beta=0.93$, $T={periods}$, $\phi=0.07$, "
        r"$\alpha=0.77$, $\delta_{\mathrm{PB}}=1.0$, $A_{\max}=35$, $H_{\max}=5$, $w_{\max}=40$."
    )
    
    # Generate all tables
    print(f"\nGenerating tables in: {results_dir}")
    generate_accuracy_latex(df, results_dir / "housing_accuracy.tex", model_params)
    generate_timing_latex(df, results_dir / "housing_timing.tex", model_params)
    generate_accuracy_markdown(df, results_dir / "housing_accuracy.md", model_params)
    generate_timing_markdown(df, results_dir / "housing_timing.md", model_params)
    
    print(f"\nAll tables generated in: {results_dir}")


if __name__ == "__main__":
    main()













