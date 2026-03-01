#!/usr/bin/env python3
"""
Generate paper tables from sweep results.

Usage:
    # Generate tables for a specific trial (looks in /scratch for sweep_results.csv):
    python gen_tables.py test_0.3-paper-sweep-10
    
    # Generate tables from a specific CSV file:
    python gen_tables.py --results /path/to/sweep_results.csv --output results/housing_renting/my-trial
    
    # Specify number of periods (default: 20):
    python gen_tables.py test_0.3-paper-sweep-10 --periods 5
"""

import sys
import argparse
from pathlib import Path

# Add examples to path for imports
sys.path.insert(0, str(Path(__file__).parent / "examples" / "housing_renting"))

from helpers.generate_paper_tables import (
    load_sweep_results,
    generate_accuracy_latex,
    generate_timing_latex,
    generate_accuracy_markdown,
    generate_timing_markdown,
)


SOLUTIONS_ROOT = Path("/scratch/tp66/as3442/FUES/solutions/housing_renting")
RESULTS_ROOT = Path(__file__).parent / "results" / "housing_renting"


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper tables from sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("trial", nargs="?", type=str, default=None,
                        help="Trial ID (e.g., test_0.3-paper-sweep-10)")
    parser.add_argument("--results", type=Path, default=None,
                        help="Path to sweep_results.csv (overrides trial)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: results/housing_renting/<trial>/)")
    parser.add_argument("--periods", type=int, default=20,
                        help="Number of periods T (default: 20)")
    args = parser.parse_args()
    
    # Determine results path
    if args.results:
        results_path = args.results
        trial = args.results.parent.name
    elif args.trial:
        results_path = SOLUTIONS_ROOT / args.trial / "sweep_results.csv"
        trial = args.trial
    else:
        parser.print_help()
        print("\nError: Please provide either a trial ID or --results path")
        sys.exit(1)
    
    print(f"Loading results from: {results_path}")
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    
    # Load data
    df = load_sweep_results(results_path)
    
    # Print summary
    print(f"\nLoaded {len(df)} configurations:")
    print(f"  Methods: {sorted(df['method'].unique())}")
    print(f"  Grid sizes: {sorted(df['grid'].unique())}")
    print(f"  H sizes: {sorted(df['H'].unique())}")
    
    # Model parameters string - exact text from user template
    model_params = (
        rf"Parameters: $r=0.06$, $\beta=0.93$, $T={args.periods}$, $\alpha=0.77$, $\phi=0.07$, "
        r"$\kappa=0.075$, $\iota=0.01$, $A_{\max}=40$, $H_{\max}=5$, $w_{\max}=40$, $P^{r}=0.1$, $\vartheta=0.5$. "
        r"Income: $\log y = z + \varepsilon$; $z_t = \rho z_{t-1} + \eta_t$ with "
        r"$\rho=0.977$, $\sigma_{\eta}=0.024$, $\sigma_{\varepsilon}=0.063$ on $7 \times 7$ grid."
    )
    
    # Output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = RESULTS_ROOT / trial
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all tables
    generate_accuracy_latex(df, output_dir / "housing_accuracy.tex", model_params)
    generate_timing_latex(df, output_dir / "housing_timing.tex", model_params)
    generate_accuracy_markdown(df, output_dir / "housing_accuracy.md", model_params)
    generate_timing_markdown(df, output_dir / "housing_timing.md", model_params)
    
    print(f"\n✓ All tables generated in: {output_dir}")


if __name__ == "__main__":
    main()


