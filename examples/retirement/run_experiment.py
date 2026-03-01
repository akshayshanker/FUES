#!/usr/bin/env python3
"""Run retirement model experiments.

Usage:
    python run_experiment.py --grid-size 3000 --plot-age 5 --output-dir results/retirement
    python run_experiment.py --params params/baseline.yml
    python run_experiment.py --run-timings  # Run full timing sweep
"""

import argparse
import os
import sys
import yaml

# Add repo root + src/ to path so `dcsmm` imports work without installation
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SRC_ROOT)

from code.retirement import Operator_Factory, RetirementModel, euler
from code.plots import plot_egrids, plot_cons_pol, plot_dcegm_cf
from code.benchmarks import test_Timings

UE_METHODS = ('RFC', 'FUES', 'DCEGM', 'CONSAV')


def parse_list(s, dtype=float):
    """Parse comma-separated string into list."""
    return [dtype(x.strip()) for x in s.split(',')]


def load_params(params_file):
    """Load model and benchmark parameters from YAML file."""
    # Handle relative paths from script directory
    if not os.path.isabs(params_file):
        params_file = os.path.join(SCRIPT_DIR, params_file)

    with open(params_file, 'r') as f:
        config = yaml.safe_load(f)

    model_params = config.get('model', {})
    benchmark_params = config.get('benchmark', {})
    return model_params, benchmark_params


def solve_method(backward_induction, cp, method):
    """Warmup (JIT compile) then timed solve. Returns named dict."""
    backward_induction(cp, method=method)              # warmup
    result = backward_induction(cp, method=method)     # timed
    return {
        'endog_grid':     result[0],
        'vf_unrefined':   result[1],
        'vf_refined':     result[2],
        'c_unrefined':    result[3],
        'c_refined':      result[4],
        'dela_unrefined': result[5],
        'timing':         result[6],   # [avg_ue_time, avg_total_time]
    }


def parse_cli():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run retirement model experiments')

    # Parameter file
    parser.add_argument('--params', type=str, default='params/baseline.yml',
                        help='YAML parameter file (default: params/baseline.yml)')

    # Overrides (these override params file if specified)
    parser.add_argument('--grid-size', type=int, default=None,
                        help='Grid size (overrides params file)')
    parser.add_argument('--plot-age', type=int, default=5,
                        help='Age to plot EGM grids (default: 5)')
    parser.add_argument('--output-dir', type=str, default='results/retirement',
                        help='Output directory (default: results/retirement)')

    # Timing sweep settings
    parser.add_argument('--run-timings', action='store_true',
                        help='Run full timing comparison sweep')
    parser.add_argument('--sweep-grids', type=str, default='500,1000,2000,3000,10000',
                        help='Comma-separated grid sizes for sweep')
    parser.add_argument('--sweep-deltas', type=str, default='0.25,0.5,1,2',
                        help='Comma-separated delta values for sweep')
    parser.add_argument('--sweep-runs', type=int, default=3,
                        help='Number of runs per config (best of n)')

    return parser.parse_args()


def main():
    # ── Parse CLI ──
    args = parse_cli()

    # ── Load parameters (I/O boundary) ──
    params, benchmark_params = load_params(args.params)
    print(f'Loaded parameters from: {args.params}')

    # ── Setup output dirs ──
    save_path = os.path.join(args.output_dir, 'plots')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tables'), exist_ok=True)

    grid_size = args.grid_size if args.grid_size is not None else params.get('grid_size', 3000)
    smooth_sigma = float(params.get('smooth_sigma', 0.0))
    sigma_tag = "sigma0" if abs(smooth_sigma) < 1e-12 else f"sigma{int(round(smooth_sigma * 100)):02d}"

    print(f'Running with grid_size={grid_size}, plot_age={args.plot_age}')
    print(f'Output: {args.output_dir}')
    print(f'smooth_sigma={smooth_sigma} (tag={sigma_tag})')

    # ── (Optional) Timing sweep ──
    if args.run_timings:
        grid_sizes = parse_list(args.sweep_grids, int)
        delta_values = parse_list(args.sweep_deltas, float)
        m_bar = params.get('m_bar', 1.2)
        true_grid_size = benchmark_params.get('true_grid_size', 20000)
        true_method = benchmark_params.get('true_method', 'DCEGM')
        print(f'\nRunning timing comparison...')
        print(f'  Grid sizes: {grid_sizes}')
        print(f'  Delta values: {delta_values}')
        print(f'  Runs per config: {args.sweep_runs}')
        print(f'  m_bar: {m_bar}')
        print(f'  True solution: {true_method} with {true_grid_size} grid points')
        test_Timings(grid_sizes, delta_values, n=args.sweep_runs,
                     results_dir=args.output_dir, m_bar=m_bar,
                     true_grid_size=true_grid_size, true_method=true_method)

    # ── Calibrate model ──
    print('\nCreating model and solving...')
    cp = RetirementModel(
        r=params.get('r', 0.02),
        beta=params.get('beta', 0.98),
        delta=params.get('delta', 1.0),
        y=params.get('y', 20),
        b=params.get('b', 1e-10),
        grid_max_A=params.get('grid_max_A', 500),
        grid_size=grid_size,
        T=params.get('T', 50),
        smooth_sigma=params.get('smooth_sigma', 0),
        m_bar=params.get('m_bar', 1.2),
    )

    # ── Build operators ──
    _, _, backward_induction = Operator_Factory(cp)

    # ── Solve (compare 4 UE methods) ──
    solutions = {}
    for method in UE_METHODS:
        solutions[method] = solve_method(backward_induction, cp, method)

    # ── Evaluate (Euler errors) ──
    errors = {}
    for method in UE_METHODS:
        errors[method] = euler(cp, solutions[method]['c_refined'])

    # ── Report ──
    print()
    print('| Method | Euler Error    | Avg UE time(ms) | Total time(ms) |')
    print('|--------|----------------|-----------------|----------------|')
    for method in UE_METHODS:
        t = solutions[method]['timing']
        print(f'| {method:<6s} | {errors[method]:<14.6f} | {t[0]*1000:<15.3f} | {t[1]*1000:<14.3f} |')
    print()

    # ── Plot ──
    print(f'Generating plots to {save_path}...')
    rfc = solutions['RFC']
    plot_egrids(args.plot_age, rfc['endog_grid'], rfc['vf_unrefined'],
                rfc['c_unrefined'], rfc['dela_unrefined'],
                grid_size, cp, save_path, tag=sigma_tag)
    plot_cons_pol(solutions['FUES']['c_refined'], cp, save_path)
    plot_dcegm_cf(args.plot_age, grid_size, rfc['endog_grid'], rfc['vf_unrefined'],
                  rfc['c_unrefined'], rfc['dela_unrefined'],
                  cp.asset_grid_A, cp, save_path, tag=sigma_tag)

    print('Done!')


if __name__ == '__main__':
    main()
