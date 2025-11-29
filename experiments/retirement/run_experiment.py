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

# Add repo root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, REPO_ROOT)

from examples.retirement.plots import plot_egrids, plot_cons_pol, plot_dcegm_cf
from examples.retirement.benchmarks import test_Timings
from dc_smm.models.retirement.retirement import Operator_Factory, RetirementModel, euler


def parse_list(s, dtype=float):
    """Parse comma-separated string into list."""
    return [dtype(x.strip()) for x in s.split(',')]


def load_params(params_file):
    """Load model parameters from YAML file."""
    # Handle relative paths from script directory
    if not os.path.isabs(params_file):
        params_file = os.path.join(SCRIPT_DIR, params_file)

    with open(params_file, 'r') as f:
        config = yaml.safe_load(f)

    return config.get('model', {})


def main():
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

    args = parser.parse_args()

    # Load parameters from YAML
    params = load_params(args.params)
    print(f'Loaded parameters from: {args.params}')

    # Setup output directories
    save_path = os.path.join(args.output_dir, 'plots')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tables'), exist_ok=True)

    # Grid size: command line overrides params file
    grid_size = args.grid_size if args.grid_size is not None else params.get('grid_size', 3000)

    print(f'Running with grid_size={grid_size}, plot_age={args.plot_age}')
    print(f'Output: {args.output_dir}')

    # Run timing comparison if requested
    if args.run_timings:
        grid_sizes = parse_list(args.sweep_grids, int)
        delta_values = parse_list(args.sweep_deltas, float)
        m_bar = params.get('m_bar', 1.2)
        print(f'\nRunning timing comparison...')
        print(f'  Grid sizes: {grid_sizes}')
        print(f'  Delta values: {delta_values}')
        print(f'  Runs per config: {args.sweep_runs}')
        print(f'  m_bar: {m_bar}')
        test_Timings(grid_sizes, delta_values, n=args.sweep_runs,
                     results_dir=args.output_dir, m_bar=m_bar)

    # Create model and solve
    print('\nCreating model and solving...')
    cp = RetirementModel(
        r=params.get('r', 0.02),
        beta=params.get('beta', 0.98),
        delta=params.get('delta', 1.0),
        y=params.get('y', 20),
        b=params.get('b', 1e-10),
        grid_max_A=params.get('grid_max_A', 500),
        grid_size=grid_size,
        T=params.get('T', 20),
        smooth_sigma=params.get('smooth_sigma', 0),
        m_bar=params.get('m_bar', 1.2),
    )

    Ts_ret, Ts_work, iter_bell = Operator_Factory(cp)

    # Precompile and run methods
    _ = iter_bell(cp, method='RFC')
    e_grid, vf_unref, vf_ref, c_unref, c_RFC, dela, t_RFC = iter_bell(cp, method='RFC')

    _ = iter_bell(cp, method='FUES')
    _, _, _, _, c_FUES, _, t_FUES = iter_bell(cp, method='FUES')

    _ = iter_bell(cp, method='DCEGM')
    _, _, _, _, c_DCEGM, _, t_DCEGM = iter_bell(cp, method='DCEGM')

    _ = iter_bell(cp, method='CONSAV')
    _, _, _, _, c_CONSAV, _, t_CONSAV = iter_bell(cp, method='CONSAV')

    # Compute Euler errors
    err_RFC = euler(cp, c_RFC)
    err_FUES = euler(cp, c_FUES)
    err_DCEGM = euler(cp, c_DCEGM)
    err_CONSAV = euler(cp, c_CONSAV)

    print()
    print('| Method | Euler Error    | Avg UE time(ms) | Total time(ms) |')
    print('|--------|----------------|-----------------|----------------|')
    print(f'| RFC    | {err_RFC:<14.6f} | {t_RFC[0]*1000:<15.3f} | {t_RFC[1]*1000:<14.3f} |')
    print(f'| FUES   | {err_FUES:<14.6f} | {t_FUES[0]*1000:<15.3f} | {t_FUES[1]*1000:<14.3f} |')
    print(f'| DCEGM  | {err_DCEGM:<14.6f} | {t_DCEGM[0]*1000:<15.3f} | {t_DCEGM[1]*1000:<14.3f} |')
    print(f'| CONSAV | {err_CONSAV:<14.6f} | {t_CONSAV[0]*1000:<15.3f} | {t_CONSAV[1]*1000:<14.3f} |')
    print()

    # Generate plots
    print(f'Generating plots to {save_path}...')
    plot_egrids(args.plot_age, e_grid, vf_unref, c_unref, dela,
                args.grid_size, cp, save_path)
    plot_cons_pol(c_FUES, cp, save_path)
    plot_dcegm_cf(args.plot_age, args.grid_size, e_grid, vf_unref, c_unref,
                  dela, cp.asset_grid_A, cp, save_path)

    print('Done!')


if __name__ == '__main__':
    main()
