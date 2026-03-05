#!/usr/bin/env python3
"""Run retirement model experiments via the canonical pipeline.

Usage:
    python run.py --output-dir results/retirement
    python run.py --config-override grid_size=5000
    python run.py --calib-override beta=0.96 --method DCEGM
    python run.py --override-file ../../experiments/retirement/params/high_beta.yml
    python run.py --run-timings
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add repo root + src/ to path so `dcsmm` imports work without installation
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SRC_ROOT)

from examples.retirement.solve import solve_nest
from examples.retirement.outputs import (
    plot_egrids, plot_cons_pol, plot_dcegm_cf,
    euler, get_policy, get_timing,
)
from examples.retirement.benchmark import test_Timings

SYNTAX_DIR = Path(__file__).resolve().parent / "syntax"

UE_METHODS = ('RFC', 'FUES', 'DCEGM', 'CONSAV')


def parse_overrides(raw_list):
    """Parse 'key=value' strings into a dict, coercing types."""
    result = {}
    for item in (raw_list or []):
        if '=' not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, val = item.split('=', 1)
        result[key.strip()] = yaml.safe_load(val.strip())
    return result


def load_override_file(path):
    """Load overrides from a YAML file (flat key-value format)."""
    if not os.path.isabs(path):
        path = os.path.join(SCRIPT_DIR, path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Support both flat format and nested 'overrides:' wrapper
    if isinstance(raw, dict) and 'overrides' in raw:
        return raw['overrides']
    return raw or {}


def parse_cli():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run retirement model via canonical pipeline',
    )

    # Override mechanism (three-functor)
    parser.add_argument(
        '--calib-override', action='append', default=[],
        help='Economic param override: key=value (repeatable)',
    )
    parser.add_argument(
        '--config-override', action='append', default=[],
        help='Numerical setting override: key=value (repeatable)',
    )
    parser.add_argument(
        '--override-file', type=str, default=None,
        help='YAML file with sparse overrides',
    )
    parser.add_argument(
        '--method', type=str, default='FUES',
        choices=['RFC', 'FUES', 'DCEGM', 'CONSAV'],
        help='Upper-envelope method (default: FUES)',
    )

    # Convenience flags
    parser.add_argument(
        '--grid-size', type=int, default=None,
        help='Grid size (shorthand for --config-override grid_size=N)',
    )
    parser.add_argument(
        '--plot-age', type=int, default=5,
        help='Age to plot EGM grids (default: 5)',
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/retirement',
        help='Output directory (default: results/retirement)',
    )

    # Timing sweep
    parser.add_argument(
        '--run-timings', action='store_true',
        help='Run full timing comparison sweep',
    )
    parser.add_argument(
        '--sweep-grids', type=str, default='500,1000,2000,3000,10000',
        help='Comma-separated grid sizes for sweep',
    )
    parser.add_argument(
        '--sweep-deltas', type=str, default='0.25,0.5,1,2',
        help='Comma-separated delta values for sweep',
    )
    parser.add_argument(
        '--sweep-runs', type=int, default=3,
        help='Number of runs per config (best of n)',
    )
    parser.add_argument(
        '--latex-grids', type=str, default=None,
        help='Comma-separated grid sizes for LaTeX tables (subset of sweep-grids)',
    )

    return parser.parse_args()


def main():
    args = parse_cli()

    # ── Build override dicts ──
    calib_overrides = parse_overrides(args.calib_override)
    config_overrides = parse_overrides(args.config_override)

    # Override file: split keys into calib vs config based on settings.yaml keys
    if args.override_file:
        file_overrides = load_override_file(args.override_file)
        settings_path = SYNTAX_DIR / 'settings.yaml'
        with open(settings_path) as f:
            settings_keys = set(yaml.safe_load(f).get('settings', {}).keys())
        for k, v in file_overrides.items():
            if k in settings_keys:
                config_overrides[k] = v
            else:
                calib_overrides[k] = v

    # Convenience: --grid-size N -> config override
    if args.grid_size is not None:
        config_overrides['grid_size'] = args.grid_size

    print(f'Syntax dir: {SYNTAX_DIR}')
    if calib_overrides:
        print(f'Calib overrides: {calib_overrides}')
    if config_overrides:
        print(f'Config overrides: {config_overrides}')

    # ── Setup output dirs ──
    save_path = os.path.join(args.output_dir, 'plots')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tables'), exist_ok=True)

    # ── (Optional) Timing sweep ──
    if args.run_timings:
        grid_sizes = [int(x) for x in args.sweep_grids.split(',')]
        delta_values = [float(x) for x in args.sweep_deltas.split(',')]
        print(f'\nRunning timing comparison...')
        print(f'  Grid sizes: {grid_sizes}')
        print(f'  Delta values: {delta_values}')
        print(f'  Runs per config: {args.sweep_runs}')
        latex_grids = [int(x) for x in args.latex_grids.split(',')] \
            if args.latex_grids else None
        test_Timings(
            grid_sizes, delta_values, n=args.sweep_runs,
            results_dir=args.output_dir,
            calib_overrides=calib_overrides,
            config_overrides=config_overrides,
            latex_grids=latex_grids,
        )

    # ── Solve via canonical pipeline (compare 4 UE methods) ──
    print('\nSolving via canonical pipeline...')
    solutions = {}
    for method in UE_METHODS:
        # Warmup (JIT compile)
        solve_nest(
            SYNTAX_DIR, method=method,
            calib_overrides=calib_overrides,
            config_overrides=config_overrides,
        )
        # Timed run
        nest, model, _ = solve_nest(
            SYNTAX_DIR, method=method,
            calib_overrides=calib_overrides,
            config_overrides=config_overrides,
        )
        solutions[method] = {
            'nest': nest,
            'model': model,
            'endog_grid': get_policy(nest, 'egrid', stage='work_cons'),
            'vf_unrefined': get_policy(nest, 'q_hat', stage='work_cons'),
            'c_unrefined': get_policy(nest, 'c_hat', stage='work_cons'),
            'dela_unrefined': get_policy(nest, 'da_pre_ue', stage='work_cons'),
            'c_refined': get_policy(nest, 'c', stage='labour_mkt_decision'),
            'c_worker': get_policy(nest, 'c', stage='work_cons'),
            'timing': get_timing(nest),
        }

    # Use model from first solve for euler errors
    model = solutions[UE_METHODS[0]]['model']
    grid_size = model.grid_size
    smooth_sigma = model.smooth_sigma
    sigma_tag = "sigma0" if abs(smooth_sigma) < 1e-12 \
        else f"sigma{int(round(smooth_sigma * 100)):02d}"

    # ── Evaluate (Euler errors) ──
    errors = {}
    for method in UE_METHODS:
        errors[method] = euler(model, solutions[method]['c_refined'])

    # ── Report ──
    print()
    print('| Method | Euler Error    | Avg UE time(ms) | Total time(ms) |')
    print('|--------|----------------|-----------------|----------------|')
    for method in UE_METHODS:
        t = solutions[method]['timing']
        print(
            f'| {method:<6s} | {errors[method]:<14.6f} '
            f'| {t[0]*1000:<15.3f} | {t[1]*1000:<14.3f} |'
        )
    print()

    # ── Plot ──
    print(f'Generating plots to {save_path}...')
    rfc = solutions['RFC']
    plot_egrids(
        args.plot_age, rfc['endog_grid'], rfc['vf_unrefined'],
        rfc['c_unrefined'], rfc['dela_unrefined'],
        grid_size, model, save_path, tag=sigma_tag,
    )
    plot_cons_pol(solutions['FUES']['c_worker'], model, save_path)
    plot_dcegm_cf(
        args.plot_age, grid_size, rfc['endog_grid'],
        rfc['vf_unrefined'], rfc['c_unrefined'],
        rfc['dela_unrefined'], model.asset_grid_A,
        model, save_path, tag=sigma_tag,
    )

    print('Done!')


if __name__ == '__main__':
    main()
