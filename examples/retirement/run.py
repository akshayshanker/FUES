#!/usr/bin/env python3
"""Run retirement model experiments via the canonical pipeline.

Usage:
    python -m examples.retirement.run
    python -m examples.retirement.run --config-override grid_size=5000
    python -m examples.retirement.run --calib-override beta=0.96
    python -m examples.retirement.run --config-override run_timings=1
    python -m examples.retirement.run --config-override plot_age=10
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SRC_ROOT)

from kikku.run.cli import (
    add_override_args, add_method_args,
    add_output_args, add_sweep_args, build_overrides,
    make_run_dir, load_effective_settings,
)

from examples.retirement.solve import solve_nest
from examples.retirement.outputs import (
    plot_egrids, plot_cons_pol, plot_dcegm_cf,
    euler, get_policy, get_timing,
)
from examples.retirement.benchmark import test_Timings

SYNTAX_DIR = Path(__file__).resolve().parent / "syntax"
SETTINGS_PATH = SYNTAX_DIR / 'settings.yaml'

UE_METHODS = ('RFC', 'FUES', 'DCEGM', 'CONSAV')


def main():
    parser = argparse.ArgumentParser(
        description='Run retirement model via canonical pipeline',
    )
    add_override_args(parser)
    add_method_args(parser, choices=list(UE_METHODS))
    add_output_args(parser, default_dir='results/retirement')
    add_sweep_args(parser)

    args = parser.parse_args()

    calib_overrides, config_overrides = build_overrides(
        args, settings_path=SETTINGS_PATH)

    eff = load_effective_settings(SETTINGS_PATH, config_overrides)
    plot_age = int(eff.get('plot_age', 5))
    run_timings = bool(int(eff.get('run_timings', 0)))
    sweep_deltas_str = str(eff.get('sweep_deltas', '0.25,0.5,1,2'))
    latex_grids_str = eff.get('latex_grids', None)

    run_dir = make_run_dir(args.output_dir, tag=args.run_tag)

    print(f'Syntax dir: {SYNTAX_DIR}')
    print(f'Output directory: {run_dir}')
    if calib_overrides:
        print(f'Calib overrides: {calib_overrides}')
    if config_overrides:
        print(f'Config overrides: {config_overrides}')

    save_path = os.path.join(run_dir, 'plots')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'tables'), exist_ok=True)

    if run_timings:
        grid_sizes_str = args.sweep_grids or '500,1000,2000,3000,10000'
        grid_sizes = [int(x) for x in grid_sizes_str.split(',')]
        delta_values = [float(x) for x in sweep_deltas_str.split(',')]
        print(f'\nRunning timing comparison...')
        print(f'  Grid sizes: {grid_sizes}')
        print(f'  Delta values: {delta_values}')
        print(f'  Runs per config: {args.sweep_runs}')
        latex_grids = None
        if latex_grids_str is not None:
            latex_grids = [int(x) for x in str(latex_grids_str).split(',')]
        test_Timings(
            grid_sizes, delta_values, n=args.sweep_runs,
            results_dir=run_dir,
            calib_overrides=calib_overrides,
            config_overrides=config_overrides,
            latex_grids=latex_grids,
        )

    print('\nSolving via canonical pipeline...')
    solutions = {}
    for method in UE_METHODS:
        _, m_, ops_, w_ = solve_nest(
            SYNTAX_DIR, method=method,
            calib_overrides=calib_overrides,
            config_overrides=config_overrides,
        )
        nest, model, _, _ = solve_nest(
            SYNTAX_DIR, method=method,
            calib_overrides=calib_overrides,
            config_overrides=config_overrides,
            model=m_, stage_ops=ops_, waves=w_,
        )
        solutions[method] = {
            'nest': nest,
            'model': model,
            'endog_grid': get_policy(nest, 'x_dcsn_hat', stage='work_cons'),
            'vf_unrefined': get_policy(nest, 'v_dcsn_hat', stage='work_cons'),
            'c_unrefined': get_policy(nest, 'c_dcsn_hat', stage='work_cons'),
            'dela_unrefined': get_policy(nest, 'dela_dcsn_hat', stage='work_cons'),
            'c_refined': get_policy(nest, 'c', stage='labour_mkt_decision'),
            'c_worker': get_policy(nest, 'c', stage='work_cons'),
            'timing': get_timing(nest),
        }

    model = solutions[UE_METHODS[0]]['model']
    grid_size = model.grid_size
    smooth_sigma = model.smooth_sigma
    sigma_tag = "sigma0" if abs(smooth_sigma) < 1e-12 \
        else f"sigma{int(round(smooth_sigma * 100)):02d}"

    errors = {}
    for method in UE_METHODS:
        errors[method] = euler(model, solutions[method]['c_refined'])

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

    print(f'Generating plots to {save_path}...')
    rfc = solutions['RFC']
    plot_egrids(
        plot_age, rfc['endog_grid'], rfc['vf_unrefined'],
        rfc['c_unrefined'], rfc['dela_unrefined'],
        grid_size, model, save_path, tag=sigma_tag,
    )
    plot_cons_pol(solutions['FUES']['c_worker'], model, save_path)
    plot_dcegm_cf(
        plot_age, grid_size, rfc['endog_grid'],
        rfc['vf_unrefined'], rfc['c_unrefined'],
        rfc['dela_unrefined'], model.asset_grid_A,
        model, save_path, tag=sigma_tag,
    )

    print('Done!')


if __name__ == '__main__':
    main()
