#!/usr/bin/env python3
"""Run retirement model experiments via the canonical pipeline.

Usage:
    python -m examples.retirement.run
    python -m examples.retirement.run --calib-override beta=0.96
    python -m examples.retirement.run --setting-override grid_size=5000
    python -m examples.retirement.run --setting-override run_timings=1
    python -m examples.retirement.run --setting-override plot_age=10
"""

import os
from pathlib import Path

from kikku.run import parse_run

from examples.retirement.solve import solve_nest, METHOD_SHORTCUT

from examples.retirement.outputs import (
    plot_egrids, plot_cons_pol, plot_dcegm_cf,
    euler, get_policy, get_timing,
)
from examples.retirement.benchmark import test_Timings

UE_METHODS = ('RFC', 'FUES', 'DCEGM', 'CONSAV')


def _solver_config(run):
    """Merge settings tier + config tier for ``load_syntax`` (settings.yaml overlay)."""
    return {**dict(run.settings), **dict(run.config)}


def main():
    run = parse_run(
        name='retirement',
        syntax='examples/retirement/syntax',
        methods=list(UE_METHODS),
        modes=['sweep'],
        output='results/retirement',
    )

    calib_overrides = run.calib or None
    solver_cfg = _solver_config(run)
    config_overrides = solver_cfg if solver_cfg else None

    eff = dict(solver_cfg)
    plot_age = int(eff.get('plot_age', 5))
    run_timings = bool(int(eff.get('run_timings', 0)))
    sweep_deltas_str = str(eff.get('sweep_deltas', '0.25,0.5,1,2'))
    latex_grids_str = eff.get('latex_grids', None)

    run_dir = str(run.output_dir)
    syntax_dir = run.model_dir

    print(f'Model dir: {syntax_dir}')
    print(f'Output directory: {run_dir}')
    if calib_overrides:
        print(f'Calib overrides: {calib_overrides}')
    if config_overrides:
        print(f'Config overrides: {config_overrides}')

    save_path = os.path.join(run_dir, 'plots')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'tables'), exist_ok=True)

    if run_timings:
        grid_sizes_str = ','.join(str(g) for g in run.sweep_grids) \
            if run.sweep_grids else '500,1000,2000,3000,10000'
        grid_sizes = [int(x) for x in grid_sizes_str.split(',')]
        delta_values = [float(x) for x in sweep_deltas_str.split(',')]
        print(f'\nRunning timing comparison...')
        print(f'  Grid sizes: {grid_sizes}')
        print(f'  Delta values: {delta_values}')
        print(f'  Runs per config: {run.sweep_runs}')
        latex_grids = None
        if latex_grids_str is not None:
            latex_grids = [int(x) for x in str(latex_grids_str).split(',')]
        test_Timings(
            grid_sizes, delta_values, n=run.sweep_runs,
            results_dir=os.path.join(run_dir, 'tables'),
            calib_overrides=calib_overrides,
            config_overrides=config_overrides,
            latex_grids=latex_grids,
        )

    print('\nSolving via canonical pipeline...')
    solutions = {}
    for method in UE_METHODS:
        if run.method_overrides:
            ue = {t: method for t in METHOD_SHORTCUT}
            ue.update(run.method_overrides)
        else:
            ue = method
        _, m_, ops_, w_ = solve_nest(
            syntax_dir,
            ue_method=ue,
            draw={"calibration": calib_overrides, "settings": config_overrides},
        )
        nest, model, _, _ = solve_nest(
            syntax_dir,
            ue_method=ue,
            draw={"calibration": calib_overrides, "settings": config_overrides},
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
