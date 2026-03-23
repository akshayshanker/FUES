"""Run the durables2_0 DDSL pipeline: single-point, method comparison, or sweep.

Usage:
    # Method comparison (FUES vs NEGM)
    python -m examples.durables2_0.run --compare
    python -m examples.durables2_0.run --compare --calib-override t0=50

    # Single-point (default)
    python -m examples.durables2_0.run
    python -m examples.durables2_0.run --method FUES --simulate
    python -m examples.durables2_0.run --calib-override t0=50

    # Override plot ages or EGM grid storage via settings
    python -m examples.durables2_0.run --config-override plot_ages=50,55
    python -m examples.durables2_0.run --config-override store_cntn=1

    # Parameter sweep
    python -m examples.durables2_0.run --sweep --sweep-grids 100,200,300
"""

import os
import argparse
import numpy as np

from kikku.run.cli import (
    add_override_args, add_method_args,
    add_output_args, add_simulation_args,
    add_sweep_args, build_overrides,
    make_run_dir, load_effective_settings,
)
from kikku.run.sweep import param_grid, sweep, parse_sweep_args
from kikku.run.metrics import format_table, write_table

from .solve import solve
from .outputs import (
    plot_policies, plot_grids, plot_lifecycle, get_timing, derive_savings,
    consumption_deviation, compute_euler_stats, print_euler_stats,
    generate_comparison_table,
)
from .simulate import simulate_and_euler

SETTINGS_PATH = 'examples/durables2_0/syntax/settings.yaml'


def run_single(
    syntax_dir='examples/durables2_0/syntax',
    method='FUES',
    calib_overrides=None,
    config_overrides=None,
    output_dir='results/durables2_0/plots',
    plot_ages=None,
    store_cntn=False,
    sim=False,
    n_sim=10000,
    seed=42,
    use_empirical_init=False,
):
    """Solve and optionally simulate + plot.

    Returns
    -------
    dict
        ``{'nest', 'cp', 'grids', 'callables', 'settings',
        'results_row', 'euler_stats'}``.
        ``euler_stats`` is ``None`` when ``sim=False``.
    """
    os.environ['FUES_RETURN_GRIDS'] = '1' if store_cntn else '0'

    cfg = dict(config_overrides or {})
    if store_cntn:
        cfg['store_cntn'] = 1

    nest, cp, grids, callables, settings = solve(
        syntax_dir, method=method, verbose=False,
        calib_overrides=calib_overrides,
        config_overrides=cfg if cfg else None)
    print(f'{len(nest["solutions"])} periods solved')

    timing = get_timing(nest)
    print(f'Mean timing — keeper: {timing["keeper_ms"]:.1f}ms, '
          f'adj: {timing["adj_ms"]:.1f}ms')

    results_row = {
        'Method': method,
        'Keeper (ms)': timing['keeper_ms'],
        'Adj (ms)': timing['adj_ms'],
        'Total (ms)': timing['keeper_ms'] + timing['adj_ms'],
    }
    euler_stats = None

    if output_dir:
        all_t = sorted(s['t'] for s in nest['solutions'])
        if plot_ages is None:
            plot_ages = [all_t[-3] if len(all_t) >= 3 else all_t[-1]]

        savings = derive_savings(nest, grids, cp.tau)

        for age in plot_ages:
            if age not in all_t:
                print(f'Age {age} not in solution '
                      f'(range {all_t[0]}–{all_t[-1]}), skipping.')
                continue
            plot_policies(nest, grids, savings,
                          output_dir=output_dir, plot_t=age)
            if store_cntn:
                plot_grids(nest, grids,
                           output_dir=output_dir, plot_t=age)

    if sim:
        euler, sim_data = simulate_and_euler(
            nest, cp, grids, callables, settings,
            N=n_sim, seed=seed,
            use_empirical_init=use_empirical_init)

        euler_stats = compute_euler_stats(euler, sim_data['discrete'])
        print_euler_stats(euler_stats)

        d = sim_data['discrete']
        adj_rate = np.mean(d[d >= 0]) * 100

        if 'combined' in euler_stats:
            results_row['Euler Combined'] = euler_stats['combined']['mean']
            results_row['Euler Keeper'] = euler_stats['keeper']['mean']
            results_row['Euler Adjuster'] = euler_stats['adjuster']['mean']
            results_row['Adj Rate'] = adj_rate

        if 'npv_utility' in sim_data:
            npv = sim_data['npv_utility']
            print(f"  NPV utility: mean={np.mean(npv):.4f}, "
                  f"std={np.std(npv):.4f}")

        if output_dir:
            plot_lifecycle(sim_data, euler, cp, output_dir=output_dir)

    return {
        'nest': nest, 'cp': cp, 'grids': grids,
        'callables': callables, 'settings': settings,
        'results_row': results_row, 'euler_stats': euler_stats,
    }


def run_comparison(
    syntax_dir='examples/durables2_0/syntax',
    calib_overrides=None,
    config_overrides=None,
    output_dir='results/durables2_0',
    plot_ages=None,
    store_cntn=False,
    sim=False,
    n_sim=10000,
    seed=42,
    use_empirical_init=False,
):
    """Solve with both FUES and NEGM, print and save a comparison table.

    Returns
    -------
    dict
        ``{'FUES': {nest, cp, ...}, 'NEGM': {nest, cp, ...}}``
    """
    methods = ['FUES', 'NEGM']
    all_results = {}
    rows = []
    euler_stats_by_method = {}

    for method in methods:
        method_dir = os.path.join(output_dir, 'plots', method)
        print(f'\n{"="*60}')
        print(f'  {method}')
        print(f'{"="*60}')

        result = run_single(
            syntax_dir=syntax_dir,
            method=method,
            calib_overrides=calib_overrides,
            config_overrides=config_overrides,
            output_dir=method_dir,
            plot_ages=plot_ages,
            store_cntn=store_cntn,
            sim=sim,
            n_sim=n_sim,
            seed=seed,
            use_empirical_init=use_empirical_init,
        )
        all_results[method] = result
        rows.append(result['results_row'])
        if result['euler_stats'] is not None:
            euler_stats_by_method[method] = result['euler_stats']

    table_dir = os.path.join(output_dir, 'tables')
    os.makedirs(table_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print('  Method comparison')
    print(f'{"="*60}')

    md_table = generate_comparison_table(rows, fmt='md',
                                         caption='Durables Model Comparison')
    print(md_table)

    md_path = os.path.join(table_dir, 'comparison.md')
    with open(md_path, 'w') as f:
        f.write(md_table)

    tex_table = generate_comparison_table(rows, fmt='tex',
                                          caption='Durables Model Comparison')
    tex_path = os.path.join(table_dir, 'comparison.tex')
    with open(tex_path, 'w') as f:
        f.write(tex_table)

    print(f'\nTables saved to {md_path}, {tex_path}')

    if sim and euler_stats_by_method:
        detail_lines = ['# Euler Error Detail\n']
        for method, es in euler_stats_by_method.items():
            detail_lines.append(f'## {method}\n')
            if 'combined' not in es:
                detail_lines.append(
                    f"Mean: {es['mean']:.4f}, "
                    f"Median: {es['median']:.4f}\n")
                continue
            detail_lines.append(
                f"Adjustment rate: {es['pct_adjuster']:.2f}% "
                f"({es['n_adjuster']} adj, {es['n_keeper']} keep)\n")
            hdr = f"{'':12} {'Combined':>12} {'Adjuster':>12} {'Keeper':>12}"
            detail_lines.append(hdr)
            detail_lines.append('-' * 60)
            for key in ['mean', 'median', 'std', 'p5', 'p95']:
                detail_lines.append(
                    f"{key:12} {es['combined'][key]:>12.4f} "
                    f"{es['adjuster'][key]:>12.4f} "
                    f"{es['keeper'][key]:>12.4f}")
            detail_lines.append(
                f"{'Frac>10^-3':12} "
                f"{es['combined']['frac_above_neg3']:>12.4f} "
                f"{es['adjuster']['frac_above_neg3']:>12.4f} "
                f"{es['keeper']['frac_above_neg3']:>12.4f}")
            detail_lines.append(
                f"{'Frac>10^-4':12} "
                f"{es['combined']['frac_above_neg4']:>12.4f} "
                f"{es['adjuster']['frac_above_neg4']:>12.4f} "
                f"{es['keeper']['frac_above_neg4']:>12.4f}")
            detail_lines.append(
                f"{'N obs':12} {es['combined']['n_obs']:>12} "
                f"{es['adjuster']['n_obs']:>12} "
                f"{es['keeper']['n_obs']:>12}")
            detail_lines.append('')

        detail_path = os.path.join(table_dir, 'euler_detail.md')
        with open(detail_path, 'w') as f:
            f.write('\n'.join(detail_lines))
        print(f'Euler detail saved to {detail_path}')
    elif not sim:
        print('\n(Run with --simulate for Euler accuracy columns)')

    return all_results


def run_sweep(
    syntax_dir='examples/durables2_0/syntax',
    method='FUES',
    grid_sizes=None,
    n_reps=3,
    warmup=True,
    calib_overrides=None,
    config_overrides=None,
    output_dir='results/durables2_0',
    comm=None,
):
    """Parameter sweep over grid sizes with timing metrics.

    Returns
    -------
    list[dict] — flat results from sweep().
    """
    if grid_sizes is None:
        grid_sizes = [100, 200, 300]

    grid = param_grid(n_a=grid_sizes)

    base_calib = dict(calib_overrides or {})
    base_config = dict(config_overrides or {})

    def solve_fn(ov):
        cfg = {**base_config, 'n_a': ov['n_a']}
        nest, cp, grids, callables, settings = solve(
            syntax_dir, method=method,
            calib_overrides=base_calib,
            config_overrides=cfg,
            verbose=False)
        return {'nest': nest, 'cp': cp, 'grids': grids}

    metric_fns = {
        'solve_ms': lambda r: get_timing(r['nest'])['solve_time'] * 1000,
        'keeper_ms': lambda r: get_timing(r['nest'])['keeper_ms'],
        'adj_ms': lambda r: get_timing(r['nest'])['adj_ms'],
    }

    results = sweep(
        solve_fn, grid, metric_fns,
        n_reps=n_reps, warmup=warmup,
        best='min', comm=comm)

    if results:
        cols = ['n_a', 'solve_ms', 'keeper_ms', 'adj_ms']
        print('\n' + format_table(results, cols))
        if output_dir:
            os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
            write_table(
                os.path.join(output_dir, 'tables', 'sweep.md'),
                results, cols)

    return results


def _parse_plot_ages(raw):
    """Convert plot_ages setting (list, string, int, or empty) to list or None."""
    if isinstance(raw, str):
        return [int(a) for a in raw.split(',') if a.strip()]
    if isinstance(raw, (int, float)):
        return [int(raw)]
    if isinstance(raw, list) and raw:
        return [int(a) for a in raw]
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Run durables2_0 model')
    add_override_args(parser)
    add_method_args(parser, choices=['FUES', 'NEGM'], default='FUES')
    add_output_args(parser, default_dir='results/durables2_0')
    add_simulation_args(parser)
    add_sweep_args(parser)

    parser.add_argument(
        '--compare', action='store_true', default=False,
        help='Run both FUES and NEGM and print comparison table',
    )
    parser.add_argument(
        '--sweep', action='store_true', default=False,
        help='Run parameter sweep instead of single-point',
    )

    args = parser.parse_args()
    calib_ov, config_ov = build_overrides(
        args, settings_path=SETTINGS_PATH)

    eff = load_effective_settings(SETTINGS_PATH, config_ov)
    plot_ages = _parse_plot_ages(eff.get('plot_ages', []))
    store_cntn = bool(eff.get('store_cntn', 0))
    use_empirical_init = eff.get('init_method', 'standard') == 'empirical'

    run_dir = make_run_dir(args.output_dir, tag=args.run_tag)
    print(f'Output directory: {run_dir}')

    if args.compare:
        run_comparison(
            calib_overrides=calib_ov or None,
            config_overrides=config_ov or None,
            output_dir=run_dir,
            plot_ages=plot_ages,
            store_cntn=store_cntn,
            sim=args.simulate,
            n_sim=args.n_sim,
            seed=args.seed,
            use_empirical_init=use_empirical_init)
    elif args.sweep:
        sweep_grid = parse_sweep_args(args)
        grid_sizes = [p['grid_size'] for p in sweep_grid] if sweep_grid else None
        run_sweep(
            method=args.method,
            grid_sizes=grid_sizes,
            n_reps=args.sweep_runs,
            calib_overrides=calib_ov,
            config_overrides=config_ov,
            output_dir=run_dir)
    else:
        run_single(
            method=args.method,
            calib_overrides=calib_ov or None,
            config_overrides=config_ov or None,
            output_dir=run_dir,
            plot_ages=plot_ages,
            store_cntn=store_cntn,
            sim=args.simulate,
            n_sim=args.n_sim,
            seed=args.seed,
            use_empirical_init=use_empirical_init)


if __name__ == '__main__':
    main()
