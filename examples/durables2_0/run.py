"""Run the durables2_0 DDSL pipeline: single-point, method comparison, or sweep.

Usage:
    # Single-point (default)
    python -m examples.durables2_0.run
    python -m examples.durables2_0.run --method NEGM --simulate

    # Method comparison (FUES vs NEGM)
    python -m examples.durables2_0.run --compare FUES NEGM

    # Parameter sweep
    python -m examples.durables2_0.run --sweep --sweep-grids 100,200,300

    # Overrides
    python -m examples.durables2_0.run --calib-override t0=50
    python -m examples.durables2_0.run --setting-override plot_ages=50,55
"""

import os
from dataclasses import replace
from pathlib import Path
import numpy as np

from kikku.run import parse_run
from kikku.run.sweep import param_grid, sweep
from kikku.run.metrics import format_table, write_table

from .solve import solve, read_scheme_method
from .outputs import (
    plot_policies, plot_grids, plot_lifecycle, get_timing, derive_savings,
    consumption_deviation, compute_euler_stats, print_euler_stats,
    generate_comparison_table,
    write_euler_detail,
)
from .simulate import (
    simulate_lifecycle,
    evaluate_euler_c,
    evaluate_euler_h,
)


def _solver_config(run):
    """Merge settings tier + config tier for ``load_syntax`` (settings.yaml overlay)."""
    return {**dict(run.settings), **dict(run.config)}


def _parse_plot_ages(raw):
    """Convert plot_ages setting (list, string, int, or empty) to list or None."""
    if isinstance(raw, str):
        return [int(a) for a in raw.split(',') if a.strip()]
    if isinstance(raw, (int, float)):
        return [int(raw)]
    if isinstance(raw, list) and raw:
        return [int(a) for a in raw]
    return None


def run_single(run):
    """Solve and optionally simulate + plot.

    Returns ``{'nest', 'grids', 'results_row', 'euler_stats'}``.
    """
    store_cntn = bool(run.settings.get('store_cntn', 0))

    cfg = _solver_config(run)
    nest, grids = solve(
        str(run.syntax_dir), method=run.method,
        method_overrides=run.method_overrides,
        verbose=False,
        calib_overrides=run.calib or None,
        setting_overrides=cfg if cfg else None)
    adj0 = nest["periods"][0]["stages"]["adjuster_cons"]
    method_label = run.method or read_scheme_method(adj0, 'upper_envelope')
    print(f'{len(nest["solutions"])} periods solved')

    stage0 = nest["periods"][0]["stages"]["keeper_cons"]
    # Post-solve: runner keys are merged onto stage.calibration with economic params.
    plot_ages = _parse_plot_ages(stage0.calibration.get("plot_ages", []))
    use_empirical_init = (
        stage0.calibration.get("init_method", "standard") == "empirical")

    timing = get_timing(nest)
    print(f'Mean timing — keeper: {timing["keeper_ms"]:.1f}ms, '
          f'adj: {timing["adj_ms"]:.1f}ms')

    results_row = {
        'Method': method_label,
        'Keeper (ms)': timing['keeper_ms'],
        'Adj (ms)': timing['adj_ms'],
        'Total (ms)': timing['keeper_ms'] + timing['adj_ms'],
    }
    euler_stats = None

    base_dir = str(run.output_dir)
    plots_dir = os.path.join(base_dir, 'plots')
    tables_dir = os.path.join(base_dir, 'tables')

    all_t = sorted(s['t'] for s in nest['solutions'])
    if plot_ages is None:
        plot_ages = [all_t[-3] if len(all_t) >= 3 else all_t[-1]]

    tau = float(stage0.calibration["tau"])
    savings = derive_savings(nest, grids, tau)

    for age in plot_ages:
        if age not in all_t:
            print(f'Age {age} not in solution '
                  f'(range {all_t[0]}–{all_t[-1]}), skipping.')
            continue
        plot_policies(nest, grids, savings,
                      output_dir=plots_dir, plot_t=age)
        if store_cntn:
            plot_grids(nest, grids,
                       output_dir=plots_dir, plot_t=age)
    if plot_ages:
        print(f'Plots saved to {plots_dir}/')

    if run.simulate:
        sim_data = simulate_lifecycle(
            nest, grids,
            N=run.n_sim, seed=run.seed,
            use_empirical_init=use_empirical_init)

        euler_c = evaluate_euler_c(sim_data, nest, grids)
        euler_h = evaluate_euler_h(sim_data, nest, grids)
        d = sim_data['discrete']
        ec_stats = compute_euler_stats(euler_c, d)
        eh_stats = compute_euler_stats(euler_h, d)
        euler_stats = {'consumption': ec_stats, 'housing': eh_stats}

        print("Consumption Euler (c FOC):")
        print_euler_stats(ec_stats)
        print("\nHousing FOC (adjusters):")
        print_euler_stats(eh_stats)

        adj_rate = np.mean(d[d >= 0]) * 100

        if 'combined' in ec_stats:
            results_row['Euler c (keeper)'] = ec_stats['keeper']['mean']
            results_row['Euler c (adj)'] = ec_stats['adjuster']['mean']
            results_row['Euler c (all)'] = ec_stats['combined']['mean']
            results_row['Adj Rate'] = adj_rate
        if 'combined' in eh_stats:
            results_row['Euler h (keeper)'] = eh_stats['keeper']['mean']
            results_row['Euler h (adj)'] = eh_stats['adjuster']['mean']
            results_row['Euler h (all)'] = eh_stats['combined']['mean']

        if 'npv_utility' in sim_data:
            npv = sim_data['npv_utility']
            print(f"  NPV utility: mean={np.mean(npv):.4f}, "
                  f"std={np.std(npv):.4f}")

        plot_lifecycle(sim_data, euler_c, nest, output_dir=plots_dir)

    # Save single-run summary table
    os.makedirs(tables_dir, exist_ok=True)
    from kikku.run.metrics import format_table, write_table
    print('\n' + format_table([results_row],
          [k for k in results_row.keys()]))
    write_table(os.path.join(tables_dir, 'summary.md'),
                [results_row], [k for k in results_row.keys()])

    return {
        'nest': nest, 'grids': grids,
        'results_row': results_row, 'euler_stats': euler_stats,
    }


def _parse_compare_spec(spec):
    """Parse one --compare entry into (label, method, method_overrides).

    A bare name (e.g. 'FUES') expands via METHOD_SHORTCUT in solve().
    An override spec (e.g. 'keeper_cons.upper_envelope=MSS') is used directly.
    """
    from kikku.dynx.methods import parse_method_override_str

    if '=' in spec:
        key, tag = parse_method_override_str(spec)
        label = f"{key[0]}.{key[2]}={tag}"
        return label, None, {key: tag}
    else:
        return spec, spec, None


def run_comparison(run):
    """Compare methods, print and save comparison tables.

    Each ``--compare`` entry is either a bare method name (expanded via
    ``METHOD_SHORTCUT``) or a ``stage.scheme=TAG`` override spec.
    Produces one combined table with all entries.

    Returns ``{label: {nest, grids, ...}, ...}``.
    """
    specs = list(run.compare_methods)
    base_dir = str(run.output_dir)
    table_dir = os.path.join(base_dir, 'tables')

    all_results = {}
    rows = []
    euler_stats_by_label = {}

    for spec in specs:
        label, method, method_overrides = _parse_compare_spec(spec)

        print(f'\n{"="*60}')
        print(f'  {label}')
        print(f'{"="*60}')

        method_dir = os.path.join(base_dir, label)
        spec_run = replace(
            run,
            method=method,
            method_overrides=method_overrides or run.method_overrides,
            output_dir=Path(method_dir),
        )
        result = run_single(spec_run)
        result['results_row']['Method'] = label
        all_results[label] = result
        rows.append(result['results_row'])
        if result['euler_stats'] is not None:
            euler_stats_by_label[label] = result['euler_stats']

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

    if run.simulate and euler_stats_by_label:
        path = write_euler_detail(euler_stats_by_label, table_dir)
        print(f'Euler detail saved to {path}')
    elif not run.simulate:
        print('\n(Run with --simulate for Euler accuracy columns)')

    return all_results


def run_sweep(run):
    """Parameter sweep over grid sizes with timing and optional Euler metrics.

    With ``run.simulate``, a single ``trial_fn`` runs solve + simulate +
    post-hoc Euler once per grid point (and per rep); metrics are plain
    key lookups on the returned dict.

    Returns list[dict] — flat results from sweep().
    """
    grid_sizes = run.sweep_grids or [100, 200, 300]
    grid = param_grid(n_a=grid_sizes)

    base_calib = dict(run.calib or {})
    base_config = _solver_config(run)

    if run.simulate:
        def trial_fn(ov):
            cfg = {**base_config, 'n_a': ov['n_a']}
            nest, grids = solve(
                str(run.syntax_dir), method=run.method,
                method_overrides=run.method_overrides,
                calib_overrides=base_calib,
                setting_overrides=cfg,
                verbose=False)
            adj0 = nest["periods"][0]["stages"]["adjuster_cons"]
            method_label = run.method or read_scheme_method(adj0, 'upper_envelope')
            st0 = nest["periods"][0]["stages"]["keeper_cons"]
            use_emp = st0.calibration.get("init_method", "standard") == "empirical"
            sim_data = simulate_lifecycle(
                nest, grids,
                N=run.n_sim, seed=run.seed,
                use_empirical_init=use_emp)
            euler_c = evaluate_euler_c(
                sim_data, nest, grids)
            euler_h = evaluate_euler_h(
                sim_data, nest, grids)
            timing = get_timing(nest)
            d = sim_data['discrete']
            ec_stats = compute_euler_stats(euler_c, d)
            eh_stats = compute_euler_stats(euler_h, d)
            return {
                'method': method_label,
                'solve_ms': timing['solve_time'] * 1000,
                'keeper_ms': timing['keeper_ms'],
                'adj_ms': timing['adj_ms'],
                'euler_c_mean': ec_stats.get('combined', {}).get(
                    'mean', np.nan),
                'euler_h_mean': eh_stats.get('combined', {}).get(
                    'mean', np.nan),
                'adj_rate': np.mean(d[d >= 0]) * 100,
            }

        metric_keys = [
            'solve_ms', 'keeper_ms', 'adj_ms',
            'euler_c_mean', 'euler_h_mean', 'adj_rate',
        ]
        metric_fns = {
            k: (lambda k=k: (lambda r: r[k]))()
            for k in metric_keys
        }
        metric_fns['method'] = lambda r: r['method']
        results = sweep(
            trial_fn, grid, metric_fns,
            n_reps=run.sweep_runs, warmup=run.warmup,
            best='min')
        cols = [
            'n_a', 'method', 'solve_ms', 'keeper_ms', 'adj_ms',
            'euler_c_mean', 'euler_h_mean', 'adj_rate',
        ]
    else:
        def solve_fn(ov):
            cfg = {**base_config, 'n_a': ov['n_a']}
            nest, grids = solve(
                str(run.syntax_dir), method=run.method,
                method_overrides=run.method_overrides,
                calib_overrides=base_calib,
                setting_overrides=cfg,
                verbose=False)
            adj0 = nest["periods"][0]["stages"]["adjuster_cons"]
            method_label = run.method or read_scheme_method(adj0, 'upper_envelope')
            return {'nest': nest, 'method': method_label}

        metric_fns = {
            'solve_ms': lambda r: get_timing(r['nest'])['solve_time'] * 1000,
            'keeper_ms': lambda r: get_timing(r['nest'])['keeper_ms'],
            'adj_ms': lambda r: get_timing(r['nest'])['adj_ms'],
            'method': lambda r: r['method'],
        }
        results = sweep(
            solve_fn, grid, metric_fns,
            n_reps=run.sweep_runs, warmup=run.warmup,
            best='min')
        cols = ['n_a', 'method', 'solve_ms', 'keeper_ms', 'adj_ms']

    if results:
        print('\n' + format_table(results, cols))
        output_dir = str(run.output_dir)
        os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
        write_table(
            os.path.join(output_dir, 'tables', 'sweep.md'),
            results, cols)

    return results


def main():
    run = parse_run(
        name='durables2_0',
        syntax='examples/durables2_0/syntax',
        methods=['FUES', 'NEGM'],
        modes=['compare', 'sweep', 'simulate'],
        output='results/durables2_0',
    )

    print(f'Output directory: {run.output_dir}')

    if run.mode == 'compare':
        run_comparison(run)
    elif run.mode == 'sweep':
        run_sweep(run)
    else:
        run_single(run)


if __name__ == '__main__':
    main()
