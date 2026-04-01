"""Run the durables DDSL pipeline: single-point, method comparison, or sweep.

Usage:
    # Single-point (default)
    python -m examples.durables.run
    python -m examples.durables.run --method NEGM --simulate

    # Method comparison (FUES vs NEGM)
    python -m examples.durables.run --compare FUES NEGM

    # Parameter sweep
    python -m examples.durables.run --sweep --sweep-grids 100,200,300

    # Overrides
    python -m examples.durables.run --calib-override t0=50
    python -m examples.durables.run --setting-override plot_ages=50,55
"""

import itertools
import math
import os
from dataclasses import replace
from pathlib import Path
import numpy as np
import yaml

from kikku.dynx.methods import parse_method_override_str
from kikku.run import parse_run
from kikku.run.sweep import param_grid, sweep
from kikku.run.metrics import format_table, write_table

from .solve import solve, read_scheme_method
from .outputs import (
    plot_policies, plot_grids, plot_lifecycle, get_timing, derive_savings,
    consumption_deviation, compute_euler_stats, print_euler_stats,
    generate_comparison_table,
    generate_sweep_table,
    write_euler_detail,
)
from .horses.simulate import (
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


def build_sweep_grid(run):
    """Parse ``run.sweep_params`` into Cartesian sweep points (model-specific).

    Each point supplies calib/settings/method overrides for one ``solve`` call.
    """
    axes = {}
    for spec in run.sweep_params:
        if '=' not in spec:
            raise ValueError(
                f"Invalid --sweep-params entry (expected key=val1,val2,...): {spec!r}")
        key, rest = spec.split('=', 1)
        key = key.strip()
        vals = [
            yaml.safe_load(p.strip())
            for p in rest.split(',')
            if p.strip()
        ]
        if not vals:
            raise ValueError(f"Empty value list for sweep axis {key!r}")
        axes[key] = vals

    if run.sweep_grids is not None and 'n_a' not in axes:
        axes['n_a'] = list(run.sweep_grids)

    calib_keys = set(run.calib.keys())
    settings_keys = set(run.settings.keys())
    roles = {}
    for key in axes:
        if key == 'method':
            roles[key] = 'method'
        elif key in calib_keys:
            roles[key] = 'calib'
        elif key in settings_keys:
            roles[key] = 'setting'
        elif '.' in key:
            roles[key] = 'method_override'
        else:
            raise ValueError(
                f"Unknown sweep axis {key!r}: not 'method', not in "
                f"calibration/settings keys, and not a dotted method path.")

    if 'method' in axes:
        for m in axes['method']:
            if m not in run.methods:
                raise ValueError(
                    f"Invalid sweep method {m!r}; allowed: {list(run.methods)}")

    key_order = sorted(axes.keys())
    if 'method' in key_order:
        key_order = ['method'] + [k for k in key_order if k != 'method']

    points = []
    for combo in itertools.product(*(axes[k] for k in key_order)):
        params = dict(zip(key_order, combo))
        calib_overrides = {}
        setting_overrides = {}
        method = None
        mo_point = {}
        for k, v in params.items():
            role = roles[k]
            if role == 'method':
                method = v
            elif role == 'calib':
                calib_overrides[k] = v
            elif role == 'setting':
                setting_overrides[k] = v
            else:
                triple, tag = parse_method_override_str(f"{k}={v}")
                mo_point[triple] = str(tag).strip()
        label = ','.join(f"{k}={v}" for k, v in params.items())
        points.append({
            'calib_overrides': calib_overrides,
            'setting_overrides': setting_overrides,
            'method': method,
            'method_overrides': mo_point if mo_point else None,
            'label': label,
            'params': dict(params),
        })
    return points


def _effective_setting(key, point, base_config):
    if key in point['params']:
        return point['params'][key]
    return base_config.get(key)


def _effective_calib(key, point, base_calib):
    if key in point['params']:
        return point['params'][key]
    return base_calib.get(key)


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


def _get_mpi_comm():
    """Return MPI communicator if available and size > 1, else None."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            return comm
    except ImportError:
        pass
    return None


def _run_sweep_params_path(run, base_calib, base_config):
    """Multi-axis sweep using kikku's sweep() with MPI support.

    Builds the Cartesian product of sweep axes, wraps each point
    into a solve_fn compatible with kikku.run.sweep.sweep(), and
    lets sweep() handle MPI distribution + best-of-n timing.
    """
    comm = _get_mpi_comm()
    points = build_sweep_grid(run)

    syntax = str(run.syntax_dir)

    # Pre-compute key classification for solve_fn
    calib_keys = set(base_calib.keys())
    settings_keys = set(base_config.keys())

    def solve_fn(params):
        """Solve one sweep point. params is flat: {method: .., n_a: .., tau: ..}."""
        method = params.get('method', run.method)
        mo = dict(run.method_overrides or {})
        calib_use = dict(base_calib)
        cfg_use = dict(base_config)
        for k, v in params.items():
            if k == 'method':
                continue
            elif k in calib_keys:
                calib_use[k] = v
            elif k in settings_keys:
                cfg_use[k] = v

        nest, grids = solve(
            syntax, method=method,
            method_overrides=mo,
            calib_overrides=calib_use,
            setting_overrides=cfg_use,
            verbose=False)

        adj0 = nest["periods"][0]["stages"]["adjuster_cons"]
        method_label = method or read_scheme_method(adj0, 'upper_envelope')
        timing = get_timing(nest)

        result = {
            'nest': nest, 'grids': grids,
            'method_label': method_label,
            'timing': timing,
            'calib_use': calib_use,
        }

        if run.simulate:
            st0 = nest["periods"][0]["stages"]["keeper_cons"]
            use_emp = st0.calibration.get("init_method", "standard") == "empirical"
            sim_data = simulate_lifecycle(
                nest, grids,
                N=run.n_sim, seed=run.seed,
                use_empirical_init=use_emp)
            euler_c = evaluate_euler_c(sim_data, nest, grids)
            euler_h = evaluate_euler_h(sim_data, nest, grids)
            d = sim_data['discrete']
            result['ec_stats'] = compute_euler_stats(euler_c, d)
            result['eh_stats'] = compute_euler_stats(euler_h, d)
            result['adj_rate'] = np.mean(d[d >= 0]) * 100

        return result

    # Build metric extraction functions
    metric_fns = {
        'solve_ms': lambda r: r['timing']['solve_time'] * 1000,
        'keeper_ms': lambda r: r['timing']['keeper_ms'],
        'adj_ms': lambda r: r['timing']['adj_ms'],
        'method': lambda r: r['method_label'],
    }
    if run.simulate:
        metric_fns.update({
            'euler_c_mean': lambda r: r['ec_stats'].get('combined', {}).get('mean', np.nan),
            'euler_c_keeper': lambda r: r['ec_stats'].get('keeper', {}).get('mean', np.nan),
            'euler_c_adjuster': lambda r: r['ec_stats'].get('adjuster', {}).get('mean', np.nan),
            'euler_h_mean': lambda r: r['eh_stats'].get('combined', {}).get('mean', np.nan),
            'adj_rate': lambda r: r['adj_rate'],
        })

    # Pass flat param dicts to kikku's sweep (MPI-aware)
    flat_grid = [p['params'] for p in points]

    raw_results = sweep(
        solve_fn, flat_grid, metric_fns,
        n_reps=run.sweep_runs,
        warmup=run.warmup,
        best='min',
        on_error='skip',
        comm=comm,
    )

    # Build summaries for LaTeX table (only on rank 0 after gather)
    results = []
    summaries = []
    for row in raw_results:
        n_a_eff = row.get('n_a', base_config.get('n_a', 0))
        tau_eff = row.get('tau', base_calib.get('tau', 0.0))

        results.append(row)

        cal_params = {}
        for k in ('beta', 'gamma_c', 'gamma_h', 'alpha', 'delta',
                  'R', 'R_H', 'phi_w', 'sigma_w'):
            if k in base_calib:
                cal_params[k] = base_calib[k]
        if 'R' in cal_params:
            cal_params['r'] = cal_params.pop('R')
        if 'R_H' in cal_params:
            cal_params['r_H'] = cal_params.pop('R_H')
        cal_params['N_sim'] = run.n_sim

        summaries.append({
            'Grid_Size': int(n_a_eff) if n_a_eff is not None else 0,
            'Tau': float(tau_eff) if tau_eff is not None else 0.0,
            'Method': row.get('method', ''),
            'Avg_Keeper_ms': row.get('keeper_ms', 0.0),
            'Avg_Adj_ms': row.get('adj_ms', 0.0),
            'Euler_Combined': row.get('euler_c_mean', np.nan),
            'Euler_Keeper': row.get('euler_c_keeper', np.nan),
            'Euler_Adjuster': row.get('euler_c_adjuster', np.nan),
            'Euler_H_Adjuster': row.get('euler_h_mean', np.nan),
            **cal_params,
        })

    return results, summaries


def run_sweep(run):
    """Parameter sweep over grid sizes with timing and optional Euler metrics.

    With ``run.simulate``, a single ``trial_fn`` runs solve + simulate +
    post-hoc Euler once per grid point (and per rep); metrics are plain
    key lookups on the returned dict.

    Returns list[dict] — flat results from sweep().
    """
    base_calib = dict(run.calib or {})
    base_config = _solver_config(run)

    if run.sweep_params:
        results, summaries = _run_sweep_params_path(
            run, base_calib, base_config)

        # Only rank 0 has gathered results; non-root ranks get []
        if not results:
            return results

        omit_md = {'euler_c_keeper', 'euler_c_adjuster'}
        if run.simulate:
            metric_tail = [
                'n_a', 'method', 'solve_ms', 'keeper_ms', 'adj_ms',
                'euler_c_mean', 'euler_h_mean', 'adj_rate',
            ]
        else:
            metric_tail = [
                'n_a', 'method', 'solve_ms', 'keeper_ms', 'adj_ms',
            ]
        row0 = results[0]
        param_cols = sorted(
            k for k in row0 if k not in set(metric_tail) | omit_md)
        cols = [c for c in param_cols + metric_tail if c in row0]
        print('\n' + format_table(results, cols))
        output_dir = str(run.output_dir)
        tdir = os.path.join(output_dir, 'tables')
        os.makedirs(tdir, exist_ok=True)
        write_table(os.path.join(tdir, 'sweep.md'), results, cols)
        tex = generate_sweep_table(
            summaries, fmt='tex',
            caption='Durables Model: Per-Period Timing and Accuracy')
        with open(os.path.join(tdir, 'sweep.tex'), 'w') as f:
            f.write(tex)
        return results

    grid_sizes = run.sweep_grids or [100, 200, 300]
    grid = param_grid(n_a=grid_sizes)

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
        name='durables',
        syntax='examples/durables/mod/separable',
        methods=['FUES', 'NEGM'],
        modes=['compare', 'sweep', 'simulate'],
        output='results/durables',
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
