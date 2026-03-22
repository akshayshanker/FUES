"""Run the durables2_0 DDSL pipeline and plot policies."""

import os
import numpy as np
from .solve import solve
from .outputs import (
    plot_policies, plot_grids, plot_lifecycle, get_timing, derive_savings,
)
from .simulate import euler_errors


def run(syntax_dir='examples/durables2_0/syntax',
        output_dir='results/durables2_0/plots',
        verbose=True, store_cntn=False, plot_ages=None,
        calib_overrides=None, config_overrides=None,
        sim=False, N_sim=10000, seed=42,
        use_empirical_init=False):
    """Solve and plot.

    Parameters
    ----------
    syntax_dir : str
    output_dir : str
        Default follows readme_examples.md convention.
    verbose : bool
    store_cntn : bool
        Store cntn perch (EGM grids) and produce EGM plots.
    plot_ages : list of int, optional
        Ages to plot. Each age gets its own subfolder.
        Default: third-to-last age only.
    calib_overrides : dict, optional
        Calibration overrides (e.g. ``{'t0': 20}``).
    config_overrides : dict, optional
        Settings overrides.
    """
    os.environ['FUES_RETURN_GRIDS'] = '1' if store_cntn else '0'

    cfg = dict(config_overrides or {})
    if store_cntn:
        cfg['store_cntn'] = 1

    nest, cp, grids, callables = solve(
        syntax_dir, verbose=verbose,
        calib_overrides=calib_overrides,
        config_overrides=cfg if cfg else None)
    print(f'{len(nest["solutions"])} periods solved')

    timing = get_timing(nest)
    print(f'Mean timing — solve: {timing["solve_time"]*1000:.1f}ms, '
          f'keeper: {timing["keeper_ms"]:.1f}ms, '
          f'adj: {timing["adj_ms"]:.1f}ms, '
          f'discrete: {timing["discrete_ms"]:.1f}ms')

    all_t = sorted(s['t'] for s in nest['solutions'])

    if plot_ages is None:
        plot_ages = [all_t[-3] if len(all_t) >= 3 else all_t[-1]]

    savings = derive_savings(nest, grids, cp.tau)

    for age in plot_ages:
        if age not in all_t:
            print(f'Age {age} not in solution (range {all_t[0]}–{all_t[-1]}), skipping.')
            continue
        plot_policies(nest, grids, savings, output_dir=output_dir, plot_t=age)
        if store_cntn:
            plot_grids(nest, grids, output_dir=output_dir, plot_t=age)

    if sim:
        euler, sim_data = euler_errors(
            nest, cp, grids, callables, N=N_sim, seed=seed,
            use_empirical_init=use_empirical_init)

        valid = euler[~np.isnan(euler)]
        d = sim_data['discrete']
        keep_mask = (d == 0) & ~np.isnan(euler)
        adj_mask = (d == 1) & ~np.isnan(euler)

        print(f"\nEuler errors (log10):")
        if len(valid) == 0:
            print("  WARNING: zero finite Euler observations — "
                  "check simulation diagnostics")
        else:
            print(f"  Combined: mean={np.mean(valid):.4f}, "
                  f"median={np.median(valid):.4f}")
            if np.any(keep_mask):
                print(f"  Keeper:   mean={np.mean(euler[keep_mask]):.4f}")
            if np.any(adj_mask):
                print(f"  Adjuster: mean={np.mean(euler[adj_mask]):.4f}")
        print(f"  Adj rate: {np.mean(d[d >= 0])*100:.1f}%")
        print(f"  Agents:   {N_sim}, periods: {cp.T - cp.t0}")

        # Utility stats
        if 'npv_utility' in sim_data:
            npv = sim_data['npv_utility']
            print(f"  NPV utility: mean={np.mean(npv):.4f}, "
                  f"std={np.std(npv):.4f}")

        # Lifecycle plots
        plot_lifecycle(sim_data, euler, cp, output_dir=output_dir)

    return nest, cp, grids, callables


if __name__ == '__main__':
    import sys

    store = '--grids' in sys.argv
    simulate = '--simulate' in sys.argv
    empirical = '--empirical-init' in sys.argv

    ages = None
    for i, arg in enumerate(sys.argv):
        if arg == '--ages' and i + 1 < len(sys.argv):
            ages = [int(a) for a in sys.argv[i + 1].split(',')]

    out_dir = 'results/durables2_0/plots'
    calib_ov = {}
    for i, arg in enumerate(sys.argv):
        if arg == '--output-dir' and i + 1 < len(sys.argv):
            out_dir = sys.argv[i + 1]
        if arg == '--t0' and i + 1 < len(sys.argv):
            calib_ov['t0'] = int(sys.argv[i + 1])

    run(store_cntn=store, plot_ages=ages, output_dir=out_dir,
        calib_overrides=calib_ov or None, sim=simulate,
        use_empirical_init=empirical)
