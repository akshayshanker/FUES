"""Retirement model example - FUES vs DC-EGM comparison.

Callers should import directly from submodules:
    from examples.retirement.solve import solve_nest
    from examples.retirement.benchmark import test_Timings
    from examples.retirement.outputs import euler, get_policy
    from examples.retirement.outputs import plot_egrids, plot_cons_pol, plot_dcegm_cf

The package used to re-export these names at the top level, but that forced
`import examples.retirement` to eagerly pull seaborn + HARK via outputs/plots.py.
Keeping this module empty lets the sweep path stay free of plotting deps.
"""
