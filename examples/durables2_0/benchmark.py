"""Side-by-side: durables2_0 (DDSL pipeline) vs original.

Runs ``solve_nest`` (YAML-driven) and ``solveEGM`` (monolithic)
and verifies array-level equivalence.
"""

import numpy as np
import time
import sys

# Patch UCGrid before any durables import
import interpolation.splines as _splines
_orig = _splines.UCGrid
def _patch(*args):
    return _orig(*[(float(a), float(b), int(n))
                   for a, b, n in args])
_splines.UCGrid = _patch

from examples.durables.durables import ConsumerProblem
from examples.durables.durables_plot import solveEGM

from .solve import solve


def run_comparison(syntax_dir, calib_overrides=None,
                   config_overrides=None, verbose=True):
    """Run both pipelines and compare all arrays.

    Parameters
    ----------
    syntax_dir : str
        Path to durables2_0 syntax directory.
    calib_overrides, config_overrides : dict, optional
    verbose : bool

    Returns
    -------
    dict
        ``{'max_diff', 'all_match', ...}``
    """
    print("=" * 60)
    print("Durables2.0 DDSL pipeline vs Original")
    print("=" * 60)

    # --- New pipeline (DDSL) ---
    print("\n--- New (solve_nest from YAML) ---")
    t0 = time.time()
    nest, model, stage_ops, waves = solve(
        syntax_dir,
        calib_overrides=calib_overrides,
        config_overrides=config_overrides,
        verbose=False,
    )
    solutions = nest["solutions"]
    print(f"  Time: {time.time() - t0:.2f}s")
    print(f"  Waves: {waves}")
    print(f"  Periods solved: {len(solutions)}")

    # --- Original pipeline ---
    print("\n--- Original (solveEGM) ---")
    cp = model.cp  # reuse same ConsumerProblem
    t0 = time.time()
    old = solveEGM(cp, verbose=False)
    print(f"  Time: {time.time() - t0:.2f}s")

    # --- Compare ---
    print("\n--- Comparison ---")
    max_diffs = {}

    for sol in solutions:
        t = sol['t']
        if t not in old:
            continue
        o = old[t]
        ad = sol['tenure']
        kp = sol['keeper_cons']
        aj = sol['adjuster_cons']

        # Compare keeper/adjuster policies (C, A, H)
        for name, new_arr, old_key in [
            ('C_keep', kp['C'], 'Ckeeper'),
            ('A_keep', kp['A'], 'Akeeper'),
            ('A_adj', aj['A'], 'Aadj'),
            ('C_adj', aj['C'], 'Cadj'),
            ('H_adj', aj['H'], 'Hadj'),
        ]:
            if old_key in o:
                d = np.max(np.abs(new_arr - o[old_key]))
                max_diffs[f'{name}[t={t}]'] = d

    overall = max(max_diffs.values()) if max_diffs else 0.0
    ok = overall < 1e-12

    sorted_d = sorted(max_diffs.items(), key=lambda x: -x[1])
    print(f"\n  Max difference: {overall:.2e}")
    print(f"  Match (< 1e-12): {ok}")
    if not ok and verbose:
        print("\n  Worst 10:")
        for name, d in sorted_d[:10]:
            print(f"    {name}: {d:.2e}")

    return {'max_diff': overall, 'all_match': ok}


if __name__ == '__main__':
    result = run_comparison(
        'examples/durables2_0/syntax',
    )
    sys.exit(0 if result['all_match'] else 1)
