"""Notebook utilities for durables.

Thin helpers that keep notebook cells clean: stdout filtering,
CE transforms, solve wrappers with progress display.
"""

import sys
import numpy as np


class FilteredStdout:
    """Proxy that suppresses lines matching a pattern (e.g. numba debug).

    Usage::

        real = sys.stdout
        sys.stdout = FilteredStdout(real, 'SCAN DEBUG')
        try:
            ...  # numba prints suppressed
        finally:
            sys.stdout = real
    """

    def __init__(self, real, pattern='SCAN DEBUG'):
        self._real = real
        self._pattern = pattern

    def write(self, s):
        if self._pattern not in s:
            self._real.write(s)

    def flush(self):
        self._real.flush()


def ce_utility(npv_array, rho):
    """Certainty-equivalent from NPV utility array.

    Parameters
    ----------
    npv_array : array-like
        Per-agent discounted NPV utility (from ``sim_data['npv_utility']``).
    rho : float
        Risk aversion parameter.

    Returns
    -------
    float
        CE of the mean NPV: ``((1-rho) * mean(V))^(1/(1-rho))``.
    """
    vals = np.asarray(npv_array)
    mean_v = np.mean(vals[np.isfinite(vals)])
    if abs(rho - 1.0) < 1e-8:
        return np.exp(mean_v)
    inner = (1.0 - rho) * mean_v
    return inner ** (1.0 / (1.0 - rho)) if inner > 0 else np.nan




def print_solve_summary(results):
    """Print a clean calibration + timing summary from solve results."""
    labels = {'FUES': 'EGM(FUES)', 'NEGM': 'NEGM(FUES)'}
    _st = results['FUES']['nest']['periods'][0]['stages']['keeper_cons']
    _cal, _sett = _st.calibration, _st.settings
    n_periods = len(results['FUES']['nest']['solutions'])

    _skip = {'t0', 'T', 'b', 'normalisation', 'phi_w', 'sigma_w', 'N_wage',
             'return_grids', 'z_vals', 'Pi', 'age'}
    _params = ', '.join(f'{k}={v}' for k, v in _cal.items()
                        if k not in _skip and not k.startswith('_'))

    print(f"\n  {n_periods} periods, ages {int(_cal['t0'])}\u2013{_sett['T']}")
    print(f"  {_params}")
    print(f"  n_a={_sett['n_a']}, n_h={_sett['n_h']}, n_w={_sett['n_w']}, "
          f"N_wage={_sett['N_wage']}")
    print()
    print(f"  {'Method':12s}  {'Keeper':>10s}  {'Adjuster':>10s}  {'Tenure':>10s}  {'Total':>8s}")
    print(f"  {'':12s}  {'(ms/period)':>10s}  {'(ms/period)':>10s}  {'(ms/period)':>10s}  {'(sec)':>8s}")
    print(f"  {'\u2500'*60}")
    for method in ['FUES', 'NEGM']:
        t = results[method]['timing']
        print(f"  {labels[method]:12s}  {t['keeper_ms']:10.0f}  {t['adj_ms']:10.0f}  "
              f"{t['discrete_ms']:10.0f}  {results[method]['elapsed']:8.0f}")


def build_comparison_row(method, results, euler_results):
    """Build one row dict for ``generate_vertical_comparison``.

    Extracts timing, Euler stats, CE utility, and simulation aggregates.
    """
    labels = {'FUES': 'EGM(FUES)', 'NEGM': 'NEGM(FUES)'}
    t = results[method]['timing']
    row = {
        'Method': labels[method],
        'Keeper (ms)': t['keeper_ms'],
        'Adj (ms)': t['adj_ms'],
        'Total (ms)': t['keeper_ms'] + t['adj_ms'] + t['discrete_ms'],
    }
    ec = euler_results[method]['stats_c']
    eh = euler_results[method]['stats_h']
    if 'combined' in ec:
        row['Euler c (keeper)'] = ec['keeper']['mean']
        row['Euler c (adj)'] = ec['adjuster']['mean']
        row['Euler c (all)'] = ec['combined']['mean']
    if 'combined' in eh:
        row['Euler h (adj)'] = eh['adjuster']['mean']
    sd = euler_results[method]['sim_data']
    d = sd['discrete']
    row['Adj Rate'] = np.mean(d[d >= 0]) * 100

    # CE utility + mean aggregates (denormalised)
    _st = results[method]['nest']['periods'][0]['stages']['keeper_cons']
    rho = float(_st.calibration.get('gamma_c', _st.calibration.get('rho', 2.0)))
    norm = 1.0 / float(_st.settings['normalisation'])
    row['CE Utility'] = ce_utility(sd['npv_utility'], rho) * norm
    row['Mean Consumption'] = np.nanmean(sd['c']) * norm
    row['Mean Financial Assets'] = np.nanmean(sd['a']) * norm
    row['Mean Housing'] = np.nanmean(sd['h']) * norm
    return row
