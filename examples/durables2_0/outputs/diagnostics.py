"""Nest accessors and solution diagnostics for durables2_0."""

import numpy as np


def derive_savings(nest, grids, tau):
    """Derive savings from budget constraint for all periods.

    Keeper: a_nxt = w - c (keeper domain is cash-on-hand).
    Adjuster: a_nxt = w - c - (1+tau)*h_choice.

    Returns
    -------
    dict
        ``{t: {'keeper': (n_z, n_a, n_h), 'adjuster': (n_z, n_w)}}``
    """
    a_grid = grids['a']
    we_grid = grids['we']
    tau_adj = 1 + tau
    result = {}
    for sol in nest['solutions']:
        t = sol['t']
        c_keep = sol['keeper_cons']['dcsn']['c']
        a_keep = a_grid[np.newaxis, :, np.newaxis] - c_keep

        c_adj = sol['adjuster_cons']['dcsn']['c']
        h_adj = sol['adjuster_cons']['dcsn']['h_choice']
        a_adj = we_grid[np.newaxis, :] - c_adj - tau_adj * h_adj

        result[t] = {'keeper': a_keep, 'adjuster': a_adj}
    return result


def get_policy(nest, key, stage='keeper_cons', perch='dcsn'):
    """Extract a policy array across all solved periods.

    Parameters
    ----------
    nest : dict
        Solved nest from ``solve()``.
    key : str
        Field name within the stage/perch dict
        (e.g. ``"c"``, ``"V"``, ``"a"``, ``"h"``, ``"adj"``).
    stage : str
        Stage name: ``'keeper_cons'``, ``'adjuster_cons'``, or ``'tenure'``.
    perch : str
        Perch within the stage (default ``'dcsn'``).

    Returns
    -------
    dict
        ``{t: ndarray}`` mapping calendar age to the policy array.
        Shape depends on stage (keeper: ``(n_z, n_a, n_h)``,
        adjuster: ``(n_z, n_we)``).
    """
    return {
        sol['t']: sol[stage][perch][key]
        for sol in nest['solutions']
    }


def get_timing(nest, warmup=3):
    """Mean timing across periods, skipping initial warmup.

    Parameters
    ----------
    nest : dict
        Solved nest.
    warmup : int
        Number of initial backward periods to skip (JIT warmup).

    Returns
    -------
    dict
        ``{'solve_time', 'keeper_ms', 'adj_ms', 'discrete_ms'}``
        as mean values in the appropriate units.
    """
    sols = [s for s in nest['solutions'] if s['h'] > warmup]
    if not sols:
        sols = nest['solutions']

    def _mean(key):
        vals = [s[key] for s in sols if key in s]
        return np.mean(vals) if vals else 0.0

    return {
        'solve_time': _mean('solve_time'),
        'keeper_ms': _mean('keeper_ms'),
        'adj_ms': _mean('adj_ms'),
        'discrete_ms': _mean('discrete_ms'),
    }


def get_solution_at_age(nest, t):
    """Get solution dict for calendar age *t*.

    Parameters
    ----------
    nest : dict
        Solved nest.
    t : int
        Calendar age.

    Returns
    -------
    dict or None
        Solution dict for age *t*, or None if not found.
    """
    for sol in nest['solutions']:
        if sol['t'] == t:
            return sol
    return None


def compute_euler_stats(euler, discrete=None):
    """Summary statistics for Euler errors.

    Parameters
    ----------
    euler : ndarray (T, N)
        Log10 relative Euler errors.
    discrete : ndarray (T, N), optional
        Discrete choice array (1 = adjuster, 0 = keeper).
        If provided, returns separate stats for each type.

    Returns
    -------
    dict
        If ``discrete`` is None: flat stats dict.
        Otherwise: ``{'combined', 'adjuster', 'keeper', 'pct_adjuster',
        'n_adjuster', 'n_keeper'}``.
    """
    def _stats(errors):
        valid = errors[~np.isnan(errors)]
        if len(valid) == 0:
            return {
                'mean': np.nan, 'median': np.nan, 'std': np.nan,
                'p5': np.nan, 'p95': np.nan,
                'frac_above_neg3': np.nan, 'frac_above_neg4': np.nan,
                'n_obs': 0,
            }
        return {
            'mean': np.mean(valid),
            'median': np.median(valid),
            'std': np.std(valid),
            'p5': np.percentile(valid, 5),
            'p95': np.percentile(valid, 95),
            'frac_above_neg3': np.mean(valid > -3),
            'frac_above_neg4': np.mean(valid > -4),
            'n_obs': len(valid),
        }

    if discrete is None:
        return _stats(euler)

    valid_mask = ~np.isnan(euler)
    adj_mask = (discrete == 1) & valid_mask
    keep_mask = (discrete == 0) & valid_mask

    n_valid = np.sum(valid_mask)
    n_adj = np.sum(adj_mask)
    n_keep = np.sum(keep_mask)

    return {
        'combined': _stats(euler),
        'adjuster': _stats(euler[adj_mask]),
        'keeper': _stats(euler[keep_mask]),
        'pct_adjuster': (n_adj / n_valid * 100) if n_valid > 0 else np.nan,
        'n_adjuster': int(n_adj),
        'n_keeper': int(n_keep),
    }


def print_euler_stats(stats):
    """Print Euler error statistics in formatted table.

    Parameters
    ----------
    stats : dict
        From :func:`compute_euler_stats`.
    """
    if 'combined' not in stats:
        print(f"Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}, "
              f"Std: {stats['std']:.4f}")
        print(f"P5: {stats['p5']:.4f}, P95: {stats['p95']:.4f}")
        print(f"Frac > 10^-3: {stats['frac_above_neg3']:.4f}, "
              f"Frac > 10^-4: {stats['frac_above_neg4']:.4f}")
        return

    print(f"\nEuler Error Statistics (log10 scale)")
    print("=" * 60)
    print(f"Adjustment rate: {stats['pct_adjuster']:.2f}% "
          f"({stats['n_adjuster']} adj, {stats['n_keeper']} keep)")
    print("-" * 60)
    print(f"{'':12} {'Combined':>12} {'Adjuster':>12} {'Keeper':>12}")
    print("-" * 60)

    for key in ['mean', 'median', 'std', 'p5', 'p95']:
        print(f"{key:12} {stats['combined'][key]:>12.4f} "
              f"{stats['adjuster'][key]:>12.4f} "
              f"{stats['keeper'][key]:>12.4f}")

    print("-" * 60)
    print(f"{'Frac>10^-3':12} {stats['combined']['frac_above_neg3']:>12.4f} "
          f"{stats['adjuster']['frac_above_neg3']:>12.4f} "
          f"{stats['keeper']['frac_above_neg3']:>12.4f}")
    print(f"{'Frac>10^-4':12} {stats['combined']['frac_above_neg4']:>12.4f} "
          f"{stats['adjuster']['frac_above_neg4']:>12.4f} "
          f"{stats['keeper']['frac_above_neg4']:>12.4f}")
    print(f"{'N obs':12} {stats['combined']['n_obs']:>12} "
          f"{stats['adjuster']['n_obs']:>12} "
          f"{stats['keeper']['n_obs']:>12}")


def consumption_deviation(nest, grids, nest_true, grids_true):
    """Mean log10 deviation of keeper consumption vs a reference solution.

    Compares keeper consumption on the tested solution's asset grid
    against a high-resolution reference, interpolating onto a common
    grid.

    Parameters
    ----------
    nest : dict
        Solved nest (method being tested).
    grids : dict
        Grids for the tested solution.
    nest_true : dict
        High-resolution reference nest.
    grids_true : dict
        Grids for the reference solution.

    Returns
    -------
    float
        Mean log10 absolute relative deviation.
    """
    a_grid = grids['a']
    a_grid_true = grids_true['a']
    n_h = len(grids['h'])
    i_h = n_h // 2
    i_z = 0

    c_by_t = get_policy(nest, 'c', stage='keeper_cons')
    c_true_by_t = get_policy(nest_true, 'c', stage='keeper_cons')

    common_ages = sorted(set(c_by_t.keys()) & set(c_true_by_t.keys()))
    if not common_ages:
        return np.nan

    deviations = []
    for t in common_ages:
        c_test = c_by_t[t][i_z, :, i_h]
        c_ref_raw = c_true_by_t[t][i_z, :, i_h]
        c_ref = np.interp(a_grid, a_grid_true, c_ref_raw)

        for i_a in range(len(a_grid)):
            if c_ref[i_a] > 1e-10 and c_test[i_a] > 1e-10:
                rel = np.abs(c_test[i_a] - c_ref[i_a]) / c_ref[i_a]
                deviations.append(np.log10(rel + 1e-16))

    return np.mean(deviations) if deviations else np.nan
