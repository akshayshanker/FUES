"""Nest accessors and solution diagnostics."""

import numpy as np


def get_policy(nest, key, stage='labour_mkt_decision'):
    """Get T x n array from nest solutions, indexed by age t.

    Parameters
    ----------
    nest : dict
        Solved nest from :func:`build_and_solve_nest`.
    key : str
        Field name within the stage solution dict
        (e.g. ``"c"``, ``"v"``, ``"dv"``).
    stage : str
        Stage name (default: ``labour_mkt_decision``).

    Returns
    -------
    ndarray (T x n)
    """
    sols = nest["solutions"]
    T, n = len(sols), len(sols[0][stage][key])
    arr = np.empty((T, n))
    for sol in sols:
        arr[sol["t"]] = sol[stage][key]
    return arr


def get_timing(nest):
    """Mean UE time and solve time (skipping first 3 warmup).

    Returns
    -------
    list
        ``[mean_ue_time, mean_solve_time]``.
    """
    ue = [s["ue_time"] for s in nest["solutions"]
          if s["h"] > 2]
    total = [s["solve_time"] for s in nest["solutions"]
             if s["h"] > 2]
    return [
        np.mean(ue) if ue else 0.0,
        np.mean(total) if total else 0.0,
    ]


def get_solution_at_age(nest, t):
    """Get solution dict for calendar age *t*.

    Parameters
    ----------
    nest : dict
        Solved nest.
    t : int
        Calendar time (age), where ``t = T-1`` is the
        last decision period.

    Returns
    -------
    dict
        Solution dict for age *t*.
    """
    T = len(nest["solutions"])
    return nest["solutions"][T - 1 - t]


def euler(cp, sigma_work):
    """Mean log10 Euler equation error across periods.

    For each grid point and period, computes the residual
    of the consumption Euler equation and returns the
    mean of ``log10(|residual / c| + 1e-16)``.

    Parameters
    ----------
    cp : RetirementModel
        Model instance (provides grid, R, beta, du, uc_inv).
    sigma_work : ndarray (T x grid_size)
        Consumption policy on the asset grid.

    Returns
    -------
    float
        Mean log10 Euler error (more negative = better).
    """
    a_grid = cp.asset_grid_A
    errors = np.zeros((cp.T - 1, cp.eulerK))
    errors.fill(np.nan)

    for t in range(cp.T - 1):
        for i_a in range(cp.eulerK):
            a = a_grid[i_a]
            c = np.interp(a, a_grid, sigma_work[t])
            a_prime = a * cp.R + cp.y - c

            if a_prime < 0.001 or a_prime > 300:
                continue

            c_plus = np.interp(
                a, a_grid, sigma_work[t + 1],
            )
            RHS = cp.beta * cp.R * cp.du(c_plus)
            euler_raw = c - cp.uc_inv(RHS)
            errors[t, i_a] = np.log10(
                np.abs(euler_raw / c) + 1e-16,
            )

    return np.nanmean(errors)


def consumption_deviation(cp, c_solution, c_true,
                          a_grid_true):
    """Mean log10 deviation from a high-resolution solution.

    Compares consumption on a coarser grid to a
    high-resolution reference (e.g. DCEGM with 20k points).

    Parameters
    ----------
    cp : RetirementModel
        Model parameters for the solution being tested.
    c_solution : ndarray (T x grid_size)
        Consumption policy from the method being tested.
    c_true : ndarray (T x true_grid_size)
        High-resolution reference solution.
    a_grid_true : ndarray
        Asset grid of the reference solution.

    Returns
    -------
    float
        Mean log10 absolute relative deviation.
    """
    a_grid = cp.asset_grid_A
    T = cp.T
    deviations = np.zeros((T - 1, len(a_grid)))
    deviations.fill(np.nan)

    for t in range(T - 1):
        c_true_interp = np.interp(
            a_grid, a_grid_true, c_true[t],
        )
        c_test = c_solution[t]
        for i_a in range(len(a_grid)):
            if (c_true_interp[i_a] > 1e-10
                    and c_test[i_a] > 1e-10):
                rel_error = (
                    np.abs(c_test[i_a] - c_true_interp[i_a])
                    / c_true_interp[i_a]
                )
                deviations[t, i_a] = np.log10(
                    rel_error + 1e-16,
                )

    return np.nanmean(deviations)
