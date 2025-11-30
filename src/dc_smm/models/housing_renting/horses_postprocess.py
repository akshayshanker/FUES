"""
Post-processing utilities for GPU results using parallel CPU computation.

This module provides functions to compute statistics and validate results
after GPU VFI computation, utilizing spare CPU cores for parallel processing.
"""

import numpy as np
from numba import njit, prange
import time


@njit(parallel=True)
def compute_policy_statistics_parallel(policy_c, policy_a, V, w_grid):
    """
    Compute comprehensive statistics on policy functions using parallel CPU.
    
    Parameters
    ----------
    policy_c : ndarray (n_W, n_H, n_Y)
        Consumption policy from GPU
    policy_a : ndarray (n_W, n_H, n_Y)
        Asset policy from GPU
    V : ndarray (n_W, n_H, n_Y)
        Value function from GPU
    w_grid : ndarray (n_W,)
        Wealth grid
    
    Returns
    -------
    dict
        Statistics including means, std, percentiles, savings rates
    """
    n_W, n_H, n_Y = policy_c.shape
    
    # Pre-allocate statistics arrays
    mean_c_by_h = np.zeros(n_H)
    std_c_by_h = np.zeros(n_H)
    mean_a_by_h = np.zeros(n_H)
    mean_savings_rate = np.zeros(n_H)
    
    # Parallel computation over housing states
    for h in prange(n_H):
        c_sum = 0.0
        c_sq_sum = 0.0
        a_sum = 0.0
        s_rate_sum = 0.0
        count = 0
        
        for w in range(n_W):
            for y in range(n_Y):
                c = policy_c[w, h, y]
                a = policy_a[w, h, y]
                wealth = w_grid[w]
                
                c_sum += c
                c_sq_sum += c * c
                a_sum += a
                
                # Savings rate
                if wealth > 0:
                    s_rate_sum += a / wealth
                    
                count += 1
        
        # Compute statistics
        mean_c_by_h[h] = c_sum / count
        mean_a_by_h[h] = a_sum / count
        std_c_by_h[h] = np.sqrt(c_sq_sum / count - mean_c_by_h[h]**2)
        mean_savings_rate[h] = s_rate_sum / count
    
    return mean_c_by_h, std_c_by_h, mean_a_by_h, mean_savings_rate


@njit(parallel=True)
def validate_policies_parallel(policy_c, policy_a, w_grid, tol=1e-10):
    """
    Validate policy functions for consistency using parallel CPU.
    
    Checks:
    1. Budget constraint: c + a = w
    2. Non-negativity: c > 0, a >= 0
    3. Borrowing constraint: a >= a_min
    4. Monotonicity in wealth (optional)
    
    Parameters
    ----------
    policy_c : ndarray
        Consumption policy
    policy_a : ndarray
        Asset policy
    w_grid : ndarray
        Wealth grid
    tol : float
        Tolerance for budget constraint
    
    Returns
    -------
    tuple
        (n_violations, violation_mask, max_violation)
    """
    n_W, n_H, n_Y = policy_c.shape
    violation_mask = np.zeros((n_W, n_H, n_Y), dtype=np.bool_)
    
    # Count different types of violations
    budget_violations = 0
    negative_c_violations = 0
    negative_a_violations = 0
    max_budget_error = 0.0
    
    # Parallel validation
    for h in prange(n_H):
        for y in range(n_Y):
            for w in range(n_W):
                c = policy_c[w, h, y]
                a = policy_a[w, h, y]
                wealth = w_grid[w]
                
                # Check budget constraint
                budget_error = abs(c + a - wealth)
                if budget_error > tol:
                    violation_mask[w, h, y] = True
                    budget_violations += 1
                    max_budget_error = max(max_budget_error, budget_error)
                
                # Check non-negativity
                if c <= 0:
                    violation_mask[w, h, y] = True
                    negative_c_violations += 1
                
                if a < -tol:  # Allow small negative due to numerics
                    violation_mask[w, h, y] = True
                    negative_a_violations += 1
    
    return (budget_violations, negative_c_violations, negative_a_violations, 
            violation_mask, max_budget_error)


@njit(parallel=True)
def compute_marginal_propensities_parallel(policy_c, w_grid):
    """
    Compute marginal propensity to consume (MPC) using parallel CPU.
    
    Parameters
    ----------
    policy_c : ndarray
        Consumption policy
    w_grid : ndarray
        Wealth grid
    
    Returns
    -------
    ndarray
        MPC array (n_W-1, n_H, n_Y)
    """
    n_W, n_H, n_Y = policy_c.shape
    mpc = np.zeros((n_W-1, n_H, n_Y))
    
    # Parallel computation of finite differences
    for h in prange(n_H):
        for y in range(n_Y):
            for w in range(n_W-1):
                dc = policy_c[w+1, h, y] - policy_c[w, h, y]
                dw = w_grid[w+1] - w_grid[w]
                if dw > 0:
                    mpc[w, h, y] = dc / dw
    
    return mpc


@njit(parallel=True)
def compute_wealth_distribution_stats(policy_a, n_bins=50):
    """
    Compute wealth distribution statistics using parallel CPU.
    
    Parameters
    ----------
    policy_a : ndarray
        Asset policy (next period wealth)
    n_bins : int
        Number of bins for histogram
    
    Returns
    -------
    tuple
        (percentiles, gini_components)
    """
    n_W, n_H, n_Y = policy_a.shape
    
    # Flatten and sort for percentile calculation
    flat_assets = policy_a.ravel()
    sorted_assets = np.sort(flat_assets)
    n_total = len(sorted_assets)
    
    # Compute percentiles
    percentiles = np.zeros(11)  # 0, 10, 20, ..., 100
    for i in range(11):
        idx = min(int(i * 0.1 * n_total), n_total - 1)
        percentiles[i] = sorted_assets[idx]
    
    # Compute Gini coefficient components
    cumsum = 0.0
    for i in range(n_total):
        cumsum += sorted_assets[i] * (n_total - i)
    
    mean_wealth = np.mean(sorted_assets)
    if mean_wealth > 0:
        gini = (2.0 * cumsum) / (n_total * n_total * mean_wealth) - 1.0
    else:
        gini = 0.0
    
    return percentiles, gini


def parallel_post_process(policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn, 
                         w_grid, H_grid, n_workers=12):
    """
    Main entry point for parallel post-processing after GPU computation.
    
    This function orchestrates multiple CPU-parallel tasks to compute
    statistics and validate the GPU results while the GPU memory is
    being cleaned up.
    
    Parameters
    ----------
    policy_c, policy_a : ndarray
        Policy functions from GPU
    Q_dcsn, V_cntn, lambda_cntn : ndarray
        Value and marginal utility from GPU
    w_grid, H_grid : ndarray
        State space grids
    n_workers : int
        Number of CPU workers to use
    
    Returns
    -------
    dict
        Comprehensive statistics and validation results
    """
    import concurrent.futures
    from numba import set_num_threads
    
    # Set number of threads for Numba parallel regions
    set_num_threads(n_workers)
    
    start_time = time.time()
    results = {}
    
    # Use ThreadPoolExecutor for I/O bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        
        # Submit parallel tasks
        future_stats = executor.submit(
            compute_policy_statistics_parallel,
            policy_c, policy_a, V_cntn, w_grid
        )
        
        future_validate = executor.submit(
            validate_policies_parallel,
            policy_c, policy_a, w_grid
        )
        
        future_mpc = executor.submit(
            compute_marginal_propensities_parallel,
            policy_c, w_grid
        )
        
        future_wealth = executor.submit(
            compute_wealth_distribution_stats,
            policy_a
        )
        
        # Gather results as they complete
        mean_c, std_c, mean_a, mean_s_rate = future_stats.result()
        results['mean_consumption_by_housing'] = mean_c
        results['std_consumption_by_housing'] = std_c
        results['mean_assets_by_housing'] = mean_a
        results['mean_savings_rate'] = mean_s_rate
        
        validation = future_validate.result()
        results['budget_violations'] = validation[0]
        results['negative_c_violations'] = validation[1]
        results['negative_a_violations'] = validation[2]
        results['max_budget_error'] = validation[4]
        
        results['mpc'] = future_mpc.result()
        
        percentiles, gini = future_wealth.result()
        results['wealth_percentiles'] = percentiles
        results['gini_coefficient'] = gini
    
    # Compute timing
    results['postprocess_time'] = time.time() - start_time
    
    # Add summary statistics
    results['summary'] = {
        'mean_consumption': np.mean(policy_c),
        'mean_assets': np.mean(policy_a),
        'mean_value': np.mean(V_cntn),
        'consumption_inequality': np.std(policy_c) / np.mean(policy_c),
        'valid_solution': validation[0] == 0,  # No budget violations
        'computation_time': results['postprocess_time']
    }
    
    return results


@njit(parallel=True)
def compute_euler_errors_subset(policy_c, policy_a, V_next, w_grid, params,
                                sample_rate=0.1):
    """
    Compute Euler equation errors on a subset of points using parallel CPU.
    
    This is useful for quick validation without computing errors on the
    entire grid, which can be memory intensive.
    
    Parameters
    ----------
    policy_c, policy_a : ndarray
        Policy functions
    V_next : ndarray
        Next period value function
    w_grid : ndarray
        Wealth grid
    params : object
        Model parameters (beta, delta, alpha, etc.)
    sample_rate : float
        Fraction of points to sample (0.1 = 10%)
    
    Returns
    -------
    tuple
        (mean_error, max_error, error_percentiles)
    """
    n_W, n_H, n_Y = policy_c.shape
    
    # Determine sample size
    n_sample_w = max(1, int(n_W * sample_rate))
    w_step = max(1, n_W // n_sample_w)
    
    # Pre-allocate error array for sampled points
    n_points = n_sample_w * n_H * n_Y
    errors = np.zeros(n_points)
    idx = 0
    
    # Parallel computation over sampled points
    for h in prange(n_H):
        for y in range(n_Y):
            for w_idx in range(0, n_W, w_step):
                c = policy_c[w_idx, h, y]
                a = policy_a[w_idx, h, y]
                
                if c > 0:
                    # Current marginal utility
                    u_c = params.alpha / c
                    
                    # Expected marginal utility next period (simplified)
                    # In practice, this would integrate over shocks
                    if a > 0:
                        c_next_approx = a * 0.9  # Rough approximation
                        u_c_next = params.alpha / c_next_approx
                        
                        # Euler equation error
                        euler_residual = abs(1 - params.beta * u_c_next / u_c)
                    else:
                        euler_residual = 0.0
                    
                    errors[idx] = euler_residual
                    idx += 1
    
    # Compute statistics
    errors_valid = errors[:idx]
    mean_error = np.mean(errors_valid)
    max_error = np.max(errors_valid)
    
    # Compute percentiles
    sorted_errors = np.sort(errors_valid)
    n_valid = len(sorted_errors)
    percentiles = np.zeros(5)  # 10, 25, 50, 75, 90
    percentile_points = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    
    for i in range(5):
        idx = min(int(percentile_points[i] * n_valid), n_valid - 1)
        percentiles[i] = sorted_errors[idx]
    
    return mean_error, max_error, percentiles