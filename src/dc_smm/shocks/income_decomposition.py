"""
Income shock decomposition using transitory and permanent components.
"""

import numpy as np
import quantecon as qe
from typing import Dict, Any, Tuple


def create_income_shock_decomposition(
    rho_p: float, 
    sigma_p: float, 
    n_p: int,
    sigma_t: float, 
    n_t: int,
    rho_t: float = 0.0,
    mu_p: float = 0.0, 
    mu_t: float = 0.0,
    cover_p: float = 3.0, 
    cover_t: float = 3.0,
    method: str = 'tauchen'
) -> Dict[str, Any]:
    """
    Create transitory-permanent income shock decomposition.
    
    Income process: log(y_t) = p_t + ε_t, where:
    - p_t = ρ_p * p_{t-1} + η_t  (persistent)
    - ε_t = ρ_t * ε_{t-1} + ν_t  (transitory)
    
    Parameters
    ----------
    rho_p : float
        Persistent AR(1) parameter
    sigma_p : float
        Persistent shock standard deviation
    n_p : int
        Number of persistent states
    sigma_t : float
        Transitory shock standard deviation
    n_t : int
        Number of transitory states
    rho_t : float
        Transitory AR(1) parameter (default: 0 for i.i.d.)
    mu_p, mu_t : float
        Unconditional means (default: 0)
    cover_p, cover_t : float
        Standard deviations to cover (default: 3)
    method : str
        'tauchen' or 'rouwenhorst'
        
    Returns
    -------
    dict with:
        - income_values: Combined income grid (n_p × n_t)
        - trans_matrix: Combined transition matrix
        - persistent_states: Persistent component values
        - transitory_states: Transitory component values
        - persistent_trans: Persistent transition matrix
        - transitory_trans: Transitory transition matrix
        - stationary_dist: Stationary distribution
        - markov_chain: QuantEcon MarkovChain object
        - state_map: Mapping (p_idx, t_idx) -> combined_idx
    """
    # Create individual processes using QuantEcon
    if method == 'tauchen':
        mc_p = qe.markov.approximation.tauchen(rho_p, sigma_p, m=cover_p, n=n_p)
        mc_t = qe.markov.approximation.tauchen(rho_t, sigma_t, m=cover_t, n=n_t)
    elif method == 'rouwenhorst':
        mc_p = qe.markov.approximation.rouwenhorst(n_p, rho_p, sigma_p, mu_p)
        mc_t = qe.markov.approximation.rouwenhorst(n_t, rho_t, sigma_t, mu_t)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Extract states and add means
    p_states = mc_p.state_values + mu_p
    t_states = mc_t.state_values + mu_t
    
    # Create combined income grid: y = exp(p + ε)
    income_grid_2d = np.exp(p_states[:, np.newaxis] + t_states[np.newaxis, :])
    income_values = income_grid_2d.flatten()
    
    # Combined transition matrix via Kronecker product
    trans_matrix = np.kron(mc_p.P, mc_t.P)
    
    # State mapping
    state_map = {(i, j): i * n_t + j for i in range(n_p) for j in range(n_t)}
    inverse_map = {v: k for k, v in state_map.items()}
    
    # Create combined MarkovChain and get stationary distribution
    mc_combined = qe.MarkovChain(trans_matrix, state_values=income_values)
    stationary_dist = mc_combined.stationary_distributions[0]
    
    return {
        'income_values': income_values,
        'trans_matrix': trans_matrix,
        'persistent_states': p_states,
        'transitory_states': t_states,
        'persistent_trans': mc_p.P,
        'transitory_trans': mc_t.P,
        'stationary_dist': stationary_dist,
        'persistent_stationary': mc_p.stationary_distributions[0],
        'transitory_stationary': mc_t.stationary_distributions[0],
        'markov_chain': mc_combined,
        'state_map': state_map,
        'inverse_map': inverse_map,
        'n_states': n_p * n_t,
        'n_persistent': n_p,
        'n_transitory': n_t,
        'income_grid_2d': income_grid_2d
    }


def compute_income_statistics(shock_params: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute key statistics from the income process.
    
    Returns mean, variance, Gini, and autocorrelation.
    """
    mc = shock_params['markov_chain']
    stat_dist = shock_params['stationary_dist']
    income_values = shock_params['income_values']
    
    # Basic moments
    mean_income = np.dot(stat_dist, income_values)
    var_income = np.dot(stat_dist, (income_values - mean_income)**2)
    
    # Log income variance
    log_income = np.log(income_values)
    mean_log = np.dot(stat_dist, log_income)
    var_log = np.dot(stat_dist, (log_income - mean_log)**2)
    
    # Component variances
    p_states = shock_params['persistent_states']
    t_states = shock_params['transitory_states']
    p_stat = shock_params['persistent_stationary']
    t_stat = shock_params['transitory_stationary']
    
    var_p = np.dot(p_stat, (p_states - np.dot(p_stat, p_states))**2)
    var_t = np.dot(t_stat, (t_states - np.dot(t_stat, t_states))**2)
    
    # Autocorrelation via simulation
    sim = mc.simulate(ts_length=10000)
    autocorr = np.corrcoef(sim[:-1], sim[1:])[0, 1]
    
    # Gini coefficient
    sorted_idx = np.argsort(income_values)
    sorted_vals = income_values[sorted_idx]
    sorted_probs = stat_dist[sorted_idx]
    cum_probs = np.cumsum(sorted_probs)
    cum_income = np.cumsum(sorted_vals * sorted_probs)
    cum_income = cum_income / cum_income[-1]
    gini = 1 - 2 * np.trapz(cum_income, cum_probs)
    
    return {
        'mean_income': mean_income,
        'var_income': var_income,
        'var_log_income': var_log,
        'var_persistent': var_p,
        'var_transitory': var_t,
        'autocorr_income': autocorr,
        'gini': gini
    }


def get_state_indices(combined_idx: int, n_transitory: int) -> Tuple[int, int]:
    """Convert combined index to (persistent, transitory) indices."""
    return combined_idx // n_transitory, combined_idx % n_transitory


def get_combined_index(p_idx: int, t_idx: int, n_transitory: int) -> int:
    """Convert (persistent, transitory) indices to combined index."""
    return p_idx * n_transitory + t_idx