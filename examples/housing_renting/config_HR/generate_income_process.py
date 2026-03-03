#!/usr/bin/env python3
"""
Generate income process based on Fella (2014) parameters.

This script generates the income process using the parameters from 
Fella's 2014 paper "A generalized endogenous grid method for non-smooth 
and non-concave problems" and saves it in YAML format for use with FUES.
"""

import numpy as np
import yaml
from pathlib import Path
import sys

# Add FUES repo root to path (4 levels up from config_HR/generate_income_process.py)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))

from dcsmm.shocks import create_income_shock_decomposition


def generate_fella_income_process():
    """
    Generate income process using Fella (2014) parameters.
    
    From Table 1 in Fella (2014):
    - ρ (rho) = 0.977 (persistence parameter)
    - ση (sigma_eta) = 0.024 (standard deviation of persistent shock)
    - σy (sigma_y) = 0.063 (total income volatility)
    
    The income process is:
    log(yt) = zt + εt
    zt = ρ * zt-1 + ηt
    
    where εt ~ N(0, σε²) and ηt ~ N(0, ση²)
    """
    
    # Parameters from Fella (2014) - Table 1
    rho = 0.977           # Persistence parameter
    sigma_eta = 0.024     # Standard deviation of persistent shock
    
    # From Storesletten et al. (2000) as referenced in Fella:
    # The income process has persistent and transitory components
    # Based on the parametrization in the paper, we need to decompose
    # the total variance into persistent and transitory parts
    
    # The paper mentions using Storesletten et al. (2000) estimates
    # For a log-normal process with persistent and transitory components:
    # log(y) = z + ε, where z is persistent and ε is transitory
    
    # Standard deviation of transitory shock (from literature)
    # Storesletten et al. report σ_ε ≈ 0.063 for transitory shocks
    sigma_epsilon = 0.063  # Transitory shock standard deviation
    
    # Note: The total unconditional variance of log income is:
    # Var(log y) = Var(z) + Var(ε) = σ_η²/(1-ρ²) + σ_ε²
    
    print(f"Income process parameters:")
    print(f"  Persistence (ρ): {rho}")
    print(f"  Persistent shock std (ση): {sigma_eta}")
    print(f"  Transitory shock std (σε): {sigma_epsilon}")
    
    # Number of grid points
    n_persistent = 7  # Number of persistent states
    n_transitory = 7  # Number of transitory states
    
    
    # Use create_income_shock_decomposition with Fella's parameters
    # Use Rouwenhorst method for highly persistent process
    shock_dict = create_income_shock_decomposition(
        rho_p=rho,              # Persistent AR(1) parameter
        sigma_p=sigma_eta,      # Persistent shock std
        n_p=n_persistent,       # Number of persistent states
        sigma_t=sigma_epsilon,  # Transitory shock std
        n_t=n_transitory,       # Number of transitory states
        rho_t=0.0,             # Transitory shocks are i.i.d.
        mu_p=0.0,              # Zero mean
        mu_t=0.0,              # Zero mean
        cover_p=3.0,           # Coverage
        cover_t=3.0,           # Coverage
        method='rouwenhorst'   # Use Rouwenhorst for high persistence
    )
    
    # Extract the combined values and transition matrix
    y_vals = shock_dict['income_values']  # Already in levels
    Pi = shock_dict['trans_matrix']
    
    # Normalize income values to have mean 1
    y_vals = y_vals / np.mean(y_vals)
    
    # Create output dictionary in the required format
    income_process = {
        'Pi': Pi.tolist() if isinstance(Pi, np.ndarray) else Pi,
        'z_vals': y_vals.tolist() if isinstance(y_vals, np.ndarray) else y_vals
    }
    
    # Print summary statistics
    n_states = len(y_vals)
    print(f"\nGenerated income process with {n_states} states")
    print(f"Income values range: [{np.min(y_vals):.3f}, {np.max(y_vals):.3f}]")
    print(f"Mean income: {np.mean(y_vals):.3f}")
    print(f"Std of income: {np.std(y_vals):.3f}")
    
    # Check ergodic distribution
    Pi_array = np.array(Pi) if not isinstance(Pi, np.ndarray) else Pi
    eigenvalues, eigenvectors = np.linalg.eig(Pi_array.T)
    stationary_idx = np.argmax(np.abs(eigenvalues))
    stationary_dist = np.real(eigenvectors[:, stationary_idx])
    stationary_dist = stationary_dist / stationary_dist.sum()
    
    y_vals_array = np.array(y_vals) if not isinstance(y_vals, np.ndarray) else y_vals
    ergodic_mean = np.dot(stationary_dist, y_vals_array)
    ergodic_std = np.sqrt(np.dot(stationary_dist, (y_vals_array - ergodic_mean)**2))
    
    print(f"\nErgodic distribution statistics:")
    print(f"  Mean: {ergodic_mean:.3f}")
    print(f"  Std: {ergodic_std:.3f}")
    
    return income_process


def save_income_process_yaml(income_process, filename='income_process_fella2014.yml'):
    """Save income process to YAML file."""
    output_path = Path(__file__).parent / filename
    
    with open(output_path, 'w') as f:
        yaml.dump(income_process, f, default_flow_style=None, sort_keys=False)
    
    print(f"\nIncome process saved to: {output_path}")
    
    # Also save a simplified version for testing
    if len(income_process['z_vals']) > 3:
        # Create a 3-state approximation for testing
        n = len(income_process['z_vals'])
        indices = [0, n//2, n-1]  # Low, middle, high
        
        simple_process = {
            'Pi': [[income_process['Pi'][i][j] for j in indices] for i in indices],
            'z_vals': [income_process['z_vals'][i] for i in indices]
        }
        
        # Renormalize transition matrix
        for i in range(3):
            row_sum = sum(simple_process['Pi'][i])
            simple_process['Pi'][i] = [p/row_sum for p in simple_process['Pi'][i]]
        
        simple_filename = filename.replace('.yml', '_simple.yml')
        simple_path = Path(__file__).parent / simple_filename
        
        with open(simple_path, 'w') as f:
            yaml.dump(simple_process, f, default_flow_style=None, sort_keys=False)
        
        print(f"Simplified 3-state version saved to: {simple_path}")


def main():
    """Generate and save the income process."""
    print("Generating income process based on Fella (2014) parameters...")
    print("="*60)
    
    income_process = generate_fella_income_process()
    save_income_process_yaml(income_process)
    
    print("\nDone!")


if __name__ == '__main__':
    main()