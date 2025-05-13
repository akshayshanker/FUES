#!/usr/bin/env python
"""
Housing model terminal stage test.

This script tests the initialization and solving of terminal period
stages in the housing model, using the StageCraft architecture.

Usage:
    python solve_stage_test.py
"""

import os
import sys
import numpy as np
import copy

# Add modcraft root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import from ModCraft
from src.stagecraft import Stage
from src.stagecraft.config_loader import initialize_model_Circuit
from src.heptapod_b.io.yaml_loader import load_config
from src.heptapod_b.num.generate import compile_num as generate_numerical_model

# Add the models directory to the Python path
# This ensures models.housing can be found
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models"))
sys.path.insert(0, os.path.dirname(models_dir))

# Import housing model whisperer functions
from models.housing.whisperer import (
    solve_stage,
    initialize_terminal_values,
)

def load_configs():
    """Load all configuration files."""
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    master_path = os.path.join(config_dir, "housing_master.yml")
    ownh_path = os.path.join(config_dir, "OWNH_stage.yml") 
    ownc_path = os.path.join(config_dir, "OWNC_stage.yml")
    renth_path = os.path.join(config_dir, "RNTH_stage.yml")
    rentc_path = os.path.join(config_dir, "RNTC_stage.yml")
    tenu_path = os.path.join(config_dir, "TENU_stage.yml")
    connections_path = os.path.join(config_dir, "connections.yml")
    
    # Load configurations
    print("Loading configurations...")
    master_config = load_config(master_path)
    ownh_config = load_config(ownh_path)
    ownc_config = load_config(ownc_path)
    renth_config = load_config(renth_path)
    rentc_config = load_config(rentc_path)
    tenu_config = load_config(tenu_path)
    connections_config = load_config(connections_path)
    
    return {
        "master": master_config,
        "ownh": ownh_config,
        "ownc": ownc_config,
        "renth": renth_config,
        "rentc": rentc_config,
        "tenu": tenu_config,
        "connections": connections_config
    }

def main():
    """Run the terminal period test."""
    print("\n===== RUNNING TERMINAL PERIOD DEBUG TEST =====")
    
    # Load configurations
    configs = load_configs()
    
    # Create stage configs dictionary
    stage_configs = {
        "OWNH": configs["ownh"],
        "OWNC": configs["ownc"],
        "RNTH": configs["renth"],
        "RNTC": configs["rentc"],
        "TENU": configs["tenu"]
    }
    
    # Create a model circuit with just one period (for terminal period test)
    model_circuit = initialize_model_Circuit(
        master_config=configs["master"],
        stage_configs=stage_configs,
        connections_config=configs["connections"]
    )
    
    print("model_circuit")
    
    # Access the last period (which we'll treat as terminal)
    terminal_period = model_circuit.get_period(0)
    
    # Get the stages in the terminal period
    ownh_stage = terminal_period.get_stage("OWNH")
    ownc_stage = terminal_period.get_stage("OWNC")
    renth_stage = terminal_period.get_stage("RNTH")
    rentc_stage = terminal_period.get_stage("RNTC")
    tenu_stage = terminal_period.get_stage("TENU")
    
    # Set numerical representation for both stages
    ownh_stage.num_rep = generate_numerical_model
    ownc_stage.num_rep = generate_numerical_model
    renth_stage.num_rep = generate_numerical_model
    rentc_stage.num_rep = generate_numerical_model
    tenu_stage.num_rep = generate_numerical_model
    
    # Set external mode for solving
    ownh_stage.model_mode = "external"
    ownc_stage.model_mode = "external"
    renth_stage.model_mode = "external"
    rentc_stage.model_mode = "external"
    tenu_stage.model_mode = "external"
    
    # Mark OWNC as the terminal stage
    ownc_stage.status_flags["is_terminal"] = True
    
    # Build computational models for both stages
    print("Building computational models...")
    ownh_stage.build_computational_model()
    ownc_stage.build_computational_model()
    renth_stage.build_computational_model()
    rentc_stage.build_computational_model()
    tenu_stage.build_computational_model()

    print("Done.")

     
    # Initialize terminal values for OWNC stage
    print("Initializing terminal values...")
    initialize_terminal_values(ownc_stage)
    initialize_terminal_values(rentc_stage)

    # Solve OWNC stage
    print("Solving OWNC stage...")
    solve_stage(ownc_stage)
    solve_stage(rentc_stage)

    print("Connecting stages...")
    ownh_stage.cntn.sol = {
        "vlu": copy.deepcopy(ownc_stage.arvl.sol["vlu"]),
        "lambda": copy.deepcopy(ownc_stage.arvl.sol["lambda"])
    }

    renth_stage.cntn.sol = {
        "vlu": copy.deepcopy(rentc_stage.arvl.sol["vlu"]),
        "lambda": copy.deepcopy(rentc_stage.arvl.sol["lambda"])
    }
    print("Done.")
    
    # Solve stages in order: OWNC, then OWNH
    print("Solving OWNH stage...")
    solve_stage(ownh_stage)
    solve_stage(renth_stage)
    print("Done.")
    
   
    # Print some debugging information
    print("\n===== DEBUG INFORMATION =====")
    print(f"OWNH grid dimensions:")
    print(f"  Asset grid: {ownh_stage.arvl.grid.a.shape}, range: [{ownh_stage.arvl.grid.a.min()}, {ownh_stage.arvl.grid.a.max()}]")
    print(f"  Housing grid: {ownh_stage.arvl.grid.H.shape}, range: [{ownh_stage.arvl.grid.H.min()}, {ownh_stage.arvl.grid.H.max()}]")
    
    print(f"\nOWNC grid dimensions:")
    print(f"  Cash-on-hand grid: {ownc_stage.dcsn.grid.w.shape}, range: [{ownc_stage.dcsn.grid.w.min()}, {ownc_stage.dcsn.grid.w.max()}]")
    
    print(f"\nSolution information:")
    print(f"  OWNH housing policy shape: {ownh_stage.dcsn.sol['H_policy'].shape}")
    print(f"  OWNC consumption policy shape: {ownc_stage.dcsn.sol['policy'].shape}")
    
    # Sample values from policy functions (middle of grids)
    h_idx = len(ownh_stage.dcsn.grid.H) // 2  # Middle housing value
    y_idx = 0  # First income state
    a_mid_idx = len(ownh_stage.dcsn.grid.a) // 2  # Middle asset value
    w_mid_idx = len(ownc_stage.dcsn.grid.w) // 2  # Middle cash-on-hand value
    
    print(f"\nSample policy values (mid-points):")
    print(f"  Housing choice at a={ownh_stage.dcsn.grid.a[a_mid_idx]:.2f}: {ownh_stage.cntn.grid.H_nxt[ownh_stage.dcsn.sol['H_policy'][a_mid_idx, h_idx, y_idx]]:.2f}")
    print(f"  Consumption at w={ownc_stage.dcsn.grid.w[w_mid_idx]:.2f}: {ownc_stage.dcsn.sol['policy'][w_mid_idx, h_idx, y_idx]:.2f}")
    
    print("\n===== TERMINAL PERIOD TEST COMPLETED =====")
    

if __name__ == "__main__":
    main()

    print("Done.")