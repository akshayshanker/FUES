#!/usr/bin/env python
"""
Housing model with renting driver for ModCraft.

This script loads, initializes, and solves the housing model with renting
using the StageCraft and Heptapod-B architecture.

Usage:
    python housing.py [--periods N] [--plot] [--no-solve] [--ue-method METHOD]

Options:
    --periods N     Number of periods to simulate (default: 3)
    --plot          Generate and save plots
    --no-solve      Skip solving (for testing loading only)
    --ue-method     Upper-envelope cleaning method (default: FUES)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# Import from ModCraft
from dynx.stagecraft import Stage
from dynx.stagecraft.config_loader import (
    initialize_model_Circuit,
    compile_all_stages,
)
from dynx.stagecraft.saver import save_circuit, load_circuit
from dynx.heptapodx.io.yaml_loader import load_config
from dynx.heptapodx.core.api import initialize_model
from dynx.heptapodx.num.generate import compile_num as generate_numerical_model

current_dir = os.path.dirname(os.path.abspath(__file__))

# Import housing model

from .whisperer import (
    build_operators,
    solve_stage,
    run_time_iteration,
)
# Import plotting helpers and error metrics (also local to this example)
from .helpers.plots import generate_plots, plot_compare_value_Q

# ------------------------------------------------------------------
# Global axis-bounds for plotting.  Keys → (xmin, xmax, ymin, ymax)
# Use None to keep default on an individual edge.
# ------------------------------------------------------------------
BOUNDS = {
    # "cons_owner": (0, 20, 0, 15),
    "egm_value":  (19, 21, 1.49, 1.7),
    "egm_assets": (19, 21, 10, 14.5),
    "Q":          (None, None, -2, 4),
}

def load_configs():
    """Load all configuration files."""
    config_dir = os.path.join(os.path.dirname(__file__), "config_HR")
    master_path = os.path.join(config_dir, "master.yml")
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

def initialize_housing_model():
    """Initialize the housing model with renting using YAML configuration.
    
    Returns
    -------
    dict
        Dictionary containing all five stages (TENU, OWNH, OWNC, RNTH, RNTC)
    """
    print("Initializing housing model with renting...")
    
    # Load configurations
    configs = load_configs()
    # Operator-specific settings will be injected later (via CLI --ue-method)
    
    # Prepare stages config dictionary with all five stages
    stage_configs = {
        "TENU": configs["tenu"],
        "OWNH": configs["ownh"],
        "OWNC": configs["ownc"],
        "RNTH": configs["renth"],
        "RNTC": configs["rentc"]
    }

    
    
    # Let the framework handle all the complexity of creating stages,
    # perches, movers, and connections
    print("Building model circuit using configuration...")
    model_circuit = initialize_model_Circuit(
        master_config=configs["master"],
        stage_configs=stage_configs,
        connections_config=configs["connections"]
    )

    compile_all_stages(model_circuit)
    
    # Get the individual stages from the model circuit
    # For the housing model with renting, we have one period with five stages
    period = model_circuit.get_period(0)
    tenu_stage = period.get_stage("TENU")
    ownh_stage = period.get_stage("OWNH")
    ownc_stage = period.get_stage("OWNC")
    rnth_stage = period.get_stage("RNTH")
    rntc_stage = period.get_stage("RNTC")
    
    # Debug: Print information about the Markov shock process
    print("\nShock information for TENU stage:")
    shock_info = tenu_stage.model.num.shocks.income_shock
    print(f"Transition matrix shape: {shock_info.transition_matrix.shape}")
    
    # Mark final period stages
    # Only consumption stages are marked as terminal
    ownh_stage.status_flags["is_terminal"] = False
    ownc_stage.status_flags["is_terminal"] = True
    rnth_stage.status_flags["is_terminal"] = False
    rntc_stage.status_flags["is_terminal"] = True
    tenu_stage.status_flags["is_terminal"] = False
    
    # Set external mode for stages
    ownh_stage.model_mode = "external"
    ownc_stage.model_mode = "external"
    rnth_stage.model_mode = "external"
    rntc_stage.model_mode = "external"
    tenu_stage.model_mode = "external"
    
    # Return all stages as a dictionary
    return {
        "TENU": tenu_stage,
        "OWNH": ownh_stage,
        "OWNC": ownc_stage,
        "RNTH": rnth_stage,
        "RNTC": rntc_stage
    }

def create_multi_period_model(n_periods=3, configs=None):
    """Create a multi-period model circuit for the housing model with renting.
    
    Parameters
    ----------
    n_periods : int
        Number of periods to create
    configs : dict, optional
        Pre-loaded configuration dictionaries. If provided, these configs are used
        instead of loading from files. Useful for modifying configs before model creation.
        
    Returns
    -------
    ModelCircuit
        The multi-period model circuit
    """
    print(f"Creating multi-period model with {n_periods} periods...")
    
    # Load configurations or use provided configs
    if configs is None:
        configs = load_configs()
    
    # Set the number of periods in the master config
    # This is critical to properly create a multi-period model
    master_config = copy.deepcopy(configs["master"])
    master_config["horizon"] = n_periods

    configs["master"]["horizon"] = n_periods
    
    # Prepare stages config dictionary with all five stages
    stage_configs = {
        "TENU": configs["tenu"],
        "OWNH": configs["ownh"],
        "OWNC": configs["ownc"],
        "RNTH": configs["renth"],
        "RNTC": configs["rentc"]
    }
    
    # Build the multi-period model circuit
    print("Building multi-period model circuit...")
    model_circuit = initialize_model_Circuit(
        master_config=master_config,
        stage_configs=stage_configs,
        connections_config=configs["connections"]
    )

    compile_all_stages(model_circuit)

    print("done")
    
    return model_circuit, configs

def create_image_dir():
    """Create directory for output images."""
    image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "images"))
    if not os.path.exists(image_dir):
        print(f"Creating image directory: {image_dir}")
        os.makedirs(image_dir)
    return image_dir

def create_save_dir():
    """Create directory for output images."""
    sol_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "solutions"))
    if not os.path.exists(sol_dir):
        print(f"Creating solutions directory: {sol_dir}")
        os.makedirs(sol_dir)
    return sol_dir

def main(argv=None):
    """Main driver function.

    Parameters
    ----------
    argv : list[str] or None
        If given, parsed instead of sys.argv[1:].  Handy for calling
        `housing.main(["--ue-method","DCEGM"])` from IPython.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Housing model with renting driver")
    parser.add_argument("--periods", type=int, default=3, help="Number of periods to simulate")
    parser.add_argument("--plot", action="store_true", help="Generate and save plots")
    parser.add_argument("--no-solve", action="store_true", help="Skip solving (for testing loading only)")
    parser.add_argument("--ue-method", default="FUES", choices=["FUES", "DCEGM", "RFC","CONSAV","FUES2DEV" ,"simple", "VFI", "VFI_GRID"], help="Upper-envelope cleaning method")
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    
    # Create image directory
    image_dir = create_image_dir()
    
    # Initialize model
    #print("periods", args.periods)
    model_circuit, master_config = create_multi_period_model(n_periods=args.periods)

    # ------------------------------------------------------------------
    # Inject chosen upper-envelope method into operator settings of each
    # stage that uses EGM (typically OWNC, RNTC, etc.).  horses_c.py readsgrid
    # the setting from model.operator.get("upper_envelope", "FUES")
    # ------------------------------------------------------------------
    for period in model_circuit.periods_list:
        for stage_name, stage in period.stages.items():
            if stage_name in ["OWNC", "RNTC"]:  # Only consumption stages use EGM
                # The EGM loop is in the backward cntn_to_dcsn mover
                mover = stage.cntn_to_dcsn
                if hasattr(mover, "model") and mover.model:
                    # Create operator dict if it doesn't exist
                    if not hasattr(mover.model, "operator"):
                        mover.model.operator = {}
                    # Inject the upper envelope method
                    mover.model.methods["upper_envelope"] = args.ue_method.upper()
                    #print(args.ue_method.upper())
                    if args.ue_method.upper() == "VFI":
                        mover.model.methods["solution"] = "VFI"
                    elif args.ue_method.upper() == "VFI_GRID":
                         mover.model.methods["solution"] = "VFI_GRID"
                    else:
                        mover.model.methods["solution"] = "EGM"
                    print(f"Set {stage_name}.cntn_to_dcsn.model.methods['upper_envelope'] = {args.ue_method.upper()}")
                    #print(mover.model.methods["solution"])
                else:
                    print(f"Warning: {stage_name}.cntn_to_dcsn has no model")

    # Solve the multi-period model - set verbose to True for detailed output
    all_stages_solved = run_time_iteration(model_circuit, n_periods=args.periods, verbose=True)

    # Main policy & EGM plots
    generate_plots(model_circuit, args.ue_method, image_dir,
                   plot_period=0, bounds=BOUNDS, save_dir=None)

    # Return the solved model circuit so that callers can build composites
    return model_circuit, master_config

if __name__ == "__main__":

    IMAGE_DIR = create_image_dir()
    SAVE_DIR = create_save_dir()
    

    # Select the methods you want to compare here.
    METHODS_TO_RUN = [
        "FUES2DEV",
        "DCEGM",
        # "DCEGM",  # uncomment when desired
    ]

    solved_models = []


    for mtd in METHODS_TO_RUN:
        mdl, configs = main(["--periods", "3", "--ue-method", mtd, "--plot"])
        solved_models.append(mdl)

        # ---- NEW: persist this circuit --------------------------------------
        # Use the config directory we already know (config_HR) as `config_src`.
        config_src_dir = os.path.join(current_dir, "config_HR")
        # cfg_bundle is the dict returned above; unpack wanted entries
        master_cfg      = configs["master"]
        stage_cfgs_dict = {k: configs[k] for k in ["tenu", "ownh", "ownc", "renth", "rentc"]}
        connections_cfg = configs["connections"]

        # need to put in config_src_dir first to ensure edited config files are overwritten!
        # TODO: needs fix at DYNX saver. 
        config_sources = [config_src_dir, master_cfg, *stage_cfgs_dict.values(), connections_cfg] 
        save_path = save_circuit(mdl, SAVE_DIR, config_sources,
                                model_id=f"{mtd}")

        print(f"Model '{mtd}' saved to: {save_path}")
        # ---------------------------------------------------------------------

        loaded_circuit = load_circuit(save_path)

        # make a sibling directory called “…/images_loaded/<method>/”
        loaded_img_dir = os.path.join(
            os.path.dirname(IMAGE_DIR), "images_loaded", mtd.lower()
        )
        os.makedirs(loaded_img_dir, exist_ok=True)

        # regenerate one set of plots from the *loaded* circuit
        generate_plots(
            loaded_circuit,
            mtd,
            loaded_img_dir,
            plot_period=0,
            bounds=BOUNDS,
            save_dir=None,
        )
        print(f"Reloaded circuit plotted to: {loaded_img_dir}")
    # ---------------------------------------------------------------------


    print("Done.")


    

    