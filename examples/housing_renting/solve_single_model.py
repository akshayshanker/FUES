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
from pathlib import Path

# Import from ModCraft
from dynx.stagecraft import Stage
from dynx.stagecraft.makemod import (
    initialize_model_Circuit,
    compile_all_stages,
)
from dynx.stagecraft.io import save_circuit, load_circuit
from dynx.stagecraft.io import load_config         # ← NEW canonical loader
from dynx.heptapodx.core.api import initialize_model
from dynx.heptapodx.num.generate import compile_num as generate_numerical_model
from dynx.stagecraft.makemod import initialize_model_Circuit, compile_all_stages

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


def create_multi_period_model(
    n_periods: int = 3,
    config_dir: str | Path | None = None,
    configs: dict | None = None,
    ue_method: str = "FUES",
):
    """
    Build a multi-period housing-with-renting ModelCircuit.

    Parameters
    ----------
    n_periods : int
        Desired planning horizon (number of periods).
    config_dir : str | Path, optional
        Path to the folder that contains *master.yml*, *stages/*, and
        *connections.yml*.  Defaults to ``<this_file>/config_HR``.
    configs : dict, optional
        An already-loaded config dictionary (same structure returned by
        ``load_config``).  If supplied, it will be *mutated in-place* to
        reflect the requested horizon.

    Returns
    -------
    model_circuit : ModelCircuit
    configs       : dict
        The (possibly modified) configuration dictionary actually used.
    """
    print(f"Creating multi-period model with {n_periods} periods …")

    # ------------------------------------------------------------------
    # 1. Obtain the configuration dict in the *new* canonical format
    # ------------------------------------------------------------------
    if configs is None:
        if config_dir is None:
            # default to sibling folder  …/examples/config_HR
            config_dir = Path(__file__).with_suffix("").parent / "config_HR"
        configs = load_config(config_dir)  # <- uses dynx.stagecraft.io helper

    # ------------------------------------------------------------------
    # 2. Mutate horizon only once
    # ------------------------------------------------------------------
    configs["master"] = copy.deepcopy(configs["master"])
    configs["master"]["horizon"] = n_periods     # (sometimes called 'periods')
    
    #---
    #2.1 Update methods
    #---
    if ue_method == "FUES":
        configs["master"]["methods"] = {"upper_envelope": "FUES"}
    elif ue_method == "DCEGM":
        configs["master"]["methods"] = {"upper_envelope": "DCEGM"}
    elif ue_method == "RFC":
        configs["master"]["methods"] = {"upper_envelope": "RFC"}
    elif ue_method == "CONSAV":
        configs["master"]["methods"] = {"upper_envelope": "CONSAV"}
    elif ue_method == "VFI":
        configs["master"]["methods"] = {"upper_envelope": "VFI"}
    elif ue_method == "VFI_HDGRID":
        configs["master"]["methods"] = {"upper_envelope": "VFI"}
    else:
        configs["master"]["methods"] = {"upper_envelope": "EGM"}

    if ue_method == "VFI" or ue_method == "VFI_HDGRID":
        # mover just takes the stage level methods and only the stage level methods
        # TODO a fix to mover methods assign. 
        configs["stages"]["OWNC"]["stage"]["methods"]["solution"] = ue_method
        configs["stages"]["RNTC"]["stage"]["methods"]["solution"] = ue_method
    else:
        configs["stages"]["OWNC"]["stage"]["methods"]["solution"] = "EGM"
        configs["stages"]["RNTC"]["stage"]["methods"]["solution"] = "EGM"

    # ------------------------------------------------------------------
    # 3. Build and compile the circuit
    # ------------------------------------------------------------------
    print("Building multi-period model circuit …")
    model_circuit = initialize_model_Circuit(
        master_config=configs["master"],
        stage_configs=configs["stages"],      # dict: name → stage-dict
        connections_config=configs["connections"],
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
    parser.add_argument("--ue-method", default="FUES", choices=["FUES", "DCEGM", "RFC","CONSAV","FUES" ,"simple", "VFI", "VFI_HDGRID"], help="Upper-envelope cleaning method")
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    
    # Create image directory
    image_dir = create_image_dir()
    
    # Initialize model
    #print("periods", args.periods)
    model_circuit, master_config = create_multi_period_model(n_periods=args.periods, ue_method=args.ue_method.upper())

    # ------------------------------------------------------------------
    # Inject chosen upper-envelope method into operator settings of each
    # stage that uses EGM (typically OWNC, RNTC, etc.).  horses_c.py readsgrid
    # the setting from model.operator.get("upper_envelope", "FUES")
    # ------------------------------------------------------------------

    ''''
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
    '''

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
        "FUES",
        "DCEGM",
        # "DCEGM",  # uncomment when desired
    ]

    solved_models = []


    # ------------------------------------------------------------------
    # run / save / reload / re-plot for each UE method
    # ------------------------------------------------------------------
    for mtd in METHODS_TO_RUN:
        # 1. solve and get canonical config container
        mdl, cfg = main(["--periods", "4", "--ue-method", mtd, "--plot"])
        solved_models.append(mdl)

        # 2. persist ----------------------------------------------------
        save_path = save_circuit(
            mdl,
            SAVE_DIR,
            cfg,              # ← single canonical dict now accepted
            model_id=mtd.lower(),
        )
        print(f"Model '{mtd}' saved to: {save_path}")

        # 3. reload -----------------------------------------------------
        loaded_circuit = load_circuit(save_path)

        # 4. plot from the re-loaded circuit ---------------------------
        loaded_img_dir = os.path.join(
            os.path.dirname(IMAGE_DIR), "images_loaded", mtd.lower()
        )
        os.makedirs(loaded_img_dir, exist_ok=True)

        generate_plots(
            loaded_circuit,
            mtd,
            loaded_img_dir,
            plot_period=0,
            bounds=BOUNDS,
            save_dir=None,    # send plots to disk only
        )
        print(f"Reloaded circuit plotted to: {loaded_img_dir}")

    print("Done.")



    print("Done.")


    

    