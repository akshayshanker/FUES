#!/usr/bin/env python
"""
Script to debug the UE time reporting issue
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dynx.runner import CircuitRunner

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the whisperer module to examine timing
from examples.housing_renting.whisperer import solve_stage, run_time_iteration
from examples.housing_renting.circuit_runner_solving import initialize_housing_model, load_configs

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the HRModel for monkey patching
from src.dc_smm.models.housing_renting.horses_c import HRModel

def main():
    try:
        print("Loading configurations...")
        configs = load_configs()

        # Create the base config
        base_cfg = {
            "master": configs["master"],
            "stages": {
                "OWNH": configs["ownh"],
                "OWNC": configs["ownc"],
                "RNTH": configs["renth"],
                "RNTC": configs["rentc"],
                "TENU": configs["tenu"],
            },
            "connections": configs["connections"],
        }

        # Set upper envelope method
        base_cfg["master"]["methods"]["upper_envelope"] = "FUES"

        print("Creating model with 1 period...")
        model = initialize_housing_model(
            base_cfg["master"], base_cfg["stages"], base_cfg["connections"], n_periods=1
        )

        # Set terminal flags for last period's consumption stages
        final_period = model.get_period(len(model.periods_list) - 1)
        final_period.get_stage("OWNC").status_flags["is_terminal"] = True
        final_period.get_stage("RNTC").status_flags["is_terminal"] = True

        # Add debug prints to track UE timing
        # Monkey patch the _solve_egm_loop function to track UE timing
        original_solve_egm_loop = HRModel._solve_egm_loop
        
        def debug_solve_egm_loop(self, *args, **kwargs):
            print("\n==== DEBUGGING _solve_egm_loop ====")
            print(f"Method: {self.methods['upper_envelope']}\n")
            
            result = original_solve_egm_loop(self, *args, **kwargs)
            
            # Print timing info from result if available
            if isinstance(result, dict) and 'timing_info' in result:
                print("\n==== TIMING INFO FROM _solve_egm_loop ====")
                for key, value in result['timing_info'].items():
                    print(f"{key}: {value}")
            
            if isinstance(result, dict) and 'refined_grids' in result:
                refined_grids = result['refined_grids']
                for i_y in range(self.n_y):
                    for i_h in range(self.n_h):
                        key = (i_y, i_h)
                        if key in refined_grids:
                            timing = refined_grids[key].get('timing', {})
                            print(f"\nRefined grid timing for y={i_y}, h={i_h}:")
                            for timing_key, timing_value in timing.items():
                                print(f"  {timing_key}: {timing_value}")
            
            return result
        
        # Apply the monkey patch
        HRModel._solve_egm_loop = debug_solve_egm_loop

        # Also monkey patch the solve_stage function to see if UE time is being captured
        original_solve_stage = solve_stage
        
        def debug_solve_stage(stage, solve_type, verbose=False, **kwargs):
            print(f"\n==== DEBUGGING solve_stage: {stage.name} ({solve_type}) ====")
            
            result = original_solve_stage(stage, solve_type, verbose, **kwargs)
            
            # Check if we have UE time in the results
            if isinstance(result, dict) and "dcsn" in result:
                dcsn_data = result["dcsn"]
                print(f"\nDcsn data keys: {list(dcsn_data.keys()) if isinstance(dcsn_data, dict) else 'Not a dict'}")
                
                if isinstance(dcsn_data, dict) and "timing" in dcsn_data:
                    print("\n==== TIMING INFO FROM solve_stage ====")
                    print(f"UE time: {dcsn_data['timing'].get('ue_time_avg', 'Not found')}")
                    for key, value in dcsn_data['timing'].items():
                        print(f"{key}: {value}")
            
            return result
        
        # Apply the solve_stage patch
        import examples.housing_renting.whisperer
        examples.housing_renting.whisperer.solve_stage = debug_solve_stage

        # Run time iteration with verbose timing
        print("\nRunning time iteration with verbose timing...")
        start_time = time.time()
        run_time_iteration(model, verbose=True, verbose_timings=True)
        end_time = time.time()

        print(f"\nTotal execution time: {end_time - start_time:.4f} seconds")

        # Add extra debug information
        print("\nDebug information:")
        try:
            print("1. Upper envelope method:", model.methods["upper_envelope"])
            print("\n2. Check stage dcsn_data for timing info:")
            
            # Examine one stage in detail to see if we can detect the UE timing
            period = model.get_period(0)
            ownc_stage = period.get_stage("OWNC")
            
            # Check if any of the operators has timing information
            if hasattr(ownc_stage, 'cntn_dcsn_ops'):
                print("\nExamining OWNC stage operators:")
                for op_name, operator in ownc_stage.cntn_dcsn_ops.items():
                    print(f"Operator {op_name}")
                    if hasattr(operator, 'timing'):
                        print(f"  Has timing attribute: {operator.timing}")
                    if hasattr(operator, 'ue_time'):
                        print(f"  Has ue_time attribute: {operator.ue_time}")
            
            # Try to access the solution data
            if hasattr(ownc_stage, 'dcsn') and hasattr(ownc_stage.dcsn, 'sol'):
                dcsn_sol = ownc_stage.dcsn.sol
                print(f"\nDcsn solution keys: {list(dcsn_sol.keys()) if isinstance(dcsn_sol, dict) else 'Not a dict'}")
                
                if isinstance(dcsn_sol, dict) and 'timing' in dcsn_sol:
                    print("Timing information found in dcsn.sol:")
                    for key, value in dcsn_sol['timing'].items():
                        print(f"  {key}: {value}")
            
            # Try to extract the timing information from any cached results
            if hasattr(examples.housing_renting.whisperer, '_timing_cache'):
                print("\nTiming cache found in whisperer:")
                for key, value in examples.housing_renting.whisperer._timing_cache.items():
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"Error during debug: {e}")
    except Exception as e:
        print(f"\nDebug information:\n{e}")

if __name__ == "__main__":
    main() 