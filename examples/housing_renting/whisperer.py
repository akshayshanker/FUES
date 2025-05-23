"""
Housing model with renting - external whisperer

This module provides the external solver for the Housing model with renting, including:
1. build_operators: Associates operator factories with movers
2. solve_stage: Solves a single stage using backward iteration
3. run_time_iteration: Solves multiple time periods by operating on an existing model circuit

The model includes five stages:
- TENU: Tenure choice stage (own vs. rent)
- OWNH: Owner housing choice stage
- OWNC: Owner consumption choice stage
- RNTH: Renter housing choice stage
- RNTC: Renter consumption choice stage
"""

import os
import sys
import numpy as np
import copy
import time  # Add time module for timing measurements

# Import operator factories directly from their modules
from dc_smm.models.housing_renting.horses_h import F_shocks_dcsn_to_arvl, F_h_cntn_to_dcsn_owner, F_h_cntn_to_dcsn_renter
from dc_smm.models.housing_renting.horses_c import F_ownc_cntn_to_dcsn, F_ownc_dcsn_to_cntn
from dc_smm.models.housing_renting.horses_common import F_id
from dc_smm.models.housing_renting.horses_t import F_t_cntn_to_dcsn

def build_operators(stage):
    """Build operator mappings for a given stage.
    
    Parameters
    ----------
    stage : Stage
        The stage to build operators for
        
    Returns
    -------
    dict
        Dictionary mapping mover names to operator functions
    """
    # Initialize empty operator dictionary
    operators = {}
    
    # Check stage type to determine which operators to use
    if "OWNH" in stage.name:
        # Owner housing choice stage
        operators["cntn_to_dcsn"] = F_h_cntn_to_dcsn_owner(stage.cntn_to_dcsn)
        operators["dcsn_to_arvl"] = F_id(stage.dcsn_to_arvl)

    elif "OWNC" in stage.name:
        # Owner consumption stage
        operators["cntn_to_dcsn"] = F_ownc_cntn_to_dcsn(stage.cntn_to_dcsn)
        operators["dcsn_to_arvl"] = F_id(stage.dcsn_to_arvl)

    elif "RNTH" in stage.name:
        # Renter housing choice stage
        operators["cntn_to_dcsn"] = F_h_cntn_to_dcsn_renter(stage.cntn_to_dcsn)
        operators["dcsn_to_arvl"] = F_id(stage.dcsn_to_arvl)

    elif "RNTC" in stage.name:
        # Renter consumption stage
        operators["cntn_to_dcsn"] = F_ownc_cntn_to_dcsn(stage.cntn_to_dcsn)
        operators["dcsn_to_arvl"] = F_id(stage.dcsn_to_arvl)
    
    elif "TENU" in stage.name:
        # Tenure stage with branching
        operators["dcsn_to_arvl"] = F_shocks_dcsn_to_arvl(stage.dcsn_to_arvl)
        operators["cntn_to_dcsn"] = F_t_cntn_to_dcsn(stage.cntn_to_dcsn)
        
    else:
        raise ValueError(f"Unknown stage type: {stage.name}")
    
    return operators

def solve_stage(stage, max_iter=None, tol=None, verbose=False):
    """Solve a stage by applying backward operators once.
    
    Parameters
    ----------
    stage : Stage
        The stage to solve
    max_iter : int, optional
        Not used in this non-iterative implementation
    tol : float, optional
        Not used in this non-iterative implementation
    verbose : bool, optional
        Whether to print verbose output, default False
        
    Returns
    -------
    Stage
        The solved stage
    dict
        Timing information for stage operations
    """
    start_time = time.time()
    
    if verbose:
        print(f"Solving stage: {stage.name}")
    
    # Build operators for this stage
    operators = build_operators(stage)
    
    # Check if this is a terminal stage
    is_terminal = stage.status_flags.get("is_terminal", False)
    
    # If this is the final period, initialize analytically
    if is_terminal:
        # Initialize terminal continuation values (CRRA utility)
        initialize_terminal_values(stage, verbose=verbose)
    
    # Apply backward operators once in sequence
    if verbose:
        print(f"  Applying backward operators...")
    
    # Step 1: Continuation to decision transformation
    cntn_to_dcsn_start = time.time()
    
    if "TENU" in stage.name:
        # Special handling for tenure choice with branched continuations
        # Get solutions from both continuation perches
        # Create an empty dictionary if not initialized
        if not hasattr(stage, "cntn"):
            raise ValueError(f"TENU stage {stage.name} is missing cntn perch structure")
            
        if not hasattr(stage.cntn, "sol") or stage.cntn.sol is None:
            stage.cntn.sol = {}
            
        # Initialize branches if they don't exist
        if "from_owner" not in stage.cntn.sol:
            stage.cntn.sol["from_owner"] = None
        if "from_renter" not in stage.cntn.sol:
            stage.cntn.sol["from_renter"] = None
            
        # Get the data from continuation perches
        own_data = stage.cntn.sol["from_owner"]
        rent_data = stage.cntn.sol["from_renter"]
        
        # Create combined continuation data with branch keys
        cntn_data = {
            "from_owner": own_data,
            "from_renter": rent_data
        }
    else:
        # Regular stage with single continuation
        cntn_data = stage.cntn.sol
    
    # Apply continuation to decision operator
    dcsn_data = operators["cntn_to_dcsn"](cntn_data)
    cntn_to_dcsn_time = time.time() - cntn_to_dcsn_start
    
    # Extract UE time if available in the result (from EGM computation)
    ue_time = 0
    if isinstance(dcsn_data, dict) and "timing" in dcsn_data:
        ue_time = dcsn_data["timing"].get("ue_time_avg", 0)
    
    # Store the solution data (excluding timing info)
    solution_keys = [k for k in dcsn_data.keys() if k != "timing"]
    stage.dcsn.sol = {k: dcsn_data[k] for k in solution_keys}
    
    # Step 2: Decision to arrival transformation
    dcsn_to_arvl_start = time.time()
    arvl_data = operators["dcsn_to_arvl"](stage.dcsn.sol)
    stage.arvl.sol = arvl_data
    dcsn_to_arvl_time = time.time() - dcsn_to_arvl_start
    
    # Mark the stage as solved
    stage.status_flags["solved"] = True
    total_time = time.time() - start_time
    
    if verbose:
        print(f"  Stage {stage.name} solved with backward pass.")
    
    # Return timing information, including terminal flag
    timing_info = {
        "stage_name": stage.name,
        "total_time": total_time,
        "cntn_to_dcsn_time": cntn_to_dcsn_time,
        "dcsn_to_arvl_time": dcsn_to_arvl_time,
        "ue_time": ue_time,
        "is_terminal": is_terminal
    }
    
    return stage, timing_info

def initialize_terminal_values(stage, verbose=False):
    """Initialize terminal continuation values analytically.
    
    Parameters
    ----------
    stage : Stage
        The stage to initialize
    verbose : bool, optional
        Whether to print verbose output, default False
    """
    if verbose:
        print(f"Initializing terminal values for {stage.name}")
    
    # Extract parameters using attribute-style access
    params = stage.model.param
    
    if "OWNC" in stage.name:
        # For owner consumption stage, initialize CRRA utility of consuming assets
        a_nxt_grid = stage.cntn.grid.a_nxt
        H_nxt_grid = stage.cntn.grid.H_nxt
        
        n_a = len(a_nxt_grid)
        n_H = len(H_nxt_grid)
        n_y = len(stage.cntn.grid.y)  # Number of income states
        
        # Initialize arrays
        vlu_cntn = np.zeros((n_a, n_H, n_y))
        lambda_cntn = np.zeros((n_a, n_H, n_y))
        
        # Use model's utility function for terminal values
        u_func = stage.model.num.functions.owner_utility
        uc_func = stage.model.num.functions.owner_marginal_utility
        
        # For each state
        for i_y in range(n_y):
            for i_h, h_val in enumerate(H_nxt_grid):
                for i_a, a_val in enumerate(a_nxt_grid):
                    # Terminal value is utility from consuming everything
                    vlu_cntn[i_a, i_h, i_y] = u_func(c=a_val, H_nxt=h_val)
                    lambda_cntn[i_a, i_h, i_y] = uc_func(c=a_val, H_nxt=h_val)
        
        # Terminal: Q = v because V_e = 0
        Q_cntn = vlu_cntn.copy()
        
        # Attach to continuation perch
        stage.cntn.sol = {
            "vlu": vlu_cntn,
            "lambda": lambda_cntn,
            "Q": Q_cntn
        }
    
    elif "RNTC" in stage.name:
        # For renter consumption stage, initialize CRRA utility of consuming assets
        a_nxt_grid = stage.cntn.grid.a_nxt
        S_grid = stage.cntn.grid.H_nxt
        
        n_a = len(a_nxt_grid)
        n_S = len(S_grid)
        n_y = len(stage.cntn.grid.y)  # Number of income states
        
        # Initialize arrays
        vlu_cntn = np.zeros((n_a, n_S, n_y))
        lambda_cntn = np.zeros((n_a, n_S, n_y))
        
        # Use model's renter utility function for terminal values
        u_func = stage.model.num.functions.renter_utility
        uc_func = stage.model.num.functions.marginal_utility
        
        # For each state
        for i_y in range(n_y):
            for i_s, s_val in enumerate(S_grid):
                for i_a, a_val in enumerate(a_nxt_grid):
                    # Terminal value is utility from consuming everything
                    vlu_cntn[i_a, i_s, i_y] = u_func(c=a_val, S=s_val)
                    lambda_cntn[i_a, i_s, i_y] = uc_func(c=a_val, S=s_val)
        
        # Terminal: Q = v because V_e = 0
        Q_cntn = vlu_cntn.copy()
        
        # Attach to continuation perch
        stage.cntn.sol = {
            "vlu": vlu_cntn,
            "lambda": lambda_cntn,
            "Q": Q_cntn
        }
    
    elif "OWNH" in stage.name:
        # For owner housing stage, initialize placeholder values
        w_grid = stage.cntn.grid.w_own
        H_nxt_grid = stage.cntn.grid.H_nxt
        
        n_w = len(w_grid)
        n_H = len(H_nxt_grid)
        n_y = len(stage.cntn.grid.y)  # Number of income states
        
        # Initialize with placeholder values
        vlu_cntn = np.zeros((n_w, n_H, n_y))
        lambda_cntn = np.zeros((n_w, n_H, n_y))
        
        # Use utility function for placeholder values
        u_func = stage.model.num.functions.owner_utility
        uc_func = stage.model.num.functions.marginal_utility
        
        for i_y in range(n_y):
            for i_h, h_val in enumerate(H_nxt_grid):
                for i_w, w_val in enumerate(w_grid):
                    # Terminal value is utility from consuming everything
                    vlu_cntn[i_w, i_h, i_y] = u_func(c=w_val, H_nxt=h_val)
                    lambda_cntn[i_w, i_h, i_y] = uc_func(c=w_val, H_nxt=h_val)
        
        # Terminal: Q = v because V_e = 0
        Q_cntn = vlu_cntn.copy()
        
        # Attach to continuation perch
        stage.cntn.sol = {
            "vlu": vlu_cntn,
            "lambda": lambda_cntn,
            "Q": Q_cntn
        }
    
    elif "RNTH" in stage.name:
        # For renter housing stage, initialize placeholder values
        w_grid = stage.cntn.grid.w_rent
        S_grid = stage.cntn.grid.S
        
        n_w = len(w_grid)
        n_S = len(S_grid)
        n_y = len(stage.cntn.grid.y)  # Number of income states
        
        # Initialize with placeholder values
        vlu_cntn = np.zeros((n_w, n_S, n_y))
        lambda_cntn = np.zeros((n_w, n_S, n_y))
        
        # Use renter utility function for placeholder values
        u_func = stage.model.num.functions.renter_utility
        uc_func = stage.model.num.functions.marginal_utility
        
        for i_y in range(n_y):
            for i_s, s_val in enumerate(S_grid):
                for i_w, w_val in enumerate(w_grid):
                    # Terminal value is utility from consuming everything
                    vlu_cntn[i_w, i_s, i_y] = u_func(c=w_val, S=s_val + 0.5)
                    lambda_cntn[i_w, i_s, i_y] = uc_func(c=w_val, S=s_val + 0.5)
        
        # Terminal: Q = v because V_e = 0
        Q_cntn = vlu_cntn.copy()
        
        # Attach to continuation perch
        stage.cntn.sol = {
            "vlu": vlu_cntn,
            "lambda": lambda_cntn,
            "Q": Q_cntn
        }
    
    elif "TENU" in stage.name:
        # For tenure choice stage with branched continuation
        # Initialize continuation with branching structure
        if not hasattr(stage, "cntn"):
            raise ValueError(f"TENU stage {stage.name} is missing cntn perch")
        
        if not hasattr(stage.cntn, "sol") or stage.cntn.sol is None:
            stage.cntn.sol = {}
        
        # Ensure branch keys exist
        if "from_owner" not in stage.cntn.sol:
            # Initialize owner branch
            a_grid = stage.model.num.state_space.cntn_own.grids.a
            H_grid = stage.model.num.state_space.cntn_own.grids.H
            y_grid = stage.model.num.state_space.cntn_own.grids.y
            
            n_a = len(a_grid)
            n_H = len(H_grid)
            n_y = len(y_grid)
            
            # Initialize own path values
            vlu_own = np.zeros((n_a, n_H, n_y))
            lambda_own = np.zeros((n_a, n_H, n_y))
            
            # Use owner utility for placeholder values
            u_func = stage.model.num.functions.owner_utility
            uc_func = stage.model.num.functions.marginal_utility
            
            for i_y in range(n_y):
                for i_h, h_val in enumerate(H_grid):
                    for i_a, a_val in enumerate(a_grid):
                        # Simple placeholder - utility of consuming assets and having housing
                        vlu_own[i_a, i_h, i_y] = u_func(c=a_val, H_nxt=h_val)
                        lambda_own[i_a, i_h, i_y] = uc_func(c=a_val, H_nxt=h_val)
            
            # Terminal: Q = v because V_e = 0
            Q_own = vlu_own.copy()
            
            # Attach to owner branch
            stage.cntn.sol["from_owner"] = {
                "vlu": vlu_own,
                "lambda": lambda_own,
                "Q": Q_own
            }
        
        if "from_renter" not in stage.cntn.sol:
            # Initialize renter branch
            w_grid = stage.model.num.state_space.cntn.grids.w
            y_grid = stage.model.num.state_space.cntn.grids.y
            
            n_w = len(w_grid)
            n_y = len(y_grid)
            
            # Initialize rent path values
            vlu_rent = np.zeros((n_w, n_y))
            lambda_rent = np.zeros((n_w, n_y))
            
            # Use renter utility for placeholder
            u_func = stage.model.num.functions.renter_utility
            uc_func = stage.model.num.functions.marginal_utility
            
            for i_y in range(n_y):
                for i_w, w_val in enumerate(w_grid):
                    # Simple placeholder - utility of consuming wealth and minimal housing
                    vlu_rent[i_w, i_y] = u_func(c=w_val, S=0.1)  # Minimal rental housing
                    lambda_rent[i_w, i_y] = uc_func(c=w_val, S=0.1)
            
            # Terminal: Q = v because V_e = 0
            Q_rent = vlu_rent.copy()
            
            # Attach to renter branch
            stage.cntn.sol["from_renter"] = {
                "vlu": vlu_rent,
                "lambda": lambda_rent,
                "Q": Q_rent
            }

def run_time_iteration(model_circuit, n_periods=None, verbose=False,verbose_timings =False, recorder=None):
    """Run time iteration by solving all periods in a pre-created model circuit.
    
    This function systematically solves all five stages of the housing-rental model
    in the correct backward order, following the natural flow of decision making.
    
    Parameters
    ----------
    model_circuit : ModelCircuit
        Pre-created model circuit with periods
    n_periods : int, optional
        Number of periods in the model circuit. If None, automatically determined
        from the model_circuit.
    verbose : bool, optional
        Whether to print verbose output, default False
    recorder : RunRecorder, optional
        If provided, metrics will be recorded during solving
        
    Returns
    -------
    list
        List of stage dictionaries for each period
    """
    total_start_time = time.time()
    
    # Automatically calculate the number of periods if not provided
    if n_periods is None:
        n_periods = len(model_circuit.periods_list)
        
    if verbose:
        print(f"Running time iteration for {n_periods} periods")
    
    # Collect stages by period for return
    all_stages_by_period = []
    
    # Initialize timing collections
    period_timings = []
    stage_timings = []
    
    # Track times for non-terminal periods
    total_ue_time = 0.0
    total_nonterminal_time = 0.0
    
    # Mark terminal periods - set terminal flags for the final period's consumption stages
    final_period = model_circuit.get_period(n_periods - 1)
    
    # Get all stages from the final period
    final_tenu = final_period.get_stage("TENU")
    final_ownh = final_period.get_stage("OWNH")
    final_ownc = final_period.get_stage("OWNC")
    final_rnth = final_period.get_stage("RNTH")
    final_rntc = final_period.get_stage("RNTC")
    
    # Set terminal flags for consumption stages
    final_ownc.status_flags["is_terminal"] = True
    final_rntc.status_flags["is_terminal"] = True
    
    # Initialize terminal continuation values for the last period's consumption stages
    initialize_terminal_values(final_ownc, verbose=verbose)
    initialize_terminal_values(final_rntc, verbose=verbose)
    
    # Set external mode for all stages
    for period_idx in range(n_periods):
        period = model_circuit.get_period(period_idx)
        for stage_name, stage in period.stages.items():
            stage.model_mode = "external"
    
    # Solve backward from the last period to the first
    for period_idx in reversed(range(n_periods)):
        period_start_time = time.time()
        period = model_circuit.get_period(period_idx)
        
        # Flag whether this is the terminal period (first one we solve)
        is_terminal_period = (period_idx == n_periods - 1)
        
        if verbose:
            print(f"\nSolving period {period_idx+1} of {n_periods}")
        
        # Get the stages for this period
        tenu_stage = period.get_stage("TENU")
        ownh_stage = period.get_stage("OWNH")
        ownc_stage = period.get_stage("OWNC")
        rnth_stage = period.get_stage("RNTH")
        rntc_stage = period.get_stage("RNTC")
        
        period_ue_time = 0.0
        
        # -------------------------------------------------------------------------------
        # Solve stages in proper backward order following CDC dependency structure:
        # 1. First the consumption stages (OWNC and RNTC)
        # -------------------------------------------------------------------------------
        ownc_stage, ownc_timing = solve_stage(ownc_stage, verbose=verbose)
        stage_timings.append(ownc_timing)
        period_ue_time += ownc_timing.get("ue_time", 0)
        
        rntc_stage, rntc_timing = solve_stage(rntc_stage, verbose=verbose)
        stage_timings.append(rntc_timing)
        period_ue_time += rntc_timing.get("ue_time", 0)
        
        # -------------------------------------------------------------------------------
        # 2. Connect consumption stages to housing stages
        # -------------------------------------------------------------------------------
        # OWNC.arvl → OWNH.cntn
        ownh_stage.cntn.sol = ownc_stage.arvl.sol
        
        # RNTC.arvl → RNTH.cntn
        rnth_stage.cntn.sol = rntc_stage.arvl.sol
        
        # -------------------------------------------------------------------------------
        # 3. Solve housing choice stages (OWNH and RNTH)
        # -------------------------------------------------------------------------------
        ownh_stage, ownh_timing = solve_stage(ownh_stage, verbose=verbose)
        stage_timings.append(ownh_timing)
        period_ue_time += ownh_timing.get("ue_time", 0)
        
        rnth_stage, rnth_timing = solve_stage(rnth_stage, verbose=verbose)
        stage_timings.append(rnth_timing)
        period_ue_time += rnth_timing.get("ue_time", 0)
        
        # -------------------------------------------------------------------------------
        # 4. Connect housing stages to tenure choice stage
        # -------------------------------------------------------------------------------
        # OWNH.arvl → TENU.cntn_own
        tenu_stage.cntn.sol["from_owner"] = ownh_stage.arvl.sol
        
        # RNTH.arvl → TENU.cntn_rent
        tenu_stage.cntn.sol["from_renter"] = rnth_stage.arvl.sol
        
        # -------------------------------------------------------------------------------
        # 5. Solve the tenure choice stage (TENU)
        # -------------------------------------------------------------------------------
        tenu_stage, tenu_timing = solve_stage(tenu_stage, verbose=verbose)
        stage_timings.append(tenu_timing)
        period_ue_time += tenu_timing.get("ue_time", 0)
        
        # -------------------------------------------------------------------------------
        # 6. Inter-period connection: If not first period, connect to previous period
        # -------------------------------------------------------------------------------
        if period_idx > 0:
            # Get the previous period's tenure choice stage
            prev_period = model_circuit.get_period(period_idx - 1)
            prev_rentc = prev_period.get_stage("RNTC")
            prev_ownc = prev_period.get_stage("OWNC")
            
            # Connect inter-period: TENU.arvl → previous TENU.cntn
            prev_rentc.cntn.sol = tenu_stage.arvl.sol
            prev_ownc.cntn.sol = tenu_stage.arvl.sol
        
        # Store this period's stages in dictionary format for the return value
        stages_dict = {
            "TENU": tenu_stage,
            "OWNH": ownh_stage,
            "OWNC": ownc_stage,
            "RNTH": rnth_stage,
            "RNTC": rntc_stage
        }
            
        all_stages_by_period.append(stages_dict)
        
        # Record period timing
        period_time = time.time() - period_start_time
        period_data = {
            "period": period_idx + 1,
            "time": period_time,
            "ue_time": period_ue_time,
            "is_terminal": is_terminal_period
        }
        period_timings.append(period_data)
        
        # Track times for non-terminal periods only
        if not is_terminal_period:
            total_ue_time += period_ue_time
            total_nonterminal_time += period_time
    
    # Reverse the list to have periods in correct order (T-n to T-1)
    all_stages_by_period.reverse()
    
    # Calculate total solution time
    total_time = time.time() - total_start_time
    
    # Record metrics if a recorder was provided
    if recorder is not None:
        recorder.add(
            total_solution_time=total_time,
            total_ue_time=total_ue_time,
            total_nonterminal_time=total_nonterminal_time,
            ue_time_percent=((total_ue_time / total_nonterminal_time) * 100) if total_nonterminal_time > 0 else 0,
            period_timings=period_timings,
            stage_timings=stage_timings
        )
    
    # Print timing summary - always show this regardless of verbose setting
    if verbose_timings:
        print("\n----- Solution Time Summary -----")
        print(f"Total solution time: {total_time:.4f} seconds")
        
        # Calculate UE time as percentage of non-terminal time
        if total_nonterminal_time > 0:
            ue_percent = (total_ue_time / total_nonterminal_time) * 100
            print(f"Upper envelope time: {total_ue_time:.4f}s ({ue_percent:.1f}% of non-terminal time)")
        
        print("\nPeriod timings:")
        for timing in reversed(period_timings):
            period_label = f"Period {timing['period']:2d}"
            if timing['is_terminal']:
                period_label += " (terminal)"
            print(f"  {period_label}: {timing['time']:.4f}s")
            if not timing['is_terminal'] and timing['ue_time'] > 0:
                ue_pct = (timing['ue_time'] / timing['time']) * 100
                print(f"    Upper envelope: {timing['ue_time']:.4f}s ({ue_pct:.1f}%)")
        
        print("\nStage timings (slowest first):")
        # Sort stages by total time (descending)
        sorted_stages = sorted(stage_timings, key=lambda x: x["total_time"], reverse=True)
        for i, timing in enumerate(sorted_stages[:5]):  # Show top 5 slowest stages
            print(f"  {i+1}. {timing['stage_name']}: {timing['total_time']:.4f} seconds")
            print(f"     cntn→dcsn: {timing['cntn_to_dcsn_time']:.4f}s, dcsn→arvl: {timing['dcsn_to_arvl_time']:.4f}s")
            if not timing.get("is_terminal", False) and timing.get("ue_time", 0) > 0:
                ue_pct = (timing['ue_time'] / timing['total_time']) * 100
                print(f"     upper envelope: {timing['ue_time']:.4f}s ({ue_pct:.1f}%)")
        
        print("---------------------------------")
    
    return all_stages_by_period