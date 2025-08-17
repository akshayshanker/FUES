"""
Housing model with renting - external whisperer

This module provides the external solver for the Housing model with renting, including:
1. build_operators: Associates operator factories with movers
2. solve_stage: Solves a single stage using backward iteration
3. run_time_iteration: Solves multiple time periods by operating on an existing model circuit (backward induction)

The model includes five stages:
- TENU: Tenure choice stage (own vs. rent)
- OWNH: Owner housing choice stage
- OWNC: Owner consumption choice stage
- RNTH: Renter housing choice stage
- RNTC: Renter consumption choice stage
"""

import numpy as np
import time  # Add time module for timing measurements
import gc  # For garbage collection after freeing memory

# Import operator factories directly from their modules
from dc_smm.models.housing_renting.horses_h import F_shocks_dcsn_to_arvl, F_h_cntn_to_dcsn_owner, F_h_cntn_to_dcsn_renter
from dc_smm.models.housing_renting.horses_c import (
    F_ownc_cntn_to_dcsn, F_rntc_cntn_to_dcsn, F_ownc_cntn_to_dcsn_gpu,
    F_rntc_cntn_to_dcsn_gpu
)
from dc_smm.models.housing_renting.horses_common import F_id
from dc_smm.models.housing_renting.horses_t import F_t_cntn_to_dcsn
from dynx.stagecraft.solmaker import Solution
# MPI utilities are now handled by circuit_runner

def build_operators(stage, use_mpi=False, comm=None):
    """Build operator mappings for a given stage.
    
    Parameters
    ----------
    stage : Stage
        The stage to build operators for
    use_mpi : bool, optional
        Whether to use MPI parallelization
    comm : MPI communicator, optional
        MPI communicator (if use_mpi=True)
        
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
        operators["cntn_to_dcsn"] = F_h_cntn_to_dcsn_owner(stage.cntn_to_dcsn, use_mpi=use_mpi, comm=comm)
        operators["dcsn_to_arvl"] = F_id(stage.dcsn_to_arvl)

    elif "OWNC" in stage.name:
        # Owner consumption stage
        if stage.model.methods.get("compute") == "GPU":
            operators["cntn_to_dcsn"] = F_ownc_cntn_to_dcsn_gpu(stage.cntn_to_dcsn)
        else:
            operators["cntn_to_dcsn"] = F_ownc_cntn_to_dcsn(stage.cntn_to_dcsn, use_mpi=use_mpi, comm=comm)
        operators["dcsn_to_arvl"] = F_id(stage.dcsn_to_arvl)

    elif "RNTH" in stage.name:
        # Renter housing choice stage

        operators["cntn_to_dcsn"] = F_h_cntn_to_dcsn_renter(stage.cntn_to_dcsn, use_mpi=use_mpi, comm=comm)
        operators["dcsn_to_arvl"] = F_id(stage.dcsn_to_arvl)

    elif "RNTC" in stage.name:
        # Renter consumption stage
        if stage.model.methods.get("compute") == "GPU":
            operators["cntn_to_dcsn"] = F_rntc_cntn_to_dcsn_gpu(stage.cntn_to_dcsn)
        else:
            operators["cntn_to_dcsn"] = F_rntc_cntn_to_dcsn(stage.cntn_to_dcsn, use_mpi=use_mpi, comm=comm)
        operators["dcsn_to_arvl"] = F_id(stage.dcsn_to_arvl)
    
    elif "TENU" in stage.name:
        # Tenure stage with branching
        operators["dcsn_to_arvl"] = F_shocks_dcsn_to_arvl(stage.dcsn_to_arvl)
        operators["cntn_to_dcsn"] = F_t_cntn_to_dcsn(stage.cntn_to_dcsn)
        
    else:
        raise ValueError(f"Unknown stage type: {stage.name}")
    
    return operators


def build_operators_for_circuit(model_circuit, use_mpi=False, comm=None):
    """Build operators for all stages in a model circuit.
    
    Parameters
    ---------- 
    model_circuit : ModelCircuit
        The model circuit containing all stages
    use_mpi : bool, optional
        Whether to use MPI parallelization
    comm : MPI communicator, optional
        MPI communicator (if use_mpi=True)
    """
    # Store MPI parameters on model circuit for run_time_iteration access
    model_circuit._use_mpi = use_mpi
    model_circuit._comm = comm
    
    # Store MPI parameters on stages for solve_stage access, but don't build operators here
    # to avoid double build (they'll be built and cached on first use in solve_stage)
    for period_idx in range(len(model_circuit.periods_list)):
        period = model_circuit.get_period(period_idx)
        for stage_name, stage in period.stages.items():
            # Store MPI parameters on the stage for use during solving
            stage._use_mpi = use_mpi
            stage._comm = comm


def solve_stage(stage, max_iter=None, tol=None, verbose=False, use_mpi=False, comm=None):
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
    use_mpi : bool, optional
        Whether to use MPI parallelization
    comm : MPI communicator, optional
        MPI communicator (if use_mpi=True)
        
    Returns
    -------
    Stage
        The solved stage
    dict
        Timing information for stage operations
    """
    start_time = time.time()
    
    # Get MPI parameters from stage if they were attached previously
    use_mpi = getattr(stage, '_use_mpi', use_mpi)
    comm = getattr(stage, '_comm', comm)

    # Early-exit on workers **only when the stage is NOT an MPI stage**
    # Get MPI parameters from stage if not provided
    if use_mpi is False and hasattr(stage, '_use_mpi'):
        use_mpi = stage._use_mpi
    if comm is None and hasattr(stage, '_comm'):
        comm = stage._comm
    
    # -------------------------------------------------------------
    # 0.  Check if this is an MPI stage (workers exit early for non-MPI stages)
    # -------------------------------------------------------------
    needs_mpi = stage.model.methods.get("compute", "MPI") == "MPI"

    # Early-exit on workers **only when the stage is NOT an MPI stage**
    if use_mpi and comm is not None and not needs_mpi and comm.rank != 0:
        return stage, {
            "stage_name": stage.name,
            "total_time": 0.0,
            "cntn_to_dcsn_time": 0.0,
            "dcsn_to_arvl_time": 0.0,
            "ue_time": 0.0,
            "is_terminal": stage.status_flags.get("is_terminal", False),
        }
    
    
    if verbose and (not use_mpi or comm is None or comm.rank == 0):
        print(f"Solving stage: {stage.name}")
    
    # Build operators for this stage (cached to avoid double build)
    stage._ops = build_operators(stage, use_mpi=use_mpi, comm=comm)
    operators = stage._ops
    
    # Check if this is a terminal stage
    is_terminal = stage.status_flags.get("is_terminal", False)
    
    # Apply backward operators once in sequence
    if verbose and (not use_mpi or comm is None or comm.rank == 0):
        print(f"  Applying backward operators...")
    
    # Step 1: Continuation to decision transformation
    cntn_to_dcsn_start = time.time()

    cntn_data = stage.cntn.sol
    
    ''' 
    ## TODO: the step below under if TENU is probably redundant tenure stage cntn
    ## is already a dict in the format we need for dcsn
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
    '''
    
    # Apply continuation to decision operator
    dcsn_data = operators["cntn_to_dcsn"](cntn_data)
    cntn_to_dcsn_time = time.time() - cntn_to_dcsn_start
    
    # Clear cntn_data if it's no longer needed (especially for GPU stages)
    if "OWNC" in stage.name or "RNTC" in stage.name:
        # For consumption stages, cntn_data can be large (value functions)
        # We don't need it after the operator has consumed it
        del cntn_data
        gc.collect()
    
    # Extract UE time if available in the result (from EGM computation)
    ue_time = 0
    if isinstance(dcsn_data, Solution):
        # Handle Solution object
        if "ue_time_avg" in dcsn_data.timing:
            ue_time = dcsn_data.timing["ue_time_avg"]
        stage.dcsn.sol = dcsn_data
    else:
        # Legacy dict handling
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
    
    if verbose and (not use_mpi or comm is None or comm.rank == 0):
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

def initialize_terminal_values(stage, verbose=False, use_mpi=False, comm=None):
    """Initialize terminal continuation values analytically.
    
    Only consumption stages (OWNC, RNTC) need genuine terminal values with CRRA 
    analytic closure. Housing and tenure stages receive their continuation objects
    from the solved consumption movers one step later.
    
    Parameters
    ----------
    stage : Stage
        The stage to initialize
    verbose : bool, optional
        Whether to print verbose output, default False
    use_mpi : bool, optional
        Whether MPI is being used
    comm : MPI communicator, optional
        MPI communicator
    """
    if verbose and (not use_mpi or comm is None or comm.rank == 0):
        print(f"Initializing terminal values for {stage.name}")

    
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
        
        # Attach to continuation perch as Solution object
        sol = Solution()
        sol.vlu = vlu_cntn
        sol.lambda_ = lambda_cntn
        sol.Q = Q_cntn
        stage.cntn.sol = sol
    
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
        
        # Attach to continuation perch as Solution object
        sol = Solution()
        sol.vlu = vlu_cntn
        sol.lambda_ = lambda_cntn
        sol.Q = Q_cntn
        stage.cntn.sol = sol
    


def run_time_iteration(model_circuit, n_periods=None, verbose=False,verbose_timings =False, recorder=None, free_memory=True, periods_to_keep=None):
    """Run time iteration by solving all periods in a pre-created model circuit.
    
    This function systematically solves all five stages of the housing-rental model
    in the correct backward order, following the natural flow of decision making.
    
    All stages now use rank-aware operators that return lightweight stubs on workers,
    eliminating the need for heavy .sol object synchronization.
    
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
    free_memory : bool, optional
        Whether to free solution memory after data is passed to next stage.
        Default True. Set to False if you need to save the full model.
    periods_to_keep : list of int, optional
        List of period indices to keep all data for (e.g., [0, 1] for Euler error).
        If None, keeps data based on free_memory flag only.
        
    Returns
    -------
    list
        List of stage dictionaries for each period
    """
    total_start_time = time.time()
    
    # Get MPI parameters from model circuit if available  
    use_mpi = getattr(model_circuit, '_use_mpi', False)
    comm = getattr(model_circuit, '_comm', None)
    
    # Set memory management flag on model circuit
    model_circuit.free_after_use = free_memory
    
    # Default periods to keep if using free_memory
    if free_memory and periods_to_keep is None:
        # By default, keep periods 0 and 1 for Euler error
        periods_to_keep = [0, 1]
    
    # Automatically calculate the number of periods if not provided
    if n_periods is None:
        n_periods = len(model_circuit.periods_list)
        
    if verbose and (not use_mpi or comm is None or comm.rank == 0):
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
    #print(f"final_period: {final_period}")
    #print(f"final_period: {final_period}")
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
    if comm is None or comm.rank == 0:
        initialize_terminal_values(final_ownc, verbose=verbose, use_mpi=use_mpi, comm=comm)
        initialize_terminal_values(final_rntc, verbose=verbose, use_mpi=use_mpi, comm=comm)

    if verbose and (not use_mpi or comm is None or comm.rank == 0):
        print("terminal values initialized")
    
    # All ranks need to be aligned after terminal initialization
    _barrier_if_mpi(use_mpi, comm)


    #assert final_ownc.cntn.sol.vlu is not None, "OWNC terminal cntn.sol.vlu missing!"
    #assert final_rntc.cntn.sol is not None, "RNTC terminal cntn.sol missing!"

    
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

        #print(f"period_idx: {period_idx}")
        #print(f"Is terminal: {is_terminal_period}")
        
        if verbose and (not use_mpi or comm is None or comm.rank == 0):
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
        # Skip already-solved VFI_HDGRID stages (from baseline preload)

        ownc_stage, ownc_timing = solve_stage(ownc_stage, verbose=verbose, use_mpi=use_mpi, comm=comm)
        _sync_perch_solutions(ownc_stage, use_mpi, comm)
        stage_timings.append(ownc_timing)
        period_ue_time += ownc_timing.get("ue_time", 0)
        
        rntc_stage, rntc_timing = solve_stage(rntc_stage, verbose=verbose, use_mpi=use_mpi, comm=comm)
        _sync_perch_solutions(rntc_stage, use_mpi, comm)
        stage_timings.append(rntc_timing)
        period_ue_time += rntc_timing.get("ue_time", 0)
        
        # -------------------------------------------------------------------------------
        # 2. Connect consumption stages to housing stages
        # -------------------------------------------------------------------------------
        # OWNC.arvl → OWNH.cntn
        ownh_stage.cntn.sol = ownc_stage.arvl.sol
        # Free memory if not needed for saving
        # (For periods not in keep list, we'll free everything at end of loop)
        if getattr(model_circuit, 'free_after_use', True) and (periods_to_keep is None or period_idx in periods_to_keep):
            ownc_stage.arvl.sol = None
            # Keep dcsn.sol for Euler error calculation (has policy functions)
            # Keep cntn.sol as it might be needed for next period
        
        # RNTC.arvl → RNTH.cntn
        rnth_stage.cntn.sol = rntc_stage.arvl.sol
        # Free memory if not needed for saving
        # (For periods not in keep list, we'll free everything at end of loop)
        if getattr(model_circuit, 'free_after_use', True) and (periods_to_keep is None or period_idx in periods_to_keep):
            rntc_stage.arvl.sol = None
            # Keep dcsn.sol for Euler error calculation (has policy functions)
            # Keep cntn.sol as it might be needed for next period
        
        # -------------------------------------------------------------------------------
        # 3. Solve housing choice stages (OWNH and RNTH)
        # -------------------------------------------------------------------------------
        ownh_stage, ownh_timing = solve_stage(ownh_stage, verbose=verbose, use_mpi=use_mpi, comm=comm)
        _sync_perch_solutions(ownh_stage, use_mpi, comm)
        stage_timings.append(ownh_timing)
        period_ue_time += ownh_timing.get("ue_time", 0)
        
        rnth_stage, rnth_timing = solve_stage(rnth_stage, verbose=verbose, use_mpi=use_mpi, comm=comm)
        _sync_perch_solutions(rnth_stage, use_mpi, comm)
        stage_timings.append(rnth_timing)
        period_ue_time += rnth_timing.get("ue_time", 0)
        
        # -------------------------------------------------------------------------------
        # 4. Connect housing stages to tenure choice stage
        # -------------------------------------------------------------------------------
        # OWNH.arvl → TENU.cntn_own
        tenu_stage.cntn.sol["from_owner"] = ownh_stage.arvl.sol
        # Free memory if not needed for saving
        # (For periods not in keep list, we'll free everything at end of loop)
        if getattr(model_circuit, 'free_after_use', True) and (periods_to_keep is None or period_idx in periods_to_keep):
            ownh_stage.arvl.sol = None
            # Keep dcsn.sol for Euler error calculation (has policy functions)
            ownh_stage.cntn.sol = None  # Already consumed
        
        # RNTH.arvl → TENU.cntn_rent
        tenu_stage.cntn.sol["from_renter"] = rnth_stage.arvl.sol
        # Free memory if not needed for saving
        # (For periods not in keep list, we'll free everything at end of loop)
        if getattr(model_circuit, 'free_after_use', True) and (periods_to_keep is None or period_idx in periods_to_keep):
            rnth_stage.arvl.sol = None
            # Keep dcsn.sol for Euler error calculation (has policy functions)
            rnth_stage.cntn.sol = None  # Already consumed
        
        # -------------------------------------------------------------------------------
        # 5. Solve the tenure choice stage (TENU)
        # -------------------------------------------------------------------------------
        tenu_stage, tenu_timing = solve_stage(tenu_stage, verbose=verbose, use_mpi=use_mpi, comm=comm)
        _sync_perch_solutions(tenu_stage, use_mpi, comm)
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
            
            # Free TENU memory after period connection if not needed for saving
            if getattr(model_circuit, 'free_after_use', True):
                # Only keep arvl.sol if it's the last solved period (period_idx == 0)
                if period_idx > 1:  # Not the second-to-last period
                    tenu_stage.arvl.sol = None
                # Keep dcsn.sol for Euler error calculation (has policy functions)
                tenu_stage.cntn.sol = None  # Already consumed
        
        # Store this period's stages in dictionary format for the return value
        stages_dict = {
            "TENU": tenu_stage,
            "OWNH": ownh_stage,
            "OWNC": ownc_stage,
            "RNTH": rnth_stage,
            "RNTC": rntc_stage
        }
            
        all_stages_by_period.append(stages_dict)
        
        # Free all memory from this period if it's not in the keep list
        if free_memory and periods_to_keep is not None and period_idx not in periods_to_keep:
            if verbose and (not use_mpi or comm is None or comm.rank == 0):
                print(f"  Freeing all memory from period {period_idx} (not in keep list: {periods_to_keep})")
            _free_period_memory(period)
        
        # Even for periods we keep, clear Q and lambda arrays which are rarely needed
        elif free_memory and period_idx in periods_to_keep:
            for stage_name in ["OWNC", "RNTC"]:
                try:
                    stage = period.get_stage(stage_name)
                    if stage.dcsn.sol and hasattr(stage.dcsn.sol, 'Q'):
                        stage.dcsn.sol.Q = None
                    if stage.dcsn.sol and hasattr(stage.dcsn.sol, 'lambda_'):
                        stage.dcsn.sol.lambda_ = None
                except:
                    pass
            gc.collect()
        
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
        if period_idx < n_periods - 2:
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
    
    # CRITICAL FIX #6: Guard verbose timing computation to save time when not needed
    if verbose_timings and (not use_mpi or comm is None or comm.rank == 0):
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

# Helper for rank alignment when needed
def _barrier_if_mpi(use_mpi: bool, comm):
    """Simple barrier for rank alignment when needed."""
    if use_mpi and comm is not None:
        comm.Barrier()


# ------------------------------------------------------------------
#  Synchronise perch .sol objects after a non-MPI stage was solved
# ------------------------------------------------------------------
def _sync_perch_solutions(stage, use_mpi, comm):
    """
    Broadcast stage.cntn/dcsn/arvl .sol from rank-0 to all other ranks
    *only* when the stage was executed on a single rank.

    Called *immediately after* solve_stage() for owner/renter consumption
    and (potentially) housing/tenure stages that run in serial mode.
    """
    if not (use_mpi and comm is not None and comm.size > 1):
        return  # nothing to do in serial mode

    for perch_name in ("cntn", "dcsn", "arvl"):
        perch = getattr(stage, perch_name, None)
        if perch is not None:
            perch.sol = comm.bcast(perch.sol if comm.rank == 0 else None,
                                   root=0)
    comm.Barrier()                    # keep ranks aligned


def _free_period_memory(period):
    """
    Completely free all solution data from a period's stages.
    Used for periods that are not needed for metrics.
    """
    for stage_name in ["TENU", "OWNH", "OWNC", "RNTH", "RNTC"]:
        try:
            stage = period.get_stage(stage_name)
            # Free all perch solution data
            for perch_name in ["arvl", "dcsn", "cntn"]:
                perch = getattr(stage, perch_name, None)
                if perch is not None and hasattr(perch, 'sol'):
                    # If it's a Solution object, clear its internal arrays too
                    if hasattr(perch.sol, 'vlu'):
                        perch.sol.vlu = None
                    if hasattr(perch.sol, 'lambda_'):
                        perch.sol.lambda_ = None
                    if hasattr(perch.sol, 'Q'):
                        perch.sol.Q = None
                    if hasattr(perch.sol, 'policy'):
                        perch.sol.policy = {}
                    perch.sol = None
            
            # Also clear any cached operators
            if hasattr(stage, '_ops'):
                stage._ops = None
        except:
            pass  # Stage might not exist
    
    # Force garbage collection
    gc.collect()