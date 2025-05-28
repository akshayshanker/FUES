import numpy as np
import time  # Add import for timing
from dc_smm.models.housing_renting.horses_common import (
    _safe_interp, egm_preprocess, build_njit_utility, piecewise_gradient, piecewise_gradient_3rd
)  # Use relative import
from numba import njit
from dc_smm.uenvelope.upperenvelope import EGM_UE
from dc_smm.fues.helpers import interp_as
from functools import lru_cache
from quantecon.optimize.scalar_maximization import brent_max

# --- Operator Factory for OWNC Consumption Choice ---
def F_ownc_cntn_to_dcsn(mover):
    """Create operator for ownc_cntn_to_dcsn mover.
    Implements EGM or VFI for consumption choice.
    """
    # Extract model and perches
    model = mover.model
    
    # In StageCraft, Movers have source_name and target_name, not perch objects
    # We need to access the stage to get the perches
    source_name = mover.source_name  # Should be 'cntn'
    target_name = mover.target_name  # Should be 'dcsn'
    
    # Get the stage to access perches
    
    # Get solution method from settings
    method = model.methods.get("solution", "EGM")
    print(method)
    model.stage_name = mover.stage_name
    
    def operator(perch_data):
        vlu_cntn = perch_data["vlu"]
        lambda_cntn = perch_data.get("lambda") # Lambda might not be needed for VFI
        
        # Track total time for this operation
        start_time = time.time()
        
        if method.upper() == "EGM":
            if lambda_cntn is None:
                raise ValueError("lambda_cntn is required for EGM method.")
            policy, vlu_dcsn, lambda_dcsn, ue_time_avg, egm_grids, policy_a, Q_dcsn = (
                _solve_egm_loop(vlu_cntn, lambda_cntn, model)
            )
            # Include UE timing in the result
            timing_info = {
                "ue_time_avg": ue_time_avg,
                "total_time": time.time() - start_time
            }
            # Return policy, value function, and also store the EGM grids
            return {
                "policy": policy,
                "policy_a": policy_a,
                "vlu": vlu_dcsn,
                "lambda": lambda_dcsn,
                "Q": Q_dcsn,
                "timing": timing_info,
                "EGM": egm_grids  # Store both unrefined and refined grids
            }
        elif method.upper() == "VFI":
            policy, vlu_dcsn, lambda_dcsn, ue_time_avg, egm_grids, policy_a, Q_dcsn= _solve_vfi_loop(vlu_cntn, model)
            # No UE time for VFI
            timing_info = {
                "total_time": time.time() - start_time
            }
            return {
                "policy": policy,
                "policy_a": policy_a,
                "vlu": vlu_dcsn,
                "lambda": lambda_dcsn,
                "Q": Q_dcsn,
                "timing": timing_info,
                "EGM": egm_grids  # Store both unrefined and refined grids
            }
        else:
            raise ValueError(
                f"Unknown solution method: {method}. Choose 'EGM' or 'VFI'."
            )
    
    return operator



# --- Private Solver Loop Helpers ---
def _solve_egm_loop(vlu_cntn, lambda_cntn, model):
    """Solves the consumption problem using the EGM loop."""
    # Use grid proxies for direct grid access
    a_nxt_grid = model.num.state_space.cntn.grids.a_nxt
    H_nxt_grid = model.num.state_space.cntn.grids.H_nxt
    w_grid = model.num.state_space.dcsn.grids.w
    c_max = model.settings_dict["c_max"]
    
    # Get functions and parameters
    compiled_funcs = model.num.functions
    beta = model.param.beta
    delta = model.param.delta_pb      # NEW
    Rfree = model.param.r + 1
    
    # Get settings using attribute-style access
    ue_method = model.methods["upper_envelope"]
    m_bar = model.settings_dict["m_bar"]
    lb = model.settings_dict["lb"]
    rfc_radius = model.settings_dict["rfc_radius"]
    rfc_n_iter = model.settings_dict["rfc_n_iter"]
    n_con = model.settings_dict["n_constraint_points"]

    n_w = len(w_grid)
    n_H = vlu_cntn.shape[1]
    n_y = vlu_cntn.shape[2]

    policy = np.empty((n_w, n_H, n_y))
    policy_a = np.empty((n_w, n_H, n_y))
    vlu_dcsn = np.empty((n_w, n_H, n_y))
    Q_dcsn = np.empty((n_w, n_H, n_y))
    lambda_dcsn = np.empty((n_w, n_H, n_y))
    
    # Track total UE time and count for averaging
    total_ue_time = 0.0
    ue_count = 0

    # Store the unrefined and refined grids
    unrefined_grids = {
        'e': {},    # Endogenous grid points (cash-on-hand)
        'Q': {},    # Value function
        'c': {},    # Consumption policy
        'a': {}     # Asset policy
    }
    
    refined_grids = {
        'e': {},    # Refined endogenous grid points
        'Q': {},    # Refined value function
        'c': {},    # Refined consumption policy
        'a': {},    # Refined asset policy
        'lambda': {} # Refined marginal utility
    }

    q_inv_func = compiled_funcs.inv_marginal_utility
    g_ve_h_ind = compiled_funcs.g_ve_h_ind
    
    h_nxt_ind_array = g_ve_h_ind(H_ind = np.arange(n_H))

    if "RNT" in model.stage_name:
        thorn = model.param.thorn
    else:
        thorn = 1
    
    a_nxt_cost = np.zeros((n_w, n_H, n_y))
    # Create a utility function specific to this housing value\
    expr_str = model.math["functions"]["u_func"]["expr"]
    param_vals = {
        "alpha": model.param.alpha,
        "kappa": model.param.kappa,
        "iota" : model.param.iota,
        # add any extra constants referenced in expr_str
    }
    utility_func = build_njit_utility(expr_str, param_vals)

    # ------------------------------------------------------------------
    #  Vectorised pre-computation of c_egm for ALL (w,h,y) points
    # ------------------------------------------------------------------
    # 1.  Scale λ once
    lam_scaled = beta * delta * lambda_cntn * Rfree                     # shape (n_w,n_H_total,n_y)

    # 2.  Map continuous-grid H-indices to decision-grid indices (renters need this)
    lam_sel = np.take(lam_scaled, h_nxt_ind_array, axis=1)      # → (n_w,n_H,n_y)
    vlu_sel = np.take(vlu_cntn , h_nxt_ind_array, axis=1)       # same mapping for value

    # 3.  Broadcast housing values (after thorn) to (1,n_H,1)
    H_bcast = (H_nxt_grid * thorn)[None, :, None]

    # 4.  Evaluate inverse Euler in one NumPy call (NumPy broadcasts internally).
    #     Use the pure-Python NumPy version behind the numba function via `.py_func`.
    # Obtain a broadcasting-friendly version of the inverse-utility.
    # If the stored function is a Numba dispatcher it *does* have .py_func;
    # otherwise it is already a plain Python/Numpy function.
    inv_mu = getattr(compiled_funcs.inv_marginal_utility, "py_func", 
                     compiled_funcs.inv_marginal_utility)

    # Evaluate for the whole (w,h,y) cube in one shot.  This relies on the
    # function being written with NumPy ufuncs so it automatically
    # broadcasts `lam_sel` (n_w,n_H,n_y) with `H_bcast` (1,n_H,1).
    # If the function is purely scalar we could fall back to
    # `np.vectorize`, but most models already express it in NumPy algebra.
    c_egm_all = inv_mu(lam_sel, H_bcast)

    interpf = lambda x, y: interp_as(x, y, w_grid, extrap=True)

    for i_y in range(n_y):
        for i_h in range(n_H):

            # Slice the pre-computed λ, v and c cubes – no recomputation inside the loop
            #lambda_e = lam_sel[:, i_h, i_y]
            vlu_e    = vlu_sel[:, i_h, i_y]
            c_egm    = c_egm_all[:, i_h, i_y]
            H_val    = H_nxt_grid[i_h] * thorn
            m_egm = c_egm + a_nxt_grid
            u_params = {"c": c_egm, "H_nxt": H_val}
            q_egm = compiled_funcs.u_func(**u_params) + delta*beta * vlu_e
            q_nxt_raw = delta*beta * vlu_e

            # Process the EGM solution to ensure grid uniqueness
            if ue_method != "CONSAV":
                m_egm_unique, vlu_v_egm_unique, c_egm_unique, a_nxt_grid_unique = (
                    egm_preprocess(
                        m_egm, q_egm, c_egm, a_nxt_grid, delta*beta, 
                        utility_func, vlu_e, m_bar = m_bar, n_con=n_con,
                        c_max=c_max, h_nxt=H_val
                    )
                )
                #print("not CONSAV")
            else:
                m_egm_unique = m_egm
                vlu_v_egm_unique = q_egm
                c_egm_unique = c_egm
                a_nxt_grid_unique = a_nxt_grid

            # Store unrefined grids (before upper envelope)
            grid_key = f"{i_y}-{i_h}"
            unrefined_grids['e'][grid_key] = m_egm_unique
            unrefined_grids['Q'][grid_key] = vlu_v_egm_unique
            unrefined_grids['c'][grid_key] = c_egm_unique
            unrefined_grids['a'][grid_key] = a_nxt_grid_unique
            def partial_uc(c_vals):
                return compiled_funcs.uc_func(**{
                    "c": c_vals, 
                    "H_nxt": H_nxt_grid[i_h]*thorn
                })

   
            

            # Get upper envelope solution and timing
            refined, _, _ = EGM_UE(
                m_egm_unique, vlu_v_egm_unique, q_nxt_raw, c_egm_unique, 
                a_nxt_grid_unique, w_grid, partial_uc, 
                u_func={"func": utility_func, "args": H_val},
                ue_method=ue_method, m_bar=m_bar, lb=lb,
                rfc_radius=rfc_radius, rfc_n_iter=rfc_n_iter
            )
            
            # Unpack the results from the refined dictionary
            # Prefer *_ref keys, fallback to non-suffixed for broader compatibility
            m_refined = refined["x_dcsn_ref"]
            q_refined = refined["v_dcsn_ref"]
            c_refined = refined["kappa_ref"] # kappa_ref for consumption
            a_refined = refined["x_cntn_ref"] # x_cntn_ref for next period assets
            
            lambda_refined = refined["lambda_ref"]

            if ue_method != "CONSAV":
                
                Q_dcsn[:, i_h, i_y] = interpf(m_refined, q_refined)
                policy[:, i_h, i_y]   = interpf(m_refined, c_refined)

                
                # Calculate decision objective Q_dcsn for present-biased utility
                #uc_today = compiled_funcs.uc_func(**{
                #    "c": policy[:, i_h, i_y],
                #    "H_nxt": H_nxt_grid[i_h]
                #})
                #c_prime = piecewise_gradient(policy[:, i_h, i_y], w_grid, m_bar=m_bar)
                #lambda_dcsn[:, i_h, i_y] = (uc_today + (1-delta)*c_prime*uc_today) /delta
                #vlu_dcsn[:, i_h, i_y] = (Q_dcsn[:, i_h, i_y] + u_func(c=policy[:, i_h, i_y], H_nxt=H_val))/delta

            else:
                Q_dcsn[:, i_h, i_y] = q_refined
                policy[:, i_h, i_y] = c_refined
                policy_a[:, i_h, i_y] = a_refined
                lambda_dcsn[:, i_h, i_y] = lambda_refined

            
            policy[policy<0] = 1e-10

                
            # Calculate decision objective Q_dcsn for present-biased utility
            uc_today = compiled_funcs.uc_func(**{
                "c": policy[:, i_h, i_y],
                "H_nxt": H_val
            })

            c_prime = piecewise_gradient(policy[:, i_h, i_y], w_grid, m_bar=m_bar)
            #print(c_prime)
            #print(policy[:, i_h, i_y])
            lambda_dcsn[:, i_h, i_y] = (uc_today - (1-delta)*c_prime*uc_today) /delta
            vlu_dcsn[:, i_h, i_y] = (Q_dcsn[:, i_h, i_y] - (1-delta)*compiled_funcs.u_func(c=policy[:, i_h, i_y], H_nxt=H_val))/delta

            #lambda_dcsn[lambda_dcsn<0] = 1e-10

            ue_time = refined.get("ue_time", 0.0)
            
            # Store refined grids (after upper envelope) 
            # using consistently retrieved variables
            refined_grids['e'][grid_key] = m_refined # cash-on-hand / endogenous grid
            refined_grids['Q'][grid_key] = q_refined # value function
            refined_grids['c'][grid_key] = c_refined # consumption policy
            refined_grids['a'][grid_key] = a_refined # asset policy
            refined_grids['lambda'][grid_key] = lambda_refined
                            
            # Track upper envelope time
            total_ue_time += ue_time
            ue_count += 1
                
    # Calculate average UE time
    avg_ue_time = total_ue_time / max(ue_count, 1)
    
    # Include the grid data in the returned tuple
    egm_grids = {
        'unrefined': unrefined_grids,
        'refined': refined_grids
    }
    
    return policy, vlu_dcsn, lambda_dcsn, avg_ue_time, egm_grids, policy_a, Q_dcsn


@njit
def bellman_obj(a_nxt, w_val, H_val, beta, delta,
                a_grid, V_slice, u_func):
    c = w_val - a_nxt
    if c <= 0.0:
        return -np.inf
    # ▸ remove arbitrary 0.1 floor
    # if a_nxt <= 0.1:
    #     return -np.inf
    V_nxt = interp_as(a_grid, V_slice, np.array([a_nxt]))[0]
    return u_func(c, H_val) + beta * delta * V_nxt

@njit
def _solve_vfi_numba(V_next, w_grid, a_grid, H_grid,
                     beta, delta, m_bar,
                     u_func, h_nxt_ind_array, thorn):
    """
    Fully-compiled Laibson VFI.
    Returns:
        policy_c   : c*(w,h,y)
        policy_a   : a'*(w,h,y)
        Q_dcsn     : u(c)+βδV_next
        V_cntn     : continuation value
        lambda_cntn: continuation marginal value
    """
    n_W  = w_grid.size
    n_A, n_H, n_Y = V_next.shape

    policy_c   = np.empty((n_W, n_H, n_Y))
    policy_a   = np.empty_like(policy_c)
    Q_dcsn     = np.empty_like(policy_c)
    V_cntn     = np.empty_like(policy_c)
    lambda_cntn= np.empty_like(policy_c)
    
    #h_nxt_ind_array = g_ve_h_ind(H_ind = np.arange(n_H))

    for h in range(n_H):
        H_val = H_grid[h]*thorn
        for y in range(n_Y):
            h_nxt_ind = h_nxt_ind_array[h]
            V_slice = V_next[:, h_nxt_ind, y]                # contiguous 1-D

            for iw in range(n_W):
                w_val  = w_grid[iw]
                a_low  = a_grid[0]
                a_high = min(w_val + 1e-1, a_grid[-1])
                


                a_star, Q_star, _ = brent_max(
                    bellman_obj,
                    a_low, a_high,
                    args=(w_val, H_val, beta, delta,
                          a_grid, V_slice, u_func),
                    xtol=1e-12
                )
                c_star            = w_val - a_star

                if np.isinf(Q_star):
                    c_star = 1e-10
                    Q_star = -10
                    a_star = 1e-10

                policy_c[iw, h, y] = c_star
                policy_a[iw, h, y] = a_star
                Q_dcsn[iw, h, y]   = Q_star

  


            # ---- continuation value + λ --------------------------------
            c_prime = piecewise_gradient(policy_c[:, h, y], w_grid, m_bar)

            for iw in range(n_W):
                c_now  = policy_c[iw, h, y]
                uc_now = 1/c_now
                V_cntn[iw, h, y] = (Q_dcsn[iw, h, y]
                                    - (1.0 - delta)*u_func(c_now, H_val)) / delta
                lambda_cntn[iw, h, y] = (uc_now +
                                          (1.0 - delta)*c_prime[iw]*uc_now) / delta
                
                #print(V_cntn[iw, h, y][np.isinf(V_cntn[iw, h, y])])

    return policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn

from numba import njit, prange
import numpy as np

# ------------------------------------------------------------------
# Dense-grid replacement for Brent search
# ------------------------------------------------------------------

#@njit
@njit
def _solve_vfi_numba_grid(V_next, w_grid, a_grid, H_grid,
                        beta, delta, m_bar,
                        u_func, h_nxt_ind_array, thorn, n_grid = 2000):
    """
    Dense-grid VFI (no Brent).  Exact same I/O shape as original.
    """

    # ── tuning: number of candidate points for a′ on [a_low, a_high] ──
    n_grid = n_grid          # raise for higher accuracy, lower for speed
    # ------------------------------------------------------------------

    n_W  = w_grid.size
    n_A, n_H, n_Y = V_next.shape

    policy_c    = np.empty((n_W, n_H, n_Y))
    policy_a    = np.empty_like(policy_c)
    Q_dcsn      = np.empty_like(policy_c)
    V_cntn      = np.empty_like(policy_c)
    lambda_cntn = np.empty_like(policy_c)

    step_inv = 1.0 / (n_grid - 1)        # pre-compute to avoid div inside loop

    for h in prange(n_H):
        H_val     = H_grid[h] * thorn
        h_nxt_ind = h_nxt_ind_array[h]

        for y in range(n_Y):
            V_slice = V_next[:, h_nxt_ind, y]      # contiguous 1-D view

            for iw in range(n_W):
                w_val  = w_grid[iw]
                a_low  = a_grid[0]
                a_high = min(w_val - 1e-12, a_grid[-1])   # ensure c > 0

                # handle degenerate case (wealth at borrowing limit)
                if a_high <= a_low + 1e-14:
                    a_high = a_low

                best_Q = -1e110
                best_a = a_low

                # exhaustive search over dense grid
                for g in range(n_grid):
                    a_try = a_low + (a_high - a_low) * g * step_inv
                    Q_try = bellman_obj(                # ← your existing fn
                        a_try, w_val, H_val,
                        beta, delta, a_grid, V_slice, u_func
                    )
                    if Q_try > best_Q:
                        best_Q = Q_try
                        best_a = a_try

                c_star = w_val - best_a

                # guard against inf / nan
                if np.isinf(best_Q) or c_star <= 0.0:
                    c_star = 1e-10
                    best_Q = -1e100
                    best_a = 1e-10

                policy_c[iw, h, y] = c_star
                policy_a[iw, h, y] = best_a
                Q_dcsn[iw, h, y]   = best_Q

            # ---- continuation value + λ --------------------------------
            c_prime = piecewise_gradient(policy_c[:, h, y], w_grid, m_bar)

            for iw in range(n_W):
                c_now  = policy_c[iw, h, y]
                uc_now = 1.0 / c_now
                V_cntn[iw, h, y] = (Q_dcsn[iw, h, y]
                                    - (1 - delta)*u_func(c_now, H_val)) / delta
                lambda_cntn[iw, h, y] = (uc_now +
                                          (1 - delta)*c_prime[iw]*uc_now) / delta

    return policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn


def _solve_vfi_loop(vlu_cntn, model):
    """
    Drop-in replacement that matches the EGM solver’s 7-value return.
    """
    # --- grids ---------------------------------------------------
    w_grid = model.num.state_space.dcsn.grids.w.astype(np.float64)
    a_grid = model.num.state_space.cntn.grids.a_nxt.astype(np.float64)
    H_grid = model.num.state_space.cntn.grids.H_nxt.astype(np.float64)
    if "RNT" in model.stage_name:
        thorn = model.param.thorn
    else:
        thorn = 1

    # --- parameters ---------------------------------------------
    beta   = float(model.param.beta)
    delta  = float(model.param.delta_pb)
    m_bar  = float(model.settings_dict["m_bar"])
    N_arg_grid_vfi = model.settings_dict["N_arg_grid_vfi"]

    # --- compiled utility functions -----------------------------
    u_jit  = model.num.functions.u_func       # scalar (c,H) → u
    uc_jit = model.num.functions.uc_func      # scalar (c,H) → u_c

    # --- contiguous 3-D continuation value array ----------------
    V_next = np.ascontiguousarray(vlu_cntn, dtype=np.float64)

    expr_str = model.math["functions"]["u_func"]["expr"]
    param_vals = {
        "alpha": model.param.alpha,
        "kappa": model.param.kappa,
        "iota" : model.param.iota,
        # add any extra constants referenced in expr_str
    }
    utility_func = build_njit_utility(expr_str, param_vals)
    g_ve_h_ind = model.num.functions.g_ve_h_ind
    h_nxt_ind_array = g_ve_h_ind(H_ind = np.arange(H_grid.size))

    # --- run kernel ---------------------------------------------
    if model.methods["solution"] == "VFI":
        policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn = _solve_vfi_numba(
            V_next, w_grid, a_grid, H_grid,
            beta, delta, m_bar,
            utility_func,h_nxt_ind_array,thorn
        )
    if model.methods["solution"] == "VFI_GRID":
        policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn = _solve_vfi_numba_grid(
            V_next, w_grid, a_grid, H_grid,
            beta, delta, m_bar,
            utility_func,h_nxt_ind_array,thorn, n_grid = N_arg_grid_vfi
        )

    # The EGM solver returns an average UE time and grid dicts.
    # For VFI these have no meaning, so we stub them out.
    avg_ue_time = 0.0

    # fill in empty egm grids as in the egm code so plotter does not error
    unrefined_grids = {
        'e': np.empty((0,0,0)),    # Endogenous grid points (cash-on-hand)
        'Q': np.empty((0,0,0)),    # Value function
        'c': np.empty((0,0,0)),    # Consumption policy
        'a': np.empty((0,0,0))     # Asset policy
    }
    refined_grids = {
        'e': np.empty((0,0,0)),    # Refined endogenous grid points
        'Q': np.empty((0,0,0)),    # Refined value function
        'c': np.empty((0,0,0)),    # Refined consumption policy
        'a': np.empty((0,0,0)),    # Refined asset policy
        'lambda': {} # Refined marginal utility
    }
    egm_grids   = {"unrefined": unrefined_grids, "refined": refined_grids}

    return (policy_c,             # same as `policy`
            V_cntn,               # vlu_dcsn
            lambda_cntn,          # lambda_dcsn
            avg_ue_time,
            egm_grids,
            policy_a,             # asset policy, matches EGM output
            Q_dcsn)               # Q_dcsn (decision objective)

# --- End Private Solver Loop Helpers ---

# Need F_ownc_dcsn_to_cntn - If it exists, move it here. 
# Otherwise, it needs to be defined. Let's assume it needs definition:

def F_ownc_dcsn_to_cntn(mover):
    """Create operator for ownc_dcsn_to_cntn mover (Forward step).
    
    Calculates end-of-period assets based on decision-period state and policy.
    Maps (w, H_nxt, y) -> a_nxt.
    The value/lambda mapping is usually handled by backward steps or identity.
    This primarily calculates the implied asset state.
    NOTE: This simple version doesn't map values/lambda, assumes handled elsewhere.
    """
    # Extract model and stage
    model = mover.model
    
    # In StageCraft, Movers have source_name and target_name, not perch objects
    # We need to access the stage to get the perches
    source_name = mover.source_name  # Should be 'dcsn'
    target_name = mover.target_name  # Should be 'cntn'
    
    # Get the stage to access perches
    
    def operator(perch_data):
        """Transforms dcsn state & policy to cntn state (assets).
        
        This is often simple as cntn state a_nxt = w - policy(w, H_nxt, y).
        Value/lambda propagation depends on model structure.
        Often, the backward step (cntn -> dcsn) calculates the necessary value/lambda
        at the dcsn state, making this forward step potentially just state mapping.
        Here, we return an empty dict assuming value/lambda mapping isn't needed
        in this specific forward operator.
        """
        # policy = perch_data["policy"] # Consumption policy C(w, H_nxt, y)
        # w_grid = stage.dcsn.grid.w
        # a_nxt = w_grid[:, None, None] - policy
        
        # Minimal implementation: returns empty dict assuming value/lambda 
        # mapping is not done by this specific forward operator.
        # The framework connects states, value calculation happens in backward steps.
        return {}
        
    return operator 