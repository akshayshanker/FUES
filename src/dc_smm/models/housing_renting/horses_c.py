import numpy as np
import time  # Add import for timing
from dc_smm.models.housing_renting.horses_common import (
    _safe_interp, egm_preprocess
)  # Use relative import
from numba import njit
from dc_smm.uenvelope.upperenvelope import EGM_UE
from dc_smm.fues.helpers import interp_as
from dynx.heptapodx.num.compile import compile_numba_function
from typing import Dict, Callable



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
    method = model.operator.get("solution", "EGM")
    model.stage_name = mover.stage_name
    
    def operator(perch_data):
        vlu_cntn = perch_data["vlu"]
        lambda_cntn = perch_data.get("lambda") # Lambda might not be needed for VFI
        
        # Track total time for this operation
        start_time = time.time()
        
        if method.upper() == "EGM":
            if lambda_cntn is None:
                raise ValueError("lambda_cntn is required for EGM method.")
            policy, vlu_dcsn, lambda_dcsn, ue_time_avg, egm_grids, policy_a = (
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
                "timing": timing_info,
                "EGM": egm_grids  # Store both unrefined and refined grids
            }
        elif method.upper() == "VFI":
            policy, vlu_dcsn, lambda_dcsn = _solve_vfi_loop(vlu_cntn, model)
            # No UE time for VFI
            timing_info = {
                "total_time": time.time() - start_time
            }
            return {
                "policy": policy,
                "vlu": vlu_dcsn,
                "lambda": lambda_dcsn,
                "timing": timing_info
            }
        else:
            raise ValueError(
                f"Unknown solution method: {method}. Choose 'EGM' or 'VFI'."
            )
    
    return operator


def build_njit_utility(
    expr: str,
    params: Dict[str, float],
    h_placeholder: str = "H_nxt",
) -> Callable[[float, float], float]:
    """
    Compile a two-argument utility u(c, H) that is Numba nopython.

    Parameters
    ----------
    expr : str
        Raw expression, e.g. "alpha*np.log(c)+(1-alpha)*np.log(kappa*(H_nxt+iota))"
    params : dict
        Literal parameter values referenced in *expr* (alpha, kappa, …).
    h_placeholder : str, optional
        Token for housing inside *expr* (default "H_nxt").

    Returns
    -------
    callable
        nopython-compiled function u(c, H) → float
    """

    # 1.  replace the placeholder with a run-time variable name 'H'
    #patched = expr.replace(h_placeholder, "H")
    patched = expr.replace(h_placeholder, "H")

    # 2.  build source code for a pure Python function
    func_src = "def _u(c, H):\n    return " + patched

    # 3.  execute in a tiny namespace containing numpy and constants
    ns = {"np": np, **params}
    exec(func_src, ns)          # defines _u in ns
    py_func = ns["_u"]

    # 4.  JIT-compile to nopython; result takes (c, H) positional args
    return njit(py_func)

# --- Private Solver Loop Helpers ---
def _solve_egm_loop(vlu_cntn, lambda_cntn, model):
    """Solves the consumption problem using the EGM loop."""
    # Use grid proxies for direct grid access
    a_nxt_grid = model.num.state_space.cntn.grids.a_nxt
    H_nxt_grid = model.num.state_space.cntn.grids.H_nxt
    w_grid = model.num.state_space.dcsn.grids.w
    
    # Get functions and parameters
    compiled_funcs = model.num.functions
    beta = model.param.beta
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
    lambda_dcsn = np.empty((n_w, n_H, n_y))
    
    # Track total UE time and count for averaging
    total_ue_time = 0.0
    ue_count = 0

    # Store the unrefined and refined grids
    unrefined_grids = {
        'e': {},    # Endogenous grid points (cash-on-hand)
        'v': {},    # Value function
        'c': {},    # Consumption policy
        'a': {}     # Asset policy
    }
    
    refined_grids = {
        'e': {},    # Refined endogenous grid points
        'v': {},    # Refined value function
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
    lam_scaled = beta * lambda_cntn * Rfree                     # shape (n_w,n_H_total,n_y)

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
            vlu_v_egm = compiled_funcs.u_func(**u_params) + beta * vlu_e
            v_nxt_raw = beta * vlu_e

            # Process the EGM solution to ensure grid uniqueness
            if ue_method != "CONSAV":
                m_egm_unique, vlu_v_egm_unique, c_egm_unique, a_nxt_grid_unique = (
                    egm_preprocess(
                        m_egm, vlu_v_egm, c_egm, a_nxt_grid, beta, 
                        compiled_funcs.u_func, vlu_e, n_con=n_con,
                        h_nxt=H_nxt_grid[i_h]*thorn
                    )
                )
                #print("not CONSAV")
            else:
                m_egm_unique = m_egm
                vlu_v_egm_unique = vlu_v_egm
                c_egm_unique = c_egm
                a_nxt_grid_unique = a_nxt_grid

            # Store unrefined grids (before upper envelope)
            grid_key = f"{i_y}-{i_h}"
            unrefined_grids['e'][grid_key] = m_egm_unique
            unrefined_grids['v'][grid_key] = vlu_v_egm_unique
            unrefined_grids['c'][grid_key] = c_egm_unique
            unrefined_grids['a'][grid_key] = a_nxt_grid_unique
            def partial_uc(c_vals):
                return compiled_funcs.uc_func(**{
                    "c": c_vals, 
                    "H_nxt": H_nxt_grid[i_h]*thorn
                })
   
            # Get upper envelope solution and timing
            refined, _, _ = EGM_UE(
                m_egm_unique, vlu_v_egm_unique, v_nxt_raw, c_egm_unique, 
                a_nxt_grid_unique, w_grid, partial_uc, 
                u_func={"func": utility_func, "args": H_val},
                ue_method=ue_method, m_bar=m_bar, lb=lb,
                rfc_radius=rfc_radius, rfc_n_iter=rfc_n_iter
            )
            
            # Unpack the results from the refined dictionary
            # Prefer *_ref keys, fallback to non-suffixed for broader compatibility
            m_refined = refined["x_dcsn_ref"]
            v_refined = refined["v_dcsn_ref"]
            c_refined = refined["kappa_ref"] # kappa_ref for consumption
            a_refined = refined["x_cntn_ref"] # x_cntn_ref for next period assets
            
            lambda_refined = refined["lambda_ref"]

            if ue_method != "CONSAV":
                
                vlu_dcsn[:, i_h, i_y] = interpf(m_refined, v_refined)
                policy[:, i_h, i_y]   = interpf(m_refined, c_refined)
                #policy_a[:, i_h, i_y] = interpf(a_refined)
                lambda_dcsn[:, i_h, i_y] = compiled_funcs.uc_func(**{
                    "c": policy[:, i_h, i_y], 
                    "H_nxt": H_nxt_grid[i_h]
                })

            else:
                vlu_dcsn[:, i_h, i_y] = v_refined
                policy[:, i_h, i_y] = c_refined
                policy_a[:, i_h, i_y] = a_refined
                lambda_dcsn[:, i_h, i_y] = lambda_refined

            ue_time = refined.get("ue_time", 0.0)
            
            # Store refined grids (after upper envelope) 
            # using consistently retrieved variables
            refined_grids['e'][grid_key] = m_refined # cash-on-hand / endogenous grid
            refined_grids['v'][grid_key] = v_refined # value function
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
    
    return policy, vlu_dcsn, lambda_dcsn, avg_ue_time, egm_grids, policy_a

def _solve_vfi_loop(vlu_cntn, model):
    """Solves the consumption problem using the VFI loop with brent_max."""
    # Import brent_max here to avoid import error if VFI is not used
    try:
        from quantecon.optimize.scalar_maximization import brent_max
    except ImportError:
        raise ImportError(
            "brent_max not found for VFI. Please install quantecon: "
            "pip install quantecon"
        )

    # Use grid proxies for direct grid access
    w_grid = model.num.state_space.dcsn.grid.w
    a_nxt_grid = model.num.state_space.cntn.grid.a_nxt
    H_nxt_grid = model.num.state_space.cntn.grid.H_nxt
    
    # Get functions and parameters
    compiled_funcs = model.num.functions
    beta = model.param.beta

    n_w = len(w_grid)
    n_H = vlu_cntn.shape[1]
    n_y = vlu_cntn.shape[2]

    policy = np.zeros((n_w, n_H, n_y))
    vlu_dcsn = np.zeros((n_w, n_H, n_y))
    lambda_dcsn = np.zeros((n_w, n_H, n_y))

    for i_y in range(n_y):
        for i_h in range(n_H):
            interp_v_nxt = _safe_interp(a_nxt_grid, vlu_cntn[:, i_h, i_y])

            for i_w, w_val in enumerate(w_grid):
                
                def bellman_objective(a_nxt_candidate):
                    c_candidate = w_val - a_nxt_candidate
                    if c_candidate <= 1e-12:
                        return -np.inf
                    u_params = {"c": c_candidate, "H_nxt": H_nxt_grid[i_h]}
                    u_current = compiled_funcs.u_func(**u_params)
                    vlu_nxt = interp_v_nxt(a_nxt_candidate)
                    return u_current + beta * vlu_nxt

                lower_bound_a = a_nxt_grid[0]
                upper_bound_a = w_val

                if upper_bound_a <= lower_bound_a + 1e-9:
                    a_nxt_opt = lower_bound_a
                    v_opt = bellman_objective(a_nxt_opt)
                else:
                    lower_brent = lower_bound_a
                    upper_brent = upper_bound_a
                    try:
                        a_nxt_opt, v_opt, _ = brent_max(
                            bellman_objective, 
                            lower_brent, 
                            upper_brent, 
                            xtol=1e-6
                        )
                    except ValueError:
                        print(
                            f"Warning: brent_max failed at w={w_val}, "
                            f"h={i_h}, y={i_y}. Using endpoint."
                        )
                        val_at_lower = bellman_objective(lower_brent)
                        val_at_upper = bellman_objective(upper_brent)
                        if val_at_lower >= val_at_upper:
                            a_nxt_opt = lower_brent
                            v_opt = val_at_lower
                        else:
                            a_nxt_opt = upper_brent
                            v_opt = val_at_upper
                
                c_opt = w_val - a_nxt_opt
                c_opt = max(c_opt, 1e-12)
                lambda_opt = compiled_funcs.uc_func(**{
                    "c": c_opt, 
                    "H_nxt": H_nxt_grid[i_h]
                })

                policy[i_w, i_h, i_y] = c_opt
                vlu_dcsn[i_w, i_h, i_y] = v_opt
                lambda_dcsn[i_w, i_h, i_y] = lambda_opt

    return policy, vlu_dcsn, lambda_dcsn

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