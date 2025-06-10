import os, time, numba, numpy as np
import logging
from dc_smm.models.housing_renting.horses_common import (
    egm_preprocess, build_njit_utility, piecewise_gradient, piecewise_gradient_3rd, get_u_func, bellman_obj
)  # Use relative import
from numba import njit, prange
from dc_smm.uenvelope.upperenvelope import EGM_UE
from dc_smm.fues.helpers import interp_as
from quantecon.optimize.scalar_maximization import brent_max
from dynx.stagecraft.solmaker import Solution

from dc_smm.helpers.mpi_utils import (
    get_comm, scatter_dict_list, gather_nested, barrier_print, broadcast_arrays, chunk_indices
)

# horses_c.py   (top of file)

logger = logging.getLogger(__name__)

if logger.isEnabledFor(logging.INFO):
    logger.info(f"NUMBA threads: {numba.get_num_threads()}, OMP_NUM_THREADS: {os.getenv('OMP_NUM_THREADS')}")

# --- Operator Factory for OWNC Consumption Choice ---

def F_ownc_cntn_to_dcsn(mover, use_mpi=False, comm=None):
    """Create operator for ownc_cntn_to_dcsn mover.
    Implements EGM or VFI for consumption choice.
    
    Args:
        mover: StageCraft mover object
        use_mpi: Whether to use MPI parallelization
        comm: MPI communicator (if use_mpi=True)
    """
    # Extract mover model
    model = mover.model

    # Get solution method from settings
    method = model.methods["solution"]
    model.stage_name = mover.stage_name

    # Produce the operator given the mover model
    def operator(perch_data):
        """Solves the agent's consumption and savings problem.

        This operator takes the continuation value and marginal utility
        from the 'cntn' (continuation) perch and calculates the optimal
        consumption policy, value function, and marginal utility at the
        'dcsn' (decision) perch. It supports both EGM and VFI solution methods.

        Args:
            perch_data: A Solution object containing data from the source perch,
                        including 'vlu' (value function) and 'lambda_' (marginal utility).

        Returns:
            A Solution object populated with the decision policy ('c', 'a'),
            value function ('vlu'), marginal utility ('lambda_'),
            decision objective ('Q'), EGM grids (if applicable),
            and timing information.

        Raises:
            ValueError: If the EGM method is selected but 'lambda_cntn' is not provided.
            ValueError: If an unknown solution method is specified.
        """

        # Extract the source perch data (Solution)
        vlu_cntn = perch_data.vlu
        lambda_cntn = perch_data.lambda_

        # Track total time for the operator
        start_time = time.time()

        # EGM loop
        #  - recall method is solution method (not upper envelope method)
        if method.upper() == "EGM":
            if lambda_cntn is None:
                raise ValueError("lambda_cntn is required for EGM method.")

            # Heavy lifting loop for EGM
            policy, vlu_dcsn, lambda_dcsn, ue_time_avg, egm_grids, policy_a, Q_dcsn = (
                _solve_egm_loop(vlu_cntn, lambda_cntn, model)
            )
            # Include UE timing in the result
            timing_info = {
                "ue_time_avg": ue_time_avg,
                "total_time": time.time() - start_time
            }

            # Create Solution object for successor perch
            sol = Solution()
            sol.policy["c"] = policy
            sol.policy["a"] = policy_a
            sol.vlu = vlu_dcsn
            sol.lambda_ = lambda_dcsn
            sol.Q = Q_dcsn
            sol.timing = timing_info

            # Attaching EGM grids to Solution
            # TODO: SIMPLIFY THIS.
            for key, arr in egm_grids["unrefined"]["e"].items():
                sol.EGM.unrefined[f"e_{key}"] = arr
            for key, arr in egm_grids["unrefined"]["Q"].items():
                sol.EGM.unrefined[f"Q_{key}"] = arr
            for key, arr in egm_grids["unrefined"]["c"].items():
                sol.EGM.unrefined[f"c_{key}"] = arr
            for key, arr in egm_grids["unrefined"]["a"].items():
                sol.EGM.unrefined[f"a_{key}"] = arr

            for key, arr in egm_grids["refined"]["e"].items():
                sol.EGM.refined[f"e_{key}"] = arr
            for key, arr in egm_grids["refined"]["Q"].items():
                sol.EGM.refined[f"Q_{key}"] = arr
            for key, arr in egm_grids["refined"]["c"].items():
                sol.EGM.refined[f"c_{key}"] = arr
            for key, arr in egm_grids["refined"]["a"].items():
                sol.EGM.refined[f"a_{key}"] = arr
            for key, arr in egm_grids["refined"]["lambda_"].items():
                sol.EGM.refined[f"lambda_{key}"] = arr

            return sol

        # VFI methods
        elif method.upper() == "VFI" or method.upper() == "VFI_HDGRID":
            # Heavy lifting loop for VFI
            policy, vlu_dcsn, lambda_dcsn, ue_time_avg, egm_grids, policy_a, Q_dcsn = (
                _solve_vfi_loop(vlu_cntn, model, use_mpi=use_mpi, comm=comm)
            )

            # No UE time for VFI
            timing_info = {
                "total_time": time.time() - start_time
            }

            # Create Solution object
            sol = Solution()
            sol.policy["c"] = policy
            sol.policy["a"] = policy_a
            sol.vlu = vlu_dcsn
            sol.lambda_ = lambda_dcsn
            sol.Q = Q_dcsn
            sol.timing = timing_info

            # Store empty EGM grids for VFI (if any)
            # TODO: SIMPLIFY THIS.
            if "e" in egm_grids["unrefined"]:
                for key, arr in egm_grids["unrefined"]["e"].items():
                    sol.EGM.unrefined[f"e_{key}"] = arr
            if "Q" in egm_grids["unrefined"]:
                for key, arr in egm_grids["unrefined"]["Q"].items():
                    sol.EGM.unrefined[f"Q_{key}"] = arr
            if "c" in egm_grids["unrefined"]:
                for key, arr in egm_grids["unrefined"]["c"].items():
                    sol.EGM.unrefined[f"c_{key}"] = arr
            if "a" in egm_grids["unrefined"]:
                for key, arr in egm_grids["unrefined"]["a"].items():
                    sol.EGM.unrefined[f"a_{key}"] = arr

            # Store refined grids for VFI (if any)
            if "e" in egm_grids["refined"]:
                for key, arr in egm_grids["refined"]["e"].items():
                    sol.EGM.refined[f"e_{key}"] = arr
            if "Q" in egm_grids["refined"]:
                for key, arr in egm_grids["refined"]["Q"].items():
                    sol.EGM.refined[f"Q_{key}"] = arr
            if "c" in egm_grids["refined"]:
                for key, arr in egm_grids["refined"]["c"].items():
                    sol.EGM.refined[f"c_{key}"] = arr
            if "a" in egm_grids["refined"]:
                for key, arr in egm_grids["refined"]["a"].items():
                    sol.EGM.refined[f"a_{key}"] = arr
            if "lambda_" in egm_grids["refined"]:
                for key, arr in egm_grids["refined"]["lambda_"].items():
                    sol.EGM.refined[f"lambda_{key}"] = arr

            return sol
        else:
            raise ValueError(
                f"Unknown solution method: {method}. Choose 'EGM', 'VFI', or 'VFI_HDGRID'."
            )

    return operator


def F_rntc_cntn_to_dcsn(mover, use_mpi=False, comm=None):
    """Create operator for rntc_cntn_to_dcsn mover.
    Implements EGM or VFI for renter consumption choice.
    
    Args:
        mover: StageCraft mover object
        use_mpi: Whether to use MPI parallelization
        comm: MPI communicator (if use_mpi=True)
    """
    # This is essentially the same as the owner version
    # The difference is handled inside the solver via model.stage_name
    return F_ownc_cntn_to_dcsn(mover, use_mpi=use_mpi, comm=comm)


# --- Private Solver Loop Helpers ---
def _solve_egm_loop(vlu_cntn, lambda_cntn, model):
    """Solves the consumption problem using the EGM loop."""

    # ------------------------------------------------------------------
    # 1. Unpack everything we need from the model numerical object
    # ------------------------------------------------------------------

    # grids
    a_nxt_grid = model.num.state_space.cntn.grids.a_nxt
    H_nxt_grid = model.num.state_space.cntn.grids.H_nxt
    w_grid = model.num.state_space.dcsn.grids.w

    # functions
    compiled_funcs = model.num.functions
    g_ve_h_ind = compiled_funcs.g_ve_h_ind

    # utility function fully jited
    # TODO: this should be compiled by Heptapod.
    expr_str = model.math["functions"]["u_func"]["expr"]
    param_vals = {
        "alpha": model.param.alpha,
        "kappa": model.param.kappa,
        "iota": model.param.iota,
        # add any extra constants referenced in expr_str
    }

    if model.methods["upper_envelope"] == "CONSAV":
        utility_func = build_njit_utility(expr_str, param_vals)
    else:
        utility_func = get_u_func(expr_str, param_vals)
    

    # parameters
    beta = model.param.beta
    delta = model.param.delta_pb      # NEW
    Rfree = model.param.r + 1

    # thorn for renters not for owners
    if "RNT" in model.stage_name:
        thorn = model.param.thorn
    else:
        thorn = 1

    # settings
    c_max = model.settings_dict["c_max"]

    # Get functions and parameters
    m_bar = model.settings_dict["m_bar"]
    lb = model.settings_dict["lb"]
    rfc_radius = model.settings_dict["rfc_radius"]
    rfc_n_iter = model.settings_dict["rfc_n_iter"]
    n_con = model.settings_dict["n_constraint_points"]
    n_con_nxt = model.settings_dict["n_constraint_points_nxt"]

    # methods
    ue_method = model.methods["upper_envelope"]

    # ------------------------------------------------------------------
    # 2. Produce the grids we will fill
    # TODO: can be pre-filled into Sol?
    # ------------------------------------------------------------------

    n_w = len(w_grid)
    n_H = vlu_cntn.shape[1]
    n_y = vlu_cntn.shape[2]

    policy = np.empty((n_w, n_H, n_y))
    policy_a = np.empty((n_w, n_H, n_y))
    vlu_dcsn = np.empty((n_w, n_H, n_y))
    Q_dcsn = np.empty((n_w, n_H, n_y))
    lambda_dcsn = np.empty((n_w, n_H, n_y))

    # Container for EGM grids
    unrefined_grids = {k: {} for k in ('e', 'Q', 'c', 'a')}
    refined_grids = {k: {} for k in ('e', 'Q', 'c', 'a', 'lambda_')}
    egm_grids = {"unrefined": unrefined_grids,
                 "refined":   refined_grids}

    # array for indices of post-state housing (not services!)
    # for owners, it is the same as the housing index in the loop
    # for renters, it is zero. for renters, housing index in loop
    # is services only.
    h_nxt_ind_array = g_ve_h_ind(H_ind=np.arange(n_H))

    # Track total UE time and count for averaging
    total_ue_time = 0.0
    ue_count = 0

    # ------------------------------------------------------------------
    #  4. Vectorised pre-computation of c_egm for ALL (w,h,y) points
    # ------------------------------------------------------------------
    # 1.  Scale λ once
    lam_scaled = beta * delta * lambda_cntn * \
        Rfree  # shape (n_w,n_H_total,n_y)

    # 2.  Map continuous-grid H-indices to decision-grid indices (renters need this)
    lam_sel = np.take(lam_scaled, h_nxt_ind_array,
                      axis=1)      # → (n_w,n_H,n_y)
    # same mapping for value
    vlu_sel = np.take(vlu_cntn, h_nxt_ind_array, axis=1)

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
    q_egm_all = compiled_funcs.u_func(
        c_egm_all, H_bcast) + delta*beta * vlu_sel

    # ------------------------------------------------------------------
    # 5. Loop over all income and housing points to do the upper envelope
    # ------------------------------------------------------------------
    for i_y in range(n_y):
        for i_h in range(n_H):

            # Slice the pre-computed λ, v and c cubes – no recomputation inside the loop
            # lambda_e = lam_sel[:, i_h, i_y]
            vlu_e = vlu_sel[:, i_h, i_y]
            c_egm = c_egm_all[:, i_h, i_y]
            H_val = H_nxt_grid[i_h] * thorn
            m_egm = c_egm + a_nxt_grid
            # u_params = {"c": c_egm, "H_nxt": H_val}
            # q_egm = compiled_funcs.u_func(**u_params) + delta*beta * vlu_e

            q_egm = q_egm_all[:, i_h, i_y]  # current value
            # continuation value from the POV of current individual
            q_nxt_raw = delta*beta * vlu_e

            # Pre-process the EGM solution
            # Add constraint points for OBC
            # Add nxt period binding solutions (if time inconsistency)
            # Only applies to DCEGM, FUES, RFC.
            if ue_method != "CONSAV":
                m_egm_unique, vlu_q_egm_unique, c_egm_unique, a_nxt_grid_unique = (
                    egm_preprocess(
                        m_egm, q_egm, c_egm, a_nxt_grid, delta*beta,
                        utility_func, vlu_e, m_bar=m_bar, n_con=n_con,
                        n_con_nxt=n_con_nxt, c_max=c_max, h_nxt=H_val
                    )
                )

            else:
                m_egm_unique = m_egm
                vlu_q_egm_unique = q_egm
                c_egm_unique = c_egm
                a_nxt_grid_unique = a_nxt_grid

            # Store unrefined grids (before upper envelope)
            grid_key = f"{i_y}-{i_h}"
            unrefined_grids['e'][grid_key] = m_egm_unique
            unrefined_grids['Q'][grid_key] = vlu_q_egm_unique
            unrefined_grids['c'][grid_key] = c_egm_unique
            unrefined_grids['a'][grid_key] = a_nxt_grid_unique

            # The upper envelope wrapper calculates the marginal utility
            # this can be removed.
            # TODO: streamline muc here
            def partial_uc(c_vals):
                return compiled_funcs.uc_func(**{
                    "c": c_vals,
                    "H_nxt": H_nxt_grid[i_h]*thorn
                })

            # Get upper envelope solution and timing
            refined, _, _ = EGM_UE(
                m_egm_unique, vlu_q_egm_unique, q_nxt_raw, c_egm_unique,
                a_nxt_grid_unique, w_grid, partial_uc,
                u_func={"func": utility_func, "args": H_val},
                ue_method=ue_method, m_bar=m_bar, lb=lb,
                rfc_radius=rfc_radius, rfc_n_iter=rfc_n_iter
            )

            # Unpack the results from the refined dictionary
            # Prefer *_ref keys, fallback to non-suffixed for broader compatibility
            m_refined = refined["x_dcsn_ref"]
            q_refined = refined["v_dcsn_ref"]
            c_refined = refined["kappa_ref"]  # kappa_ref for consumption
            # x_cntn_ref for next period assets
            a_refined = refined["x_cntn_ref"]

            lambda_refined = refined["lambda_ref"]

            if ue_method != "CONSAV":

                Q_dcsn[:, i_h, i_y] = interp_as(
                    m_refined, q_refined, w_grid, extrap=True)
                policy[:, i_h, i_y] = interp_as(
                    m_refined, c_refined, w_grid, extrap=True)

            else:
                Q_dcsn[:, i_h, i_y] = q_refined
                policy[:, i_h, i_y] = c_refined
                policy_a[:, i_h, i_y] = a_refined
                lambda_dcsn[:, i_h, i_y] = lambda_refined

            policy[policy < 0] = 1e-10

            # Calculate decision objective Q_dcsn and derivative of consumption
            # for present-biased utility
            uc_today = compiled_funcs.uc_func(**{
                "c": policy[:, i_h, i_y],
                "H_nxt": H_val
            })

            if delta < 1:
                c_prime = piecewise_gradient_3rd(
                    policy[:, i_h, i_y], w_grid, m_bar=m_bar)
            else:
                c_prime = np.zeros_like(policy[:, i_h, i_y])

            # Shadow value with marginal utility
            lambda_dcsn[:, i_h, i_y] = (
                uc_today - (1-delta)*c_prime*uc_today) / delta
            vlu_dcsn[:, i_h, i_y] = (Q_dcsn[:, i_h, i_y] - (1-delta) *
                                     compiled_funcs.u_func(c=policy[:, i_h, i_y], H_nxt=H_val))/delta

            ue_time = refined.get("ue_time", 0.0)

            # Store refined grids (after upper envelope)
            # using consistently retrieved variables
            # for consav, refined grids are just the interpolants (consav package returns)
            # cash-on-hand / endogenous grid
            refined_grids['e'][grid_key] = m_refined
            refined_grids['Q'][grid_key] = q_refined  # value function
            refined_grids['c'][grid_key] = c_refined  # consumption policy
            refined_grids['a'][grid_key] = a_refined  # asset policy
            refined_grids['lambda_'][grid_key] = lambda_refined

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





@njit(nopython=False)
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
    n_W = w_grid.size
    n_A, n_H, n_Y = V_next.shape

    policy_c = np.empty((n_W, n_H, n_Y))
    policy_a = np.empty_like(policy_c)
    Q_dcsn = np.empty_like(policy_c)
    V_cntn = np.empty_like(policy_c)
    lambda_cntn = np.empty_like(policy_c)

    for h in range(n_H):
        H_val = H_grid[h]*thorn
        for y in range(n_Y):
            h_nxt_ind = h_nxt_ind_array[h]
            V_slice = V_next[:, h_nxt_ind, y]                # contiguous 1-D

            for iw in range(n_W):
                w_val = w_grid[iw]
                a_low = a_grid[0]
                a_high = min(w_val + 1e-1, a_grid[-1])

                a_star, Q_star, _ = brent_max(
                    bellman_obj,
                    a_low, a_high,
                    args=(w_val, H_val, beta, delta,
                          a_grid, V_slice, u_func),
                    xtol=1e-12
                )
                c_star = w_val - a_star

                if np.isinf(Q_star):
                    c_star = 1e-100
                    Q_star = -1e100
                    a_star = 1e-100

                policy_c[iw, h, y] = c_star
                policy_a[iw, h, y] = a_star
                Q_dcsn[iw, h, y] = Q_star

            # ---- continuation value + λ --------------------------------
            c_prime = piecewise_gradient(policy_c[:, h, y], w_grid, m_bar)

            for iw in range(n_W):
                c_now = policy_c[iw, h, y]
                uc_now = 1/c_now
                V_cntn[iw, h, y] = (Q_dcsn[iw, h, y]
                                    - (1.0 - delta)*u_func(c_now, H_val)) / delta
                lambda_cntn[iw, h, y] = (uc_now -
                                         (1.0 - delta)*c_prime[iw]*uc_now) / delta

                # print(V_cntn[iw, h, y][np.isinf(V_cntn[iw, h, y])])

    return policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn


@njit
def _solve_vfi_block(h_idx, y_idx, V_next, w_grid, a_grid, H_grid,
                     beta, delta, m_bar, u_func, h_nxt_ind_array,
                     thorn, n_grid):
    """
    Solve VFI for a block of (h,y) pairs.
    
    Parameters
    ----------
    h_idx : np.ndarray(int32)
        Housing indices, shape (B,)
    y_idx : np.ndarray(int32) 
        Income indices, shape (B,)
    V_next : np.ndarray
        Value function continuation
    w_grid : np.ndarray
        Wealth grid
    a_grid : np.ndarray  
        Asset grid
    H_grid : np.ndarray
        Housing grid
    beta, delta, m_bar : float
        Parameters
    u_func : callable
        Utility function
    h_nxt_ind_array : np.ndarray
        Housing index mapping
    thorn : float
        Rental efficiency
    n_grid : int
        Grid density
        
    Returns
    -------
    tuple
        (policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn) each shape (B, n_W)
    """
    B = h_idx.size
    n_W = w_grid.size
    
    # Output arrays (B, n_W)
    policy_c = np.empty((B, n_W))
    policy_a = np.empty((B, n_W))
    Q_dcsn = np.empty((B, n_W))
    V_cntn = np.empty((B, n_W))
    lambda_cntn = np.empty((B, n_W))
    
    step_inv = 1.0 / (n_grid - 1)
    #print(f"Solving block {h_idx[0]}-{y_idx[0]}")
    # Process each (h,y) pair in the block
    for b in range(B):
        h = h_idx[b]
        y = y_idx[b]
        
        H_val = H_grid[h] * thorn
        h_nxt_ind = h_nxt_ind_array[h]
        V_slice = V_next[:, h_nxt_ind, y]
        
        # Solve for all wealth points for this (h,y)
        for iw in range(n_W):
            w_val = w_grid[iw]
            a_low = a_grid[0]
            a_high = min(w_val - 1e-12, a_grid[-1])
            
            if a_high <= a_low + 1e-14:
                a_high = a_low
                
            best_Q = -1e110
            best_a = a_low
            
            # Dense grid search
            for g in range(n_grid):
                a_try = a_low + (a_high - a_low) * g * step_inv
                Q_try = bellman_obj(a_try, w_val, H_val, beta, delta, a_grid, V_slice, u_func)
                if Q_try > best_Q:
                    best_Q = Q_try
                    best_a = a_try
                    
            c_star = w_val - best_a
            
            if np.isinf(best_Q) or c_star <= 0.0:
                c_star = 1e-10
                best_Q = -1e100
                best_a = 1e-100
                
            policy_c[b, iw] = c_star
            policy_a[b, iw] = best_a
            Q_dcsn[b, iw] = best_Q
        
        # Calculate continuation value and lambda for this (h,y)
        c_prime = piecewise_gradient(policy_c[b, :], w_grid, m_bar)
        
        for iw in range(n_W):
            c_now = policy_c[b, iw]
            uc_now = 1.0 / c_now
            V_cntn[b, iw] = (Q_dcsn[b, iw] - (1 - delta) * u_func(c_now, H_val)) / delta
            lambda_cntn[b, iw] = (uc_now - (1 - delta) * c_prime[iw] * uc_now) / delta
    
    return policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn


def solve_vfi_grid_serial(V_next, w_grid, a_grid, H_grid,
                          beta, delta, m_bar, u_func, h_nxt_ind_array, 
                          thorn, n_grid=2000):
    """
    Serial VFI using block kernel.
    """
    n_A, n_H, n_Y = V_next.shape
    n_W = w_grid.size
    
    # Create index arrays for all (h,y) pairs
    h_idx = np.repeat(np.arange(n_H), n_Y).astype(np.int32)
    y_idx = np.tile(np.arange(n_Y), n_H).astype(np.int32)
    
    # Call block kernel
    policy_c_block, policy_a_block, Q_block, V_block, lambda_block = _solve_vfi_block(
        h_idx, y_idx, V_next, w_grid, a_grid, H_grid,
        beta, delta, m_bar, u_func, h_nxt_ind_array, thorn, n_grid
    )
    
    # Reshape to (n_W, n_H, n_Y) format
    policy_c = np.empty((n_W, n_H, n_Y))
    policy_a = np.empty((n_W, n_H, n_Y))
    Q_dcsn = np.empty((n_W, n_H, n_Y))
    V_cntn = np.empty((n_W, n_H, n_Y))
    lambda_cntn = np.empty((n_W, n_H, n_Y))
    
    for b in range(len(h_idx)):
        h = h_idx[b]
        y = y_idx[b]
        policy_c[:, h, y] = policy_c_block[b, :]
        policy_a[:, h, y] = policy_a_block[b, :]
        Q_dcsn[:, h, y] = Q_block[b, :]
        V_cntn[:, h, y] = V_block[b, :]
        lambda_cntn[:, h, y] = lambda_block[b, :]
    
    return policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn


def solve_vfi_grid_mpi(vlu_cntn, model, comm, use_mpi=True):
    """
    MPI-parallel VFI grid solver using block kernel.
    """
    # Extract parameters (same as serial version)
    w_grid = model.num.state_space.dcsn.grids.w.astype(np.float64)
    a_grid = model.num.state_space.cntn.grids.a_nxt.astype(np.float64)
    H_grid = model.num.state_space.cntn.grids.H_nxt.astype(np.float64)
    
    if "RNT" in model.stage_name:
        thorn = model.param.thorn
    else:
        thorn = 1
        
    beta = float(model.param.beta)
    delta = float(model.param.delta_pb)
    m_bar = float(model.settings_dict["m_bar"])
    N_arg_grid_vfi = model.settings_dict["N_arg_grid_vfi"]
    
    # Build utility function
    expr_str = model.math["functions"]["u_func"]["expr"]
    param_vals = {
        "alpha": model.param.alpha,
        "kappa": model.param.kappa,
        "iota": model.param.iota,
    }
    utility_func = build_njit_utility(expr_str, param_vals)
    
    g_ve_h_ind = model.num.functions.g_ve_h_ind
    h_nxt_ind_array = g_ve_h_ind(H_ind=np.arange(H_grid.size))
    
    V_next = np.ascontiguousarray(vlu_cntn, dtype=np.float64)
    n_A, n_H, n_Y = V_next.shape
    n_W = w_grid.size
    
    # Broadcast read-only data to all ranks
    broadcast_data = {
        'V_next': V_next,
        'w_grid': w_grid,
        'a_grid': a_grid,
        'H_grid': H_grid,
        'h_nxt_ind_array': h_nxt_ind_array,
        'beta': beta,
        'delta': delta,
        'm_bar': m_bar,
        'thorn': thorn,
        'n_grid': N_arg_grid_vfi
    }
    broadcast_data = broadcast_arrays(comm, broadcast_data, root=0)
    
    # Create and distribute (h,y) index pairs
    if comm.rank == 0:
        # Create all pairs
        h_all = np.repeat(np.arange(n_H), n_Y).astype(np.int32)
        y_all = np.tile(np.arange(n_Y), n_H).astype(np.int32)
        
        # Split across ranks
        chunks = []
        total_pairs = len(h_all)
        #print(f"Total pairs: {total_pairs}")
        for rank in range(comm.size):
            rank_indices = chunk_indices(total_pairs, comm.size, rank)
            h_chunk = h_all[rank_indices]
            y_chunk = y_all[rank_indices]
            chunks.append({'h_idx': h_chunk, 'y_idx': y_chunk})
    else:
        chunks = None
    
    # Scatter index chunks
    local_chunk = scatter_dict_list(comm, chunks)
    
    if local_chunk:
        h_local = local_chunk['h_idx']
        y_local = local_chunk['y_idx']
    else:
        h_local = np.array([], dtype=np.int32)
        y_local = np.array([], dtype=np.int32)
    
    # Process local chunk with block kernel (guarded for empty chunks)
    if h_local.size > 0:
        policy_c_local, policy_a_local, Q_local, V_local, lambda_local = _solve_vfi_block(
            h_local, y_local,
            broadcast_data['V_next'],
            broadcast_data['w_grid'],
            broadcast_data['a_grid'],
            broadcast_data['H_grid'],
            broadcast_data['beta'],
            broadcast_data['delta'],
            broadcast_data['m_bar'],
            utility_func,
            broadcast_data['h_nxt_ind_array'],
            broadcast_data['thorn'],
            broadcast_data['n_grid']
        )
        
        # Package results for gathering
        local_results = {
            'h_idx': h_local,
            'y_idx': y_local,
            'policy_c': policy_c_local,
            'policy_a': policy_a_local,
            'Q_dcsn': Q_local,
            'V_cntn': V_local,
            'lambda_cntn': lambda_local
        }
    else:
        # Empty chunk - return empty arrays with correct shapes
        local_results = {
            'h_idx': np.empty(0, dtype=np.int32),
            'y_idx': np.empty(0, dtype=np.int32),
            'policy_c': np.empty((0, n_W)),
            'policy_a': np.empty((0, n_W)),
            'Q_dcsn': np.empty((0, n_W)),
            'V_cntn': np.empty((0, n_W)),
            'lambda_cntn': np.empty((0, n_W))
        }
    
    # Gather results to root
    all_results = gather_nested(comm, [local_results], root=0)
    
    if comm.rank == 0:
        # Reconstruct full arrays
        policy_c = np.empty((n_W, n_H, n_Y))
        policy_a = np.empty((n_W, n_H, n_Y))
        Q_dcsn = np.empty((n_W, n_H, n_Y))
        V_cntn = np.empty((n_W, n_H, n_Y))
        lambda_cntn = np.empty((n_W, n_H, n_Y))
        
        for result in all_results:          # result *is* a dict
            h_idx = result['h_idx']
            y_idx = result['y_idx']
            for k in range(h_idx.size):          # handle multiple pairs per worker
                h = h_idx[k]
                y = y_idx[k]
                policy_c[:, h, y]   = result['policy_c'][k]
                policy_a[:, h, y]   = result['policy_a'][k]
                Q_dcsn[:, h, y]     = result['Q_dcsn'][k]
                V_cntn[:, h, y]     = result['V_cntn'][k]
                lambda_cntn[:, h, y]= result['lambda_cntn'][k]
        
        # Package full results for broadcasting
        full = (policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn)
    else:
        full = None
    
    # Synchronize before broadcast to ensure slow ranks don't miss it
    comm.Barrier()
    # Broadcast results to all ranks
    policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn = comm.bcast(full, root=0)
    return policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn


def _get_solver(method, use_mpi, comm):
    """
    Select appropriate VFI solver based on method and MPI settings.
    
    Returns:
        callable: Either solve_vfi_grid_serial or solve_vfi_grid_mpi
    """
    if method == "VFI_HDGRID" and use_mpi and comm and comm.size > 1:
        return solve_vfi_grid_mpi
    else:
        return solve_vfi_grid_serial


def _solve_vfi_loop(vlu_cntn, model, use_mpi=False, comm=None):
    """
    Drop-in replacement that matches the EGM solver's 7-value return.
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
    beta = float(model.param.beta)
    delta = float(model.param.delta_pb)
    m_bar = float(model.settings_dict["m_bar"])
    N_arg_grid_vfi = model.settings_dict["N_arg_grid_vfi"]

    # --- compiled utility functions -----------------------------
    u_jit = model.num.functions.u_func       # scalar (c,H) → u
    uc_jit = model.num.functions.uc_func      # scalar (c,H) → u_c

    # --- contiguous 3-D continuation value array ----------------
    V_next = np.ascontiguousarray(vlu_cntn, dtype=np.float64)

    expr_str = model.math["functions"]["u_func"]["expr"]
    param_vals = {
        "alpha": model.param.alpha,
        "kappa": model.param.kappa,
        "iota": model.param.iota,
        # add any extra constants referenced in expr_str
    }
    utility_func = build_njit_utility(expr_str, param_vals)
    g_ve_h_ind = model.num.functions.g_ve_h_ind
    h_nxt_ind_array = g_ve_h_ind(H_ind=np.arange(H_grid.size))

    # --- run kernel ---------------------------------------------
    if model.methods["solution"] == "VFI":
        # Handle MPI for serial VFI - solve on root and broadcast
        if use_mpi and comm and comm.size > 1:
            if comm.rank == 0:
                policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn = _solve_vfi_numba(
                    V_next, w_grid, a_grid, H_grid,
                    beta, delta, m_bar,
                    utility_func, h_nxt_ind_array, thorn
                )
                out = (policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn)
            else:
                out = None
            # Broadcast tuple of 5 nd-arrays to all ranks
            out = comm.bcast(out, root=0)
            policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn = out
        else:
            # Serial execution
            policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn = _solve_vfi_numba(
                V_next, w_grid, a_grid, H_grid,
                beta, delta, m_bar,
                utility_func, h_nxt_ind_array, thorn
            )
    elif model.methods["solution"] == "VFI_HDGRID":
        # Choose solver based on MPI settings
        if use_mpi and comm and comm.size > 1:
            policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn = solve_vfi_grid_mpi(
                vlu_cntn, model, comm, use_mpi
            )
        else:
            policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn = solve_vfi_grid_serial(
                V_next, w_grid, a_grid, H_grid,
                beta, delta, m_bar,
                utility_func, h_nxt_ind_array, thorn, n_grid=N_arg_grid_vfi
            )
    



    # The EGM solver returns an average UE time and grid dicts.
    # For VFI these have no meaning, so we stub them out.
    avg_ue_time = 0.0

    # fill in empty egm grids as in the egm code so plotter does not error
    unrefined_grids = {k: {} for k in ('e', 'Q', 'c', 'a')}
    refined_grids = {k: {} for k in ('e', 'Q', 'c', 'a', 'lambda_')}
    egm_grids = {"unrefined": unrefined_grids, "refined": refined_grids}

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
        """Transforms decision state & policy to continuation state (assets).

        This operator calculates the end-of-period assets based on the
        decision-period state and the consumption policy. The primary purpose
        is to map the decision state (w, H_nxt, y) and policy C(w, H_nxt, y)
        to the next-period asset state a_nxt.

        In this simplified version, it's assumed that the value and marginal
        utility (lambda) propagation is handled by other parts of the model
        framework, particularly the backward solution steps. Therefore, this
        operator focuses solely on the state transformation for assets.

        Args:
            perch_data: A dictionary or Solution object containing data from
                        the source 'dcsn' perch. Expected to hold the
                        consumption 'policy' and relevant state grids if needed
                        (though currently commented out as it returns an empty dict).

        Returns:
            dict: An empty dictionary. This operator, in its current form,
                  does not populate any fields in the target 'cntn' perch
                  as it assumes value/lambda mapping is done elsewhere.
                  A more complete implementation might return {'a_nxt': a_nxt_array}.
        """
        # policy = perch_data["policy"] # Consumption policy C(w, H_nxt, y)
        # w_grid = stage.dcsn.grid.w

        # Minimal implementation: returns empty dict assuming value/lambda
        # mapping is not done by this specific forward operator.
        # The framework connects states, value calculation happens in backward steps.
        return {}

    return operator
