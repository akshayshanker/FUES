"""Solution of Ishkakov et al (2017) retirement choice
model using FUES-EGM by Dobrescu and Shanker (2022).

Author: Akshay Shanker, University of Sydney, akshay.shanker@me.com.


Todo
----

1. Complete docstrings for key functions
    and relabel policy functions for
    clarity in exposition

"""

import numpy as np
import time
from numba import njit, types
from numba.typed import Dict
from dcsmm.fues.helpers.math_funcs import interp_as, correct_jumps1d
from dcsmm.uenvelope import EGM_UE as egm_ue_global



class RetirementModel:

    """
    A class that stores primitives for the retirement choice model.

    Parameters
    ----------
    r: float
                    interest rate
    beta: float
                    discount rate
    delta: float
                    fixed cost to work
    smooth_sigma: float
                    smoothing parameter
    y: float
                    wage for worker
    b: float
                    lower bound for assets
    grid_max_A: float
                    max liquid asset
    grid_size: int
                    grid size for liquid asset
    T: int
            terminal age

    Attributes
    ----------

    du: callable
                     marginal utility of consumption
    u: callable
                     utility
    uc_inv: callable
                     inverse of marginal utility

    """

    def __init__(self,
                 r=0.02,
                 beta=.945,
                 delta=1,
                 smooth_sigma=0,
                 y=1,
                 b=1e-2,
                 grid_max_A=50,
                 grid_size=50,
                 T=60,
                 m_bar=1.2,
                 padding_mbar=0):

        self.grid_size = grid_size
        self.r, self.R = r, 1 + r
        self.beta = beta
        self.delta = delta
        self.smooth_sigma = smooth_sigma
        self.b = b
        self.T = T
        self.y = y
        self.grid_max_A = grid_max_A
        self.m_bar = m_bar
        self.padding_mbar = padding_mbar

        

        self.asset_grid_A = np.linspace(b, grid_max_A, grid_size)

        self.eulerK = len(self.asset_grid_A)

        # define functions
        @njit
        def du(x):

            return 1 / x

        @njit(cache=True)
        def u(x):

            cons_u = np.log(x)

            return cons_u

        @njit
        def uc_inv(x):

            return 1 / x
        
        @njit 
        def ddu(x):
            return -1/(x**2)

        self.u, self.du, self.uc_inv, self.ddu = u, du, uc_inv, ddu
        

#@njit
def euler(cp,sigma_work):
    
    a_grid = cp.asset_grid_A

    euler = np.zeros((cp.T-1,cp.eulerK))
    euler.fill(np.nan)

    # b. loop over time
    for t in range(cp.T-1):
        for i_a in range(cp.eulerK):
                
                # i. state
                a = a_grid[i_a]
                
                # iii. continuous choice
                c = np.interp(a,a_grid,sigma_work[t])
                a_prime = a*cp.R + cp.y - c 
                
                if a_prime < 0.001 or a_prime>300: continue

                
                c_plus =  np.interp(a,a_grid,sigma_work[t+1])

                # oooo. accumulate
                RHS = cp.beta*cp.R*cp.du(c_plus)    

                # v. euler error
                euler_raw = c - cp.uc_inv(RHS)
                
                euler[t, i_a] = np.log10(np.abs(euler_raw/c)+1e-16)
    
    return np.nanmean(euler)


def consumption_deviation(cp, c_solution, c_true, a_grid_true):
    """Compute mean log absolute deviation from high-resolution 'true' solution.

    Uses the same metric as Euler error: log10(|c - c_true| / c_true).
    Compares a solution computed on a coarser grid to a high-resolution
    reference solution (e.g., DCEGM with 20,000 points).

    Parameters
    ----------
    cp : RetirementModel
        Model parameters for the solution being tested.
    c_solution : ndarray (T x grid_size)
        Consumption policy from method being tested.
    c_true : ndarray (T x true_grid_size)
        High-resolution "true" solution.
    a_grid_true : ndarray
        Asset grid for the true solution.

    Returns
    -------
    float
        Mean log10 absolute relative deviation across periods and grid points.
    """
    a_grid = cp.asset_grid_A
    T = cp.T

    deviations = np.zeros((T - 1, len(a_grid)))
    deviations.fill(np.nan)

    for t in range(T - 1):
        # Interpolate true solution to the test grid
        c_true_interp = np.interp(a_grid, a_grid_true, c_true[t])
        c_test = c_solution[t]

        for i_a in range(len(a_grid)):
            if c_true_interp[i_a] > 1e-10 and c_test[i_a] > 1e-10:
                # Same metric as Euler: log10(|c - c_true| / c_true)
                rel_error = np.abs(c_test[i_a] - c_true_interp[i_a]) / c_true_interp[i_a]
                deviations[t, i_a] = np.log10(rel_error + 1e-16)

    return np.nanmean(deviations)


def Operator_Factory(cp):
    """Build stage solvers (movers) for the retirement model.

    Parameters
    ----------
    cp : RetirementModel
        Model instance with calibrated parameters and grids.

    Returns
    -------
    dict
        ``{'retiree': solver_retiree_stage,
           'worker':  solver_worker_stage,
           'branch':  lab_mkt_choice_stage}``
    """

    # unpack parameters from class
    beta, delta = cp.beta, cp.delta
    asset_grid_A = cp.asset_grid_A
    u, du, uc_inv = cp.u, cp.du, cp.uc_inv
    ddu = cp.ddu
    y = cp.y
    smooth_sigma = cp.smooth_sigma
    grid_size = cp.grid_size
    R = cp.R
    m_bar = cp.m_bar

    @njit
    def solver_retiree_stage(c_cntn,        # c[>]: consumption at continuation perch
                             v_cntn,        # V[>]: value at continuation perch
                             dlambda_cntn,  # dlambda[>]: second-order marginal at cntn
                             t):
        """Retiree consumption stage solver (EGM, no upper envelope).

        Implements the RetireeConsumption stage:
          InvEuler:                c = (beta * R * du(c[>]))^{-1}
          cntn_to_dcsn_transition: a_ret = (c + b_ret) / R
          Bellman:                 V = u(c) + beta * V[>]
          MarginalBellman:         dlambda = ddu(c) * (R - da)

        Returns
        -------
        c_arvl       : consumption on arrival grid
        v_arvl       : value on arrival grid
        da_arvl      : da'/da on arrival grid
        dlambda_arvl : second-order marginal for upstream connector
        """
        c_dcsn_hat = np.zeros(grid_size)
        v_dcsn_hat = np.zeros(grid_size)
        endog_grid = np.zeros(grid_size)
        da_dcsn_hat = np.zeros(grid_size)

        for i in range(len(asset_grid_A)):
            b_ret = asset_grid_A[i]              # poststate grid point
            c_next = c_cntn[i]
            # InvEuler
            uc_cntn = beta * R * du(c_next)
            c_dcsn = uc_inv(uc_cntn)
            # cntn_to_dcsn_transition: a_ret = (c + b_ret) / R
            a_ret_hat = (c_dcsn + b_ret) / R
            endog_grid[i] = a_ret_hat
            c_dcsn_hat[i] = c_dcsn
            # Bellman
            v_dcsn_hat[i] = u(c_dcsn) + beta * v_cntn[i]
            # Asset derivative
            da_dcsn_hat[i] = R * ddu(c_dcsn) / (ddu(c_dcsn) + beta * R * dlambda_cntn[i])

        # Interpolate from endogenous to exogenous arrival grid
        min_a_val = endog_grid[0]
        c_arvl = interp_as(endog_grid, c_dcsn_hat, asset_grid_A)
        v_arvl = interp_as(endog_grid, v_dcsn_hat, asset_grid_A)
        da_arvl = interp_as(endog_grid, da_dcsn_hat, asset_grid_A)

        # Constrained region: consume everything
        constrained_idx = np.where(asset_grid_A <= min_a_val)
        c_arvl[constrained_idx] = asset_grid_A[constrained_idx]
        v_arvl[constrained_idx] = u(asset_grid_A[constrained_idx]) + beta * v_cntn[0]
        da_arvl[constrained_idx] = 0

        # MarginalBellman: dlambda = ddu(c) * (R - da)
        dlambda_arvl = ddu(c_arvl) * (R - da_arvl)

        return c_arvl, v_arvl, da_arvl, dlambda_arvl

    @njit
    def _invert_euler(lambda_worker_cntn, dlambda_worker_cntn, v_worker_cntn):
        """Inverse Euler step for the worker consumption stage.

        Given continuation-perch marginals, invert to get
        decision-perch consumption, value, endogenous grid,
        and asset derivative (all on continuation grid, pre-UE).
        """
        cons_cntn_hat = np.zeros(grid_size)
        q_cntn_hat = np.zeros(grid_size)
        endog_grid = np.zeros(grid_size)
        del_a_unrefined = np.zeros(grid_size)

        for i in range(grid_size):
            uc_cntn = beta * R * lambda_worker_cntn[i]
            c_cntn = uc_inv(uc_cntn)
            q_cntn_hat[i] = u(c_cntn) + beta * v_worker_cntn[i] - delta
            cons_cntn_hat[i] = c_cntn
            endog_grid[i] = c_cntn + asset_grid_A[i]
            del_a_unrefined[i] = R * ddu(c_cntn) / (ddu(c_cntn) + beta * R * dlambda_worker_cntn[i])

        return cons_cntn_hat, q_cntn_hat, endog_grid, del_a_unrefined


    @njit
    def _approx_dcsn_state_functions(egrid1, vf_clean, sigma_clean, dela_clean,
                                      min_a_val, VF_prime_work):

        asset_grid_wealth = R * asset_grid_A + y
        
        vf_work_t = interp_as(egrid1, vf_clean, asset_grid_wealth)
        sigma_work_t = interp_as(egrid1, sigma_clean, asset_grid_wealth)
        dela_work_t = interp_as(egrid1, dela_clean, asset_grid_wealth)
        
        # Apply jump correction to smooth out discontinuities
        gradient_jump_threshold = 2  # This threshold can be adjusted
        policy_value_dict = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64[:]
        )
        policy_value_dict['sigma'] = sigma_work_t
        policy_value_dict['dela'] = dela_work_t
        
        vf_work_t_corrected, corrected_policies = correct_jumps1d(
            vf_work_t, asset_grid_wealth, gradient_jump_threshold, policy_value_dict
        )
        vf_work_t = vf_work_t_corrected
        sigma_work_t = corrected_policies['sigma']
        dela_work_t = corrected_policies['dela']

        constrained_indices = np.where(asset_grid_wealth < min_a_val)
        sigma_work_t[constrained_indices] = asset_grid_wealth[constrained_indices] - asset_grid_A[0]
        vf_work_t[constrained_indices] = u(asset_grid_wealth[constrained_indices]) + beta * VF_prime_work[0] - delta
        dela_work_t[constrained_indices] = 0

        return vf_work_t, sigma_work_t, dela_work_t

    @njit
    def lab_mkt_choice_stage(
        v_cntn_work,        # V[>][work]:  value from worker cons stage
        v_cntn_ret,         # V[>][retire]: value from retiree cons stage
        c_cntn_work,        # c[>][work]:  consumption policy, work branch
        c_cntn_ret,         # c[>][retire]: consumption policy, retire branch
        da_cntn_work,       # da[>][work]: da'/dm, work branch
        da_cntn_ret,        # da[>][retire]: da'/dm, retire branch
    ):
        """Branching-stage mover: discrete work/retire choice.

        Aggregates branch-keyed continuation values and policies into
        decision-perch objects via hard max (smooth_sigma=0) or logit.

        Returns
        -------
        v       : value at decision perch (Bellman)
        c       : mixed consumption policy
        lambda_ : marginal value du(c) (MarginalBellman)
        dlambda : ddu(c)*(R - da), second-order marginal for upstream EGM
        """

        if smooth_sigma == 0:
            work_prob = v_cntn_work > v_cntn_ret
        else:
            exp_v_work = np.exp(v_cntn_work / smooth_sigma)
            exp_v_ret = np.exp(v_cntn_ret / smooth_sigma)
            work_prob = exp_v_work / (exp_v_ret + exp_v_work)
            work_prob = np.where(np.isnan(work_prob) | np.isinf(work_prob), 0, work_prob)

        c = work_prob * c_cntn_work + (1 - work_prob) * c_cntn_ret
        c[np.where(c < 0.0001)] = 0.0001

        v = work_prob * v_cntn_work + (1 - work_prob) * v_cntn_ret
        da = work_prob * da_cntn_work + (1 - work_prob) * da_cntn_ret

        lambda_arvl = du(c)
        dlambda_arvl = ddu(c) * (R - da)

        return v, c, lambda_arvl, dlambda_arvl

    def solver_worker_stage(lambda_worker_cntn, #lambda_worker[>]
                            dlambda_worker_cntn,  #dlambda_worker[>]
                            v_worker_cntn,        #v_worker[>]
                            method = 'FUES'):
        """
        Orchestrates the time t policy generation for a worker.
        """

        # Step 1: Invert Euler equation to get unrefined consumption, q function, 
        # dcsn state and derivative of asset function defined on continuation state 
        cons_cntn_hat, q_cntn_hat, endog_grid, del_a_unrefined = \
            _invert_euler(lambda_worker_cntn, dlambda_worker_cntn, v_worker_cntn)
        
        min_a_val = endog_grid[0]

        # Step 2: Upper-envelope via egm_ue_global
        time_start_fues = time.time()
        refined, _, _ = egm_ue_global(
            endog_grid,                          # x_dcsn_hat
            q_cntn_hat,                          # qf_hat
            beta * v_worker_cntn - delta,        # v_nxt_raw
            cons_cntn_hat,                       # c_hat
            asset_grid_A,                        # a_hat
            asset_grid_A,                        # w_grid (evaluation grid)
            du,                                  # uc_func_partial
            {"func": u, "args": {}},             # u_func placeholder
            ue_method=method.upper(),
            m_bar=m_bar,
            lb=20,
            rfc_radius=0.75,
            rfc_n_iter=40,
        )
        egrid1 = refined["x_dcsn_ref"]
        q_cntn = refined["v_dcsn_ref"]
        c_cntn = refined["kappa_ref"]
        a_prime_clean = refined["x_cntn_ref"]
        dela_clean = np.gradient(a_prime_clean, egrid1) if len(egrid1) > 1 \
            else np.zeros_like(a_prime_clean)
        time_end_fues = time.time()
        
        # Step 3: Approximate worker policy and VF on arvl grid
        # Note we interopolate directly on arvl grid since there is no shock

        v_worker_arvl, c_worker_arvl, dela_work_t = _approx_dcsn_state_functions(egrid1, q_cntn, c_cntn, dela_clean, min_a_val, v_worker_cntn)

        ue_time = time.time() - time_start_fues

        return (v_worker_arvl,        # v_worker[>]
                c_worker_arvl,        # c_worker[>]
                dela_work_t,          # dela_work[>]
                ue_time,              # time taken for UE step
                cons_cntn_hat,        # unrefined consumption (pre-UE)
                q_cntn_hat,           # unrefined value (pre-UE)
                endog_grid,           # endogenous grid (pre-UE)
                del_a_unrefined)      # unrefined asset derivative


    return {
        'retiree': solver_retiree_stage,
        'worker':  solver_worker_stage,
        'branch':  lab_mkt_choice_stage,
    }