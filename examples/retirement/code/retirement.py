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
import matplotlib.pyplot as plt


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
    """
    Operator takes in a RetirementModel and returns functions
    to solve the model.

    Parameters
    ----------
    rm: RetirementModel
                    instance of retirement model

    Returns
    -------

    Ts_ret: callabe
                    Solver for retirees using EGM
    Ts_work: callable
                    Solver for workers using EGM

    """

    # unpack parameters from class
    beta, delta = cp.beta, cp.delta
    asset_grid_A = cp.asset_grid_A
    grid_max_A = cp.grid_max_A
    u, du, uc_inv = cp.u, cp.du, cp.uc_inv
    ddu = cp.ddu
    y = cp.y
    smooth_sigma = cp.smooth_sigma
    grid_size = cp.grid_size

    R = cp.R
    b = cp.b
    T = cp.T
    m_bar = cp.m_bar
    padding_mbar = cp.padding_mbar

    def EGM_UE(endog_grid,
                q_cntn_hat,
                v_cntn,
                c_cntn_hat,
                a_hat,
                del_a_unrefined,
                m_bar=2,
                method='DCEGM', padding_mbar=0):
        """Call the shared `helpers.egm_upper_envelope.EGM_UE`.
        """

        @njit(cache=True)
        def u_interp(c, *args):  # Accept and ignore any extra arguments passed by the solver
            return u(c)

        refined, _, _ = egm_ue_global(
            endog_grid,                # x_dcsn_hat
            q_cntn_hat,             # qf_hat
            v_cntn,             # v_nxt_raw (unused for FUES/DCEGM/RFC)
            c_cntn_hat,                 # c_hat
            a_hat,                      # a_hat
            asset_grid_A,              # w_grid (evaluation grid)
            du,                        # uc_func_partial
            {"func": u, "args": {}}, # u_func placeholder
            ue_method=method.upper(),
            m_bar=m_bar,
            lb=20,
            rfc_radius=0.75,
            rfc_n_iter=40,
        )

        m_ref = refined["x_dcsn_ref"]
        v_ref = refined["v_dcsn_ref"]
        c_ref = refined["kappa_ref"]
        a_ref = refined["x_cntn_ref"]

        if len(m_ref) > 1:
            del_a_ref = np.gradient(a_ref, m_ref)
        else:
            del_a_ref = np.zeros_like(a_ref)

        return m_ref, v_ref, c_ref, a_ref, del_a_ref

    @njit
    def retiree_solver(sigma_prime_ret,
            VF_prime_ret,
            uc_pprime_dcp_ret,
            t):
        """
        Generates time t policy for retiree. JIT-compiled.
        """
        sigma_ret_t_inv = np.zeros(grid_size)
        vf_ret_t_inv = np.zeros(grid_size)
        endog_grid = np.zeros(grid_size)
        dela_a_ret_t_inv = np.zeros(grid_size)

        for i in range(len(asset_grid_A)):
            a_prime = asset_grid_A[i]
            c_prime = sigma_prime_ret[i]
            uc_prime = beta * R * du(c_prime)
            c_t = uc_inv(uc_prime)
            a_t = (c_t + a_prime) / R
            endog_grid[i] = a_t
            sigma_ret_t_inv[i] = c_t
            vf_ret_t_inv[i] = u(c_t) + beta * VF_prime_ret[i]
            dela_a_ret_t_inv[i] = R*ddu(c_t)/(ddu(c_t) + beta * R*uc_pprime_dcp_ret[i])

        min_a_val = endog_grid[0]
        sigma_ret_t = interp_as(endog_grid, sigma_ret_t_inv, asset_grid_A)
        vf_ret_t = interp_as(endog_grid, vf_ret_t_inv, asset_grid_A)
        del_a_ret_t = interp_as(endog_grid, dela_a_ret_t_inv, asset_grid_A)

        constrained_idx = np.where(asset_grid_A <= min_a_val)
        sigma_ret_t[constrained_idx] = asset_grid_A[constrained_idx]
        vf_ret_t[constrained_idx] = u(asset_grid_A[constrained_idx]) + beta * VF_prime_ret[0]
        del_a_ret_t[constrained_idx] = 0
        uc_pprime_dcp_ret = ddu(sigma_ret_t)*(R - del_a_ret_t)

        return sigma_ret_t, vf_ret_t, del_a_ret_t, uc_pprime_dcp_ret

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
                                      min_a_val, VF_prime_work,
                                      sigma_ret_t, vf_ret_t, dela_ret_t):

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
        v   : value at decision perch (Bellman)
        c   : mixed consumption policy
        dv  : marginal value du(c) (MarginalBellman)
        ddv : ddu(c)*(R - da), second-order marginal for upstream EGM
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

        dv = du(c)
        ddv = ddu(c) * (R - da)

        return v, c, dv, ddv

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

        # Step 2: Upper-envelope (remains in Python mode)
        time_start_fues = time.time()
        egrid1, q_cntn, c_cntn, a_prime_clean, dela_clean = EGM_UE(
            endog_grid, q_cntn_hat, beta * v_worker_cntn - delta, cons_cntn_hat,
            asset_grid_A, del_a_unrefined, m_bar=m_bar, method=method, padding_mbar=padding_mbar)
        time_end_fues = time.time()
        
        # Step 3: Approximate worker policy and VF on arvl grid
        # Note we interopolate directly on arvl grid since there is no shock

        v_worker_arvl, c_worker_arvl, dela_work_t = _approx_dcsn_state_functions(egrid1, q_cntn, c_cntn, dela_clean, min_a_val, v_worker_cntn)

        ue_time = time.time() - time_start_fues

        return (v_worker_arvl,   # v_worker[>]
                c_worker_arvl,   # c_worker[>]
                dela_work_t,     # dela_work[>]
                ue_time)         # time taken to solve EGM


    def iter_bell(policy_params, method  = 'FUES'):
        max_age = policy_params.T
        grid_len = policy_params.grid_size

        initial_asset_grid = np.copy(policy_params.asset_grid_A)
        initial_value_func = policy_params.u(initial_asset_grid)
        consumption_derivative_terminal = ddu(initial_asset_grid) * R

        retiree_consumption = np.empty((max_age, grid_len))
        retiree_values = np.empty((max_age, grid_len))
        retiree_asset_derivatives = np.empty((max_age, grid_len))

        worker_unrefined_values = np.empty((max_age, grid_len))
        worker_refined_values = np.empty((max_age, grid_len))
        worker_unrefined_consumption = np.empty((max_age, grid_len))
        worker_endog_grid = np.empty((max_age, grid_len))
        worker_refined_consumption = np.empty((max_age, grid_len))
        asset_pol_derivative_unrefined = np.empty((max_age, grid_len))

        UE_times = np.zeros(max_age)
        all_times = np.zeros(max_age)

        next_consumption, next_value_func, next_cons_derivative = (
            np.copy(initial_asset_grid),
            np.copy(initial_value_func),
            np.copy(consumption_derivative_terminal))

        for i in range(max_age):
            age = int(max_age - i - 1)
            consumption, value, asset_derivative, cons_derivative = (
                retiree_solver(next_consumption, next_value_func, next_cons_derivative, age))
            
            retiree_consumption[age, :] = consumption
            retiree_values[age, :] = value
            retiree_asset_derivatives[age, :] = asset_derivative

            next_consumption, next_value_func, next_cons_derivative = (
                consumption, value, cons_derivative)

        next_worker_value = policy_params.u(initial_asset_grid)
        next_worker_cons_derivative = policy_params.du(initial_asset_grid)
        next_euler_derivative = ddu(initial_asset_grid) * R

        for i in range(max_age):
            age = int(max_age - i - 1)
            results = solver_worker_stage(
                next_worker_cons_derivative, next_euler_derivative, next_worker_value,
                retiree_consumption[age, :], retiree_values[age, :],
                retiree_asset_derivatives[age, :], age, policy_params.m_bar, method=method)
            
            (worker_cons_derivative, unrefined_consumption, worker_value, 
            unrefined_worker_value, endogenous_grid, refined_consumption, 
            euler_derivative, unrefined_asset_derivative, EU_time, total_time) = results

            worker_unrefined_values[age, :] = unrefined_worker_value
            worker_refined_values[age, :] = worker_value
            worker_unrefined_consumption[age, :] = unrefined_consumption
            worker_endog_grid[age, :] = endogenous_grid
            worker_refined_consumption[age, :] = refined_consumption
            asset_pol_derivative_unrefined[age, :] = unrefined_asset_derivative

            next_worker_cons_derivative = worker_cons_derivative
            next_worker_value = worker_value
            next_euler_derivative = euler_derivative
            if i>2:
                UE_times[age] = EU_time
                all_times[age] = total_time

        mask = UE_times > 0
        average_times = [np.mean(UE_times[mask]), np.mean(all_times[mask])]

        return (worker_endog_grid, worker_unrefined_values, worker_refined_values, 
                worker_unrefined_consumption, worker_refined_consumption, 
                asset_pol_derivative_unrefined, average_times)

    return retiree_solver, solver_worker_stage, iter_bell