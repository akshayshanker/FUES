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


from dc_smm.fues.helpers.math_funcs import interp_as, correct_jumps1d
from dc_smm.uenvelope import EGM_UE as egm_ue_global



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
                 T=60, padding_mbar= 0):

        self.grid_size = grid_size
        self.r, self.R = r, 1 + r
        self.beta = beta
        self.delta = delta
        self.smooth_sigma = smooth_sigma
        self.b = b
        self.T = T
        self.y = y
        self.grid_max_A = grid_max_A
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
    padding_mbar = cp.padding_mbar

    def EGM_UE(endog_grid,
                vf_work_t_inv,
                v_nxt_raw,
                sigma_work_t_inv,
                asset_grid_A,
                del_a_unrefined,
                m_bar=2,
                method='DCEGM', padding_mbar=0):
        """Call the shared `helpers.egm_upper_envelope.EGM_UE` façade.
        """

        @njit(cache=True)
        def u_interp(c, *args):  # Accept and ignore any extra arguments passed by the solver
            return u(c)

        refined, _, _ = egm_ue_global(
            endog_grid,                # x_dcsn_hat
            vf_work_t_inv,             # qf_hat
            v_nxt_raw,             # v_nxt_raw (unused for FUES/DCEGM/RFC)
            sigma_work_t_inv,          # c_hat
            asset_grid_A,              # a_hat
            asset_grid_A,              # w_grid (evaluation grid)
            du,                        # uc_func_partial
            {"func": u, "args": {}}, # u_func placeholder
            ue_method=method.upper(),
            m_bar=m_bar,
            lb=4,
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
    def _worker_solver_egm_step_jit(uc_prime_work, uc_pprime_dcp_work, VF_prime_work):
        """
        Performs the initial EGM step for the worker problem.
        This part is JIT-compiled for performance.
        """
        sigma_work_t_inv = np.zeros(grid_size)
        vf_work_t_inv = np.zeros(grid_size)
        endog_grid = np.zeros(grid_size)
        del_a_unrefined = np.zeros(grid_size)

        for i in range(grid_size):
            uc_prime = beta * R * uc_prime_work[i]
            c_t = uc_inv(uc_prime)
            vf_work_t_inv[i] = u(c_t) + beta * VF_prime_work[i] - delta
            sigma_work_t_inv[i] = c_t
            endog_grid[i] = c_t + asset_grid_A[i]
            del_a_unrefined[i] = R * ddu(c_t) / (ddu(c_t) + beta * R * uc_pprime_dcp_work[i])
        
        return sigma_work_t_inv, vf_work_t_inv, endog_grid, del_a_unrefined

    @njit
    def _worker_solver_final_step_jit(egrid1, vf_clean, sigma_clean, dela_clean,
                                      min_a_val, VF_prime_work,
                                      sigma_ret_t, vf_ret_t, dela_ret_t):
        """
        Performs interpolation and discrete choice after the upper envelope.
        This part is JIT-compiled for performance.
        """
        asset_grid_wealth = R * asset_grid_A + y
        
        vf_work_t = interp_as(egrid1, vf_clean, asset_grid_wealth)
        sigma_work_t = interp_as(egrid1, sigma_clean, asset_grid_wealth)
        dela_work_t = interp_as(egrid1, dela_clean, asset_grid_wealth)
        
        # Apply jump correction to smooth out discontinuities
        gradient_jump_threshold = 10  # This threshold can be adjusted
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

        if smooth_sigma == 0:
            work_choice = vf_work_t > vf_ret_t
        else:
            exp_v_work = np.exp(vf_work_t / smooth_sigma)
            exp_v_ret = np.exp(vf_ret_t / smooth_sigma)
            work_choice = exp_v_work / (exp_v_ret + exp_v_work)
            work_choice = np.where(np.isnan(work_choice) | np.isinf(work_choice), 0, work_choice)

        sigma_t = work_choice * sigma_work_t + (1 - work_choice) * sigma_ret_t
        sigma_t[np.where(sigma_t < 0.0001)] = 0.0001
        
        vf_t = work_choice * vf_work_t + (1 - work_choice) * vf_ret_t
        dela_t = work_choice * dela_work_t + (1 - work_choice) * dela_ret_t
        
        uc_t = du(sigma_t)
        uddca_t = ddu(sigma_t) * (R - dela_t)
        
        return uc_t, vf_t, sigma_t, uddca_t

    def worker_solver(uc_prime_work,
                uc_pprime_dcp_work, 
                VF_prime_work,
                sigma_ret_t,
                vf_ret_t,
                dela_ret_t, 
                t, m_bar, method = 'FUES'):
        """
        Orchestrates the time t policy generation for a worker.
        """
        time_start_all = time.time()
        
        # Step 1: JIT-compiled EGM step
        sigma_work_t_inv, vf_work_t_inv, endog_grid, del_a_unrefined = \
            _worker_solver_egm_step_jit(uc_prime_work, uc_pprime_dcp_work, VF_prime_work)
        
        min_a_val = endog_grid[0]

        # Step 2: Upper-envelope (remains in Python mode)
        time_start_fues = time.time()
        egrid1, vf_clean, sigma_clean, a_prime_clean, dela_clean = EGM_UE(
            endog_grid, vf_work_t_inv, beta * VF_prime_work - delta, sigma_work_t_inv, 
            asset_grid_A, del_a_unrefined, m_bar=1, method=method, padding_mbar=padding_mbar)
        time_end_fues = time.time()
        
        # Step 3: JIT-compiled final interpolation and discrete choice
        uc_t, vf_t, sigma_t, uddca_t = \
            _worker_solver_final_step_jit(egrid1, vf_clean, sigma_clean, dela_clean,
                                          min_a_val, VF_prime_work,
                                          sigma_ret_t, vf_ret_t, dela_ret_t)

        time_end_all = time.time()
        time_fues = time_end_fues - time_start_fues
        time_all = time_end_all - time_start_all

        return uc_t, sigma_work_t_inv, vf_t, vf_work_t_inv,\
            endog_grid, sigma_t, uddca_t, del_a_unrefined, time_fues, time_all


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
            results = worker_solver(
                next_worker_cons_derivative, next_euler_derivative, next_worker_value,
                retiree_consumption[age, :], retiree_values[age, :],
                retiree_asset_derivatives[age, :], age, 2, method = method)
            
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

        average_times = [np.mean(UE_times), np.mean(all_times)]

        return (worker_endog_grid, worker_unrefined_values, worker_refined_values, 
                worker_unrefined_consumption, worker_refined_consumption, 
                asset_pol_derivative_unrefined, average_times)

    return retiree_solver, worker_solver, iter_bell