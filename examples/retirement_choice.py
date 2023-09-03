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
from numba import jit
import time
import dill as pickle
from numba import njit, prange

from FUES.FUES import FUES

from FUES.math_funcs import interp_as, upper_envelope

from HARK.interpolation import LinearInterp
#from HARK.dcegm import calc_segments, calc_multiline_envelope, calc_cross_points
from HARK.dcegm import calc_nondecreasing_segments, upper_envelope, calc_linear_crossing
from interpolation import interp


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
                 T=60):

        self.grid_size = grid_size
        self.r, self.R = r, 1 + r
        self.beta = beta
        self.delta = delta
        self.smooth_sigma = smooth_sigma
        self.b = b
        self.T = T
        self.y = y
        self.grid_max_A = grid_max_A

        self.asset_grid_A = np.linspace(b, grid_max_A, grid_size)

        # define functions
        @njit
        def du(x):

            return 1 / x

        @njit
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

    @njit
    def Ts_ret(sigma_prime_ret,
            VF_prime_ret,
            uc_pprime_dcp_ret,
            t):
        """
        Generates time t policy for retiree

        Parameters
        ----------
        sigma_prime_ret : 1D array
                        t+1 period consumption function
        VF_prime_ret : 1D array
                    t+1 period value function (retired)
        uc_pprime_dcp_ret : 1D array
                            derivative of the consumption policy with respect to asset
        t : int
            Age

        Returns
        -------
        sigma_ret_t: 1D array
                    consumption policy on assets at start of time t
        vf_ret_t: 1D array
                time t value
        del_a_ret_t: 1D array
                    derivative of the asset policy with respect to asset
        uc_pprime_dcp_ret: 1D array
                        derivative of the consumption policy with respect to asset

        Notes
        -----
        Whether or not to work decision in time t is made
        at the start of time t. Thus, if agent chooses to retire,
        total cash at hand will be a(t)(1+r).
        """

        # Empty grids for time t consumption, vf, enog grid
        sigma_ret_t_inv = np.zeros(grid_size)
        vf_ret_t_inv = np.zeros(grid_size)
        endog_grid = np.zeros(grid_size)
        dela_a_ret_t_inv = np.zeros(grid_size)

        # loop over exogenous grid to create endogenous grid
        for i in range(len(asset_grid_A)):
            a_prime = asset_grid_A[i]
            c_prime = sigma_prime_ret[i]
            uc_prime = beta * R * du(c_prime)

            # current period consumption using inverse of next period MUC
            c_t = uc_inv(uc_prime)

            # evaluate endogenous grid points
            a_t = (c_t + a_prime) / R
            endog_grid[i] = a_t

            # evaluate candidate policy function and value function
            sigma_ret_t_inv[i] = c_t
            vf_ret_t_inv[i] = u(c_t) + beta * VF_prime_ret[i]

            # derivative of current period asset policy
            dela_a_ret_t_inv[i] = R*ddu(c_t)/(ddu(c_t) + beta * R*uc_pprime_dcp_ret[i])

        # min value of time t assets before which t+1 constraint binds
        min_a_val = endog_grid[0]

        # interpolate policy and value function on even grid
        sigma_ret_t = interp_as(endog_grid, sigma_ret_t_inv, asset_grid_A)
        vf_ret_t = interp_as(endog_grid, vf_ret_t_inv, asset_grid_A)
        del_a_ret_t = interp_as(endog_grid, dela_a_ret_t_inv, asset_grid_A)

        # impose lower bound on liquid assets where a_t < min_a_val
        sigma_ret_t[np.where(asset_grid_A <= min_a_val)] = asset_grid_A[np.where(asset_grid_A <= min_a_val)]
        vf_ret_t[np.where(asset_grid_A <= min_a_val)] = u(asset_grid_A[np.where(asset_grid_A <= min_a_val)]) + beta * VF_prime_ret[0]
        del_a_ret_t[np.where(asset_grid_A <= min_a_val)] = 0
        uc_pprime_dcp_ret = ddu(sigma_ret_t)*(R - del_a_ret_t)

        return sigma_ret_t, vf_ret_t, del_a_ret_t, uc_pprime_dcp_ret


    #@njit
    def Ts_work(uc_prime_work,
                uc_pprime_dcp_work, 
                VF_prime_work,
                sigma_ret_t,
                vf_ret_t,
                dela_ret_t, 
                t, m_bar):
        """
        Generates time t policy for worker.

        Parameters
        ----------
        uc_prime_work : 1D array
            t+1 period MUC on t+1 state if work choice = 1 at t.
        VF_prime_work : 1D array
            t+1 period VF on t+1 state if work choice = 1 at t.
        uc_pprime_dcp_work : 1D array
            t+1 period derivative of RHS of Euler wrt to t+1 state.
        sigma_ret_t : 1D array
            t consumption on t+1 state if work choice at = 0.
        vf_ret_t : 1D array
            t VF if work decision at t = 0.
        dela_ret_t : 1D array
            t derivative of asset policy if work decision at t = 0.
        t : int
            Age.
        m_bar : float 
            Jump detection threshold for FUES.

        Returns
        -------
        uc_t : 1D array
            Unconditioned time t MUC on assets(t).
        sigma_work_t_inv : 1D array
            Unrefined consumption for worker at time t on wealth.
        vf_t : 1D array
            Unconditioned time t value on assets(t).
        vf_work_t_inv : 1D array
            Unrefined time t value for worker at time t on wealth.
        endog_grid : 1D array
            Unrefined endogenous grid.
        sigma_work_t : 1D array
            Refined work choice for worker at time t on start of time t assets.
        del_a : 1D array
            Derivative of worker asset policy wrt to assets.

        uddca_t : 1D array
            Derivative of RHS of the time t Euler wrt to t assets.

        Notes
        -----
        - uddca_t gives the derivative of the RHS of the Euler wrt to t assets,
        not t-1 assets. This is the derivative wrt to start of time t period 
        assets.
        
        - uddca_t is evaluated unconditional on the time t choice of whether or 
        not to work in t+1.

        - del_a_unrefined is the derivative of the unrefined asset policy wrt to 
        start of time t for someone choosing to work in t+1, defined on the 
        unrefined endogenous grid.

        - del_a, not returned by function, is the derivative of the refined asset 
        policy wrt to start of time t.
        """

        # Empty grids for time t consumption, vf, enog grid
        time_start_all = time.time()
        sigma_work_t_inv = np.zeros(grid_size)
        vf_work_t_inv = np.zeros(grid_size)
        endog_grid = np.zeros(grid_size)
        del_a_unrefined = np.zeros(grid_size)

        # Loop through each time T+1 state in the exogenous grid
        for i in range(len(asset_grid_A)):
            # marginal utility of next period consumption on T+1 state
            uc_prime = beta * R * uc_prime_work[i]
            c_t = uc_inv(uc_prime)

            # current period value function on T+1 state
            vf_work_t_inv[i] = u(c_t) + beta * VF_prime_work[i] - delta
            sigma_work_t_inv[i] = c_t

            # endogenous grid of current period wealth
            endog_grid[i] = c_t + asset_grid_A[i]

            # derivative of current period asset policy
            del_a_unrefined[i] = R * ddu(c_t)/(ddu(c_t) + beta * R * uc_pprime_dcp_work[i])
            #print(uc_pprime_dcp_work[i])

        min_a_val = endog_grid[0]

        # wealth grid points located at time t asset grid points
        asset_grid_wealth = R * asset_grid_A + y

        # remove sub-optimal points using FUES
        time_start_fues = time.time()
        egrid1, vf_clean, sigma_clean, a_prime_clean, dela_clean = FUES(
            endog_grid, vf_work_t_inv, sigma_work_t_inv, asset_grid_A, del_a_unrefined, m_bar=2)
        time_end_fues = time.time()

        # interpolate on even start of period t asset grid for worker
        vf_work_t = interp_as(egrid1, vf_clean, asset_grid_wealth)
        sigma_work_t = interp_as(egrid1, sigma_clean, asset_grid_wealth)
        dela_work_t = interp_as(egrid1, dela_clean, asset_grid_wealth)

        #pos = np.where(np.abs(np.diff(a_prime_clean)/np.diff(egrid1)) > 1)[0] + 1
        #for p in pos:
        #    sigma_clean[p+1] = sigma_clean[p]
        #    vf_clean[p+1] = vf_clean[p]

        # binding t+1 asset constraint points
        sigma_work_t[np.where(asset_grid_wealth < min_a_val)]\
            = asset_grid_wealth[np.where(asset_grid_wealth < min_a_val)] - asset_grid_A[0]
        vf_work_t[np.where(asset_grid_wealth < min_a_val)]\
            = u(asset_grid_wealth[np.where(asset_grid_wealth < min_a_val)])\
            + beta * VF_prime_work[0] - delta

        # asset policy derivative zero at constrained points 
        dela_work_t[np.where(asset_grid_wealth < min_a_val)] = 0

        # make current period discrete choices and unconditioned policies
        if smooth_sigma == 0:
            work_choice = vf_work_t > vf_ret_t
        else:
            work_choice = np.exp(vf_work_t/smooth_sigma)/((np.exp(vf_ret_t/smooth_sigma)\
                            + np.exp(vf_work_t/smooth_sigma)))

        sigma_t = work_choice * sigma_work_t + (1 - work_choice) * sigma_ret_t
        vf_t = work_choice * (vf_work_t) + (1 - work_choice) * vf_ret_t
        dela_t = work_choice * (dela_work_t) + (1 - work_choice) * dela_ret_t
        uc_t = du(sigma_t)
        uddca_t = ddu(sigma_t)*(R - dela_t)

        time_end_all = time.time()
        time_fues = time_end_fues - time_start_fues 
        time_all = time_end_all - time_start_all

        return uc_t, sigma_work_t_inv, vf_t, vf_work_t_inv,\
            endog_grid, sigma_t, uddca_t,del_a_unrefined, time_fues, time_all


    def iter_bell(cp):
        T = cp.T
        grid_size = cp.grid_size

        # Values at terminal states
        sigma_prime_terminal = np.copy(cp.asset_grid_A)
        v_prime_terminal = cp.u(sigma_prime_terminal)
        uc_pprime_dcp_terminal = ddu(sigma_prime_terminal) * R

        # Empty grids for retirees for each t
        sigma_retirees = np.empty((cp.T, cp.grid_size))
        vf_retirees = np.empty((cp.T, cp.grid_size))
        dela_retirees = np.empty((cp.T, cp.grid_size))

        # Empty grids for workers for each t
        vf_work_unref = np.empty((cp.T, cp.grid_size))
        vf_refined = np.empty((cp.T, cp.grid_size))
        c_worker_unref = np.empty((cp.T, cp.grid_size))
        e_grid_worker_unref = np.empty((cp.T, cp.grid_size))
        c_refined = np.empty((cp.T, cp.grid_size))
        dela_unrefined = np.empty((cp.T, cp.grid_size))

        iter_time_age_fues = np.zeros(cp.T)
        iter_time_age_all = np.zeros(cp.T)

        # Step 1: Solve retiree policy
        sigma_prime_ret, VF_prime_ret, uc_pprime_dcp_ret = np.copy(sigma_prime_terminal), np.copy(v_prime_terminal), np.copy(uc_pprime_dcp_terminal)

        # backward induction to solve for retirees each period
        for i in range(T):
            age = int(T - i - 1)
            sigma_ret_t, vf_ret_t, del_a_ret_t, uc_pprime_dcp_ret_t = Ts_ret(sigma_prime_ret, VF_prime_ret, uc_pprime_dcp_ret, age)
            sigma_retirees[age, :] = sigma_ret_t
            vf_retirees[age, :] = vf_ret_t
            dela_retirees[age, :] = del_a_ret_t
            sigma_prime_ret, VF_prime_ret, uc_pprime_dcp_ret = sigma_ret_t, vf_ret_t, uc_pprime_dcp_ret_t

        # Step 2: Solve general policy for workers
        VF_prime_work = cp.u(sigma_prime_terminal)
        uc_prime_work = cp.du(sigma_prime_terminal)
        uc_pprime_dcp_work = ddu(sigma_prime_terminal) * R 

        # backward induction to solve for workers each period
        for i in range(T):
            age = int(T - i - 1)
            uc_t, sigma_work_t_inv, vf_t, vf_work_t_inv, endog_grid, cons_pol, uddca_t, dela_t_inv, time_fues, time_all = Ts_work(
                uc_prime_work, uc_pprime_dcp_work, VF_prime_work,
                sigma_retirees[age, :], vf_retirees[age, :],
                dela_retirees[age, :], age, 2)

            # store the grids for plotting
            vf_work_unref[age, :] = vf_work_t_inv
            vf_refined[age, :] = vf_t
            c_worker_unref[age, :] = sigma_work_t_inv
            e_grid_worker_unref[age, :] = endog_grid
            c_refined[age, :] = cons_pol
            dela_unrefined[age, :] = dela_t_inv

            # next period inputs to solver
            uc_prime_work = uc_t
            VF_prime_work = vf_t
            uc_pprime_dcp_work = uddca_t

            iter_time_age_fues[age] = time_fues
            iter_time_age_all[age] = time_all

        mean_times = np.zeros(2)
        mean_times[1] = np.mean(iter_time_age_all)
        mean_times[0] = np.mean(iter_time_age_fues)

        return e_grid_worker_unref, vf_work_unref, vf_refined, c_worker_unref, c_refined,dela_unrefined, mean_times



    return Ts_ret, Ts_work, iter_bell
