"""

Module that contains class and solvers for discrete
choice housing model (Fella, 2014).

Author: Akshay Shanker, University of Sydney, a.shanker@unsw.edu.au

"""
import numpy as np
import time
import dill as pickle
from sklearn.utils.extmath import cartesian
from numba import njit, prange
import matplotlib.pylab as pl
from quantecon.optimize.scalar_maximization import brent_max
from numba import njit
from numba.typed import Dict
from numba.core import types
from quantecon import MarkovChain



import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
cwd = os.getcwd()
sys.path.append('..')
os.chdir(cwd)
from FUES.FUES import FUES
from FUES.RFC_simple import rfc
from FUES.DCEGM import dcegm
from FUES.math_funcs import interp_as

class ConsumerProblem:
    """
    A class that stores primitives for the consumer problem for
    model with fixed adjustment cost and discrete housing grid. The
    income process is assumed to be a finite state Markov chain.

    Parameters
    ----------
    r : float, optional(default=0.01)
            A strictly positive scalar giving the interest rate
    r_H : float
             return on housing
    beta : float, optional(default=0.96)
             The discount factor, must satisfy (1 + r) * beta < 1
    delta: float
            Depreciation rate
    Pi : array_like 
            A 2D NumPy array giving the Markov matrix for shocks
    z_vals : 1D array
            The state space of shocks 
    b : float
            The borrowing constraint lower bound
    grid_max_A: float
                 Max liquid asset
    grid_size : int
                 Liq. asset grid size 
    gamma_1: float
    phi : float
            Ratio of h_prime that becomes fixed adjustment cost
    xi: float
    kappa: float
            Constant for housing in utility
    theta: float
            Non-durable consumption share
    iota: float
            Housing utility constant shift term

    u : callable, optional(default=np.log)
            The utility function
    du : callable, optional(default=lambda x: 1/x)
            The derivative of u

    Attributes
    ----------

    X_all: 3D array
            full state-sapce of shocks, assets, housing
    X_all_big: 4D array
                Conditioned state space
                full state-space + housing choice made at t
    X_exog: 3D array 
                small state-space of discrete states
                shocks, housing at t and housing choice made at t
    u: callable
        utility 
    uc_inv: callable
             inverse of marginal utility of cons. 
    du: callable
         marginal utility of consumption


    Notes
    ----

    To understand the grids above, 
    recall agent enters with housing state H(t). 
    Then agent makes a housing choice H(t+1). 


    """

    def __init__(self,
                 r=0.074,
                 r_H=.1,
                 beta=.945,
                 delta=0.1,
                 Pi=((0.09, 0.91), (0.06, 0.94)),
                 z_vals=(0.1, 1.0),
                 b=1e-2,
                 grid_max_A=50,
                 grid_max_H=4,
                 grid_size=200,
                 grid_size_H=3,
                 gamma_1=0.2,
                 phi=0.06,
                 xi=0.1,
                 kappa=0.075,
                 theta=0.77,
                 iota=.001):

        self.grid_size = grid_size
        self.grid_size_H = grid_size_H
        self.r, self.R = r, 1 + r
        self.r_H, self.R_H = r_H, 1 + r_H
        self.beta = beta
        self.delta = delta
        self.b = b
        self.phi = phi
        self.grid_max_A, self.grid_max_H = grid_max_A, grid_max_H

        self.gamma_1, self.xi = gamma_1, xi

        self.Pi, self.z_vals = np.array(Pi), np.asarray(z_vals)

        self.asset_grid_A = np.linspace(b, grid_max_A, grid_size)
        self.asset_grid_H = np.linspace(b, grid_max_H, grid_size_H)
        self.asset_grid_M = np.linspace(b, grid_max_A + grid_max_H, grid_size*2)

        # time t state-space
        self.X_all = cartesian([np.arange(len(z_vals)),
                                np.arange(len(self.asset_grid_A)),
                                np.arange(len(self.asset_grid_H))])

        # time state-space plus t+1 housing
        self.X_all_big = cartesian([np.arange(len(z_vals)),
                                    np.arange(len(self.asset_grid_A)),
                                    np.arange(len(self.asset_grid_H)),
                                    np.arange(len(self.asset_grid_H))])

        # time t discrete state, t+1 discrete state and exog state
        self.X_exog = cartesian([np.arange(len(z_vals)),
                                 np.arange(len(self.asset_grid_H))])

        self.iota, self.kappa, self.theta = iota, kappa, theta

        # define functions
        @njit
        def u(x, h):
            if x <= 0:
                return - np.inf
            else:
                return theta * np.log(x) + (1 - theta) * \
                    np.log(kappa * (h + iota))
            
        @njit 
        def u_vec(x, h):
            return theta * np.log(x) + (1 - theta) * np.log(kappa * (h + iota))

        @njit
        def term_du(x):
            return theta / x

        @njit
        def du_inv(uc):
            return theta / uc

        self.u = u
        self.uc_inv = du_inv
        self.du = term_du
        self.u_vec = u_vec

def euler_error_fella(cp, z_series, H_post_state, c_state, a_state, c_post_state):
    """
    Calculate the Euler error for the Fella housing model with a stationary policy function.
    
    Parameters:
    cp : ConsumerProblem
        The consumer problem with model parameters.
    z_series : np.array
        The series of exogenous shock states (z) over time.
    H_post_state : np.array
        The policy function for housing choices (h_prime) over the post-state space (z, a, h).
    c_state : np.array
        The stationary policy function for consumption (c) over the state space (z, m, h).
    a_state : np.array
        The stationary policy function for liquid assets (a_prime) over the state space (z, m, h).
    c_on_post_state : np.array
        The future stationary policy function for consumption (c) over the post-state space (z, a, h).
    
    Returns:
    float
        The average log10 Euler error across exogenous states, asset grid points, and housing grid points.
    """

    a_grid = cp.asset_grid_M  # Liquid assets grid
    h_grid = cp.asset_grid_H  # Housing grid
    z_vals = cp.z_vals        # Exogenous shock state

    # Initialize the Euler error array
    euler = np.full(len(z_series), np.nan)
    
    a_t = 0.1
    i_h_t = 1
    h_grid_index = np.arange(len(h_grid))

    # Loop over exogenous states (z), asset grid (a), and housing grid (h)
    for t in range(len(z_series)):
        i_z = z_series[t]
        z = z_vals[i_z]

        i_h_prime = np.interp(a_t, a_grid, H_post_state[i_z, :, i_h_t])
        # Find nearest housing grid point
        i_h_prime = np.argmin(np.abs(h_grid_index - i_h_prime))
        h = h_grid[i_h_t]

        chi = 0
        if i_h_t != i_h_prime:
            chi = 1

        wealth = (cp.R * a_t + z + chi * (h - h_grid[i_h_prime]) 
                  - chi * np.abs(h_grid[i_h_prime]) * cp.phi)

        a_prime = np.interp(wealth, a_grid, a_state[i_z, :, i_h_prime])
        c = np.interp(wealth, a_grid, c_state[i_z, :, i_h_t])
        
        if a_prime < cp.b or wealth < cp.b:
            continue

        RHS = 0
        for i_z_plus in range(len(z_vals)):
            c_plus = np.interp(a_prime, a_grid, c_post_state[i_z_plus, :, i_h_prime])
            RHS += cp.Pi[i_z, i_z_plus] * cp.du(c_plus)

        # Compute the raw Euler error
        euler_raw = c - cp.uc_inv(cp.R * RHS * cp.beta)
        euler[t] = np.log10(np.abs(euler_raw / c) + 1e-16)

        # Update states
        a_t = a_prime
        i_h_t = i_h_prime

    # Return the average Euler error across all states
    return np.nanmean(euler), euler


def EGM_UE(egrid, vf, c, a, dela, endog_mbar=False, method='FUES', m_bar=1.2):
    """
    Wrapper function to select between FUES, RFC and DC-EGM upper envelope methods
    """

    if method == 'FUES':
        # FUES method
        policies_dict = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64[:],
        )

        policies_dict['a'] = np.array(a)
        policies_dict['c'] = np.array(c)
        policies_dict['vf'] = np.array(vf)
        test_pols = np.array(a)

        egrid_refined_1D, vf_refined_1D, c_refined_1D, a_prime_refined_1D, dela = \
            FUES(
                egrid, vf, policies_dict['c'], policies_dict['a'],
                policies_dict['a'], m_bar=m_bar, LB=2, endog_mbar=False
            )

    elif method == 'DCEGM':
        # DCEGM method
        a_prime_refined_1D, egrid_refined_1D, c_refined_1D, vf_refined_1D, \
            dela_clean = dcegm(c, c, vf, a, egrid)

    elif method == 'RFC':
        # RFC method
        grad = cp.du(c)
        xr = np.array([egrid]).T
        vfr = np.array([vf]).T
        gradr = np.array([grad]).T
        pr = np.array([a]).T
        mbar = 1.2
        radius = 0.75

        # Run RFC vectorized
        sub_points, roofRfc, close_ponts = rfc(xr, gradr, vfr, pr, mbar, radius, 20)

        mask = np.ones(egrid.shape[0], dtype=bool)
        mask[sub_points] = False
        egrid_refined_1D = egrid[mask]
        vf_refined_1D = vfr[mask][:, 0]
        c_refined_1D = c[mask]
        a_prime_refined_1D = a[mask]

    return egrid_refined_1D, vf_refined_1D, c_refined_1D, a_prime_refined_1D, vf


def Operator_Factory(cp):
    """ Operator that generates functions to solve model"""

    # unpack all variables
    beta, delta = cp.beta, cp.delta
    gamma_1 = cp.gamma_1
    xi = cp.xi
    asset_grid_A, asset_grid_H, z_vals, Pi = cp.asset_grid_A, cp.asset_grid_H,\
        cp.z_vals, cp.Pi
    asset_grid_M = cp.asset_grid_M
    grid_max_A, grid_max_H = cp.grid_max_A, cp.grid_max_H
    u = cp.u
    u_vec = cp.u_vec
    uc_inv = cp.uc_inv
    uc = cp.du
    phi = cp.phi
    r = cp.r

    R, R_H = cp.R, cp.R_H
    X_all = cp.X_all
    b = cp.b
    X_all_big = cp.X_all_big
    X_exog = cp.X_exog
    z_idx = np.arange(len(z_vals))

    shape = (len(z_vals), len(asset_grid_A), len(asset_grid_H))
    
    shape_big = (
        len(z_vals),
        len(asset_grid_A),
        len(asset_grid_H),
        len(asset_grid_H))

    @njit
    def obj(a_prime,\
                 a,
                 h,
                 i_h,
                 h_prime,
                 i_h_prime,
                 z,
                 i_z,
                 V,
                 R,
                 R_H,
                 t):

        """Objective function to be *maximised* by Bellman operator

        Parameters
        ----------
        a_prime: float 
                    next period liquid assets 
        a: float 
            current period liquid asset 
        h: float
            current period housing
        i_h: int
              current period housing index
        h_prime: float
                    next period housing level
        i_h_prime: int
                    next period housing index
        z: float
            exog shock
        i_z: int
                shock index 
        V: 3D array
            Value function
        R: float 
            interest rate + 1
        R_H: float 
        t: int 

        Returns
        -------
        u: float
            utility

        """
        if i_h != i_h_prime:
            chi = 1
        else:
            chi = 0

        # wealth is cash at hand after cost of adjusting house paid
        wealth = R * a + z - (h_prime - h)* chi - phi * np.abs(h_prime) * chi
        consumption = wealth - a_prime

        # get the t+1 value function
        Ev_prime = interp_as(
            asset_grid_A, V[i_z, :, i_h_prime], np.array([a_prime]))[0]

        # evaluate time t value conditional on consumption
        if a_prime >= b:
            return u(consumption, h_prime) + beta * Ev_prime
        else:
            return - np.inf

    @njit
    def bellman_operator(t, V):
        """
        The approximate Bellman operator, which computes and returns the
        updated value function TV (or the V-greedy policy c if
        return_policy is True)

        Parameters
        ----------
        V : 3D array 
                Value function at t interation 
        t: int 

        Returns
        -------
        new_a_prime: 3D array 
                        t-1 asset policy
        new_h_prime: 3D array
                        t-1 housing policy
        new_V: 3D array
                    t-1 value function 
        new_z_prime: 3D array
                    total cash at hand after adjustment 
        new_V_adj_big: 4D array
                        Value function conditioned on H_prime choice

        new_a_big: 4D array 
                    Asset policy conditioned on H_prime choice 

        new_c_prime: 3D array 
                        Consumption policy 

        """

        # Solve r.h.s. of Bellman equation
        # First generate the empty grids for next
        # iteration value and policy

        new_V = np.empty(V.shape)
        new_h_prime = np.empty(V.shape)
        new_a_prime = np.empty(V.shape)
        new_V_adj = np.empty(V.shape)
        new_V_noadj = np.empty(V.shape)
        new_z_prime = np.empty(V.shape)
        new_V_adj_big = np.empty(shape_big)
        new_a_big = np.empty(shape_big)
        new_c_prime = np.empty(shape)

        # loop over the time t state space
        for state in prange(len(X_all)):
            a = asset_grid_A[X_all[state][1]]
            h = asset_grid_H[X_all[state][2]]
            i_a = int(X_all[state][1])
            i_h = int(X_all[state][2])
            i_z = int(X_all[state][0])
            z = z_vals[i_z]

            v_vals_hprime = np.zeros(len(asset_grid_H))
            ap_vals_hprime = np.zeros(len(asset_grid_H))
            z_vals_prime = np.zeros(len(asset_grid_H))
            cvals_prime = np.zeros(len(asset_grid_H))

            # loop over t+1 housing discrete choices 
            for i_h_prime in range(len(asset_grid_H)):
                h_prime = asset_grid_H[i_h_prime]
                lower_bound = asset_grid_A[0]

                if i_h != i_h_prime:
                    chi = 1
                else:
                    chi = 0

                upper_bound = max(
                    asset_grid_A[0], R * a + z - (h_prime - h)*chi - phi * np.abs(h_prime) * chi) + b

                args_adj = (
                    a,
                    h,
                    i_h,
                    h_prime,
                    i_h_prime,
                    z,
                    i_z,
                    V,
                    R,
                    R_H,
                    t)

                xf, xvf, flag = brent_max(
                    obj, lower_bound, upper_bound, args=args_adj, xtol=1e-12)
                v_vals_hprime[i_h_prime] = xvf
                new_V_adj_big[i_z, i_a, i_h, i_h_prime] = xvf

                ap_vals_hprime[i_h_prime] = xf

                z_vals_prime[i_h_prime] = upper_bound

                # wealth is cash at hand after housing adjustment paid for 
                wealth = R * a + z - (h_prime - h)* chi - phi * np.abs(h_prime) * chi
                new_a_big[i_z, i_a, i_h, i_h_prime] = xf
                cvals_prime[i_h_prime] = wealth - xf

            # make the time t discrete choice aout h(t+1)
            h_prime_index = int(np.argmax(v_vals_hprime))

            new_h_prime[i_z, i_a, i_h] = h_prime_index
            new_a_prime[i_z, i_a, i_h] = ap_vals_hprime[h_prime_index]
            new_V[i_z, i_a, i_h] = v_vals_hprime[h_prime_index]
            new_z_prime[i_z, i_a, i_h] = z_vals_prime[h_prime_index]
            new_c_prime[i_z, i_a, i_h] = cvals_prime[h_prime_index]

        return new_a_prime, new_h_prime, new_V, new_z_prime, new_V_adj_big, new_a_big, new_c_prime

    def condition_V(new_V_uc, new_Ud_a_uc, new_Ud_h_uc):
        """ Condition the t+1 value or marginal value
        time t information"""

        new_V = np.zeros(np.shape(new_V_uc))
        new_UD_a = np.zeros(np.shape(new_Ud_a_uc))
        new_UD_h = np.zeros(np.shape(new_Ud_h_uc))

        # numpy dot sum product over last axis of matrix_A (t+1 continuation value unconditioned)
        # see nunpy dot docs
        for state in range(len(X_all)):
            i_a = int(X_all[state][1])
            i_h = int(X_all[state][2])
            i_z = int(X_all[state][0])

            new_V[i_z, i_a, i_h] = np.dot(Pi[i_z, :], new_V_uc[:, i_a, i_h])
            new_UD_a[i_z, i_a, i_h] = np.dot(
                Pi[i_z, :], new_Ud_a_uc[:, i_a, i_h])
            new_UD_h[i_z, i_a, i_h] = np.dot(
                Pi[i_z, :], new_Ud_h_uc[:, i_a, i_h])

        return new_V, new_UD_a, new_UD_h
        
    @njit 
    def invertEuler(V, sigma, dela):
        """
        Invert the Euler equation to find the consumption policy function
        """

        c_raw = np.zeros(shape)
        v_raw = np.zeros(shape)
        e_grid_raw = np.zeros(shape)

        for state in range(len(X_all)):
            i_z = int(X_all[state][0])
            i_a_prime = int(X_all[state][1])
            i_h_prime = int(X_all[state][2])
            h_prime = asset_grid_H[X_all[state][2]]
            a_prime = asset_grid_A[X_all[state][1]]

            UC_primes = beta * R * uc(sigma[:, i_a_prime, i_h_prime])
            VF_primes = beta * V[:, i_a_prime, i_h_prime]

            
            c_t = uc_inv(np.dot(UC_primes, Pi[i_z, :]))

            vf_prime = np.dot(Pi[i_z, :], VF_primes)
            v_curr =  u(c_t, h_prime) + vf_prime 
            market_resources = a_prime + c_t
            #print(market_resources)
            e_grid_raw[i_z, i_a_prime, i_h_prime] = market_resources
            c_raw[i_z, i_a_prime, i_h_prime] = c_t
            v_raw[i_z, i_a_prime, i_h_prime] = v_curr
        #print(e_grid_raw)
        return c_raw, v_raw, e_grid_raw
    
    @njit 
    def H_choice(new_v_refined, a_prime_refined, c_refined):
        """ 
        Interpolate the value function and policy function on the 
        start of period state space and choose the optimal housing choice. 

        Parameters
        ----------
        new_v_refined: 3D array
                        Value function on active state space and conditioned on housing choice
        a_prime_refined: 3D array
                        Asset policy on active state space and conditioned on housing choice
        c_refined: 3D array
                    Consumption policy on active state space and conditioned on housing choice
        
        Returns
        -------
        new_v_refined: 3D array
                        Value function on state space
        a_new: 3D array
                Asset policy on state space
        H_new: 3D array
                Housing policy on state space
        
        """

        a_new = np.zeros(shape)
        H_new = np.zeros(shape)
        c_new = np.zeros(shape)

        V_new_big = np.zeros(shape_big)
        sigma_new_big = np.zeros(shape_big)
        a_new_big = np.zeros(shape_big)

        for i in range(len(X_all_big)):
            i_z = int(X_all_big[i][0])
            i_a = int(X_all_big[i][1])
            i_h = int(X_all_big[i][2])
            i_h_prime = int(X_all_big[i][3])
            a = asset_grid_A[i_a]

            h = asset_grid_H[i_h]
            h_prime = asset_grid_H[i_h_prime]

            chi = 0

            if i_h != i_h_prime:
                chi = 1

            wealth_curr = a* R + z_vals[i_z] + chi*(h - h_prime) - phi * np.abs(h_prime)*chi

            if wealth_curr < 0:
                V_new_big[i_z, i_a, i_h, i_h_prime] = -np.inf
            
            else:
                V_new_big[i_z, i_a, i_h, i_h_prime] = np.interp(wealth_curr, asset_grid_M, new_v_refined[i_z, :, i_h_prime])
                sigma_new_big[i_z, i_a, i_h, i_h_prime] = np.interp(wealth_curr, asset_grid_M, c_refined[i_z, :, i_h_prime])
                a_new_big[i_z, i_a, i_h, i_h_prime] = np.interp(wealth_curr, asset_grid_M, a_prime_refined[i_z, :, i_h_prime])
        
        for i in range(len(X_all)):

            i_z = int(X_all[i][0])
            i_a = int(X_all[i][1])
            i_h = int(X_all[i][2])

            # pick out max element
            max_index = int(np.argmax(V_new_big[i_z, i_a, i_h, :]))
    
            new_v_refined[i_z, i_a, i_h] = V_new_big[i_z, i_a, i_h,max_index]
            H_new[i_z, i_a, i_h] = max_index

            a_new[i_z, i_a,i_h] = a_new_big[i_z, i_a, i_h, max_index]
            c_new[i_z,i_a, i_h] = sigma_new_big[i_z, i_a, i_h, max_index]
        
        return new_v_refined, c_new, a_new, H_new
            
    #@njit
    def Euler_Operator(V, sigma, dela, method='FUES'):
        """
        Euler operator finds next period policy function using EGM and FUES
        """

        # The value function should be conditioned on time t
        # continuous state, time t discrete state, and time t+1 discrete state choice

        c_raw, v_raw, e_grid_raw = invertEuler(V, sigma, dela)

        new_a_prime_refined1 = np.zeros(shape)
        new_c_refined1 = np.zeros(shape)
        new_v_refined1 = np.zeros(shape)

        for i in range(len(X_exog)):

            h_prime = asset_grid_H[X_exog[i][1]]  # t+1 housing
            i_h_prime = int(X_exog[i][1])
            i_z = int(X_exog[i][0])

            egrid_unrefined_1D = e_grid_raw[i_z, :, i_h_prime]
            a_prime_unrefined_1D = np.copy(asset_grid_A)
            c_unrefined_1D = c_raw[i_z, :, i_h_prime]
            vf_unrefined_1D = v_raw[i_z, :, i_h_prime]

            start = time.time()
            egrid_refined_1D, vf_refined_1D, c_refined_1D, a_prime_refined_1D, dela_out = \
                EGM_UE(egrid_unrefined_1D, vf_unrefined_1D, c_unrefined_1D,
                    a_prime_unrefined_1D, vf_unrefined_1D, method=method,
                    m_bar=0.8)
            UE_time = time.time() - start

            min_c_val = np.min(c_unrefined_1D)
            c_array = np.linspace(0.00001, min_c_val, 100)
            e_array = c_array
            h_prime_array = np.zeros(100)
            h_prime_array.fill(h_prime)
            vf_array = u_vec(c_array, h_prime_array) + beta * np.dot(
                Pi[i_z, :], V[:, 0, i_h_prime])
            b_array = np.zeros(100)
            b_array.fill(asset_grid_A[0])

            egrid_refined_1D = np.concatenate((e_array, egrid_refined_1D))
            vf_refined_1D = np.concatenate((vf_array, vf_refined_1D))
            c_refined_1D = np.concatenate((c_array, c_refined_1D))
            a_prime_refined_1D = np.concatenate((b_array, a_prime_refined_1D))

            new_a_prime_refined1[i_z, :, i_h_prime] = interp_as(
                egrid_refined_1D, a_prime_refined_1D, asset_grid_M, extrap=False)
            new_c_refined1[i_z, :, i_h_prime] = interp_as(
                egrid_refined_1D, c_refined_1D, asset_grid_M, extrap=False)
            new_v_refined1[i_z, :, i_h_prime] = interp_as(
                egrid_refined_1D, vf_refined_1D, asset_grid_M, extrap=False)

        new_v_refined, new_c_refined, new_a_prime_refined, new_H_refined = \
            H_choice(new_v_refined1, new_a_prime_refined1, new_c_refined1)

        results = {'post_state': {}, 'state': {}}

        results['post_state']['c'] = new_c_refined
        results['post_state']['a'] = new_a_prime_refined
        results['post_state']['vf'] = new_v_refined
        results['post_state']['H_prime'] = new_H_refined

        results['state']['c'] = new_c_refined1
        results['state']['a'] = new_a_prime_refined1
        results['state']['vf'] = new_v_refined1

        return results, UE_time

    return bellman_operator, Euler_Operator, condition_V


def iterate_euler(cp, method="FUES", max_iter=200, tol=1e-4, verbose = True):
    """
    Function to perform Euler iteration using the given method (FUES, RFC, or DCEGM).

    Parameters:
    cp : ConsumerProblem instance
        The consumer problem with model parameters.
    method : str, optional
        Method to use for Euler iteration ('FUES', 'RFC', or 'DCEGM').
    max_iter : int, optional
        Maximum number of iterations for convergence (default=200).
    tol : float, optional
        Convergence tolerance (default=1e-4).
    
    Returns:
    dict
        Dictionary containing the final value function, consumption policy, 
        asset policy, and time taken.
    """

    # Unpack necessary functions
    _, Euler_Operator, _ = Operator_Factory(cp)

    # Initial values for value function, consumption, and assets
    shape = (len(cp.z_vals), len(cp.asset_grid_A), len(cp.asset_grid_H))
    V_init = np.ones(shape)
    c_init = np.ones(shape) * (cp.asset_grid_A[:, None] / 3)  # Initial consumption policy
    dela = np.ones((len(cp.z_vals), len(cp.asset_grid_H), len(cp.asset_grid_H)))

    # Initialize error and iteration counter
    bhask_error = np.inf
    k = 0

    # Copies of initial conditions
    V_new = np.copy(V_init)
    c_new = np.copy(c_init)
    
    start_time = time.time()  # Track time
    UE_time = 0
    # Euler iteration loop
    while k < max_iter and bhask_error > tol:
        # Perform one step of Euler operator based on the selected method
        results, UE_time1 = Euler_Operator(V_new, c_new, dela, method=method)
        
        # Update error and policies
        bhask_error = np.max(np.abs(results['post_state']['vf'] - V_new))  # Calculate error based on policy function changes
        V_new = np.copy(results['post_state']['vf'])  # Update value function
        c_new = np.copy(results['post_state']['c'])  # Update consumption policy
        
        # average time taken for UE
        UE_time = (UE_time + UE_time1)/2

        k += 1  # Increment iteration count
        results['UE_time'] = UE_time
        
        if verbose == True:
            print(f'{method} Iteration {k}, Error: {bhask_error:.6f}')

    end_time = time.time()

    return results


def compare_methods_grid(cp1, grid_sizes_A, grid_sizes_H, max_iter=100, tol=1e-03):
    """
    Compare the performance of FUES, DCEGM, and RFC over different grid sizes.
    """
    results_summary = []

    # simulate markoc chain for model and fix 

    mc = MarkovChain(cp1.Pi)
    z_series = mc.simulate(ts_length=100000,  init = 1)


    for grid_size_A in grid_sizes_A:
        for grid_size_H in grid_sizes_H:
            # Update the grid size in the model parameters (assuming cp has the grid size properties)
            ##cp.asset_grid_A = np.linspace(0, 1, grid_size_A)  # Adjust asset grid as needed
            cp = ConsumerProblem(
                    r=0.06,
                    r_H=0,
                    beta=0.93,
                    delta=0,
                    Pi=((0.99, 0.01, 0), (0.01, 0.98, 0.01), (0, 0.09, 0.91)),
                    z_vals=(0.1, 0.526, 4.66),
                    b=1e-10,
                    grid_max_A=5,
                    grid_max_H=grid_size_H,
                    grid_size=grid_size_A,
                    grid_size_H=7,
                    gamma_1=0,
                    xi=0,
                    kappa=0.07,
                    phi=0.07,
                    theta=0.77
                )

            for method in ['FUES', 'DCEGM', 'RFC']:
                best_time = np.inf
                best_euler_error = np.inf
                total_runtime = 0

                for _ in range(1):  # Run each method 4 times
                    start_time = time.time()
                    results = iterate_euler(cp, method=method, max_iter=max_iter, tol=tol, verbose = False)
                    runtime = time.time() - start_time

                    total_runtime += runtime
                    if results['UE_time'] < best_time:
                        best_time = results['UE_time']
                        best_results = results

                    # Calculate Euler error for each iteration
                    E_error, _ = euler_error_fella(cp,z_series, 
                                                        results['post_state']['H_prime'],
                                                        results['state']['c'], 
                                                        results['state']['a'], 
                                                        results['post_state']['c'])
                    if E_error < best_euler_error:
                        best_euler_error = E_error

                # Record the average total time and best UE_time
                avg_total_runtime = total_runtime / 4
                results_summary.append({
                    'Grid_Size_A': grid_size_A,
                    'Grid_Size_H': grid_size_H,
                    'Method': method,
                    'Best_UE_time': best_time,
                    'Avg_Total_Runtime': avg_total_runtime,
                    'Best_Euler_Error': best_euler_error
                })

    # save results summary as pickle file
    with open('../results/fella_timings.pkl', 'wb') as file:
        pickle.dump(results_summary, file)

    # Create a LaTeX table
    latex_table = create_latex_table(results_summary)

    file_path = '../results/fella_timings.tex'

    with open(file_path, 'w') as file:
        file.write(latex_table)

    return results_summary, latex_table

def create_latex_table(results_summary):
    """
    Generates a LaTeX table that includes timing (in milliseconds) and 
    Euler error comparisons for RFC, FUES, and DCEGM methods across 
    different grid sizes (A and H), all on the same row.
    """
    table = "\\begin{table}[htbp]\n\\centering\n\\small\n"
    table += (
        "\\begin{tabular}{ccccc|ccc}\n\\toprule\n"
        "\\multirow{2}{*}{\\textit{Grid Size A}} & "
        "\\multirow{2}{*}{\\textit{Grid Size H}} & "
        "\\multicolumn{3}{c}{\\textbf{Timing (milliseconds)}} & "
        "\\multicolumn{3}{c}{\\textbf{Euler error (Log10)}} \\\\\n"
        " & & \\textbf{RFC} & \\textbf{FUES} & \\textbf{DCEGM} & "
        "\\textbf{RFC} & \\textbf{FUES} & \\textbf{DCEGM} \\\\\n"
        "\\midrule\n"
    )

    current_grid_size_A = None

    # Sort the results by Grid_Size_A and Grid_Size_H for proper ordering
    results_summary = sorted(results_summary, key=lambda x: (x['Grid_Size_A'], x['Grid_Size_H']))

    # Initialize a dictionary to store results for each grid size
    grid_results = {}

    # Iterate through the results and organize data by Grid_Size_A and Grid_Size_H
    for result in results_summary:
        grid_size_A = result['Grid_Size_A']
        grid_size_H = result['Grid_Size_H']
        method = result['Method']

        # Convert time to milliseconds
        time_ms = result['Best_UE_time'] * 1000
        # Euler error formatted without scientific notation
        euler_error = f"{result['Best_Euler_Error']:.6f}"

        # Initialize dictionary entries for grid_size_A and grid_size_H
        if grid_size_A not in grid_results:
            grid_results[grid_size_A] = {}
        if grid_size_H not in grid_results[grid_size_A]:
            grid_results[grid_size_A][grid_size_H] = {
                'RFC': {'time': '-', 'error': '-'},
                'FUES': {'time': '-', 'error': '-'},
                'DCEGM': {'time': '-', 'error': '-'}
            }

        # Store the result in the appropriate method slot
        grid_results[grid_size_A][grid_size_H][method] = {
            'time': f"{time_ms:.3f}",
            'error': euler_error
        }

    # Now iterate through grid_results and construct the table rows
    for grid_size_A, grid_size_H_data in grid_results.items():
        first_row = True  # Track if this is the first row for this grid size A
        for grid_size_H, methods_data in grid_size_H_data.items():
            # Add \midrule between Grid_Size_A groups (except for the first row)
            if not first_row:
                table += "\\midrule\n"
            first_row = False

            # Output the row with Grid_Size_A, Grid_Size_H, and the results of RFC, FUES, and DCEGM
            table += (
                f"\\multirow{{1}}{{*}}{{\\textit{{{grid_size_A}}}}} & {grid_size_H} & "
                f"{methods_data['RFC']['time']} & {methods_data['FUES']['time']} & {methods_data['DCEGM']['time']} & "
                f"{methods_data['RFC']['error']} & {methods_data['FUES']['error']} & {methods_data['DCEGM']['error']} \\\\\n"
            )

    # Finish the table
    table += "\\bottomrule\n\\end{tabular}\n"
    table += (
        "\\caption{\\small Speed and accuracy of FUES, DCEGM, and RFC "
        "across different grid sizes A and H.}\n\\end{table}"
    )

    return table

import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pl
from matplotlib.ticker import FormatStrFormatter

# Main block of code
if __name__ == "__main__":\


    # open pickle file with table
    #with open('../results/fella_timings.pkl', 'rb') as file:
    #    results_summary = pickle.load(file)
    
    # Create a LaTeX table
    #table = latex_table = create_latex_table(results_summary)
    
    #print(table)

    #file_path = '.,/results/fella_timings.tex'

    #with open(file_path, 'w') as file:
    #    file.write(latex_table)


    # Instantiate the consumer problem with parameters
    cp = ConsumerProblem(
        r=0.06,
        r_H=0,
        beta=0.93,
        delta=0,
        Pi=((0.99, 0.01, 0), (0.01, 0.98, 0.01), (0, 0.09, 0.91)),
        z_vals=(0.1, 0.526, 4.66),
        b=1e-10,
        grid_max_A=20,
        grid_max_H=5,
        grid_size=500,
        grid_size_H=10,
        gamma_1=0,
        xi=0,
        kappa=0.07,
        phi=0.11,
        theta=0.77
    )

    # Timing comparisons and table
    #grid_sizes_A = [500, 1000,2000]
    #grid_sizes_H = [3,4,5,7]
    
    #compare_methods_grid(cp, grid_sizes_A, grid_sizes_H,
    #                     max_iter=100, tol=1e-03)

    # Bellman operator and plotting

    mc = MarkovChain(cp.Pi)
    z_series = mc.simulate(ts_length=10000,  init = 1)


    bellman_operator, euler_operator, condition_V = Operator_Factory(cp)

    # Initialize empty grids
    shape = (len(cp.z_vals), len(cp.asset_grid_A), len(cp.asset_grid_H))
    V_init = np.ones(shape)
    a_policy = np.empty(shape)
    h_policy = np.empty(shape)
    value_func = np.empty(shape)

    bell_error = 1
    bell_toll = 1e-3
    iteration = 0
    new_V = V_init
    max_iter = 200
    pl.close()

    sns.set(style="whitegrid",
            rc={"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10})

    # Solve via VFI and plot
    start_time = time.time()
    while bell_error > bell_toll and iteration < max_iter:
        V = np.copy(new_V)
        a_new_policy, h_new_policy, V_new_policy, _, _, _, _ = \
            bellman_operator(iteration, V)

        new_V, _, _ = condition_V(V_new_policy, V_new_policy, V_new_policy)
        a_policy, h_policy, value_func = np.copy(a_new_policy), \
                                         np.copy(h_new_policy), \
                                         np.copy(new_V)

        bell_error = np.max(np.abs(V - value_func))
        print(f"Iteration {iteration + 1}, error is {bell_error}")
        iteration += 1

    print(f"VFI in {time.time() - start_time} seconds")

    # Euler error and additional policy function plotting
    results_FUES = iterate_euler(cp, method='FUES', max_iter=max_iter, 
                                 tol=1e-03)
    print(f"FUES in {results_FUES['UE_time']} seconds")

    vf_FUES, c_FUES, a_FUES, h_FUES = results_FUES['post_state']['vf'], \
                                      results_FUES['post_state']['c'], \
                                      results_FUES['post_state']['a'], \
                                      results_FUES['post_state']['H_prime']

    euler_error_FUES, _ = euler_error_fella(cp, z_series,
                                                results_FUES['post_state']['H_prime'],
                                                 results_FUES['state']['c'], 
                                                 results_FUES['state']['a'], 
                                                 results_FUES['post_state']['c'])
    print(euler_error_FUES)

    results_DCEGM = iterate_euler(cp, method='DCEGM', max_iter=max_iter, 
                                  tol=1e-03)
    print(f"DCEGM in {results_DCEGM['UE_time']} seconds")

    vf_DCEGM, c_DCEGM, a_DCEGM, h_DCEGM = results_DCEGM['post_state']['vf'], \
                                          results_DCEGM['post_state']['c'], \
                                          results_DCEGM['post_state']['a'], \
                                          results_DCEGM['post_state']['H_prime']

    euler_error_DCEGM, _ = euler_error_fella(cp, z_series,
                                                results_DCEGM['post_state']['H_prime'],
                                                  results_DCEGM['state']['c'], 
                                                  results_FUES['state']['a'], 
                                                  results_DCEGM['post_state']['c'])
    print(euler_error_DCEGM)

    # Plotting
    fig, ax = pl.subplots(1, 2)
    ax[0].set_xlabel('Assets (t)', fontsize=11)
    ax[0].set_ylabel('Assets (t+1)', fontsize=11)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_yticklabels(ax[0].get_yticks(), size=9)
    ax[0].set_xticklabels(ax[0].get_xticks(), size=9)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    ax[1].set_xlabel('Assets (t)', fontsize=11)
    ax[1].set_ylabel('Assets (t+1)', fontsize=11)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_yticklabels(ax[0].get_yticks(), size=9)
    ax[1].set_xticklabels(ax[0].get_xticks(), size=9)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    for i, col, lab in zip([1, 4, 7], ['blue', 'red', 'black'], 
                           ['H = low', 'H = med.', 'H = high']):
        ax[1].plot(cp.asset_grid_A, results_FUES['state']['c'][1, :, i], 
                   color=col, label=lab)
        ax[1].plot(cp.asset_grid_A, results_DCEGM['state']['c'][1, :, i], 
                   color='black', linestyle='--')

    ax[0].legend(frameon=False, prop={'size': 10})
    ax[0].set_title("VFI", fontsize=11)
    ax[1].set_title("FUES-EGM", fontsize=11)

    fig.tight_layout()
    pl.savefig('../results/plots/fella/Fella_policy.png')

