"""

Module that contains class and solvers for discrete
choice housing model (Fella, 2014) using FUES, RFC, DC-EGM. 

Author: Akshay Shanker, University of Sydney, a.shanker@unsw.edu.au

"""
import numpy as np
import time
import dill as pickle
from sklearn.utils.extmath import cartesian
from numba import njit, prange
import matplotlib.pylab as pl
from quantecon.optimize.scalar_maximization import brent_max
from quantecon.markov.approximation import tauchen
from numba import njit
from numba.typed import Dict
from numba.core import types
from quantecon import MarkovChain
import os
import sys
from itertools import groupby


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
import fues.ue_kernels
from fues.fues import FUES as fues_alg, uniqueEG
from fues.rfc_simple import rfc
from fues.dcegm import dcegm
from fues.helpers.math_funcs import interp_as, correct_jumps1d

# Commenting out missing module
# from fues.bonnFues import fues_numba_unconstrained 

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
                 m_bar=1.2,
                 lb = 3,
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
        self.n_con = grid_size
        self.lb = lb

        self.gamma_1, self.xi = gamma_1, xi

        self.Pi, self.z_vals = np.array(Pi), np.asarray(z_vals)

        self.asset_grid_A = np.linspace(b, grid_max_A, grid_size*2)
        self.asset_grid_H = np.linspace(b, grid_max_H, grid_size_H)
        self.asset_grid_M = np.linspace(b, grid_max_A + grid_max_H, grid_size)

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
        self.m_bar = m_bar

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

def welfare_loss_log_utility(v_fues, v_dcegm, c_dcegm):
    """
    Calculate welfare loss in consumption terms using log utility.

    Parameters:
    v_fues: np.array
        Value function estimates from fues method.
    v_dcegm: np.array
        Value function estimates from DC-EGM method.
    c_dcegm: np.array
        Consumption level under DC-EGM method.

    Returns:
    welfare_loss: float
        Equivalent variation welfare loss in consumption terms.
    """

    # Calculate the equivalent consumption that gives the same welfare as FUES
    c_equivalent = c_dcegm * (v_fues / v_dcegm)

    # Compute the welfare loss in consumption terms
    welfare_loss = np.mean(np.abs(c_equivalent - c_dcegm))

    return welfare_loss

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
    n_con = cp.n_con

    shape = (len(z_vals), len(asset_grid_A), len(asset_grid_H))
    shape_active = (len(z_vals), len(asset_grid_M), len(asset_grid_H))
    
    shape_big = (
        len(z_vals),
        len(asset_grid_A),
        len(asset_grid_H),
        len(asset_grid_H))
    

    def EGM_UE(egrid, vf, c, a, dela, endog_mbar=False, method='FUES', m_bar=1.2, lb = 4):
        """
        Wrapper function to select between FUES, RFC and DC-EGM upper envelope methods
        """


        #uniqueIds = uniqueEG(egrid, vf)
        #egrid = egrid[uniqueIds]
        ##vf = vf[uniqueIds]
        ##c = c[uniqueIds]
        #a = a[uniqueIds]

        if method == 'FUES':
            # FUES method original 
            policies_dict = Dict.empty(
                key_type=types.unicode_type,
                value_type=types.float64[:],
            )

            policies_dict['a'] = np.array(a)
            policies_dict['c'] = np.array(c)
            policies_dict['vf'] = np.array(vf)
            test_pols = np.array(a)

            egrid_refined_1D, vf_refined_1D, a_prime_refined_1D, c_refined_1D, dela = \
                FUES(
                    egrid, vf, policies_dict['a'], policies_dict['c'],
                    policies_dict['a'], m_bar=m_bar, LB=lb, endog_mbar=False
                )
            
        if method == 'FUES_numba':
            # FUES coded by OSE - Module not available, using standard FUES instead
            policies_dict = Dict.empty(
                key_type=types.unicode_type,
                value_type=types.float64[:],
            )

            policies_dict['a'] = np.array(a)
            policies_dict['c'] = np.array(c)
            policies_dict['vf'] = np.array(vf)
            test_pols = np.array(a)

            egrid_refined_1D, vf_refined_1D, a_prime_refined_1D, c_refined_1D, dela = \
                FUES(
                    egrid, vf, policies_dict['a'], policies_dict['c'],
                    policies_dict['a'], m_bar=m_bar, LB=lb, endog_mbar=False
                )

        elif method == 'DCEGM':
            # DCEGM method
            # -------------------- DCEGM DIAGNOSTIC BLOCK --------------------
            debug_dcegm = True
            if debug_dcegm:
                import uuid, sys
                import numpy as np
                _dcegm_inputs = {
                    'c': c, 'grad': c, 'vf': vf, 'a': a, 'egrid': egrid
                }
                fname = f"inputs_from_fella_{uuid.uuid4().hex[:6]}.npz"
                try:
                    np.savez(fname, **_dcegm_inputs)
                    print(f"[FELLA DCEGM DEBUG] Saved inputs to {fname}", file=sys.stderr)
                    print(f"[FELLA DCEGM DEBUG] Input length = {len(egrid)}", file=sys.stderr)
                except Exception as err:
                    print("[FELLA DCEGM DEBUG] Failed to save inputs:", err, file=sys.stderr)
            # -----------------------------------------------------------------

            a_prime_refined_1D, egrid_refined_1D, c_refined_1D, vf_refined_1D, \
                dela_clean = dcegm(c, c, vf, a, egrid)
            print("Solved using DCEGM")

        elif method == 'RFC':
            # RFC method
            grad = cp.du(c)
            xr = np.array([egrid]).T
            vfr = np.array([vf]).T
            gradr = np.array([grad]).T
            pr = np.array([a]).T
            #mbar = 1.2
            radius = 0.75

            # Run RFC vectorized
            sub_points, roofRfc, close_ponts = rfc(xr, gradr, vfr, pr, m_bar, radius, 20)

            mask = np.ones(egrid.shape[0], dtype=bool)
            mask[sub_points] = False
            egrid_refined_1D = egrid[mask]
            vf_refined_1D = vfr[mask][:, 0]
            c_refined_1D = c[mask]
            a_prime_refined_1D = a[mask]

        return egrid_refined_1D, vf_refined_1D, c_refined_1D, a_prime_refined_1D, vf
    
    @njit
    def euler_error_fella(z_series, H_post_state, c_state, a_state, c_post_state):
        """
        Calculate the Euler error for the Fella housing model with a stationary policy function.
        
        Parameters:
        z_series : np.array
            The series of exogenous shock states (z) over time.
        H_post_state : np.array
            The policy function for housing choices (h_prime) over the post-state space (z, a, h).
        c_state : np.array
            The stationary policy function for consumption (c) over the state space (z, m, h).
        a_state : np.array
            The stationary policy function for liquid assets (a_prime) over the state space (z, m, h).
        c_post_state : np.array
            The future stationary policy function for consumption (c) over the post-state space (z, a, h).
        
        Returns:
        float
            The average log10 Euler error across exogenous states, asset grid points, and housing grid points.
        """

        # Initialize the Euler error array
        euler = np.full(len(z_series), np.nan)
        utility = np.full(len(z_series), np.nan)
        cons = np.full(len(z_series), np.nan)
        
        a_t = 0.1
        i_h_t = 1
        h_grid_index = np.arange(len(asset_grid_H))

        # Loop over exogenous states (z), asset grid (a), and housing grid (h)
        for t in range(len(z_series)):
            i_z = z_series[t]
            z = z_vals[i_z]

            i_h_prime = np.interp(a_t, asset_grid_A, H_post_state[i_z, :, i_h_t])
            # Find nearest housing grid point
            i_h_prime = np.argmin(np.abs(h_grid_index - i_h_prime))
            h = asset_grid_H[i_h_t]

            chi = 0
            if i_h_t != i_h_prime:
                chi = 1

            wealth = (R * a_t + z + chi * (h - asset_grid_H[i_h_prime]) 
                    - chi * np.abs(asset_grid_H[i_h_prime]) * phi)

            a_prime = np.interp(wealth, asset_grid_M, a_state[i_z, :, i_h_prime])
            c = np.interp(wealth, asset_grid_M, c_state[i_z, :, i_h_t])
            
            if a_prime < b or wealth < b or a_prime >= grid_max_A:
                continue

            RHS = 0
            for i_z_plus in range(len(z_vals)):
                c_plus = np.interp(a_prime,  asset_grid_A, c_post_state[i_z_plus, :, i_h_prime])
                RHS += Pi[i_z, i_z_plus] * uc(c_plus)

            # Compute the raw Euler error
            euler_raw = c - uc_inv(R * RHS * beta)
            euler[t] = np.log10(np.abs(euler_raw / c) + 1e-16)
            utility[t] = u(c, h)
            cons[t] = c

            # Update states
            a_t = a_prime
            i_h_t = i_h_prime

        
        # n cross_section 
        n_cross = int(np.sqrt(len(z_series)))

        # break utility into n_cross sections
        utility = utility.reshape(n_cross, n_cross)
        cons = cons.reshape(n_cross, n_cross)
        v_estimate = np.zeros(n_cross)
        for i in range(n_cross):
            for t in range(n_cross):
                v_estimate[i] = v_estimate[i] + utility[i, t]*beta**t

        # Return the average Euler error across all states
        return np.nanmean(euler), euler, v_estimate, cons[:,0]

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
    def invertEuler(V, sigma, lambda_e):
        """
        Invert the Euler equation to find the raw consumption policy function
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

            #UC_primes = beta * R * uc(sigma[:, i_a_prime, i_h_prime])
            VF_primes = beta * V[:, i_a_prime, i_h_prime]

            UC_primes = beta * R *lambda_e[:, i_a_prime, i_h_prime]

            c_t = uc_inv(np.dot(UC_primes, Pi[i_z, :]))

            vf_prime = np.dot(Pi[i_z, :], VF_primes)
            v_curr =  u(c_t, h_prime) + vf_prime 
            market_resources = a_prime + c_t
            e_grid_raw[i_z, i_a_prime, i_h_prime] = market_resources
            c_raw[i_z, i_a_prime, i_h_prime] = c_t
            v_raw[i_z, i_a_prime, i_h_prime] = v_curr

        return c_raw, v_raw, e_grid_raw
    
    @njit
    def H_choice(new_v_refined, a_prime_refined, c_refined, lambda_e_refined, m_bar =1.1):
        """ 
        Interpolate the refined value function and policy function on the 
        start-of-period state space and choose the optimal housing choice.

        Parameters
        ----------
        new_v_refined: 3D array
            Value function on active state space and conditioned on housing
            choice
        a_prime_refined: 3D array
            Asset policy on active state space and conditioned on housing
            choice
        c_refined: 3D array
            Consumption policy on active state space and conditioned on housing
            choice
        
        Returns
        -------
        v_new: 3D array
            Value function on start of period state space
        c_new: 3D array
            Consumption policy on start of period state
        a_new: 3D array
            Asset policy on start of period state space
        H_new: 3D array
            Housing policy on start of period state space
        """

        a_new = np.zeros(shape)
        H_new = np.zeros(shape)
        c_new = np.zeros(shape)
        v_new = np.zeros(shape)
        lambda_e_new = np.zeros(shape)

        V_new_big = np.zeros(shape_big)
        sigma_new_big = np.zeros(shape_big)
        a_new_big = np.zeros(shape_big)
        lambda_e_new_big = np.zeros(shape_big)

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

            wealth_curr = (a * R + z_vals[i_z] + chi * (h - h_prime) - 
                        phi * np.abs(h_prime) * chi)

            if wealth_curr < 0:
                V_new_big[i_z, i_a, i_h, i_h_prime] = -np.inf
            else:
                V_new_big[i_z, i_a, i_h, i_h_prime] = np.interp(
                    wealth_curr, asset_grid_M, new_v_refined[i_z, :, i_h_prime])
                sigma_new_big[i_z, i_a, i_h, i_h_prime] = np.interp(
                    wealth_curr, asset_grid_M, c_refined[i_z, :, i_h_prime])
                a_new_big[i_z, i_a, i_h, i_h_prime] = np.interp(
                    wealth_curr, asset_grid_M, a_prime_refined[i_z, :, i_h_prime])
                lambda_e_new_big[i_z, i_a, i_h, i_h_prime] = np.interp(
                    wealth_curr, asset_grid_M, lambda_e_refined[i_z, :, i_h_prime])

        # sharpen the jumps
        #V_new_big, sigma_new_big, a_new_big = sharpen_jumps(
         #   V_new_big, sigma_new_big, a_new_big, asset_grid_A, gradient_jump_threshold=m_bar)
        
        for i in range(len(X_all)):
            i_z = int(X_all[i][0])
            i_a = int(X_all[i][1])
            i_h = int(X_all[i][2])

            max_index = int(np.argmax(V_new_big[i_z, i_a, i_h, :]))

            v_new[i_z, i_a, i_h] = V_new_big[i_z, i_a, i_h, max_index]
            H_new[i_z, i_a, i_h] = max_index
            a_new[i_z, i_a, i_h] = a_new_big[i_z, i_a, i_h, max_index]
            c_new[i_z, i_a, i_h] = sigma_new_big[i_z, i_a, i_h, max_index]
            lambda_e_new[i_z, i_a, i_h] = uc(c_new[i_z, i_a, i_h])
        return v_new, c_new, a_new, H_new, lambda_e_new
    

    @njit
    def sharpen_jumps(V_new_big, sigma_new_big, a_new_big, x_values, gradient_jump_threshold=1.1):
        """
        Corrects jumps for each i,:,j,k in 4D arrays V_new_big, sigma_new_big, and a_new_big using the
        correct_jumps_gradient_with_policy_value_funcs function.

        Args:
            V_new_big (numpy.ndarray): 4D array representing the value function.
            sigma_new_big (numpy.ndarray): 4D array representing the policy function.
            a_new_big (numpy.ndarray): 4D array representing the auxiliary function.
            x_values (numpy.ndarray): 1D array representing the x-values along the second axis.
            gradient_jump_threshold (float): The threshold to detect and correct jumps.
        
        Returns:
            tuple: Corrected arrays V_new_big, sigma_new_big, and a_new_big.
        """
        # Loop over each i, j, k in the 4D arrays
        for i in range(V_new_big.shape[0]):  # Loop over the first dimension (i)
            for j in range(V_new_big.shape[2]):  # Loop over the third dimension (j)
                for k in range(V_new_big.shape[3]):  # Loop over the fourth dimension (k)

                    # Extract the 1D slices corresponding to i,:,j,k for each array
                    V_slice = V_new_big[i, :, j, k]
                    sigma_slice = sigma_new_big[i, :, j, k]
                    a_slice = a_new_big[i, :, j, k]

                    # Convert slices into a typed dict for policy and value functions
                    policy_value_funcs = Dict()
                    policy_value_funcs['v'] = V_slice
                    policy_value_funcs['a'] = a_slice

                    # Correct the jumps in these slices using the generic function
                    corrected_sigma_slice, corrected_policy_value_funcs = correct_jumps1d(
                        sigma_slice, x_values, gradient_jump_threshold, policy_value_funcs
                    )

                    # Store the corrected slices back into the original arrays
                    sigma_new_big[i, :, j, k] = corrected_sigma_slice
                    V_new_big[i, :, j, k] = corrected_policy_value_funcs['v']
                    a_new_big[i, :, j, k] = corrected_policy_value_funcs['a']

        return V_new_big, sigma_new_big, a_new_big


            

    #@njit
    def Euler_Operator(V, sigma, lambda_e, method='FUES', m_bar=1.2, lb=4):
        """
        Euler operator finds next period policy function using EGM and FUES

        Parameterrs
        ----------
        V: 3D array
            Value function for time t+1 conditioned to time t end of period 
        sigma: 3D array
            Consumption policy for time t+1 on t+1 start of period state
        method: str 
            Method to use for upper envelope (FUES, RFC, DCEGM)

        Returns
        -------
        results: dict
            Dictionary containing the updated policy functions and value functions



        """

        # The value function should be conditioned on time t
        # continuous state, time t discrete state, and time t+1 discrete state choice

        c_raw, v_raw, e_grid_raw = invertEuler(V, sigma, lambda_e)

        new_a_prime_refined1 = np.zeros(shape_active)
        new_c_refined1 = np.zeros(shape_active)
        new_v_refined1 = np.zeros(shape_active)
        new_lambda_e_refined1 = np.zeros(shape_active)
        new_v_refined1_dict = {}
        new_c_refined1_dict = {}
        new_a_prime_refined1_dict = {}
        new_e_refined_dict = {}

        for i in range(len(X_exog)):

            h_prime = asset_grid_H[X_exog[i][1]]  # t+1 housing
            i_h_prime = int(X_exog[i][1])
            i_z = int(X_exog[i][0])

            egrid_unrefined_1D = e_grid_raw[i_z, :, i_h_prime]
            a_prime_unrefined_1D = np.copy(asset_grid_A)
            c_unrefined_1D = c_raw[i_z, :, i_h_prime]
            vf_unrefined_1D = v_raw[i_z, :, i_h_prime]


            min_c_val = np.min(c_unrefined_1D)
            c_array = np.linspace(1e-100, min_c_val, n_con)
            e_array = c_array
            h_prime_array = np.zeros(n_con)
            h_prime_array.fill(h_prime)
            vf_array = u_vec(c_array, h_prime_array) + beta * np.dot(
                Pi[i_z, :], V[:, 0, i_h_prime])
            b_array = np.zeros(n_con)
            b_array.fill(asset_grid_A[0])

            egrid_unrefined_1D = np.concatenate((e_array, egrid_unrefined_1D))
            vf_unrefined_1D = np.concatenate((vf_array, vf_unrefined_1D))
            c_unrefined_1D = np.concatenate((c_array, c_unrefined_1D))
            a_prime_unrefined_1D = np.concatenate((b_array, a_prime_unrefined_1D))

            uniqueIds = uniqueEG(egrid_unrefined_1D, vf_unrefined_1D)
            egrid_unrefined_1D = egrid_unrefined_1D[uniqueIds]
            vf_unrefined_1D = vf_unrefined_1D[uniqueIds]
            c_unrefined_1D = c_unrefined_1D[uniqueIds]
            a_prime_unrefined_1D = a_prime_unrefined_1D[uniqueIds]

            start = time.time()
            
            egrid_refined_1D, vf_refined_1D, c_refined_1D, a_prime_refined_1D, dela_out = \
                EGM_UE(egrid_unrefined_1D, vf_unrefined_1D, c_unrefined_1D,
                    a_prime_unrefined_1D, vf_unrefined_1D, method=method,
                    m_bar=m_bar, lb=lb)
            
            new_v_refined1_dict[f"{i_z}-{i_h_prime}"] = vf_refined_1D
            new_c_refined1_dict[f"{i_z}-{i_h_prime}"] = c_refined_1D
            new_a_prime_refined1_dict[f"{i_z}-{i_h_prime}"] = a_prime_refined_1D
            new_e_refined_dict[f"{i_z}-{i_h_prime}"] = egrid_refined_1D

            UE_time = time.time() - start

            new_a_prime_refined1[i_z, :, i_h_prime] = interp_as(
                egrid_refined_1D, a_prime_refined_1D, asset_grid_M,extrap=False)
            new_c_refined1[i_z, :, i_h_prime] = interp_as(
                egrid_refined_1D, c_refined_1D, asset_grid_M, extrap=False)
            new_v_refined1[i_z, :, i_h_prime] = interp_as(
                egrid_refined_1D, vf_refined_1D, asset_grid_M, extrap=False)
            
            new_lambda_e_refined1[i_z, :, i_h_prime] = uc(new_c_refined1[i_z, :, i_h_prime])
        
        new_v_refined_bar, new_c_refined, new_a_prime_refined, new_H_refined, lambda_e_refined_bar = \
            H_choice(new_v_refined1, new_a_prime_refined1, new_c_refined1, new_lambda_e_refined1, m_bar=m_bar)

        results = {'post_state': {}, 'state': {}, 'EGM': {}}

        results['post_state']['c'] = new_c_refined
        results['post_state']['a'] = new_a_prime_refined
        results['post_state']['vf'] = new_v_refined_bar
        results['post_state']['H_prime'] = new_H_refined
        results['post_state']['lambda_e'] = lambda_e_refined_bar
        results['state']['c'] = new_c_refined1
        results['state']['a'] = new_a_prime_refined1
        results['state']['vf'] = new_v_refined1

        results['EGM']['unrefined'] = {}
        results['EGM']['refined'] = {}
        results['EGM']['unrefined']['v'] = v_raw
        results['EGM']['unrefined']['c'] = c_raw
        results['EGM']['unrefined']['e'] = e_grid_raw

        results['EGM']['refined']['v'] = new_v_refined1_dict
        results['EGM']['refined']['c'] = new_c_refined1_dict
        results['EGM']['refined']['a'] = new_a_prime_refined1_dict
        results['EGM']['refined']['e'] = new_e_refined_dict
    
        return results, UE_time

    return bellman_operator, Euler_Operator, condition_V, euler_error_fella

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
    _, Euler_Operator, _,_= Operator_Factory(cp)

    utility_values = cp.u_vec(cp.asset_grid_A[:, None], cp.asset_grid_H)

    # Initial values for value function, consumption, and assets
    shape = (len(cp.z_vals), len(cp.asset_grid_A), len(cp.asset_grid_H))
    V_init = np.zeros(shape)
    lambda_e_init = np.zeros(shape)
    for i_z in range(len(cp.z_vals)):
        V_init[i_z, :, :] = utility_values
    c_init = np.ones(shape) * (cp.asset_grid_A[:, None])  # Initial consumption policy
    lambda_e_init = cp.du(c_init)
    dela = np.ones((len(cp.z_vals), len(cp.asset_grid_H), len(cp.asset_grid_H)))
    lambda_e_new = np.copy(lambda_e_init)
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
        results, UE_time1 = Euler_Operator(V_new, c_new,lambda_e_new, method=method, m_bar = cp.m_bar, lb  = cp.lb)
        
        # Update error and policies
        bhask_error = np.max(np.abs(results['post_state']['vf'] - V_new))  # Calculate error based on policy function changes
        V_new = np.copy(results['post_state']['vf'])  # Update value function
        c_new = np.copy(results['post_state']['c'])  # Update consumption policy
        lambda_e_new = np.copy(results['post_state']['lambda_e'])  # Update lambda_e policy
        
        # average time taken for UE
        UE_time = (UE_time + UE_time1)/2

        k += 1  # Increment iteration count
        results['UE_time'] = UE_time
        
        if verbose == True:
            print(f'{method} Iteration {k}, Error: {bhask_error:.6f}')

    end_time = time.time()

    results['Total_iterations'] =k 

    return results

# Add a simple test function
def test_model():
    """Run a simple test of the housing model with default parameters"""
    print("Creating consumer problem instance...")
    cp = ConsumerProblem(
        grid_size=50,      # Smaller grid for faster testing
        grid_size_H=3
    )
    
    print("Running 5 iterations of Euler method (FUES)...")
    results = iterate_euler(cp, method="FUES", max_iter=5, tol=1e-6)
    
    print("\nTest completed successfully!")
    print(f"- Total iterations: {results['Total_iterations']}")
    print(f"- Average upper envelope time: {results['UE_time']*1000:.6f} ms")
    
    # Plot a simple figure of the consumption policy
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    i_z, i_h = 0, 1  # Use first shock state and middle housing value
    
    # Fix: check dimensions and make sure they match
    x_vals = cp.asset_grid_M
    y_vals = results['post_state']['c'][i_z, :, i_h]
    
    print(f"Debug - x_vals shape: {x_vals.shape}, y_vals shape: {y_vals.shape}")
    
    # Ensure same dimensions
    if len(x_vals) != len(y_vals):
        # If dimensions don't match, resample to match dimensions
        if len(x_vals) < len(y_vals):
            indices = np.linspace(0, len(y_vals)-1, len(x_vals), dtype=int)
            y_vals = y_vals[indices]
        else:
            indices = np.linspace(0, len(x_vals)-1, len(y_vals), dtype=int)
            x_vals = x_vals[indices]
    
    plt.plot(x_vals, y_vals, 'b-', label='Consumption Policy')
    plt.title(f'Consumption Policy Function (z={cp.z_vals[i_z]}, h={cp.asset_grid_H[i_h]})')
    plt.xlabel('Cash-on-Hand (M)')
    plt.ylabel('Consumption (c)')
    plt.grid(True)
    plt.legend()
    plt.savefig('fella_test_consumption.png')
    print("- Plot saved to 'fella_test_consumption.png'")

# Run the test if this file is executed directly
if __name__ == "__main__":
    print("Testing the Fella housing model...")
    test_model()




                           