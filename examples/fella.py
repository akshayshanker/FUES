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

import os
import sys

from HARK.interpolation import LinearInterp
#from HARK.dcegm import calc_segments, calc_multiline_envelope, calc_cross_points
from HARK.dcegm import calc_nondecreasing_segments, upper_envelope, calc_linear_crossing
from interpolation import interp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




# Import local modules
cwd = os.getcwd()
sys.path.append('..')
os.chdir(cwd)
from FUES.FUES2 import FUES
from FUES.RFC_simple import rfc
from FUES.DCEGM import dcegm



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
            Caling for housing in utility
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



def euler_fella_stationary(cp, sigma_work, hpol):
    """
    Calculate the Euler error for the Fella housing model with a stationary policy function.
    
    Parameters:
    cp : ConsumerProblem
        The consumer problem with model parameters.
    sigma_work : np.array
        The stationary policy function for consumption over the state space (z, a, h).
    hpol : np.array
        The stationary policy function for housing choices (h_prime) over the state space (z, a, h).
    
    Returns:
    float
        The average log10 Euler error across exogenous states, asset grid points, and housing grid points.
    """
    
    a_grid = cp.asset_grid_A  # Liquid assets grid
    h_grid = cp.asset_grid_H  # Housing grid
    z_vals = cp.z_vals        # Exogenous shock state

    # Initialize the Euler error array
    euler = np.zeros((len(z_vals), len(a_grid), len(h_grid)))
    euler.fill(np.nan)

    # Loop over exogenous states (z), asset grid (a), and housing grid (h)
    for i_z in range(len(z_vals)):
        z = z_vals[i_z]

        for i_a in range(len(a_grid)):
            a = a_grid[i_a]

            for i_h in range(len(h_grid)):
                h = h_grid[i_h]

                # 1. Interpolate consumption and housing policy for current state (a, h, z)
                c = np.interp(a, a_grid, sigma_work[i_z, :, i_h])
                i_h_prime = int(hpol[i_z, i_a, i_h])

                h_prime = h_grid[i_h_prime]

                # 2. Compute next period's assets (a_prime) and housing (h_prime)
                a_prime = cp.R * a + z - c - (h_prime - h) - np.abs(h_prime) * cp.phi  # Adjust for housing adjustment cost
                
                if a_prime < 0.001 or a_prime>5:
                    continue  # Avoid near-zero or negative assets

                # 3. Interpolate consumption for next period based on a_prime and h_prime (stationary policy)
                c_plus = np.interp(a_prime, a_grid, sigma_work[i_z, :, i_h_prime])

                # 4. Calculate the right-hand side of the Euler equation (discounted marginal utility)
                RHS = cp.beta * cp.R * cp.du(c_plus)

                # 5. Compute the raw Euler error
                euler_raw = c - cp.uc_inv(RHS)

                # 6. Normalize the Euler error and take log10 (added small value to avoid log(0))
                euler[i_z, i_a, i_h] = np.log10(np.abs(euler_raw / c) + 1e-16)

    # Return the average Euler error across all states
    return np.nanmean(euler)


## Function to wrap the upper envelope and select FUES or RFC 

def EGM_UE(egrid, vf, c, a, dela, endog_mbar = False, method = 'FUES', m_bar = 1.2):

    if method == 'FUES':

        policies_dict = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64[:],
            )
        
        policies_dict['a'] = np.array(a)
        policies_dict['c'] = np.array(c)
        policies_dict['vf'] = np.array(vf)
        test_pols = c

        egrid_refined_1D, vf_refined_1D, policies_clean_out, test_pols_clean = FUES(egrid,vf, policies_dict, test_pols,test_pols,   m_bar = 1.1, LB = 19, endog_mbar= False)

        c_refined_1D = policies_clean_out['c']
        a_prime_refined_1D = policies_clean_out['a']

    if method == 'DCEGM':
        a_prime_refined_1D, egrid_refined_1D,c_refined_1D, vf_refined_1D,dela_clean = dcegm(c,c,vf, a,egrid)
    
    elif method == 'RFC':
        #Generate inputs for RFC
        #print("RFC")
        grad = cp.du(c)
        xr = np.array([egrid]).T
        vfr =  np.array([vf]).T
        gradr = np.array([grad]).T
        pr =  np.array([a]).T
        mbar = 1.2
        radius = 0.5
        
        #run rfc_vectorized
        sub_points, roofRfc, close_ponts = rfc(xr,gradr,vfr,pr,mbar,radius, 20)


        mask = np.ones(egrid.shape[0] ,dtype=bool)
        mask[sub_points] = False
        egrid_refined_1D = egrid[mask]
        vf_refined_1D = vfr[mask][:,0]
        c_refined_1D = c[mask]
        a_prime_refined_1D = a[mask]
        #get del_a array
        #dela_clean = del_a_unrefined[mask]

        #print(vf_clean.shape)
    
    return egrid_refined_1D, vf_refined_1D, c_refined_1D, a_prime_refined_1D, vf

def Operator_Factory(cp):
    """ Operator that generates functions to solve model"""

    # tolerances
    tol_bell = 10e-10

    # unpack all variables
    beta, delta = cp.beta, cp.delta
    gamma_1 = cp.gamma_1
    xi = cp.xi
    asset_grid_A, asset_grid_H, z_vals, Pi = cp.asset_grid_A, cp.asset_grid_H,\
        cp.z_vals, cp.Pi
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
    shape_egrid = (len(z_vals), len(asset_grid_H), len(asset_grid_H))
    shape_big = (
        len(z_vals),
        len(asset_grid_A),
        len(asset_grid_H),
        len(asset_grid_H))

    @njit
    def interp_as(xp, yp, x, extrap=False):
        """Function  interpolates 1D
        with linear extraplolation

        Parameters
        ----------
        xp : 1D array
                        points of x values
        yp : 1D array
                        points of y values
        x  : 1D array
                        points to interpolate

        Returns
        -------
        evals: 1D array
                        y values at x

        """

        evals = np.zeros(len(x))
        if extrap and len(xp) > 1:
            for i in range(len(x)):
                if x[i] < xp[0]:
                    if (xp[1] - xp[0]) != 0:
                        evals[i] = yp[0] + (x[i] - xp[0]) * (yp[1] - yp[0])\
                            / (xp[1] - xp[0])
                    else:
                        evals[i] = yp[0]

                elif x[i] > xp[-1]:
                    if (xp[-1] - xp[-2]) != 0:
                        evals[i] = yp[-1] + (x[i] - xp[-1]) * (yp[-1] - yp[-2])\
                            / (xp[-1] - xp[-2])
                    else:
                        evals[i] = yp[-1]
                else:
                    evals[i] = np.interp(x[i], xp, yp)
        else:
            evals = np.interp(x, xp, yp)
            
        return evals

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
        """ Condition the t+1 continuation vaue on
        time t information"""

        # make the exogenuos state index the last
        #matrix_A_V = new_V_uc.transpose((1,2,0))
        #matrix_A_ua = new_Ud_a_uc.transpose((1,2,0))
        #matrix_A_uh = new_Ud_h_uc.transpose((1,2,0))

        # rows of EBA_P2 correspond to time t all exogenous state index
        # cols of EBA_P2 correspond to transition to t+1 exogenous state index
        #matrix_B = Pi

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
    
    #@njit 
    def H_choice(new_v_refined, new_a_prime_refined, new_c_refined):

        sigma_new = np.zeros(shape)
        a_new = np.zeros(shape)
        V_new = np.zeros(shape)
        H_new = np.zeros(shape)

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
                V_new_big[i_z, i_a, i_h, i_h_prime] = np.interp(wealth_curr, asset_grid_A, new_v_refined[i_z, :, i_h_prime])
                sigma_new_big[i_z, i_a, i_h, i_h_prime] = np.interp(wealth_curr, asset_grid_A, new_c_refined[i_z, :, i_h_prime])
                a_new_big[i_z, i_a, i_h, i_h_prime] = np.interp(wealth_curr, asset_grid_A, new_a_prime_refined[i_z, :, i_h_prime])
        
        for i in range(len(X_all)):

            i_z = int(X_all[i][0])
            i_a = int(X_all[i][1])
            i_h = int(X_all[i][2])
            assets = np.copy(asset_grid_A)* R + z_vals[i_z] + h
            
            # pick out max element
            max_index = int(np.argmax(V_new_big[i_z, i_a, i_h, :]))
    
            new_v_refined[i_z, i_a, i_h] = V_new_big[i_z, i_a, i_h,max_index]
            H_new[i_z, i_a, i_h] = max_index

            a_new[i_z, i_a,i_h] = a_new_big[i_z, i_a, i_h, max_index]
            new_c_refined[i_z,i_a, i_h] = sigma_new_big[i_z, i_a, i_h, max_index]
        
        return new_v_refined, new_c_refined, a_new, H_new
            
    
    #@njit
    def Euler_Operator(V, sigma, dela, method = 'FUES'):
        """
        Euler operator finds next period policy function
        using EGM and FUES"""

        # The value function should be conditioned on time t
        # continuous state, time t discrete state and time
        # t+1 discrete state choice

        c_raw, v_raw, e_grid_raw = invertEuler(V, sigma, dela)

        new_a_prime_refined = np.zeros(shape)
        new_c_refined = np.zeros(shape)
        new_v_refined = np.zeros(shape)

        for i in range(len(X_exog)):

            h_prime = asset_grid_H[X_exog[i][1]]  # t+1 housing
            i_h_prime = int(X_exog[i][1])
            i_z = int(X_exog[i][0])

            egrid_unrefined_1D = e_grid_raw[i_z, :, i_h_prime]
            a_prime_unrefined_1D = np.copy(asset_grid_A)
            c_unrefined_1D = c_raw[i_z, :, i_h_prime]
            vf_unrefined_1D = v_raw[i_z, :, i_h_prime]
            
            min_c_val = np.min(c_unrefined_1D)
            c_array = np.linspace(0.00001, min_c_val, 100)
            e_array = phi * np.abs(h_prime) + c_array
            h_prime_array = np.zeros(100)
            h_prime_array.fill(h_prime)
            vf_array = u_vec(c_array, h_prime_array) + beta * np.dot(Pi[i_z, :], V[:, 0, i_h_prime])
            b_array = np.zeros(100)
            b_array.fill(asset_grid_A[0])

            #print(egrid_unrefined_1D)

            egrid_unrefined_1D = np.concatenate((e_array, egrid_unrefined_1D))
            vf_unrefined_1D = np.concatenate((vf_array, vf_unrefined_1D))
            c_unrefined_1D = np.concatenate((c_array, c_unrefined_1D))
            a_prime_unrefined_1D = np.concatenate((b_array, a_prime_unrefined_1D))

            
            egrid_refined_1D, vf_refined_1D, c_refined_1D, a_prime_refined_1D, dela_out = \
                EGM_UE(egrid_unrefined_1D, vf_unrefined_1D, c_unrefined_1D, a_prime_unrefined_1D, vf_unrefined_1D, method = "FUES", m_bar =2)
                        
            new_a_prime_refined[i_z, :, i_h_prime] = interp_as(
                egrid_refined_1D, a_prime_refined_1D, asset_grid_A, extrap = True)
            new_c_refined[i_z, :, i_h_prime] = interp_as(
                egrid_refined_1D, c_refined_1D, asset_grid_A, extrap = True)
            new_v_refined[i_z, :, i_h_prime] = interp_as(
                egrid_refined_1D, vf_refined_1D, asset_grid_A, extrap = True)
            

        new_v_refined, new_c_refined, new_a_prime_refined, new_H_refined = H_choice(new_v_refined, new_a_prime_refined, new_c_refined)

        return new_v_refined, new_c_refined, new_a_prime_refined, new_H_refined,new_H_refined

    return bellman_operator, Euler_Operator, condition_V


def iterate_euler(cp, method="FUES", max_iter=200, tol=1e-4):
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
    h_new = np.ones(shape)
    c_init = np.ones(shape) * (cp.asset_grid_A[:, None] / 3)  # Initial consumption policy
    dela = np.ones((len(cp.z_vals), len(cp.asset_grid_H), len(cp.asset_grid_H)))

    # Initialize error and iteration counter
    bhask_error = np.inf
    k = 0

    # Copies of initial conditions
    V_new = np.copy(V_init)
    c_new = np.copy(c_init)
    h_new = np.copy(h_init)
    
    start_time = time.time()  # Track time

    # Euler iteration loop
    while k < max_iter and bhask_error > tol:
        # Perform one step of Euler operator based on the selected method
        V, cpol, apol, new_H_refined, dela_new = Euler_Operator(V_new, c_new, dela, method=method)
        
        # Update error and policies
        bhask_error = np.max(np.abs(cpol - c_new))  # Calculate error based on policy function changes
        V_new = np.copy(V)  # Update value function
        dela = np.copy(dela_new)  # Update dela
        c_new = np.copy(cpol)  # Update consumption policy
        h_new = np.copy(new_H_refined)  # Update housing policy

        k += 1  # Increment iteration count
        print(f'{method} Iteration {k}, Error: {bhask_error:.6f}')

    end_time = time.time()

    return {
        'V': V_new,  # Final value function
        'cpol': c_new,  # Final consumption policy
        'apol': apol,  # Final asset policy
        'h_new': h_new,  # Final housing policy
        'time': end_time - start_time,  # Time taken
        'iterations': k  # Number of iterations
    }


if __name__ == "__main__":
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter


    # Instantize the consumer problem with parameters 
    
    cp = ConsumerProblem(r=0,
                         r_H=0,
                         beta=.92,
                         delta=0,
                         Pi=((.5, 0.5), (.5, 0.5)),
                         z_vals=(4, 4),
                         b=1e-100,
                         grid_max_A=30,
                         grid_max_H=5,
                         grid_size=300,
                         grid_size_H=4,
                         gamma_1=0,
                         xi=0, kappa=0.75, phi=0.07, theta=0.77)

    bellman_operator, Euler_Operator, condition_V = Operator_Factory(cp)

    # Inital empty grids 
    shape = (len(cp.z_vals), len(cp.asset_grid_A), len(cp.asset_grid_H))
    shape_big = (len(cp.z_vals), len(cp.asset_grid_A),
                 len(cp.asset_grid_H), len(cp.asset_grid_H))
    shape_egrid = (len(cp.z_vals), len(cp.asset_grid_H), len(cp.asset_grid_H))
    V_init, h_init, a_init = np.empty(shape), np.empty(shape), np.empty(shape)
    V_init, Ud_prime_a_init, Ud_prime_h_init = np.ones(
        shape), np.ones(shape), np.ones(shape)
    V_pols, h_pols, a_pols = np.empty(shape), np.empty(shape), np.empty(shape)

    bell_error = 10
    bell_toll = 1e-4
    t = 0
    new_V = V_init
    max_iter = 20
    pl.close()

    sns.set(style="whitegrid",
            rc={"font.size": 10,
                "axes.titlesize": 10,
                "axes.labelsize": 10})
    fig, ax = pl.subplots(1, 2)

    # Solve via VFI and plot 
    start_time = time.time()
    while bell_error > bell_toll and t < max_iter:

        V = np.copy(new_V)
        a_pols_new, h_pols_new, V_pols_new, new_z_prime, new_V_adj_big, new_a_big, new_c_prime = bellman_operator(
            t, V)
        new_V, new_UD_a, new_UD_h = condition_V(
            V_pols_new, V_pols_new, V_pols_new)
        a_pols, h_pols, V_pols = np.copy(
            a_pols_new), np.copy(h_pols_new), np.copy(new_V)
        bell_error = np.max(np.abs(V - V_pols))
        print(t)
        new_V = V_pols
        t = t + 1
        print('Iteration {}, error is {}'.format(t, bell_error))

    print("VFI in {} seconds".format(time.time() - start_time))

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

    #E_error_bell = euler_fella_stationary(cp, new_c_prime, h_pols)

    #print(E_error_bell)
    #print(np.mean(new_V))


    results = iterate_euler(cp, method='FUES', max_iter=max_iter, tol=1e-03)

    V_new, c_new, apol, hpol  = results['V'], results['cpol'], results['apol'], results['h_new']

    E_error = euler_fella_stationary(cp, c_new, hpol)

    print(E_error)

    for i, col, lab in zip([1, 2, 3], ['blue', 'red', 'black'], [
            'H = low', ' H = med.', ' H = high']):
        ax[1].plot(cp.asset_grid_A, apol[1, :, i], color=col, label=lab)
        ax[0].plot(cp.asset_grid_A, a_pols_new[1, :, i], color=col, label=lab)

    ax[0].legend(frameon=False, prop={'size': 10})
    ax[0].set_title("VFI", fontsize=11)
    ax[1].set_title("FUES-EGM", fontsize=11)

    fig.tight_layout()
    pl.savefig('../plots/fella/Fella_policy.png')

    print(np.mean(V_new))
