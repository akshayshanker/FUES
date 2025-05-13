"""
Author: Akshay Shanker, University of New South Wales, a.shanker@unsw.edu.au

"""
import numpy as np
import quantecon.markov as Markov
import quantecon as qe
from quantecon.optimize.root_finding import brentq
from numba import jit, prange
from numba.typed import Dict
import time
import dill as pickle
from sklearn.utils.extmath import cartesian
from numba import njit, prange
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from interpolation.splines import extrap_options as xto
from interpolation import interp

import scipy
import matplotlib.pylab as pl
import os, sys

# Import local modules
cwd = os.getcwd()
sys.path.append('..')
os.chdir(cwd)
from dc_smm.fues.fues import FUES as fues_alg, uniqueEG
from dc_smm.fues.rfc_simple import rfc
from dc_smm.fues.dcegm import dcegm
from dc_smm.fues.helpers.math_funcs import interp_as, rootsearch, correct_jumps1d

class ConsumerProblem:
    """
    A class that stores primitives for the Consumer Problem for
    model with fixed adjustment cost. The
    income process is assumed to be a finite state Markov chain.

    Parameters
    ----------
    r : scalar(float), optional(default=0.01)
            A strictly positive scalar giving the interest rate
    Lambda: scalar(float), optional(default = 0.1)
            The shadow social value of accumulation
    beta : scalar(float), optional(default=0.96)
            The discount factor, must satisfy (1 + r) * beta < 1
    Pi : array_like(float), optional(default=((0.60, 0.40),(0.05, 0.95))
            A 2D NumPy array giving the Markov matrix for {z_t}
    z_vals : array_like(float), optional(default=(0.5, 0.95))
            The state space of {z_t}
    b : scalar(float), optional(default=0)
            The borrowing constraint
    grid_max : scalar(float), optional(default=16)
            Max of the grid used to solve the problem
    grid_size : scalar(int), optional(default=50)
            Number of grid points to solve problem, a grid on [-b, grid_max]
    u : callable, optional(default=np.log)
            The utility function
    du_c : callable, optional(default=lambda x: 1/x)
            The derivative of u

    Attributes
    ----------
    r, beta, Pi, z_vals, b, u, du_c : see Parameters
    asset_grid : np.ndarray
            One dimensional grid for assets

    """

    def __init__(self,
                 config,
                 r=0.024,
                 sigma=1,
                 r_H=.1,
                 beta=.945,
                 alpha=0.66,
                 delta=0.1,
                 Pi=((0.09, 0.91), (0.06, 0.94)),
                 z_vals=(0.1, 1.0),
                 b=1e-2,
                 grid_max_A=50,
                 grid_max_WE=100,
                 grid_max_H=50,
                 grid_size=50,
                 grid_size_H=50,
                 grid_size_W=50,
                 gamma_c=1.458,
                 K=200,
                 theta=2,
                 tau=0.2,
                 chi=0,
                 EGM_N=25,
                 tol_bel=1e-4,
                 root_eps=1e-1,
                 m_bar=2,
                 stat = True, 
                 tol_timeiter = 1e-4,
                 T=60,T1=60, t0=50):
        
        self.grid_size = int(grid_size)
        self.r, self.R = r, 1 + r
        self.r_H, self.R_H = r_H, 1 + r_H
        self.beta = beta
        self.delta = delta
        self.gamma_c, self.chi = gamma_c, chi
        self.b = b
        self.T = T
        self.T1 = T1
        self.grid_max_A, self.grid_max_H = grid_max_A, grid_max_H
        self.sigma = sigma
        lambdas = np.array(config['lambdas'])
        self.alpha = alpha
        self.Pi, self.z_vals = np.array(Pi), np.asarray(z_vals)
        self.asset_grid_A = np.linspace(b, np.float64(grid_max_A), grid_size)
        self.asset_grid_H = np.linspace(b, np.float64(grid_max_H), grid_size_H)
        
        self.asset_grid_HE = np.linspace(b, np.float64(grid_max_H), grid_size_H*3)
        self.asset_grid_WE = np.linspace(b, np.float64(grid_max_WE), grid_size_W)
        
        self.X_all = cartesian([np.arange(len(z_vals)),
                                np.arange(len(self.asset_grid_A)),
                                np.arange(len(self.asset_grid_H))])

        self.UGgrid_all = UCGrid((b, grid_max_A, grid_size),
                                 (b, grid_max_H, grid_size_H))
        self.tau = tau
        self.EGM_N = EGM_N
        self.tol_bel = tol_bel
        self.m_bar = m_bar
        self.t0 = t0
        self.root_eps = root_eps
        self.tol_timeiter = tol_timeiter
        self.stat = stat

       # define functions
        @njit
        def du_c(x):
            #if x <= 0:
            #    return 1e250
            #else:
            return np.power(x, - gamma_c)

        @njit
        def du_c_inv(x):
            if x <= 0:
                return 1e250
            else:
                return np.power(x, -1 / gamma_c)

        @njit
        def du_h(y):
            #if y <= 0:
            #return 1e250
            #else:
            return alpha / y

        @njit
        def term_du(x):
            return theta * np.power(K + x, - gamma_c)

        @njit
        def term_u(x):
            return theta * (np.power(K + x, 1 - gamma_c) - 1) / (1 - gamma_c)

        @njit
        def u(x, y, chi):
            if x <= 0:
                cons_u = - np.inf
            elif y <= 0:
                cons_u - np.inf
            else:
                cons_u = (np.power(x, 1 - gamma_c) - 1) / (1 - gamma_c) \
                    + alpha * np.log(y)

            return cons_u - chi

        @njit
        def y_func(t, xi):

            if stat == True:
                t = T1
            else:
                t = t

            wage_age = np.dot(np.array([1, t, np.power(t, 2), np.power(
                t, 3), np.power(t, 4)]).astype(np.float64), lambdas[0:5])
            wage_tenure = t * lambdas[5] + np.power(t, 2) * lambdas[6]

            return np.exp(wage_age + wage_tenure + xi) * 1e-5

        self.u, self.du_c, self.term_u, self.term_du, self.y_func\
            = u, du_c, term_u, term_du, y_func
        self.du_c_inv = du_c_inv
        self.du_h = du_h

def Operator_Factory(cp):

    # tolerances
    tol_bell = cp.tol_bel
    m_bar = cp.m_bar

    beta, delta = cp.beta, cp.delta
    asset_grid_A, asset_grid_H, z_vals, Pi = cp.asset_grid_A, cp.asset_grid_H,\
        cp.z_vals, cp.Pi
    asset_grid_HE = cp.asset_grid_HE
    bg = np.zeros(len(asset_grid_A))
    bg.fill(cp.b)
    asset_grid_AC = np.concatenate((bg, asset_grid_A))
    grid_max_A, grid_max_H = cp.grid_max_A, cp.grid_max_H
    u, du_c, term_u = cp.u, cp.du_c, cp.term_u
    y_func = cp.y_func
    asset_grid_WE = cp.asset_grid_WE

    R, R_H = cp.R, cp.R_H
    X_all = cp.X_all
    b = cp.b
    T = cp.T
    chi = cp.chi
    sigma = cp.sigma
    tau = cp.tau
    du_c_inv = cp.du_c_inv
    du_h = cp.du_h
    EGM_N = cp.EGM_N
    z_idx = np.arange(len(z_vals))

    shape = (len(z_vals), len(asset_grid_A), len(asset_grid_H))
    V_init, h_init, c_init = np.empty(shape), np.empty(shape), np.empty(shape)
    UGgrid_all = cp.UGgrid_all

    root_eps = cp.root_eps

    @njit
    def roots(f, a, l, h_prime, z, Ud_prime_a, Ud_prime_h, t, eps=root_eps):

        sols_array = np.zeros(EGM_N)
        i = 0
        while True:
            x1, x2 = rootsearch(f, a, l, eps, h_prime, z,
                                Ud_prime_a, Ud_prime_h, t)
            if np.isnan(x1) == False:
                a = x2
                root = brentq(f, x1, x2,
                              args=(h_prime, z, Ud_prime_a, Ud_prime_h, t),
                              xtol=1e-12)
                if root is not None:
                    sols_array[i] = root[0]

            else:
                break
            i = i + 1

        return sols_array

    @njit
    def obj_noadj(a_prime, w, h, z, i_z, V, R, R_H, chi, t):

        # objective function to be *maximised* for non-adjusters

        if w - a_prime[0] > 0 and a_prime[0] > b:
            h_prime_nad = h
            point = np.array([a_prime[0], h])
            # if t > T-1:
            #	Ev_prime = term_u(h_prime*R_H*(1-delta) + a_prime[0]*R)
            # else:
            Ev_prime = eval_linear(UGgrid_all, V[i_z], point, xto.NEAREST)
            consumption = w - a_prime[0]

            return np.exp(u(consumption, h_prime_nad, chi) + beta * Ev_prime)
        else:
            return -1e250

    @njit
    def obj_noadj_scalr(a_prime, w, h, z, i_z, V, R, R_H, chi, t):

        return obj_noadj(np.array([a_prime]), w, h, z, i_z, V, R, R_H, chi, t)
        # objective function to be *maximised* for non-adjusters

    @njit
    def obj_adj(x_prime, a, h, z, i_z, V, R, R_H, t):

        # objective function to be *maximised* for adjusters

        h_prime = x_prime

        w_2 = R * a + R_H * h * (1 - delta) + \
            y_func(t, z) - h_prime - tau * h_prime

        if w_2 > 0 and h_prime >= b:
            args_nadj_2 = (w_2, h_prime, z, i_z, V, R, R_H, chi, t)
            bnds_nadj_2 = np.array([[0.05, w_2]])
            x0_no_adj = np.array([w_2 / 2 + b])

            x_prime_nadj_star_1 = qe.optimize.nelder_mead(obj_noadj,
                                                          x0_no_adj,
                                                          bounds=bnds_nadj_2,
                                                          args=args_nadj_2,
                                                          tol_x=1e-10)[0][0]
            a_prime = x_prime_nadj_star_1

            point = np.array([a_prime, h_prime])
            Ev_prime = eval_linear(UGgrid_all, V[i_z, :], point, xto.NEAREST)
            consumption = w_2 - a_prime

            return np.exp(u(consumption, h_prime, chi) + beta * Ev_prime)
        else:
            return -1e250
        
    @njit
    def obj_adj_nested(x_prime, wealth, i_z, V, keeper_pol_c, t):

        # objective function to be *maximised* for adjusters

        h_prime = x_prime

        w_2 = wealth - h_prime - tau * h_prime

        if w_2 > 0 and h_prime >0:

            c_keeper = eval_linear(UGgrid_all, keeper_pol_c, np.array([w_2, h_prime]), xto.NEAREST)

            a_prime = w_2 - c_keeper

            point = np.array([a_prime, h_prime])
            Ev_prime = eval_linear(UGgrid_all, V[i_z, :], point, xto.NEAREST)
            
            return u(c_keeper, h_prime, 0) + beta * Ev_prime
        else:
            return -np.inf
        
            
    @njit
    def obj_adj_nested_b(x_prime, wealth, i_z, V, keeper_pol, t):

        # objective function to be *maximised* for adjusters

        h_prime = x_prime

        w_2 = wealth - h_prime - tau * h_prime

        if w_2 > 0 and h_prime >= b:


            point = np.array([b, h_prime])
            Ev_prime = eval_linear(UGgrid_all, V[i_z, :], point, xto.NEAREST)
            
            consumption = w_2 - b

            return u(consumption, h_prime, 0) + beta * Ev_prime
        else:
            return -np.inf

    @njit
    def iterVFI(t, V):
        """
        The approximate Bellman operator, which computes and returns the
        updated value function TV (or the V-greedy policy c if
        return_policy is True).

        Parameters
        ----------
        V : array_like(float)
                A NumPy array of dim len(cp.asset_grid) times len(cp.z_vals)
        cp : ConsumerProblem
                An instance of ConsumerProblem that stores primitives
        return_policy : bool, optional(default=False)
                Indicates whether to return the greed policy given V or the
                updated value function TV.  Default is TV.

        Returns
        -------
        array_like(float)
                Returns either the greed policy given V or the updated value
                function TV.

        """

        Vcurr = np.empty(V.shape)
        Hnxt = np.empty(V.shape)  # next period capital
        Cnxt = np.empty(V.shape)
        Anxt = np.empty(V.shape)  # lisure
        Vcurr_adj = np.empty(V.shape)
        Vcurr_noadj = np.empty(V.shape)
        Aadj = np.empty(V.shape)
        Hadj = np.empty(V.shape)

        for state in range(len(X_all)):

            # unpack the state-space vals and indices
            a = asset_grid_A[X_all[state][1]]
            h = asset_grid_H[X_all[state][2]]
            i_a = X_all[state][1]
            i_h = X_all[state][2]
            i_z = int(X_all[state][0])
            z = z_vals[i_z]

            # bounds for adjusters, initial val and args
            # adjuster is maximising over h_prime at the top level
            # obj_adjh
            w_2_max_H = (R * a + R_H * h * (1 - delta) +
                         y_func(t, z)) / (1 + tau)

            bnds_adj = np.array([b, w_2_max_H])
            args_adj = (a, h, z, i_z, V, R, R_H, t)

            # maximise over h_prime, implicitly maximising over a_prime
            h_prime_adj_star = qe.optimize.brent_max(obj_adj, bnds_adj[0],
                                                     bnds_adj[1],
                                                     args=args_adj,
                                                     xtol=1e-10)[0]

            # set no adjust bounds, initial val and args
            # we take in wealth/ cash at hand as input state to obj_noadj

            w_noadj = R * a + y_func(t, z)
            bnds_nadj = np.array([[b, w_noadj]])
            x0_noadj = np.array([w_noadj / 2])
            args_nadj = (w_noadj, h * (1 - delta), z, i_z, V, R, R_H, 0, t)

            a_prime_nadj_star = qe.optimize.nelder_mead(obj_noadj, x0_noadj,
                                                        bounds=bnds_nadj,
                                                        args=args_nadj,
                                                        tol_f=1e-12,
                                                        tol_x=1e-12)[0]

            consumption_noadj = w_noadj - a_prime_nadj_star[0]

            v_adj = obj_adj(h_prime_adj_star,
                            a, h, z, i_z, V, R, R_H, t)

            v_nadj = obj_noadj_scalr(a_prime_nadj_star[0],
                                     w_noadj, h * (1 - delta), z, i_z, V, R, R_H, 0, t)

            # Get a_prime values back for adjusters
            w_adj_1 = R * a + R_H * h * (1 - delta) + y_func(t, z)\
                - h_prime_adj_star - tau * h_prime_adj_star

            args_nadj_1 = (w_adj_1, h_prime_adj_star,
                           z, i_z, V, R, R_H, chi, t)
            bnds_nadj_2 = np.array([[b, w_adj_1]])
            x0_nadj_2 = np.array([w_adj_1 / 2 + b])

            a_prime_adj_star_1 = qe.optimize.nelder_mead(
                obj_noadj, x0_nadj_2, bnds_nadj_2, args=args_nadj_1)[0][0]


            consumption_adj = R * a + R_H * h * (1 - delta)\
                + y_func(t, z)\
                - h_prime_adj_star\
                - a_prime_adj_star_1\
                - tau * h_prime_adj_star

            if v_adj >= v_nadj:

                d_adj = 1
            else:
                d_adj = 0

            v = d_adj * np.log(v_adj) + (1 - d_adj) * np.log(v_nadj)

            h_prime = d_adj * h_prime_adj_star\
                + (1 - d_adj) * R_H * h * (1 - delta)

            a_prime = d_adj * a_prime_adj_star_1\
                + (1 - d_adj) * a_prime_nadj_star[0]

            Hnxt[i_z, i_a, i_h], Anxt[i_z, i_a, i_h],\
                Vcurr[i_z, i_a, i_h] = h_prime, a_prime, v
            Cnxt[i_z, i_a, i_h] \
                = consumption_adj * d_adj + (1 - d_adj) * consumption_noadj

            Aadj[i_z, i_a, i_h] = a_prime_adj_star_1
            Hadj[i_z, i_a, i_h] = h_prime_adj_star

        return Vcurr, Anxt, Hnxt, Cnxt, Aadj, Hadj

    @njit
    def root_A_liq_euler_inv(a_prime, h_prime, z, Ud_prime_a, t):
        """ Gives inverse of liquid asset Euler and EGM point
                for non-adjusters

        Parameters
        ----------
        a_prime: float64
        h_prime: float64
                                t+1 housing value adjusted
        z: float64
                value of shock
        Ud_prime_a: 2D array
                                 discounted marginal utility of liq assets
                                 for given shock value today
        Ud_prime_h: 2D array
                                 discounted marginal utility of housing assets
                                 for given shock value today
        t: int
                time

        Returns
        -------
        egm_a: float64
                EGM point for a today
        c: float64
                consumption
        """

        #h_prime = (1-delta)*R_H*h
        point = np.array([a_prime, h_prime])

        Ud_prime_a_val = eval_linear(UGgrid_all,
                                     Ud_prime_a,
                                     point, xto.NEAREST)

        c = du_c_inv(Ud_prime_a_val)
        egm_a = c + a_prime  # - y_func(t,z)

        return egm_a, c

    @njit
    def housing_euler_resid_(a_prime, h_prime, z, Ud_prime_a, Ud_prime_h, t):
        """ Euler residual to housing
                Euler given h_prime and a_prime

        Parameters
        ----------
        a_prime: float64
        h_prime: float64
                                t+1 housing value adjusted
        z: float64
                value of shock
        Ud_prime_a: 2D array
                                 discounted marginal utility of liq assets
                                 for given shock value today
        Ud_prime_h: 2D array
                                 discounted marginal utility of housing assets
                                 for given shock value today
        t: int
                time

        Returns
        -------
        resid: float64


        """

        egm_a, c = root_A_liq_euler_inv(a_prime, h_prime, z,
                                        Ud_prime_a, t)

        point = np.array([a_prime, h_prime])
        Ud_prime_h_val = eval_linear(UGgrid_all, Ud_prime_h,
                                     point, xto.NEAREST)

        return du_c(c) * (1 + tau) - Ud_prime_h_val - du_h(h_prime)

    @njit
    def root_H_UPRIME_func(h_prime, z, Ud_prime_a, Ud_prime_h, t):
        """ Function returns a_prime roots of housing Euler equation
                for adjusters given h_prime.

        Parameters
        ----------
        h_prime: float64
                                t+1 housing value adjusted
        z: float64
                value of shock
        Ud_prime_a: 2D array
                                 discounted marginal utility of liq assets
                                 for given shock value today
        Ud_prime_h: 2D array
                                 discounted marginal utility of housing assets
                                 for given shock value today
        t: int
                time

        Returns
        -------
        a_prime_points: 1D array
                                         list with a_prime roots as non-zeros
        e_grid_points: 1D array
                                         EGM points of wealth/ cash at hand
                                         as non-zerops

        Notes
        -----
        Arrays a_prime_points and e_grid_points have len 100
        and Zeros in a_prime_points and e_grid_points are not roots.


        """

        # make empty array of zeros for EGM points
        e_grid_points = np.zeros(EGM_N)

        # get array of points of a_prime
        a_prime_points = roots(housing_euler_resid_,
                               asset_grid_A[0],
                               asset_grid_A[-1],
                               h_prime, z, Ud_prime_a, Ud_prime_h, t)

        for j in range(len(a_prime_points)):
            # recover consumption associated with liquid asset Euler
            if a_prime_points[j] > 0:
                point_for_c = np.array([a_prime_points[j], h_prime])

                Ud_prime_h_val = eval_linear(UGgrid_all, Ud_prime_h,
                                             point_for_c, xto.NEAREST)

                c = du_c_inv((Ud_prime_h_val
                              + du_h(h_prime)) / (1 + tau))

                egm_wealth = c + a_prime_points[j] + h_prime * (1 + tau)

                e_grid_points[j] = egm_wealth
            else:
                break

        if len(np.where(a_prime_points == b)[0]) == 0:

            point_at_amin = np.array([b, h_prime])

            Ud_prime_h_val = eval_linear(UGgrid_all, Ud_prime_h,
                                         point_at_amin, xto.NEAREST)

            Ud_prime_a_val = eval_linear(UGgrid_all, Ud_prime_a,
                                         point_at_amin, xto.NEAREST)

            c_at_amin = du_c_inv((Ud_prime_h_val + du_h(h_prime)) / (1 + tau))
            a_prime_points[-1] = b
            egm_wealth_min = c_at_amin + b + h_prime * (1 + tau)
            e_grid_points[-1] = egm_wealth_min

        return a_prime_points, e_grid_points

    @njit
    def _keeperEGM(Ud_prime_a, V, t):
        """
        Function produces unrefinded endogenous grids for non-adjusters

        Parameters
        ----------
        Ud_prime_a: 3D array
                                t+1 marginal discounted expected marginal shadow
                                value of liq. asssets
        Ud_prime_h: 3D array
                                t+1 marginal discounted expected marginal shadow
                                value of liq. asssets
        V: 3D array
                t+1 Value function undiscounted

        t: int
                time

        Returns
        -------
        endog_grid_unrefined: 3D array
                                                        unrefined EGM points (liq. assets)
        vf_unrefined: 3D array
                                         value at t for each EGM point and a_prime point
                                         as non-zerops
        c_unrefined: 3D array
                                        consumption for each EGM point

        Notes
        -----
        Input arrays are interpolants conditioned on time t states.

        """

        endog_grid_unrefined = np.ones((len(z_vals), int(len(asset_grid_A)*2), len(asset_grid_H)))
        
        vf_unrefined = np.ones((len(z_vals), int(len(asset_grid_A)*2), len(asset_grid_H)))
        c_unrefined = np.ones((len(z_vals), int(len(asset_grid_A)*2), len(asset_grid_H)))


        for index_z in range(len(z_vals)):
            for index_h_today in range(len(asset_grid_H)):
                z = z_vals[index_z]
                h_prime = asset_grid_H[index_h_today] * (1 - delta)
                egm_a0, c0 = root_A_liq_euler_inv(b, h_prime, z,
                                                    Ud_prime_a[index_z, :], t)
                c_at_a_zero_max = c0
                C_array = np.linspace(1e-60, max(1e-50,c0-1e-50), len(asset_grid_A))
                #print(C_array)
                point = np.array([b, h_prime])
                v_prime = beta * eval_linear(UGgrid_all,
                                                    V[index_z, :],
                                                    point, xto.NEAREST)
                
                for k in range(len(asset_grid_A)):
                    vf_unrefined[index_z, k, index_h_today] =  u(C_array[k], h_prime, 0) + v_prime
                
                    endog_grid_unrefined[index_z, k, index_h_today] = C_array[k] + b
                    c_unrefined[index_z, k, index_h_today] = C_array[k]
                
                for index_a_prime in range(len(asset_grid_A)):
                        
                        index_a_db = int(len(asset_grid_A) + index_a_prime)
                        a_prime = asset_grid_AC[index_a_db]
                        h_prime = asset_grid_H[index_h_today] * (1 - delta)

                        egm_a, c = root_A_liq_euler_inv(a_prime, h_prime, z,
                                                        Ud_prime_a[index_z, :], t)
                        #print(c)
                        #c = max(1e-100, c)
                        endog_grid_unrefined[index_z,
                                            index_a_db, index_h_today] = egm_a
                        c_unrefined[index_z,
                                    index_a_db, index_h_today] = c

                        point = np.array([a_prime, h_prime])

                        v_prime = beta * eval_linear(UGgrid_all,
                                                    V[index_z, :],
                                                    point, xto.NEAREST)

                        vf_unrefined[index_z, index_a_db, index_h_today]\
                            = u(c, h_prime, 0) + v_prime

        return endog_grid_unrefined, vf_unrefined, c_unrefined

    #@njit
    def _refineKeeper(endog_grid_unrefined,
                      vf_unrefined, c_unrefined, V_prime, t, method ='FUES', m_bar = 1.4):
        """
        Refine EGM grid for non-adjusters with FUES

        Parameters
        ----------
        endog_grid_unrefined: 3D array
                                t+1 marginal discounted expected marginal shadow
                                value of liq. asssets
        vf_unrefined: 3D array
                                t+1 marginal discounted expected marginal shadow
                                value of liq. asssets
        c_unrefined: 3D array
                t+1 Value function undiscounted

        V_prime: int
                time
        t

        Returns
        -------


        """

        # empty refined grids conditioned of time t+1 housing
        new_a_prime_refined = np.ones(shape)
        new_c_refined = np.ones(shape)
        new_v_refined = np.ones(shape)

        # keep today's housing fixed
        for index_h_today in range(len(asset_grid_H)):
            for index_z in range(len(z_vals)):
                
                vf_unrefined_points = vf_unrefined[index_z, :, index_h_today]
                endog_grid_unrefined_points\
                    = endog_grid_unrefined[index_z, :, index_h_today]
                c_unrefined_points = c_unrefined[index_z, :, index_h_today]
                
                #print(endog_grid_unrefined_points)
                uniqueIds = uniqueEG(endog_grid_unrefined_points, vf_unrefined_points)
                endog_grid_unrefined_points = endog_grid_unrefined_points[uniqueIds]
                vf_unrefined_points = vf_unrefined_points[uniqueIds]
                c_unrefined_points = c_unrefined_points[uniqueIds]
                asset_grid_AC_unique = asset_grid_AC[uniqueIds]
                
                if method == 'FUES':
                
                    e_grid_clean, vf_clean, a_prime_clean, c_clean, dela = \
                        FUES(endog_grid_unrefined_points, vf_unrefined_points,\
                                asset_grid_AC_unique,c_unrefined_points, vf_unrefined_points,\
                                m_bar=m_bar, LB=2, endog_mbar=False
                                )
                    
                if method == 'RFC':

                    v_raw = np.array([vf_unrefined_points]).T
                    sigma_raw= np.column_stack((c_unrefined_points,asset_grid_AC_unique))
                    gr_raw = np.array([du_c(c_unrefined_points)]).T
                    M_raw = np.array([endog_grid_unrefined_points]).T
                    grid = np.array([asset_grid_A]).T
                    gr_raw = np.array([du_c(c_unrefined_points)]).T

                    sub_points, tngv, closest_indices = rfc(M_raw,gr_raw,v_raw,sigma_raw, J_bar = m_bar, radius = 0.5, k =20)
                    
            
                    mask = np.ones(M_raw.shape[0] ,dtype=bool)
                    mask[sub_points] = False
                    e_grid_clean = M_raw[mask][:,0]
                    vf_clean = v_raw[mask][:,0]
                    c_clean = sigma_raw[mask,0]
                    a_prime_clean = sigma_raw[mask,1]
                
                if method == 'DCEGM':
                    a_prime_clean, e_grid_clean, c_clean,vf_clean, dela = \
                        dcegm(c_unrefined_points, c_unrefined_points, vf_unrefined_points,\
                                asset_grid_AC_unique,endog_grid_unrefined_points
                            )

                sortindex = np.argsort(e_grid_clean)    
                e_grid_clean = e_grid_clean[sortindex]
                vf_clean = vf_clean[sortindex]
                c_clean = c_clean[sortindex]
                a_prime_clean = a_prime_clean[sortindex]


                new_a_prime_refined[index_z, :, index_h_today]\
                    = interp_as(e_grid_clean, a_prime_clean, asset_grid_A)
                new_c_refined[index_z, :, index_h_today]\
                    = interp_as(e_grid_clean, c_clean, asset_grid_A)
                new_v_refined[index_z, :, index_h_today]\
                    = interp_as(e_grid_clean, vf_clean, asset_grid_A)

                new_a_prime_refined[index_z, :, index_h_today][np.where(
                    new_a_prime_refined[index_z, :, index_h_today] < b)] = b
                
                # remove jumps

                policy_value_funcs = Dict()
                policy_value_funcs['v'] = new_v_refined[index_z, :, index_h_today]
                policy_value_funcs['a'] = new_a_prime_refined[index_z, :, index_h_today]

                sharp_c, corrected_policy_value_funcs = correct_jumps1d(new_c_refined[index_z, :, index_h_today],\
                                                            asset_grid_A,
                                                            m_bar,
                                                            policy_value_funcs)
                
                new_a_prime_refined[index_z, :, index_h_today] = corrected_policy_value_funcs['a']
                new_c_refined[index_z, :, index_h_today] = sharp_c
                new_v_refined[index_z, :, index_h_today] = corrected_policy_value_funcs['v']

                
        return new_a_prime_refined, new_c_refined, new_v_refined,\
            e_grid_clean, vf_clean, c_clean, a_prime_clean

    @njit
    def _adjEGM(Ud_prime_a, Ud_prime_h, V, t):

        endog_grid_unrefined = np.zeros(
            (len(z_vals), len(asset_grid_HE), EGM_N))
        vf_unrefined = np.zeros((len(z_vals), len(asset_grid_HE), EGM_N))
        a_prime_unrefined = np.zeros((len(z_vals), len(asset_grid_HE), EGM_N))
        h_prime_unrefined = np.zeros((len(z_vals), len(asset_grid_HE), EGM_N))

        for index_h_prime in range(len(asset_grid_HE)):
            for index_z in range(len(z_vals)):

                h_prime = asset_grid_HE[index_h_prime]
                a_primes, e_grid_points = root_H_UPRIME_func(h_prime,
                                                             z_vals[index_z],
                                                             Ud_prime_a[index_z, :],
                                                             Ud_prime_h[index_z, :], t)
                
                endog_grid_unrefined[index_z, index_h_prime, :] = e_grid_points
                a_prime_unrefined[index_z, index_h_prime, :] = a_primes

                for i in range(len(a_primes)):
                    if a_primes[i] > 0:

                        point = np.array([a_primes[i], h_prime])
                        c_val = e_grid_points[i] - \
                            h_prime * (1 + tau) - a_primes[i]
                        v_prime = beta * eval_linear(UGgrid_all, V[index_z, :],
                                                     point, xto.NEAREST)

                        vf_unrefined[index_z, index_h_prime, i] = u(
                            c_val, h_prime, chi) + v_prime
                        h_prime_unrefined[index_z, index_h_prime, i] = h_prime

                    else:
                        pass

        return endog_grid_unrefined, vf_unrefined, \
            a_prime_unrefined, h_prime_unrefined

    #@njit
    def refine_adj(endog_grid_unrefined,
                   vf_unrefined,
                   a_prime_unrefined,
                   h_prime_unrefined, method= 'FUES', m_bar = 1.4):

        # unrefined grids conditioned of time t+1 housing
        # returns function on *wealth*

        new_a_prime_refined = np.ones((len(z_vals), len(asset_grid_WE)))
        new_h_prime_refined = np.ones((len(z_vals), len(asset_grid_WE)))
        new_v_refined = np.ones((len(z_vals), len(asset_grid_WE)))
        new_c_refined = np.ones((len(z_vals), len(asset_grid_WE)))


        for index_z in range(len(z_vals)):
            a_prime_unrefined_ur = np.ravel(a_prime_unrefined[index_z, :])

            vf_unrefined_points = np.ravel(vf_unrefined[index_z, :])[
                np.where(a_prime_unrefined_ur > 0)]
            hprime_unrefined_points = np.ravel(h_prime_unrefined[index_z, :])[
                np.where(a_prime_unrefined_ur > 0)]
            aprime_unrefined_points = np.ravel(a_prime_unrefined[index_z, :])[
                np.where(a_prime_unrefined_ur > 0)]
            egrid_unref_points = np.ravel(endog_grid_unrefined[index_z, :])[
                np.where(a_prime_unrefined_ur > 0)]
            

            c_unrefined_points = egrid_unref_points - hprime_unrefined_points * (1 + tau) - aprime_unrefined_points
            
            gr_h  = du_c(c_unrefined_points) 

            
            # Remove duplicates of EGM points by taking the max 
            uniqueIds = uniqueEG(egrid_unref_points, vf_unrefined_points)
            
            egrid_unref_points = egrid_unref_points[uniqueIds]
            vf_unrefined_points = vf_unrefined_points[uniqueIds]
            c_unrefined_points = c_unrefined_points[uniqueIds]
            aprime_unrefined_points = aprime_unrefined_points[uniqueIds]

            if method == 'FUES':
                e_grid_clean, vf_clean, a_prime_clean,hprime_clean, c_clean = \
                    FUES(egrid_unref_points, vf_unrefined_points,\
                            aprime_unrefined_points, hprime_unrefined_points,c_unrefined_points,\
                            m_bar=m_bar, LB=2, endog_mbar=False
                            )
            
            if method == 'RFC':

                # evaluate using RFC interpolation
                v_raw = np.array([vf_unrefined_points]).T
                sigma_raw= np.column_stack((hprime_unrefined_points,aprime_unrefined_points, c_unrefined_points))
                gr_raw = np.array([gr_h]).T
                M_raw = np.array([egrid_unref_points]).T
                grid = np.array([asset_grid_WE]).T

                sub_points, tngv, closest_indices = rfc(M_raw,gr_raw,v_raw,sigma_raw, J_bar = m_bar, radius = 0.05, k =50)

                mask = np.ones(M_raw.shape[0] ,dtype=bool)
                mask[sub_points] = False
                e_grid_clean = M_raw[mask][:,0]
                vf_clean = v_raw[mask][:,0]
                hprime_clean = sigma_raw[mask,0]
                a_prime_clean = sigma_raw[mask,1]
                c_clean = sigma_raw[mask,2]

    
            sortindex = np.argsort(e_grid_clean)
            e_grid_clean = e_grid_clean[sortindex]
            vf_clean = vf_clean[sortindex]
            hprime_clean = hprime_clean[sortindex]
            a_prime_clean = a_prime_clean[sortindex]
            c_clean = c_clean[sortindex]

            sigma_intersect = None
            M_intersect = None
            c_clean = e_grid_clean - hprime_clean * (1 + tau) - a_prime_clean
            
            new_a_prime_refined[index_z, :]\
                = interp_as(e_grid_clean, a_prime_clean, asset_grid_WE)
            new_h_prime_refined[index_z, :]\
                = interp_as(e_grid_clean, hprime_clean, asset_grid_WE)
            new_v_refined[index_z, :]\
                = interp_as(e_grid_clean, vf_clean, asset_grid_WE)
            new_c_refined[index_z, :]\
                = interp_as(e_grid_clean, c_clean, asset_grid_WE)

        return new_a_prime_refined, new_c_refined, new_h_prime_refined, new_v_refined,\
            e_grid_clean, vf_clean, hprime_clean, a_prime_clean,\
            vf_unrefined_points, hprime_unrefined_points,\
            aprime_unrefined_points,\
            egrid_unref_points,sigma_intersect, M_intersect
    
    #@njit
    def iterEGM(t, V_prime,
                         ELambdaAnxt,
                         ELambdaHnxt, method = 'FUES', m_bar = 1.4):
        
        """"

        Iterates on the Coleman operator

        Note: V_prime,Ud_prime_a, Ud_prime_h assumed
                        to be conditioned on time t shock

                 - the t+1 marginal utilities are not multiplied
                   by the discount factor and rate of return
        """

        # Declare empty grids
        # uc indicates conditioned on time t shock
        Anxt = np.empty(V_prime.shape)
        Hnxt = np.empty(V_prime.shape)
        Cnxt = np.empty(V_prime.shape)
        Dnxt = np.empty(V_prime.shape)

        Aadj = np.empty(V_prime.shape)
        Hadj = np.empty(V_prime.shape)
        #Cadj = np.empty(V_prime.shape)
 
        Vcurr = np.empty(V_prime.shape)
        LambdaAcurr = np.empty(V_prime.shape)
        LambdaHcurr = np.empty(V_prime.shape)

        endog_grid_unrefined_noadj, vf_unrefined_noadj, c_unrefined_noadj\
            = _keeperEGM(ELambdaAnxt, V_prime, t)

        Akeeper, Ckeeper, Vkeeper, e_grid_clean, vf_clean,\
            c_clean, a_prime_clean\
            = _refineKeeper(endog_grid_unrefined_noadj,
                            vf_unrefined_noadj,
                            c_unrefined_noadj,
                            V_prime, t,  method = method, m_bar = m_bar)

        endog_grid_unrefined_adj, vf_unrefined_adj, a_prime_unrefined_adj,\
            h_prime_unrefined_adj\
            = _adjEGM(ELambdaAnxt, ELambdaHnxt, V_prime, t)

        Aadj, Cadj, Hadj, Vadj,\
            e_grid_clean, vf_clean, hprime_clean, a_prime_clean,\
            vf_unrefined_adj_1, h_prime_unrefined_adj_1, a_prime_unrefined_adj_1,\
            endog_grid_unrefined_adj_1, sigma_intersect, M_intersect\
            = refine_adj(endog_grid_unrefined_adj, vf_unrefined_adj,
                         a_prime_unrefined_adj, h_prime_unrefined_adj, method = method, m_bar = m_bar)
        
        AdjGrids = {}
        AdjGrids["endog_grid_unrefined_adj"] = endog_grid_unrefined_adj_1
        AdjGrids["vf_unrefined_adj"] = vf_unrefined_adj_1
        AdjGrids["a_prime_unrefined_adj"] = a_prime_unrefined_adj_1
        AdjGrids["h_prime_unrefined_adj"] = h_prime_unrefined_adj_1
        AdjGrids["e_grid_clean"] = e_grid_clean
        AdjGrids["vf_clean"] = vf_clean
        AdjGrids["hprime_clean"] = hprime_clean
        AdjGrids["a_prime_clean"] = a_prime_clean
        AdjGrids['sigma_intersect'] = sigma_intersect
        AdjGrids['M_intersect'] = M_intersect
        AdjPol = {}
        AdjPol["A"] = Aadj
        AdjPol["H"] = Hadj
        AdjPol["V"] = Vadj
        AdjPol["C"] = Cadj
        KeeperPol = {}
        KeeperPol["A"] = Akeeper
        KeeperPol["C"] = Ckeeper
        KeeperPol["V"] = Vkeeper

        # Evaluate discrete choice on  the pre-state 
        for state in range(len(X_all)):

            a = asset_grid_A[X_all[state][1]]
            h = asset_grid_H[X_all[state][2]]
            i_a = X_all[state][1]
            i_h = X_all[state][2]
            i_z = int(X_all[state][0])
            z = z_vals[i_z]
            
            # non-adjusters
            wealth_nadj = R * a + y_func(t, z)
            v_nadj_val = interp_as(
                asset_grid_A, Vkeeper[i_z, :, i_h], np.array([wealth_nadj]))[0]
            c_nadj_val = interp_as(
                asset_grid_A, Ckeeper[i_z, :, i_h], np.array([wealth_nadj]), extrap= False)[0]
            a_prime_nadj_val = interp_as(
                asset_grid_A, Akeeper[i_z, :, i_h], np.array([wealth_nadj]))[0]
            h_prime_nadj_val = (1 - delta) * h

            # adjusters
            wealth = R * a + R_H * h * (1 - delta) + y_func(t, z)

            a_adj_val = interp_as(asset_grid_WE, Aadj[i_z, :],
                                  np.array([wealth]))[0]
            h_adj_val = interp_as(asset_grid_WE, Hadj[i_z, :],
                                  np.array([wealth]))[0]
            c_adj_val = interp_as(asset_grid_WE, Cadj[i_z, :],
                                    np.array([wealth]))[0]

            points_adj = np.array([a_adj_val, h_adj_val])

            v_adj_val = u(c_adj_val, h_adj_val, chi)\
                + beta * eval_linear(UGgrid_all,
                                     V_prime[i_z],
                                     points_adj,
                                     xto.NEAREST)

            if v_adj_val >= v_nadj_val:
                d_adj = 1
            else:
                d_adj = 0

            Dnxt[i_z, i_a, i_h] = d_adj
            point_nadj = np.array([a_prime_nadj_val, h_prime_nadj_val])
            Hnxt[i_z, i_a, i_h] = d_adj * h_adj_val \
                + (1 - d_adj) * h_prime_nadj_val
            Anxt[i_z, i_a, i_h] = d_adj * a_adj_val\
                + (1 - d_adj) * a_prime_nadj_val

            Cnxt[i_z, i_a, i_h] = d_adj * c_adj_val\
                + (1 - d_adj) * c_nadj_val
            Vcurr[i_z, i_a, i_h] = d_adj * v_adj_val\
                + (1 - d_adj) * v_nadj_val
            LambdaAcurr[i_z, i_a, i_h] = beta * R * d_adj * du_c(c_adj_val) \
                + (1 - d_adj) * beta * R * du_c(c_nadj_val)
            Phi_t = du_h(h_prime_nadj_val) \
                + eval_linear(UGgrid_all,
                              ELambdaHnxt[i_z],
                              point_nadj, xto.NEAREST)
            LambdaHcurr[i_z, i_a, i_h] = beta * R_H * (1 - delta)\
                * (d_adj * du_c(c_adj_val)
                   + (1 - d_adj) * Phi_t)

        return Vcurr, Hnxt, Cnxt,Dnxt, LambdaAcurr,LambdaHcurr, AdjPol, KeeperPol, AdjGrids
    
    
    @njit
    def _solveAdjNEGM(EVnxt,Ckeeper,
                            Akeeper,t):
        """"

        """

        a_prime_adj  = np.ones((len(z_vals), len(asset_grid_WE)))
        h_prime_adj  = np.ones((len(z_vals), len(asset_grid_WE)))
        c_adj = np.ones((len(z_vals), len(asset_grid_WE)))
        v_adj = np.ones((len(z_vals), len(asset_grid_WE)))
        #c_adj = np.ones((len(z_vals), len(asset_grid_WE)))

        for i_a in range(len(asset_grid_WE)):
            #for i_h in range(len(asset_grid_H)):
            for i_z in range(len(z_vals)):

                # Active state variable 
                liqAct = asset_grid_WE[i_a]
                z = z_vals[i_z]

                # Bounds for adjusters, initial val and args
                bnds_adj = np.array([b, liqAct/(1+tau)+b])
                args_adj = (liqAct, i_z, EVnxt, Ckeeper[i_z],t)

                # Maximize over h_prime, implicitly maximising over a_prime
                outadj = qe.optimize.brent_max(obj_adj_nested, 
                                                        bnds_adj[0],\
                                                        bnds_adj[1],\
                                                        args=args_adj)
                h_prime_adj_star1,v_prime_adj_star = outadj[0],outadj[1]

                # calculate max h_prime for a_prime = b
                outb =  qe.optimize.brent_max(obj_adj_nested_b, 
                                                        bnds_adj[0],\
                                                        bnds_adj[1],\
                                                            args=args_adj)
                
                h_prime_a_bound, v_prime_a_bound = outb[0],outb[1]
                
                # Wealth after housing expenditure                                       
                left_over_wealth = liqAct - h_prime_adj_star1* (1 + tau)
                
                a_prime_adj_star1 = eval_linear(UGgrid_all,\
                                                Akeeper[i_z],\
                                                np.array([left_over_wealth,\
                                                        h_prime_adj_star1]),\
                                                xto.NEAREST)
                
                abound_flag  = v_prime_adj_star<v_prime_a_bound
                h_prime_adj_star  = abound_flag*h_prime_a_bound\
                                     + (1-abound_flag)*h_prime_adj_star1
                a_prime_adj_star = abound_flag*b\
                                     + (1-abound_flag)*a_prime_adj_star1
                v_adj[i_z, i_a] = (1-abound_flag)*v_prime_adj_star\
                                     + abound_flag*v_prime_a_bound
                
                h_prime_adj[i_z, i_a] = h_prime_adj_star
                a_prime_adj[i_z, i_a] = a_prime_adj_star
                c_adj[i_z, i_a] = liqAct - h_prime_adj_star*(1 + tau) - a_prime_adj_star
        
        return a_prime_adj,c_adj, h_prime_adj, v_adj

    
    
    def iterNEGM(EVnxt, LambdaAnxt,ELambdaHnxt, t):
        
        """ 
        Solve iteration of using the nested EGM algorithm
        or hypbrid EGM-VFI algorithm

        Parameters
        ----------
        EVnxt: 3D array
                t+1 Expected Value function undiscounted

        
        """

        Anxt = np.empty(EVnxt.shape)
        Hnxt = np.empty(EVnxt.shape)
        Cnxt = np.empty(EVnxt.shape)
        Dnxt = np.empty(EVnxt.shape)
        Vcurr = np.empty(EVnxt.shape)
        LambdaAcurr = np.empty(EVnxt.shape)
        LambdaHcurr = np.empty(EVnxt.shape)

        # Eval the keeper policy using EGM
        endog_grid_unrefined_noadj, vf_unrefined_noadj, c_unrefined_noadj\
            = _keeperEGM(LambdaAnxt, EVnxt, t)

        Akeeper, Ckeeper, Vkeeper,_, _,_,_,\
            = _refineKeeper(endog_grid_unrefined_noadj,
                            vf_unrefined_noadj,
                            c_unrefined_noadj,
                            EVnxt, t, method = 'DCEGM')
        
        # VFI part for each level of liquidf wealth after wages are realised
        # and housing has been liquidated 

        Aadj, Cadj,Hadj, Vadj = _solveAdjNEGM(EVnxt,Ckeeper,Akeeper, t)

        AdjPol = {}
        AdjPol["A"] = Aadj
        AdjPol["H"] = Hadj
        AdjPol["V"] = Vadj
        AdjPol["C"] = Cadj
        KeeperPol = {}
        KeeperPol["A"] = Akeeper
        KeeperPol["C"] = Ckeeper
        KeeperPol["V"] = Vkeeper
       
        # Evaluate t+1 liq assets as a function of t+1 illiquid
        for state in range(len(X_all)):

            a = asset_grid_A[X_all[state][1]]
            h = asset_grid_H[X_all[state][2]]
            i_a = X_all[state][1]
            i_h = X_all[state][2]
            i_z = int(X_all[state][0])
            z = z_vals[i_z]

            # non-adjusters
            wealth_nadj = R * a + y_func(t, z)
            v_nadj_val = interp_as(
                asset_grid_A, Vkeeper[i_z, :, i_h], np.array([wealth_nadj]))[0]
            c_nadj_val = interp_as(
                asset_grid_A, Ckeeper[i_z, :, i_h], np.array([wealth_nadj]))[0]
            a_prime_nadj_val = interp_as(
                asset_grid_A, Akeeper[i_z, :, i_h], np.array([wealth_nadj]))[0]
            h_prime_nadj_val = (1 - delta) * h

            # adjusters
            wealth = R * a + R_H * h * (1 - delta) + y_func(t, z)

            a_adj_val = interp_as(asset_grid_WE, Aadj[i_z, :],
                                  np.array([wealth]))[0]
            h_adj_val = interp_as(asset_grid_WE, Hadj[i_z, :],
                                  np.array([wealth]))[0]
            c_adj_val = wealth - a_adj_val - h_adj_val * (1 + tau)

            points_adj = np.array([a_adj_val, h_adj_val])

            v_adj_val = u(c_adj_val, h_adj_val, chi)\
                + beta * eval_linear(UGgrid_all,
                                     EVnxt[i_z],
                                     points_adj,
                                     xto.NEAREST)
            d_adj = 0
            d_adj = v_adj_val >= v_nadj_val
            Dnxt[i_z, i_a, i_h] = d_adj
            
            Hnxt[i_z, i_a, i_h] = d_adj * h_adj_val \
                + (1 - d_adj) * h_prime_nadj_val

            #Hadj[i_z, i_a, i_h] = h_adj_val

            Anxt[i_z, i_a, i_h] = d_adj * a_adj_val\
                + (1 - d_adj) * a_prime_nadj_val
            #Aadj[i_z, i_a, i_h] = a_adj_val

            Cnxt[i_z, i_a, i_h] = d_adj * c_adj_val\
                + (1 - d_adj) * c_nadj_val
            #Cadj[i_z, i_a, i_h] = c_adj_val
            Vcurr[i_z, i_a, i_h] = d_adj * v_adj_val\
                + (1 - d_adj) * v_nadj_val

            LambdaAcurr[i_z, i_a, i_h] = beta * R * d_adj * du_c(c_adj_val) \
                + (1 - d_adj) * beta * R * du_c(c_nadj_val)
            
            
            point_nadj = np.array([a_prime_nadj_val, h_prime_nadj_val])
            Phi_t = du_h(h_prime_nadj_val) \
                + eval_linear(UGgrid_all,
                              ELambdaHnxt[i_z],
                              point_nadj, xto.NEAREST)
            LambdaHcurr[i_z, i_a, i_h] = beta * R_H * (1 - delta)\
                                                * (d_adj * du_c(c_adj_val)
                                                + (1 - d_adj) * Phi_t)
            
        return Vcurr,Hnxt, Cnxt,Dnxt,LambdaAcurr, LambdaHcurr, AdjPol, KeeperPol
    
    def condition_V(Vcurr, LambdaAcurr, LambdaHcurr):
        """ Condition the t+1 continuation vaue on
        time t information"""

        new_V = np.zeros(np.shape(Vcurr))
        new_UD_a = np.zeros(np.shape(LambdaAcurr))
        new_UD_h = np.zeros(np.shape(LambdaHcurr))

        # numpy dot sum product over last axis of matrix_A
        # (t+1 continuation value unconditioned)
        # see nunpy dot docs
        for state in range(len(X_all)):
            i_a = X_all[state][1]
            i_h = X_all[state][2]
            i_z = int(X_all[state][0])

            new_V[i_z, i_a, i_h] = np.dot(Pi[i_z, :], Vcurr[:, i_a, i_h])
            new_UD_a[i_z, i_a, i_h] = np.dot(
                Pi[i_z, :], LambdaAcurr[:, i_a, i_h])
            new_UD_h[i_z, i_a, i_h] = np.dot(
                Pi[i_z, :], LambdaHcurr[:, i_a, i_h])

        return new_V, new_UD_a, new_UD_h

    return iterVFI, iterEGM, condition_V, iterNEGM