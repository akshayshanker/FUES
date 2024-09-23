"""

Author: Akshay Shanker, University of New South Wales, a.shanker@unsw.edu.au

"""
import numpy as np
import quantecon.markov as Markov
import quantecon as qe
from quantecon.optimize.root_finding import brentq
from numba import jit, prange
import time
import dill as pickle
from sklearn.utils.extmath import cartesian
from numba import njit, prange
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from interpolation.splines import extrap_options as xto
from interpolation import interp

import scipy
import matplotlib.pylab as pl

# local modules
from FUES.math_funcs import rootsearch, f, interp_as
from FUES.FUES import FUES


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
                 grid_size=200,
                 grid_size_H=200,
                 grid_size_W=200,
                 gamma_c=1.458,
                 K=200,
                 theta=2,
                 tau=0.2,
                 chi=0,
                 EGM_N=100,
                 tol_bel=1e-4,
                 m_bar=2,
                 T=60):
        self.grid_size = int(grid_size)
        self.r, self.R = r, 1 + r
        self.r_H, self.R_H = r_H, 1 + r_H
        self.beta = beta
        self.delta = delta
        self.gamma_c, self.chi = gamma_c, chi
        self.b = b
        self.T = T
        self.grid_max_A, self.grid_max_H = grid_max_A, grid_max_H
        self.sigma = sigma
        lambdas = np.array(config['lambdas'])
        self.alpha = alpha

        self.Pi, self.z_vals = np.array(Pi), np.asarray(z_vals)

        self.asset_grid_A = np.linspace(b, grid_max_A, grid_size)
        self.asset_grid_H = np.linspace(b, grid_max_H, grid_size_H)
        self.asset_grid_WE = np.linspace(b, grid_max_WE, grid_size_W)

        self.X_all = cartesian([np.arange(len(z_vals)),
                                np.arange(len(self.asset_grid_A)),
                                np.arange(len(self.asset_grid_H))])

        self.UGgrid_all = UCGrid((b, grid_max_A, grid_size),
                                 (b, grid_max_H, grid_size_H))

        self.tau = tau

        self.EGM_N = EGM_N
        self.tol_bel = tol_bel
        self.m_bar = m_bar

        # define functions
        @njit
        def du_c(x):
            if x <= 0:
                return 1e250
            else:
                return np.power(x, - gamma_c)

        @njit
        def du_c_inv(x):
            if x <= 0:
                return 1e250
            else:
                return np.power(x, -1 / gamma_c)

        @njit
        def du_h(y):
            if y <= 0:
                return 1e250
            else:
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

    @njit
    def roots(f, a, l, h_prime, z, Ud_prime_a, Ud_prime_h, t, eps=1e-3):

        sols_array = np.zeros(EGM_N)
        sols_array.fill(np.nan)
        i = 0
        while True:
            x1, x2 = rootsearch(f, a, l, eps, h_prime, z,
                                Ud_prime_a, Ud_prime_h, t)
            if np.isnan(x1) == False:
                a = x2
                root = brentq(f, x1, x2,
                              args=(h_prime, z, Ud_prime_a, Ud_prime_h, t),
                              xtol=1e-12)
                if root is not None and root[-1] is True:
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
            Ev_prime = eval_linear(UGgrid_all, V[i_z], point, xto.LINEAR)
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

            #obj_x_prime_noadj_zero = obj_noadj_scalr(b, w_2, h_prime, z, i_z, V,R, R_H,chi,t)

            # if obj_x_prime_noadj_zero> x_prime_nadj_star_1:
            #	x_prime_nadj_star = b
            # else:
            #	x_prime_nadj_star = x_prime_nadj_star_1

            a_prime = x_prime_nadj_star_1

            point = np.array([a_prime, h_prime])
            Ev_prime = eval_linear(UGgrid_all, V[i_z, :], point, xto.LINEAR)
            consumption = w_2 - a_prime

            return np.exp(u(consumption, h_prime, chi) + beta * Ev_prime)
        else:
            return -1e250

    @njit
    def bellman_operator(t, V):
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

        new_V = np.empty(V.shape)
        new_h_prime = np.empty(V.shape)  # next period capital
        new_c = np.empty(V.shape)
        new_a_prime = np.empty(V.shape)  # lisure
        new_V_adj = np.empty(V.shape)
        new_V_noadj = np.empty(V.shape)
        new_a_prime_adj = np.empty(V.shape)
        new_h_prime_adj = np.empty(V.shape)

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

            #obj_at_a_prime_adj_star_zero  = obj_noadj_scalr(b, w_adj_1, h_prime_adj_star, z, i_z, V,R, R_H,chi,t)

            # if obj_at_a_prime_adj_star_zero> a_prime_adj_star_2:
            #	a_prime_adj_star_1 = b
            # else:
            #	a_prime_adj_star_1 = a_prime_adj_star_2

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

            new_h_prime[i_z, i_a, i_h], new_a_prime[i_z, i_a, i_h],\
                new_V[i_z, i_a, i_h] = h_prime, a_prime, v
            new_c[i_z, i_a, i_h] \
                = consumption_adj * d_adj + (1 - d_adj) * consumption_noadj

            new_a_prime_adj[i_z, i_a, i_h] = a_prime_adj_star_1
            new_h_prime_adj[i_z, i_a, i_h] = h_prime_adj_star

        return new_a_prime, new_h_prime, new_V, new_c, \
            new_a_prime_adj, new_h_prime_adj

    @njit
    def root_A_liq_euler_inv(a_prime, h_prime, z, Ud_prime_a, Ud_prime_h, t):
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
                                     point, xto.LINEAR)

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
                                        Ud_prime_a, Ud_prime_h, t)

        point = np.array([a_prime, h_prime])
        Ud_prime_h_val = eval_linear(UGgrid_all, Ud_prime_h,
                                     point, xto.LINEAR)

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
            if np.isnan(a_prime_points[j]) == False:
                point_for_c = np.array([a_prime_points[j], h_prime])

                Ud_prime_h_val = eval_linear(UGgrid_all, Ud_prime_h,
                                             point_for_c, xto.LINEAR)

                c = du_c_inv((Ud_prime_h_val
                              + du_h(h_prime)) / (1 + tau))

                egm_wealth = c + a_prime_points[j] + h_prime * (1 + tau)

                e_grid_points[j] = egm_wealth
            else:
                break

        if len(np.where(a_prime_points == b)[0]) == 0:

            point_at_amin = np.array([b, h_prime])

            Ud_prime_h_val = eval_linear(UGgrid_all, Ud_prime_h,
                                         point_at_amin, xto.LINEAR)

            Ud_prime_a_val = eval_linear(UGgrid_all, Ud_prime_a,
                                         point_at_amin, xto.LINEAR)

            c_at_amin = du_c_inv((Ud_prime_h_val + du_h(h_prime)) / (1 + tau))

            # if du_c(c_at_amin)>= Ud_prime_a_val:
            a_prime_points[-1] = b
            egm_wealth_min = c_at_amin + b + h_prime * (1 + tau)
            e_grid_points[-1] = egm_wealth_min

        return a_prime_points, e_grid_points

    @njit
    def eval_no_adj_pol_egm_(Ud_prime_a, Ud_prime_h, V, t):
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

        endog_grid_unrefined = np.ones(shape)
        vf_unrefined = np.ones(shape)
        c_unrefined = np.ones(shape)

        for index_a_prime in range(len(asset_grid_A)):
            for index_h_today in range(len(asset_grid_H)):
                for index_z in range(len(z_vals)):
                    z = z_vals[index_z]

                    a_prime = asset_grid_A[index_a_prime]
                    h_prime = asset_grid_H[index_h_today] * (1 - delta)

                    egm_a, c = root_A_liq_euler_inv(a_prime, h_prime, z,
                                                    Ud_prime_a[index_z, :],
                                                    Ud_prime_h[index_z, :], t)
                    #c = max(1e-100, c)
                    endog_grid_unrefined[index_z,
                                         index_a_prime, index_h_today] = egm_a
                    c_unrefined[index_z,
                                index_a_prime, index_h_today] = c

                    point = np.array([a_prime, h_prime])

                    v_prime = beta * eval_linear(UGgrid_all,
                                                 V[index_z, :],
                                                 point, xto.LINEAR)

                    vf_unrefined[index_z, index_a_prime, index_h_today]\
                        = u(c, h_prime, 0) + v_prime

        return endog_grid_unrefined, vf_unrefined, c_unrefined

    @njit
    def refine_no_adj(endog_grid_unrefined,
                      vf_unrefined, c_unrefined, V_prime, t):
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

                e_grid_cean, vf_clean, c_clean, a_prime_clean, dela\
                    = FUES(endog_grid_unrefined_points, vf_unrefined_points,
                           c_unrefined_points, asset_grid_A,c_unrefined_points, b, m_bar=m_bar, endog_mbar = False)
                # print(endog_grid_unrefined_points)
                a_under_bar = min(e_grid_cean)

                new_a_prime_refined[index_z, :, index_h_today]\
                    = interp_as(e_grid_cean, a_prime_clean, asset_grid_A)
                new_c_refined[index_z, :, index_h_today]\
                    = interp_as(e_grid_cean, c_clean, asset_grid_A)
                new_v_refined[index_z, :, index_h_today]\
                    = interp_as(e_grid_cean, vf_clean, asset_grid_A)

                new_a_prime_refined[index_z, :, index_h_today][np.where(
                    new_a_prime_refined[index_z, :, index_h_today] < b)] = b

                for i_a in range(len(asset_grid_A)):
                    if asset_grid_A[i_a] < a_under_bar:
                        new_a_prime_refined[index_z, i_a, index_h_today] = b
                        point = np.array(
                            [b, asset_grid_H[index_h_today] * (1 - delta)])
                        c = asset_grid_A[i_a] - b
                        v_at_amin = u(c,
                                      asset_grid_H[index_h_today] * (1 - delta),
                                      0) + beta * eval_linear(UGgrid_all,
                                                              V_prime[index_z,
                                                                      :],
                                                              point,
                                                              xto.LINEAR)
                        new_v_refined[index_z, i_a, index_h_today] = v_at_amin
                        new_c_refined[index_z, i_a, index_h_today] = c

        return new_a_prime_refined, new_c_refined, new_v_refined,\
            e_grid_cean, vf_clean, c_clean, a_prime_clean

    @njit
    def eval_adj_pol_egm_(Ud_prime_a, Ud_prime_h, V, t):

        endog_grid_unrefined = np.zeros(
            (len(z_vals), len(asset_grid_H), EGM_N))
        vf_unrefined = np.zeros((len(z_vals), len(asset_grid_H), EGM_N))
        a_prime_unrefined = np.zeros((len(z_vals), len(asset_grid_H), EGM_N))
        h_prime_unrefined = np.zeros((len(z_vals), len(asset_grid_H), EGM_N))

        for index_h_prime in range(len(asset_grid_H)):
            for index_z in range(len(z_vals)):
                # printprint(index_h_prime)

                h_prime = asset_grid_H[index_h_prime]
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
                                                     point, xto.LINEAR)

                        vf_unrefined[index_z, index_h_prime, i] = u(
                            c_val, h_prime, chi) + v_prime
                        h_prime_unrefined[index_z, index_h_prime, i] = h_prime

                    else:
                        pass

        return endog_grid_unrefined, vf_unrefined, \
            a_prime_unrefined, h_prime_unrefined

    # @njit
    def refine_adj(endog_grid_unrefined,
                   vf_unrefined,
                   a_prime_unrefined,
                   h_prime_unrefined):

        # unrefined grids conditioned of time t+1 housing
        # returns function on *wealth*

        new_a_prime_refined = np.ones((len(z_vals), len(asset_grid_WE)))
        new_h_prime_refined = np.ones((len(z_vals), len(asset_grid_WE)))
        new_v_refined = np.ones((len(z_vals), len(asset_grid_WE)))

        # keep today's shock fixed

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

            # print(egrid_unref_points)

            e_grid_cean, vf_clean, a_prime_clean, a_prime_clean, dela\
                = FUES(egrid_unref_points, vf_unrefined_points,
                       aprime_unrefined_points, hprime_unrefined_points,a_prime_clean
                       b, m_bar=m_bar, endog_mbar = False)

            new_a_prime_refined[index_z, :]\
                = interp_as(e_grid_cean, a_prime_clean, asset_grid_WE)
            new_h_prime_refined[index_z, :]\
                = interp_as(e_grid_cean, hprime_clean, asset_grid_WE)
            new_v_refined[index_z, :]\
                = interp_as(e_grid_cean, vf_clean, asset_grid_WE)

            #print( )

        return new_a_prime_refined, new_h_prime_refined, new_v_refined,\
            e_grid_cean, vf_clean, hprime_clean, a_prime_clean,\
            vf_unrefined_points, hprime_unrefined_points,\
            aprime_unrefined_points,\
            egrid_unref_points

    # @njit
    def coleman_operator(t, V_prime,
                         Ud_prime_a,
                         Ud_prime_h):
        """"

        Iterates on the Coleman operator

        Note: V_prime,Ud_prime_a, Ud_prime_h assumed
                        to be conditioned on time t shock

                 - the t+1 marginal utilities are not multiplied
                   by the discount factor and rate of return
        """

        # Declare empty grids
        # uc indicates conditioned on time t shock
        new_a_prime = np.empty(V_prime.shape)
        new_h_prime = np.empty(V_prime.shape)
        new_c = np.empty(V_prime.shape)

        new_a_prime_adj = np.empty(V_prime.shape)
        new_h_prime_adj = np.empty(V_prime.shape)
        new_c_adj = np.empty(V_prime.shape)

        new_V_uc = np.empty(V_prime.shape)
        new_Ud_a_uc = np.empty(V_prime.shape)
        new_Ud_h_uc = np.empty(V_prime.shape)

        endog_grid_unrefined_noadj, vf_unrefined_noadj, c_unrefined_noadj\
            = eval_no_adj_pol_egm_(Ud_prime_a, Ud_prime_h, V_prime, t)

        a_prime_nadj, c_nadj, v_nadj, e_grid_cean, vf_clean,\
            c_clean, a_prime_clean\
            = refine_no_adj(endog_grid_unrefined_noadj,
                            vf_unrefined_noadj,
                            c_unrefined_noadj,
                            V_prime, t)

        endog_grid_unrefined_adj, vf_unrefined_adj, a_prime_unrefined_adj,\
            h_prime_unrefined_adj\
            = eval_adj_pol_egm_(Ud_prime_a, Ud_prime_h, V_prime, t)

        a_prime_adj, h_prime_adj, v_adj,\
            e_grid_cean, vf_clean, hprime_clean, a_prime_clean,\
            vf_unrefined_adj_1, h_prime_unrefined_adj_1, a_prime_unrefined_adj_1,\
            endog_grid_unrefined_adj_1\
            = refine_adj(endog_grid_unrefined_adj, vf_unrefined_adj,
                         a_prime_unrefined_adj, h_prime_unrefined_adj)

        adj_unrefined_grids = {}
        adj_unrefined_grids["endog_grid_unrefined_adj"] = endog_grid_unrefined_adj_1
        adj_unrefined_grids["vf_unrefined_adj"] = vf_unrefined_adj_1
        adj_unrefined_grids["a_prime_unrefined_adj"] = a_prime_unrefined_adj_1
        adj_unrefined_grids["h_prime_unrefined_adj"] = h_prime_unrefined_adj_1

        adj_nu_clean = {}
        adj_nu_clean["e_grid_cean"] = e_grid_cean
        adj_nu_clean["vf_clean"] = vf_clean
        adj_nu_clean["hprime_clean"] = hprime_clean
        adj_nu_clean["a_prime_clean"] = a_prime_clean

        adj_refined_grids = {}

        adj_refined_grids["a_prime_adj"] = a_prime_adj
        adj_refined_grids["h_prime_adj"] = h_prime_adj
        adj_refined_grids["v_adj"] = v_adj

        # evaluate t+1 liq assets as a function of t+1 illiquid
        for state in range(len(X_all)):

            a = asset_grid_A[X_all[state][1]]
            h = asset_grid_H[X_all[state][2]]
            i_a = X_all[state][1]
            i_h = X_all[state][2]
            i_z = int(X_all[state][0])
            z = z_vals[i_z]

            # for non-adjusters
            wealth_nadj = R * a + y_func(t, z)
            v_nadj_val = interp_as(
                asset_grid_A, v_nadj[i_z, :, i_h], np.array([wealth_nadj]))[0]
            c_nadj_val = interp_as(
                asset_grid_A, c_nadj[i_z, :, i_h], np.array([wealth_nadj]))[0]
            a_prime_nadj_val = interp_as(
                asset_grid_A, a_prime_nadj[i_z, :, i_h], np.array([wealth_nadj]))[0]
            h_prime_nadj_val = (1 - delta) * h

            # for adjusters
            wealth = R * a + R_H * h * (1 - delta) + y_func(t, z)

            a_adj_val = interp_as(asset_grid_WE, a_prime_adj[i_z, :],
                                  np.array([wealth]))[0]
            h_adj_val = interp_as(asset_grid_WE, h_prime_adj[i_z, :],
                                  np.array([wealth]))[0]
            c_adj_val = wealth - a_adj_val - h_adj_val * (1 + tau)

            points_adj = np.array([a_adj_val, h_adj_val])

            v_adj_val = u(c_adj_val, h_adj_val, chi)\
                + beta * eval_linear(UGgrid_all,
                                     V_prime[i_z],
                                     points_adj,
                                     xto.LINEAR)

            if v_adj_val >= v_nadj_val:
                d_adj = 1
            else:
                d_adj = 0

            point_nadj = np.array([a_prime_nadj_val, h_prime_nadj_val])

            new_h_prime[i_z, i_a, i_h] = d_adj * h_adj_val \
                + (1 - d_adj) * h_prime_nadj_val

            new_h_prime_adj[i_z, i_a, i_h] = h_adj_val

            new_a_prime[i_z, i_a, i_h] = d_adj * a_adj_val\
                + (1 - d_adj) * a_prime_nadj_val
            new_a_prime_adj[i_z, i_a, i_h] = a_adj_val

            new_c[i_z, i_a, i_h] = d_adj * c_adj_val\
                + (1 - d_adj) * c_nadj_val
            new_c_adj[i_z, i_a, i_h] = c_adj_val
            new_V_uc[i_z, i_a, i_h] = d_adj * v_adj_val\
                + (1 - d_adj) * v_nadj_val

            new_Ud_a_uc[i_z, i_a, i_h] = beta * R * d_adj * du_c(c_adj_val) \
                + (1 - d_adj) * beta * R * du_c(c_nadj_val)

            # margina value of housing today if not adjusting from POV of
            # t-1

            Phi_t = du_h(h_prime_nadj_val) \
                + eval_linear(UGgrid_all,
                              Ud_prime_h[i_z],
                              point_nadj, xto.LINEAR)

            new_Ud_h_uc[i_z, i_a, i_h] = beta * R_H * (1 - delta)\
                * (d_adj * du_c(c_adj_val)
                   + (1 - d_adj) * Phi_t)

        return new_a_prime, new_h_prime, new_V_uc, new_Ud_a_uc,\
            new_Ud_h_uc, new_c, \
            new_h_prime_adj, new_a_prime_adj, new_c_adj,\
            adj_unrefined_grids, adj_refined_grids, adj_nu_clean

    def condition_V(new_V_uc, new_Ud_a_uc, new_Ud_h_uc):
        """ Condition the t+1 continuation vaue on
        time t information"""

        new_V = np.zeros(np.shape(new_V_uc))
        new_UD_a = np.zeros(np.shape(new_Ud_a_uc))
        new_UD_h = np.zeros(np.shape(new_Ud_h_uc))

        # numpy dot sum product over last axis of matrix_A
        # (t+1 continuation value unconditioned)
        # see nunpy dot docs
        for state in range(len(X_all)):
            i_a = X_all[state][1]
            i_h = X_all[state][2]
            i_z = int(X_all[state][0])

            new_V[i_z, i_a, i_h] = np.dot(Pi[i_z, :], new_V_uc[:, i_a, i_h])
            new_UD_a[i_z, i_a, i_h] = np.dot(
                Pi[i_z, :], new_Ud_a_uc[:, i_a, i_h])
            new_UD_h[i_z, i_a, i_h] = np.dot(
                Pi[i_z, :], new_Ud_h_uc[:, i_a, i_h])

        return new_V, new_UD_a, new_UD_h

    return bellman_operator, coleman_operator, condition_V
