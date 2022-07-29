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

from FUES import FUES


class ConsumerProblem:
    """
    A class that stores primitives for the Consumer Problem for
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
    Pi : array_like(float), optional(default=((0.60, 0.40),(0.05, 0.95))
            A 2D NumPy array giving the Markov matrix for {z_t}
    z_vals : array_like(float), optional(default=(0.5, 0.95))
            The state space of {z_t}
    b : float, optional(default=0)
            The borrowing constraint
    grid_max_A: float, optional(default=16)
                            max liquid assets
    grid_size : scalar(int), optional(default=50)
                            Number of grid points to solve problem, a grid on [-b, grid_max]
    gamma_1: float
    phi : float
                    ratio of h_prime that becomes fixed adjustment cost
    xi: float
    kappa: float
                    scaling for housing in utility
    theta: float
                    non-durable consumption share
    iota: float
                    housing utility constant shift term

    u : callable, optional(default=np.log)
            The utility function
    du : callable, optional(default=lambda x: 1/x)
            The derivative of u

    Attributes
    ----------


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
                 iota=0.01):

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
                                 np.arange(len(self.asset_grid_H)),
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
        def term_du(x):
            return theta / x

        @njit
        def du_inv(uc):
            return theta / uc

        self.u = u
        self.uc_inv = du_inv
        self.du = term_du


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
    def obj(a_prime, a, h, i_h, h_prime, i_h_prime, z, i_z, V, R, R_H, t):
        """Objective function to be *maximised* by Bellman operator


        """
        if i_h != i_h_prime:
            chi = 1
        else:
            chi = 0

        # wealth is cash at hand after cost of adjusting house paid
        wealth = R * a + z - (h_prime - h) - phi * h_prime * chi
        consumption = wealth - a_prime

        # get the t+1 value function
        Ev_prime = interp_as(
            asset_grid_A, V[i_z, :, i_h_prime], np.array([a_prime]))[0]

        # evaluate time t value conditional on consumption
        if a_prime >= b:
            return u(consumption, h_prime) + beta * Ev_prime
        else:
            return - np.inf

    @njit(parallel=True)
    def bellman_operator(t, V):
        """
        The approximate Bellman operator, which computes and returns the
        updated value function TV (or the V-greedy policy c if
        return_policy is True)

        Parameters
        ----------
        V : array_like(float)
                A NumPy array of dim len(cp.asset_grid) times len(cp.z_vals)

        Returns
        -------
        array_like(float)
                Returns either the greed policy given V or the updated value
                function TV.

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

        # loop over the state space
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

            for i_h_prime in range(len(asset_grid_H)):
                h_prime = asset_grid_H[i_h_prime]
                lower_bound = asset_grid_A[0]

                if i_h != i_h_prime:
                    chi = 1
                else:
                    chi = 0

                upper_bound = max(
                    asset_grid_A[0], R * a + z - (h_prime - h) - phi * h_prime * chi) + b

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

                wealth = R * a + z - (h_prime - h) - phi * h_prime * chi
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

        # numpy dot sum product over last 
        # axis of matrix_A (t+1 continuation value unconditioned)
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
    def Euler_Operator(V, sigma, dela):

        # The value function should be conditioned on time t
        # continuous state, time t discrete state and time
        # t+1 discrete state choice

        # empty refined grids conditioned of time t+1 housing
        new_a_prime_refined_big = np.ones(shape_big)
        new_c_refined_big = np.ones(shape_big)
        new_v_refined_big = np.ones(shape_big)
        new_v_refined_big_cons = np.ones(shape_big)

        # empty refined grids unconditioned of time t+1 housing
        new_a_prime_refined = np.empty(shape)
        new_c_refined = np.empty(shape)
        new_v_refined = np.empty(shape)
        new_H_refined = np.empty(shape)

        # unrefined grids conditioned of time t+1 housing
        endog_grid_unrefined_big = np.ones(shape_big)
        vf_unrefined_big = np.ones(shape_big)
        c_unrefined_big = np.ones(shape_big)
        dela_new = np.empty(shape_egrid)

        # First generate t+1 marginal utilities conditioned on time
        # t+1 continuous state, t and t+1 discrete choice
        # and **time t shock**

        new_UC_prime_c = np.ones(shape_big)

        for state in prange(len(X_all_big)):

            a_prime = asset_grid_A[X_all_big[state][1]]
            h = asset_grid_H[X_all_big[state][2]]  # t housing
            h_prime = asset_grid_H[X_all_big[state][3]]  # t+1 housing
            i_a_prime = int(X_all_big[state][1])
            i_h_prime = int(X_all_big[state][3])
            i_h = int(X_all_big[state][2])
            i_z = int(X_all_big[state][0])
            z = z_vals[i_z]

            if i_h != i_h_prime:
                chi = 1
            else:
                chi = 0

            UC_prime_zprimes = np.empty(len(z_vals))
            V_prime_zprimes = np.empty(len(z_vals))

            for i_z_prime in range(len(z_vals)):
                #print(sigma[i_z_prime, i_a_prime, i_h_prime])
                UC_prime_zprimes[i_z_prime] = uc(
                    sigma[i_z_prime, i_a_prime, i_h_prime])
                V_prime_zprimes[i_z_prime] = V[i_z_prime, i_a_prime, i_h_prime]

            new_UC_prime_c[i_z, i_a_prime, i_h, i_h_prime] = beta * \
                R * np.dot(Pi[i_z, :], UC_prime_zprimes)
            new_V_c_prime = beta * np.dot(Pi[i_z, :], V_prime_zprimes)

            c_t = uc_inv(new_UC_prime_c[i_z, i_a_prime, i_h, i_h_prime])

            # wealth is (1+r)a + z_vals
            wealth = (a_prime + (h_prime - h) + phi *
                      h_prime * chi + c_t + b - z) / R

            endog_grid_unrefined_big[i_z, i_a_prime, i_h, i_h_prime] = wealth
            new_c_refined_big[i_z, i_a_prime, i_h, i_h_prime] = c_t
            new_v_refined_big[i_z, i_a_prime, i_h,
                              i_h_prime] = u(c_t, h) + new_V_c_prime

        # now apply FUES and interp policy functions
        # loop over points that are held exogenous

        for i in range(len(X_exog)):

            h = asset_grid_H[X_exog[i][1]]  # t housing
            h_prime = asset_grid_H[X_exog[i][2]]  # t+1 housing
            i_h_prime = int(X_exog[i][2])
            i_h = int(X_exog[i][1])
            i_z = int(X_exog[i][0])
            z = z_vals[i_z]

            egrid_unrefined_1D = endog_grid_unrefined_big[i_z,
                                                          :, i_h, i_h_prime]
            a_prime_unrefined_1D = np.copy(asset_grid_A)
            c_unrefined_1D = new_c_refined_big[i_z, :, i_h, i_h_prime]
            vf_unrefined_1D = new_v_refined_big[i_z, :, i_h, i_h_prime]

            egrid_refined_1D, vf_refined_1D, c_refined_1D, a_prime_refined_1D, dela_out = \
                FUES(egrid_unrefined_1D, vf_unrefined_1D, c_unrefined_1D,
                     a_prime_unrefined_1D, m_bar=dela[i_z, i_h, i_h_prime] * 1.2)

            min_a_prime_val = egrid_refined_1D[np.argmin(a_prime_refined_1D)]
            dela_new[i_z, i_h, i_h_prime] = np.mean(dela_out[dela_out > 0])

            if i_h != i_h_prime:
                chi = 1
            else:
                chi = 0

            wealth_grid = np.copy(asset_grid_A)
            new_a_prime_refined_big[i_z, :, i_h, i_h_prime] = interp_as(
                egrid_refined_1D, a_prime_refined_1D, wealth_grid)
            new_c_refined_big[i_z, :, i_h, i_h_prime] = interp_as(
                egrid_refined_1D, c_refined_1D, wealth_grid)
            new_v_refined_big[i_z, :, i_h, i_h_prime] = interp_as(
                egrid_refined_1D, vf_refined_1D, wealth_grid)

            for k in range(len(asset_grid_A)):
                c_const_t = z + wealth_grid[k] * R - \
                    h_prime + h - chi * h_prime * phi - b

                new_v_refined_big_cons[i_z, k, i_h, i_h_prime] = u(
                    c_const_t, h_prime) + beta * np.dot(Pi[i_z, :], V[:, 0, i_h_prime])

                if wealth_grid[k] <= min_a_prime_val and asset_grid_A[k] < 2.5:
                    new_c_refined_big[i_z, k, i_h, i_h_prime] = c_const_t
                    new_v_refined_big[i_z,k,i_h,i_h_prime]\
                     = new_v_refined_big_cons[i_z, k,i_h,i_h_prime]
                    new_a_prime_refined_big[i_z, k, i_h, i_h_prime] = b

            aftr_adj_cash = z + wealth_grid * R - h_prime + h - chi * h_prime * phi
            new_v_refined_big[i_z, :, i_h,
                              i_h_prime][aftr_adj_cash <= 0] = -np.inf

        # make discrete choice

        for i in range(len(X_all)):

            i_z = int(X_all[i][0])
            i_a = int(X_all[i][1])
            i_h = int(X_all[i][2])
            # pick out max element
            new_v_refined[i_z, i_a, i_h] = np.max(
                new_v_refined_big[i_z, i_a, i_h, :])

            max_index = int(np.argmax(new_v_refined_big[i_z, i_a, i_h, :]))
            new_H_refined[i_z, i_a, i_h] = max_index

            new_a_prime_refined[i_z,i_a,i_h]\
             = new_a_prime_refined_big[i_z,i_a,i_h,max_index]

            new_c_refined[i_z,i_a,i_h]\
             = new_c_refined_big[i_z,i_a,i_h, max_index]

        return new_v_refined, new_c_refined, new_a_prime_refined, new_H_refined, dela_new

    return bellman_operator, Euler_Operator, condition_V


if __name__ == "__main__":
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter

    cp = ConsumerProblem(r=0,
                         r_H=0,
                         beta=.92,
                         delta=0,
                         Pi=((.5, 0.5), (.5, 0.5)),
                         z_vals=(4, 0.1),
                         b=1e-100,
                         grid_max_A=20,
                         grid_max_H=5,
                         grid_size=1000,
                         grid_size_H=4,
                         gamma_1=0,
                         xi=0, kappa=0.75, phi=0.27, theta=0.77)

    bellman_operator, Euler_Operator, condition_V = Operator_Factory(cp)

    # === Solve r.h.s. of Bellman equation === #
    shape = (len(cp.z_vals), len(cp.asset_grid_A), len(cp.asset_grid_H))
    shape_big = (len(cp.z_vals), len(cp.asset_grid_A),
                 len(cp.asset_grid_H), len(cp.asset_grid_H))
    shape_egrid = (len(cp.z_vals), len(cp.asset_grid_H), len(cp.asset_grid_H))
    V_init, h_init, a_init = np.empty(shape), np.empty(shape), np.empty(shape)
    V_init, Ud_prime_a_init, Ud_prime_h_init = np.ones(
        shape), np.ones(shape), np.ones(shape)
    V_pols, h_pols, a_pols = np.empty(shape), np.empty(shape), np.empty(shape)

    bell_error = 1
    bell_toll = 1e-4
    t = 0
    new_V = V_init
    max_iter = 200
    pl.close()

    sns.set(style="whitegrid",
            rc={"font.size": 10,
                "axes.titlesize": 10,
                "axes.labelsize": 10})
    fig, ax = pl.subplots(1, 2)

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

    #ax[0].set_ylim(7.75, 8.27)
    #ax[0].set_xlim(44, 54.5)
    # ax[0].set_ylim(7.75,8.27)
    # ax[0].set_xlim(48,56)
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
    #ax[1].legend(frameon=False, prop={'size': 10})
    ax[1].set_yticklabels(ax[0].get_yticks(), size=9)
    ax[1].set_xticklabels(ax[0].get_xticks(), size=9)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    for i, col, lab in zip([1, 2, 3], ['blue', 'red', 'black'], [
            'Low H(t)', 'Med. H(t)', 'High H(t)']):
        ax[0].plot(cp.asset_grid_A, a_pols_new[1, :, i], color=col, label=lab)

    # Euler iteration
    # Initial values
    V_init, c_init, a_init = np.ones(shape), np.ones(shape), np.ones(shape)

    for i in range(len(cp.X_all)):

        i_z = int(cp.X_all[i][0])
        i_a = int(cp.X_all[i][1])
        i_h = int(cp.X_all[i][2])

        c_init[i_z, i_a, i_h] = cp.asset_grid_A[i_a] / 3

    c_init = c_init

    bhask_error = 1
    bhask_toll = 1e-04
    max_iter = 200
    k = 0
    V_new = np.copy(V_init)
    c_new = np.copy(c_init)
    a_new = np.copy(a_init)
    dela = np.ones(shape_egrid) * 2

    start_time = time.time()
    while k < max_iter and bhask_error > bhask_toll:

        V, cpol, apol, new_H_refined, dela_new = Euler_Operator(
            V_new, c_new, dela)
        bhask_error = np.max(np.abs(cpol - c_new))
        V_new = np.copy(V)
        dela = np.copy(dela_new)
        c_new = np.copy(cpol)

        k = k + 1

        print('Euler iteration {}, error is {}'.format(k, bhask_error))
    print("EGM in {} seconds".format(time.time() - start_time))

    for i, col, lab in zip([1, 2, 3], ['blue', 'red', 'black'], [
            'H = low', ' H = med.', ' H = high']):
        ax[1].plot(cp.asset_grid_A, apol[1, :, i], color=col, label=lab)

    ax[0].legend(frameon=False, prop={'size': 10})
    ax[0].set_title("VFI", fontsize=11)
    ax[1].set_title("FUES-EGM", fontsize=11)

    fig.tight_layout()
    pl.savefig('plots/fella/Fella_policy.png')
