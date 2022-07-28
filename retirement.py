"""Solution of Ishkakov et al (2017) retirement choice
model using FUES-EGM by Dobrescu and Shanker (2022).

Author: Akshay Shanker, University of Sydney, akshay.shanker@me.com.

"""

import numpy as np
from numba import jit
import time
import dill as pickle
from numba import njit, prange

from FUES import FUES


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
                 smooth_sigma = 0,
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

        self.u, self.du, self.uc_inv = u, du, uc_inv


def Operator_Factory(rm):
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
    y = cp.y
    smooth_sigma = cp.smooth_sigma

    R = cp.R
    b = cp.b
    T = cp.T

    @njit
    def interp_as(xp, yp, x, extrap=True):
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
    def Ts_ret(sigma_prime_ret,
               VF_prime_ret,
               t):
        """
        Generates time t policy for retiree

        Parameters
        ----------
        sigma_prime_ret : 1D array
                          t+1 period consumption function
        yp : VF_prime_ret
                t+1 period value function (retired)
        x  : 1D array
             points to interpolate
        t : int
             Age

        Returns
        -------
        sigma_ret_t: 1D array
                        consumption policy on assets at start of time t
        vf_ret_t: 1D array
                        time t value

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

            # evaluate cancidate policy function and value function
            sigma_ret_t_inv[i] = c_t
            vf_ret_t_inv[i] = u(c_t) + beta * VF_prime_ret[i]

        # min value of time t assets before which t+1 constraint binds
        min_a_val = endog_grid[0]

        # interpolate policy and value function on even grid
        sigma_ret_t = interp_as(endog_grid, sigma_ret_t_inv, asset_grid_A)
        vf_ret_t = interp_as(endog_grid, vf_ret_t_inv, asset_grid_A)

        # impose lower bound on liquid assets where a_t < min_a_val
        sigma_ret_t[np.where(asset_grid_A <= min_a_val)]\
            = asset_grid_A[np.where(asset_grid_A <= min_a_val)]
        vf_ret_t[np.where(asset_grid_A <= min_a_val)]\
            = u(asset_grid_A[np.where(asset_grid_A <= min_a_val)])\
            + beta * VF_prime_ret[0]

        return sigma_ret_t, vf_ret_t

    @njit
    def Ts_work(uc_prime_work,
                VF_prime_work,
                sigma_ret_t,
                vf_ret_t,
                t, m_bar):
        """
        Generates time t policy for worker

        Parameters
        ----------
        uc_prime_work: 1D array
                        t+1 period MUC. on t+1 state
                        if work choice = 1 at t
        VF_prime_work : 1D array
                        t+1 period VF on t+1 state
                        if work choice = 1 at t
        sigma_ret_t  : 1D array
                        t+1 consumption on t+1 state
                        if work choice at = 0
        vf_ret_t : 1D array
                    t+1 VF if work dec at t = 0
        t : int
             Age
        m_bar: float 
                jump detection threshold for FUES

        Returns
        -------
        uc_t: 1D array
                   unconditioned time t MUC on assets(t)
        sigma_work_t_inv: 1D array
                        unrefined consumption for worker at time t on wealth
        vf_t: 1D array
                        unconditioned time t value on assets(t)
        vf_work_t_inv: 1D array
                        unrefined time t value for worker at time t on wealth
        endog_grid: 1D array
                        unrefined endogenous grid
        sigma_work_t: 1D array
                        refined work choice for worke at time t
                        on start of  time t assets
        Notes
        -----

        """

        # Empty grids for time t consumption, vf, enog grid
        sigma_work_t_inv = np.zeros(grid_size)
        vf_work_t_inv = np.zeros(grid_size)
        endog_grid = np.zeros(grid_size)

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

        min_a_val = endog_grid[0]
        #print(min_a_val)

        # wealth grid points located at time t asset grid points
        asset_grid_wealth = R * asset_grid_A + y

        # remove sub-optimal points using FUES
        egrid1, vf_clean, sigma_clean,a_prime_clean, dela = FUES(
            endog_grid, vf_work_t_inv, sigma_work_t_inv, asset_grid_A, m_bar = 2)

        # interpolate on even start of period t asset grid for worker
        vf_work_t = interp_as(egrid1, vf_clean, asset_grid_wealth)
        sigma_work_t = interp_as(egrid1, sigma_clean, asset_grid_wealth)

        # binding t+1 asset constraint points
        sigma_work_t[np.where(asset_grid_wealth < min_a_val)]\
            = asset_grid_wealth[np.where(asset_grid_wealth < min_a_val)] - asset_grid_A[0]
        vf_work_t[np.where(asset_grid_wealth < min_a_val)]\
            = u(asset_grid_wealth[np.where(asset_grid_wealth < min_a_val)])\
            + beta * VF_prime_work[0] - delta

        # make current period discrete choices and unconditioned policies
        if smooth_sigma == 0:
            work_choice = vf_work_t > vf_ret_t
        else:
            work_choice = np.exp(vf_work_t/smooth_sigma)/((np.exp(vf_ret_t/smooth_sigma)\
                             + np.exp(vf_work_t/smooth_sigma)))

        sigma_t = work_choice * sigma_work_t + (1 - work_choice) * sigma_ret_t
        vf_t = work_choice * (vf_work_t) + (1 - work_choice) * vf_ret_t
        uc_t = du(sigma_t)

        return uc_t, sigma_work_t_inv, vf_t, vf_work_t_inv,\
            endog_grid, sigma_work_t

    return Ts_ret, Ts_work


if __name__ == "__main__":

    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter
    import matplotlib.pylab as pl

    # Create instance of RetirementModel 
    cp = RetirementModel(r=0.02,
                         beta=0.98,
                         delta=1,
                         y=20,
                         b=1E-1,
                         grid_max_A=500,
                         grid_size=2000,
                         T=20, 
                         smooth_sigma = 0)
    # Unpack solvers
    Ts_ret, Ts_work = Operator_Factory(cp)

    # Parameters
    T = cp.T
    grid_size = cp.grid_size

    # Values at terminal states 
    # Recall state-space is assets at t! before interest 
    # Recall exogenous grid is a (t+1) assets
    sigma_prime_terminal = np.copy(cp.asset_grid_A)
    v_prime_terminal = cp.u(sigma_prime_terminal)

    # Empty grids for retirees for each t
    sigma_retirees = np.empty((cp.T, cp.grid_size)) 
    vf_retirees = np.empty((cp.T, cp.grid_size))
    

    # Empty grids for workers for each t
    vf_work = np.empty((cp.T, cp.grid_size)) # value function for worker (unrefined)
    vf_uncond = np.empty((cp.T, cp.grid_size)) # value function refined unconditioned
    sigma_work = np.empty((cp.T, cp.grid_size)) # consumption function refined unconditioned
    sigma_work_dt = np.empty((cp.T, cp.grid_size)) # consumption policy function for worker (unrefined)
    e_grid = np.empty((cp.T, cp.grid_size)) # endogenous grid for worker (unrefined)

    # Step 1: Solve retiree policy
    sigma_prime_ret, VF_prime_ret\
        = np.copy(sigma_prime_terminal), np.copy(v_prime_terminal)

     # backward induction to solve for retirees each period 
    for i in range(T):
        age = int(T - i - 1)
        sigma_ret_t, vf_ret_t = Ts_ret(sigma_prime_ret, VF_prime_ret, age)
        sigma_retirees[age, :] = sigma_ret_t
        vf_retirees[age, :] = vf_ret_t
        sigma_prime_ret, VF_prime_ret = sigma_ret_t, vf_ret_t

    # Step 2: Solve general policy i.e. for person who enters as worker at start of t
    VF_prime_work = cp.u(sigma_prime_terminal)
    uc_prime_work = cp.du(sigma_prime_terminal)

   #time_start = time.time()
   # backward induction to solve for workers each period 
    for i in range(T):

        age = int(T - i - 1)
        uc_t, sigma_work_t_inv, vf_t, vf_work_t_inv, endog_grid, cons_pol = Ts_work(
            uc_prime_work, VF_prime_work, sigma_retirees[age, :], vf_retirees[age, :], age, 2)

        # store the grids for plotting 
        vf_work[age, :] = vf_work_t_inv
        vf_uncond[age, :] = vf_t
        sigma_work_dt[age, :] = sigma_work_t_inv
        e_grid[age, :] = endog_grid
        sigma_work[age] = cons_pol

        # next period inputs to solver 
        uc_prime_work = uc_t
        VF_prime_work = vf_t

    #print(time.time() - time_start)

    # Plot unrefined vs refined endogenous grid 
    for t in [17]:

        # get unrefined endogenous grid, value function and consumption 
        # for worker at time t
        x = e_grid[t]
        vf = vf_work[t]
        c = sigma_work_dt[t]
        a_prime = cp.asset_grid_A

        # generate refined grid, value function and policy using FUES 
        x_clean, vf_clean, c_clean,a_prime_clean, dela = FUES(x, vf, c, a_prime,2)

        for j in range(1):

            # Plot  upper envelope and asset policy 
            pl.close()
            fig, ax = pl.subplots(1, 2)
            sns.set(
                style="white", rc={
                    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})
            
            ax[0].scatter(
                x,
                vf * cp.beta - cp.delta,
                s=20,
                facecolors='none',
                edgecolors='r')
            ax[0].plot(
                x_clean,
                vf_clean * cp.beta - cp.delta,
                color='black',
                linewidth=1,
                label='Value function')
            ax[0].scatter(
                x_clean,
                vf_clean * cp.beta - cp.delta,
                color='blue',
                s=15,
                marker='x',
                linewidth=0.75)

            ax[0].set_ylim(7.75, 8.27)
            ax[0].set_xlim(44, 54.5)
            #ax[0].set_ylim(7.75,8.27)
            #ax[0].set_xlim(48,56)
            ax[0].set_xlabel('Assets (t)', fontsize=11)
            ax[0].set_ylabel('Value', fontsize=11)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['top'].set_visible(False)
            ax[0].legend(frameon=False, prop={'size': 10})
            ax[0].set_yticklabels(ax[0].get_yticks(), size=9)
            ax[0].set_xticklabels(ax[0].get_xticks(), size=9)
            ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

            ax[1].scatter(
                np.sort(x),
                np.take(
                    x  - c,
                    np.argsort(x)),
                s=20,
                facecolors='none',
                edgecolors='r',
                label='EGM points')
            ax[1].scatter(
                np.sort(x_clean),
                np.take(
                    x_clean - c_clean,
                    np.argsort(x_clean)),
                s=20,
                color='blue',
                marker='x',
                linewidth=0.75,
                label='Optimal points')

            ax[1].set_ylim(20,40)
            ax[1].set_xlim(44, 54.2)
            #ax[1].set_ylim(20,55)
            #ax[1].set_xlim(48,60)
            ax[1].set_ylabel('Assets (t+1)', fontsize=11)
            ax[1].set_xlabel('Assets (t)', fontsize=11)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[1].set_yticklabels(ax[1].get_yticks(), size=9)
            ax[1].set_xticklabels(ax[1].get_xticks(), size=9)
            ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            ax[1].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            fig.tight_layout()
            ax[1].legend(frameon=False, prop={'size': 10})
            fig.savefig('ret_vf_aprime_all_{}.png'.format(t))

    # Plot consumption policy 

    pl.close()
    sns.set(style="whitegrid",
            rc={"font.size": 10,
                "axes.titlesize": 10,
                "axes.labelsize": 10})
    fig, ax = pl.subplots(1, 1)
    
    for t, col, lab in zip([17, 10, 0], ['blue', 'red', 'black'], [
                           't=18', 't=10', 't=1']):

        cons_pol = np.copy(sigma_work[t])

        # remove jump joints 
        pos = np.where(np.abs(np.diff(cons_pol)) > 0.1)[0] + 1
        y1 = np.insert(cons_pol, pos, np.nan)
        x1 = np.insert(cp.asset_grid_A, pos, np.nan)

        ax.plot(x1, y1, color=col, label=lab)
        ax.set_xlim(0, 380)
        ax.set_ylim(0, 40)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticklabels(ax.get_yticks(), size=9)
        ax.set_xticklabels(ax.get_xticks(), size=9)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.set_ylabel('Consumption', fontsize=11)
        ax.set_xlabel('Assets (t)', fontsize=11)

    ax.legend(frameon=False, prop={'size': 10})
    fig.savefig('ret_cons_all.png'.format(t))

