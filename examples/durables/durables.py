"""
Author: Akshay Shanker, University of New South Wales, a.shanker@unsw.edu.au

"""
import numpy as np
import quantecon.markov as Markov
import quantecon as qe
from quantecon.optimize.root_finding import brentq
from numba import jit
import time
import dill as pickle
from sklearn.utils.extmath import cartesian
from numba import njit
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from interpolation.splines import extrap_options as xto
from interpolation import interp

import scipy
import matplotlib.pylab as pl
import os, sys
from consav import golden_section_search

# Import local modules
cwd = os.getcwd()
sys.path.append('..')
os.chdir(cwd)
from dc_smm.fues import FUES_jit_core, get_fues_defaults  # JIT-compiled core
from dc_smm.fues.fues_v0dev import uniqueEG  # Duplicate removal utility
from dc_smm.fues.helpers.math_funcs import interp_as_scalar  # 1D scalar interpolation

# Extract FUES defaults as module-level constants for numba compatibility
_fd = get_fues_defaults()
_FUES_ENDOG_MBAR = _fd['endog_mbar']
_FUES_INCLUDE_INTER = _fd['include_intersections']
_FUES_NO_DOUBLE = _fd['no_double_jumps']
_FUES_SINGLE_INTER = _fd['single_intersection']
_FUES_DISABLE_CHECKS = _fd['disable_jump_checks']
_FUES_LEFT_STRICT = _fd['left_turn_no_jump_strict']
_FUES_POST_STATE = _fd['use_post_state_jump_test']
_FUES_DETECT_DEC = _fd['detect_decreasing_policy']
_FUES_POST_CLEAN = _fd['post_clean']
_FUES_PAD_MBAR = _fd['padding_mbar']
_FUES_JUMP_TOL = _fd['jump_check_tol']
_FUES_EPS_D = _fd['eps_d']
_FUES_EPS_SEP = _fd['eps_sep']
_FUES_EPS_FWD = _fd['eps_fwd_back']
_FUES_PAR_GUARD = _fd['parallel_guard']
del _fd
from dc_smm.fues.rfc_simple import rfc
from dc_smm.fues.helpers.math_funcs import (interp_as, rootsearch, rootsearch_wf,
                                             bisect_wf, correct_jumps1d_arr,
                                             find_roots_piecewise_linear)


# =============================================================================
# NaN/Inf Clamping Helpers (prevents crashes from interpolation edge cases)
# =============================================================================

@njit
def clamp_value(arr, nan_val=-1e10):
    """Clamp NaN/inf in value functions. NaN → large negative so state is never chosen."""
    result = arr.copy()
    for i in range(len(result)):
        if np.isnan(result[i]) or np.isinf(result[i]):
            result[i] = nan_val
    return result


@njit
def clamp_policy(arr, min_val, max_val):
    """Clamp NaN/inf in policy functions to valid bounds."""
    result = arr.copy()
    for i in range(len(result)):
        if np.isnan(result[i]):
            result[i] = min_val
        elif result[i] < min_val:
            result[i] = min_val
        elif result[i] > max_val or np.isinf(result[i]):
            result[i] = max_val
    return result


@njit
def clamp_scalar(val, min_val, max_val, nan_replacement):
    """Clamp a single scalar value. Returns nan_replacement if NaN."""
    if np.isnan(val):
        return nan_replacement
    elif np.isinf(val):
        return max_val if val > 0 else min_val
    elif val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    return val


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
    grid_size_A : scalar(int), optional(default=50)
            Number of grid points for asset grid A, a grid on [-b, grid_max]
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
                 Pi=None,  # Will be generated from AR(1) if None
                 z_vals=None,  # Will be generated from AR(1) if None
                 phi_w=0.9170411,  # AR(1) persistence
                 sigma_w=0.0817012,  # AR(1) shock std dev
                 N_wage=3,  # Number of wage states for Tauchen
                 b=1e-2,
                 grid_max_A=50,
                 grid_max_WE=100,
                 grid_max_H=50,
                 grid_size_A=50,
                 grid_size_H=50,
                 grid_size_W=50,
                 gamma_c=1.458,  # Not used in log utility (kept for compatibility)
                 gamma_h=1.0,   # Not used in log utility (kept for compatibility)
                 kappa=1.0,     # Housing scaling factor in utility
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
                 T=60,T1=60, t0=50,
                 N_sim=10000):

        self.grid_size_A = int(grid_size_A)
        self.grid_size_H = int(grid_size_H)
        self.N_sim = config.get('N_sim', N_sim)  # Number of agents for simulation
        self.N_HD_LAMBDA = int(config.get('N_HD_LAMBDA', 1))  # HD grid multiplier for Euler errors
        # Read return_grids from environment variable (set in PBS script)
        self.return_grids = os.environ.get('FUES_RETURN_GRIDS', '0') == '1'
        self.r, self.R = r, 1 + r
        self.r_H, self.R_H = r_H, 1 + r_H
        self.beta = beta
        self.delta = delta
        self.gamma_c, self.gamma_h, self.chi = gamma_c, gamma_h, chi
        self.kappa = kappa
        self.b = b
        self.T = T
        self.T1 = T1
        self.grid_max_A, self.grid_max_H = grid_max_A, grid_max_H
        self.sigma = sigma
        self.alpha = alpha

        # Generate AR(1) wage process using Tauchen method
        # AR(1): z' = phi_w * z + sigma_w * epsilon, epsilon ~ N(0,1)
        self.phi_w = phi_w
        self.sigma_w = sigma_w
        self.N_wage = N_wage

        if Pi is None or z_vals is None:
            # Use Tauchen method to discretize AR(1) process
            # qe.markov.tauchen(n, rho, sigma, mu=0.0, n_std=3)
            # n = num states, rho = persistence, sigma = std of innovation, n_std = width in std devs
            labour_mc = qe.markov.tauchen(N_wage, phi_w, sigma_w, mu=0.0, n_std=3)
            self.z_vals = np.asarray(labour_mc.state_values)  # Log wage states
            self.Pi = np.asarray(labour_mc.P)  # Transition matrix

            # Enforce minimum probability floor to avoid tiny numbers
            min_prob = 1e-3
            self.Pi = np.maximum(self.Pi, min_prob)
            # Rescale rows to sum to 1
            self.Pi = self.Pi / self.Pi.sum(axis=1, keepdims=True)

            # Print the wage process
            print("\n" + "="*60)
            print("AR(1) Wage Process (Tauchen Method)")
            print("="*60)
            print(f"  Persistence (phi_w):     {phi_w:.6f}")
            print(f"  Shock std dev (sigma_w): {sigma_w:.6f}")
            print(f"  Number of states:        {N_wage}")
            print(f"\n  State values (log shock): {self.z_vals}")
            print(f"\n  Transition matrix P:")
            for i, row in enumerate(self.Pi):
                print(f"    State {i}: {row}")
            print("="*60 + "\n")
        else:
            # Use manually specified values
            self.Pi, self.z_vals = np.array(Pi), np.asarray(z_vals)

        self.asset_grid_A = np.linspace(b, np.float64(grid_max_A), grid_size_A)
        self.asset_grid_H = np.linspace(b, np.float64(grid_max_H), grid_size_H)

        self.asset_grid_HE = np.linspace(b, np.float64(grid_max_H), grid_size_H)
        self.asset_grid_WE = np.linspace(b, np.float64(grid_max_WE), grid_size_W)

        self.X_all = cartesian([np.arange(len(self.z_vals)),
                                np.arange(len(self.asset_grid_A)),
                                np.arange(len(self.asset_grid_H))])

        self.UGgrid_all = UCGrid((b, grid_max_A, grid_size_A),
                                 (b, grid_max_H, grid_size_H))
        self.tau = tau
        self.EGM_N = EGM_N
        self.tol_bel = tol_bel
        self.m_bar = m_bar
        self.t0 = t0
        self.root_eps = root_eps
        self.tol_timeiter = tol_timeiter
        self.stat = stat

        # Income function parameters
        lambdas = np.array(config.get('lambdas', [0, 0, 0, 0, 0, 0, 0]))
        tau_av = int(config.get('tau_av', 15))
        normalisation = float(config.get('normalisation', 1e-5))  # Ensure float for numba
        tzero = 20  # Starting age for tenure calculation (fixed at 20)

        # Store for reference
        self.lambdas = lambdas
        self.tau_av = tau_av
        self.normalisation = normalisation
        self.tzero = tzero

        # Print income function parameters
        print("\n" + "-"*60)
        print("Income Function Parameters")
        print("-"*60)
        print(f"  Age polynomial coeffs (lambdas[0:5]):")
        print(f"    {lambdas[0:5]}")
        print(f"  Tenure polynomial coeffs (lambdas[5:7]):")
        print(f"    {lambdas[5:7]}")
        print(f"  Max tenure (tau_av): {tau_av}")
        print(f"  Starting age (tzero): {tzero}")
        print(f"  Normalisation: {normalisation}")

        # Compute actual wages at age 35 for each shock state
        t_example = 35
        tau_example = min(tau_av, t_example - tzero)
        age_factors = np.array([1.0, float(t_example), float(t_example)**2,
                                float(t_example)**3, float(t_example)**4])
        wage_age = np.dot(age_factors, lambdas[0:5])
        tenure_factors = np.array([float(tau_example), float(tau_example)**2])
        wage_tenure = np.dot(tenure_factors, lambdas[5:7])
        wages_at_35 = np.exp(wage_age + wage_tenure + self.z_vals) * normalisation
        print(f"\n  Example: Wages at age {t_example} (tenure={tau_example}):")
        for i, (z, w) in enumerate(zip(self.z_vals, wages_at_35)):
            print(f"    State {i}: shock={z:.4f}, wage={w:.6f}")
        print("-"*60 + "\n")

       # define functions
        # CRRA utility form: u(c,h) = alpha * c^(1-gamma_c)/(1-gamma_c) + (1-alpha) * (kappa*h)^(1-gamma_h)/(1-gamma_h)
        # alpha = consumption weight, (1-alpha) = housing weight, kappa = housing scaling
        @njit
        def du_c(x):
            # Marginal utility of consumption: alpha * c^(-gamma_c)
            if x <= 0:
                return 1e250
            return alpha * np.power(x, -gamma_c)

        @njit
        def du_c_inv(m):
            # Inverse marginal utility: given du_c = alpha * c^(-gamma_c), c = (alpha/m)^(1/gamma_c)
            if m <= 0:
                return 1e250
            return np.power(alpha / m, 1.0 / gamma_c)

        @njit
        def du_h(y):
            # Marginal utility of housing: (1-alpha) * kappa^(1-gamma_h) * h^(-gamma_h)
            if y <= 0:
                return 1e250
            return (1.0 - alpha) * np.power(kappa, 1.0 - gamma_h) * np.power(y, -gamma_h)

        @njit
        def term_du(x):
            # Terminal marginal utility (CRRA form)
            if x <= 0:
                return 1e250
            return theta * alpha * np.power(K + x, -gamma_c)

        @njit
        def term_u(x):
            # Terminal utility (CRRA form)
            if x <= 0:
                return -np.inf
            return theta * alpha * np.power(K + x, 1.0 - gamma_c) / (1.0 - gamma_c)

        @njit
        def u(x, y, chi):
            # CRRA utility: alpha * c^(1-gamma_c)/(1-gamma_c) + (1-alpha) * (kappa*h)^(1-gamma_h)/(1-gamma_h)
            if x <= 0:
                cons_u = -np.inf
            elif y <= 0:
                cons_u = -np.inf
            else:
                c_utility = alpha * np.power(x, 1.0 - gamma_c) / (1.0 - gamma_c)
                h_utility = (1.0 - alpha) * np.power(kappa * y, 1.0 - gamma_h) / (1.0 - gamma_h)
                cons_u = c_utility + h_utility

            return cons_u - chi

        # Retirement parameters
        retirement_age = 60
        retirement_income = 0.1  # Nominal $10,000/year after normalisation

        @njit
        def y_func(t, xi):
            """
            Income function with age-tenure profile and AR(1) shock.

            Parameters
            ----------
            t : int
                Time period (age)
            xi : float
                Log wage shock from AR(1) process (from z_vals)

            Returns
            -------
            float
                Income level

            Notes
            -----
            - Before retirement: Wage = exp(age_polynomial + tenure_polynomial + xi) * normalisation
            - After retirement (age > 60): Fixed income of 0.1 (with certainty, no shock)
            - Tenure is capped at tau_av
            """
            # After retirement age, return fixed income with certainty
            if t > retirement_age:
                return retirement_income

            # Tenure: years since tzero, capped at tau_av
            tau_tenure = min(tau_av, t - tzero)

            # Age polynomial: lambdas[0:5] on [1, t, t^2, t^3, t^4]
            age_factors = np.array([1.0, float(t), np.power(float(t), 2),
                                    np.power(float(t), 3), np.power(float(t), 4)])
            wage_age = np.dot(age_factors, lambdas[0:5])

            # Tenure polynomial: lambdas[5:7] on [tau, tau^2]
            tenure_factors = np.array([float(tau_tenure), np.power(float(tau_tenure), 2)])
            wage_tenure = np.dot(tenure_factors, lambdas[5:7])

            return np.exp(wage_age + wage_tenure + xi) * normalisation

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

    # HD grid for Lambda_H computation (N_HD_LAMBDA times finer)
    N_HD_LAMBDA = getattr(cp, 'N_HD_LAMBDA', 1)
    n_a_hd = len(asset_grid_A) * N_HD_LAMBDA
    n_h_hd = len(asset_grid_H) * N_HD_LAMBDA
    asset_grid_A_HD = np.linspace(b, grid_max_A, n_a_hd)
    asset_grid_H_HD = np.linspace(b, grid_max_H, n_h_hd)
    UGgrid_HD = UCGrid((b, grid_max_A, n_a_hd), (b, grid_max_H, n_h_hd))
    use_hd_lambda = N_HD_LAMBDA > 1
    du_h = cp.du_h
    EGM_N = cp.EGM_N
    gamma_c = cp.gamma_c
    alpha = cp.alpha
    z_idx = np.arange(len(z_vals))

    shape = (len(z_vals), len(asset_grid_A), len(asset_grid_H))
    V_init, h_init, c_init = np.empty(shape), np.empty(shape), np.empty(shape)
    UGgrid_all = cp.UGgrid_all

    root_eps = cp.root_eps
    return_grids = getattr(cp, 'return_grids', False)  # Whether to store EGM grids

    @njit
    def roots(f, a, l, h_prime, z, Ud_prime_a, Ud_prime_h, t, eps=root_eps):
        sols_array = np.zeros(EGM_N)
        i = 0
        while i < EGM_N:  # Guard against overflow
            x1, x2 = rootsearch(f, a, l, eps, h_prime, z,
                                Ud_prime_a, Ud_prime_h, t)
            # NaN check: x1 != x1 is True only for NaN (faster than np.isnan)
            if x1 != x1:
                break
            a = x2
            root = brentq(f, x1, x2,
                          args=(h_prime, z, Ud_prime_a, Ud_prime_h, t),
                          xtol=1e-08)
            if root is not None:
                sols_array[i] = root[0]
                i += 1
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
            a_prime = x_prime_nadj_star_1

            point = np.array([a_prime, h_prime])
            Ev_prime = eval_linear(UGgrid_all, V[i_z, :], point, xto.LINEAR)
            consumption = w_2 - a_prime

            return np.exp(u(consumption, h_prime, chi) + beta * Ev_prime)
        else:
            return -1e250

    @njit
    def obj_adj_min(x_prime, a, h, z, i_z, V, R, R_H, t):
        """Negated objective for golden_section_search (minimizer)."""
        return -obj_adj(x_prime, a, h, z, i_z, V, R, R_H, t)

    @njit
    def obj_adj_nested(x_prime, wealth, i_z, V, keeper_pol_c, t):

        # objective function to be *maximised* for adjusters

        h_prime = x_prime

        w_2 = wealth - h_prime - tau * h_prime

        # Check bounds to avoid extrapolation issues
        if w_2 > 0 and h_prime > 0 and h_prime <= grid_max_H and w_2 <= grid_max_A:

            c_keeper = eval_linear(UGgrid_all, keeper_pol_c, np.array([w_2, h_prime]), xto.LINEAR)

            # Ensure c_keeper is valid
            if c_keeper <= 0 or c_keeper > w_2:
                return -np.inf

            a_prime = max(b, w_2 - c_keeper)

            # Check a_prime is within grid
            if a_prime > grid_max_A:
                return -np.inf

            point = np.array([a_prime, h_prime])
            Ev_prime = eval_linear(UGgrid_all, V[i_z, :], point, xto.LINEAR)

            return u(c_keeper, h_prime, 0) + beta * Ev_prime
        else:
            return -np.inf

    @njit
    def obj_adj_nested_min(x_prime, wealth, i_z, V, keeper_pol_c, t):
        """Negated objective for golden_section_search (minimizer)."""
        return -obj_adj_nested(x_prime, wealth, i_z, V, keeper_pol_c, t)


    @njit
    def obj_adj_nested_b(x_prime, wealth, i_z, V, keeper_pol, t):

        # objective function to be *maximised* for adjusters (at borrowing constraint a'=b)

        h_prime = x_prime

        w_2 = wealth - h_prime - tau * h_prime

        # Check bounds to avoid extrapolation issues
        if w_2 > 0 and h_prime >= b and h_prime <= grid_max_H:

            consumption = w_2 - b

            # Ensure consumption is valid
            if consumption <= 0:
                return -np.inf

            point = np.array([b, h_prime])
            Ev_prime = eval_linear(UGgrid_all, V[i_z, :], point, xto.LINEAR)

            return u(consumption, h_prime, 0) + beta * Ev_prime
        else:
            return -np.inf

    @njit
    def obj_adj_nested_b_min(x_prime, wealth, i_z, V, keeper_pol, t):
        """Negated objective for golden_section_search (minimizer)."""
        return -obj_adj_nested_b(x_prime, wealth, i_z, V, keeper_pol, t)

    @njit
    def gs_max_multisect(obj, obj_neg, a, b, args, n_sections=5, xtol=1e-10):
        """Golden section maximization over multiple sections to handle multiple optima.

        Divides [a, b] into n_sections intervals, runs golden_section_search on each,
        returns the (x, val) that gives the maximum across all sections.

        Parameters
        ----------
        obj : callable
            Objective function to maximize (not negated)
        obj_neg : callable
            Negated objective function for minimization (golden_section_search)
        a : float
            Lower bound
        b : float
            Upper bound
        args : tuple
            Additional arguments to obj and obj_neg
        n_sections : int
            Number of sections to divide interval into
        xtol : float
            Tolerance for optimization

        Returns
        -------
        best_x : float
            x value that maximizes obj
        best_val : float
            Maximum value found
        """
        # Handle edge case where interval is too small
        if b - a < xtol:
            val = obj(a, *args)
            return a, val

        # Pre-allocate arrays to store results from each section
        x_opts = np.zeros(n_sections)
        vals = np.zeros(n_sections)
        vals[:] = -1e250  # Initialize to large negative (safer than -inf for numba)

        section_width = (b - a) / n_sections

        for i in range(n_sections):
            lo = a + i * section_width
            hi = a + (i + 1) * section_width

            # Skip sections that are too narrow
            if hi - lo < xtol:
                continue

            # Run golden_section_search (minimizer) with negated objective
            x_opt = golden_section_search.optimizer(obj_neg, lo, hi, args=args, tol=xtol)
            x_opts[i] = x_opt
            vals[i] = obj(x_opt, *args)  # Evaluate original objective for max value

        # Find the section with the maximum value
        best_idx = 0
        best_val = vals[0]
        for i in range(1, n_sections):
            if vals[i] > best_val:
                best_val = vals[i]
                best_idx = i

        best_x = x_opts[best_idx]

        # Fallback if all sections returned invalid
        if best_val < -1e200:
            mid = (a + b) / 2
            best_x = mid
            best_val = obj(mid, *args)

        return best_x, best_val

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
            # Use golden_section_search (minimizer) with negated objective
            h_prime_adj_star = golden_section_search.optimizer(obj_adj_min, bnds_adj[0],
                                                               bnds_adj[1],
                                                               args=args_adj,
                                                               tol=1e-10)

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
    def root_A_liq_euler_inv_2d(a_prime, h_prime, z, Ud_prime_a, t):
        """ Gives inverse of liquid asset Euler and EGM point
                for non-adjusters (2D version for off-grid h_prime)

        Parameters
        ----------
        a_prime: float64
        h_prime: float64
                                t+1 housing value (may be off-grid)
        z: float64
                value of shock
        Ud_prime_a: 2D array
                                 discounted marginal utility of liq assets
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

        point = np.array([a_prime, h_prime])
        Ud_prime_a_val = eval_linear(UGgrid_all, Ud_prime_a, point, xto.LINEAR)

        c = du_c_inv(Ud_prime_a_val)
        egm_a = c + a_prime

        return egm_a, c

    @njit
    def root_A_liq_euler_inv(a_prime, h_prime, z, Ud_prime_a_1d, t):
        """ Gives inverse of liquid asset Euler and EGM point
                for adjusters (1D version for on-grid h_prime)

        Parameters
        ----------
        a_prime: float64
        h_prime: float64
                                t+1 housing value adjusted (on grid)
        z: float64
                value of shock
        Ud_prime_a_1d: 1D array
                                 discounted marginal utility of liq assets
                                 for given shock value today, sliced at h_prime index
        t: int
                time

        Returns
        -------
        egm_a: float64
                EGM point for a today
        c: float64
                consumption
        """

        # 1D interpolation along a dimension (h_prime is on grid)
        Ud_prime_a_val = interp_as_scalar(asset_grid_A, Ud_prime_a_1d, a_prime)

        c = du_c_inv(Ud_prime_a_val)
        egm_a = c + a_prime  # - y_func(t,z)

        return egm_a, c

    @njit
    def housing_euler_resid_(a_prime, h_prime, z, Ud_prime_a_1d, Ud_prime_h_1d, t):
        """ Euler residual for housing Euler given h_prime and a_prime.

        Optimized: Uses du_c(c) = Ud_prime_a_val directly (from liquid asset Euler)
        instead of computing c = du_c_inv(Ud_prime_a_val) then du_c(c).

        Parameters
        ----------
        a_prime: float64
        h_prime: float64
                                t+1 housing value adjusted
        z: float64
                value of shock
        Ud_prime_a_1d: 1D array
                                 discounted marginal utility of liq assets
                                 for given shock value today, sliced at h_prime index
        Ud_prime_h_1d: 1D array
                                 discounted marginal utility of housing assets
                                 for given shock value today, sliced at h_prime index
        t: int
                time

        Returns
        -------
        resid: float64
        """
        # Direct interpolation - no consumption calculation needed
        # At liquid asset Euler root: du_c(c) = Ud_prime_a_val
        Ud_prime_a_val = interp_as_scalar(asset_grid_A, Ud_prime_a_1d, a_prime)
        Ud_prime_h_val = interp_as_scalar(asset_grid_A, Ud_prime_h_1d, a_prime)

        # Housing Euler: du_c(c) * (1+tau) = Ud_prime_h_val + du_h(h_prime)
        return Ud_prime_a_val * (1 + tau) - Ud_prime_h_val - du_h(h_prime)

    @njit
    def root_H_UPRIME_func(h_prime, z, Ud_prime_a_1d, Ud_prime_h_1d, t):
        """ Function returns a_prime roots of housing Euler equation
                for adjusters given h_prime.

        Parameters
        ----------
        h_prime: float64
                                t+1 housing value adjusted (on grid)
        z: float64
                value of shock
        Ud_prime_a_1d: 1D array
                                 discounted marginal utility of liq assets
                                 for given shock value today, sliced at h_prime index
        Ud_prime_h_1d: 1D array
                                 discounted marginal utility of housing assets
                                 for given shock value today, sliced at h_prime index
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
                               h_prime, z, Ud_prime_a_1d, Ud_prime_h_1d, t)

        for j in range(len(a_prime_points)):
            # recover consumption associated with liquid asset Euler
            if a_prime_points[j] > 0:
                # 1D interpolation along a dimension (h_prime is on grid)
                Ud_prime_h_val = interp_as_scalar(asset_grid_A, Ud_prime_h_1d, a_prime_points[j])

                c = du_c_inv((Ud_prime_h_val
                              + du_h(h_prime)) / (1 + tau))

                egm_wealth = c + a_prime_points[j] + h_prime * (1 + tau)

                e_grid_points[j] = egm_wealth
            else:
                break

        if len(np.where(a_prime_points == b)[0]) == 0:
            # 1D interpolation at a=b (h_prime is on grid)
            Ud_prime_h_val = interp_as_scalar(asset_grid_A, Ud_prime_h_1d, b)

            c_at_amin = du_c_inv((Ud_prime_h_val + du_h(h_prime)) / (1 + tau))
            a_prime_points[-1] = b
            egm_wealth_min = c_at_amin + b + h_prime * (1 + tau)
            e_grid_points[-1] = egm_wealth_min

        return a_prime_points, e_grid_points

    @njit
    def root_H_UPRIME_func_fast(h_prime, z, Ud_prime_a_1d, Ud_prime_h_1d, t):
        """Fast version using piecewise linear root-finding with coarse sampling.

        Samples at root_eps intervals to avoid O(n_A) operations.
        Time complexity: O(grid_range / root_eps) instead of O(n_grid).

        Parameters
        ----------
        h_prime: float64
            t+1 housing value adjusted (on grid)
        z: float64
            value of shock
        Ud_prime_a_1d: 1D array
            discounted marginal utility of liq assets
        Ud_prime_h_1d: 1D array
            discounted marginal utility of housing assets
        t: int
            time

        Returns
        -------
        a_prime_points: 1D array
            a_prime roots (zeros are padding, not roots)
        e_grid_points: 1D array
            EGM wealth points corresponding to roots
        """
        # Compute du_h(h_prime) once
        du_h_val = du_h(h_prime)

        # Determine sample points based on root_eps
        n_A = len(asset_grid_A)
        grid_range = asset_grid_A[-1] - asset_grid_A[0]
        avg_spacing = grid_range / (n_A - 1)

        if root_eps > avg_spacing:
            # Coarse sampling: only evaluate at root_eps intervals
            step = int(root_eps / avg_spacing)
            n_samples = (n_A - 1) // step + 1

            # Build sample grid and compute residual only at samples
            sample_grid = np.empty(n_samples)
            resid = np.empty(n_samples)

            for k in range(n_samples):
                idx = min(k * step, n_A - 1)
                sample_grid[k] = asset_grid_A[idx]
                resid[k] = Ud_prime_a_1d[idx] * (1.0 + tau) - Ud_prime_h_1d[idx] - du_h_val

            # Find roots on coarse grid (no root_eps param needed - already sampled)
            a_prime_points, n_roots = find_roots_piecewise_linear(
                resid, sample_grid, EGM_N, 0.0)
        else:
            # Fine grid: compute residual at all points
            resid = Ud_prime_a_1d * (1.0 + tau) - Ud_prime_h_1d - du_h_val
            a_prime_points, n_roots = find_roots_piecewise_linear(
                resid, asset_grid_A, EGM_N, 0.0)

        # Make EGM wealth points array
        e_grid_points = np.zeros(EGM_N)

        # Recover consumption and EGM wealth for each root
        for j in range(n_roots):
            a_p = a_prime_points[j]
            if a_p > 0:
                # 1D interpolation along a dimension (h_prime is on grid)
                Ud_prime_h_val = interp_as_scalar(asset_grid_A, Ud_prime_h_1d, a_p)

                c = du_c_inv((Ud_prime_h_val + du_h_val) / (1 + tau))
                egm_wealth = c + a_p + h_prime * (1 + tau)
                e_grid_points[j] = egm_wealth

        # Add borrowing constraint point if not already a root
        has_b = False
        for j in range(n_roots):
            if a_prime_points[j] == b:
                has_b = True
                break

        if not has_b:
            # 1D interpolation at a=b (h_prime is on grid)
            Ud_prime_h_val = interp_as_scalar(asset_grid_A, Ud_prime_h_1d, b)
            c_at_amin = du_c_inv((Ud_prime_h_val + du_h_val) / (1 + tau))
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
                # Keeper: h_prime is off-grid (depreciated), use 2D interpolation
                h_prime = asset_grid_H[index_h_today] * (1 - delta)
                egm_a0, c0 = root_A_liq_euler_inv_2d(b, h_prime, z,
                                                    Ud_prime_a[index_z, :], t)
                c_at_a_zero_max = c0
                C_array = np.linspace(1e-08, max(1e-08,c0-1e-10), len(asset_grid_A))
                #print(C_array)
                point = np.array([b, h_prime])
                v_prime = beta * eval_linear(UGgrid_all,
                                                    V[index_z, :],
                                                    point, xto.LINEAR)

                for k in range(len(asset_grid_A)):
                    vf_unrefined[index_z, k, index_h_today] =  u(C_array[k], h_prime, 0) + v_prime

                    endog_grid_unrefined[index_z, k, index_h_today] = C_array[k] + b
                    c_unrefined[index_z, k, index_h_today] = C_array[k]

                for index_a_prime in range(len(asset_grid_A)):

                        index_a_db = int(len(asset_grid_A) + index_a_prime)
                        a_prime = asset_grid_AC[index_a_db]
                        h_prime = asset_grid_H[index_h_today] * (1 - delta)

                        # Keeper: h_prime is off-grid (depreciated), use 2D interpolation
                        egm_a, c = root_A_liq_euler_inv_2d(a_prime, h_prime, z,
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
                                                    point, xto.LINEAR)

                        vf_unrefined[index_z, index_a_db, index_h_today]\
                            = u(c, h_prime, 0) + v_prime

        return endog_grid_unrefined, vf_unrefined, c_unrefined

    @njit
    def _refineKeeper(endog_grid_unrefined,
                      vf_unrefined, c_unrefined, V_prime, t, m_bar=1.1, return_grids=0):
        """
        Refine EGM grid for non-adjusters using FUES (JIT-compiled).

        If return_grids=1, also returns intermediate grids for plotting.
        """
        # Cache dimensions
        n_z = len(z_vals)
        n_h = len(asset_grid_H)
        n_a = len(asset_grid_A)

        # empty refined grids conditioned on time t housing
        new_a_prime_refined = np.ones((n_z, n_a, n_h))
        new_c_refined = np.ones((n_z, n_a, n_h))
        new_v_refined = np.ones((n_z, n_a, n_h))

        # keep today's housing fixed
        for index_h_today in range(n_h):
            for index_z in range(n_z):
                
                vf_unrefined_points = vf_unrefined[index_z, :, index_h_today]
                endog_grid_unrefined_points = endog_grid_unrefined[index_z, :, index_h_today]
                c_unrefined_points = c_unrefined[index_z, :, index_h_today]
                
                # Remove duplicates by taking max value
                uniqueIds = uniqueEG(endog_grid_unrefined_points, vf_unrefined_points)
                endog_grid_unrefined_points = endog_grid_unrefined_points[uniqueIds]
                vf_unrefined_points = vf_unrefined_points[uniqueIds]
                c_unrefined_points = c_unrefined_points[uniqueIds]
                asset_grid_AC_unique = asset_grid_AC[uniqueIds]
                
                # Sort inputs by e_grid (required by FUES_jit_core)
                sortidx = np.argsort(endog_grid_unrefined_points)
                e_sorted = endog_grid_unrefined_points[sortidx]
                v_sorted = vf_unrefined_points[sortidx]
                c_sorted = c_unrefined_points[sortidx]
                a_sorted = asset_grid_AC_unique[sortidx]

                _FUES_POST_STATE = True
                _FUES_POST_CLEAN = False
                _FUES_INCLUDE_INTER = False
                _FUES_DISABLE_CHECKS = True
                _FUES_EPS_FWD = 0.05
                
                # Call FUES_jit_core with all parameters (post_clean=0 for keeper)
                keep, intersections, n_inter, final_mask, n_final = FUES_jit_core(
                    e_sorted, v_sorted, c_sorted, a_sorted, a_sorted,  # e, v, p1, p2, del_a
                    m_bar, 10,  # m_bar, LB
                    _FUES_ENDOG_MBAR, _FUES_INCLUDE_INTER, _FUES_NO_DOUBLE,
                    _FUES_SINGLE_INTER, _FUES_DISABLE_CHECKS, _FUES_LEFT_STRICT,
                    _FUES_POST_STATE, _FUES_DETECT_DEC, 0,  # post_clean=0
                    _FUES_PAD_MBAR, _FUES_JUMP_TOL,
                    _FUES_EPS_D, _FUES_EPS_SEP, _FUES_EPS_FWD, _FUES_PAR_GUARD
                )

                # Pre-allocate and copy directly (faster than concatenate)
                n_kept = np.sum(keep)
                n_total = n_kept + n_inter
                e_grid_clean = np.empty(n_total)
                vf_clean = np.empty(n_total)
                c_clean = np.empty(n_total)
                a_prime_clean = np.empty(n_total)

                # Copy kept points
                idx = 0
                for i in range(len(keep)):
                    if keep[i]:
                        e_grid_clean[idx] = e_sorted[i]
                        vf_clean[idx] = v_sorted[i]
                        c_clean[idx] = c_sorted[i]
                        a_prime_clean[idx] = a_sorted[i]
                        idx += 1

                # Copy intersections directly
                for i in range(n_inter):
                    e_grid_clean[idx] = intersections[i, 0]
                    vf_clean[idx] = intersections[i, 1]
                    c_clean[idx] = intersections[i, 2]
                    a_prime_clean[idx] = intersections[i, 3]
                    idx += 1

                # Sort if we have intersections
                if n_inter > 0:
                    sortindex = np.argsort(e_grid_clean)
                    e_grid_clean = e_grid_clean[sortindex]
                    vf_clean = vf_clean[sortindex]
                    c_clean = c_clean[sortindex]
                    a_prime_clean = a_prime_clean[sortindex]
                
                # Interpolate to output grid
                new_a_prime_refined[index_z, :, index_h_today] = interp_as(
                    e_grid_clean, a_prime_clean, asset_grid_A, extrap=True)
                new_c_refined[index_z, :, index_h_today] = interp_as(
                    e_grid_clean, c_clean, asset_grid_A, extrap=True)
                new_v_refined[index_z, :, index_h_today] = interp_as(
                    e_grid_clean, vf_clean, asset_grid_A, extrap=True)

                # Clamp NaN/inf after interpolation (allow off-grid)
                new_a_prime_refined[index_z, :, index_h_today] = clamp_policy(
                    new_a_prime_refined[index_z, :, index_h_today], b, grid_max_A*2)
                new_c_refined[index_z, :, index_h_today] = clamp_policy(
                    new_c_refined[index_z, :, index_h_today], 1e-10, 1e10)
                new_v_refined[index_z, :, index_h_today] = clamp_value(
                    new_v_refined[index_z, :, index_h_today])

                # Enforce borrowing constraint
                #for i in range(n_a):
                #    if new_a_prime_refined[index_z, i, index_h_today] < b:
                #        new_a_prime_refined[index_z, i, index_h_today] = b
                
                # remove jumps - uses consumption to detect jumps, corrects v and a
                # For keeper, no durable policy to correct, so pass dummy (v_arr used as placeholder)
                #(new_c_refined[index_z, :, index_h_today],
                # new_v_refined[index_z, :, index_h_today],
                # _,  # dummy d output (not used for keeper)
                # new_a_prime_refined[index_z, :, index_h_today]) = correct_jumps1d_arr(
                #    new_c_refined[index_z, :, index_h_today],
                #    asset_grid_A,
                #    m_bar,
                #    new_v_refined[index_z, :, index_h_today],
                #    new_v_refined[index_z, :, index_h_today].copy(),  # dummy for d_arr
                #    new_a_prime_refined[index_z, :, index_h_today]
                #)

        # Return dummy arrays when plotting not needed
        if return_grids == 0:
            dummy = np.empty(1)
            return new_a_prime_refined, new_c_refined, new_v_refined, dummy, dummy, dummy, dummy
        else:
            return new_a_prime_refined, new_c_refined, new_v_refined, e_grid_clean, vf_clean, c_clean, a_prime_clean

    @njit
    def _adjEGM(Ud_prime_a, Ud_prime_h, V, t):
        # Cache dimensions
        n_z = len(z_vals)
        n_he = len(asset_grid_HE)
        n_egm = EGM_N
        tau_adj = 1.0 + tau  # Pre-compute

        # Use zeros - need 0 for invalid points
        endog_grid_unrefined = np.zeros((n_z, n_he, n_egm))
        vf_unrefined = np.zeros((n_z, n_he, n_egm))
        a_prime_unrefined = np.zeros((n_z, n_he, n_egm))
        h_prime_unrefined = np.zeros((n_z, n_he, n_egm))

        for index_h_prime in range(n_he):
            h_prime = asset_grid_HE[index_h_prime]
            h_adj = h_prime * tau_adj  # Pre-compute for inner loop

            for index_z in range(n_z):
                z = z_vals[index_z]
                # Slice 2D arrays to 1D along h dimension (h_prime is on grid)
                Ud_a_z_1d = Ud_prime_a[index_z, :, index_h_prime]
                Ud_h_z_1d = Ud_prime_h[index_z, :, index_h_prime]
                V_z_1d = V[index_z, :, index_h_prime]

                # ============================================================
                # SWITCH: Use root_H_UPRIME_func_fast for O(n) piecewise linear
                #         Use root_H_UPRIME_func for iterative brentq (slower)
                # ============================================================
                a_primes, e_grid_points = root_H_UPRIME_func_fast(h_prime, z,
                                                                   Ud_a_z_1d, Ud_h_z_1d, t)

                endog_grid_unrefined[index_z, index_h_prime] = e_grid_points
                a_prime_unrefined[index_z, index_h_prime] = a_primes

                for i in range(len(a_primes)):
                    a_p = a_primes[i]
                    if a_p > 0.0:
                        c_val = e_grid_points[i] - h_adj - a_p
                        # 1D interpolation along a dimension (h_prime is on grid)
                        v_prime = beta * interp_as_scalar(asset_grid_A, V_z_1d, a_p)
                        vf_unrefined[index_z, index_h_prime, i] = u(c_val, h_prime, chi) + v_prime
                        h_prime_unrefined[index_z, index_h_prime, i] = h_prime

        return endog_grid_unrefined, vf_unrefined, a_prime_unrefined, h_prime_unrefined

    @njit
    def refine_adj(endog_grid_unrefined,
                   vf_unrefined,
                   a_prime_unrefined,
                   h_prime_unrefined, m_bar=1.4, return_grids=0):
        """
        Refine adjuster EGM grids using FUES (JIT-compiled).
        Input arrays are 3D: (n_z, n_he, n_egm) - ravel 2nd/3rd dims.
        Returns function on *wealth*.

        If return_grids=1, also returns intermediate grids for plotting.
        """
        n_z = len(z_vals)
        n_we = len(asset_grid_WE)
        tau_adj = 1.0 + tau  # Pre-compute
        grid_we = asset_grid_WE  # Local ref avoids closure lookup

        # Use empty instead of ones - values overwritten
        new_a_prime_refined = np.empty((n_z, n_we))
        new_h_prime_refined = np.empty((n_z, n_we))
        new_v_refined = np.empty((n_z, n_we))
        new_c_refined = np.empty((n_z, n_we))


        for index_z in range(n_z):
            # Ravel the 2D slice (n_he x n_egm) -> 1D for this z
            a_prime_unrefined_ur = a_prime_unrefined[index_z].ravel()
            
            # Boolean mask for valid points
            mask = a_prime_unrefined_ur > 0.0
            
            vf_unrefined_points = vf_unrefined[index_z].ravel()[mask]
            hprime_unrefined_points = h_prime_unrefined[index_z].ravel()[mask]
            aprime_unrefined_points = a_prime_unrefined_ur[mask]
            egrid_unref_points = endog_grid_unrefined[index_z].ravel()[mask]

            c_unrefined_points = egrid_unref_points - hprime_unrefined_points * tau_adj - aprime_unrefined_points
            
            # Remove duplicates of EGM points by taking the max 
            uniqueIds = uniqueEG(egrid_unref_points, vf_unrefined_points)
            
            egrid_unref_points = egrid_unref_points[uniqueIds]
            vf_unrefined_points = vf_unrefined_points[uniqueIds]
            c_unrefined_points = c_unrefined_points[uniqueIds]
            aprime_unrefined_points = aprime_unrefined_points[uniqueIds]
            hprime_unrefined_points = hprime_unrefined_points[uniqueIds]

            # Sort inputs by e_grid (required by FUES_jit_core)
            sortidx = np.argsort(egrid_unref_points)
            e_sorted = egrid_unref_points[sortidx]
            v_sorted = vf_unrefined_points[sortidx]
            a_sorted = aprime_unrefined_points[sortidx]
            h_sorted = hprime_unrefined_points[sortidx]
            c_sorted = c_unrefined_points[sortidx]
            _FUES_POST_STATE = True
            _FUES_POST_CLEAN = False
            _FUES_INCLUDE_INTER = False
            _FUES_DISABLE_CHECKS = True
            _FUES_EPS_FWD = 0.05
            # Call FUES_jit_core with all parameters
            keep, intersections, n_inter, final_mask, n_final = FUES_jit_core(
                e_sorted, v_sorted, h_sorted, a_sorted, c_sorted,  # e, v, p1, p2, del_a
                m_bar, 5,  # m_bar, LB
                0, _FUES_INCLUDE_INTER, _FUES_NO_DOUBLE,  # endog_mbar=False
                _FUES_SINGLE_INTER, _FUES_DISABLE_CHECKS, _FUES_LEFT_STRICT,
                _FUES_POST_STATE, _FUES_DETECT_DEC, _FUES_POST_CLEAN,
                _FUES_PAD_MBAR, _FUES_JUMP_TOL,
                _FUES_EPS_D, _FUES_EPS_SEP, _FUES_EPS_FWD, _FUES_PAR_GUARD
            )

            # Pre-allocate and copy directly (faster than concatenate)
            n_kept = np.sum(keep)
            n_total = n_kept + n_inter
            e_grid_clean = np.empty(n_total)
            vf_clean = np.empty(n_total)
            a_prime_clean = np.empty(n_total)
            hprime_clean = np.empty(n_total)

            # Copy kept points
            idx = 0
            for i in range(len(keep)):
                if keep[i]:
                    e_grid_clean[idx] = e_sorted[i]
                    vf_clean[idx] = v_sorted[i]
                    a_prime_clean[idx] = a_sorted[i]
                    hprime_clean[idx] = h_sorted[i]
                    idx += 1

            # Copy intersections directly
            for i in range(n_inter):
                e_grid_clean[idx] = intersections[i, 0]
                vf_clean[idx] = intersections[i, 1]
                a_prime_clean[idx] = intersections[i, 2]
                hprime_clean[idx] = intersections[i, 3]
                idx += 1

            # Sort if we have intersections
            if n_inter > 0:
                sortindex = np.argsort(e_grid_clean)
                e_grid_clean = e_grid_clean[sortindex]
                vf_clean = vf_clean[sortindex]
                a_prime_clean = a_prime_clean[sortindex]
                hprime_clean = hprime_clean[sortindex]

            # Recompute c_clean from values
            c_clean = e_grid_clean - hprime_clean * tau_adj - a_prime_clean
            
            # Interpolate to output grid
            new_a_prime_refined[index_z] = interp_as(e_grid_clean, a_prime_clean, grid_we, extrap= True)
            new_h_prime_refined[index_z] = interp_as(e_grid_clean, hprime_clean, grid_we, extrap= True)
            new_v_refined[index_z] = interp_as(e_grid_clean, vf_clean, grid_we, extrap= True)
            new_c_refined[index_z] = interp_as(e_grid_clean, c_clean, grid_we, extrap= True)

            # Clamp NaN/inf after interpolation
            new_a_prime_refined[index_z] = clamp_policy(new_a_prime_refined[index_z], b, grid_max_A*2)
            new_h_prime_refined[index_z] = clamp_policy(new_h_prime_refined[index_z], b, grid_max_H*2)
            new_v_refined[index_z] = clamp_value(new_v_refined[index_z])
            new_c_refined[index_z] = clamp_policy(new_c_refined[index_z], 1e-10, 1e10)

            # Post-clean: correct double jumps in interpolated policies
            # Uses consumption as primary array to detect jumps, corrects v, h_prime, a_prime
            (new_c_refined[index_z], new_v_refined[index_z],
             new_h_prime_refined[index_z], new_a_prime_refined[index_z]) = correct_jumps1d_arr(
                new_c_refined[index_z], grid_we, m_bar,
                new_v_refined[index_z], new_h_prime_refined[index_z], new_a_prime_refined[index_z]
            )

        # Return dummy arrays when plotting not needed (saves nothing on compute, but clearer API)
        if return_grids == 0:
            dummy = np.empty(1)
            return (new_a_prime_refined, new_c_refined, new_h_prime_refined, new_v_refined,
                    dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy)
        else:
            return (new_a_prime_refined, new_c_refined, new_h_prime_refined, new_v_refined,
                    e_grid_clean, vf_clean, hprime_clean, a_prime_clean,
                    vf_unrefined_points, hprime_unrefined_points,
                    aprime_unrefined_points, egrid_unref_points)

    @njit
    def _discrete_choice_loop(t, V_prime, ELambdaHnxt,
                              Akeeper, Ckeeper, Vkeeper,
                              Aadj, Cadj, Hadj,
                              Vcurr, Anxt, Hnxt, Cnxt, Dnxt,
                              LambdaAcurr, LambdaHcurr,
                              ELambdaH_HD_nxt, LambdaH_HD_curr,
                              c_from_budget=0):
        """JIT-compiled discrete choice evaluation over all states.

        c_from_budget: if 1, compute c_adj from budget constraint (NEGM style)
                       if 0, interpolate Cadj (EGM style)

        ELambdaH_HD_nxt: HD Lambda from next period (n_z, n_a_hd, n_h_hd)
                         Used for more accurate Phi_t evaluation
        LambdaH_HD_curr: Output HD Lambda for current period (n_z, n_a_hd, n_h_hd)
        """
        tau_adj = 1.0 + tau
        n_z = len(z_vals)

        # First pass: compute policies on normal grid
        for state in range(len(X_all)):
            a = asset_grid_A[X_all[state][1]]
            h = asset_grid_H[X_all[state][2]]
            i_a = X_all[state][1]
            i_h = X_all[state][2]
            i_z = int(X_all[state][0])
            z = z_vals[i_z]

            # non-adjusters
            wealth_nadj = R * a + y_func(t, z)
            v_nadj_val = interp_as_scalar(asset_grid_A, Vkeeper[i_z, :, i_h], wealth_nadj)
            c_nadj_val = interp_as_scalar(asset_grid_A, Ckeeper[i_z, :, i_h], wealth_nadj)
            a_prime_nadj_val = interp_as_scalar(asset_grid_A, Akeeper[i_z, :, i_h], wealth_nadj)
            h_prime_nadj_val = (1 - delta) * h

            # Clamp NaN/inf after keeper interpolations (allow off-grid)
            v_nadj_val = clamp_scalar(v_nadj_val, -1e10, 1e10, -1e10)
            c_nadj_val = clamp_scalar(c_nadj_val, 1e-10, 1e10, 1e-10)
            a_prime_nadj_val = clamp_scalar(a_prime_nadj_val, b, grid_max_A*2, b)

            # adjusters
            wealth = R * a + R_H * h * (1 - delta) + y_func(t, z)
            a_adj_val = interp_as_scalar(asset_grid_WE, Aadj[i_z, :], wealth)
            h_adj_val = interp_as_scalar(asset_grid_WE, Hadj[i_z, :], wealth)

            # Clamp NaN/inf after adjuster interpolations
            a_adj_val = clamp_scalar(a_adj_val, b, grid_max_A*2, b)
            h_adj_val = clamp_scalar(h_adj_val, b, grid_max_H*2, b)

            if c_from_budget == 1:
                # NEGM: compute from budget constraint
                c_adj_val = wealth - a_adj_val - h_adj_val * tau_adj
            else:
                # EGM: interpolate
                c_adj_val = interp_as_scalar(asset_grid_WE, Cadj[i_z, :], wealth)
            c_adj_val = clamp_scalar(c_adj_val, 1e-10, 1e10, 1e-10)

            points_adj = np.array([a_adj_val, h_adj_val])
            v_adj_val = u(c_adj_val, h_adj_val, chi) \
                + beta * eval_linear(UGgrid_all, V_prime[i_z], points_adj, xto.LINEAR)

            if v_adj_val >= v_nadj_val:
                d_adj = 1
            else:
                d_adj = 0

            Dnxt[i_z, i_a, i_h] = d_adj
            point_nadj = np.array([a_prime_nadj_val, h_prime_nadj_val])
            Hnxt[i_z, i_a, i_h] = d_adj * h_adj_val + (1 - d_adj) * h_prime_nadj_val
            Anxt[i_z, i_a, i_h] = d_adj * a_adj_val + (1 - d_adj) * a_prime_nadj_val
            Cnxt[i_z, i_a, i_h] = d_adj * c_adj_val + (1 - d_adj) * c_nadj_val
            Vcurr[i_z, i_a, i_h] = d_adj * v_adj_val + (1 - d_adj) * v_nadj_val
            LambdaAcurr[i_z, i_a, i_h] = beta * R * d_adj * du_c(c_adj_val) \
                + (1 - d_adj) * beta * R * du_c(c_nadj_val)

            # Use HD Lambda for Phi_t if available, otherwise use normal grid
            if use_hd_lambda:
                Phi_t = du_h(h_prime_nadj_val) \
                    + eval_linear(UGgrid_HD, ELambdaH_HD_nxt[i_z], point_nadj, xto.LINEAR)
            else:
                Phi_t = du_h(h_prime_nadj_val) \
                    + eval_linear(UGgrid_all, ELambdaHnxt[i_z], point_nadj, xto.LINEAR)

            LambdaHcurr[i_z, i_a, i_h] = beta * R_H * (1 - delta) \
                * (d_adj * du_c(c_adj_val) + (1 - d_adj) * Phi_t)

        # Second pass: compute Lambda_H on HD grid (if using HD)
        if use_hd_lambda:
            for i_z in range(n_z):
                for i_a_hd in range(n_a_hd):
                    a_hd = asset_grid_A_HD[i_a_hd]
                    for i_h_hd in range(n_h_hd):
                        h_hd = asset_grid_H_HD[i_h_hd]

                        # Interpolate discrete choice from normal grid
                        point_hd = np.array([a_hd, h_hd])
                        d_adj_hd = eval_linear(UGgrid_all, Dnxt[i_z], point_hd, xto.LINEAR)
                        d_adj_hd = int(min(max(round(d_adj_hd), 0), 1))

                        if d_adj_hd == 1:
                            # Adjuster: Lambda = beta * R_H * (1-delta) * u'_c(c)
                            c_hd = eval_linear(UGgrid_all, Cnxt[i_z], point_hd, xto.LINEAR)
                            c_hd = max(c_hd, 1e-10)
                            LambdaH_HD_curr[i_z, i_a_hd, i_h_hd] = beta * R_H * (1 - delta) * du_c(c_hd)
                        else:
                            # Keeper: Lambda = beta * R_H * (1-delta) * (u'_h(h') + E[Lambda'])
                            a_prime_hd = eval_linear(UGgrid_all, Anxt[i_z], point_hd, xto.LINEAR)
                            a_prime_hd = max(a_prime_hd, b)
                            h_prime_hd = (1 - delta) * h_hd

                            point_next_hd = np.array([a_prime_hd, h_prime_hd])
                            E_lambda_h_next = eval_linear(UGgrid_HD, ELambdaH_HD_nxt[i_z], point_next_hd, xto.LINEAR)
                            if np.isnan(E_lambda_h_next) or np.isinf(E_lambda_h_next):
                                E_lambda_h_next = 0.0

                            Phi_t_hd = du_h(h_prime_hd) + E_lambda_h_next
                            LambdaH_HD_curr[i_z, i_a_hd, i_h_hd] = beta * R_H * (1 - delta) * Phi_t_hd

        return Vcurr, Anxt, Hnxt, Cnxt, Dnxt, LambdaAcurr, LambdaHcurr, LambdaH_HD_curr

    #@njit
    def iterEGM(t, V_prime,
                         ELambdaAnxt,
                         ELambdaHnxt, ELambdaH_HD_nxt=None, m_bar=1.4, plot_age=-1, verbose_timing=True):
        """
        Iterates on the Coleman operator

        Note: V_prime,Ud_prime_a, Ud_prime_h assumed
                        to be conditioned on time t shock

                 - the t+1 marginal utilities are not multiplied
                   by the discount factor and rate of return

        If plot_age == t, returns intermediate grids for plotting.
        If verbose_timing=True, prints timing for each step.

        ELambdaH_HD_nxt: HD Lambda from next period (n_z, n_a_hd, n_h_hd) or None
                         Used for more accurate Phi_t evaluation in discrete choice
        """
        # return_grids is set from cp.return_grids (via FUES_RETURN_GRIDS env var)
        # When False, intermediate grids are not stored (faster, less memory)

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

        # HD Lambda for current period (only allocated if using HD)
        if use_hd_lambda:
            LambdaH_HD_curr = np.zeros((len(z_vals), n_a_hd, n_h_hd))
            if ELambdaH_HD_nxt is None:
                # Initialize with zeros for terminal period
                ELambdaH_HD_nxt = np.zeros((len(z_vals), n_a_hd, n_h_hd))
        else:
            LambdaH_HD_curr = np.zeros((1, 1, 1))  # Dummy
            ELambdaH_HD_nxt = np.zeros((1, 1, 1))  # Dummy

        t0 = time.perf_counter()
        endog_grid_unrefined_noadj, vf_unrefined_noadj, c_unrefined_noadj\
            = _keeperEGM(ELambdaAnxt, V_prime, t)
        t_keeperEGM = time.perf_counter() - t0

        t0 = time.perf_counter()
        Akeeper, Ckeeper, Vkeeper, e_grid_clean, vf_clean,\
            c_clean, a_prime_clean\
            = _refineKeeper(endog_grid_unrefined_noadj,
                            vf_unrefined_noadj,
                            c_unrefined_noadj,
                            V_prime, t, m_bar=m_bar, return_grids=return_grids)
        t_refineKeeper = time.perf_counter() - t0

        t0 = time.perf_counter()
        endog_grid_unrefined_adj, vf_unrefined_adj, a_prime_unrefined_adj,\
            h_prime_unrefined_adj\
            = _adjEGM(ELambdaAnxt, ELambdaHnxt, V_prime, t)
        t_adjEGM = time.perf_counter() - t0

        t0 = time.perf_counter()
        (Aadj, Cadj, Hadj, Vadj,
         e_grid_clean, vf_clean, hprime_clean, a_prime_clean,
         vf_unrefined_adj_1, h_prime_unrefined_adj_1, a_prime_unrefined_adj_1,
         endog_grid_unrefined_adj_1) = refine_adj(
            endog_grid_unrefined_adj, vf_unrefined_adj,
            a_prime_unrefined_adj, h_prime_unrefined_adj, m_bar=m_bar, return_grids=return_grids)
        t_refine_adj = time.perf_counter() - t0

        if verbose_timing:
            print(f"  [t={t}] _keeperEGM: {t_keeperEGM*1000:.1f}ms, "
                  f"_refineKeeper: {t_refineKeeper*1000:.1f}ms, "
                  f"_adjEGM: {t_adjEGM*1000:.1f}ms, "
                  f"refine_adj: {t_refine_adj*1000:.1f}ms")
        
        AdjGrids = {}
        AdjGrids["endog_grid_unrefined_adj"] = endog_grid_unrefined_adj_1
        AdjGrids["vf_unrefined_adj"] = vf_unrefined_adj_1
        AdjGrids["a_prime_unrefined_adj"] = a_prime_unrefined_adj_1
        AdjGrids["h_prime_unrefined_adj"] = h_prime_unrefined_adj_1
        AdjGrids["e_grid_clean"] = e_grid_clean
        AdjGrids["vf_clean"] = vf_clean
        AdjGrids["hprime_clean"] = hprime_clean
        AdjGrids["a_prime_clean"] = a_prime_clean
        AdjGrids["sigma_intersect"] = None  # Not computed in FUES-only mode
        AdjGrids["M_intersect"] = None  # Not computed in FUES-only mode
        AdjPol = {}
        AdjPol["A"] = Aadj
        AdjPol["H"] = Hadj
        AdjPol["V"] = Vadj
        AdjPol["C"] = Cadj
        KeeperPol = {}
        KeeperPol["A"] = Akeeper
        KeeperPol["C"] = Ckeeper
        KeeperPol["V"] = Vkeeper

        # Evaluate discrete choice on the pre-state (JIT-compiled)
        t0 = time.perf_counter()
        Vcurr, Anxt, Hnxt, Cnxt, Dnxt, LambdaAcurr, LambdaHcurr, LambdaH_HD_curr = \
            _discrete_choice_loop(t, V_prime, ELambdaHnxt,
                                  Akeeper, Ckeeper, Vkeeper,
                                  Aadj, Cadj, Hadj,
                                  Vcurr, Anxt, Hnxt, Cnxt, Dnxt,
                                  LambdaAcurr, LambdaHcurr,
                                  ELambdaH_HD_nxt, LambdaH_HD_curr,
                                  c_from_budget=1)
        t_discrete = time.perf_counter() - t0

        # Compute aggregate timing
        t_keeper = t_keeperEGM + t_refineKeeper
        t_adj = t_adjEGM + t_refine_adj
        t_total = t_keeper + t_adj + t_discrete

        if verbose_timing:
            print(f"  [t={t}] discrete_choice: {t_discrete*1000:.1f}ms, "
                  f"TOTAL: {t_total*1000:.1f}ms")

        # Build timing dict
        timing = {
            'keeper_ms': t_keeper * 1000,
            'adj_ms': t_adj * 1000,
            'discrete_ms': t_discrete * 1000,
            'total_ms': t_total * 1000
        }

        return Vcurr, Hnxt, Cnxt, Dnxt, LambdaAcurr, LambdaHcurr, LambdaH_HD_curr, AdjPol, KeeperPol, AdjGrids, timing
    
    
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
                # Cap upper bound at grid_max_H to avoid extrapolation
                bnds_adj_lo = b
                bnds_adj_hi = min(liqAct/(1+tau)+b, grid_max_H)
                args_adj = (liqAct, i_z, EVnxt, Ckeeper[i_z], t)

                # Maximize over h_prime using multi-section golden section (handles multiple optima)
                h_prime_adj_star1, v_prime_adj_star = gs_max_multisect(
                    obj_adj_nested, obj_adj_nested_min, bnds_adj_lo, bnds_adj_hi,
                    args=args_adj, n_sections=1, xtol=1e-06)

                # calculate max h_prime for a_prime = b
                h_prime_a_bound, v_prime_a_bound = gs_max_multisect(
                    obj_adj_nested_b, obj_adj_nested_b_min, bnds_adj_lo, bnds_adj_hi,
                    args=args_adj, n_sections=1, xtol=1e-06)
                
                # Wealth after housing expenditure                                       
                left_over_wealth = liqAct - h_prime_adj_star1* (1 + tau)
                
                a_prime_adj_star1 = eval_linear(UGgrid_all,\
                                                Akeeper[i_z],\
                                                np.array([left_over_wealth,\
                                                        h_prime_adj_star1]),\
                                                xto.LINEAR)
                
                abound_flag  = v_prime_adj_star<v_prime_a_bound
                h_prime_adj_star  = abound_flag*h_prime_a_bound\
                                     + (1-abound_flag)*h_prime_adj_star1
                a_prime_adj_star = abound_flag*b\
                                     + (1-abound_flag)*a_prime_adj_star1
                v_adj[i_z, i_a] = (1-abound_flag)*v_prime_adj_star\
                                     + abound_flag*v_prime_a_bound
                
                # Clamp to grid bounds (consistent with EGM)
                h_prime_adj_star = min(max(h_prime_adj_star, b), grid_max_H*2)
                a_prime_adj_star = min(max(a_prime_adj_star, b), grid_max_A*2)

                h_prime_adj[i_z, i_a] = h_prime_adj_star
                a_prime_adj[i_z, i_a] = a_prime_adj_star
                c_adj[i_z, i_a] = liqAct - h_prime_adj_star*(1 + tau) - a_prime_adj_star

        return a_prime_adj,c_adj, h_prime_adj, v_adj

    
    
    def iterNEGM(EVnxt, LambdaAnxt, ELambdaHnxt, t, ELambdaH_HD_nxt=None, verbose_timing=True):
        """
        Solve iteration of using the nested EGM algorithm
        or hypbrid EGM-VFI algorithm

        Parameters
        ----------
        EVnxt: 3D array
                t+1 Expected Value function undiscounted
        ELambdaH_HD_nxt: HD Lambda from next period (n_z, n_a_hd, n_h_hd) or None
                         Used for more accurate Phi_t evaluation in discrete choice
        verbose_timing: bool
                If True, print timing for each step
        """

        Anxt = np.empty(EVnxt.shape)
        Hnxt = np.empty(EVnxt.shape)
        Cnxt = np.empty(EVnxt.shape)
        Dnxt = np.empty(EVnxt.shape)
        Vcurr = np.empty(EVnxt.shape)
        LambdaAcurr = np.empty(EVnxt.shape)
        LambdaHcurr = np.empty(EVnxt.shape)

        # HD Lambda for current period (only allocated if using HD)
        if use_hd_lambda:
            LambdaH_HD_curr = np.zeros((len(z_vals), n_a_hd, n_h_hd))
            if ELambdaH_HD_nxt is None:
                # Initialize with zeros for terminal period
                ELambdaH_HD_nxt = np.zeros((len(z_vals), n_a_hd, n_h_hd))
        else:
            LambdaH_HD_curr = np.zeros((1, 1, 1))  # Dummy
            ELambdaH_HD_nxt = np.zeros((1, 1, 1))  # Dummy

        # Eval the keeper policy using EGM
        t0 = time.perf_counter()
        endog_grid_unrefined_noadj, vf_unrefined_noadj, c_unrefined_noadj\
            = _keeperEGM(LambdaAnxt, EVnxt, t)
        t_keeperEGM = time.perf_counter() - t0

        t0 = time.perf_counter()
        Akeeper, Ckeeper, Vkeeper, _, _, _, _ = _refineKeeper(
            endog_grid_unrefined_noadj,
            vf_unrefined_noadj,
            c_unrefined_noadj,
            EVnxt, t, m_bar=cp.m_bar)
        t_refineKeeper = time.perf_counter() - t0

        # VFI part for each level of liquid wealth after wages are realised
        # and housing has been liquidated
        t0 = time.perf_counter()
        Aadj, Cadj, Hadj, Vadj = _solveAdjNEGM(EVnxt, Ckeeper, Akeeper, t)
        t_solveAdjNEGM = time.perf_counter() - t0

        if verbose_timing:
            print(f"  [t={t}] _keeperEGM: {t_keeperEGM*1000:.1f}ms, "
                  f"_refineKeeper: {t_refineKeeper*1000:.1f}ms, "
                  f"_solveAdjNEGM: {t_solveAdjNEGM*1000:.1f}ms")

        AdjPol = {}
        AdjPol["A"] = Aadj
        AdjPol["H"] = Hadj
        AdjPol["V"] = Vadj
        AdjPol["C"] = Cadj
        KeeperPol = {}
        KeeperPol["A"] = Akeeper
        KeeperPol["C"] = Ckeeper
        KeeperPol["V"] = Vkeeper

        # Evaluate discrete choice on the pre-state (JIT-compiled, NEGM style)
        t0 = time.perf_counter()
        Vcurr, Anxt, Hnxt, Cnxt, Dnxt, LambdaAcurr, LambdaHcurr, LambdaH_HD_curr = \
            _discrete_choice_loop(t, EVnxt, ELambdaHnxt,
                                  Akeeper, Ckeeper, Vkeeper,
                                  Aadj, Cadj, Hadj,
                                  Vcurr, Anxt, Hnxt, Cnxt, Dnxt,
                                  LambdaAcurr, LambdaHcurr,
                                  ELambdaH_HD_nxt, LambdaH_HD_curr,
                                  c_from_budget=1)
        t_discrete = time.perf_counter() - t0

        # Compute aggregate timing
        t_keeper = t_keeperEGM + t_refineKeeper
        t_adj = t_solveAdjNEGM
        t_total = t_keeper + t_adj + t_discrete

        if verbose_timing:
            print(f"  [t={t}] discrete_choice: {t_discrete*1000:.1f}ms, "
                  f"TOTAL: {t_total*1000:.1f}ms")

        # Build timing dict
        timing = {
            'keeper_ms': t_keeper * 1000,
            'adj_ms': t_adj * 1000,
            'discrete_ms': t_discrete * 1000,
            'total_ms': t_total * 1000
        }

        return Vcurr, Hnxt, Cnxt, Dnxt, LambdaAcurr, LambdaHcurr, LambdaH_HD_curr, AdjPol, KeeperPol, timing
    
    @njit
    def condition_V(Vcurr, LambdaAcurr, LambdaHcurr):
        """Condition the t+1 continuation value on time t information (JIT-compiled).

        Computes E[V|z] = sum over z' of Pi[z,z'] * V[z',a,h] for all (z,a,h).
        Optimized to avoid non-contiguous array slicing in np.dot.
        """
        n_z, n_a, n_h = Vcurr.shape
        new_V = np.zeros((n_z, n_a, n_h))
        new_UD_a = np.zeros((n_z, n_a, n_h))
        new_UD_h = np.zeros((n_z, n_a, n_h))

        # Loop over (a,h) and do matrix-vector multiply for all z at once
        # Make contiguous copies to avoid NumbaPerformanceWarning
        for i_a in range(n_a):
            for i_h in range(n_h):
                # Extract contiguous slices along z dimension
                v_slice = np.ascontiguousarray(Vcurr[:, i_a, i_h])
                a_slice = np.ascontiguousarray(LambdaAcurr[:, i_a, i_h])
                h_slice = np.ascontiguousarray(LambdaHcurr[:, i_a, i_h])

                # Pi @ slice computes all z values at once
                new_V[:, i_a, i_h] = np.dot(Pi, v_slice)
                new_UD_a[:, i_a, i_h] = np.dot(Pi, a_slice)
                new_UD_h[:, i_a, i_h] = np.dot(Pi, h_slice)

        return new_V, new_UD_a, new_UD_h

    @njit
    def condition_V_HD(LambdaH_HD_curr):
        """Condition HD Lambda on current shock state (JIT-compiled).

        Computes E[Lambda_H_HD|z] = sum over z' of Pi[z,z'] * Lambda_H_HD[z',a,h]
        for all (z,a,h) on the HD grid.
        """
        n_z, n_a_hd_loc, n_h_hd_loc = LambdaH_HD_curr.shape
        new_HD = np.zeros((n_z, n_a_hd_loc, n_h_hd_loc))

        # Loop over (a_hd, h_hd) and do matrix-vector multiply for all z at once
        for i_a in range(n_a_hd_loc):
            for i_h in range(n_h_hd_loc):
                # Extract contiguous slice along z dimension
                hd_slice = np.ascontiguousarray(LambdaH_HD_curr[:, i_a, i_h])
                # Pi @ slice computes all z values at once
                new_HD[:, i_a, i_h] = np.dot(Pi, hd_slice)

        return new_HD

    return iterVFI, iterEGM, condition_V, condition_V_HD, iterNEGM