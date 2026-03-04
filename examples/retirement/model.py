"""Iskhakov et al. (2017) retirement choice model solved
via FUES-EGM (Dobrescu and Shanker, 2022).

Three-stage branching period::

    employed-->labour_mkt_decision (branching, max)
                    |-- work   -> worker-consumption-stage   (EGM + upper envelope) -> employed
                    +-- retire -> retiree-consumption-stage (EGM, no upper envelope) -> retired
    retired -> retiree-consumption-stage (EGM, no upper envelope) -> retired

Author: Akshay Shanker, akshay.shanker@me.com.
"""

import numpy as np
import time
from numba import njit
from dcsmm.fues.helpers.math_funcs import interp_as, interp_as_2, interp_as_3
from dcsmm.uenvelope import EGM_UE as egm_ue_global


# ============================================================================
# Default equation callables (log utility)
#
# These are the standalone @njit callables that define the model's equations.
# Phase 0 extraction: previously hard-coded inside RetirementModel.__init__.
# The whisperer (Phase 2) will compile these from YAML stage declarations.
# ============================================================================

@njit(cache=True)
def _default_u(c):
    """Utility: u(c) = log(c)."""
    return np.log(c)


@njit(cache=True)
def _default_du(c):
    """Marginal utility: u'(c) = 1/c."""
    return 1.0 / c


@njit(cache=True)
def _default_uc_inv(x):
    """Inverse marginal utility: (u')^{-1}(x) = 1/x."""
    return 1.0 / x


@njit(cache=True)
def _default_ddu(c):
    """Second derivative of utility: u''(c) = -1/c^2."""
    return -1.0 / (c ** 2)


class RetirementModel:

    """Calibration and grids for the retirement choice model.

    Stores calibrated parameters, the asset grid, and
    equation callables (utility and its derivatives).

    All parameters are required — canonical values live in
    ``syntax/calibration.yaml`` and ``syntax/settings.yaml``.
    Use :meth:`from_period` or :meth:`with_test_defaults` to
    construct instances.

    Parameters
    ----------
    r : float
        Interest rate.
    beta : float
        Discount factor (not rate).
    delta : float
        Fixed utility cost of working.
    smooth_sigma : float
        Logit smoothing parameter (0 = hard max).
    y : float
        Wage income for workers.
    b : float
        Lower bound for asset grid.
    grid_max_A : float
        Upper bound for asset grid.
    grid_size : int
        Number of asset grid points.
    T : int
        Number of periods (horizon).
    m_bar : float
        FUES jump detection threshold.
    padding_mbar : float
        Padding for m_bar.
    equations : dict, optional
        Override equation callables (keys: ``u``, ``du``,
        ``uc_inv``, ``ddu``).  Defaults to log utility.

    Attributes
    ----------
    R : float
        Gross return ``1 + r``.
    asset_grid_A : ndarray
        Asset grid of size *grid_size*.
    eulerK : int
        Number of Euler equation check points.
    u, du, uc_inv, ddu : callable
        Equation callables (``@njit``).
    """

    def __init__(self,
                 r, beta, delta, smooth_sigma, y, b,
                 grid_max_A, grid_size, T, m_bar,
                 padding_mbar=0, equations=None):

        self.grid_size = int(grid_size)
        self.r, self.R = float(r), 1 + float(r)
        self.beta = float(beta)
        self.delta = float(delta)
        self.smooth_sigma = float(smooth_sigma)
        self.b = float(b)
        self.T = int(T)
        self.y = float(y)
        self.grid_max_A = float(grid_max_A)
        self.m_bar = float(m_bar)
        self.padding_mbar = float(padding_mbar)

        self.asset_grid_A = np.linspace(
            self.b, self.grid_max_A, self.grid_size,
        )
        self.eulerK = len(self.asset_grid_A)

        # Equation callables: use provided overrides or module-level defaults.
        # The whisperer (Phase 2) will compile these from YAML declarations.
        if equations is None:
            equations = {}
        self.u = equations.get('u', _default_u)
        self.du = equations.get('du', _default_du)
        self.uc_inv = equations.get('uc_inv', _default_uc_inv)
        self.ddu = equations.get('ddu', _default_ddu)

    @classmethod
    def from_period(cls, period, equations=None):
        """Construct from a dolo-plus calibrated period dict.

        Reads calibration and settings from the ``work_cons``
        stage's ``.calibration`` and ``.settings`` attributes.
        These are populated by the calibrate and configure
        functors during :func:`build_period`.

        Parameters
        ----------
        period : dict
            ``{<stage_name>: <calibrated SymbolicModel>}``.
            Must contain a ``'work_cons'`` key whose value
            has ``.calibration`` and ``.settings`` dicts.
        equations : dict, optional
            Override equation callables.

        Returns
        -------
        RetirementModel
        """
        cal = period['work_cons'].calibration
        settings = period['work_cons'].settings or {}
        # Parameters come from wherever the dolo-plus pipeline
        # put them.  Economic params land in .calibration via
        # calibrate_stage; structural params land in .settings
        # via configure_stage.  We check both.
        def _get(key, default=None):
            if key in cal:
                return cal[key]
            if key in settings:
                return settings[key]
            if default is not None:
                return default
            raise KeyError(
                f"Parameter '{key}' not found in stage "
                f"calibration or settings"
            )

        return cls(
            r=_get('r'),
            beta=_get('beta'),
            delta=_get('delta'),
            smooth_sigma=_get('smooth_sigma', 0),
            y=_get('y'),
            b=_get('b', 1e-10),
            grid_max_A=_get('grid_max_A', 500),
            grid_size=_get('grid_size', 3000),
            T=_get('T', 20),
            m_bar=_get('m_bar', 1.2),
            padding_mbar=_get('padding_mbar', 0),
            equations=equations,
        )

    @classmethod
    def with_test_defaults(cls, **overrides):
        """Construct with test defaults (for unit tests only).

        Canonical values match ``syntax/calibration.yaml``
        and ``syntax/settings.yaml``.

        Parameters
        ----------
        **overrides
            Any parameter to override from the defaults.

        Returns
        -------
        RetirementModel
        """
        defaults = dict(
            r=0.02, beta=0.98, delta=1.0, smooth_sigma=0,
            y=20, b=1e-10, grid_max_A=500, grid_size=3000,
            T=20, m_bar=1.2, padding_mbar=0,
        )
        defaults.update(overrides)
        return cls(**defaults)


def Operator_Factory(cp, equations=None):
    """Build stage operators for the retirement model.

    Returns three operators corresponding to the three
    stages in the period template:

    - ``retire_cons``: retiree EGM (no upper envelope).
    - ``work_cons``: worker EGM + upper envelope (FUES/
      DCEGM/RFC/CONSAV).
    - ``labour_mkt_decision``: branching max/logit
      aggregator over work and retire branches.

    All operators are closures over the calibrated
    parameters and equation callables.

    Parameters
    ----------
    cp : RetirementModel
        Model instance with calibrated parameters and
        grids.
    equations : dict, optional
        Override equation callables.  Keys: ``u``, ``du``,
        ``uc_inv``, ``ddu``.  Each must be ``@njit``.
        When *None*, uses the callables on *cp* (which
        default to log-utility).

    Returns
    -------
    dict
        ``{'retire_cons':  solver_retiree_stage,
           'work_cons':    solver_worker_stage,
           'labour_mkt_decision': lab_mkt_choice_stage}``
    """

    # unpack parameters from class
    beta, delta = cp.beta, cp.delta
    asset_grid_A = cp.asset_grid_A
    y = cp.y
    smooth_sigma = cp.smooth_sigma
    grid_size = cp.grid_size
    R = cp.R
    m_bar = cp.m_bar

    # Equation callables: use provided overrides or model defaults.
    if equations is None:
        equations = {}
    u = equations.get('u', cp.u)
    du = equations.get('du', cp.du)
    uc_inv = equations.get('uc_inv', cp.uc_inv)
    ddu = equations.get('ddu', cp.ddu)

    @njit
    def solver_retiree_stage(c_cntn,        # c[>]: consumption at continuation perch
                             v_cntn,
                             # V[>]: value at continuation perch
                             dlambda_cntn,
                             # dlambda[>]: second-order marginal at cntn
                             t):
        """Retiree consumption stage (EGM, no upper
        envelope).

        Solves the ``retire_cons`` stage via the EGM
        backward mover:

        1. **InvEuler** on the poststate grid:
           ``c = uc_inv(beta * R * du(c_cntn))``
        2. **cntn_to_dcsn transition** (endogenous grid):
           ``a_ret = (c + b_ret) / R``
        3. **Bellman**:
           ``V = u(c) + beta * V_cntn``
        4. **Asset derivative**:
           ``da = R * ddu(c) / (ddu(c) + beta*R*ddv_cntn)``
        5. Interpolate from endogenous to exogenous
           arrival grid.
        6. **Constrained region**: consume down to
           borrowing constraint for grid points below
           the minimum endogenous value.
        7. **MarginalBellman**:
           ``ddv = ddu(c) * (R - da)``

        Parameters
        ----------
        c_cntn : ndarray
            Consumption at the continuation perch (c[>]).
        v_cntn : ndarray
            Value at the continuation perch (V[>]).
        dlambda_cntn : ndarray
            Second-order marginal (ddv) at continuation.
        t : int
            Period index (unused; reserved for
            time-varying extensions).

        Returns
        -------
        c_arvl : ndarray
            Consumption on the arrival grid.
        v_arvl : ndarray
            Value on the arrival grid.
        da_arvl : ndarray
            Asset derivative da'/da on the arrival grid.
        dlambda_arvl : ndarray
            Second-order marginal ddv for the upstream
            twister (feeds the worker continuation chain).
        """
        c_dcsn_hat = np.zeros(grid_size)
        v_dcsn_hat = np.zeros(grid_size)
        endog_grid = np.zeros(grid_size)
        da_dcsn_hat = np.zeros(grid_size)

        for i in range(len(asset_grid_A)):
            b_ret = asset_grid_A[i]              # poststate grid point
            c_next = c_cntn[i]
            # InvEuler
            uc_cntn = beta * R * du(c_next)
            c_dcsn = uc_inv(uc_cntn)
            # cntn_to_dcsn_transition: a_ret = (c + b_ret) / R
            a_ret_hat = (c_dcsn + b_ret) / R
            endog_grid[i] = a_ret_hat
            c_dcsn_hat[i] = c_dcsn
            # Bellman
            v_dcsn_hat[i] = u(c_dcsn) + beta * v_cntn[i]
            # Asset derivative
            da_dcsn_hat[i] = R * ddu(c_dcsn) / \
                (ddu(c_dcsn) + beta * R * dlambda_cntn[i])

        # Interpolate from endogenous to exogenous arrival grid
        min_a_val = endog_grid[0]
        c_arvl, v_arvl, da_arvl = interp_as_3(
            endog_grid, c_dcsn_hat, v_dcsn_hat, da_dcsn_hat, asset_grid_A)

        # Constrained region: consume down to borrowing constraint
        constrained_idx = np.where(asset_grid_A <= min_a_val)
        c_arvl[constrained_idx] = asset_grid_A[constrained_idx]
        v_arvl[constrained_idx] = u(
            asset_grid_A[constrained_idx]) + beta * v_cntn[0]
        da_arvl[constrained_idx] = 0

        # MarginalBellman: dlambda = ddu(c) * (R - da)
        dlambda_arvl = ddu(c_arvl) * (R - da_arvl)

        return c_arvl, v_arvl, da_arvl, dlambda_arvl

    @njit
    def _invert_euler(lambda_worker_cntn, dlambda_worker_cntn, v_worker_cntn):
        """Inverse Euler step for the worker stage (pre-UE).

        Inverts the Euler equation on the poststate grid
        to recover decision-perch consumption, the
        Q-function, the endogenous wealth grid, and the
        asset derivative.  All outputs are *unrefined*
        (before upper-envelope filtering).

        Parameters
        ----------
        lambda_worker_cntn : ndarray
            Marginal value dV at continuation (dv[>]).
        dlambda_worker_cntn : ndarray
            Second-order marginal ddv at continuation.
        v_worker_cntn : ndarray
            Value V at continuation (V[>]).

        Returns
        -------
        cons_cntn_hat : ndarray
            Consumption (pre-UE) on the poststate grid.
        q_cntn_hat : ndarray
            Q-function u(c) + beta*V[>] - delta (pre-UE).
        endog_grid : ndarray
            Endogenous wealth grid w = c + a (pre-UE).
        del_a_unrefined : ndarray
            Asset derivative da'/da (pre-UE).
        """
        cons_cntn_hat = np.zeros(grid_size)
        q_cntn_hat = np.zeros(grid_size)
        endog_grid = np.zeros(grid_size)
        del_a_unrefined = np.zeros(grid_size)

        for i in range(grid_size):
            uc_cntn = beta * R * lambda_worker_cntn[i]
            c_cntn = uc_inv(uc_cntn)
            q_cntn_hat[i] = u(c_cntn) + beta * v_worker_cntn[i] - delta
            cons_cntn_hat[i] = c_cntn
            endog_grid[i] = c_cntn + asset_grid_A[i]
            del_a_unrefined[i] = R * ddu(c_cntn) / \
                (ddu(c_cntn) + beta * R * dlambda_worker_cntn[i])

        return cons_cntn_hat, q_cntn_hat, endog_grid, del_a_unrefined

    @njit
    def _approx_dcsn_state_functions(egrid1, vf_clean, sigma_clean, dela_clean,
                                     min_a_val, VF_prime_work):

        """Interpolate refined worker policies to the arrival grid.

        After upper-envelope refinement, the policies and
        value function live on the refined endogenous grid.
        This function:

        1. Computes the arrival wealth grid:
           ``w = R * a + y``.
        2. Interpolates the refined value, consumption, and
           asset derivative from the endogenous grid to the
           arrival wealth grid.
        3. Applies jump correction to smooth residual
           discontinuities from the upper envelope.
        4. Handles the constrained region (consume down
           to borrowing constraint) for wealth below the
           minimum endogenous grid point.

        Parameters
        ----------
        egrid1 : ndarray
            Refined endogenous grid (post-UE).
        vf_clean : ndarray
            Refined Q-function on *egrid1*.
        sigma_clean : ndarray
            Refined consumption on *egrid1*.
        dela_clean : ndarray
            Refined asset derivative on *egrid1*.
        min_a_val : float
            Minimum endogenous grid value (pre-UE).
        VF_prime_work : ndarray
            Continuation value V[>] (for constrained
            region fallback).

        Returns
        -------
        vf_work_t : ndarray
            Value on the arrival wealth grid.
        sigma_work_t : ndarray
            Consumption on the arrival wealth grid.
        dela_work_t : ndarray
            Asset derivative on the arrival wealth grid.
        """

        asset_grid_wealth = R * asset_grid_A + y

        vf_work_t, sigma_work_t = interp_as_2(
            egrid1, vf_clean, sigma_clean, asset_grid_wealth)
        dela_work_t = np.zeros(grid_size)

        constrained_indices = np.where(asset_grid_wealth < min_a_val)
        sigma_work_t[constrained_indices] = asset_grid_wealth[constrained_indices] - asset_grid_A[0]
        vf_work_t[constrained_indices] = u(
            asset_grid_wealth[constrained_indices]) + beta * VF_prime_work[0] - delta

        return vf_work_t, sigma_work_t, dela_work_t

    @njit
    def lab_mkt_choice_stage(
        v_cntn_work,        # V[>][work]:  value from worker cons stage
        v_cntn_ret,         # V[>][retire]: value from retiree cons stage
        c_cntn_work,        # c[>][work]:  consumption policy, work branch
        c_cntn_ret,         # c[>][retire]: consumption policy, retire branch
        da_cntn_work,       # da[>][work]: da'/dm, work branch
        da_cntn_ret,        # da[>][retire]: da'/dm, retire branch
    ):
        """Branching stage: discrete work/retire choice.

        Aggregates the two branch-keyed continuation values
        and policies into arrival-perch objects via hard max
        (``smooth_sigma=0``) or logit smoothing.

        When ``smooth_sigma=0`` (hard max), the winner's
        value, consumption, and derivatives are selected
        pointwise.  When ``smooth_sigma>0`` (logit /
        taste shocks), outputs are softmax-weighted
        mixtures.

        Parameters
        ----------
        v_cntn_work : ndarray
            Value from the worker consumption stage.
        v_cntn_ret : ndarray
            Value from the retiree consumption stage.
        c_cntn_work : ndarray
            Consumption policy, work branch.
        c_cntn_ret : ndarray
            Consumption policy, retire branch.
        da_cntn_work : ndarray
            Asset derivative da'/da, work branch.
        da_cntn_ret : ndarray
            Asset derivative da'/da, retire branch.

        Returns
        -------
        v : ndarray
            Mixed value at the arrival perch.
        c : ndarray
            Mixed consumption policy.
        lambda_arvl : ndarray
            Marginal value dV = du(c).
        dlambda_arvl : ndarray
            Second-order marginal ddV = ddu(c)*(R - da).
        """

        if smooth_sigma == 0:
            work_prob = v_cntn_work > v_cntn_ret
        else:
            exp_v_work = np.exp(v_cntn_work / smooth_sigma)
            exp_v_ret = np.exp(v_cntn_ret / smooth_sigma)
            work_prob = exp_v_work / (exp_v_ret + exp_v_work)
            work_prob = np.where(
                np.isnan(work_prob) | np.isinf(work_prob), 0, work_prob)

        c = work_prob * c_cntn_work + (1 - work_prob) * c_cntn_ret
        c[np.where(c < 0.0001)] = 0.0001

        v = work_prob * v_cntn_work + (1 - work_prob) * v_cntn_ret
        da = work_prob * da_cntn_work + (1 - work_prob) * da_cntn_ret

        lambda_arvl = du(c)
        dlambda_arvl = ddu(c) * (R - da)

        return v, c, lambda_arvl, dlambda_arvl

    def solver_worker_stage(lambda_worker_cntn,  # lambda_worker[>]
                            dlambda_worker_cntn,  # dlambda_worker[>]
                            v_worker_cntn,  # v_worker[>]
                            method='FUES'):
        """Worker consumption stage (EGM + upper envelope).

        Solves the ``work_cons`` stage in three steps:

        1. **Invert Euler** (``_invert_euler``): recover
           unrefined consumption, Q-function, endogenous
           grid, and asset derivative on the poststate grid.
        2. **Upper envelope** (``egm_ue_global``): apply
           FUES / DCEGM / RFC / CONSAV to resolve
           non-convexities from the fixed cost of working.
        3. **Interpolate to arrival grid**
           (``_approx_dcsn_state_functions``): map refined
           policies to the arrival wealth grid and handle
           the constrained region.

        Parameters
        ----------
        lambda_worker_cntn : ndarray
            Marginal value dV at continuation (dv[>]).
        dlambda_worker_cntn : ndarray
            Second-order marginal ddv at continuation.
        v_worker_cntn : ndarray
            Value V at continuation (V[>]).
        method : str
            Upper-envelope method: ``FUES``, ``DCEGM``,
            ``RFC``, or ``CONSAV``.

        Returns
        -------
        v_worker_arvl : ndarray
            Value on the arrival wealth grid.
        c_worker_arvl : ndarray
            Consumption on the arrival wealth grid.
        dela_work_t : ndarray
            Asset derivative on the arrival wealth grid.
        ue_time : float
            Wall-clock time for the upper-envelope step.
        cons_cntn_hat : ndarray
            Unrefined consumption (pre-UE, diagnostics).
        q_cntn_hat : ndarray
            Unrefined Q-function (pre-UE, diagnostics).
        endog_grid : ndarray
            Endogenous grid (pre-UE, diagnostics).
        del_a_unrefined : ndarray
            Unrefined asset derivative (pre-UE, diagnostics).
        """

        # Step 1: Invert Euler
        t_s1 = time.time()
        cons_cntn_hat, q_cntn_hat, endog_grid, del_a_unrefined = \
            _invert_euler(
                lambda_worker_cntn,
                dlambda_worker_cntn,
                v_worker_cntn)
        t_invert = time.time() - t_s1

        min_a_val = endog_grid[0]

        # Step 2: Upper-envelope via egm_ue_global
        t_s2 = time.time()
        refined, _, _ = egm_ue_global(
            endog_grid,                          # x_dcsn_hat
            q_cntn_hat,                          # qf_hat
            beta * v_worker_cntn - delta,        # v_nxt_raw
            cons_cntn_hat,                       # c_hat
            asset_grid_A,                        # a_hat
            asset_grid_A,                        # w_grid (evaluation grid)
            du,                                  # uc_func_partial
            {"func": u, "args": {}},             # u_func placeholder
            ue_method=method.upper(),
            m_bar=m_bar,
            lb=10,
            rfc_radius=0.75,
            rfc_n_iter=40,
        )
        ue_time = time.time() - t_s2

        # Step 2b: Unpack refined dict
        t_s2b = time.time()
        egrid1 = refined["x_dcsn_ref"]
        q_cntn = refined["v_dcsn_ref"]
        c_cntn = refined["kappa_ref"]
        a_prime_clean = refined["x_cntn_ref"]
        dela_clean = np.zeros_like(a_prime_clean)
        t_unpack = time.time() - t_s2b

        # Step 3: Approximate worker policy and VF on arvl grid
        t_s3 = time.time()
        v_worker_arvl, c_worker_arvl, dela_work_t = _approx_dcsn_state_functions(
            egrid1, q_cntn, c_cntn, dela_clean, min_a_val, v_worker_cntn)
        t_approx = time.time() - t_s3

        return (v_worker_arvl,        # v_worker[<] (arrival)
                c_worker_arvl,        # c_worker[<] (arrival)
                dela_work_t,          # da[<] (arrival)
                ue_time,              # wall-clock time for UE
                cons_cntn_hat,        # c (pre-UE, diagnostics)
                q_cntn_hat,           # Q (pre-UE, diagnostics)
                endog_grid,           # w_endo (pre-UE, diagnostics)
                del_a_unrefined)      # da (pre-UE, diagnostics)

    return {
        'retire_cons': solver_retiree_stage,
        'work_cons': solver_worker_stage,
        'labour_mkt_decision': lab_mkt_choice_stage,
    }
