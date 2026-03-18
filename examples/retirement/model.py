"""Numerical resources (rho output) for the retirement model.

The calibrated stage objects are the single source of truth
for parameter values and in principle, the callables.  This module provides:

- Default ``@njit`` equation callables (log utility).
- ``RetirementModel``: thin container that holds a reference
  to the period dict, constructs grids, and stores callables and numerical objects like the asset grid.
  Scalar params delegate to stage ``.calibration`` /
  ``.settings`` via ``__getattr__``.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _default_u(c):
    return np.log(c)


@njit(cache=True)
def _default_du(c):
    return 1.0 / c


@njit(cache=True)
def _default_uc_inv(x):
    return 1.0 / x


@njit(cache=True)
def _default_ddu(c):
    return -1.0 / (c ** 2)

# ============================================================================
# EGM sub-equation callables (cntn_to_dcsn_mover)
#
# Each callable follows a standardized signature:
#   fn(pointwise_inputs..., fixed_state, params)
#
# - `fixed_state`: non-optimized state variables held constant
#   during the EGM inversion (e.g. housing stock h, exog shock z
#   for a durables keeper).  For the retirement model this is
#   unused (0.0 placeholder).
# - `params`: model-specific scalar array.
#
# Retirement params layout:
#   params[0] = beta,  params[1] = R,  params[2] = delta
#
# Signatures (the "EGM recipe"):
#   fn_inv_euler(dv_cntn_i, fixed_state, params)          → c_i
#   fn_bellman_rhs(c_i, v_cntn_i, fixed_state, params)    → v_i
#   fn_cntn_to_dcsn(c_i, x_cntn_i, fixed_state, params)  → x_dcsn_i
#   fn_concavity(c_i, ddv_cntn_i, fixed_state, params)    → del_a_i
# ============================================================================

@njit(cache=True)
def fn_inv_euler(dv_cntn_i, fixed_state, params):
    """InvEuler: c = (beta * R * dV[>])^{-1}  (log utility)."""
    beta, R = params[0], params[1]
    return 1.0 / (beta * R * dv_cntn_i)


@njit(cache=True)
def fn_bellman_rhs(c_i, v_cntn_i, fixed_state, params):
    """Bellman RHS: V = u(c) + beta * V[>] - delta."""
    beta, delta = params[0], params[2]
    return np.log(c_i) + beta * v_cntn_i - delta


@njit(cache=True)
def fn_cntn_to_dcsn(c_i, x_cntn_i, fixed_state, params):
    """cntn_to_dcsn transition: w[>] = c + a."""
    return c_i + x_cntn_i


@njit(cache=True)
def fn_concavity(c_i, ddv_cntn_i, fixed_state, params):
    """Concavity factor: del_a = R * u''(c) / (u''(c) + beta*R*d²V)."""
    beta, R = params[0], params[1]
    ddu_c = -1.0 / (c_i ** 2)
    return R * ddu_c / (ddu_c + beta * R * ddv_cntn_i)


# ── Retiree-specific overrides ──
# The retiree has no work cost (delta=0) and a different
# cntn_to_dcsn transition: a_ret = (c + b_ret) / R.

@njit(cache=True)
def fn_bellman_rhs_ret(c_i, v_cntn_i, fixed_state, params):
    """Bellman RHS (retiree): V = u(c) + beta * V[>]  (no delta)."""
    beta = params[0]
    return np.log(c_i) + beta * v_cntn_i


@njit(cache=True)
def fn_cntn_to_dcsn_ret(c_i, x_cntn_i, fixed_state, params):
    """cntn_to_dcsn transition (retiree): a_ret = (c + b_ret) / R."""
    R = params[1]
    return (c_i + x_cntn_i) / R


DEFAULT_CALLABLES = {
    'u': _default_u,
    'du': _default_du,
    'uc_inv': _default_uc_inv,
    'ddu': _default_ddu,
}

WORKER_EGM_FNS = {
    'inv_euler': fn_inv_euler,
    'bellman_rhs': fn_bellman_rhs,
    'cntn_to_dcsn': fn_cntn_to_dcsn,
    'concavity': fn_concavity,
}

RETIREE_EGM_FNS = {
    'inv_euler': fn_inv_euler,
    'bellman_rhs': fn_bellman_rhs_ret,
    'cntn_to_dcsn': fn_cntn_to_dcsn_ret,
    'concavity': fn_concavity,
}

# ============================================================================
# Arrival-to-decision transition callables (dcsn_to_arvl_mover)
#
# These encode the forward transition g_{prec~sim} for each stage.
# Manually defined for now; will be extracted from the syntax
# automatically in a future version.
#
# Signature: g(x_arvl_grid, params) → x_dcsn_grid
#   params layout: [beta, R, delta, y]
# ============================================================================

@njit(cache=True)
def g_arvl_to_dcsn_worker(a_grid, params):
    """arvl_to_dcsn_transition (worker): w = (1+r)*a + y."""
    R, y = params[1], params[3]
    return R * a_grid + y


@njit(cache=True)
def g_arvl_to_dcsn_retiree(a_ret_grid, params):
    """arvl_to_dcsn_transition (retiree): w_ret = (1+r)*a_ret."""
    R = params[1]
    return R * a_ret_grid


class RetirementModel:
    """Numerical resources (rho output) for the retirement model.

    Holds a reference to the calibrated period dict, constructs
    the asset grid, and stores ``@njit`` equation callables.

    Scalar parameters (``beta``, ``delta``, ``y``, etc.) are
    **not** stored — ``__getattr__`` delegates to the stage's
    ``.calibration`` and ``.settings``.

    Parameters
    ----------
    period : dict
        Canonical period dict with ``"stages"`` key.
    callables : dict, optional
        Override equation callables.  Defaults to log utility.
    """

    def __init__(self, period, callables=None):
        stages = period["stages"] if "stages" in period else period
        self._work = stages["work_cons"]

        cal = self._work.calibration
        settings = self._work.settings or {}

        b = float(cal.get('b', settings.get('b', 1e-10)))
        grid_max = float(settings.get('grid_max_A', 500))
        n = int(settings.get('grid_size', 3000))

        self.asset_grid_A = np.linspace(b, grid_max, n)
        self.grid_size = n
        self.eulerK = n

        eqs = {**DEFAULT_CALLABLES, **(callables or {})}
        self.u = eqs['u']
        self.du = eqs['du']
        self.uc_inv = eqs['uc_inv']
        self.ddu = eqs['ddu']

    @property
    def R(self):
        return 1.0 + float(self._cal('r'))

    def _cal(self, key, default=None):
        """Look up a param in calibration then settings."""
        cal = self._work.calibration
        settings = self._work.settings or {}
        if key in cal:
            return cal[key]
        if key in settings:
            return settings[key]
        if default is not None:
            return default
        raise KeyError(f"'{key}' not in calibration or settings")

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._cal(name)
        except KeyError:
            raise AttributeError(name)

    @classmethod
    def with_test_defaults(cls, **overrides):
        """Construct with test defaults (no dolo-plus needed)."""
        defaults = dict(
            r=0.02, beta=0.98, delta=1.0, smooth_sigma=0,
            y=20, b=1e-10, grid_max_A=500, grid_size=3000,
            T=20, m_bar=1.2, padding_mbar=0,
        )
        defaults.update(overrides)

        class _MockStage:
            def __init__(self, p):
                self.calibration = {
                    k: p[k] for k in
                    ('r', 'beta', 'delta', 'smooth_sigma', 'y', 'b')
                }
                self.settings = {
                    k: p[k] for k in
                    ('grid_max_A', 'grid_size', 'T',
                     'm_bar', 'padding_mbar')
                }

        return cls({"stages": {"work_cons": _MockStage(defaults)}})
