"""Numerical resources for durables2_0.

Exposes the functorial triple (cp, grids, callables):

  cp        — ConsumerProblem instance (bound stage proxy)
  grids     — space discretizations (created once)
  callables — structural equation functions (created once)

Age-varying income is bound separately via make_y_func(cp, age).
"""

import numpy as np
import interpolation.splines as _splines
from numba import njit
from examples.durables.durables import ConsumerProblem


# ============================================================
# UCGrid patch (numba dtype assertion workaround)
# ============================================================

_orig_UCGrid = _splines.UCGrid


def _patched_UCGrid(*args):
    fixed = []
    for tup in args:
        lo, hi, n = tup
        fixed.append((float(lo), float(hi), int(n)))
    return _orig_UCGrid(*fixed)


_splines.UCGrid = _patched_UCGrid


# ============================================================
# Equation callables (module-level @njit)
#
# Raw functions with explicit parameters. make_callables(cp)
# binds the parameters to produce thin wrappers with the
# signatures expected by operators: du_c(c), u(c,h,chi), etc.
# ============================================================

@njit(cache=True)
def _u(c, h, chi, alpha, gamma_c, gamma_h, kappa):
    """CRRA utility: alpha*c^(1-gc)/(1-gc) + (1-alpha)*(kappa*h)^(1-gh)/(1-gh) - chi."""
    if c <= 0 or h <= 0:
        return -np.inf
    c_util = alpha * c ** (1.0 - gamma_c) / (1.0 - gamma_c)
    h_util = (1.0 - alpha) * (kappa * h) ** (1.0 - gamma_h) / (1.0 - gamma_h)
    return c_util + h_util - chi


@njit(cache=True)
def _du_c(c, alpha, gamma_c):
    """Marginal utility of consumption: alpha * c^(-gamma_c)."""
    if c <= 0:
        return 1e250
    return alpha * c ** (-gamma_c)


@njit(cache=True)
def _du_c_inv(m, alpha, gamma_c):
    """Inverse marginal utility: c = (alpha/m)^(1/gamma_c)."""
    if m <= 0:
        return 1e250
    return (alpha / m) ** (1.0 / gamma_c)


@njit(cache=True)
def _du_h(h, alpha, gamma_h, kappa):
    """Marginal utility of housing: (1-alpha)*kappa^(1-gh)*h^(-gh)."""
    if h <= 0:
        return 1e250
    return (1.0 - alpha) * kappa ** (1.0 - gamma_h) * h ** (-gamma_h)


@njit(cache=True)
def _term_u(w, theta, alpha, gamma_c, K):
    """Terminal utility."""
    if w <= 0:
        return -np.inf
    return theta * alpha * (K + w) ** (1.0 - gamma_c) / (1.0 - gamma_c)


@njit(cache=True)
def _term_du(w, theta, alpha, gamma_c, K):
    """Terminal marginal utility."""
    if w <= 0:
        return 1e250
    return theta * alpha * (K + w) ** (-gamma_c)


@njit(cache=True)
def _y_func(t, xi, lambdas, tau_av, tzero, normalisation):
    """Income: age-tenure polynomial + AR(1) shock."""
    if t > 60:
        return 0.1
    tau_tenure = min(tau_av, t - tzero)
    age_factors = np.array([1.0, float(t), float(t)**2,
                            float(t)**3, float(t)**4])
    wage_age = np.dot(age_factors, lambdas[0:5])
    tenure_factors = np.array([float(tau_tenure),
                               float(tau_tenure)**2])
    wage_tenure = np.dot(tenure_factors, lambdas[5:7])
    return np.exp(wage_age + wage_tenure + xi) * normalisation


# ============================================================
# make_callables: bind parameters from cp
# ============================================================

def make_callables(cp):
    """Return structural callables from ConsumerProblem (age-invariant)."""
    return {
        "u": cp.u,
        "du_c": cp.du_c,
        "du_c_inv": cp.du_c_inv,
        "du_h": cp.du_h,
        "term_u": cp.term_u,
        "term_du": cp.term_du,
    }


def make_y_func(cp, age):
    """Bind age-specific income callable y_func(z) for one period."""
    lambdas = cp.lambdas
    tau_av = cp.tau_av
    tzero = cp.tzero
    normalisation = cp.normalisation

    @njit
    def y_func(xi):
        return _y_func(age, xi, lambdas, tau_av, tzero, normalisation)

    return y_func


# ============================================================
# EGM recipe callables (kikku make_egm_1d convention)
#
# fn(pointwise_input, fixed_state, params) -> result
#
# Keeper params layout:
#   params[0] = beta,  [1] = alpha,  [2] = gamma_c
#   params[3] = gamma_h,  [4] = kappa
# ============================================================

@njit(cache=True)
def keeper_inv_euler(dv_cntn_i, fixed_state, params):
    """InvEuler: c = (alpha / dV)^{1/gamma_c}."""
    alpha = params[1]
    gamma_c = params[2]
    if dv_cntn_i <= 0:
        return 1e10
    return (alpha / dv_cntn_i) ** (1.0 / gamma_c)


@njit(cache=True)
def keeper_bellman_rhs(c_i, v_cntn_i, fixed_state, params):
    """Bellman RHS: u(c, h') + beta * V_cntn."""
    beta = params[0]
    alpha = params[1]
    gamma_c = params[2]
    gamma_h = params[3]
    kappa = params[4]
    h_prime = fixed_state
    if c_i <= 0 or h_prime <= 0:
        return -1e20
    c_util = alpha * c_i ** (1.0 - gamma_c) / (1.0 - gamma_c)
    h_util = (1.0 - alpha) * (kappa * h_prime) ** (1.0 - gamma_h) / (1.0 - gamma_h)
    return c_util + h_util + beta * v_cntn_i


@njit(cache=True)
def keeper_cntn_to_dcsn(c_i, x_cntn_i, fixed_state, params):
    """Endogenous grid transition: w = c + a'."""
    return c_i + x_cntn_i


@njit(cache=True)
def _keeper_concavity(c_i, ddv_cntn_i, fixed_state, params):
    return 0.0


KEEPER_EGM_FNS = {
    "inv_euler": keeper_inv_euler,
    "bellman_rhs": keeper_bellman_rhs,
    "cntn_to_dcsn": keeper_cntn_to_dcsn,
    "concavity": _keeper_concavity,
}


# ============================================================
# Construction helpers
# ============================================================

def make_cp(calibration, settings):
    """Create ConsumerProblem from YAML calibration + settings."""
    config = dict(calibration)
    config.update(settings)
    return ConsumerProblem(
        config,
        r=float(calibration["r"]),
        beta=float(calibration["beta"]),
        alpha=float(calibration.get("alpha", 0.7)),
        delta=float(calibration.get("delta", 0.0)),
        kappa=float(calibration.get("kappa", 1.0)),
        sigma=float(calibration.get("sigma", 0.001)),
        r_H=float(calibration.get("r_H", 0.0)),
        b=float(calibration.get("b", 1e-8)),
        gamma_c=float(calibration.get("gamma_c", 3.0)),
        gamma_h=float(calibration.get("gamma_h", 1.0)),
        chi=float(calibration.get("chi", 0.0)),
        tau=float(calibration.get("tau", 0.2)),
        K=float(calibration.get("K", 200.0)),
        theta=float(calibration.get("theta", 2.0)),
        grid_max_A=float(settings.get("a_max", 50.0)),
        grid_max_H=float(settings.get("h_max", 50.0)),
        grid_max_WE=float(settings.get("w_max", 100.0)),
        grid_size_A=int(settings.get("n_a", 50)),
        grid_size_H=int(settings.get("n_h", 50)),
        grid_size_W=int(settings.get("n_w", 50)),
        m_bar=float(settings.get("m_bar", 1.4)),
        T=int(settings.get("T", 60)),
        t0=int(calibration.get("t0", settings.get("T", 60))),
        N_wage=int(calibration.get("N_wage", 3)),
        phi_w=float(calibration.get("phi_w", 0.917)),
        sigma_w=float(calibration.get("sigma_w", 0.082)),
    )


def make_grids(cp):
    """Extract space discretizations from cp. Created once."""
    return {
        "a": cp.asset_grid_A,
        "h": cp.asset_grid_H,
        "we": cp.asset_grid_WE,
        "z": cp.z_vals,
        "Pi": cp.Pi,
        "X_all": cp.X_all,
        "UGgrid_all": cp.UGgrid_all,
    }
