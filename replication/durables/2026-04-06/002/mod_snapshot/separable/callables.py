"""Single source for all callable construction in durables.

Equation primitives (module-level @njit), EGM recipe callables baked
into make_callables(period_h). All callables are age-invariant —
income transitions (keep_w, adj_w) accept age as a runtime argument
so callables can be built once and reused across all periods.

Sections:
  1. Equation primitives — raw @njit functions with explicit parameters
  2. Equation callable factory — make_callables(period_h)
"""

import numpy as np
from numba import njit




# ============================================================
# Section 1: Equation primitives (module-level @njit)
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
# Section 2: Equation callable factory (stage-scoped + age)
# ============================================================

def make_callables(period_h):
    """Return structural callables (age-invariant).

    Reads all parameters from the calibrated stage objects' ``.calibration``
    dict. Income transitions (``keep_w``, ``adj_w``) accept ``age`` as a
    runtime argument rather than baking it in — this allows callables to be
    constructed once and reused across all periods.

    Parameters
    ----------
    period_h : dict
        Dolo+ period after ``recalibrate_period``.

    All parameters are baked into @njit closures at construction time.
    """
    # Period calibration is merged into every stage; keeper_cons holds the
    # full preference / income block used by all three callable bundles.
    cal = period_h["stages"]["keeper_cons"].calibration

    chi = float(cal["chi"])
    alpha = float(cal["alpha"])
    gamma_c = float(cal["gamma_c"])
    gamma_h = float(cal["gamma_h"])
    kappa = float(cal["kappa"])
    theta = float(cal["theta"])
    K = float(cal["K"])
    beta = float(cal["beta"])
    R = float(cal["R"])
    R_H = float(cal["R_H"])
    delta = float(cal["delta"])
    tau = float(cal["tau"])

    lambdas = np.asarray(cal["lambdas"], dtype=np.float64)
    tau_av = int(cal["tau_av"])
    tzero = int(cal.get("tzero", 20))
    normalisation = float(cal["normalisation"])

    @njit
    def u(c, h):
        return _u(c, h, chi, alpha, gamma_c, gamma_h, kappa)

    @njit
    def d_c_u(c, h):
        """Marginal utility of c. h accepted for CD compatibility but ignored (separable)."""
        return _du_c(c, alpha, gamma_c)

    @njit
    def d_c_u_inv(m, h):
        """Inverse marginal utility. h accepted for CD compatibility but ignored (separable)."""
        return _du_c_inv(m, alpha, gamma_c)

    @njit
    def d_h_u(c, h):
        """Marginal utility of h. c accepted for CD compatibility but ignored (separable)."""
        return _du_h(h, alpha, gamma_h, kappa)

    @njit
    def term_u(w):
        return _term_u(w, theta, alpha, gamma_c, K)

    @njit
    def term_du(w):
        return _term_du(w, theta, alpha, gamma_c, K)

    @njit
    def bellman_obj(u_val, v_cntn):
        return u_val + beta * v_cntn

    @njit
    def bellman_discount(v):
        return beta * v

    @njit
    def housing_cost(h):
        return (1.0 + tau) * h

    @njit
    def marginalBellman_d_a(dvw):
        return beta * R * dvw

    @njit
    def marginalBellman_d_h_keep(phi):
        return beta * (1.0 - delta) * phi

    @njit
    def marginalBellman_d_h_adj(dvw):
        return beta * R_H * (1.0 - delta) * dvw

    @njit
    def marginalBellman_d_a_terminal(w):
        return beta * R * term_du(w)

    @njit
    def marginalBellman_d_h_terminal(w):
        return beta * R_H * (1.0 - delta) * term_du(w)

    @njit
    def invEuler_foc_h_residual(dv_a, dv_h, h):
        """For CD compat, d_h_u needs c — recovered from dv_a via Euler.
        For separable, d_h_u(c, h) ignores c, so we pass a dummy."""
        c = d_c_u_inv(dv_a, h)
        return (1.0 + tau) * dv_a - dv_h - d_h_u(c, h)

    @njit
    def invEuler_foc_h_rhs(d_h_u_val, phi):
        return (d_h_u_val + phi) / (1.0 + tau)

    # --- FOC diagnostics and c-recovery (compose from bound closures) ---

    @njit
    def euler_error_c(c, rhs, h):
        """h accepted for CD compatibility but ignored (separable)."""
        c_hat = d_c_u_inv(rhs, h)
        return np.log10(abs((c - c_hat) / c) + 1e-16)

    @njit
    def euler_error_h(c, h_nxt, phi):
        lhs = invEuler_foc_h_rhs(d_h_u(c, h_nxt), phi)
        c_hat = d_c_u_inv(lhs, h_nxt)
        return np.log10(abs((c - c_hat) / c) + 1e-16)

    @njit
    def invEuler_foc_h_c(du_h_val, phi):
        """Recover c from durable FOC (separable factoring).
        For CD, the adjuster uses d_c_u_inv(dv_a, h) instead."""
        return d_c_u_inv(invEuler_foc_h_rhs(du_h_val, phi), 0.0)

    # --- Keeper EGM recipe (params baked in; signature pointwise, fixed_state) ---

    @njit
    def keeper_inv_euler(dv_cntn_i, fixed_state):
        if dv_cntn_i <= 0:
            return 1e10
        return (alpha / dv_cntn_i) ** (1.0 / gamma_c)

    @njit
    def keeper_bellman_rhs(c_i, v_cntn_i, fixed_state):
        h_prime = fixed_state
        if c_i <= 0 or h_prime <= 0:
            return -1e20
        c_util = alpha * c_i ** (1.0 - gamma_c) / (1.0 - gamma_c)
        h_util = (1.0 - alpha) * (kappa * h_prime) ** (1.0 - gamma_h) / (1.0 - gamma_h)
        return c_util + h_util + beta * v_cntn_i

    @njit
    def keeper_cntn_to_dcsn(c_i, x_cntn_i, fixed_state):
        return c_i + x_cntn_i

    @njit
    def keeper_concavity(c_i, ddv_cntn_i, fixed_state):
        return 0.0

    keeper_egm_fns = {
        "inv_euler": keeper_inv_euler,
        "bellman_rhs": keeper_bellman_rhs,
        "cntn_to_dcsn": keeper_cntn_to_dcsn,
        "concavity": keeper_concavity,
    }

    # --- Age-invariant transition closures ---

    @njit
    def tr_keeper_cntn_to_dcsn(a_nxt, c):
        return a_nxt + c

    @njit
    def tr_keeper_dcsn_to_cntn(w_keep, c):
        return w_keep - c

    @njit
    def tr_adj_cntn_to_dcsn_m(a_nxt, c, h_choice):
        return a_nxt + c + (1.0 + tau) * h_choice

    @njit
    def tr_adj_cntn_to_dcsn_budget(m, a_nxt, h_choice):
        return m - a_nxt - (1.0 + tau) * h_choice

    @njit
    def tr_tenure_keep_h(h):
        return (1.0 - delta) * h

    @njit
    def tr_terminal_wealth(a, h):
        return R * a + R_H * (1.0 - delta) * h

    # --- Income transitions (age is a runtime argument, not baked in) ---

    @njit
    def y_func(age, xi):
        return _y_func(age, xi, lambdas, tau_av, tzero, normalisation)

    @njit
    def keep_w(a, z, age):
        return R * a + y_func(age, z)

    @njit
    def adj_w(a, h, z, age):
        return R * a + R_H * (1.0 - delta) * h + y_func(age, z)

    return {
        "keeper_cons": {
            "u": u,
            "d_c_u": d_c_u,
            "d_c_u_inv": d_c_u_inv,
            "d_h_u": d_h_u,
            "bellman_obj": bellman_obj,
            "bellman_discount": bellman_discount,
            "euler_error_c": euler_error_c,
            "keeper_egm_fns": keeper_egm_fns,
            "transitions": {
                "cntn_to_dcsn": tr_keeper_cntn_to_dcsn,
                "dcsn_to_cntn": tr_keeper_dcsn_to_cntn,
            },
        },
        "adjuster_cons": {
            "u": u,
            "d_c_u": d_c_u,
            "d_c_u_inv": d_c_u_inv,
            "d_h_u": d_h_u,
            "bellman_obj": bellman_obj,
            "bellman_discount": bellman_discount,
            "housing_cost": housing_cost,
            "invEuler_foc_h_residual": invEuler_foc_h_residual,
            "invEuler_foc_h_rhs": invEuler_foc_h_rhs,
            "euler_error_h": euler_error_h,
            "invEuler_foc_h_c": invEuler_foc_h_c,
            "transitions": {
                "cntn_to_dcsn_m": tr_adj_cntn_to_dcsn_m,
                "cntn_to_dcsn_budget": tr_adj_cntn_to_dcsn_budget,
            },
        },
        "tenure": {
            "u": u,
            "bellman_obj": bellman_obj,
            "housing_cost": housing_cost,
            "term_u": term_u,
            "term_du": term_du,
            "marginalBellman_d_a": marginalBellman_d_a,
            "marginalBellman_d_h_keep": marginalBellman_d_h_keep,
            "marginalBellman_d_h_adj": marginalBellman_d_h_adj,
            "marginalBellman_d_a_terminal": marginalBellman_d_a_terminal,
            "marginalBellman_d_h_terminal": marginalBellman_d_h_terminal,
            "transitions": {
                "keep_h": tr_tenure_keep_h,
                "terminal_wealth": tr_terminal_wealth,
                "keep_w": keep_w,
                "adj_w": adj_w,
            },
        },
    }
