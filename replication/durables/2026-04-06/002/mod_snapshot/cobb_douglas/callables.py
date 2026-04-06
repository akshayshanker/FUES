"""Cobb-Douglas CRRA callables for durables.

Same interface as callables.py (separable CRRA) — returns the same
dict of function names. The solver dispatches based on calibration
keys: 'rho' in cal → this file, 'gamma_c' in cal → callables.py.

Utility:  u(c, h) = (c^α · (h + h̄)^(1-α))^(1-ρ) / (1-ρ)

Key difference from separable:
  - du_c(c, h) depends on h   (separable: du_c(c) independent of h)
  - du_c_inv(m, h) depends on h
  - du_h(c, h) depends on c   (separable: du_h(h) independent of c)
  - keeper_inv_euler USES fixed_state=h (separable ignores it)

Math sourced from: examples/durable-cons/durable-cons/utility.py
"""

import numpy as np
from numba import njit


# ============================================================
# Section 1: Equation primitives (module-level @njit)
# ============================================================

@njit(cache=True)
def _u_cd(c, h, alpha, rho, d_ubar):
    """Cobb-Douglas CRRA: (c^α · (h+h̄)^(1-α))^(1-ρ) / (1-ρ)."""
    if c <= 0 or h + d_ubar <= 0:
        return -np.inf
    htot = h + d_ubar
    c_total = c ** alpha * htot ** (1.0 - alpha)
    return c_total ** (1.0 - rho) / (1.0 - rho)


@njit(cache=True)
def _du_c_cd(c, h, alpha, rho, d_ubar):
    """du/dc = α · c^(α(1-ρ)-1) · (h+h̄)^((1-α)(1-ρ))."""
    if c <= 0:
        return 1e250
    htot = h + d_ubar
    if htot <= 0:
        return 1e250
    c_power = alpha * (1.0 - rho) - 1.0
    d_power = (1.0 - alpha) * (1.0 - rho)
    return alpha * (c ** c_power) * (htot ** d_power)


@njit(cache=True)
def _du_c_inv_cd(m, h, alpha, rho, d_ubar):
    """Inverse: c = (m / (α · (h+h̄)^((1-α)(1-ρ))))^(1/(α(1-ρ)-1))."""
    if m <= 0:
        return 1e250
    htot = h + d_ubar
    if htot <= 0:
        return 1e250
    c_power = alpha * (1.0 - rho) - 1.0
    d_power = (1.0 - alpha) * (1.0 - rho)
    denom = alpha * htot ** d_power
    return (m / denom) ** (1.0 / c_power)


@njit(cache=True)
def _du_h_cd(c, h, alpha, rho, d_ubar):
    """du/dh = (1-α) · c^(α(1-ρ)) · (h+h̄)^((1-α)(1-ρ)-1)."""
    if c <= 0:
        return 1e250
    htot = h + d_ubar
    if htot <= 0:
        return 1e250
    c_power = alpha * (1.0 - rho)
    d_power = (1.0 - alpha) * (1.0 - rho) - 1.0
    return (1.0 - alpha) * (c ** c_power) * (htot ** d_power)


@njit(cache=True)
def _term_u_cd(w, theta, alpha, rho, tau, d_ubar):
    """Terminal indirect utility for CD with budget c + (1+τ)h = w.

    Intratemporal FOC gives optimal split:
      c* = alpha * (w + (1+tau)*d_ubar)
      h* + d_ubar = (1-alpha) * (w + (1+tau)*d_ubar) / (1+tau)
    Composite: C = alpha^alpha * ((1-alpha)/(1+tau))^(1-alpha) * (w + (1+tau)*d_ubar)
    V_term = theta * C^(1-rho) / (1-rho)
    """
    if w <= 0:
        return -np.inf
    w_eff = w + (1.0 + tau) * d_ubar
    scale = alpha ** alpha * ((1.0 - alpha) / (1.0 + tau)) ** (1.0 - alpha)
    c_agg = scale * w_eff
    return theta * c_agg ** (1.0 - rho) / (1.0 - rho)


@njit(cache=True)
def _term_du_cd(w, theta, alpha, rho, tau, d_ubar):
    """Terminal marginal utility: dV_term/dw.

    d/dw of V_term = theta * scale^(1-rho) * (1-rho) * (w_eff)^((1-rho)-1) * 1 / (1-rho)
                   = theta * scale^(1-rho) * (w_eff)^(-rho)
    where scale = alpha^alpha * ((1-alpha)/(1+tau))^(1-alpha)
    """
    if w <= 0:
        return 1e250
    w_eff = w + (1.0 + tau) * d_ubar
    scale = alpha ** alpha * ((1.0 - alpha) / (1.0 + tau)) ** (1.0 - alpha)
    return theta * scale ** (1.0 - rho) * w_eff ** (-rho)


@njit(cache=True)
def _y_func(t, xi, lambdas, tau_av, tzero, normalisation):
    """Income: age-tenure polynomial + AR(1) shock (unchanged from separable)."""
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
    """Return Cobb-Douglas CRRA callables. Same interface as callables.py.

    Reads parameters from the calibrated stage objects.
    Expects: alpha, rho, d_ubar (NOT gamma_c, gamma_h, kappa, chi).
    """
    cal = period_h["stages"]["keeper_cons"].calibration

    alpha = float(cal["alpha"])
    rho = float(cal["rho"])
    d_ubar = float(cal["d_ubar"])
    theta = float(cal["theta"])
    beta = float(cal["beta"])
    R = float(cal["R"])
    R_H = float(cal["R_H"])
    delta = float(cal["delta"])
    tau = float(cal["tau"])

    lambdas = np.asarray(cal["lambdas"], dtype=np.float64)
    tau_av = int(cal["tau_av"])
    tzero = int(cal.get("tzero", 20))
    normalisation = float(cal["normalisation"])

    # --- Utility and marginals (CD) ---

    @njit
    def u(c, h):
        return _u_cd(c, h, alpha, rho, d_ubar)

    @njit
    def d_c_u(c, h):
        """NOTE: takes (c, h), not just (c). Separable version takes only (c)."""
        return _du_c_cd(c, h, alpha, rho, d_ubar)

    @njit
    def d_c_u_inv(m, h):
        """NOTE: takes (m, h), not just (m). Separable version takes only (m)."""
        return _du_c_inv_cd(m, h, alpha, rho, d_ubar)

    @njit
    def d_h_u(c, h):
        """NOTE: takes (c, h), not just (h). Separable version takes only (h)."""
        return _du_h_cd(c, h, alpha, rho, d_ubar)

    @njit
    def term_u(w):
        return _term_u_cd(w, theta, alpha, rho, tau, d_ubar)

    @njit
    def term_du(w):
        return _term_du_cd(w, theta, alpha, rho, tau, d_ubar)

    # --- Bellman combinators (unchanged) ---

    @njit
    def bellman_obj(u_val, v_cntn):
        return u_val + beta * v_cntn

    @njit
    def bellman_discount(v):
        return beta * v

    @njit
    def housing_cost(h):
        return (1.0 + tau) * h

    # --- Marginal Bellman operators (unchanged structure) ---

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

    # --- Adjuster FOC residuals (CD: du_c and du_h depend on both c and h) ---

    @njit
    def invEuler_foc_h_residual(dv_a, dv_h, h):
        """Durable FOC residual: (1+τ)·dv_a - dv_h - du_h(c, h)
        where c = du_c_inv(dv_a, h) from the asset Euler.

        This is the key difference from separable: we recover c from
        dv_a and h, then evaluate du_h(c, h).
        """
        c = d_c_u_inv(dv_a, h)
        return (1.0 + tau) * dv_a - dv_h - d_h_u(c, h)

    @njit
    def invEuler_foc_h_rhs(d_h_u_val, phi):
        return (d_h_u_val + phi) / (1.0 + tau)

    @njit
    def invEuler_foc_h_c(du_h_val, phi):
        """Recover c from the housing FOC. For CD, we need h to compute
        d_c_u_inv, but this function is called where h is already known
        from the FOC context. Placeholder — the adjuster horse calls
        invEuler_foc_h_residual directly.
        """
        # This is a placeholder; the adjuster horse resolves via root-finding
        # on invEuler_foc_h_residual, not through this function.
        return 0.0

    # --- Euler error diagnostics (CD versions) ---

    @njit
    def euler_error_c(c, rhs, h):
        """Euler error: compare c to inv_euler(rhs, h).
        NOTE: takes (c, rhs, h). Separable takes (c, rhs).
        """
        c_hat = d_c_u_inv(rhs, h)
        return np.log10(abs((c - c_hat) / c) + 1e-16)

    @njit
    def euler_error_h(c, h_nxt, phi):
        """Housing Euler error. du_h depends on c for CD."""
        d_h_u_val = d_h_u(c, h_nxt)
        lhs = invEuler_foc_h_rhs(d_h_u_val, phi)
        c_hat = d_c_u_inv(lhs, h_nxt)
        return np.log10(abs((c - c_hat) / c) + 1e-16)

    # --- Keeper EGM recipe (CD: inv_euler USES fixed_state=h) ---

    @njit
    def keeper_inv_euler(dv_cntn_i, fixed_state):
        """Inverse Euler for keeper. fixed_state = h_keep.

        c = inv_marg_func(beta*R*dV, h_keep)
        From utility.py: inv_marg_func_nopar(q, d, d_ubar, alpha, rho)
        """
        h = fixed_state
        if dv_cntn_i <= 0:
            return 1e10
        return _du_c_inv_cd(dv_cntn_i, h, alpha, rho, d_ubar)

    @njit
    def keeper_bellman_rhs(c_i, v_cntn_i, fixed_state):
        h_prime = fixed_state
        if c_i <= 0 or h_prime + d_ubar <= 0:
            return -1e20
        return _u_cd(c_i, h_prime, alpha, rho, d_ubar) + beta * v_cntn_i

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

    # --- Transitions (unchanged from separable) ---

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

    # --- Income transitions (unchanged) ---

    @njit
    def y_func(age, xi):
        return _y_func(age, xi, lambdas, tau_av, tzero, normalisation)

    @njit
    def keep_w(a, z, age):
        return R * a + y_func(age, z)

    @njit
    def adj_w(a, h, z, age):
        return R * a + R_H * (1.0 - delta) * h + y_func(age, z)

    # --- Return dict: SAME KEYS as callables.py ---

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
