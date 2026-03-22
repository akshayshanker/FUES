"""Transition callables for durables2_0.

Each function corresponds to a transition equation declared in
the stage YAML syntax. Direction-first naming convention:

  g_{source_perch}_to_{target_perch}_{stage}[_{branch}]

These are the g-functions (morphisms between perch spaces) from
doloplus-foundations §4.2. Movers reference them; they have
independent mathematical identity.

Parameters are pre-bound at construction time via make_transitions(cp, y_func).
"""

from numba import njit


def make_transitions(cp, y_func):
    """Build all transition callables with parameters pre-bound.

    Parameters
    ----------
    cp : ConsumerProblem
        For R, R_H, delta, tau, beta.
    y_func : callable
        Age-bound income function y_func(z).

    Returns
    -------
    dict
        Transition callables keyed by DDSL direction-first names.
    """
    R = cp.R
    R_H = cp.R_H
    delta = cp.delta
    tau = cp.tau
    beta = cp.beta

    # ============================================================
    # Tenure stage: dcsn_to_cntn_transition
    # ============================================================

    @njit
    def g_dcsn_to_cntn_keep_w(a, z):
        """tenure.dcsn_to_cntn_transition.keep: w_keep[>] = R*a + y(z)"""
        return R * a + y_func(z)

    @njit
    def g_dcsn_to_cntn_keep_h(h):
        """tenure.dcsn_to_cntn_transition.keep: h_keep[>] = (1-delta)*h"""
        return (1 - delta) * h

    @njit
    def g_dcsn_to_cntn_adj_w(a, h, z):
        """tenure.dcsn_to_cntn_transition.adjust: w_adj[>] = R*a + R_H*(1-delta)*h + y(z)"""
        return R * a + R_H * (1 - delta) * h + y_func(z)

    # ============================================================
    # Keeper stage: transitions
    # ============================================================

    @njit
    def g_cntn_to_dcsn(a_nxt, c):
        """keeper.cntn_to_dcsn_transition: w_keep[>] = a_nxt + c[>]"""
        return a_nxt + c

    @njit
    def g_dcsn_to_cntn(w_keep, c):
        """keeper.dcsn_to_cntn_transition: a_nxt = w_keep - c"""
        return w_keep - c

    # ============================================================
    # Adjuster stage: transitions
    # ============================================================

    @njit
    def g_cntn_to_dcsn_m(a_nxt, c, h_choice):
        """adjuster.cntn_to_dcsn_transition: m[>] = a_nxt + c[>] + (1+tau)*h_choice[>]"""
        return a_nxt + c + (1 + tau) * h_choice

    @njit
    def g_cntn_to_dcsn_c(m, a_nxt, h_choice):
        """adjuster.cntn_to_dcsn_transition (inverted for c): c = m - a_nxt - (1+tau)*h_choice"""
        return m - a_nxt - (1 + tau) * h_choice

    # ============================================================
    # Terminal condition
    # ============================================================

    @njit
    def g_terminal_wealth(a, h):
        """Terminal wealth: w = R*a + R_H*(1-delta)*h"""
        return R * a + R_H * (1 - delta) * h

    return {
        "tenure": {
            "dcsn_to_cntn": {
                "keep_w": g_dcsn_to_cntn_keep_w,
                "keep_h": g_dcsn_to_cntn_keep_h,
                "adj_w": g_dcsn_to_cntn_adj_w,
            },
        },
        "keeper": {
            "cntn_to_dcsn": g_cntn_to_dcsn,
            "dcsn_to_cntn": g_dcsn_to_cntn,
        },
        "adjuster": {
            "cntn_to_dcsn": g_cntn_to_dcsn_m,
            "cntn_to_dcsn_budget": g_cntn_to_dcsn_c,
        },
        "terminal": g_terminal_wealth,
    }
