"""Backward induction for the durables model (DDSL)."""

import numpy as np
import time
from pathlib import Path
from kikku.dynx import (
    load_syntax, instantiate_period,
    period_to_graph, backward_paths,
)
from dolo.compiler.calibration import (
    calibrate as calibrate_stage,
)

from .model import make_cp, make_grids, make_settings
from .callables import make_callables, make_y_func
from .horses.keeper_egm import make_keeper_ops
from .horses.branching import make_tenure_ops
from .horses.adjuster_egm import make_adjuster_ops
from .horses.adjuster_negm import make_adjuster_negm_ops
from .horses.conditioning import make_conditioners


def _terminal_vlu_cntn(grids, tenure_callables):
    """Terminal condition: consume everything."""
    z_vals = grids["z"]
    a_grid = grids["a"]
    h_grid = grids["h"]
    x_all = grids["X_all"]
    term_u = tenure_callables["term_u"]
    d_a_term = tenure_callables["marginalBellman_d_a_terminal"]
    d_h_term = tenure_callables["marginalBellman_d_h_terminal"]
    g_w = tenure_callables["transitions"]["terminal_wealth"]

    shape = (len(z_vals), len(a_grid), len(h_grid))
    V = np.empty(shape)
    d_aV = np.empty(shape)
    d_hV = np.empty(shape)

    for state in range(len(x_all)):
        i_z, i_a, i_h = x_all[state]
        a = a_grid[i_a]
        h = h_grid[i_h]
        w = g_w(a, h)
        V[i_z, i_a, i_h] = term_u(w)
        d_aV[i_z, i_a, i_h] = d_a_term(w)
        d_hV[i_z, i_a, i_h] = d_h_term(w)

    return {"V": V, "d_aV": d_aV, "d_hV": d_hV}


def _apply_twister(prev_sol, twister):
    """Twister: previous arvl -> next vlu_cntn.

    Identity map because state space (a, h) is period-invariant.
    """
    return prev_sol["tenure"]["arvl"]


def recalibrate_period(period, calib_h):
    """Pure fn: return new period with updated calibration."""
    new_stages = {}
    for name, stage in period["stages"].items():
        new_stages[name] = calibrate_stage(stage, calib_h)
    return {
        "stages": new_stages,
        "connectors": period.get("connectors", []),
    }


def make_params_for_age(base_calibration, T):
    """Age-specific calibration schedule."""

    def params_for_age(h):
        calib = dict(base_calibration)
        calib["age"] = T - h
        return calib

    return params_for_age


def solve_period(stage_ops, vlu_cntn, grids, store_cntn=False,
                 verbose=False):
    """Solve one period in wave order."""
    t0 = time.perf_counter()

    A_keep, C_keep, V_keep, dVw_keep, phi_keep, keeper_egm = (
        stage_ops["keeper_cons"]["dcsn_mover"](vlu_cntn, grids)
    )
    t_keeper = time.perf_counter() - t0

    # If NEGM, inject keeper output into adjuster before it runs
    if "inject_keeper" in stage_ops["adjuster_cons"]:
        stage_ops["adjuster_cons"]["inject_keeper"](C_keep, A_keep)

    t1 = time.perf_counter()
    A_adj, C_adj, H_adj, V_adj, dVw_adj, adj_egm = (
        stage_ops["adjuster_cons"]["dcsn_mover"](vlu_cntn, grids)
    )
    t_adj = time.perf_counter() - t1

    t2 = time.perf_counter()
    vlu_dcsn, pol_dcsn = stage_ops["tenure"]["dcsn_mover"](
        vlu_cntn,
        grids,
        A_keep,
        C_keep,
        V_keep,
        dVw_keep,
        phi_keep,
        A_adj,
        C_adj,
        H_adj,
        V_adj,
        dVw_adj,
    )
    t_discrete = time.perf_counter() - t2

    vlu_arvl = stage_ops["tenure"]["arvl_mover"](vlu_dcsn)

    solve_time = time.perf_counter() - t0
    if verbose:
        print(
            "  keeper: {:.1f}ms, adj: {:.1f}ms, discrete: {:.1f}ms".format(
                t_keeper * 1000, t_adj * 1000, t_discrete * 1000
            )
        )

    # --- Assemble solution per solution_scheme.md ---

    keeper_sol = {
        "dcsn": {
            "V": V_keep,
            "d_wV": dVw_keep,
            "d_hV": phi_keep,
            "c": C_keep,
        },
    }

    adjuster_sol = {
        "dcsn": {
            "V": V_adj,
            "d_wV": dVw_adj,
            "c": C_adj,
            "h_choice": H_adj,
        },
    }

    tenure_sol = {
        "dcsn": {
            "V": vlu_dcsn["V"],
            "d_aV": vlu_dcsn["d_aV"],
            "d_hV": vlu_dcsn["d_hV"],
            "adj": pol_dcsn["adj"],
        },
        "arvl": vlu_arvl,
    }

    if store_cntn:
        keeper_cntn = {
            "V": vlu_cntn["V"],
            "d_a_nxtV": vlu_cntn["d_aV"],
            "d_h_nxtV": vlu_cntn["d_hV"],
        }
        if keeper_egm is not None:
            keeper_cntn["c"] = keeper_egm["c"]
            keeper_cntn["m_endog"] = keeper_egm["m_endog"]
        keeper_sol["cntn"] = keeper_cntn

        if adj_egm is not None:
            adjuster_sol["cntn"] = {
                "c": adj_egm["c"],
                "m_endog": adj_egm["m_endog"],
                "a_nxt_eval": adj_egm["a_nxt_eval"],
                "h_nxt_eval": adj_egm["h_nxt_eval"],
                "_refined": adj_egm["_refined"],
            }

    return {
        "keeper_cons": keeper_sol,
        "adjuster_cons": adjuster_sol,
        "tenure": tenure_sol,
        "solve_time": solve_time,
        "keeper_ms": t_keeper * 1000,
        "adj_ms": t_adj * 1000,
        "discrete_ms": t_discrete * 1000,
    }


def accrete_and_solve(
    H, period_inst, inter_conn, params_for_age, cp, grids, callables,
    settings,
    method="FUES", store_cntn=False, verbose=False,
):
    """Accrete backward, solving each period."""
    nest = {"periods": [], "twisters": [], "solutions": []}

    keep_h_fn = callables["tenure"]["transitions"]["keep_h"]

    # Build age-invariant pieces once.
    keeper_ops = {"dcsn_mover": make_keeper_ops(
        callables["keeper_cons"], keep_h_fn, grids, settings)}
    condition_V, condition_V_HD = make_conditioners(grids["Pi"])
    if method == "NEGM":
        adjuster_ops = make_adjuster_negm_ops(
            callables["adjuster_cons"], grids, settings)
    else:
        adjuster_ops = make_adjuster_ops(
            callables["adjuster_cons"], grids, settings)

    vlu_cntn = _terminal_vlu_cntn(grids, callables["tenure"])

    for h in range(H + 1):
        age = cp.T - h

        period_h = recalibrate_period(period_inst, params_for_age(h))
        nest["periods"].append(period_h)
        nest["twisters"].append(None if h == 0 else inter_conn)

        if h > 0:
            vlu_cntn = _apply_twister(nest["solutions"][h - 1], nest["twisters"][h])

        # Per-period income transitions (y_func bound to age)
        y_func_h = make_y_func(cp, age)
        income_trans_h = callables["tenure"]["make_income_transitions"](y_func_h)

        tenure_dcsn, tenure_arvl, tenure_arvl_hd = make_tenure_ops(
            callables["tenure"], income_trans_h, grids, settings,
            condition_V, condition_V_HD,
        )
        stage_ops_h = {
            "keeper_cons": keeper_ops,
            "adjuster_cons": adjuster_ops,
            "tenure": {
                "dcsn_mover": tenure_dcsn,
                "arvl_mover": tenure_arvl,
                "arvl_mover_hd": tenure_arvl_hd,
            },
        }

        sol = solve_period(stage_ops_h, vlu_cntn, grids,
                           store_cntn=store_cntn, verbose=verbose)
        sol["h"] = h
        sol["t"] = age
        nest["solutions"].append(sol)

    return nest


def solve(
    syntax_dir,
    method="FUES",
    calib_overrides=None,
    config_overrides=None,
    cp=None,
    grids=None,
    callables=None,
    settings=None,
    verbose=False,
):
    """Full DDSL pipeline: load -> accrete+solve."""
    syntax_dir = Path(syntax_dir)

    calibration, yaml_settings, stage_sources, period_template, inter_conn = load_syntax(
        syntax_dir, calib_overrides, config_overrides
    )
    period_inst = instantiate_period(
        calibration, yaml_settings, stage_sources, period_template
    )

    if cp is None:
        cp = make_cp(calibration, yaml_settings)
    if grids is None:
        grids = make_grids(cp)
    if callables is None:
        callables = make_callables(cp)
    if settings is None:
        settings = make_settings(cp)

    # Keep wave derivation for diagnostics.
    graph = period_to_graph(period_inst)
    waves = backward_paths(graph, inter_conn)
    if verbose:
        print(f"Waves: {waves}")

    H = cp.T - cp.t0
    store_cntn = bool(int(yaml_settings.get("store_cntn", 0)))
    params_for_age = make_params_for_age(calibration, cp.T)
    nest = accrete_and_solve(
        H, period_inst, inter_conn, params_for_age, cp, grids, callables,
        settings,
        method=method, store_cntn=store_cntn, verbose=verbose,
    )

    # Expose topology so the forward simulator uses the same graph
    # that was solved, not a fresh parse of the syntax directory.
    nest["graph"] = graph
    nest["inter_conn"] = inter_conn

    return nest, cp, grids, callables, settings
