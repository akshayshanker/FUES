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
from examples.durables.durables import Operator_Factory

from .model import make_cp, make_grids, make_callables, make_y_func
from .horses.keeper_egm import make_keeper_ops
from .horses.branching import make_tenure_ops


def _make_adjuster_ops(cp, internals, callables):
    """Age-invariant adjuster pieces from Operator_Factory internals."""
    _adjEGM = internals["_adjEGM"]
    _refine_adj = internals["refine_adj"]
    return_grids = cp.return_grids
    du_c_op = callables["du_c"]

    def build_for_age(age):
        def dcsn_mover(vlu_cntn, grids):
            egrid, vf, a_nxt, h_nxt = _adjEGM(
                vlu_cntn["d_aV"],
                vlu_cntn["d_hV"],
                vlu_cntn["V"],
                age,
            )
            (Aadj, Cadj, Hadj, Vadj,
             _, _, _, _, _, _, _, _) = _refine_adj(
                egrid,
                vf,
                a_nxt,
                h_nxt,
                m_bar=cp.m_bar,
                return_grids=return_grids,
            )
            n_z_loc = Cadj.shape[0]
            n_w = Cadj.shape[1]
            dVw_adj = np.empty((n_z_loc, n_w))
            for iz in range(n_z_loc):
                for iw in range(n_w):
                    dVw_adj[iz, iw] = du_c_op(Cadj[iz, iw])
            return Aadj, Cadj, Hadj, Vadj, dVw_adj

        return {"dcsn_mover": dcsn_mover}

    return build_for_age


def _terminal_vlu_cntn(cp, grids, callables):
    """Terminal condition: consume everything."""
    z_vals = grids["z"]
    a_grid = grids["a"]
    h_grid = grids["h"]
    x_all = grids["X_all"]
    term_u = callables["term_u"]
    term_du = callables["term_du"]

    shape = (len(z_vals), len(a_grid), len(h_grid))
    V = np.empty(shape)
    d_aV = np.empty(shape)
    d_hV = np.empty(shape)

    for state in range(len(x_all)):
        i_z, i_a, i_h = x_all[state]
        a = a_grid[i_a]
        h = h_grid[i_h]
        w = cp.R_H * (1 - cp.delta) * h + cp.R * a
        V[i_z, i_a, i_h] = term_u(w)
        d_aV[i_z, i_a, i_h] = cp.beta * cp.R * term_du(w)
        d_hV[i_z, i_a, i_h] = cp.beta * cp.R_H * term_du(w) * (1 - cp.delta)

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


def solve_period(stage_ops, vlu_cntn, grids, verbose=False):
    """Solve one period in wave order without explicit t/cp arguments."""
    t0 = time.perf_counter()

    A_keep, C_keep, V_keep, dVw_keep, phi_keep = stage_ops["keeper_cons"]["dcsn_mover"](
        vlu_cntn, grids
    )
    t_keeper = time.perf_counter() - t0

    t1 = time.perf_counter()
    Aadj, Cadj, Hadj, Vadj, dVw_adj = stage_ops["adjuster_cons"]["dcsn_mover"](
        vlu_cntn, grids
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
        Aadj,
        Cadj,
        Hadj,
        Vadj,
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

    return {
        "keeper_cons": {"dcsn": {"V": V_keep, "c": C_keep, "a": A_keep}},
        "adjuster_cons": {"dcsn": {"V": Vadj, "c": Cadj, "a": Aadj, "h": Hadj}},
        "tenure": {
            "dcsn": {
                "V": vlu_dcsn["V"],
                "d_aV": vlu_dcsn["d_aV"],
                "d_hV": vlu_dcsn["d_hV"],
                "d": pol_dcsn["d"],
            },
            "arvl": vlu_arvl,
        },
        "solve_time": solve_time,
    }


def accrete_and_solve(
    H, period_inst, inter_conn, params_for_age, cp, grids, callables, verbose=False
):
    """Accrete backward, solving each period."""
    nest = {"periods": [], "twisters": [], "solutions": []}
    vlu_cntn = _terminal_vlu_cntn(cp, grids, callables)

    # Build age-invariant pieces once.
    keeper_ops = {"dcsn_mover": make_keeper_ops(cp, callables)}
    (_, _, condition_V, condition_V_HD, _, internals) = Operator_Factory(cp)
    build_adjuster_for_age = _make_adjuster_ops(cp, internals, callables)

    for h in range(H + 1):
        age = cp.T - h

        period_h = recalibrate_period(period_inst, params_for_age(h))
        nest["periods"].append(period_h)
        nest["twisters"].append(None if h == 0 else inter_conn)

        if h > 0:
            vlu_cntn = _apply_twister(nest["solutions"][h - 1], nest["twisters"][h])

        y_func_h = make_y_func(cp, age)
        tenure_dcsn, tenure_arvl, tenure_arvl_hd = make_tenure_ops(
            cp, callables, y_func_h, condition_V, condition_V_HD
        )
        adjuster_ops = build_adjuster_for_age(age)

        stage_ops_h = {
            "keeper_cons": keeper_ops,
            "adjuster_cons": adjuster_ops,
            "tenure": {
                "dcsn_mover": tenure_dcsn,
                "arvl_mover": tenure_arvl,
                "arvl_mover_hd": tenure_arvl_hd,
            },
        }

        sol = solve_period(stage_ops_h, vlu_cntn, grids, verbose=verbose)
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
    verbose=False,
):
    """Full DDSL pipeline: load -> accrete+solve."""
    syntax_dir = Path(syntax_dir)

    calibration, settings, stage_sources, period_template, inter_conn = load_syntax(
        syntax_dir, calib_overrides, config_overrides
    )
    period_inst = instantiate_period(
        calibration, settings, stage_sources, period_template
    )

    if cp is None:
        cp = make_cp(calibration, settings)
    if grids is None:
        grids = make_grids(cp)
    if callables is None:
        callables = make_callables(cp)

    # Keep wave derivation for diagnostics.
    graph = period_to_graph(period_inst)
    waves = backward_paths(graph, inter_conn)
    if verbose:
        print(f"Waves: {waves}")

    H = cp.T - cp.t0
    params_for_age = make_params_for_age(calibration, cp.T)
    nest = accrete_and_solve(
        H, period_inst, inter_conn, params_for_age, cp, grids, callables, verbose=verbose
    )

    return nest, cp, grids, callables
