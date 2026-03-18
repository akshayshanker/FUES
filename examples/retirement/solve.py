"""Backward induction for the retirement choice model.

Functional pipeline::

    load_syntax → instantiate_period → period_to_graph → backward_paths
                        ↓                                       ↓
                  RetirementModel(period)                  solve_backward
                        ↓                                  (wave loop)
                  make_stage_operators ─────────────────────────┘

Each step is a distinct transform with a different semantic
contract.  The backward loop is a thin combinator that iterates
over kikku's topology-derived wave ordering.
"""

import copy
import numpy as np
import time
from pathlib import Path

from kikku.dynx import period_to_graph, backward_paths
from kikku.dynx import load_syntax, instantiate_period

from .model import RetirementModel
from .operators import make_stage_operators


# ============================================================
# Methodization override (re-binding, no mutation)
# ============================================================

def _override_ue_method(stage_sources, method):
    """Return a copy of *stage_sources* with the UE method patched."""
    patched = copy.deepcopy(stage_sources)
    for src in patched.values():
        methods = src.get("methods", {})
        for entry in methods.get("methods", []):
            for scheme in entry.get("schemes", []):
                if scheme.get("scheme") == "upper_envelope":
                    scheme["method"] = {
                        "__yaml_tag__": method.upper(),
                        "value": "",
                    }
    return patched


# ============================================================
# Continuation wiring (application-specific, pure)
# ============================================================

def _terminal_continuations(model):
    """Terminal-period continuation values (consume everything)."""
    a = model.asset_grid_A
    return {
        'retire_cons': (
            np.copy(a), model.u(a), model.ddu(a) * model.R,
        ),
        'work_cons': (
            model.du(a), model.ddu(a) * model.R, model.u(a),
        ),
    }


def _wire_continuations(prev_sol):
    """Extract continuation values from previous period's solution."""
    lmkt = prev_sol['labour_mkt_decision']
    ret = prev_sol['retire_cons']
    return {
        'work_cons': (lmkt['dv'], lmkt['ddv'], lmkt['v']),
        'retire_cons': (ret['c'], ret['v'], ret['ddv']),
    }


# ============================================================
# Stage dispatch (pure: inputs in, result out)
# ============================================================

def _run_retire_cons(op, cntn, grid):
    c, v, da, ddv = op(*cntn, grid)
    return {"c": c, "v": v, "da": da, "ddv": ddv}


def _run_work_cons(op, cntn, grid):
    (v, c, da, ue_elapsed,
     c_dcsn_hat, v_dcsn_hat, x_dcsn_hat, dela_dcsn_hat) = op(*cntn, grid)
    return {
        "v": v, "c": c, "da": da,
        "c_dcsn_hat": c_dcsn_hat, "v_dcsn_hat": v_dcsn_hat,
        "x_dcsn_hat": x_dcsn_hat, "dela_dcsn_hat": dela_dcsn_hat,
        "ue_time": ue_elapsed,
    }


def _run_labour_mkt(op, work_result, retire_result):
    v, c, dv, ddv = op(
        work_result['v'], retire_result['v'],
        work_result['c'], retire_result['c'],
        work_result['da'], retire_result['da'],
    )
    return {"v": v, "c": c, "dv": dv, "ddv": ddv}


_STAGE_RUNNERS = {
    'retire_cons': lambda ops, cntns, grid, sr: _run_retire_cons(
        ops['retire_cons'], cntns['retire_cons'], grid),
    'work_cons': lambda ops, cntns, grid, sr: _run_work_cons(
        ops['work_cons'], cntns['work_cons'], grid),
    'labour_mkt_decision': lambda ops, cntns, grid, sr: _run_labour_mkt(
        ops['labour_mkt_decision'],
        sr['work_cons'], sr['retire_cons']),
}


# ============================================================
# Backward solve
# ============================================================

def solve_period(waves, stage_ops, cntns, grid, t, h):
    """Solve one period using kikku-derived wave ordering."""
    t0 = time.time()
    sr = {}
    timings = {}

    for wave in waves:
        for name in wave:
            ts = time.time()
            sr[name] = _STAGE_RUNNERS[name](
                stage_ops, cntns, grid, sr,
            )
            timings[name] = time.time() - ts

    solve_time = time.time() - t0

    return {
        "t": t, "h": h,
        **sr,
        "ue_time": sr["work_cons"].get("ue_time", 0.0),
        "solve_time": solve_time,
        "t_retire": timings.get("retire_cons", 0.0),
        "t_work": timings.get("work_cons", 0.0),
        "t_lmkt": timings.get("labour_mkt_decision", 0.0),
    }


def solve_backward(T, model, stage_ops, waves):
    """Run backward induction over T periods.

    Pure combinator: no I/O, no pipeline, just the loop.
    """
    grid = model.asset_grid_A
    solutions = []
    for h in range(T):
        cntns = _terminal_continuations(model) if h == 0 \
            else _wire_continuations(solutions[h - 1])
        sol = solve_period(waves, stage_ops, cntns,
                           grid, t=T - 1 - h, h=h)
        solutions.append(sol)
    return solutions


# ============================================================
# Entry point
# ============================================================

def solve_nest(syntax_dir, method='FUES',
               calib_overrides=None, config_overrides=None,
               model=None, stage_ops=None, waves=None):
    """Canonical pipeline: load → build → solve backward.

    Composition algebra::

        load_syntax → calibration, settings, sources, template
        _override_ue_method(sources, method)
        instantiate_period → period
        period_to_graph → graph → backward_paths → waves
        RetirementModel(period) → model
        make_stage_operators(model, period) → ops
        solve_backward(T, model, ops, waves) → solutions

    For stationary models, pass back ``model``, ``stage_ops``,
    and ``waves`` to skip the pipeline on subsequent calls.

    Parameters
    ----------
    syntax_dir : str or Path
        Root syntax directory.
    method : str
        Upper-envelope method (FUES/DCEGM/RFC/CONSAV).
    calib_overrides, config_overrides : dict, optional
        Sparse overrides.
    model : RetirementModel, optional
        Reuse to skip reconstruction.
    stage_ops : dict, optional
        Reuse to skip JIT recompilation.
    waves : list[list[str]], optional
        Reuse to skip graph construction.

    Returns
    -------
    nest : dict
    model : RetirementModel
    stage_ops : dict
    waves : list[list[str]]
    """
    syntax_dir = Path(syntax_dir)

    if model is None or stage_ops is None or waves is None:
        calibration, settings, stage_sources, \
            period_template, inter_conn = load_syntax(
                syntax_dir, calib_overrides, config_overrides,
            )
        stage_sources = _override_ue_method(stage_sources, method)
        period = instantiate_period(
            calibration, settings, stage_sources, period_template,
        )
        graph = period_to_graph(period)
        waves = backward_paths(graph, inter_conn)
        if model is None:
            model = RetirementModel(period)
        if stage_ops is None:
            stage_ops = make_stage_operators(model, period=period)

    solutions = solve_backward(int(model.T), model, stage_ops, waves)

    return {"solutions": solutions}, model, stage_ops, waves
