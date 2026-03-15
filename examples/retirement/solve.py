"""Backward induction for the retirement choice model.

Functional pipeline::

    load_syntax  →  instantiate_period  →  period_to_graph  →  backward_paths
                          ↓                                          ↓
                  RetirementModel.from_period                   solve_period
                          ↓                                     (wave loop)
                  make_stage_operators(method) ─────────────────────┘

Each step is a distinct transform with a different semantic
contract.  The backward loop is a thin combinator that iterates
over kikku's topology-derived wave ordering.
"""

import numpy as np
import time
from pathlib import Path

from kikku.period_graphs import period_to_graph, backward_paths
from kikku.pipeline import load_syntax, instantiate_period

from .model import RetirementModel
from .operators import make_stage_operators



# ============================================================
# Methodization override
# ============================================================

def _override_ue_method(stage_sources, method):
    """Return a copy of stage_sources with the UE method patched.

    Finds any ``upper_envelope`` scheme in any stage's methods
    and replaces its method tag with *method*.  Re-binding style:
    returns a new dict, does not mutate the input.
    """
    import copy
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
    """Extract continuation values from the previous period's solution."""
    lmkt = prev_sol['labour_mkt_decision']
    ret = prev_sol['retire_cons']
    return {
        'work_cons': (lmkt['dv'], lmkt['ddv'], lmkt['v']),
        'retire_cons': (ret['c'], ret['v'], ret['ddv']),
    }


# ============================================================
# Stage dispatch (application-specific, pure)
# ============================================================

def _solve_retire_cons(op, cntn, grid):
    c, v, da, ddv = op(*cntn, grid)
    return {"c": c, "v": v, "da": da, "ddv": ddv}


def _solve_work_cons(op, cntn, grid):
    (v, c, da, ue_elapsed,
     c_hat, q_hat, egrid, da_pre_ue) = op(*cntn, grid)
    return {
        "v": v, "c": c, "da": da,
        "c_hat": c_hat, "q_hat": q_hat,
        "egrid": egrid, "da_pre_ue": da_pre_ue,
        "ue_time": ue_elapsed,
    }


def _solve_labour_mkt(op, work_result, retire_result):
    v, c, dv, ddv = op(
        work_result['v'], retire_result['v'],
        work_result['c'], retire_result['c'],
        work_result['da'], retire_result['da'],
    )
    return {"v": v, "c": c, "dv": dv, "ddv": ddv}


# ============================================================
# Backward solve: solve_period + backward loop
# ============================================================

def solve_period(waves, stage_ops, cntns, grid, t, h):
    """Solve one period using kikku-derived wave ordering.

    Parameters
    ----------
    waves : list[list[str]]
        Backward-solve waves from ``kikku.backward_paths``.
    stage_ops : dict
        Stage operators from ``make_stage_operators``.
        Method and all config are already bound.
    cntns : dict
        Continuation values keyed by stage name.
    grid : ndarray
        Asset grid (passed to operators so they can be
        reused across grid sizes without JIT recompilation).
    t : int
        Calendar time index.
    h : int
        Horizon step (0 = terminal, T-1 = first).

    Returns
    -------
    dict
        Solution dict with per-stage results and timings.
    """
    dispatchers = {
        'retire_cons': lambda sr: _solve_retire_cons(
            stage_ops['retire_cons'], cntns['retire_cons'],
            grid,
        ),
        'work_cons': lambda sr: _solve_work_cons(
            stage_ops['work_cons'], cntns['work_cons'],
            grid,
        ),
        'labour_mkt_decision': lambda sr: _solve_labour_mkt(
            stage_ops['labour_mkt_decision'],
            sr['work_cons'], sr['retire_cons'],
        ),
    }

    t0 = time.time()
    stage_results = {}
    stage_timings = {}

    for wave in waves:
        for stage_name in wave:
            ts = time.time()
            stage_results[stage_name] = dispatchers[stage_name](
                stage_results,
            )
            stage_timings[stage_name] = time.time() - ts

    solve_time = time.time() - t0

    sol = {"t": t, "h": h}
    sol.update(stage_results)
    sol["ue_time"] = stage_results["work_cons"].get("ue_time", 0.0)
    sol["solve_time"] = solve_time
    sol["t_retire"] = stage_timings.get("retire_cons", 0.0)
    sol["t_work"] = stage_timings.get("work_cons", 0.0)
    sol["t_lmkt"] = stage_timings.get("labour_mkt_decision", 0.0)
    return sol


# ============================================================
# Entry point
# ============================================================

def solve_nest(syntax_dir, method='FUES',
               calib_overrides=None, config_overrides=None,
               model=None, stage_ops=None):
    """Canonical pipeline: load → build → solve backward.

    The composition algebra is visible::

        load_syntax → calibration, settings, template
        instantiate_period(calibration, settings, template)
            → period → graph → waves
            → model  → ops(method)
        for h: cntns + waves + ops → solve_period → solution

    For stationary models, pass back the returned ``model``
    and ``stage_ops`` to avoid re-building operators (and
    re-triggering JIT compilation) on subsequent calls.

    Parameters
    ----------
    syntax_dir : str or Path
        Root syntax directory.
    method : str
        Upper-envelope method (FUES/DCEGM/RFC/CONSAV).
        Bound at operator construction, not in the solve loop.
    calib_overrides : dict, optional
        Override economic parameters.
    config_overrides : dict, optional
        Override numerical settings.
    model : RetirementModel, optional
        Pre-built model (reuse to skip reconstruction).
    stage_ops : dict, optional
        Pre-built stage operators (reuse to skip JIT).

    Returns
    -------
    nest : dict
        ``{"solutions": [...]}``.
    model : RetirementModel
    stage_ops : dict
    """
    syntax_dir = Path(syntax_dir)

    # I/O + pipeline (skip entirely when reusing model + ops)
    if model is None or stage_ops is None:
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
        stage_ops['_waves'] = waves

    waves = stage_ops['_waves']
    T = int(model.T)

    # Backward induction (thin combinator)
    solutions = []
    for h in range(T):
        cntns = _terminal_continuations(model) if h == 0 \
            else _wire_continuations(solutions[h - 1])
        sol = solve_period(waves, stage_ops, cntns,
                           model.asset_grid_A, t=T - 1 - h, h=h)
        solutions.append(sol)

    nest = {"solutions": solutions}
    return nest, model, stage_ops
