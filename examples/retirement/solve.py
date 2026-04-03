"""Backward induction for the retirement choice model.

Functional pipeline::

    load_syntax → instantiate_period → period_to_graph → backward_paths
                        ↓                                       ↓
                  RetirementModel(period)                  solve_backward
                        ↓                                  (wave loop)
                  make_retire_cons(model, callables)             │
                  make_work_cons(model, callables, ue_method)    │
                  make_labour_mkt_decision(model) ──────────────┘

Each step is a distinct transform with a different semantic
contract.  The backward loop is a thin combinator that iterates
over kikku's topology-derived wave ordering.  Operator composition
T = I ∘ B is visible inline in solve_period.
"""

import numpy as np
import time
from pathlib import Path

from kikku.dynx import period_to_graph, backward_paths
from kikku.dynx import load_syntax, instantiate_period
from kikku.dynx.methods import override_methods

from .model import RetirementModel
from .operators import make_retire_cons, make_work_cons, make_labour_mkt_decision
from .model import make_worker_egm_fns, make_retiree_egm_fns


# ============================================================
# Method shortcut (--method expands before instantiate_period)
# ============================================================

METHOD_SHORTCUT = [
    ('work_cons', 'cntn_to_dcsn_mover', 'upper_envelope'),
    ('retire_cons', 'cntn_to_dcsn_mover', 'upper_envelope'),
    ('labour_mkt_decision', 'cntn_to_dcsn_mover', 'upper_envelope'),
]


# ============================================================
# Syntax helpers
# ============================================================

def read_scheme_method(stage, scheme_name,
                       mover='cntn_to_dcsn_mover',
                       default='FUES'):
    """Read the method tag for a scheme from an instantiated stage.

    Inspects ``stage.methods[mover]['schemes']`` for a scheme
    entry matching ``scheme_name`` and returns its ``__yaml_tag__``.
    """
    if not hasattr(stage, 'methods'):
        return default
    mover_dict = stage.methods.get(mover, {})
    for scheme in mover_dict.get('schemes', []):
        if scheme.get('scheme') == scheme_name:
            tag = scheme.get('method', {})
            if isinstance(tag, dict):
                return tag.get('__yaml_tag__', default)
            return str(tag)
    return default


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
# Backward solve
# ============================================================

def solve_period(waves, stage_ops, cntns, grid, t, h):
    """Solve one period using kikku-derived wave ordering.

    Operator composition is inline — each stage's T = I ∘ B
    is visible at the call site.
    """
    t0 = time.time()
    sr = {}
    timings = {}

    for wave in waves:
        for name in wave:
            ts = time.time()

            if name == 'retire_cons':
                B = stage_ops['retire_cons']['dcsn_mover']
                I = stage_ops['retire_cons']['arvl_mover']
                x, v, c, da = B(*cntns['retire_cons'], grid)
                c_a, v_a, da_a, ddv = I(
                    x, v, c, da, grid, cntns['retire_cons'][1][0])
                sr[name] = {"c": c_a, "v": v_a, "da": da_a, "ddv": ddv}

            elif name == 'work_cons':
                B = stage_ops['work_cons']['dcsn_mover']
                I = stage_ops['work_cons']['arvl_mover']
                (x, v, c, da, ue_time,
                 c_hat, v_hat, x_hat, da_hat) = B(
                    *cntns['work_cons'], grid)
                v_a, c_a, da_a = I(x, v, c, cntns['work_cons'][2], grid)
                sr[name] = {
                    "v": v_a, "c": c_a, "da": da_a,
                    "c_dcsn_hat": c_hat, "v_dcsn_hat": v_hat,
                    "x_dcsn_hat": x_hat, "dela_dcsn_hat": da_hat,
                    "ue_time": ue_time,
                }

            elif name == 'labour_mkt_decision':
                B = stage_ops['labour_mkt_decision']['dcsn_mover']
                v, c, dv, ddv = B(
                    sr['work_cons']['v'], sr['retire_cons']['v'],
                    sr['work_cons']['c'], sr['retire_cons']['c'],
                    sr['work_cons']['da'], sr['retire_cons']['da'],
                )
                sr[name] = {"v": v, "c": c, "dv": dv, "ddv": ddv}

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

def solve_nest(syntax_dir, method_overrides=None,
               calib_overrides=None, setting_overrides=None,
               model=None, stage_ops=None, waves=None,
               # Deprecated — use method_overrides + setting_overrides
               method=None, config_overrides=None):
    """Canonical pipeline: load → build → solve backward.

    Composition algebra::

        load_syntax → calibration, settings, sources, template
        override_methods (shortcut + explicit overrides)
        instantiate_period → period
        period_to_graph → graph → backward_paths → waves
        RetirementModel(period) → model
        make_retire_cons(model, callables) → {dcsn_mover, arvl_mover}
        make_work_cons(model, callables, ue_method) → {dcsn_mover, arvl_mover}
        make_labour_mkt_decision(model) → {dcsn_mover}
        solve_backward(T, model, ops, waves) → solutions

    For stationary models, pass back ``model``, ``stage_ops``,
    and ``waves`` to skip the pipeline on subsequent calls.

    Parameters
    ----------
    syntax_dir : str or Path
        Root syntax directory.
    method : str or None
        Upper-envelope method (FUES/DCEGM/RFC/CONSAV). Expanded to
        ``METHOD_SHORTCUT`` before instantiation. If ``None``, only
        ``method_overrides`` and YAML defaults apply.
    calib_overrides, config_overrides : dict, optional
        Sparse overrides.
    model : RetirementModel, optional
        Reuse to skip reconstruction.
    stage_ops : dict, optional
        Reuse to skip JIT recompilation.
    waves : list[list[str]], optional
        Reuse to skip graph construction.
    method_overrides : dict, optional
        ``{(stage, target, scheme): tag, ...}`` merged after the
        ``method`` shortcut; passed to ``override_methods``.

    Returns
    -------
    nest : dict
    model : RetirementModel
    stage_ops : dict
    waves : list[list[str]]
    """
    syntax_dir = Path(syntax_dir)

    # Backward compat: config_overrides → setting_overrides
    if config_overrides is not None and setting_overrides is None:
        setting_overrides = config_overrides

    if model is None or stage_ops is None or waves is None:
        calibration, settings, stage_sources, \
            period_template, inter_conn = load_syntax(
                syntax_dir, calib_overrides, setting_overrides,
            )
        # method= is deprecated — callers should use method_overrides
        all_method_overrides = {}
        if method is not None:
            for target in METHOD_SHORTCUT:
                all_method_overrides[target] = method
        if method_overrides:
            all_method_overrides.update(method_overrides)
        if all_method_overrides:
            stage_sources = override_methods(stage_sources, all_method_overrides)
        period = instantiate_period(
            calibration, settings, stage_sources, period_template,
        )
        graph = period_to_graph(period)
        waves = backward_paths(graph, inter_conn)
        if model is None:
            model = RetirementModel(period)
        if stage_ops is None:
            stages_dict = period["stages"] if "stages" in period else period
            work_stage = stages_dict["work_cons"]
            ue = read_scheme_method(work_stage, 'upper_envelope')
            beta, R = model.beta, model.R
            delta, y = model.delta, model.y
            stage_ops = {
                'retire_cons': make_retire_cons(
                    model, make_retiree_egm_fns(beta, R, delta, y)),
                'work_cons': make_work_cons(
                    model, make_worker_egm_fns(beta, R, delta, y),
                    ue_method=ue),
                'labour_mkt_decision': make_labour_mkt_decision(model),
            }

    solutions = solve_backward(int(model.T), model, stage_ops, waves)

    return {"solutions": solutions}, model, stage_ops, waves
