"""Backward induction for the retirement choice model.

Functional pipeline::

    spec_factory → make → period_to_graph → backward_paths
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
from collections.abc import Mapping

from dolo.compiler.period_factory import make as make_period
from dolo.compiler.period_factory import period_to_graph
from dolo.compiler.nest_factory import backward_paths
from dolo.compiler.spec_factory import load as load_spec, make as make_spec

from .model import RetirementModel
from .operators import make_retire_cons, make_work_cons, make_labour_mkt_decision
from .model import make_worker_egm_fns, make_retiree_egm_fns


# ============================================================
# Method shortcut (--method expands before make)
# ============================================================

# Under the kernel/policy schema (spec 0.3) + no-schemes methodization
# (spec 0.1d.1), the upper-envelope method attaches to the ``upper_env`` node
# (it feeds ``evaluate``), not the old ``cntn_to_dcsn_mover`` builder. Only
# ``work_cons`` carries a real upper-envelope operator (it sits below the
# work/retire branch); ``retire_cons`` (smooth EGM) and ``labour_mkt_decision``
# (pure discrete choice) have no ``upper_env`` node, so the ``$method_switch``
# slot targets ``work_cons.upper_env`` alone.
METHOD_SHORTCUT = [
    ('work_cons', 'upper_env'),
]


def expand_method_shortcut(
    tag: str, shortcut: list[tuple[str, str]]
) -> dict:
    """Build a ``$method_switch`` slot value in the no-schemes shape.

    Emits the authored operator-slot surface (spec 0.1d.1) — one
    ``{on: <node>, method: <tag>}`` entry per target node, with NO ``schemes:``
    list and NO legacy ``scheme:`` name — then normalizes it to the internal
    ``schemes:``-list shape via the same ``methodization._normalize_methods``
    pass the authored ``*_methods.yml`` files go through. Normalizing here keeps
    the slot consistent with the (already-normalized) base ``upper_env`` entry it
    merges onto, so the keyed-list merge by ``on:`` overrides the method tag.
    """
    from dolo.compiler.methodization import _normalize_methods

    methods = [{"on": on_node, "method": tag} for _stage, on_node in shortcut]
    return _normalize_methods({"methods": methods})


def _prepare_method_slot(method_switch) -> object | None:
    if method_switch is None:
        return None
    if isinstance(method_switch, str):
        return expand_method_shortcut(method_switch, METHOD_SHORTCUT)
    if isinstance(method_switch, Mapping):
        if any(isinstance(k, tuple) for k in method_switch):
            raise TypeError("tuple keys in method_switch are not supported in v3")
        return dict(method_switch)
    raise TypeError(
        f"method_switch must be None, str, or dict; got {type(method_switch).__name__}"
    )


# ============================================================
# Syntax helpers
# ============================================================

def read_node_method(stage, node, default='FUES'):
    """Read the method tag bound to a named methodization ``node``.

    ``stage.methods`` is keyed by the methodization *node* (the ``on:`` field):
    ``upper_env`` (upper envelope), ``policy`` (the EGM/VFI/max selection),
    ``evaluate`` (interpolation), ``argmax_*`` (direct maximisation), etc. This
    reads the method tag for one node directly — by node, not by legacy scheme
    name (spec 0.1d.1).

    A legacy scheme name (``upper_envelope``, ``maximization``, …) is accepted
    and mapped to its node via ``methodization._NODE_TO_SCHEME_*`` (inverse), so
    existing call sites keep working. The no-schemes override surface stores the
    tag either at the node's top level (``method:``) or inside a normalized
    ``schemes:`` block; both are read here.
    """
    if not hasattr(stage, 'methods'):
        return default

    methods = stage.methods

    # Resolve a legacy scheme name to the node(s) that carry it.
    node_names = _nodes_for_request(node, methods)

    def _tag(value):
        if isinstance(value, Mapping):
            return value.get('__yaml_tag__', default)
        return str(value)

    for nm in node_names:
        entry = methods.get(nm)
        if not isinstance(entry, Mapping):
            continue
        # No-schemes surface: the tag lives at the node's top level.
        if 'method' in entry:
            return _tag(entry['method'])
        # Normalized internal surface: tag inside the single scheme block.
        for block in entry.get('schemes', []):
            if 'method' in block:
                return _tag(block['method'])

    return default


def _nodes_for_request(request, methods):
    """Yield the node name(s) addressed by ``request``.

    ``request`` may be a node name (used directly) or a legacy scheme name,
    which maps to every node whose ``methodization`` scheme matches.
    """
    if request in methods:
        return [request]
    from dolo.compiler.methodization import _scheme_for_node
    return [nm for nm in methods if _scheme_for_node(nm) == request] or [request]


# Backwards-compatible alias: callers pass either a node name or a legacy
# scheme name; the resolver maps a scheme name to its node internally.
def read_scheme_method(stage, scheme_name, mover=None, default='FUES'):
    return read_node_method(stage, scheme_name, default=default)


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

def solve_nest(
    registry_dir,
    spec_factory_name="spec_factory.yaml",
    draw=None,
    method_switch=None,
    model=None,
    stage_ops=None,
    waves=None,
    **more_slots,
):
    """Canonical retirement pipeline: spec_factory → build → solve backward.

    Composition algebra::

        spec_factory.load → recipe
        spec_factory.make(recipe, draw, method_switch) → SpecGraph
        period_factory.make(spec, 0, template, sym_stages) → period
        period_to_graph → graph → backward_paths → waves
        RetirementModel(period) → model
        make_retire_cons/work_cons/labour_mkt_decision → stage_ops
        solve_backward(T, model, ops, waves) → solutions

    For stationary models, pass back ``model``, ``stage_ops``,
    and ``waves`` to skip the pipeline on subsequent calls.

    Parameters
    ----------
    registry_dir : str or Path
        Path to the model registry (e.g. 'examples/retirement/syntax').
    spec_factory_name : str
        Name of the spec_factory YAML file.
    draw : dict, optional
        Tier-wrapped overrides for the `$draw` slot:
        `{'calibration': {...}, 'settings': {...}}`.
    method_switch : str or dict, optional
        v3: string tags expand via ``expand_method_shortcut``; nested
        ``{methods: [...]}`` dicts pass through. Tuple keys are not supported.
    model : RetirementModel, optional
        Reuse to skip reconstruction.
    stage_ops : dict, optional
        Reuse to skip JIT recompilation.
    waves : list[list[str]], optional
        Reuse to skip graph construction.
    more_slots
        Additional spec_factory slot bindings (e.g. ``**t.slots`` from kikku).
        Reserved names for this function: do not use ``model``, ``stage_ops``,
        ``waves`` as slot tags; they are solver kwargs.

    Returns
    -------
    nest : dict
    model : RetirementModel
    stage_ops : dict
    waves : list[list[str]]
    """
    from pathlib import Path as _Path
    from dolo.compiler.tag_tolerant_yaml import load_yaml_tag_tolerant
    from dolo.compiler.nest_factory.loader import load_inter_connector

    registry_dir = _Path(registry_dir)

    if not (registry_dir / spec_factory_name).exists():
        raise FileNotFoundError(
            f"No {spec_factory_name} in {registry_dir}. "
            f"Every retirement registry must provide a spec_factory YAML; "
            f"migrate legacy registries by mirroring examples/retirement/syntax/."
        )

    sb: dict = {k: v for k, v in more_slots.items()}
    if draw is not None:
        sb["draw"] = draw
    if method_switch is not None:
        sb["method_switch"] = _prepare_method_slot(method_switch)
    elif sb.get("method_switch") is not None:
        sb["method_switch"] = _prepare_method_slot(sb["method_switch"])

    if model is None or stage_ops is None or waves is None:
        recipe = load_spec(str(registry_dir / spec_factory_name))
        spec = make_spec(
            recipe, registry_dir=str(registry_dir), **sb
        )

        # Load sym stage sources
        stage_names = list(recipe.stages.keys())
        stages_dir = registry_dir / "stages"
        sym_stages = {}
        for name in stage_names:
            stage_yaml_path = stages_dir / name / f"{name}.yaml"
            with open(stage_yaml_path) as f:
                yaml_text = f.read()
            sym_stages[name] = {
                "yaml_text": yaml_text,
                "yaml_path": str(stage_yaml_path),
            }

        # Load period template
        period_raw = load_yaml_tag_tolerant(registry_dir / "period.yaml")
        tmpl_stage_names = []
        for entry in period_raw.get('stages', []):
            if isinstance(entry, dict):
                tmpl_stage_names.extend(entry.keys())
            else:
                tmpl_stage_names.append(str(entry))
        period_template = {
            "name": period_raw.get("name", "retirement"),
            "stages": tmpl_stage_names,
            "connectors": period_raw.get("connectors", []),
        }

        # Build period via 0.1s path (SpecGraph)
        period = make_period(spec, 0, period_template, sym_stages)

        # Derive topology
        inter_conn = load_inter_connector(registry_dir)
        graph = period_to_graph(period)
        waves = backward_paths(graph, inter_conn)

        if model is None:
            model = RetirementModel(period)

        if stage_ops is None:
            stages_dict = period["stages"]
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


