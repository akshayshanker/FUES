"""
Backward induction for the retirement choice model
(branching stage with discrete work/retire choice).

Following the bellman-ddsl canonical pattern:

  Part A -- helpers for building a single period via the
            dolo-plus pipeline (load -> methodize ->
            configure -> calibrate).
  Part B -- accretively build the nest and solve it backward,
            one period at a time.  Period construction and
            solving happen in the same loop (no separate
            build-all-then-solve-all).
"""

import numpy as np
import time
import yaml
from pathlib import Path
from dolo.compiler.model import SymbolicModel
from dolo.compiler.calibration import calibrate as calibrate_stage
from dolo.compiler.calibration import configure as configure_stage
from dolo.compiler.methodization import methodize as methodize_stage
from .retirement import Operator_Factory, RetirementModel
from .retirement import (
    _default_u, _default_du, _default_uc_inv, _default_ddu,
)

# Inter-period twister (spec 0.1h S3.3): rename at period
# boundary.  Maps current period's poststate names to next
# period's prestate names.
#
# NOTE: this dict is stored on the nest as declarative
# metadata.  The actual continuation wiring is performed
# explicitly in the solve loop (step 3), not by a generic
# twister-application function.
INTER_PERIOD_TWISTER = {"b": "a", "b_ret": "a_ret"}


# ============================================================
# Part A -- Single-period construction helpers
# ============================================================

def _load_period_template(syntax_dir):
    """Read *period.yaml* and return the stage list.

    Parameters
    ----------
    syntax_dir : str or Path
        Root syntax directory (contains ``period.yaml``).

    Returns
    -------
    dict
        ``{"name": <period name>, "stages": [<stage names>]}``.
    """
    with open(Path(syntax_dir) / "period.yaml") as f:
        raw = yaml.safe_load(f)
    stages = []
    for entry in raw.get('stages', []):
        if isinstance(entry, dict):
            stages.extend(entry.keys())
        else:
            stages.append(str(entry))
    return {"name": raw["name"], "stages": stages}


def build_period(params, syntax_dir):
    """Build one period via the dolo-plus pipeline.

    For every stage declared in the period template
    (including the branching root ``labour_mkt_decision``
    and the two branch leaves ``work_cons``,
    ``retire_cons``), applies the dolo-plus pipeline:
    load YAML -> SymbolicModel -> methodize -> configure ->
    calibrate.

    Parameters
    ----------
    params : dict
        Calibration parameters (e.g. ``{r, beta, delta, ...}``).
    syntax_dir : str or Path
        Root syntax directory containing ``period.yaml``,
        ``settings.yaml``, and ``stages/``.

    Returns
    -------
    dict
        ``{<stage_name>: <calibrated SymbolicModel>, ...}``.
        One entry per stage in the period template.
    """
    syntax_dir = Path(syntax_dir)
    stages_dir = syntax_dir / "stages"
    settings_path = str(syntax_dir / "settings.yaml")
    template = _load_period_template(syntax_dir)

    period = {}
    for name in template["stages"]:
        path = stages_dir / name / f"{name}.yaml"
        with open(path) as f:
            s = SymbolicModel(
                yaml.compose(f.read()), filename=str(path),
            )
        s = methodize_stage(
            s,
            str(stages_dir / name / f"{name}_methods.yml"),
        )
        s = configure_stage(s, settings_path)
        s = calibrate_stage(s, params)
        period[name] = s
    return period


def build_period_callables(period):
    """Return equation callables for a calibrated period.

    For the retirement model (log utility) the callables are
    the module-level defaults.  A future whisperer would
    compile these from the YAML equation strings instead.

    NOTE: the *period* argument is currently unused -- the
    callables are hard-coded defaults.  It is accepted so
    the signature is ready for whisperer integration.

    Parameters
    ----------
    period : dict
        Calibrated period from :func:`build_period`
        (unused; reserved for future whisperer).

    Returns
    -------
    dict
        ``{u, du, uc_inv, ddu}`` -- each an ``@njit``
        callable.
    """
    return {
        'u': _default_u, 'du': _default_du,
        'uc_inv': _default_uc_inv, 'ddu': _default_ddu,
    }


# ============================================================
# Part B -- Accretive nest build + solve
# ============================================================

def build_and_solve_nest(
    T, params, syntax_dir, method='FUES',
    cp=None, stage_ops=None,
):
    """Accretively build and solve the nest, one period at
    a time.

    The period contains a branching stage
    (``labour_mkt_decision``) that routes to two branch
    leaves (``work_cons``, ``retire_cons``) via a discrete
    max aggregator.

    For each ``h = 0, 1, ..., T-1`` (distance from terminal,
    where ``t = T-1-h`` is calendar time / age):

    1. Build the period via the dolo-plus pipeline.
    2. Compile equation callables and build stage operators
       (only on the first call for stationary models).
    3. Assemble continuation from the previous solution.
       At ``h == 0`` the terminal condition is
       consume-everything: ``v = u(a)``, ``c = a``.
    4. Solve the period in reverse topological order
       (leaves first):
       ``retire_cons`` -> ``work_cons`` ->
       ``labour_mkt_decision``.
    5. Append the period and its solution to the nest.

    The two continuation chains have different orderings
    that match the stage operator signatures::

        cntn_retire: (c, v, ddv)   -- retiree stage expects
                                      consumption, value,
                                      second-order marginal
        cntn_work:   (dv, ddv, v)  -- worker stage expects
                                      marginal value,
                                      second-order marginal,
                                      value

    Parameters
    ----------
    T : int
        Number of periods (horizon).
    params : dict or list[dict]
        Single dict for stationary models, or a list indexed
        by ``h`` for lifecycle models.
    syntax_dir : str or Path
        Root syntax directory.
    method : str
        Upper-envelope method: ``FUES``, ``DCEGM``, ``RFC``,
        or ``CONSAV``.
    cp : RetirementModel, optional
        Pre-built model instance (reused across periods for
        stationary models).
    stage_ops : dict, optional
        Pre-built stage operators from
        :func:`Operator_Factory`.

    Returns
    -------
    nest : dict
        ``{"periods": [...], "twisters": [...],
        "solutions": [...]}``.
        Each list has length *T*, indexed by *h*.
        Each solution is a dict keyed by stage name
        (``retire_cons``, ``work_cons``,
        ``labour_mkt_decision``) with arrays for values,
        policies, and diagnostics.
    cp : RetirementModel
        The model instance used for solving.
    """
    nest = {"periods": [], "twisters": [], "solutions": []}
    is_schedule = isinstance(params, list)

    for h in range(T):
        t = T - 1 - h

        # 1. Build period
        p = params[h] if is_schedule else params
        period = build_period(p, syntax_dir)
        nest["periods"].append(period)
        nest["twisters"].append(
            None if h == 0 else INTER_PERIOD_TWISTER
        )

        # 2. Build stage operators
        if cp is None or stage_ops is None:
            callables = build_period_callables(period)
            cp = RetirementModel.from_period(
                period, equations=callables,
            )
            stage_ops = Operator_Factory(
                cp, equations=callables,
            )

        # 3. Continuation
        if h == 0:
            # Terminal: consume everything, v = u(a)
            a = cp.asset_grid_A
            cntn_retire = (
                np.copy(a),
                cp.u(a),
                cp.ddu(a) * cp.R,
            )
            cntn_work = (
                cp.du(a),
                cp.ddu(a) * cp.R,
                cp.u(a),
            )
        else:
            prev = nest["solutions"][h - 1]
            prev_lmkt = prev["labour_mkt_decision"]
            prev_ret = prev["retire_cons"]
            cntn_work = (
                prev_lmkt["dv"],
                prev_lmkt["ddv"],
                prev_lmkt["v"],
            )
            cntn_retire = (
                prev_ret["c"],
                prev_ret["v"],
                prev_ret["ddv"],
            )

        # 4. Solve: reverse topological (leaves first)
        t0 = time.time()

        c_retire, v_retire, da_retire, ddv_retire = \
            stage_ops['retire_cons'](*cntn_retire, t)

        (v_work, c_work, da_work, ue_elapsed,
         c_hat, q_hat, egrid, da_pre_ue) = \
            stage_ops['work_cons'](
                *cntn_work, method=method,
            )

        v_lmkt, c_lmkt, dv_lmkt, ddv_lmkt = \
            stage_ops['labour_mkt_decision'](
                v_work, v_retire,
                c_work, c_retire,
                da_work, da_retire,
            )

        solve_time = time.time() - t0

        # 5. Store solution
        nest["solutions"].append({
            "t": t, "h": h,
            "retire_cons": {
                "c": c_retire, "v": v_retire,
                "da": da_retire, "ddv": ddv_retire,
            },
            "work_cons": {
                "v": v_work, "c": c_work, "da": da_work,
                "c_hat": c_hat, "q_hat": q_hat,
                "egrid": egrid, "da_pre_ue": da_pre_ue,
            },
            "labour_mkt_decision": {
                "v": v_lmkt, "c": c_lmkt,
                "dv": dv_lmkt, "ddv": ddv_lmkt,
            },
            "ue_time": ue_elapsed,
            "solve_time": solve_time,
        })

    return nest, cp


# ============================================================
# Entry points
# ============================================================

def backward_induction(cp, stage_ops, syntax_dir,
                       method='FUES'):
    """Build and solve a nest from a pre-built model.

    Extracts calibration from *cp*, accretively builds the
    nest via the dolo-plus pipeline, and solves it using
    the provided stage operators.

    Parameters
    ----------
    cp : RetirementModel
        Model instance with calibrated parameters and grids.
    stage_ops : dict
        Stage operators from :func:`Operator_Factory`
        (keys: ``retire_cons``, ``work_cons``,
        ``labour_mkt_decision``).
    syntax_dir : str or Path
        Root syntax directory.
    method : str
        Upper-envelope method.

    Returns
    -------
    dict
        The full nest dict with keys ``periods``,
        ``twisters``, ``solutions``.
    """
    params = {
        'r': cp.r, 'beta': cp.beta, 'delta': cp.delta,
        'smooth_sigma': cp.smooth_sigma,
        'y': cp.y, 'b': cp.b,
        'grid_max_A': cp.grid_max_A,
        'grid_size': cp.grid_size, 'T': cp.T,
    }
    nest, _ = build_and_solve_nest(
        cp.T, params, syntax_dir,
        method=method, cp=cp, stage_ops=stage_ops,
    )
    return nest


def solve_canonical(syntax_dir, params_override=None,
                    method='FUES'):
    """Canonical pipeline: load YAML, build nest, solve.

    Reads ``calibration.yaml``, ``settings.yaml``, and
    ``period.yaml`` from *syntax_dir*, merges any parameter
    overrides, then accretively builds and solves the nest
    via the dolo-plus pipeline.

    Parameters
    ----------
    syntax_dir : str or Path
        Root syntax directory containing ``calibration.yaml``,
        ``settings.yaml``, ``period.yaml``, and ``stages/``.
    params_override : dict, optional
        Override calibration values (e.g.
        ``{"grid_size": 3000, "delta": 2}``).
    method : str
        Upper-envelope method.

    Returns
    -------
    nest : dict
        The solved nest.
    cp : RetirementModel
        The model instance used for solving.
    """
    syntax_dir = Path(syntax_dir)
    with open(syntax_dir / "calibration.yaml") as f:
        base_params = yaml.safe_load(f)['calibration']
    if params_override:
        base_params.update(params_override)

    with open(syntax_dir / "settings.yaml") as f:
        settings = yaml.safe_load(f)['settings']

    T = base_params.get('T', settings.get('T', 20))
    return build_and_solve_nest(
        T, base_params, syntax_dir, method=method,
    )
