"""
Backward induction for the retirement choice model
(branching stage with discrete work/retire choice).

Three-functor dolo-plus pipeline:

  1. Parse     -- load stage YAML -> SymbolicModel
  2. Methodize -- attach solution method (EGM, UE, max)
  3. Configure -- bind structural/grid settings
  4. Calibrate -- bind economic parameter values

Then build operators and solve backward.
"""

import numpy as np
import time
import yaml
from pathlib import Path
from dolo.compiler.model import SymbolicModel
from dolo.compiler.calibration import calibrate as calibrate_stage
from dolo.compiler.calibration import configure as configure_stage
from dolo.compiler.methodization import methodize as methodize_stage
from .model import Operator_Factory, RetirementModel
from .model import (
    _default_u, _default_du, _default_uc_inv, _default_ddu,
)

def _load_twister(syntax_dir):
    """Load the inter-period twister from ``nest.yaml``.

    The twister maps cntn-perch poststates to arvl-perch
    prestates across period boundaries (spec 0.1h S3.3).

    Returns the first ``inter_connectors`` entry (all are
    identical for stationary models).
    """
    path = Path(syntax_dir) / "nest.yaml"

    # nest.yaml uses custom tags (!period) — add a
    # pass-through constructor so safe_load doesn't choke.
    class _Loader(yaml.SafeLoader):
        pass
    _Loader.add_multi_constructor(
        '', lambda loader, suffix, node:
        loader.construct_mapping(node)
        if isinstance(node, yaml.MappingNode)
        else None,
    )

    with open(path) as f:
        nest_yaml = yaml.load(f, Loader=_Loader)
    connectors = nest_yaml.get('inter_connectors', [])
    if connectors:
        return connectors[0]
    return {}


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


def _load_yaml(path):
    """Load a YAML file and return its contents."""
    with open(path) as f:
        return yaml.safe_load(f)


def instantiate_period(params, syntax_dir):
    """Build one period via the dolo-plus three-functor pipeline.

    For every stage declared in the period template,
    applies: parse -> methodize -> configure -> calibrate.

    Parameters
    ----------
    params : dict
        Merged calibration parameters. Only params declared
        in each stage's ``parameters:`` list are consumed by
        ``calibrate_stage``; extra keys are ignored.
    syntax_dir : str or Path
        Root syntax directory containing ``period.yaml``,
        ``settings.yaml``, and ``stages/``.

    Returns
    -------
    dict
        ``{<stage_name>: <calibrated SymbolicModel>, ...}``.
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
        # Functor 1: Methodize
        s = methodize_stage(
            s,
            str(stages_dir / name / f"{name}_methods.yml"),
        )
        # Functor 2: Configure
        s = configure_stage(s, settings_path)
        # Functor 3: Calibrate
        s = calibrate_stage(s, params)
        period[name] = s
    return period


def build_period_callables(period):
    """Return equation callables for a calibrated period.

    For the retirement model (log utility) the callables are
    the module-level defaults.  A future whisperer would
    compile these from the YAML equation strings instead.

    Parameters
    ----------
    period : dict
        Calibrated period from :func:`instantiate_period`
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

def _accrete_nest(
    T, params, syntax_dir, method='FUES',
    model=None, stage_ops=None,
):
    """Accretively build and solve the nest, one period at
    a time.

    NOTE: For stationary models, ``model`` and ``stage_ops``
    are built once (on the first period) and reused for all
    ``T`` periods.  If ``params`` is a schedule (list of
    dicts), operators are still frozen from period 0 -- this
    is an explicit stationarity assumption.

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
    model : RetirementModel, optional
        Pre-built model instance.
    stage_ops : dict, optional
        Pre-built stage operators.

    Returns
    -------
    nest : dict
        ``{"periods": [...], "twisters": [...],
        "solutions": [...]}``.
    model : RetirementModel
        The model instance used for solving.
    stage_ops : dict
        The stage operators used for solving.
    """
    nest = {"periods": [], "twisters": [], "solutions": []}
    is_schedule = isinstance(params, list)
    twister = _load_twister(syntax_dir)

    for h in range(T):
        t = T - 1 - h

        # 1. Build period via three-functor pipeline
        p = params[h] if is_schedule else params
        period = instantiate_period(p, syntax_dir)
        nest["periods"].append(period)
        nest["twisters"].append(None if h == 0 else twister)

        # 2. Build model + stage operators (once for stationary)
        if model is None or stage_ops is None:
            callables = build_period_callables(period)
            model = RetirementModel.from_period(
                period, equations=callables,
            )
            stage_ops = Operator_Factory(
                model, equations=callables,
            )

        # 3. Continuation
        if h == 0:
            # Terminal: consume everything, v = u(a)
            a = model.asset_grid_A
            cntn_retire = (
                np.copy(a),
                model.u(a),
                model.ddu(a) * model.R,
            )
            cntn_work = (
                model.du(a),
                model.ddu(a) * model.R,
                model.u(a),
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
        t_retire = time.time() - t0

        t1 = time.time()
        (v_work, c_work, da_work, ue_elapsed,
         c_hat, q_hat, egrid, da_pre_ue) = \
            stage_ops['work_cons'](
                *cntn_work, method=method,
            )
        t_work = time.time() - t1

        t2 = time.time()
        v_lmkt, c_lmkt, dv_lmkt, ddv_lmkt = \
            stage_ops['labour_mkt_decision'](
                v_work, v_retire,
                c_work, c_retire,
                da_work, da_retire,
            )
        t_lmkt = time.time() - t2

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
            "t_retire": t_retire,
            "t_work": t_work,
            "t_lmkt": t_lmkt,
        })

    return nest, model, stage_ops


# ============================================================
# Entry point
# ============================================================

def solve_nest(syntax_dir, method='FUES',
                    calib_overrides=None, config_overrides=None):
    """Canonical pipeline: load config, build nest, solve.

    Three-functor pipeline:

    1. Load ``calibration.yaml`` (economic params) and
       ``settings.yaml`` (numerical/structural settings).
    2. Apply overrides at the correct abstraction level.
    3. Build and solve the nest via
       :func:`_accrete_nest`.

    The ``method`` parameter (FUES/DCEGM/RFC/CONSAV) is a
    methodization concern -- it selects the upper-envelope
    algorithm.  Method overrides per stage are bound during
    the methodize functor in :func:`instantiate_period` via the
    ``*_methods.yml`` files.

    Parameters
    ----------
    syntax_dir : str or Path
        Root syntax directory containing ``calibration.yaml``,
        ``settings.yaml``, ``period.yaml``, and ``stages/``.
    method : str
        Upper-envelope method for the worker stage.
    calib_overrides : dict, optional
        Override economic parameters (e.g.
        ``{"beta": 0.96, "delta": 2}``).
    config_overrides : dict, optional
        Override numerical settings (e.g.
        ``{"grid_size": 5000, "T": 50}``).

    Returns
    -------
    nest : dict
        The solved nest.
    model : RetirementModel
        The model instance used for solving.
    stage_ops : dict
        The stage operators used for solving.
    """
    syntax_dir = Path(syntax_dir)

    # Load calibration (economic params)
    calibration = _load_yaml(
        syntax_dir / "calibration.yaml",
    )['calibration']
    if calib_overrides:
        calibration.update(calib_overrides)

    # Load settings (numerical/structural)
    settings = _load_yaml(
        syntax_dir / "settings.yaml",
    )['settings']
    if config_overrides:
        settings.update(config_overrides)

    # Merge into a single params dict for instantiate_period.
    # calibrate_stage only picks params declared in each
    # stage's parameters: list; extra keys are ignored.
    # Settings are consumed by configure_stage via the
    # settings.yaml file path.
    merged_params = {**calibration, **settings}

    T = int(settings.get('T', calibration.get('T', 20)))

    return _accrete_nest(
        T, merged_params, syntax_dir, method=method,
    )
