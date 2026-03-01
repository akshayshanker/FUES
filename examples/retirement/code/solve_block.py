"""
Backward Induction via Stage Accretion: retirement choice (branching)

Period structure (branching DAG):
  labour_mkt_decision (branching, max)
    |-- work   -> work_cons   (WorkerConsumption)  -> b
    +-- retire -> retire_cons (RetireeConsumption)  -> b_ret

Inter-period connector: {b: a, b_ret: a_ret}
  - b     -> a     (worker output -> branching stage arrival)
  - b_ret -> a_ret (retiree output -> retire_cons direct entry)

Topological solve order (leaves first):
  1. retire_cons  (leaf -- retiree consumption, EGM)
  2. work_cons    (leaf -- worker consumption, EGM + UE)
  3. labour_mkt_decision  (root -- branching max/logit)

Stage solvers (primitives) are built by Operator_Factory in retirement.py.
This module provides the backward_induction combinator that wires them.
"""

import numpy as np
import time
from pathlib import Path

from .retirement import Operator_Factory, RetirementModel, euler

# ---------------------------------------------------------------------------
# Paths for YAML stage definitions (structural reference)
# ---------------------------------------------------------------------------

_EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
STAGES_PATH = _EXAMPLE_ROOT / "syntax" / "syntax" / "stages"
MODEL_PATH = _EXAMPLE_ROOT / "syntax" / "syntax"

# ---------------------------------------------------------------------------
# YAML/dolo structural layer (requires dolo-plus compiler)
# ---------------------------------------------------------------------------

try:
    import yaml
    from dolo.compiler.model import SymbolicModel
    from dolo.compiler.calibration import calibrate as calibrate_stage, configure as configure_stage
    from dolo.compiler.methodization import methodize as methodize_stage
    _HAS_DOLO = True
except ImportError:
    _HAS_DOLO = False


def build_syntactic_stage(name: str):
    """Load stage syntax from YAML file (requires dolo-plus)."""
    if not _HAS_DOLO:
        raise ImportError("dolo-plus compiler not available")
    path = STAGES_PATH / name / f"{name}.yaml"
    with open(path, 'r') as f:
        return SymbolicModel(yaml.compose(f.read()), filename=str(path))


def build_period(params: dict) -> dict:
    """Build a period instance containing the three stages.

    Pipeline per stage: load YAML -> methodize -> configure -> calibrate.
    Requires dolo-plus compiler.
    """
    if not _HAS_DOLO:
        raise ImportError("dolo-plus compiler not available")

    settings_path = str(MODEL_PATH / "settings.yaml")

    # Worker Decision (branching stage)
    wd_syntax = build_syntactic_stage("worker_decision")
    wd_M = methodize_stage(wd_syntax, str(STAGES_PATH / "worker_decision" / "worker_decision_methods.yml"))
    wd_cfg = configure_stage(wd_M, settings_path)
    wd_calib = calibrate_stage(wd_cfg, params)

    # Worker Consumption (work branch)
    wc_syntax = build_syntactic_stage("worker_consumption")
    wc_M = methodize_stage(wc_syntax, str(STAGES_PATH / "worker_consumption" / "worker_consumption_methods.yml"))
    wc_cfg = configure_stage(wc_M, settings_path)
    wc_calib = calibrate_stage(wc_cfg, params)

    # Retiree Consumption (retire branch)
    rc_syntax = build_syntactic_stage("retiree_consumption")
    rc_M = methodize_stage(rc_syntax, str(STAGES_PATH / "retiree_consumption" / "retiree_consumption_methods.yml"))
    rc_cfg = configure_stage(rc_M, settings_path)
    rc_calib = calibrate_stage(rc_cfg, params)

    return {
        "labour_mkt_decision": wd_calib,
        "work_cons": wc_calib,
        "retire_cons": rc_calib,
    }


# ============================================================================
# Backward induction combinator
# ============================================================================

def backward_induction(cp, movers, method='FUES'):
    """Solve the retirement model by backward induction.

    Single loop in topological order: retiree leaf, worker leaf,
    branching root.  Two explicit inter-period connectors carry
    state between periods.

    Parameters
    ----------
    cp : RetirementModel
        Model instance with calibrated parameters and grids.
    movers : dict
        ``{'retiree': solver_retiree_stage, 'worker': solver_worker_stage,
           'branch': lab_mkt_choice_stage}``
    method : str
        Upper-envelope method passed to the worker solver.

    Returns
    -------
    tuple (7 elements)
        worker_endog_grid, worker_unrefined_values,
        worker_refined_values, worker_unrefined_consumption,
        worker_refined_consumption, asset_pol_derivative_unrefined,
        average_times
    """
    T = cp.T
    grid_size = cp.grid_size
    asset_grid_A = cp.asset_grid_A

    # Terminal conditions
    initial_c = np.copy(asset_grid_A)
    initial_v = cp.u(asset_grid_A)
    initial_dlambda = cp.ddu(asset_grid_A) * cp.R
    initial_lambda = cp.du(asset_grid_A)

    # Diagnostic storage
    worker_endog_grid = np.empty((T, grid_size))
    worker_unrefined_values = np.empty((T, grid_size))
    worker_refined_values = np.empty((T, grid_size))
    worker_unrefined_consumption = np.empty((T, grid_size))
    worker_refined_consumption = np.empty((T, grid_size))
    asset_pol_derivative_unrefined = np.empty((T, grid_size))

    UE_times = np.zeros(T)
    all_times = np.zeros(T)

    # Retiree chain connector: (c, v, dlambda)
    ret_c_next = np.copy(initial_c)
    ret_v_next = np.copy(initial_v)
    ret_dlambda_next = np.copy(initial_dlambda)

    # Mixed chain connector: (lambda, dlambda, v)
    mix_lambda_next = np.copy(initial_lambda)
    mix_dlambda_next = np.copy(initial_dlambda)
    mix_v_next = np.copy(initial_v)

    for i in range(T):
        t = T - 1 - i

        time_start = time.time()

        # Stage 1: retiree leaf
        c_ret, v_ret, da_ret, dlambda_ret = movers['retiree'](
            ret_c_next, ret_v_next, ret_dlambda_next, t)

        # Stage 2: worker leaf
        v_work, c_work, da_work, ue_time, \
            cons_hat, q_hat, egrid, da_unref = movers['worker'](
                mix_lambda_next, mix_dlambda_next, mix_v_next, method=method)

        # Stage 3: branching root
        v_mix, c_mix, lambda_mix, dlambda_mix = movers['branch'](
            v_work, v_ret, c_work, c_ret, da_work, da_ret)

        time_total = time.time() - time_start

        # Store diagnostics
        worker_endog_grid[t, :] = egrid
        worker_unrefined_values[t, :] = q_hat
        worker_refined_values[t, :] = v_work
        worker_unrefined_consumption[t, :] = cons_hat
        worker_refined_consumption[t, :] = c_mix
        asset_pol_derivative_unrefined[t, :] = da_unref

        if i > 2:
            UE_times[t] = ue_time
            all_times[t] = time_total

        # Update connectors
        ret_c_next, ret_v_next, ret_dlambda_next = c_ret, v_ret, dlambda_ret
        mix_lambda_next, mix_dlambda_next, mix_v_next = lambda_mix, dlambda_mix, v_mix

    mask = UE_times > 0
    average_times = [
        np.mean(UE_times[mask]) if np.any(mask) else 0.0,
        np.mean(all_times[mask]) if np.any(mask) else 0.0,
    ]

    return (worker_endog_grid, worker_unrefined_values, worker_refined_values,
            worker_unrefined_consumption, worker_refined_consumption,
            asset_pol_derivative_unrefined, average_times)


# ============================================================================
# Convenience: build movers + solve in one call
# ============================================================================

def solve(cp, method='FUES'):
    """Build stage solvers and run backward induction.

    Composition is explicit:  Operator_Factory -> movers -> backward_induction.
    """
    movers = Operator_Factory(cp)
    return backward_induction(cp, movers, method)
