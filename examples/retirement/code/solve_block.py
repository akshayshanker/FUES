"""
Backward Induction via Stage Accretion: retirement choice (branching)

Implements the pattern from spec_0.1l-branching.md, following the same
structure as lifecycle_cons_period.py (the sequential cons-port variant).

Key differences from the sequential case:
- The period is a DAG, not a linear chain of stages.
- The branching stage (WorkerDecision) has two continuation perches:
  work → WorkerConsumption, retire → RetireeConsumption.
- solve_period uses topological order: leaf stages first, root last.
- Inter-period connector has two entries: {b: a, b_ret: a_ret}.

Period structure (branching DAG):
  labour_mkt_decision (branching, max)
    ├── work   → work_cons   (WorkerConsumption)  → b
    └── retire → retire_cons (RetireeConsumption)  → b_ret

Inter-period connector: {b: a, b_ret: a_ret}
  - b     → a     (worker output → branching stage arrival)
  - b_ret → a_ret (retiree output → retire_cons direct entry)

NO SOLVERS HERE. Stub solve only.
"""

from pathlib import Path
from typing import Optional
import yaml

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------

_EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = _EXAMPLE_ROOT.parent.parent
STAGES_PATH = _EXAMPLE_ROOT / "syntax" / "stages"
MODEL_PATH = _EXAMPLE_ROOT / "syntax"

# ---------------------------------------------------------------------------
# Imports: Stage representation layer
# ---------------------------------------------------------------------------

from dolo.compiler.model import SymbolicModel
from dolo.compiler.calibration import calibrate as calibrate_stage, configure as configure_stage
from dolo.compiler.methodization import methodize as methodize_stage


# =============================================================================
# STUB SOLVERS (for testing construction without numerical implementation)
# =============================================================================

def apply_inter_connector(solution: dict, inter_connector: dict) -> dict:
    """Apply inter-period connector to get continuation values for this period.

    For branching models, the connector has multiple entries:
      {b: a, b_ret: a_ret}
    meaning: this period's b feeds next period's a (worker entry),
             this period's b_ret feeds next period's a_ret (retiree entry).
    """
    return {
        "inter_connector_rename": inter_connector,
        "source_solution": solution,
    }


def solve_stage(stage, continuation_value, stage_name: str = "") -> dict:
    """Stub: solve a single stage given continuation value."""
    name = stage_name or getattr(stage, 'name', 'unknown')
    kind = getattr(stage, 'kind', 'sequential')
    return {
        "stage": name,
        "kind": kind,
        "arrival": {"V": f"V_arvl_{name}"},
        "decision": {"policy": f"policy_{name}"},
    }


def solve_branching_period(period: dict, continuation_value,
                           is_terminal: bool = False) -> dict:
    """
    Solve all stages within a branching period, working backwards.

    DAG solve order (topological, leaves first):
      1. retire_cons  (leaf — retiree consumption)
      2. work_cons    (leaf — worker consumption)
      3. labour_mkt_decision  (root — branching stage, combines both)

    At terminal, continuation_value is the terminal condition.

    The branching stage receives the arrival values from both downstream
    stages and applies max (agent-controlled branch_control).
    """
    stage_solutions = {}

    # --- Step 1: Solve leaf stages (both consumption stages) ---

    # Retiree consumption: receives continuation value for retiree path
    retire_cons = period["retire_cons"]
    retire_solution = solve_stage(
        retire_cons, continuation_value, stage_name="retire_cons"
    )
    stage_solutions["retire_cons"] = retire_solution
    print(f"    [retire_cons] solved → V_arvl = {retire_solution['arrival']['V']}")

    # Worker consumption: receives continuation value for worker path
    work_cons = period["work_cons"]
    work_solution = solve_stage(
        work_cons, continuation_value, stage_name="work_cons"
    )
    stage_solutions["work_cons"] = work_solution
    print(f"    [work_cons]   solved → V_arvl = {work_solution['arrival']['V']}")

    # --- Step 2: Solve branching stage (root) ---
    # The branching stage receives both branch arrival values and takes max.
    branching_stage = period["labour_mkt_decision"]
    branch_continuation = {
        "work": work_solution["arrival"],     # Q^work(a) from worker cons
        "retire": retire_solution["arrival"], # Q^retire(a) from retiree cons
    }
    branching_solution = solve_stage(
        branching_stage, branch_continuation,
        stage_name="labour_mkt_decision"
    )
    stage_solutions["labour_mkt_decision"] = branching_solution
    print(f"    [branching]   solved → V_arvl = {branching_solution['arrival']['V']}"
          f"  (max over work/retire)")

    return {
        "type": "terminal" if is_terminal else "interior",
        "stages_solved": list(stage_solutions.keys()),
        "stage_solutions": stage_solutions,
        # Period has two arrival values: worker entry (a) and retiree entry (a_ret)
        "worker_entry": branching_solution["arrival"],
        "retiree_entry": retire_solution["arrival"],
    }


# =============================================================================
# BUILDER FUNCTIONS
# =============================================================================

def build_syntactic_stage(name: str) -> SymbolicModel:
    """Load stage syntax from YAML file."""
    path = STAGES_PATH / name / f"{name}.yaml"
    with open(path, 'r') as f:
        return SymbolicModel(yaml.compose(f.read()), filename=str(path))


def build_period(params: dict) -> dict:
    """
    Build a period instance containing the three stages.

    All three steps for each stage:
    1. Load syntactic stage from YAML
    2. Methodize (attach solution scheme)
    3. Configure (attach settings)
    4. Calibrate (substitute parameter values)

    The branching stage (WorkerDecision) uses a max aggregator.
    The consumption stages use EGM (with FUES for worker branch).
    """
    settings_path = str(MODEL_PATH / "settings.yaml")

    # --- Worker Decision (branching stage) ---
    wd_syntax = build_syntactic_stage("worker_decision")
    wd_M = methodize_stage(wd_syntax, str(STAGES_PATH / "worker_decision" / "worker_decision_methods.yml"))
    wd_cfg = configure_stage(wd_M, settings_path)
    wd_calib = calibrate_stage(wd_cfg, params)

    # --- Worker Consumption (work branch) ---
    wc_syntax = build_syntactic_stage("worker_consumption")
    wc_M = methodize_stage(wc_syntax, str(STAGES_PATH / "worker_consumption" / "worker_consumption_methods.yml"))
    wc_cfg = configure_stage(wc_M, settings_path)
    wc_calib = calibrate_stage(wc_cfg, params)

    # --- Retiree Consumption (retire branch + direct retiree entry) ---
    rc_syntax = build_syntactic_stage("retiree_consumption")
    rc_M = methodize_stage(rc_syntax, str(STAGES_PATH / "retiree_consumption" / "retiree_consumption_methods.yml"))
    rc_cfg = configure_stage(rc_M, settings_path)
    rc_calib = calibrate_stage(rc_cfg, params)

    return {
        "labour_mkt_decision": wd_calib,
        "work_cons": wc_calib,
        "retire_cons": rc_calib,
    }


# =============================================================================
# PARAMS SCHEDULE
# =============================================================================

def make_params_schedule(T: int) -> dict:
    """
    Create params_schedule mapping h → params dict.

    h is distance from terminal:
    - h=0: terminal period
    - h=T: initial period

    Following Iskhakov et al. (2017) baseline: all parameters constant.
    """
    base_params = {
        "r": 0.02,
        "beta": 0.98,
        "delta": 1.0,
        "y": 20.0,
    }

    return {h: dict(base_params) for h in range(T + 1)}


# =============================================================================
# MAIN: BUILD AND SOLVE NEST
# =============================================================================

def build_and_solve_nest(
    T: int,
    params_schedule: dict,
    terminal_condition=None,
) -> dict:
    """
    Build nest backwards from terminal, solving as we go.

    Per spec_0.1l + spec_0.1h:
    - Each time slot uses the same period template (LifecyclePeriod)
    - Inter-period connector: {b: a, b_ret: a_ret}
    - Branching is internal to each period (the nest is a linear chain)
    """
    nest = {"periods": [], "inter_connectors": [], "solutions": []}

    # The inter-period connector for the retirement model:
    # b → a (worker continuation → branching stage arrival)
    # b_ret → a_ret (retiree continuation → retire_cons direct entry)
    INTER_CONNECTOR = {"b": "a", "b_ret": "a_ret"}

    for h in range(T + 1):  # h = 0, 1, ..., T (distance from terminal)

        # --- Get parameters for this period ---
        params = params_schedule[h]

        # --- Build period: load + methodize + configure + calibrate ---
        period = build_period(params)

        nest["periods"].append(period)
        nest["inter_connectors"].append(INTER_CONNECTOR)

        # --- Continuation value ---
        if h == 0:
            continuation_value = terminal_condition
        else:
            continuation_value = apply_inter_connector(
                nest["solutions"][-1], INTER_CONNECTOR
            )

        # --- Solve this period (branching DAG solve order) ---
        print(f"  h={h:2d} (age T-{h}):")
        solution = solve_branching_period(
            period, continuation_value, is_terminal=(h == 0)
        )
        nest["solutions"].append(solution)

        print(f"    → period [{solution['type']}], "
              f"stages solved: {solution['stages_solved']}")
        print()

    return nest


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BACKWARD INDUCTION: retirement choice (branching)")
    print("=" * 70)
    print()
    print("Building nest backwards from terminal (h=0) to initial (h=T)")
    print("Period structure: labour_mkt_decision → {work_cons, retire_cons}")
    print("Inter-period connector: {b: a, b_ret: a_ret}")
    print()

    T = 5  # Small horizon for testing

    print(f"Horizon: T={T} periods")
    print()

    params_schedule = make_params_schedule(T)

    print("Building and solving nest (backwards, stub solve)...")
    print()

    nest = build_and_solve_nest(
        T=T,
        params_schedule=params_schedule,
        terminal_condition={"V": "V_terminal"},
    )

    # --- Assemble model dict (plain dict per spec_0.1h) ---
    model = {
        "periods": nest["periods"],
        "inter_connectors": nest["inter_connectors"],
    }

    print("=" * 70)
    print(f"SUMMARY: {len(model['periods'])} periods, "
          f"{len(model['inter_connectors'])} inter-period connectors")
    print()

    # --- Print stage info from the first period (verify parsing) ---
    p0 = model["periods"][0]
    for occ_name, stage in p0.items():
        kind = getattr(stage, 'kind', 'sequential')
        name = getattr(stage, 'name', occ_name)
        syms = list(stage.symbols.keys()) if hasattr(stage, 'symbols') else []
        print(f"  {occ_name}: {name} (kind={kind})")
        print(f"    symbol groups: {syms}")
        if hasattr(stage, 'branch_labels') and stage.branch_labels:
            print(f"    branch_labels: {stage.branch_labels}")
            print(f"    branch_poststates: {stage.branch_poststates}")
    print()
    # --- Branching-specific assertions (will fail until parser supports branching) ---
    print("Running branching-specific assertions...")
    print()

    wd = p0["labour_mkt_decision"]

    # 1. kind must be "branching" (not "sequential" or "adc-stage")
    assert getattr(wd, 'kind', None) == "branching", \
        f"Expected kind='branching', got kind={getattr(wd, 'kind', None)!r}"

    # 2. branch_control must be "agent"
    assert getattr(wd, 'branch_control', None) == "agent", \
        f"Expected branch_control='agent', got {getattr(wd, 'branch_control', None)!r}"

    # 3. branch_labels must be ["work", "retire"] (YAML declaration order)
    assert getattr(wd, 'branch_labels', None) == ["work", "retire"], \
        f"Expected branch_labels=['work', 'retire'], got {getattr(wd, 'branch_labels', None)!r}"

    # 4. branch_poststates must have the right structure
    bp = getattr(wd, 'branch_poststates', None)
    assert bp is not None, "branch_poststates not found on branching stage"
    assert set(bp.keys()) == {"work", "retire"}, \
        f"Expected branch keys {{work, retire}}, got {set(bp.keys())}"

    # 5. flat poststates must contain all poststate names (disjoint union)
    flat_ps = wd.symbols.get("poststates", [])
    assert "a" in flat_ps and "a_ret" in flat_ps, \
        f"Expected flat poststates to contain 'a' and 'a_ret', got {flat_ps}"

    # 6. branch_transitions must exist
    bt = getattr(wd, 'branch_transitions', None)
    assert bt is not None, "branch_transitions not found on branching stage"
    assert set(bt.keys()) == {"work", "retire"}, \
        f"Expected transition keys {{work, retire}}, got {set(bt.keys())}"

    # 7. Non-branching stages should NOT have branch attributes
    wc = p0["work_cons"]
    assert getattr(wc, 'branch_labels', None) is None, \
        "WorkerConsumption should not have branch_labels"

    # 8. values_marginal must be recognized as a symbol group
    assert "values_marginal" in wc.symbols, \
        f"values_marginal not found in WorkerConsumption symbols: {list(wc.symbols.keys())}"

    print("ALL ASSERTIONS PASSED")
    print()
    print("=" * 70)
    print("DONE — all stages parsed, period constructed, nest built.")
    print("=" * 70)
