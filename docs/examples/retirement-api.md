# Retirement Example — API Reference

API documentation for `examples/retirement/code/`.

---

## Canonical Pipeline

### `solve_canonical`

::: examples.retirement.code.solve_block.solve_canonical

```python
from examples.retirement.code import solve_canonical

nest, cp = solve_canonical(
    syntax_dir="examples/retirement/syntax/syntax",
    params_override={"grid_size": 3000, "delta": 1.0},
    method="FUES",
)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `syntax_dir` | `str` or `Path` | Root syntax directory containing `calibration.yaml`, `settings.yaml`, `period.yaml`, and `stages/`. |
| `params_override` | `dict`, optional | Override calibration values (e.g. `{"grid_size": 3000}`). |
| `method` | `str` | Upper-envelope method: `FUES`, `DCEGM`, `RFC`, or `CONSAV`. |

**Returns**

| Name | Type | Description |
|------|------|-------------|
| `nest` | `dict` | Solved nest with keys `periods`, `twisters`, `solutions`. |
| `cp` | `RetirementModel` | Model instance used for solving. |

---

### `backward_induction`

```python
from examples.retirement.code import (
    backward_induction, Operator_Factory, RetirementModel,
)

cp = RetirementModel(r=0.02, beta=0.98, delta=1.0,
                     y=20, grid_size=3000, T=20)
stage_ops = Operator_Factory(cp)
nest = backward_induction(
    cp, stage_ops,
    syntax_dir="examples/retirement/syntax/syntax",
    method="FUES",
)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `cp` | `RetirementModel` | Model instance with calibrated parameters and grids. |
| `stage_ops` | `dict` | Stage operators from `Operator_Factory`. |
| `syntax_dir` | `str` or `Path` | Root syntax directory. |
| `method` | `str` | Upper-envelope method. |

**Returns** — `dict`: the solved nest.

---

### `build_and_solve_nest`

The core function.  Accretively builds and solves the nest one period at a time.

For each \(h = 0, 1, \ldots, T{-}1\) (distance from terminal):

1. Build the period via the dolo-plus pipeline (`build_period`).
2. Compile equation callables + build stage operators.
3. Assemble continuation from the previous solution (terminal condition at \(h = 0\)).
4. Solve the period in reverse topological order: `retire_cons` → `work_cons` → `labour_mkt_decision`.
5. Append solution to the nest.

---

## Nest Structure

The nest is a plain dict:

```python
nest = {
    "periods":   [period_0, period_1, ...],   # calibrated SymbolicModel dicts
    "twisters":  [None, {"b": "a", ...}, ...], # inter-period rename
    "solutions": [sol_0, sol_1, ...],          # one per h
}
```

Each solution dict:

```python
sol = {
    "t": 19,   # calendar time (age)
    "h": 0,    # distance from terminal

    "retire_cons": {
        "c": ndarray,    # consumption on arrival grid
        "v": ndarray,    # value on arrival grid
        "da": ndarray,   # asset derivative da'/da
        "ddv": ndarray,  # second-order marginal
    },
    "work_cons": {
        "v": ndarray,    # value on arrival wealth grid
        "c": ndarray,    # consumption
        "da": ndarray,   # asset derivative
        "c_hat": ndarray,    # pre-UE consumption (diagnostics)
        "q_hat": ndarray,    # pre-UE Q-function (diagnostics)
        "egrid": ndarray,    # pre-UE endogenous grid (diagnostics)
        "da_pre_ue": ndarray, # pre-UE asset derivative (diagnostics)
    },
    "labour_mkt_decision": {
        "v": ndarray,    # mixed value (after branching max)
        "c": ndarray,    # mixed consumption policy
        "dv": ndarray,   # marginal value dV = du(c)
        "ddv": ndarray,  # second-order marginal
    },
    "ue_time": float,     # upper-envelope wall-clock time
    "solve_time": float,  # total period solve time
}
```

---

## Accessors

Read from `nest["solutions"]` without manual indexing.

### `get_policy`

```python
from examples.retirement.code import get_policy

# Consumption policy (T x n), from branching stage
c = get_policy(nest, "c")

# Worker endogenous grid (T x n)
egrid = get_policy(nest, "egrid", stage="work_cons")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `nest` | `dict` | Solved nest. |
| `key` | `str` | Field name (e.g. `"c"`, `"v"`, `"dv"`). |
| `stage` | `str` | Stage name (default: `"labour_mkt_decision"`). |

Returns `ndarray` of shape `(T, n)`, indexed by age \(t\).

### `get_timing`

```python
from examples.retirement.code import get_timing

ue_time, solve_time = get_timing(nest)
```

Returns `[mean_ue_time, mean_solve_time]` (skipping first 3 warmup periods).

### `get_solution_at_age`

```python
from examples.retirement.code import get_solution_at_age

sol = get_solution_at_age(nest, t=17)
```

Returns the solution dict for calendar age \(t\).

---

## Model

### `RetirementModel`

```python
from examples.retirement.code import RetirementModel

cp = RetirementModel(
    r=0.02, beta=0.98, delta=1.0, smooth_sigma=0,
    y=20, b=1e-10, grid_max_A=500, grid_size=3000,
    T=20, m_bar=1.2,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | `float` | 0.02 | Interest rate |
| `beta` | `float` | 0.945 | Discount factor |
| `delta` | `float` | 1 | Fixed utility cost of working |
| `smooth_sigma` | `float` | 0 | Logit smoothing (0 = hard max) |
| `y` | `float` | 1 | Wage income |
| `b` | `float` | 0.01 | Asset grid lower bound |
| `grid_max_A` | `float` | 50 | Asset grid upper bound |
| `grid_size` | `int` | 50 | Number of grid points |
| `T` | `int` | 60 | Horizon (periods) |
| `m_bar` | `float` | 1.2 | FUES jump detection threshold |
| `equations` | `dict` | `None` | Override `{u, du, uc_inv, ddu}` |

#### `RetirementModel.from_period`

Construct from a dolo-plus calibrated period dict:

```python
period = build_period(params, syntax_dir)
cp = RetirementModel.from_period(period)
```

---

## Stage Operators

### `Operator_Factory`

```python
from examples.retirement.code import Operator_Factory

stage_ops = Operator_Factory(cp)
```

Returns a dict with three stage operators:

| Key | Operator | Description |
|-----|----------|-------------|
| `"retire_cons"` | `solver_retiree_stage` | Retiree EGM (no upper envelope) |
| `"work_cons"` | `solver_worker_stage` | Worker EGM + upper envelope |
| `"labour_mkt_decision"` | `lab_mkt_choice_stage` | Branching max/logit |

#### `retire_cons` — Retiree EGM

```python
c, v, da, ddv = stage_ops["retire_cons"](
    c_cntn, v_cntn, ddv_cntn, t,
)
```

Solves via InvEuler → endogenous grid → Bellman → interpolate → constrained region → MarginalBellman.

#### `work_cons` — Worker EGM + Upper Envelope

```python
v, c, da, ue_time, c_hat, q_hat, egrid, da_pre_ue = \
    stage_ops["work_cons"](
        dv_cntn, ddv_cntn, v_cntn, method="FUES",
    )
```

Three-step: (1) invert Euler, (2) upper envelope (FUES/DCEGM/RFC/CONSAV), (3) interpolate to arrival grid.

Returns 4 solution arrays + 4 diagnostic arrays.

#### `labour_mkt_decision` — Branching Max

```python
v, c, dv, ddv = stage_ops["labour_mkt_decision"](
    v_work, v_retire, c_work, c_retire,
    da_work, da_retire,
)
```

Hard max (`smooth_sigma=0`) or logit-smoothed aggregation over work/retire branches.

---

## Diagnostics

### `euler`

```python
from examples.retirement.code import euler

c = get_policy(nest, "c")
error = euler(cp, c)  # mean log10 Euler error
```

### `consumption_deviation`

```python
from examples.retirement.code import consumption_deviation

c_true = get_policy(nest_true, "c")
dev = consumption_deviation(cp, c, c_true, a_grid_true)
```

Mean log10 deviation from a high-resolution reference solution.

---

## Period Construction

### `build_period`

```python
from examples.retirement.code.solve_block import build_period

period = build_period(
    params={"r": 0.02, "beta": 0.98, "delta": 1.0, ...},
    syntax_dir="examples/retirement/syntax/syntax",
)
```

Loads each stage from YAML via the dolo-plus pipeline: load → SymbolicModel → methodize → configure → calibrate.

### `build_period_callables`

```python
from examples.retirement.code.solve_block import build_period_callables

callables = build_period_callables(period)
# {"u": @njit, "du": @njit, "uc_inv": @njit, "ddu": @njit}
```

Returns equation callables for the period.  Currently returns log-utility defaults; a future whisperer would compile from the YAML equations.

---

## YAML Syntax

Stage declarations live in `examples/retirement/syntax/syntax/stages/`:

```
stages/
  labour_mkt_decision/    # branching (max aggregator)
  work_cons/              # worker EGM + FUES
  retire_cons/            # retiree EGM
```

Period template: `period.yaml`
Calibration: `calibration.yaml`
Settings (grids, m_bar): `settings.yaml`
Inter-period twister: `{"b": "a", "b_ret": "a_ret"}`
