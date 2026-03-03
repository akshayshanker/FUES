# Retirement Example — API Reference

API documentation for `examples/retirement/`.

---

## Canonical Pipeline

### `solve_canonical`

::: examples.retirement.solve.solve_canonical

```python
from examples.retirement.solve import solve_canonical

nest, model, stage_ops = solve_canonical(
    syntax_dir="examples/retirement/syntax",
    method="FUES",
    calib_overrides={"beta": 0.96},
    config_overrides={"grid_size": 5000, "T": 50},
)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `syntax_dir` | `str` or `Path` | Root syntax directory containing `calibration.yaml`, `settings.yaml`, `period.yaml`, and `stages/`. |
| `method` | `str` | Upper-envelope method: `FUES`, `DCEGM`, `RFC`, or `CONSAV`. |
| `calib_overrides` | `dict`, optional | Override economic parameters (e.g. `{"beta": 0.96}`). |
| `config_overrides` | `dict`, optional | Override numerical settings (e.g. `{"grid_size": 5000}`). |

**Returns**

| Name | Type | Description |
|------|------|-------------|
| `nest` | `dict` | Solved nest with keys `periods`, `twisters`, `solutions`. |
| `model` | `RetirementModel` | Model instance used for solving. |
| `stage_ops` | `dict` | Stage operators used for solving. |

---

### `_build_and_solve_nest`

The core function (internal).  Accretively builds and solves the nest one period at a time.

For each \(h = 0, 1, \ldots, T{-}1\) (distance from terminal):

1. Build the period via the dolo-plus pipeline (`build_period`).
2. Compile equation callables + build stage operators.
3. Assemble continuation from the previous solution (terminal condition at \(h = 0\)).
4. Solve the period in reverse topological order: `retire_cons` → `work_cons` → `labour_mkt_decision`.
5. Append solution to the nest.

**Note:** For stationary models, model and stage operators are built once (on the first period) and reused. This is an explicit stationarity assumption.

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
from examples.retirement.outputs import get_policy

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
from examples.retirement.outputs import get_timing

ue_time, solve_time = get_timing(nest)
```

Returns `[mean_ue_time, mean_solve_time]` (skipping first 3 warmup periods).

### `get_solution_at_age`

```python
from examples.retirement.outputs import get_solution_at_age

sol = get_solution_at_age(nest, t=17)
```

Returns the solution dict for calendar age \(t\).

---

## Model

### `RetirementModel`

All parameters are required — canonical values live in `syntax/calibration.yaml` and `syntax/settings.yaml`.

```python
from examples.retirement.model import RetirementModel

# Via canonical pipeline (preferred):
nest, model, stage_ops = solve_canonical("examples/retirement/syntax")

# Via test defaults (for unit tests only):
model = RetirementModel.with_test_defaults(grid_size=500)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `r` | `float` | Interest rate |
| `beta` | `float` | Discount factor |
| `delta` | `float` | Fixed utility cost of working |
| `smooth_sigma` | `float` | Logit smoothing (0 = hard max) |
| `y` | `float` | Wage income |
| `b` | `float` | Asset grid lower bound |
| `grid_max_A` | `float` | Asset grid upper bound |
| `grid_size` | `int` | Number of grid points |
| `T` | `int` | Horizon (periods) |
| `m_bar` | `float` | FUES jump detection threshold |
| `padding_mbar` | `float` | Padding for m_bar (default: 0) |
| `equations` | `dict` | Override `{u, du, uc_inv, ddu}` (default: log utility) |

#### `RetirementModel.from_period`

Construct from a dolo-plus calibrated period dict:

```python
period = build_period(params, syntax_dir)
model = RetirementModel.from_period(period)
```

#### `RetirementModel.with_test_defaults`

Construct with canonical defaults (for unit tests only):

```python
model = RetirementModel.with_test_defaults(grid_size=500, T=10)
```

---

## Stage Operators

### `Operator_Factory`

```python
from examples.retirement.model import Operator_Factory

stage_ops = Operator_Factory(model)
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
from examples.retirement.outputs import euler

c = get_policy(nest, "c")
error = euler(model, c)  # mean log10 Euler error
```

### `consumption_deviation`

```python
from examples.retirement.outputs import consumption_deviation

c_true = get_policy(nest_true, "c")
dev = consumption_deviation(model, c, c_true, a_grid_true)
```

Mean log10 deviation from a high-resolution reference solution.

---

## Period Construction

### `build_period`

```python
from examples.retirement.solve import build_period

period = build_period(
    params={"r": 0.02, "beta": 0.98, "delta": 1.0, ...},
    syntax_dir="examples/retirement/syntax",
)
```

Loads each stage from YAML via the dolo-plus pipeline: parse → methodize → configure → calibrate.

### `build_period_callables`

```python
from examples.retirement.solve import build_period_callables

callables = build_period_callables(period)
# {"u": @njit, "du": @njit, "uc_inv": @njit, "ddu": @njit}
```

Returns equation callables for the period.  Currently returns log-utility defaults; a future whisperer would compile from the YAML equations.

---

## YAML Syntax

Stage declarations live in `examples/retirement/syntax/stages/`:

```
stages/
  labour_mkt_decision/    # branching (max aggregator)
  work_cons/              # worker EGM + FUES
  retire_cons/            # retiree EGM
```

Period template: `period.yaml`
Calibration: `calibration.yaml` (economic params — consumed by calibrate functor)
Settings: `settings.yaml` (numerical settings — consumed by configure functor)
Inter-period twister: `{"b": "a", "b_ret": "a_ret"}`

---

## CLI

```bash
# Baseline (uses syntax/ defaults)
python examples/retirement/run.py --output-dir results/retirement

# Override economic params
python examples/retirement/run.py --calib-override beta=0.96 --calib-override delta=0.5

# Override numerical settings
python examples/retirement/run.py --config-override grid_size=5000 --config-override T=50

# Shorthand for grid size
python examples/retirement/run.py --grid-size 5000

# Use override file from experiments/
python examples/retirement/run.py --override-file ../../experiments/retirement/params/long_horizon.yml

# Run timing benchmarks
python examples/retirement/run.py --run-timings

# Choose method
python examples/retirement/run.py --method DCEGM
```
