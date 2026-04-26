# Retirement Example — API Reference

Partially auto-generated from docstrings. Treat this page as an internal
reference for the retirement example code, not as the primary user guide. For
usage and model description, see
[Retirement Choice](../examples/retirement_choice_model.md) and the
[retirement notebook](../notebooks/retirement_fues.ipynb).


## Pipeline (`solve.py`)

### `solve_nest`

```python
solve_nest(registry_dir, spec_factory_name="spec_factory.yaml", draw=None, ue_method=None, model=None, stage_ops=None, waves=None)
```

Canonical pipeline: load syntax, build the stage graph, and solve backward.

Three-functor pipeline:

1. Load ``calibration.yaml`` (economic params) and
   ``settings.yaml`` (numerical/structural settings).
2. Apply overrides at the correct abstraction level.
3. Build and solve the nest via
   :func:`_accrete_nest`.

The `ue_method` parameter selects the upper-envelope algorithm
(`FUES`, `DCEGM`, `RFC`, or `CONSAV`). The `draw` argument carries
tiered overrides of the form `{"calibration": {...}, "settings": {...}}`.

Parameters
----------
registry_dir : str or Path
    Root syntax directory containing ``calibration.yaml``,
    ``settings.yaml``, ``period.yaml``, and ``stages/``.
draw : dict, optional
    Tiered overrides, e.g.
    ``{"calibration": {"beta": 0.96}, "settings": {"grid_size": 5000}}``.
ue_method : str or dict, optional
    Upper-envelope method selection.

Returns
-------
nest : dict
    The solved nest.
model : RetirementModel
    The model instance used for solving.
stage_ops : dict
    The stage operators used for solving.

### `instantiate_period`

```python
instantiate_period(params, syntax_dir)
```

Build one period via the dolo-plus three-functor pipeline.

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

### `build_period_callables`

```python
build_period_callables(period)
```

Return equation callables for a calibrated period.

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


## Model (`model.py`)

### `RetirementModel`

Calibration and grids for the retirement choice model.

Stores calibrated parameters, the asset grid, and
equation callables (utility and its derivatives).

All parameters are required — canonical values live in
``syntax/calibration.yaml`` and ``syntax/settings.yaml``.
Use :meth:`from_period` or :meth:`with_test_defaults` to
construct instances.

Parameters
----------
r : float
    Interest rate.
beta : float
    Discount factor (not rate).
delta : float
    Fixed utility cost of working.
smooth_sigma : float
    Logit smoothing parameter (0 = hard max).
y : float
    Wage income for workers.
b : float
    Lower bound for asset grid.
grid_max_A : float
    Upper bound for asset grid.
grid_size : int
    Number of asset grid points.
T : int
    Number of periods (horizon).
m_bar : float
    FUES jump detection threshold.
padding_mbar : float
    Padding for m_bar.
equations : dict, optional
    Override equation callables (keys: ``u``, ``du``,
    ``uc_inv``, ``ddu``).  Defaults to log utility.

Attributes
----------
R : float
    Gross return ``1 + r``.
asset_grid_A : ndarray
    Asset grid of size *grid_size*.
eulerK : int
    Number of Euler equation check points.
u, du, uc_inv, ddu : callable
    Equation callables (``@njit``).

#### `RetirementModel.__init__`

```python
__init__(self, r, beta, delta, smooth_sigma, y, b, grid_max_A, grid_size, T, m_bar, padding_mbar=0, equations=None)
```

Initialize self.  See help(type(self)) for accurate signature.

#### `RetirementModel.from_period`

```python
from_period(period, equations=None)
```

Construct from a dolo-plus calibrated period dict.

Reads calibration and settings from the ``work_cons``
stage's ``.calibration`` and ``.settings`` attributes.
These are populated by the calibrate and configure
functors during :func:`build_period`.

Parameters
----------
period : dict
    ``{<stage_name>: <calibrated SymbolicModel>}``.
    Must contain a ``'work_cons'`` key whose value
    has ``.calibration`` and ``.settings`` dicts.
equations : dict, optional
    Override equation callables.

Returns
-------
RetirementModel

#### `RetirementModel.with_test_defaults`

```python
with_test_defaults(**overrides)
```

Construct with test defaults (for unit tests only).

Canonical values match ``syntax/calibration.yaml``
and ``syntax/settings.yaml``.

Parameters
----------
**overrides
    Any parameter to override from the defaults.

Returns
-------
RetirementModel

### `Operator_Factory`

```python
Operator_Factory(cp, equations=None)
```

Build stage operators for the retirement model.

Returns three operators corresponding to the three
stages in the period template:

- ``retire_cons``: retiree EGM (no upper envelope).
- ``work_cons``: worker EGM + upper envelope (FUES/
  DCEGM/RFC/CONSAV).
- ``labour_mkt_decision``: branching max/logit
  aggregator over work and retire branches.

All operators are closures over the calibrated
parameters and equation callables.

Parameters
----------
cp : RetirementModel
    Model instance with calibrated parameters and
    grids.
equations : dict, optional
    Override equation callables.  Keys: ``u``, ``du``,
    ``uc_inv``, ``ddu``.  Each must be ``@njit``.
    When *None*, uses the callables on *cp* (which
    default to log-utility).

Returns
-------
dict
    ``{'retire_cons':  solver_retiree_stage,
       'work_cons':    solver_worker_stage,
       'labour_mkt_decision': lab_mkt_choice_stage}``


## Benchmark (`benchmark.py`)

Helpers used by the canonical kikku path in `run.py`: the Cartesian product of
`--params-range`, `--settings-range`, and `--methods-range` is `run.test_set`;
`sweep` produces `list[SweepResult]`, and these functions precompute reference
policies and format tables.

### `load_baseline`

```python
load_baseline() -> tuple[dict, dict]
```

Load default calibration and settings from `syntax/calibration.yaml` and
`syntax/settings.yaml`.

### `precompute_true_solutions`

```python
precompute_true_solutions(deltas, true_grid_size, true_method, base_params, base_settings, *, comm) -> dict[float, dict]
```

On rank 0, solve once per `delta` at `true_grid_size` with `true_method`, then
broadcast the mapping to all ranks. Values are `{'c_true', 'a_grid'}` for
consumption-deviation metrics.

### `format_timing_sweep_for_tables`

```python
format_timing_sweep_for_tables(results, *, method_order=METHODS) -> dict[str, list]
```

Reshape flat `SweepResult` rows into row lists for the timing/accuracy writers
(keys: ``errors``, ``ue_ms``, ``total_ms``, ``cdev``).

### `write_timing_sweep_tables`

```python
write_timing_sweep_tables(results, results_dir, *, benchmark_params, latex_grids)
```

Call `format_timing_sweep_for_tables` and write markdown/LaTeX via
`outputs` table generators. ``latex_grids`` selects which grid sizes appear in
LaTeX output; markdown includes all completed rows.


## CLI (`run.py`)

### `parse_overrides`

```python
parse_overrides(raw_list)
```

Parse 'key=value' strings into a dict, coercing types.

### `load_override_file`

```python
load_override_file(path)
```

Load overrides from a YAML file (flat key-value format).


## Diagnostics (`outputs/diagnostics.py`)

### `get_policy`

```python
get_policy(nest, key, stage='labour_mkt_decision')
```

Get T x n array from nest solutions, indexed by age t.

Parameters
----------
nest : dict
    Solved nest from :func:`build_and_solve_nest`.
key : str
    Field name within the stage solution dict
    (e.g. ``"c"``, ``"v"``, ``"dv"``).
stage : str
    Stage name (default: ``labour_mkt_decision``).

Returns
-------
ndarray (T x n)

### `get_timing`

```python
get_timing(nest)
```

Mean UE time and solve time (skipping first 3 warmup).

Returns
-------
list
    ``[mean_ue_time, mean_solve_time]``.

### `get_solution_at_age`

```python
get_solution_at_age(nest, t)
```

Get solution dict for calendar age *t*.

Parameters
----------
nest : dict
    Solved nest.
t : int
    Calendar time (age), where ``t = T-1`` is the
    last decision period.

Returns
-------
dict
    Solution dict for age *t*.

### `euler`

```python
euler(cp, sigma_work)
```

Mean log10 Euler equation error across periods.

For each grid point and period, computes the residual
of the consumption Euler equation and returns the
mean of ``log10(|residual / c| + 1e-16)``.

Parameters
----------
cp : RetirementModel
    Model instance (provides grid, R, beta, du, uc_inv).
sigma_work : ndarray (T x grid_size)
    Consumption policy on the asset grid.

Returns
-------
float
    Mean log10 Euler error (more negative = better).

### `consumption_deviation`

```python
consumption_deviation(cp, c_solution, c_true, a_grid_true)
```

Mean log10 deviation from a high-resolution solution.

Compares consumption on a coarser grid to a
high-resolution reference (e.g. DCEGM with 20k points).

Parameters
----------
cp : RetirementModel
    Model parameters for the solution being tested.
c_solution : ndarray (T x grid_size)
    Consumption policy from the method being tested.
c_true : ndarray (T x true_grid_size)
    High-resolution reference solution.
a_grid_true : ndarray
    Asset grid of the reference solution.

Returns
-------
float
    Mean log10 absolute relative deviation.

