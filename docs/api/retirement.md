# Retirement Example — API Reference

Partially auto-generated from docstrings. Treat this page as an internal
reference for the retirement example code, not as the primary user guide. For
usage and model description, see
[Retirement Choice](../examples/retirement_choice_model.md) and the
[retirement notebook](../notebooks/retirement_fues.ipynb).


## Build and solve (`solve.py`)

### `solve_nest`

```python
solve_nest(registry_dir, spec_factory_name="spec_factory.yaml", draw=None, method_switch=None, model=None, stage_ops=None, waves=None, **more_slots)
```

The full sequence: load the model files, build the stage graph, and solve backward.

Composition:

1. `spec_factory.load` / `spec_factory.make` bind slot values
   (`draw`, `method_switch`, `**more_slots`) into a `SpecGraph`.
2. `period_factory.make` builds the calibrated period;
   `period_to_graph` + `backward_paths` derive the wave ordering.
3. `RetirementModel(period)` and the three stage-operator
   factories feed :func:`solve_backward`.

The `method_switch` parameter selects the upper-envelope algorithm
(`FUES`, `DCEGM`, `RFC`, or `CONSAV`); a string tag expands via
`expand_method_shortcut` onto the `work_cons.upper_env` node. The
`draw` argument takes tiered overrides of the form
`{"calibration": {...}, "settings": {...}}`.

Parameters
----------
registry_dir : str or Path
    Root syntax directory containing ``spec_factory.yaml``,
    ``calibration.yaml``, ``settings.yaml``, ``period.yaml``,
    and ``stages/``.
draw : dict, optional
    Tiered overrides, e.g.
    ``{"calibration": {"beta": 0.96}, "settings": {"grid_size": 5000}}``.
method_switch : str or dict, optional
    Upper-envelope method selection. String tags expand via
    ``expand_method_shortcut``; nested ``{methods: [...]}`` dicts
    pass through.
model, stage_ops, waves : optional
    Pass back the previous call's returns to skip the build
    reconstruction and JIT recompilation.
more_slots
    Additional spec_factory slot bindings (e.g. ``**t.slots``
    from kikku).

Returns
-------
nest : dict
    ``{"solutions": [...]}`` — one solution dict per period.
model : RetirementModel
    The model instance used for solving.
stage_ops : dict
    The stage operators used for solving.
waves : list[list[str]]
    Kikku-derived wave ordering of the stage graph.

### `solve_backward`

```python
solve_backward(T, model, stage_ops, waves)
```

Run backward induction over T periods.

A pure function: no file access, no rebuilding, just the backward loop. Wires
terminal-period continuations at ``h = 0``, then threads each
period's solution into the next call of :func:`solve_period`.

Returns
-------
list[dict]
    One solution dict per period (``h = 0`` is the last
    calendar period ``t = T-1``).

### `expand_method_shortcut`

```python
expand_method_shortcut(tag: str, shortcut: list[tuple[str, str]]) -> dict
```

Build a ``$method_switch`` slot value in the no-schemes shape.

Emits one ``{on: <node>, method: <tag>}`` entry per target node
(``METHOD_SHORTCUT`` targets ``work_cons.upper_env`` alone), then
normalizes it via ``methodization._normalize_methods``.

### `read_node_method`

```python
read_node_method(stage, node, default='FUES')
```

Read the method tag bound to a named methodization ``node``
(``upper_env``, ``policy``, ``evaluate``, ...). A legacy scheme
name (``upper_envelope``, ``maximization``, ...) is accepted and
mapped to its node. ``read_scheme_method(stage, scheme_name,
mover=None, default='FUES')`` is the backwards-compatible alias.


## Model (`model.py`)

### `RetirementModel`

Numerical resources (rho output) for the retirement model.

Holds a reference to the calibrated period dict, constructs
the asset grid, and stores ``@njit`` equation callables.
Scalar parameters (``beta``, ``delta``, ``y``, etc.) are
**not** stored — ``__getattr__`` delegates to the ``work_cons``
stage's ``.calibration`` and ``.settings``. Canonical values
live in ``syntax/calibration.yaml`` and ``syntax/settings.yaml``.

Parameters
----------
period : dict
    Canonical period dict with ``"stages"`` key.
callables : dict, optional
    Override equation callables (keys: ``u``, ``du``,
    ``uc_inv``, ``ddu``).  Defaults to log utility.

Attributes
----------
R : float
    Gross return ``1 + r`` (property, from calibration).
asset_grid_A : ndarray
    Asset grid of size *grid_size*.
eulerK : int
    Number of Euler equation check points.
u, du, uc_inv, ddu : callable
    Equation callables (``@njit``).

#### `RetirementModel.__init__`

```python
__init__(self, period, callables=None)
```

Construct from a calibrated period dict (as built inside
:func:`solve_nest`).

#### `RetirementModel.with_test_defaults`

```python
with_test_defaults(**overrides)
```

Construct with test defaults (no dolo-plus needed; for unit
tests only).

Canonical values match ``syntax/calibration.yaml``
and ``syntax/settings.yaml``.

Parameters
----------
**overrides
    Any parameter to override from the defaults.

Returns
-------
RetirementModel

### `make_worker_egm_fns` / `make_retiree_egm_fns`

```python
make_worker_egm_fns(beta, R, delta, _y)
make_retiree_egm_fns(beta, R, _delta, _y)
```

EGM recipe dicts (``@njit`` closures) for the ``work_cons`` and
``retire_cons`` stages: ``{'inv_euler', 'bellman_rhs',
'cntn_to_dcsn', 'concavity'}``.


## Operators (`solvers/operators.py`)

Stage operator factories. Each stage has its own factory
returning ``dcsn_mover`` (B) and, where the stage has one,
``arvl_mover`` (I); the composition T = I ∘ B happens inline
in ``solve_period``. All operators are closures over the
calibrated parameters and equation callables.

### `make_retire_cons`

```python
make_retire_cons(model, callables)
```

Retiree EGM (no upper envelope). ``callables`` is the EGM recipe
dict from ``make_retiree_egm_fns``. Returns
``{'dcsn_mover', 'arvl_mover'}``.

### `make_work_cons`

```python
make_work_cons(model, callables, ue_method='FUES')
```

Worker EGM + upper envelope (`FUES`/`DCEGM`/`RFC`/`CONSAV`).
``callables`` is the EGM recipe dict from ``make_worker_egm_fns``.
Returns ``{'dcsn_mover', 'arvl_mover'}``.

### `make_labour_mkt_decision`

```python
make_labour_mkt_decision(model)
```

Branching max/logit aggregator over work and retire branches
(hard argmax when ``smooth_sigma == 0``, logit otherwise).
Returns ``{'dcsn_mover'}`` — no arrival mover.


## Benchmark (`benchmark.py`)

Helpers used by the canonical kikku path in `run.py`: kikku's slot CLI
(`--slot-override`, `--slot-spec`, `--slot-range`) builds `run.test_set`;
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
`postprocess` table generators. ``latex_grids`` selects which grid sizes appear
in LaTeX output; markdown includes all completed rows.


## CLI (`run.py`)

Argument parsing is delegated to `kikku.run.parse_cli` (the v4 slot
surface: `--slot-override`, `--slot-spec`, `--slot-range`, plus
`--sweep-runs`, `--warmup`, `--output-dir`, ...). `run.py` adds one
extra flag, `--latex-grids` (comma list of grid sizes for LaTeX
timing/accuracy table output).

### `main`

```python
main() -> None
```

Entry point. Builds `run.test_set` via `parse_cli`; when no row
includes a `method_switch` slot, fans each row out across the four
upper-envelope methods (`RFC`, `FUES`, `DCEGM`, `CONSAV`). Method-only
sweeps take the plot path (`make_solve_test_plots`); sweeps that vary
grids or calibration take the timing path (`make_solve_test_timing`,
with a 20k-grid DCEGM reference from
`benchmark.precompute_true_solutions`).

### `make_solve_test_timing`

```python
make_solve_test_timing(wdir, true_solutions, base_c)
```

Timing kernel factory: solve → policy + Euler + consumption deviation
vs. high-grid truth → pack metrics (`ue_time`, `total_time`, `error`,
`cdev`).

### `make_solve_test_plots`

```python
make_solve_test_plots(wdir)
```

Plot kernel factory: solve → unpack stage policies/grids/value
function for figures.


## Diagnostics (`postprocess/diagnostics.py`)

### `get_policy`

```python
get_policy(nest, key, stage='labour_mkt_decision')
```

Get T x n array from nest solutions, indexed by age t.

Parameters
----------
nest : dict
    Solved nest from :func:`solve_nest`.
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

