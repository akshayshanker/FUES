# 2026-03-03 — Retirement Pipeline-First Restructure

## What was done

### 1. Matsya review of restructure plan

Queried `econ-ark-matsya` (bellman-ddsl RAG, Opus 4.6 with extended thinking) for DDSL conformance review of the pipeline-first restructure plan. Key findings incorporated into the plan:

- **Three-functor pipeline**: DDSL prescribes methodize -> configure -> calibrate as distinct functors. Original plan conflated all three into a single `merged_params`.
- **Operator signature violations**: `t` and `method` are configuration, not value-function arguments. Debug values in `work_cons` return mix solver internals with clean outputs.
- **Stage self-containment**: `from_period` should not reach into stage internals post-calibration.
- **Perch tagging inconsistencies**: `retire_cons` uses `_cntn`/`_arvl`, `work_cons` doesn't.
- **Twister naming**: Should carry perch tags. Two-track state space documented.

### 2. Implementation

#### Step 0: Default value audit

Reconciled conflicting defaults across files:
- `beta`: 0.945 (__init__) / 0.98 (baseline.yml, calibration.yaml) / 0.96 (benchmark.py) -> **0.98** canonical
- `T`: 60 (__init__) / 20 (baseline.yml, calibration.yaml) / 50 (benchmark.py) -> **20** canonical
- `b`: 1e-2 (__init__) / 1e-10 (baseline.yml) / 1e-100 (benchmark.py) -> **1e-10** canonical

#### Step 1: Flattened syntax/syntax/ to syntax/

- Moved all files up one level
- Removed `.gitkeep`
- Updated `SYNTAX_DIR` in `run_experiment.py` and `benchmark.py`

#### Step 2: Added missing params to YAML configs

- `calibration.yaml`: added `b`, `smooth_sigma`
- `settings.yaml`: added `grid_size`, `grid_max_A`, `padding_mbar`

#### Step 3: Rewrote solve.py — three-functor pipeline

- `solve_canonical()` now has `calib_overrides` and `config_overrides` (separate override hooks)
- Fixed blocking bug: settings are now properly loaded and merged
- `backward_induction()` deleted
- `build_and_solve_nest` renamed to `_build_and_solve_nest` (internal)
- Returns `(nest, model, stage_ops)` instead of `(nest, cp)`
- Renamed `cp` -> `model` throughout

#### Step 4: Updated model.py

- `RetirementModel.__init__`: all params now required (no defaults)
- Added `RetirementModel.with_test_defaults()` classmethod
- `from_period`: reads from both `.calibration` and `.settings` with `_get()` helper
- `padding_mbar` wired through `from_period`

#### Step 5: Rewrote run_experiment.py

- Uses `solve_canonical()` exclusively (no manual `RetirementModel` construction)
- New CLI: `--calib-override`, `--config-override`, `--override-file`, `--method`
- Type coercion via `yaml.safe_load(value)` for CLI overrides
- `--grid-size N` is shorthand for `--config-override grid_size=N`

#### Step 6: Rewrote benchmark.py

- Uses `solve_canonical()` for all runs (true solutions + sweep)
- Loads baseline from syntax dir (no hardcoded params)
- Simplified best-of-n loop with dict-based tracking

#### Step 7: Cleaned up params/ and docs

- Deleted `examples/retirement/params/` (replaced by syntax/ as source of truth)
- Rewrote `experiments/retirement/params/*.yml` as sparse overrides (flat key-value)
- Updated `docs/examples/retirement-api.md`, `retirement.md`, `quickstart.md`, `index.md`
- Updated `experiments/retirement/README.md`

### Files changed

```
examples/retirement/
  model.py          — Required params, with_test_defaults(), _get() in from_period
  solve.py          — Three-functor pipeline, solve_canonical, deleted backward_induction
  run_experiment.py — Canonical pipeline CLI with override mechanism
  benchmark.py      — Uses solve_canonical throughout
  __init__.py       — Updated exports
  syntax/           — Flattened from syntax/syntax/
    calibration.yaml — Added b, smooth_sigma
    settings.yaml    — Added grid_size, grid_max_A, padding_mbar

experiments/retirement/
  params/*.yml      — Sparse overrides (flat key-value)
  README.md         — Updated docs

docs/
  examples/retirement-api.md — Updated API docs
  examples/retirement.md     — Updated code structure
  getting-started/quickstart.md — Updated CLI example
  index.md                   — Updated CLI example

AI/devspecs/03032026/
  retirement_pipeline-first_restructure_58cec34c.plan.md — Incorporated matsya feedback
```

### Current state

- `solve_canonical` is the single entry point
- No more `backward_induction` bypass
- `syntax/` is the single source of truth for all parameters
- Three-functor override mechanism (calib_overrides, config_overrides)
- Operator signatures still carry `t` and `method` at call time (deferred to next session — requires changes to model.py closures)

### TODO (next session)

- Clean operator signatures: bind `t` and `method` at construction time in `Operator_Factory`
- Standardize ValueFn returns across all stages to `(v_arvl, c_arvl, da_arvl, dlambda_arvl)`
- Extract debug values from `work_cons` return into diagnostics mechanism
- Smoke test the full pipeline (requires dolo-plus installed)
- Update `experiments/retirement/retirement_timings.sh` and `run_retirement_single_core.sh` for new CLI
