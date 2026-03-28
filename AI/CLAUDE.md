# FUES — AI Session Init (`AI/CLAUDE.md`)

This file is printed at the start of each development session by:
- `./AI/scripts/init_claude_session.sh`
- `python AI/scripts/claude_session.py`

## Session start checklist (always)
1. **Review project index**: read `AI/INDEX.md` (quick links + pointers).
2. **Review recent progress**:
   - latest working notes under `AI/working/` (most recent first)
   - current tasks in `AI/todo.md` (if relevant)
3. **Check repo state**: `git status` and skim `git diff` before changing anything.
4. **Create / update today's working note** (see below) as you work: record conceptual lessons + concrete code changes.

## Daily working notes (convention)
We keep a **new folder for each day** under `AI/working/`, containing concise session notes.

- **Path**: `AI/working/DDMMYYYY/`
- **File**: `session.md` (or descriptive name)
- **Content**: conceptual lessons learned + what changed (files, key diffs, decisions, next steps).
- **At init**: quickly skim the last 1–2 session notes to get back into context.
- **At end of session**: update today's working note and add a link in `AI/INDEX.md`.

Legacy notes (older work) may exist as:
- single files like `DDMMYYYY_working.md`
- older session logs under `AI/context/history/`
- the legacy folder `AI/working/2025/`

Keep legacy notes intact, but use the `AI/working/DDMMYYYY/` convention going forward.

## ⚠️ CRITICAL: NO GUESSING ⚠️
- **Always verify names**: search before using any function/class/parameter/path.
- **Never assume APIs**: confirm with repo search + reading the file (avoid AttributeError/NameError churn).
- Prefer **small, reversible changes**; keep behavior stable unless explicitly changing it.

## CRITICAL: File system / Gadi constraints
This project runs on the NCI Gadi HPC cluster. The home drive has limited quota.

- **Do not write large outputs to home**.
- **All run outputs** (plots, solution bundles, logs) should go to: `/scratch/tp66/{user}/`
- Local development can edit the repo, but solver runs should respect the scratch-output convention on Gadi.

## Ecosystem: four repos

| Repo | Role | This repo uses |
|------|------|---------------|
| **FUES** (this) | FUES algorithm + example models | installable as `dcsmm` |
| **kikku** (bright-forest/kikku) | DDSL infrastructure: YAML parsing, stage instantiation, EGM builders, CLI, sweep | `kikku.dynx`, `kikku.asva`, `kikku.run` |
| **bellman-ddsl** | Theory: stage algebra, perches, operator composition | Concepts, not code imports |
| **dolo-plus** | YAML syntax spec for stage definitions | `dolo.compiler.calibration` |

The DDSL pipeline: YAML syntax (dolo-plus) → parsed by kikku → operators built in FUES examples → solved by dcsmm algorithms.

## Project map

### Core package (`src/dcsmm/`)

| Directory | What | Key exports |
|-----------|------|-------------|
| `fues/` | FUES upper-envelope algorithm | `FUES_jit`, `uniqueEG`; versions v0dev, v0.1dev, v0.2dev (current default) |
| `fues/helpers/` | 1D interpolation, math utils | `interp_as`, `interp_as_scalar`, `find_roots_piecewise_linear` |
| `uenvelope/` | Unified upper-envelope dispatch | `EGM_UE` (dispatches to FUES, RFC, DCEGM, MSS, LTM) |
| `helpers/` | MPI utilities | `get_comm`, `chunk_indices`, `DummyComm` (serial fallback) |
| `models/` | Legacy model code | `housing_renting/` (older, being superseded by examples/) |

### Examples (`examples/`)

Three active models, each self-contained with YAML syntax + solve + outputs:

| Model | Directory | Stages | Methods | Status |
|-------|-----------|--------|---------|--------|
| **Retirement** | `examples/retirement/` | labour_mkt_decision, retire_cons, work_cons | RFC, FUES, DCEGM, CONSAV | Stable |
| **Durables** | `examples/durables2_0/` | tenure, keeper_cons, adjuster_cons | FUES, NEGM | Active development |
| **Housing/renting** | `examples/housing_renting/` | tenure_choice, owner_cons, renter_cons, etc. | FUES, RFC, DCEGM | Do NOT edit |

Each example has:
```
examples/<model>/
├── syntax/                  # YAML stage definitions (dolo-plus format)
│   ├── calibration.yaml     # Economic parameters
│   ├── settings.yaml        # Grid sizes, solver config
│   └── stages/              # Per-stage YAML + methods
├── solve.py                 # Backward induction pipeline
├── run.py                   # CLI entry point (uses kikku.run.cli)
├── callables.py             # Equation + transition callables (durables2_0)
├── horses/                  # Per-stage EGM operators (durables2_0)
├── outputs/                 # Plots, tables, diagnostics
├── simulate.py              # Forward simulation + Euler (durables2_0)
├── benchmark.py             # Timing / accuracy comparison
└── notebooks/               # Interactive notebooks
```

### Other directories

| Directory | What |
|-----------|------|
| `experiments/` | PBS job scripts for Gadi HPC (per-model subfolders) |
| `tests/` | pytest tests (`test_imports`, `test_retirement`, `test_kikku`, `test_durables2_simulate`) |
| `docs/` | MkDocs source → `site/` |
| `results/` | Local run outputs (timestamped, not committed) |
| `replication/` | Committed paper results |
| `AI/` | Development notes, prompts, design principles |
| `AI/prompts/` | Implementation briefs (instructions *to be executed*) |
| `AI/working/` | Session logs (records *of completed work*) |
| `AI/devspecs/` | Technical specifications for major changes |
| `AI/design-principles.md` | Coding rules (Ousterhout/Backus/FFP) |

## Critical principle: the calibrated stage IS the model

The dolo+ calibrated stage (from `load_syntax → instantiate_period → calibrate`)
is the single source of truth for parameters. Callables, grids, and operators
are all derived FROM it — never from a separate parameter object.

```
CORRECT:  stage.calibration['beta']  →  callable that uses beta
WRONG:    yaml_dict['beta']  →  SomeClass.beta  →  callable that uses obj.beta
```

If you recalibrate the stage (e.g. for a different age), all downstream
objects (callables, operators) must re-derive from the new calibration.
A parallel parameter object that shadows the stage breaks the DDSL contract.

The accretive solve loop follows this chain each period:
```
calibrate(syntactic_stage, params_h)  →  stage_h  →  callables_h  →  operators_h  →  solve
```

## Critical principle: stages are distinct objects — never conflate them

A period contains multiple stages (e.g. tenure, keeper_cons, adjuster_cons).
Each stage has its own symbols, equations, calibration, and identity. Code
must never treat one stage as interchangeable with another, even when they
happen to share parameter values today.

```
WRONG:   stage_h = next(iter(period["stages"].values()))  # grab "a" stage
         make_keeper_ops(..., stage_h)                     # pass it to keeper
         make_adjuster_ops(..., stage_h)                   # pass same object to adjuster

CORRECT: make_keeper_ops(..., period["stages"]["keeper_cons"])
         make_adjuster_ops(..., period["stages"]["adjuster_cons"])
```

**Why this matters**: each operator builder should receive the stage it
belongs to. Passing an arbitrary stage works only by accident — it relies
on all stages sharing the same calibration keys, which is not guaranteed
and hides the true dependency. When a stage later gains its own parameters,
the code silently uses the wrong values.

This is an instance of the deeper rule: **every function should receive
exactly the data it depends on, identified by name, not by position in
a container.** Grabbing "the first item" from a dict, or passing one
object where another is meant, creates invisible coupling that breaks
when the structure evolves.

## DDSL stage pipeline (how models work)

Every model follows this pipeline (from bellman-ddsl theory):

```
YAML syntax → load_syntax → instantiate_period → period_to_graph → backward_paths
                                                                         │
                                                              ┌──────────┴──────────┐
                                                              │  Accretive solve    │
                                                              │  for h = 0..H:     │
                                                              │    recalibrate(age) │
                                                              │    build operators  │
                                                              │    solve_period     │
                                                              └─────────────────────┘
```

Within each period, stages are solved in wave order. Each stage has:
- **B mover** (`cntn_to_dcsn`): EGM inversion or optimisation
- **I mover** (`dcsn_to_arvl`): expectation over shocks
- Period operator: **T = I ∘ B** (always B first, then I)

Three perches per stage:
- `[<]` arrival — post-expectation, pre-optimisation
- (unmarked) decision — post-optimisation
- `[>]` continuation — post-transition, pre-expectation

## Override contract: CLI shortcut → explicit patch → YAML target

The DDSL pipeline has four override surfaces, each corresponding to a
pipeline functor. Every override patches stage sources BEFORE
`instantiate_period`. After instantiation, the stage object is the sole
source of truth — no parallel dicts, no env vars, no raw YAML re-reads.

```
CLI shortcut           →  explicit patch target            →  YAML source
--method NEGM          →  expands via METHOD_SHORTCUT      →  methods.yml
                          to stage.target.scheme triples
--method-override      →  exact (stage, target, scheme)    →  methods.yml
  adj.upper_envelope=NEGM  triple patched directly
--calib-override       →  all stages (shared calibration)  →  calibration.yaml
  beta=0.96
--setting-override     →  all stages (shared settings)     →  settings.yaml
  n_a=500
```

Key principles:

- **Mover factories read method from `stage.methods`**, not from a
  solver-level method string. The solver stays method-agnostic; operator
  builders dispatch internally from the methodized stage.

- **The solver is method-agnostic** — no `if method == "NEGM"` branching
  in the backward loop; each horse/mover maker inspects its stage’s
  method bindings.

- **`--method` is a shortcut** that expands to the same patch path as
  `--method-override`: the model’s `METHOD_SHORTCUT` list supplies
  `(stage, target, scheme)` triples, each set to the given tag before
  `instantiate_period`.

- **At the CLI level**, `--method-override` accepts
  `stage.target.scheme=TAG` or shorthand `stage.scheme=TAG` (default
  target `cntn_to_dcsn_mover`). Multiple flags merge into one override
  dict; explicit `--method-override` entries override shortcut-expanded
  entries where they collide (merge order in the solver).

- **Shared calibration is a model-specific decision**, not a universal
  rule. In durables2_0, all stages share calibration — the solver
  documents this explicitly. A different model could have stage-specific
  calibration with per-stage override flags.

- **`method=None` means "use YAML default"** — the methods YAML
  declares the default method; the CLI only overrides when the user asks.

## Simulation and Euler: decoupled post-hoc evaluation

Simulation (forward walk) and Euler error computation are independent:

```
solve  →  nest (policies per period)
            ↓
simulate_lifecycle(nest, grids)  →  sim_data {c, a_nxt, h_nxt, z_idx, discrete}
            ↓
evaluate_euler_c(sim_data, nest, grids)  →  euler_c (T, N)
evaluate_euler_h(sim_data, nest, grids)  →  euler_h (T, N)
```

Euler errors are pure functions over simulated panels + solved policies.
They are NOT computed inside the forward walk (no side-channel mutation
in forward operators). This enables:
- Simulation without Euler (for lifecycle plots or debugging)
- Multiple error types from one simulation
- Post-hoc error evaluation on saved simulation data

Forward operators do pure simulation only — no `euler_panel` writes.

## Durables2_0 specifics

The active development model. Key conventions:

- **Stage is the model**: `stage.calibration` for economics, `stage.settings` for numerics. No `ConsumerProblem`, no parallel `settings` dict.
- **Grids as arguments**: operators receive grids at call time, not in closures
- **Per-period callables**: `make_callables(period_h, age)` called fresh each h — produces ALL callables including age-bound income transitions
- **Horse makers take `(callables_h, grids, stage)`**: one callables dict + the stage for that horse's numerical settings. Each horse maker receives its OWN stage.
- **Solution structure**: `{stage: {perch: {quantity: array}}}` (see `solution_scheme.md`)
- **Marginal values**: `d_aV`, `d_hV`, `d_wV` (flat keys, not nested)
- **Discrete choice**: `adj` (0/1 indicator), not `d`
- **`sol["callables"]`**: each solution entry carries its per-period callables for simulation/Euler use

## Scope rules for editing

- **Edit freely**: `examples/durables2_0/`, `examples/retirement/`
- **Do NOT edit**: `examples/housing_renting/` (separate development stage)
- **Do NOT edit**: `examples/old/` (baseline reference, never modify)

## Prompts vs working notes
- **Prompts** (`AI/prompts/DDMMYYYY/`): implementation briefs, design specs, and task descriptions that drive future work. These are instructions *to be executed*. Always put new prompts here.
- **Working notes** (`AI/working/DDMMYYYY/`): session logs recording what *was done* — conceptual lessons, decisions made, files changed, next steps. These are records *of completed work*.

## Examples and documentation
When changing anything under `examples/` (imports, entry points, CLI args, directory layout), check and update `examples/readme_examples.md` and `experiments/*/README.md` so run instructions stay accurate.

## CRITICAL: When removing code, verify the FULL function still works

When cleaning up, refactoring, or removing diagnostic/debug code from a
function, you MUST:

1. **Read the entire function** after your edit — not just the diff.
2. **Check every variable reference** still resolves. If you removed a
   variable definition (e.g. `_log = ...`), search for ALL uses of that
   variable in the same scope and remove them too.
3. **Check the function still returns**. If you deleted code that
   contained the `return` statement, the function silently returns `None`.
4. **Run `python -c "import ast; ast.parse(open('file.py').read())"`**
   to catch syntax errors. But note: `ast.parse` does NOT catch
   `NameError` — a deleted variable referenced later will only crash
   at runtime.
5. **If the function is inside a try/except** (like `_safe_criterion` or
   `make_criterion`'s closure), a crash will be SILENTLY swallowed and
   return a default value (e.g. `BIG_LOSS`). You will not see an error
   in logs. The only symptom is wrong results.

**Incident**: On 28 March 2026, removing diagnostic variables (`_log`,
`_rss_mb`, `_trial_call_count`) from `trial()` left behind an `if _log:`
block and lost the `return panels` statement. Every CE evaluation threw
`NameError`, silently caught as `BIG_LOSS = 1e10`. Multiple Gadi jobs
(~5000 SU) burned before the cause was found.

## Coding rules (project-wide)
### 1D interpolation
For 1D interpolation, always use functions from `dc_smm.fues.helpers.math_funcs`:
- `interp_as(xp, yp, x)` (array version)
- `interp_as_scalar(xp, yp, x)` (scalar version)

Do **not** use `np.interp` or `interpolation.interp` directly. If 1D interp needs optimization, do it inside these helper functions.
