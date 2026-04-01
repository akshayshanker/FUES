# Running the examples

From the repo root (`FUES/`), run `python -m examples.<model>.run`.
Each run writes to a timestamped folder under `--output-dir` (default `results/<model>/YYYY-MM-DD_NNN`). Use `--run-tag NAME` for a memorable label instead of the auto-increment.

---

## How the CLI works

Every example loads two YAML files from its `syntax/` directory before solving:

- **`calibration.yaml`** — economic parameters (preferences, returns, shocks, transaction costs)
- **`settings.yaml`** — numerical settings (grid sizes, tolerances, horizon, FUES threshold `m_bar`)

The CLI lets you override any value without editing YAML. Overrides are routed to the correct tier automatically — the runner checks which file a key belongs to and errors if you put it in the wrong tier.

### Override tiers

| Tier | Flag | What it patches | Error if misrouted? |
|------|------|-----------------|---------------------|
| Economic parameters | `--calib-override key=val` | Keys in `calibration.yaml` | Yes — settings keys are rejected |
| Numerical settings | `--setting-override key=val` | Keys in `settings.yaml` | Yes — calibration keys are rejected |
| Solution methods | `--method-override stage.scheme=TAG` | Method tags in stage methods YAML | Raises if stage/scheme not found |
| Experiment config | `--config-override key=val` | Runner-only knobs not in either YAML | Warns if name collides |

All flags are repeatable: `--calib-override beta=0.96 --calib-override tau=0.15`.

**`--override-file path.yml`** loads overrides from a YAML file. Keys are auto-routed to the correct tier.

Values are coerced via YAML typing: `key=0.96` → float, `key=true` → bool, `key=50,55` → string.

### Method overrides

Each stage declares its solution methods in `syntax/stages/<stage>/<stage>_methods.yml`. The methods YAML specifies which **schemes** (upper_envelope, interpolation, etc.) to use on each **mover** (cntn_to_dcsn_mover, dcsn_to_arvl_mover) and which **algorithm tag** implements each scheme.

**`--method TAG`** is a convenience shortcut. Each model declares which stages it patches — typically the upper-envelope method on the main consumption stage. Omit it to use the YAML default.

**`--method-override stage.scheme=TAG`** overrides any specific scheme on any stage. When disambiguation is needed, use the full address `stage.mover.scheme=TAG`.

```bash
# Shortcut: patches the model's default target (adjuster UE in durables)
--method NEGM

# Explicit: override one stage's upper envelope
--method-override keeper_cons.upper_envelope=MSS

# Override two stages independently
--method-override adjuster_cons.upper_envelope=NEGM --method-override keeper_cons.upper_envelope=MSS
```

**Precedence**: `--method-override` > `--method` (shortcut) > YAML default.

### Comparison mode

**`--compare SPEC1 SPEC2 [SPEC3 ...]`** solves with each method configuration and produces a combined comparison table. Each spec is either a bare method name (expanded via the model's shortcut) or a `stage.scheme=TAG` override:

```bash
# Bare names: compare adjuster methods (FUES vs NEGM)
--compare FUES NEGM

# Override specs: compare keeper UE methods
--compare keeper_cons.upper_envelope=FUES keeper_cons.upper_envelope=MSS

# Mix bare + override: three-way comparison
--compare FUES NEGM keeper_cons.upper_envelope=MSS

# Add simulation for Euler error columns
--compare FUES NEGM --simulate --n-sim 10000
```

`--compare` and `--sweep` are mutually exclusive.

### Output directory layout

Every run creates a self-contained directory under `results/<model>/`:

```
results/durables/
└── 2026-03-25_001/                        # auto-dated, or --run-tag name
    │
    │  ── Single-method run ──
    ├── plots/
    │   ├── age_68/                        # policy plots per cohort
    │   │   ├── policy_keeper_consumption.png
    │   │   ├── policy_adj_*.png
    │   │   ├── discrete_choice.png
    │   │   └── value_keeper.png
    │   └── simulation/                    # with --simulate
    │       └── lifecycle.png
    └── tables/
        └── summary.md                     # timing + Euler (if simulated)
```

```
    │  ── Comparison run ──
    ├── FUES/                              # one subfolder per compared method
    │   ├── plots/age_68/...
    │   ├── plots/simulation/lifecycle.png
    │   └── tables/summary.md
    ├── NEGM/
    │   └── ...
    ├── keeper_cons.upper_envelope=MSS/    # override-spec labels
    │   └── ...
    └── tables/                            # combined comparison
        ├── comparison.md
        ├── comparison.tex
        └── euler_detail.md                # with --simulate
```

```
    │  ── Sweep run ──
    └── tables/
        └── sweep.md                       # one row per grid size
```

---

## Durables2_0 (`examples.durables.run`)

Two-asset lifecycle model: liquid assets **a**, housing **h**, discrete tenure choice (keep/adjust with transaction cost **τ**), CRRA preferences over **(c, h)**. Three DDSL stages per period: `keeper_cons`, `adjuster_cons`, `tenure`.

The `--method` shortcut targets the **adjuster** stage's upper envelope (FUES vs NEGM). Use `--method-override` to target any stage independently.

### Examples

```bash
# Default solve (FUES everywhere)
python -m examples.durables.run

# NEGM adjuster (keeper stays FUES)
python -m examples.durables.run --method NEGM

# MSS at the keeper (adjuster stays FUES)
python -m examples.durables.run --method-override keeper_cons.upper_envelope=MSS

# Compare adjuster methods
python -m examples.durables.run --compare FUES NEGM --simulate --n-sim 10000

# Compare keeper methods
python -m examples.durables.run \
    --compare keeper_cons.upper_envelope=FUES keeper_cons.upper_envelope=MSS \
    --simulate --n-sim 10000

# Three-way: FUES everywhere vs NEGM adjuster vs MSS keeper
python -m examples.durables.run \
    --compare FUES NEGM keeper_cons.upper_envelope=MSS \
    --simulate --n-sim 10000

# Finer grids
python -m examples.durables.run \
    --setting-override n_a=300 --setting-override n_h=300 --setting-override n_w=300

# Shorter horizon (faster for testing)
python -m examples.durables.run --calib-override t0=55

# EGM grid diagnostics
python -m examples.durables.run \
    --setting-override store_cntn=1 --setting-override plot_ages=69,65,60

# Grid-convergence sweep
python -m examples.durables.run --sweep --sweep-grids 100,200,300 --sweep-runs 3

# Sweep + simulation (Euler accuracy vs grid size)
python -m examples.durables.run \
    --sweep --sweep-grids 100,200,300 --simulate --n-sim 5000

# Named output directory
python -m examples.durables.run --compare FUES NEGM --run-tag paper_baseline
```

---

## Retirement (`examples.retirement.run`)

Lifecycle consumption-savings with voluntary retirement and discrete labour supply. Solves with four upper-envelope methods: **RFC**, **FUES**, **DCEGM** (Iskhakov et al. 2017), **CONSAV** (Jorgensen 2021). The `--method` shortcut targets all stages' upper envelope simultaneously.

### Examples

```bash
# Default solve + plots (FUES)
python -m examples.retirement.run

# Override method for all stages
python -m examples.retirement.run --method RFC

# Override one stage independently
python -m examples.retirement.run --method-override work_cons.upper_envelope=DCEGM

# Calibration override
python -m examples.retirement.run --calib-override beta=0.96

# Finer grid
python -m examples.retirement.run --setting-override grid_size=5000

# Timing benchmark
python -m examples.retirement.run --setting-override run_timings=1 \
    --sweep-grids 500,1000,2000,3000,10000 --sweep-runs 3
```

---

## Override system (reference)

Four override surfaces, each patching a different layer of the DDSL pipeline:

| Surface | Flag | YAML source | Validated? |
|---------|------|-------------|-----------|
| Economic parameters | `--calib-override` | `syntax/calibration.yaml` | Hard error if key is in settings |
| Numerical settings | `--setting-override` | `syntax/settings.yaml` | Hard error if key is in calibration |
| Solution methods | `--method-override` | `syntax/stages/*/…_methods.yml` | Error if stage/scheme not found |
| Experiment config | `--config-override` | *(not in YAML)* | Warning if name collides |

Method override address format:
- **Short**: `stage.scheme=TAG` (target defaults to `cntn_to_dcsn_mover`)
- **Full**: `stage.mover.scheme=TAG`

---

## MPI sweeps (HPC)

Grid sweeps can be distributed across MPI ranks. The sweep function in `kikku.run.sweep` accepts an optional `comm` communicator — each rank evaluates one parameter point.

A typical PBS job script:

```bash
mpirun -np 48 python -m examples.durables.run \
    --sweep --sweep-grids 100,200,300,500,1000 \
    --simulate --n-sim 10000 \
    --run-tag sweep_paper
```

For large experiment matrices (many grid sizes × methods × calibrations), define the sweep in an **experiment-set YAML** under `experiments/<model>/experiment_sets/`:

```yaml
# experiments/durables/experiment_sets/paper_sweep.yml
sweep:
  methods: [FUES, NEGM]
  grid_sizes: [100, 200, 300, 500, 1000]

fixed:
  t0: 40
  n_sim: 10000

metrics:
  - euler_error
```

Pass to the runner: `--experiment-set experiments/durables/experiment_sets/paper_sweep.yml`.

Each MPI rank solves one `(method, grid_size)` combination. Results are gathered on rank 0 and written to a single table. The experiment-set YAML is the reproducible record of what was run.
