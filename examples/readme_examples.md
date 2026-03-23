# Running the Examples

All commands use `python -m examples.<model>.run` from the repo root (`FUES/`).
Both runners auto-configure `sys.path`, so no `PYTHONPATH` manipulation is needed.

All model-specific settings live in `syntax/settings.yaml`; no model-specific
CLI args.  Override any setting via `--config-override key=value`.

Each run creates a timestamped output directory under `--output-dir`
(e.g. `results/durables2_0/2026-03-23_001/`).  Use `--run-tag NAME` to
replace the auto-timestamp with a fixed name.

---

## Basic run workflow

1. **Working directory** — From the repo root (`FUES/`), run modules as
   `python -m examples.<model>.run`.  No `PYTHONPATH` setup.

2. **Pick a model** — `retirement` or `durables2_0` (see sections below).

3. **Default single run** — Solves with that example’s defaults and writes
   plots/tables under `results/<model>/<run-id>/`.

   ```bash
   python -m examples.retirement.run
   python -m examples.durables2_0.run --method FUES
   ```

4. **Durables: FUES vs NEGM in one run** — One invocation solves both,
   prints/saves a comparison table, and writes plots under per-method
   subfolders (`plots/FUES/`, `plots/NEGM/`).

   ```bash
   python -m examples.durables2_0.run --compare
   ```

   Add **`--simulate`** to the same line to run forward simulation and fill
   Euler / adj-rate columns in the table (plus lifecycle plots under
   `plots/simulation/`).

   ```bash
   python -m examples.durables2_0.run --compare --simulate --n-sim 10000 --seed 42
   ```

5. **Single-method simulation (Durables)** — After a one-method solve
   (`--method FUES` or `NEGM`), add `--simulate` for Euler diagnostics
   and lifecycle plots (same flags as above without `--compare`).

6. **Tune calibration vs numerics** — Use `--calib-override` for economic
   parameters (e.g. `t0`, `beta`) and `--config-override` for grids and
   solver flags (repeatable).  Shorthand: `--grid-size N`.

7. **Name your output folder** — By default the run id is timestamped;
   use `--run-tag myrun` for a stable path (e.g. `results/durables2_0/myrun/`).

8. **Where to look** — Under the run directory: `plots/` (figures),
   `tables/` (markdown/LaTeX where applicable).  See each model’s
   **Output structure** subsection.

---

## Retirement

Upper-envelope comparison for a discrete-continuous lifecycle consumption–savings
model with voluntary retirement.  Methods: `RFC`, `FUES`, `DCEGM`, `CONSAV`.

### Single-point solve

```bash
python -m examples.retirement.run
```

### Override calibration or numerical settings

```bash
python -m examples.retirement.run --calib-override beta=0.96
python -m examples.retirement.run --config-override grid_size=5000
python -m examples.retirement.run --grid-size 5000          # shorthand
python -m examples.retirement.run --override-file experiments/retirement/params/high_beta.yml
```

### Choose a specific EGM plot age

```bash
python -m examples.retirement.run --config-override plot_age=10
```

### Timing sweep (FUES vs DCEGM vs RFC vs CONSAV)

```bash
python -m examples.retirement.run --config-override run_timings=1 \
    --sweep-grids 500,1000,2000,3000,10000 \
    --sweep-runs 3
```

Override delta values:

```bash
python -m examples.retirement.run --config-override run_timings=1 \
    --config-override sweep_deltas=0.25,0.5,1
```

LaTeX tables for a subset of grids:

```bash
python -m examples.retirement.run --config-override run_timings=1 \
    --config-override latex_grids=1000,3000,10000
```

### Run tagging

```bash
# Auto-timestamped (default)
python -m examples.retirement.run
# → results/retirement/2026-03-23_001/

# Named tag
python -m examples.retirement.run --run-tag baseline
# → results/retirement/baseline/
```

### Output structure

```
results/retirement/
└── 2026-03-23_001/          # timestamped or tagged run dir
    ├── plots/               # consumption policy, EGM grids, DCEGM comparison (PNG)
    └── tables/              # timing and accuracy tables (LaTeX + Markdown)
```

---

## Durables2_0 (DDSL)

Two-asset discrete-continuous housing/durables model with tenure choice.
Methods: `FUES` (default), `NEGM`.

### Method comparison (FUES vs NEGM)

```bash
python -m examples.durables2_0.run --compare
python -m examples.durables2_0.run --compare --calib-override t0=50
```

### Single-point solve

```bash
python -m examples.durables2_0.run --method FUES
python -m examples.durables2_0.run --method NEGM
```

### Override calibration or numerical settings

```bash
python -m examples.durables2_0.run --calib-override t0=50
python -m examples.durables2_0.run --config-override n_a=300
python -m examples.durables2_0.run --grid-size 500
python -m examples.durables2_0.run --override-file experiments/durables/params/high_dep.yml
```

### Forward simulation with Euler errors

```bash
python -m examples.durables2_0.run --simulate --n-sim 10000 --seed 42
python -m examples.durables2_0.run --simulate --config-override init_method=empirical
```

### EGM grid diagnostics

```bash
python -m examples.durables2_0.run --config-override store_cntn=1
python -m examples.durables2_0.run --config-override store_cntn=1 --config-override plot_ages=69,65,60,55,50
```

### Parameter sweep (grid-size timing)

```bash
python -m examples.durables2_0.run --sweep --sweep-grids 100,200,300 --sweep-runs 3
```

### Run tagging

```bash
# Auto-timestamped (default)
python -m examples.durables2_0.run --compare
# → results/durables2_0/2026-03-23_001/

# Named tag
python -m examples.durables2_0.run --compare --run-tag baseline
# → results/durables2_0/baseline/
```

### Output structure

```
results/durables2_0/
└── 2026-03-23_001/                      # timestamped or tagged run dir
    ├── plots/
    │   ├── FUES/                        # FUES outputs (--compare)
    │   │   └── age_{t}/                 # policy plots per age (PNG)
    │   ├── NEGM/                        # NEGM outputs (--compare)
    │   │   └── age_{t}/
    │   ├── age_{t}/                     # single-method policy plots (PNG)
    │   └── simulation/
    │       └── lifecycle.png            # lifecycle profiles (--simulate)
    └── tables/
        ├── method_comparison.md         # FUES vs NEGM table (--compare)
        └── sweep.md                     # sweep timing table (--sweep)
```

---

## Generic `--config-override` pattern

All model-specific knobs live in `syntax/settings.yaml` and are overridden
via `--config-override key=value` (repeatable).  Examples:

```bash
# Durables: plot specific ages, store EGM grids
python -m examples.durables2_0.run --config-override plot_ages=50,55 --config-override store_cntn=1

# Retirement: change plot age, enable timings
python -m examples.retirement.run --config-override plot_age=10 --config-override run_timings=1
```

---

## MPI Parallelism (future)

The sweep infrastructure in `kikku.run.sweep` already accepts a `comm` argument
for distributing parameter-grid points across MPI ranks.  When MPI support is
wired in, a launcher would look like:

```python
from mpi4py import MPI

def get_comm():
    return MPI.COMM_WORLD

comm = get_comm()
run_sweep(..., comm=comm)
```

No changes to `run.py` are required — the `comm` is threaded through `sweep()`
which partitions the grid and gathers results on rank 0.  CLI entry points will
stay serial; MPI dispatch will be handled by a thin experiment-level script or
PBS/Slurm wrapper that calls `run_sweep` directly.
