# Running Locally

From the repo root, run `python -m examples.<model>.run`.

## Models

| Model | Command | Stages |
|-------|---------|--------|
| **Durables** | `python -m examples.durables.run` | tenure, keeper_cons, adjuster_cons |
| **Retirement** | `python -m examples.retirement.run` | labour_mkt_decision, work_cons, retire_cons |

## Output

Every run creates `<output-dir>/YYYY-MM-DD/NNN/` with auto-incremented NNN:

```
results/durables/2026-03-25/001/
├── plots/
│   ├── age_68/            # policy plots
│   └── simulation/        # lifecycle plots
└── tables/
    ├── sweep.md           # sweep results
    ├── sweep.tex          # LaTeX (paper format)
    ├── comparison.md      # compare mode
    └── summary.md         # single-run
```

Default: `results/<model>/`. Override with `--output-dir`.

## Overrides

Four tiers — the runner routes each key to the correct place:

| Tier | Flag | What it patches |
|------|------|-----------------|
| Economic parameters | `--calib-override key=val` | `calibration.yaml` |
| Numerical settings | `--setting-override key=val` | `settings.yaml` |
| Solution methods | `--method-override stage.scheme=TAG` | methods YAML |
| Experiment config | `--config-override key=val` | runner-only knobs |

```bash
# Override calibration
--calib-override beta=0.96 --calib-override tau=0.12

# Override settings
--setting-override n_a=500 --setting-override n_h=300

# Override method on a specific stage
--method-override keeper_cons.upper_envelope=MSS

# Shortcut: --method patches the model's default target
--method NEGM

# Bulk overrides from YAML
--override-file params/baseline.yml
```

Precedence: `--method-override` > `--method` > YAML default. `--calib-override` > `--override-file` > `calibration.yaml`.

## Modes

### Single solve (default)

```bash
python -m examples.durables.run
python -m examples.durables.run --method NEGM
python -m examples.durables.run --simulate --n-sim 10000
```

### Compare

```bash
# Adjuster methods
python -m examples.durables.run --compare FUES NEGM --simulate

# Keeper methods (override specs)
python -m examples.durables.run \
    --compare keeper_cons.upper_envelope=FUES keeper_cons.upper_envelope=MSS

# Three-way
python -m examples.durables.run \
    --compare FUES NEGM keeper_cons.upper_envelope=MSS --simulate
```

### Sweep

```bash
# Grid convergence
python -m examples.durables.run --sweep --sweep-grids 100,200,300

# Multi-axis: grid × tau × method (paper table)
python -m examples.durables.run --sweep \
    --sweep-params n_a=250,500,750,1000 tau=0.05,0.07,0.12 method=FUES,NEGM \
    --simulate --n-sim 10000
```

## Durables examples

```bash
# Default (FUES, separable utility)
python -m examples.durables.run

# NEGM adjuster
python -m examples.durables.run --method NEGM

# MSS at keeper
python -m examples.durables.run --method-override keeper_cons.upper_envelope=MSS

# Finer grids
python -m examples.durables.run --setting-override n_a=600 --setting-override n_h=600

# Shorter horizon (faster)
python -m examples.durables.run --calib-override t0=55

# EGM grid diagnostics
python -m examples.durables.run --setting-override store_cntn=1

# Paper table
python -m examples.durables.run --sweep \
    --sweep-params n_a=250,500,750,1000 tau=0.05,0.07,0.12 method=FUES,NEGM \
    --simulate --n-sim 10000 --calib-override t0=20
```

## Retirement examples

```bash
# Default (all 4 UE methods)
python -m examples.retirement.run

# Override method
python -m examples.retirement.run --method RFC

# Timing benchmark
python -m examples.retirement.run \
    --setting-override run_timings=1 \
    --sweep-grids 1000,2000,3000,5000,10000 --sweep-runs 3

# Calibration override
python -m examples.retirement.run --calib-override beta=0.96
```

## Installation

Three tiers — pick one:

| Install | Gets you | When to use |
|---|---|---|
| `pip install -e .` | FUES + EGM_UE benchmarks (numpy, numba, scipy, interpolation, ConSav, HARK) | Just the algorithm — compare FUES vs DCEGM vs ConSav upper envelopes |
| `pip install -e ".[examples]"` | Above + matplotlib, seaborn, tqdm, pyyaml, quantecon, dill, pykdtree, `kikku[estimation]` | Run the durables + retirement pipelines, notebooks, Gadi sweeps |
| `pip install -e ".[dev]"` | `[examples]` + pytest + autopep8 | Contributing / running the test suite |

On Gadi (or local), `source setup/setup.sh` installs `[examples]` and verifies HARK / kikku / mpi4py in one shot. Re-run any time to just activate; pass `--update` to `git pull` + reinstall.
