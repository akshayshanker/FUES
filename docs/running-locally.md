# Running locally

This page assumes the repo has been cloned and `source setup/setup.sh`
has run successfully. See [Installation](getting-started/installation.md)
for the setup options. All commands below are issued from the repo root.

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
│   └── simulation/      # lifecycle plots
└── tables/
    ├── sweep.md         # sweep results
    ├── sweep.tex        # LaTeX (paper format)
    ├── comparison.md   # compare mode
    └── summary.md       # single-run
```

Default: `results/<model>/`. Override with `--output-dir`.

## Overrides (kikku RunSpec v2)

Tiers and merge order follow `kikku.run.parse_cli` — the runner maps each
key to the params / settings / methods YAMLs declared under `base_spec`.

| Tier | Flag | What it patches |
|------|------|-----------------|
| Economic parameters | `--params-override key=val` | calibration (params) |
| Numerical settings | `--settings-override key=val` | `settings` |
| Method slots (dotted) | `--methods-override stg.mover.scheme=TAG` | methods YAML |
| Merged from file | `--override-file path.yml` | params + settings split by key |

```bash
# Override calibration
--params-override beta=0.96 --params-override tau=0.12

# Override settings
--settings-override n_a=500 --settings-override n_h=300

# Override a method path (same string as in the methods YAML)
--methods-override adjuster_cons.cntn_to_dcsn_mover.upper_envelope=NEGM

# Bulk overrides from YAML
--override-file params/baseline.yml
```

Precedence is argv merge order, then the base YAMLs. A single string tag such
as `FUES` or `NEGM` is still given **via** `--methods-override` on the
`upper_envelope` path for the relevant stage, not a bare `--method` flag (that
form was removed in RunSpec v2).

## Modes

### Single solve (default)

```bash
python -m examples.durables.run
python -m examples.durables.run \
  --methods-override adjuster_cons.cntn_to_dcsn_mover.upper_envelope=NEGM
python -m examples.durables.run --simulate --n-sim 10000
```

### Compare

```bash
# Adjuster FUES vs NEGM (method path = adjuster stage upper_envelope slot)
python -m examples.durables.run \
  --compare adjuster_cons.cntn_to_dcsn_mover.upper_envelope=FUES,NEGM \
  --simulate
```

### Sweep

```bash
# One-axis settings sweep (grid_size example)
python -m examples.durables.run --sweep \
  --settings-range '[{"grid_size":100},{"grid_size":200},{"grid_size":300}]'

# Multi-axis (paper table): use --params-range / --settings-range / --methods-range
# with JSON lists of partial override dicts; kikku takes the Cartesian product.
python -m examples.durables.run --sweep \
  --params-range '[{"n_a":250,"tau":0.05},{"n_a":500,"tau":0.12}]' \
  --methods-range '[{"adjuster_cons.cntn_to_dcsn_mover.upper_envelope":"FUES"},{"adjuster_cons.cntn_to_dcsn_mover.upper_envelope":"NEGM"}]' \
  --simulate --n-sim 10000
```

## Durables examples

```bash
# Default (FUES on adjuster, separable utility)
python -m examples.durables.run

# NEGM adjuster
python -m examples.durables.run \
  --methods-override adjuster_cons.cntn_to_dcsn_mover.upper_envelope=NEGM

# Finer grids
python -m examples.durables.run --settings-override n_a=600 --settings-override n_h=600

# Shorter horizon (faster)
python -m examples.durables.run --params-override t0=55

# EGM grid diagnostics
python -m examples.durables.run --settings-override store_cntn=1

# Paper table (abridged — use full product JSON for a complete replication)
python -m examples.durables.run --sweep \
  --params-override t0=20 \
  --simulate --n-sim 10000
```

## Retirement examples

```bash
# Default (all 4 UE methods)
python -m examples.retirement.run

# Two UE tags on the work_cons upper_envelope slot
python -m examples.retirement.run \
  --compare work_cons.cntn_to_dcsn_mover.upper_envelope=FUES,RFC

# Timing benchmark: full Cartesian sweep via kikku ranges (post-solve plot uses
# the largest grid in the test set; see experiments/retirement/retirement_timings.sh)
python -m examples.retirement.run --sweep \
  --params-range '[{"delta":0.5}]' \
  --settings-range '[{"grid_size":1000},{"grid_size":2000},{"grid_size":3000}]' \
  --sweep-runs 3

# Params
python -m examples.retirement.run --params-override beta=0.96
```

## See also

- [Installation](getting-started/installation.md) for environment setup.
- [Running on PBS / Gadi](running-on-gadi.md) for cluster runs and estimation sweeps.
- [Applications](examples/index.md) for the model-level context behind each `run.py`.
