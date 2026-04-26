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

## Overrides (kikku RunSpec v3)

All overrides go through **slots** named in the model’s `spec_factory.yaml`
(typically `$draw` for calibration + settings, `$method_switch` for methods
YAML-shaped blocks). Merge order is argv order (see `kikku.run.parse_cli`).

| Use case | v3 |
|----------|-----|
| Per-key cal / settings | `slot-override` with `$draw.KEY=VAL` (FUES examples route both through the `draw` slot) |
| Methods YAML patches | `slot-spec` with a file or inline JSON; top-level `method_switch` (nested methods block) |
| Bundle from disk | `slot-spec @path` (YAML must use top-level **slot** keys such as `draw:`) |
| Sweeps (one or more axes) | Repeat `slot-range` (list of bundle dicts per flag); Cartesian product |

```bash
# Economic parameters and numerics (FUES: both route through $draw in examples)
--slot-override '$draw.beta=0.96' --slot-override '$draw.tau=0.12'
--slot-override '$draw.n_a=500' --slot-override '$draw.n_h=300'

# NEGM on adjuster upper_envelope: pass a method_switch slot (string tag expands in solve)
--slot-spec='{"method_switch":"NEGM"}'

# Bulk slot bundle from YAML (top-level keys = slot names; see experiments/retirement/params/baseline.yml)
--slot-spec @experiments/retirement/params/baseline.yml
```

String tags like `FUES` or `NEGM` for the upper envelope are typically passed
as the `method_switch` **slot** value (via `--slot-spec` or a small YAML file),
not as a removed bare `--method` flag.

## Modes

### Single solve (default)

```bash
python -m examples.durables.run
python -m examples.durables.run \
  --slot-spec='{"method_switch":"NEGM"}'
python -m examples.durables.run --simulate --n-sim 10000
```

### Compare (single scalar axis on one slot)

```bash
# Compare two calibration values on $draw.beta
python -m examples.durables.run \
  --compare '$draw.beta=0.92,0.96' \
  --simulate
```

For two method tags, use a two-row `--slot-range` (or `--sweep` with one axis)
instead of `--compare`, because method blocks are not one-level `slot.subkey`.

### Sweep

```bash
# One-axis settings sweep
python -m examples.durables.run --sweep \
  --slot-range='[{"draw":{"grid_size":100}},{"draw":{"grid_size":200}},{"draw":{"grid_size":300}}]'

# Multi-axis: repeated --slot-range; kikku takes the Cartesian product
python -m examples.durables.run --sweep \
  --slot-range='[{"draw":{"n_a":250,"tau":0.05}},{"draw":{"n_a":500,"tau":0.12}}]' \
  --slot-range='[{"method_switch":"FUES"},{"method_switch":"NEGM"}]' \
  --simulate --n-sim 10000
```

## Durables examples

```bash
# Default (FUES on adjuster, separable utility)
python -m examples.durables.run

# NEGM adjuster
python -m examples.durables.run \
  --slot-spec='{"method_switch":"NEGM"}'

# Finer grids
python -m examples.durables.run --slot-override '$draw.n_a=600' --slot-override '$draw.n_h=600'

# Shorter horizon (faster)
python -m examples.durables.run --slot-override '$draw.t0=55'

# EGM grid diagnostics
python -m examples.durables.run --slot-override '$draw.store_cntn=1'

# Paper table (abridged)
python -m examples.durables.run --sweep \
  --slot-override '$draw.t0=20' \
  --simulate --n-sim 10000
```

## Retirement examples

```bash
# Default (all 4 UE methods, expanded by the runner when no method_switch)
python -m examples.retirement.run

# Two method tags: two-row slot-range
python -m examples.retirement.run --sweep \
  --slot-range='[{"method_switch":"FUES"},{"method_switch":"RFC"}]'

# Timing benchmark: full Cartesian sweep (see experiments/retirement/retirement_timings.sh)
python -m examples.retirement.run --sweep \
  --slot-range @experiments/retirement/timing_deltas.yaml \
  --slot-range @experiments/retirement/timing_grids.yaml \
  --slot-range @experiments/retirement/timing_methods.yaml \
  --sweep-runs 3

# Params
python -m examples.retirement.run --slot-override '$draw.beta=0.96'
```

## See also

- [Installation](getting-started/installation.md) for environment setup.
- [Running on PBS / Gadi](running-on-gadi.md) for cluster runs and estimation sweeps.
- [Applications](examples/index.md) for the model-level context behind each `run.py`.
