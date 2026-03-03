# Retirement Model Experiments

Timing and accuracy comparison of FUES vs DC-EGM vs RFC vs CONSAV for the
Ishkakov et al (2017) retirement choice model.

## Setup

Install the `dcsmm` package and all dependencies (one-time):

```bash
cd FUES
bash setup/setup_venv.sh
```

## Running

All runs go through the **canonical pipeline** (`solve_canonical`).
Baseline calibration lives in `examples/retirement/syntax/` — override
individual parameters via CLI flags or YAML files.

### On Gadi

Start an interactive session, load the environment, then run:

```bash
qsub -I -q expresssr -P tp66 -l ncpus=1,mem=8GB,walltime=01:00:00,storage=scratch/tp66,wd
source setup/load_env.sh
python examples/retirement/run.py --grid-size 3000 --output-dir results/retirement
```

Full timing sweep:

```bash
python examples/retirement/run.py --run-timings
```

Or use the PBS wrapper:

```bash
qsub experiments/retirement/retirement_timings.sh
```

### On a laptop

```bash
source .venv/bin/activate
python examples/retirement/run.py --grid-size 1000
```

### Override parameters

```bash
# Override economic params
python examples/retirement/run.py --calib-override beta=0.96

# Override numerical settings
python examples/retirement/run.py --config-override grid_size=5000 --config-override T=50

# Use an override file (sparse, flat key-value YAML)
python examples/retirement/run.py --override-file experiments/retirement/params/long_horizon.yml
```

## Output

Results are saved to the `--output-dir` path (default: `results/retirement/`):

- `plots/` — PNG figures (EGM grids, consumption policy, DCEGM comparison)
- `retirement_timing.{tex,md}` — Timing table (UE and total time per method)
- `retirement_accuracy.{tex,md}` — Accuracy table (Euler error and consumption deviation)

## Parameters

Canonical calibration and settings live in `examples/retirement/syntax/`:

| File | Description |
|------|-------------|
| `calibration.yaml` | Economic params: r, beta, delta, y, b, smooth_sigma |
| `settings.yaml` | Numerical settings: grid_size, grid_max_A, T, m_bar |

Sparse override files live in `experiments/retirement/params/`:

| File | Description |
|------|-------------|
| `baseline.yml` | Benchmark config (beta=0.96, T=50) |
| `high_beta.yml` | Higher discount factor (beta=0.99) |
| `low_delta.yml` | Lower cost of working (delta=0.5) |
| `long_horizon.yml` | Longer horizon (T=50, padding_mbar) |

Override files contain only values that differ from the canonical `syntax/` defaults.

## File Layout

```
examples/retirement/
├── run.py          # CLI entry point (uses solve_canonical)
├── model.py                   # Model (RetirementModel, Operator_Factory)
├── solve.py                   # Canonical pipeline (solve_canonical)
├── benchmark.py               # Timing sweep (via solve_canonical)
├── outputs/
│   ├── diagnostics.py         # euler(), get_policy(), get_timing()
│   ├── plots.py               # Plotting functions
│   └── tables.py              # Table generation (LaTeX + Markdown)
└── syntax/                    # dolo-plus YAML (single source of truth)
    ├── calibration.yaml       # Economic params
    ├── settings.yaml          # Numerical settings
    ├── period.yaml            # Stage topology
    └── stages/                # Per-stage YAML + methods

experiments/retirement/
├── params/                    # Sparse override files
├── retirement_timings.sh      # PBS wrapper (batch submission)
├── run_retirement_single_core.sh
└── README.md                  # This file
```
