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

### On Gadi

Start an interactive session, load the environment, then run:

```bash
qsub -I -q expresssr -P tp66 -l ncpus=1,mem=8GB,walltime=01:00:00,storage=scratch/tp66,wd
source setup/load_env.sh
python examples/retirement/run_experiment.py --params params/baseline.yml --grid-size 3000
```

Full timing sweep:

```bash
python examples/retirement/run_experiment.py --params params/baseline.yml --run-timings
```

Or use the PBS wrapper:

```bash
qsub experiments/retirement/retirement_timings.sh
```

### On a laptop

```bash
source .venv/bin/activate
python examples/retirement/run_experiment.py --params params/baseline.yml --grid-size 1000
```

## Output

Results are saved to the `--output-dir` path (default: `results/retirement/`):

- `plots/` — PNG figures (EGM grids, consumption policy, DCEGM comparison)
- `retirement_timing.{tex,md}` — Timing table (UE and total time per method)
- `retirement_accuracy.{tex,md}` — Accuracy table (Euler error and consumption deviation)

## Parameters

Model and benchmark settings are in YAML files under `examples/retirement/params/`:

| File | Description |
|------|-------------|
| `baseline.yml` | Standard Ishkakov et al (2017) parameterisation |
| `high_beta.yml` | Higher discount factor |
| `low_delta.yml` | Lower cost of working |
| `long_horizon.yml` | More periods |
| `sigma05.yml` | Smoothed discrete choice |

## File Layout

```
examples/retirement/
├── run_experiment.py          # Main entry point
├── params/*.yml               # Parameter files
├── code/
│   ├── retirement.py          # Model (RetirementModel, Operator_Factory)
│   ├── solve_block.py         # backward_induction (dolo-plus pipeline)
│   ├── benchmarks.py          # Timing sweep (test_Timings)
│   └── helpers/
│       ├── helpers.py         # euler(), get_policy(), get_timing(), consumption_deviation()
│       ├── plots.py           # Plotting functions
│       └── tables.py          # Table generation (LaTeX + Markdown)
├── syntax/                    # Stage YAML declarations (dolo-plus)
└── ...

experiments/retirement/
├── retirement_timings.sh      # PBS wrapper (batch submission)
├── run_retirement_single_core.sh
└── README.md                  # This file
```
