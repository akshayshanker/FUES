# Retirement Model Experiments

Timing comparison of FUES vs DC-EGM vs CONSAV for the Ishkakov et al (2017) retirement model.

## Quick Start

```bash
cd FUES
./experiments/retirement/retirement_timings.sh
```

## Settings

Edit the top of `retirement_timings.sh`:

```bash
# Baseline model settings
GRID_SIZE=3000              # Grid size for plots
PLOT_AGE=17                 # Age to plot EGM grids
OUTPUT_DIR=""               # Leave empty for default

# Timing sweep settings
RUN_TIMINGS=false                        # Set true for full sweep
SWEEP_GRIDS="500,1000,2000,3000,10000"   # Grid sizes
SWEEP_DELTAS="0.25,0.5,1,2"              # Delta values
SWEEP_RUNS=3                             # Runs per config
```

Or pass as environment variables:
```bash
GRID_SIZE=5000 PLOT_AGE=10 ./experiments/retirement/retirement_timings.sh
```

## Running Python Directly

```bash
cd FUES
python experiments/retirement/run_experiment.py --help
python experiments/retirement/run_experiment.py --grid-size 3000 --plot-age 5
python experiments/retirement/run_experiment.py --run-timings  # Full sweep
```

## Running on Gadi

**Interactive session:**
```bash
qsub -I -P tp66 -q expressbw -l ncpus=1,mem=25GB,walltime=02:00:00
cd $HOME/dev/fues.dev/FUES
./experiments/retirement/retirement_timings.sh
```

**Batch job:**
```bash
qsub experiments/retirement/retirement_timings.sh
```

## Output

Results saved to `FUES/results/retirement/`:
- `plots/` - PNG figures
- `retirement_timing.tex/.md` - Timing table (UE and Total time per method)
- `retirement_accuracy.tex/.md` - Accuracy table (Euler error and consumption deviation)

## Benchmark Metrics

The timing benchmarks compare four upper envelope methods: **RFC**, **FUES**, **DCEGM**, and **CONSAV**.

### Timing Table
- **UE**: Average upper envelope computation time per period (ms)
- **Tot**: Total solution time including all periods (ms)

### Accuracy Table
- **Euler**: Euler equation error, `log₁₀(|u'(c) - βR·E[u'(c')]| / u'(c))`
- **Dev**: Consumption deviation from true solution, `log₁₀(|c - c_true| / c_true)`

The "true" reference solution is computed using a high-resolution grid (default: 20,000 points with DCEGM). This is configurable via YAML parameters.

## Parameter Configuration

Model and benchmark parameters are configured via YAML files in `params/`:

```yaml
# params/baseline.yml
model:
  r: 0.02
  beta: 0.98
  delta: 1.0
  # ... other model params

benchmark:
  true_grid_size: 20000  # Grid size for reference solution
  true_method: DCEGM     # Method for reference (RFC, FUES, DCEGM, CONSAV)
```

Use different parameter files:
```bash
python experiments/retirement/run_experiment.py --params params/long_horizon.yml --run-timings
```

## Files

```
experiments/retirement/
├── retirement_timings.sh  # Bash wrapper (settings, env setup)
├── run_experiment.py      # Python experiment runner
└── README.md

examples/retirement/
├── plots.py               # Plotting functions
├── tables.py              # Table generation
└── benchmarks.py          # Timing sweep logic
```
