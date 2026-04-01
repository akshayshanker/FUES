# Retirement Experiments

Timing and accuracy: FUES vs DC-EGM vs RFC vs CONSAV on the Iskhakov et al. (2017) retirement model.

## Quick start

```bash
# Gadi (interactive)
qsub -I -q expresssr -P tp66 -l ncpus=1,mem=8GB,walltime=01:00:00,storage=scratch/tp66,wd
source setup/load_env.sh
bash experiments/retirement/run_retirement_single_core.sh

# Laptop
source .venv/bin/activate
python examples/retirement/run.py --grid-size 1000
```

## Timing sweep

```bash
# PBS batch
qsub experiments/retirement/retirement_timings.sh

# Or directly
python examples/retirement/run.py --run-timings
```

## Parameter overrides

Override files in `params/` are sparse YAML — only values that differ from `examples/retirement/syntax/` defaults.

```bash
python examples/retirement/run.py --override-file experiments/retirement/params/baseline.yml
python examples/retirement/run.py --calib-override beta=0.96 --config-override grid_size=5000
```

| File | Key changes |
|------|-------------|
| `params/baseline.yml` | beta=0.96, T=50 |
| `params/high_beta.yml` | beta=0.99 |
| `params/low_delta.yml` | delta=0.5 |
| `params/long_horizon.yml` | T=50, padding_mbar |

## Output

Each run creates a **timestamped folder** named `retirement_YYYYMMDD_HHMMSS`:

- **On Gadi**: `/scratch/tp66/$USER/FUES/solutions/retirement/retirement_YYYYMMDD_HHMMSS/`
- **Locally**: `results/retirement/retirement_YYYYMMDD_HHMMSS/`

Override with `--output-dir` or the `OUTPUT_DIR` variable in `retirement_timings.sh`.

Contents of each run folder:

```
retirement_YYYYMMDD_HHMMSS/
├── plots/              # PNG plots (EGM grids, policy functions, value functions)
├── timings.md          # Markdown timing table (all grid × delta combos)
├── timings.tex         # LaTeX timing table (subset for paper)
└── ...                 # Additional solver outputs
```

## Logs

Logs are written to `$BASE_OUT/logs/retirement/` (scratch on Gadi, `/tmp` locally):

```
logs/retirement/
├── retirement_YYYYMMDD_HHMMSS.log   # stdout
└── retirement_YYYYMMDD_HHMMSS.err   # stderr
```

The `RUN_ID` can be overridden via environment variable: `RUN_ID=myrun qsub retirement_timings.sh`.

## Configuration (retirement_timings.sh)

Key variables at the top of the script:

| Variable | Default | Description |
|----------|---------|-------------|
| `PARAMS_FILE` | `params/baseline.yml` | Override file for calibration/settings |
| `GRID_SIZE` | 2000 | Baseline grid size for plots |
| `PLOT_AGE` | 16 | Age to plot EGM grids |
| `RUN_TIMINGS` | true | Run full timing comparison |
| `SWEEP_GRIDS` | 1000–15000 | Grid sizes for sweep |
| `SWEEP_DELTAS` | 0.25, 0.5, 1, 2 | Delta values for sweep |
| `SWEEP_RUNS` | 3 | Number of runs per config (best of n) |
| `LATEX_GRIDS` | 1000–10000 | Subset for paper LaTeX tables |
