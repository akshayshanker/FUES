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

Results go to `--output-dir` (default `results/retirement/`): plots (PNG), timing tables (LaTeX/Markdown), accuracy tables.
