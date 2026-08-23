# Paper results

Committed outputs for Dobrescu and Shanker (2022), "A fast upper envelope
scan method for discrete-continuous dynamic programming."

Each model directory holds dated snapshots of PBS cluster runs, one
`YYYY-MM-DD/NNN/` directory per run, copied from Gadi scratch with
`scripts/pull_gadi_results.sh`. Tables (`.tex` and `.md`) are committed;
figures are not — repo-wide `*.png` and `*.pdf` ignore rules keep binary
outputs out of git, so figures are regenerated from the runs.

## Layout

```
paper-results/
├── retirement/       # Section 2.1: 2026-03-31/002, 2026-04-15/001
├── durables/         # Section 2.2: 2026-04-06/004, 2026-04-15/001
├── housing_renting/  # Section 2.3: placeholder, not yet populated
└── README.md
```

The retirement snapshots include a parameter footer in their tables; the
2026-04-06 durables run also commits the settings YAML used.

## Reproducing

The cluster scripts are in `benchmarks/<model>/`; see
[Running on a PBS cluster](../docs/running-on-gadi.md) for site setup.

```bash
qsub benchmarks/retirement/retirement_timings.sh   # Section 2.1
qsub benchmarks/durables/run_durables.pbs          # Section 2.2
qsub benchmarks/housing_renting/run_housing_single_core.sh   # Section 2.3
```

Results land on scratch; copy a run's tables into a new dated directory
here, for example:

```bash
cp $SCRATCH/FUES/retirement/<run>/tables/*.tex paper-results/retirement/<date>/<n>/tables/
```

## Hardware

Hardware varies by application; each PBS script records its exact
resource requests. Retirement and durables timing runs use a single
Intel Xeon core on the NCI Gadi expresssr queue (Python 3.12, numba);
housing-renting runs are multi-core or GPU depending on configuration.
