# Replication

Paper artifacts for Dobrescu and Shanker (2026), "A fast upper envelope scan method for discrete-continuous dynamic programming."

Each subdirectory corresponds to a paper application. Figures and tables are committed outputs from PBS cluster runs on NCI Gadi.

## Reproducing

All results can be regenerated from source:

```bash
# Retirement model (Tables 1-2, Figure 4-5)
qsub experiments/retirement/retirement_timings.sh
```

Output lands in `$SCRATCH/FUES/solutions/retirement/`. Copy the final artifacts here:

```bash
cp $SCRATCH/FUES/solutions/retirement/<run_id>/retirement_timing.tex replication/retirement/tables/
cp $SCRATCH/FUES/solutions/retirement/<run_id>/retirement_accuracy.tex replication/retirement/tables/
cp $SCRATCH/FUES/solutions/retirement/<run_id>/plots/*.png replication/retirement/figures/
```

## Contents

```
replication/
├── retirement/
│   ├── figures/          # Paper Figures 4-5
│   └── tables/           # Paper Tables 1-2 (.tex + .md)
└── README.md
```

## Hardware

PBS runs used a single Intel Xeon core on NCI Gadi (expresssr queue), Python 3.12, numba JIT. See `experiments/retirement/retirement_timings.sh` for exact PBS settings.
