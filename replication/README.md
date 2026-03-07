# Replication

Paper artifacts for Dobrescu and Shanker (2026), "A fast upper envelope scan method for discrete-continuous dynamic programming."

Each subdirectory corresponds to a paper application. Figures and tables are committed outputs from PBS cluster runs.

## Applications

| Directory | Paper section | What |
|-----------|--------------|------|
| `retirement/` | Section 2.1, Tables 1--2, Figures 4--5 | Discrete retirement choice (Iskhakov et al. 2017) |
| `housing/` | Section 2.2 | Continuous housing investment with frictions |
| `housing_renting/` | Section 2.3 | Discrete housing choice with capital income tax |

## Reproducing

Each application has its own PBS script under `experiments/`. Results land on scratch and can be copied here.

### Retirement (Section 2.1)

```bash
qsub experiments/retirement/retirement_timings.sh
```

```bash
cp $SCRATCH/FUES/solutions/retirement/<run_id>/retirement_timing.tex replication/retirement/tables/
cp $SCRATCH/FUES/solutions/retirement/<run_id>/retirement_accuracy.tex replication/retirement/tables/
cp $SCRATCH/FUES/solutions/retirement/<run_id>/plots/*.png replication/retirement/figures/
```

### Housing investment (Section 2.2)

```bash
qsub experiments/durables/durables_timings.sh
```

### Housing-renting (Section 2.3)

```bash
qsub experiments/housing_renting/run_housing_single_core.sh
```

## Contents

```
replication/
├── retirement/
│   ├── figures/          # Figures 4-5
│   └── tables/           # Tables 1-2 (.tex + .md)
├── housing/              # (to be populated)
│   ├── figures/
│   └── tables/
├── housing_renting/      # (to be populated)
│   ├── figures/
│   └── tables/
└── README.md
```

## Hardware

Hardware varies by application. See each PBS script for exact settings.

- **Retirement**: single Intel Xeon core, NCI Gadi expresssr queue, Python 3.12, numba
- **Housing investment**: single core (same setup)
- **Housing-renting**: multi-core / GPU depending on configuration
