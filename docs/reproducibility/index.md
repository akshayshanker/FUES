# Reproducibility

This section collects the workflows that generate the benchmark outputs
in the working paper — from a first local solve to the full cluster
sweeps — and records where each kind of output lives.

## Where things live

| Folder | Contents |
|---|---|
| [`examples/<model>/`](https://github.com/akshayshanker/FUES/tree/main/examples) | Model code and `run.py` entry point |
| [`benchmarks/<model>/`](https://github.com/akshayshanker/FUES/tree/main/benchmarks) | PBS scripts and sweep drivers that produce the paper tables |
| [`paper-results/<model>/`](https://github.com/akshayshanker/FUES/tree/main/paper-results) | Committed tables and figures from PBS runs |
| `results/<model>/` | Local run outputs from non-PBS invocations (e.g. `python -m examples.retirement.run` with no `--output-dir`); dated `YYYY-MM-DD/NNN/`, gitignored |

## Notebooks

Interactive walkthroughs of the models and the method, rendered from the
repo's Jupyter notebooks:

- [EGM / FUES walkthrough](../notebooks/egm_fues_transparent.ipynb) — the
  upper-envelope problem and the scan, step by step.
- [Retirement choice](../notebooks/retirement_fues.ipynb) — the
  discrete retirement model solved with all four upper-envelope methods.
- [Durables, separable utility](../notebooks/durables_fues_separable.ipynb)
  and [Durables, Cobb–Douglas](../notebooks/durables_fues.ipynb) — the
  housing-adjustment model under its two parameterisations.

## Running locally

[Running locally](../running-locally.md) covers everything an ordinary
machine can do with the examples install: single solves, calibration
comparisons, and sweeps via the slot-keyed command line. Reported paper
wall-times are hardware-specific, but every locally feasible result can
be regenerated serially this way.

## Running on a PBS cluster

[Running on a PBS cluster](../running-on-gadi.md) covers the runs that need
a cluster: the parallel timing sweeps behind the paper tables, the
durables estimation (multi-node MPI with terabyte-scale memory), the
extra-large-grid solves, and the GPU housing-renting benchmarks. Outputs
from these runs are snapshotted into `paper-results/`.
