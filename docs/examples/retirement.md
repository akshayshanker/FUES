# Retirement Choice Model

Implementation of the Iskhakov, Jorgensen, Rust, and Schjerning (2017) retirement choice model, used as the primary benchmark in Dobrescu and Shanker (2026).

!!! tip "Interactive notebook"
    For a step-by-step walkthrough with interactive plots, see the **[Retirement Model Notebook](../notebooks/retirement_fues.ipynb)**.

## Model

A finite-horizon agent chooses consumption \(c_t\) and whether to work (\(d_{t+1} = 1\)) or retire (\(d_{t+1} = 0\)) each period. Retirement is absorbing: once retired, the agent cannot return to work.

**Budget constraint:**

\[
a_{t+1} = (1+r)a_t + d_t y - c_t
\]

**Per-period utility:**

\[
\log(c_t) - \delta d_{t+1}
\]

where \(\delta\) is the utility cost of working.

**Worker's Bellman equation:**

\[
V_t^1(a) = \max_{c, d_{t+1} \in \{0,1\}} \left\{ u(c) - d_{t+1}\delta + \beta V_{t+1}^{d_{t+1}}(a') \right\}
\]

**Retiree's Bellman equation:**

\[
V_t^0(a) = \max_c \left\{ u(c) + \beta V_{t+1}^0(a') \right\}
\]

The worker's value function is the upper envelope of multiple concave functions, one for each feasible sequence of future discrete choices. This non-concavity is precisely the problem that FUES solves.

## Running locally

All commands run from the repo root (`FUES/`).

```bash
# Single run (baseline calibration from syntax/)
python examples/retirement/run.py --grid-size 3000 --output-dir results/retirement

# Override parameters
python examples/retirement/run.py --calib-override beta=0.96 --config-override T=50

# Full timing sweep (all methods × grid sizes × delta values)
python examples/retirement/run.py --run-timings \
    --sweep-grids 1000,2000,3000,6000,10000 \
    --sweep-deltas 0.25,0.5,1,2 \
    --output-dir results/retirement
```

## Replicating paper results (PBS cluster)

The paper's timing and accuracy tables (Tables 1--2) were produced on NCI Gadi (Intel Xeon, single core). To replicate:

```bash
# On Gadi: submit the PBS batch job
cd FUES
source setup/load_env.sh
qsub experiments/retirement/retirement_timings.sh
```

This runs the full sweep (15 grid sizes × 4 delta values × 4 methods × 3 repetitions) and produces:

- `retirement_timing.md` / `.tex` — timing table (ms per period)
- `retirement_accuracy.md` / `.tex` — Euler error and consumption deviation tables
- EGM grid plots at the configured age

The LaTeX tables include only the paper grid sizes (1k, 2k, 3k, 6k, 10k) via `--latex-grids`. The markdown tables include all 15 grid sizes for the [notebook comparison](../notebooks/retirement_fues.ipynb).

### PBS settings

Edit `experiments/retirement/retirement_timings.sh` to change:

| Variable | Default | Description |
|----------|---------|-------------|
| `SWEEP_GRIDS` | 1k--15k (1k steps) | Grid sizes for the sweep |
| `SWEEP_DELTAS` | 0.25, 0.5, 1, 2 | Utility cost values |
| `LATEX_GRIDS` | 1k, 2k, 3k, 6k, 10k | Subset for LaTeX tables |
| `SWEEP_RUNS` | 3 | Best-of-n repetitions |

### Override files

Sparse YAML files in `experiments/retirement/params/` — only values that differ from `syntax/` defaults:

| File | Key changes |
|------|-------------|
| `baseline.yml` | \(\beta=0.96\), \(T=50\) |
| `high_beta.yml` | \(\beta=0.99\) |
| `low_delta.yml` | \(\delta=0.5\), \(T=50\) |
| `long_horizon.yml` | \(T=50\), padding\_mbar |

```bash
python examples/retirement/run.py --override-file experiments/retirement/params/long_horizon.yml
```

## Benchmark results

### Without taste shocks

Parameters: \(T=50\), \(\beta=0.96\), \(r=0.02\), \(y=20\), \(a \in [0, 500]\).

**Upper envelope time (ms per period):**

| Grid | \(\delta\) | RFC | FUES | MSS |
|------|-----------|-----|------|-----|
| 500 | 0.25 | 1.2 | 0.11 | 0.36 |
| 500 | 1.00 | 1.4 | 0.11 | 0.65 |
| 1000 | 0.25 | 2.9 | 0.22 | 0.64 |
| 1000 | 1.00 | 3.1 | 0.21 | 2.25 |
| 2000 | 0.25 | 5.6 | 0.43 | 1.29 |
| 2000 | 1.00 | 7.4 | 0.43 | 4.76 |
| 3000 | 0.25 | 8.4 | 0.65 | 1.98 |
| 3000 | 1.00 | 12.8 | 0.63 | 6.63 |

**Euler equation error** (\(\log_{10}\)):

| Grid | \(\delta\) | RFC | FUES | MSS |
|------|-----------|------|------|-----|
| 500 | 0.25 | -1.537 | -1.591 | -1.537 |
| 1000 | 1.00 | -1.630 | -1.658 | -1.629 |
| 3000 | 1.00 | -1.629 | -1.660 | -1.629 |

FUES is 5--20× faster than MSS and 10--20× faster than RFC across all configurations. Euler errors are comparable, with FUES slightly more accurate.

### Key observations

1. **FUES timing is stable across \(\delta\)**: MSS slows as \(\delta\) increases (more kinks in the endogenous grid), while FUES timing is nearly constant.

2. **With taste shocks** (\(\bar{s} > 0\)): the endogenous grid becomes non-monotone, with decreasing segments and isolated points. MSS must process many additional segments. FUES handles this without modification.

3. **Scaling**: FUES scales sub-linearly with grid size; MSS scales linearly; LTM scales quadratically. See the [scaling analysis in the notebook](../notebooks/retirement_fues.ipynb).

## Code structure

```
examples/retirement/
├── run.py                      # CLI entry point (uses solve_nest)
├── syntax/                     # dolo-plus YAML declarations (single source of truth)
│   ├── period.yaml             # Period template (stage list)
│   ├── calibration.yaml        # r, beta, delta, y, b, smooth_sigma
│   ├── settings.yaml           # grid_size, grid_max_A, T, m_bar
│   └── stages/
│       ├── labour_mkt_decision/  # Branching (max)
│       ├── work_cons/            # Worker EGM + FUES
│       └── retire_cons/          # Retiree EGM
├── model.py                    # Model class + Operator_Factory
├── solve.py                    # Canonical pipeline (solve_nest)
├── benchmark.py                # Timing sweeps (via solve_nest)
├── notebooks/
│   └── retirement_fues.ipynb   # Interactive walkthrough
└── outputs/
    ├── diagnostics.py          # Nest accessors, euler, deviation
    ├── plots.py                # Paper + notebook plot functions
    └── tables.py               # LaTeX + Markdown tables
```

See [API Reference](../api/retirement.md) for detailed function signatures.

## References

- Iskhakov, F., Jorgensen, T.H., Rust, J., and Schjerning, B. (2017). "The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks." *Quantitative Economics*, 8(2), 317-365.
- Dobrescu, L.I. and Shanker, A. (2026). "A fast upper envelope scan method for discrete-continuous dynamic programming."
- Druedahl, J. (2021). "A guide to solve non-convex consumption-saving models." *Computational Economics*, 58, 747-775.
- Dobrescu, L.I. and Shanker, A. (2024). "RFC: A rooftop-cut method for upper envelopes." Working paper.
