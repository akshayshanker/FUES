# Retirement Choice Model

Implementation of the Iskhakov, Jorgensen, Rust, and Schjerning (2017) retirement choice model, used as the primary benchmark in Dobrescu and Shanker (2026). Demonstrates speed and lower complexity of FUES.

!!! tip "Interactive notebook"
    For interactive run of scaling compared to benchmarks, see the **[Retirement Model Notebook](../notebooks/retirement_fues.ipynb)**.

## Model

Each period, a finite-horizon agent chooses consumption $c_t$ and whether to *continue* work ($d_{t+1} = 1$) or retire ($d_{t+1} = 0$). Retirement is absorbing: once retired, the agent cannot return to work.

**Budget constraint:**

$$
a_{t+1} = (1+r)a_t + d_t y - c_t
$$

**Per-period utility:**

$$
\log(c_t) - \tau d_{t+1}
$$

where $\tau$ is the utility cost of working, $a_{t}$ is beginning of period liquid assets, $y$ is income for a worker and $r$ is the interest rate.

**Worker's Bellman equation:**

$$
V_t^1(a) = \max_{c, d_{t+1} \in \{0,1\}} \left\{ u(c) - d_{t+1}\tau + \beta V_{t+1}^{d_{t+1}}(a') \right\}
$$
where $a^{\prime} = (1+r)a +  y - c$.

**Retiree's Bellman equation:**

$$
V_t^0(a) = \max_c \left\{ u(c) + \beta V_{t+1}^0(a') \right\}
$$

The worker's value function is the upper envelope of multiple concave functions, one for each feasible sequence of future discrete choices. Holding $d_{t+1}=1$ fixed, by selecting $a'$, the worker implicitly selects all future discrete choices $d_j$ for $j > t+1$. The upper envelope of these concave functions is not itself concave, producing the "secondary kinks" described by Iskhakov et al. (2017). This non-concavity is precisely the problem that FUES solves.

### Euler equations and applying FUES

If the agent chooses to continue working ($d_{t+1} = 1$), the worker's Euler equation at time $t$ is

$$
u'(c_t^1) \geq \beta(1+r)\, u'(c_{t+1}),
$$

where $c_t^1$ is consumption conditional on $d_{t+1}=1$ and $c_{t+1}$ is the unconditional time $t+1$ consumption. If the agent retires ($d_{t+1}=0$), the Euler equation is

$$
u'(c_t^0) \geq \beta(1+r)\, u'(c_{t+1}^0).
$$

Because the value function is not concave, worker points satisfying these first-order conditions may lie on choice-specific value functions associated with suboptimal future discrete-choice sequences. The endogenous grid method (EGM) inverts these Euler equations analytically to produce an unrefined endogenous grid $\hat{\mathbb{X}}_t$, value correspondence $\hat{\mathbb{V}}_t$, and continuation grid $\hat{\mathbb{X}}_t'$. FUES then identifies and removes suboptimal points from these EGM outputs in a single pass. See the [algorithm page](../algorithm/fues-algorithm.md) for a detailed description of the scan logic, forward/backward scans, and intersection-point construction.

## Solve interface

The primary entry point is `solve_nest` in `solve.py`. It loads the YAML declarations, builds the model, and solves backward over $T$ periods:

```python
from examples.retirement.solve import solve_nest
from examples.retirement.outputs import euler, get_policy

nest, model, stage_ops = solve_nest(
    'examples/retirement/syntax',
    method='FUES',
    config_overrides={'grid_size': 3000, 'T': 50},
)

# Euler error (log10)
c_refined = get_policy(nest, 'c', stage='labour_mkt_decision')
print(f'Euler error: {euler(model, c_refined):.4f}')
```

`solve_nest` runs a three-step pipeline on each stage declared in `syntax/stages/`:

1. **Methodize** — attach the upper-envelope method (FUES, DCEGM, RFC, or CONSAV) specified in each stage's `*_methods.yml` file.
2. **Configure** — bind numerical settings (grid sizes, bounds, $\bar{M}$) from `settings.yaml`.
3. **Calibrate** — bind economic parameters ($\beta$, $\tau$, $r$, $y$) from `calibration.yaml`.

It then solves the three stages in reverse topological order each period:

1. **`retire_cons`** — retiree EGM (standard concave problem, no upper envelope).
2. **`work_cons`** — worker EGM + upper envelope via `EGM_UE`. This is where FUES runs.
3. **`labour_mkt_decision`** — pointwise $\max(V^{\text{work}} - \tau,\; V^{\text{retire}})$ to evaluate the discrete choice.

The returned `nest` dict contains the full solution history. Use `get_policy(nest, key, stage=...)` to extract policies and `get_timing(nest)` to extract UE and total solve times.

Overrides are passed as dicts:

```python
# Lower beta and increase grid
nest, model, _ = solve_nest(
    'examples/retirement/syntax',
    method='FUES',
    calib_overrides={'beta': 0.94},
    config_overrides={'grid_size': 5000, 'T': 50},
)
```

## Running locally

`run.py` is a command-line wrapper around `solve_nest`. It loads baseline economic parameters from `syntax/calibration.yaml` and numerical settings from `syntax/settings.yaml`, applies any command-line overrides, solves with all four methods, and prints a comparison table. Parameters can be overridden at the command line.

The override mechanism works at two levels:

- **`--calib-override key=value`** overrides economic parameters (e.g., `beta`, `delta`, `y`) defined in `calibration.yaml`.
- **`--config-override key=value`** overrides numerical settings (e.g., `grid_size`, `T`, `m_bar`) defined in `settings.yaml`.
- **`--override-file path.yml`** loads a sparse YAML file; keys matching `settings.yaml` are treated as config overrides, all others as calibration overrides.

Each stage in `syntax/stages/` declares which parameters it consumes. The configure and calibrate functors bind only the relevant parameters to each stage, so extra keys are ignored.

Outputs are saved to `--output-dir` and include EGM grid plots, consumption policy plots, and a printed table of Euler errors and timing for all four upper-envelope methods (FUES, RFC, MSS, LTM).

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

Outputs are written to `experiments/retirement/results/`. Pre-computed benchmark results from the paper are available in the same directory.

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

Instead of entering parameters manually, sparse YAML override files in `experiments/retirement/params/` specify only values that differ from the `syntax/` defaults:

| File               | Key changes          |
| ------------------ | -------------------- |
| `baseline.yml`     | $\beta=0.96$, $T=50$ |
| `high_beta.yml`    | $\beta=0.99$         |
| `low_delta.yml`    | $\tau=0.5$, $T=50$ |
| `long_horizon.yml` | $T=50$,              |

```bash
python examples/retirement/run.py --override-file experiments/retirement/params/long_horizon.yml
```

## Benchmark results

Parameters: $T=50$, $\beta=0.96$, $r=0.02$, $y=20$, $a \in [0, 500]$. No taste shocks. (See Tables 1--2 in the paper for the full results.)

**Upper envelope time (ms per period):**

| Grid | $\tau$ | RFC | FUES | MSS |
|------|-----------|-----|------|-----|
| 500 | 0.25 | 1.2 | 0.11 | 0.36 |
| 500 | 1.00 | 1.4 | 0.11 | 0.65 |
| 1000 | 0.25 | 2.9 | 0.22 | 0.64 |
| 1000 | 1.00 | 3.1 | 0.21 | 2.25 |
| 2000 | 0.25 | 5.6 | 0.43 | 1.29 |
| 2000 | 1.00 | 7.4 | 0.43 | 4.76 |
| 3000 | 0.25 | 8.4 | 0.65 | 1.98 |
| 3000 | 1.00 | 12.8 | 0.63 | 6.63 |

**Euler equation error** ($\log_{10}$):

| Grid | $\tau$ | RFC | FUES | MSS |
|------|-----------|------|------|-----|
| 500 | 0.25 | -1.537 | -1.591 | -1.537 |
| 1000 | 1.00 | -1.630 | -1.658 | -1.629 |
| 3000 | 1.00 | -1.629 | -1.660 | -1.629 |

FUES is 5--20× faster than MSS and 10--20× faster than RFC across all configurations. Euler errors are comparable, with FUES slightly more accurate. FUES scales sub-linearly, MSS scales linearly and LTM scales quadratically. 


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

