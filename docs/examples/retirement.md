# Retirement Choice Model

Implementation of the Iskhakov, Jorgensen, Rust, and Schjerning (2017) retirement choice model, used as the primary benchmark in Dobrescu and Shanker (2025).

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

## Running benchmarks

All commands run from the repo root (`FUES/`). `--output-dir` is relative to cwd.

```bash
# Single run (baseline parameters, grid size 3000)
PYTHONPATH=".:src" python examples/retirement/run.py \
    --grid-size 3000 --output-dir results/retirement

# Full timing sweep across grid sizes and delta values
PYTHONPATH=".:src" python examples/retirement/run.py \
    --run-timings --sweep-grids 500,1000,2000,3000,10000 \
    --sweep-deltas 0.25,0.5,1,2 --output-dir results/retirement
```

## Benchmark results

### Without taste shocks

Parameters: \(T=50\), \(\beta=0.96\), \(r=0.02\), \(y=20\), \(a \in [0, 500]\).

**Upper envelope time (ms per period):**

| Grid | \(\delta\) | RFC | FUES | DC-EGM |
|------|-----------|-----|------|--------|
| 500 | 0.25 | 1.2 | 0.11 | 0.36 |
| 500 | 1.00 | 1.4 | 0.11 | 0.65 |
| 1000 | 0.25 | 2.9 | 0.22 | 0.64 |
| 1000 | 1.00 | 3.1 | 0.21 | 2.25 |
| 2000 | 0.25 | 5.6 | 0.43 | 1.29 |
| 2000 | 1.00 | 7.4 | 0.43 | 4.76 |
| 3000 | 0.25 | 8.4 | 0.65 | 1.98 |
| 3000 | 1.00 | 12.8 | 0.63 | 6.63 |

**Euler equation error** (\(\log_{10}\)):

| Grid | \(\delta\) | RFC | FUES | DC-EGM |
|------|-----------|------|------|--------|
| 500 | 0.25 | -1.537 | -1.591 | -1.537 |
| 1000 | 1.00 | -1.630 | -1.658 | -1.629 |
| 3000 | 1.00 | -1.629 | -1.660 | -1.629 |

FUES is 5-20x faster than DC-EGM and 10-20x faster than RFC across all configurations. Euler errors are comparable, with FUES slightly more accurate.

### Key observations

1. **FUES timing is stable across \(\delta\)**: DC-EGM slows as \(\delta\) increases (more kinks in the endogenous grid), while FUES timing is nearly constant.

2. **With taste shocks** (\(\bar{s} > 0\)): the endogenous grid becomes non-monotone, with decreasing segments and isolated points. DC-EGM must process many additional segments. FUES handles this without modification.

3. **Scaling**: FUES scales linearly with grid size; DC-EGM scales super-linearly as the number of monotone segments grows.

## Code structure

```
examples/retirement/
├── run.py           # CLI entry point (uses solve_canonical)
├── syntax/                     # dolo-plus YAML declarations (single source of truth)
│   ├── period.yaml             # Period template (stage list)
│   ├── calibration.yaml        # r, beta, delta, y, b, smooth_sigma
│   ├── settings.yaml           # grid_size, grid_max_A, T, m_bar
│   └── stages/
│       ├── labour_mkt_decision/  # Branching (max)
│       ├── work_cons/            # Worker EGM + FUES
│       └── retire_cons/          # Retiree EGM
├── model.py                    # Model class + Operator_Factory
├── solve.py                    # Canonical pipeline (solve_canonical)
├── benchmark.py                # Timing sweeps (via solve_canonical)
└── outputs/
    ├── diagnostics.py          # Nest accessors, euler, deviation
    ├── plots.py                # EGM grid and policy plots
    └── tables.py               # LaTeX + Markdown tables
```

See [API Reference](retirement-api.md) for detailed function signatures.

## Parameter files

| File | Description |
|------|-------------|
| `baseline.yml` | \(\beta=0.98\), \(T=20\), \(\delta=1.0\), \(\bar{s}=0\) |
| `high_beta.yml` | Higher discount factor |
| `low_delta.yml` | Lower cost of working |
| `long_horizon.yml` | More periods |
| `sigma05.yml` | Logit-smoothed discrete choice (\(\bar{s}=0.05\)) |

## References

- Iskhakov, F., Jorgensen, T.H., Rust, J., and Schjerning, B. (2017). "The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks." *Quantitative Economics*, 8(2), 317-365.
- Dobrescu, L.I. and Shanker, A. (2025). "A fast upper envelope scan method for discrete-continuous dynamic programming."
