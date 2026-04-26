# Application 1: A Discrete Retirement Choice Model

*Source: Dobrescu and Shanker (2026), Section 2.1. Based on Iskhakov et al. (2017).*

## Notation

| Symbol | Meaning |
|--------|---------|
| $a_t$ | Financial assets at time $t$ |
| $d_t \in \{0,1\}$ | Work status ($1$ = worker, $0$ = retiree) |
| $c_t$ | Consumption at time $t$ |
| $y$ | Per-period wage (constant) |
| $r$ | Interest rate |
| $\beta$ | Discount factor |
| $\tau$ | Utility cost of working (named `delta` in the code) |
| $S = [0, \bar{a}]$ | Financial assets space |
| $V_t^d(a)$ | Value function at time $t$ conditional on work status $d$ |
| $\sigma_t^d(a, d_{t+1})$ | Continuation state policy function conditional on discrete choice |
| $\mathcal{I}_t(a, d)$ | Discrete choice policy function |
| $\mathbb{D}$ | Set of feasible future work-retire sequences |
| $\mathbf{d} = \{d_{t+1}, d_{t+2}, \ldots, d_T\}$ | One feasible future sequence |
| $V_t^{\mathbf{d}}(a)$ | Value function conditional on a fixed future sequence $\mathbf{d}$ |

## Model Environment

Consider the finite horizon retirement and savings choice model in Iskhakov et al. (2017) where an agent consumes, works (if they so choose), and saves from time $t=0$ until $t=T$. At the beginning of each period, the agent starts as a worker or retiree, with the state variable denoting their beginning-of-period work status given by the discrete variable $d_t$. If the agent works, they earn a per-period wage $y$. Every period, they can choose to continue working in the next period by setting $d_{t+1} = 1$, or to permanently retire by setting $d_{t+1} = 0$. If the agent chooses to work the next period, they will incur a utility cost $\tau$ at time $t$. We assume all agents start as workers so $d_0 = 1$, they consume $c_t$, and save in financial assets $a_{t+1}$ with $a_{t+1} \in S$ and $S := [0, \bar{a}] \subset \mathbb{R}_+$. The intertemporal budget constraint is

$$
a_{t+1} = (1+r)a_t + d_t y - c_t.
$$

Per-period utility is given by $u(c_t) - \tau d_{t+1}$. Letting $u(c) = \log(c)$, the lifetime problem can be written as

$$
V_0^{d_0}(a_0)
= \max_{(c_t)_{t=0}^{T},\, (d_{t+1})_{t=0}^{T-1}}
\left\{
\sum_{t=0}^{T} \beta^t u(c_t)
- \sum_{t=0}^{T-1} \beta^t \tau d_{t+1}
\right\},
$$

subject to the budget constraint, $a_t \in S$ for each $t$, and $d_{t+1} = 0$ if $d_t = 0$ (i.e., the agent is unable to return to work after retiring).

Let $V_t^{d_t}$ denote the beginning-of-period value function. If the agent enters the period as a worker, their value function for any $t < T$ is characterized by the Bellman equation

$$
V_t^1(a) = \max_{c, d_{t+1} \in \{0,1\}} \left\{ u(c) - d_{t+1}\tau + \beta V_{t+1}^{d_{t+1}}(a') \right\},
$$

where $a' = (1+r)a + y - c$ and $a' \in S$; at time $T$, their value function is $V_T^1(a) = u((1+r)a + y)$. If the agent enters the period as a retiree, the time $t < T$ value function is

$$
V_t^0(a) = \max_c \left\{ u(c) + \beta V_{t+1}^0(a') \right\},
$$

with $a' = (1+r)a - c$; at time $T$, their value function is $V_T^0(a) = u((1+r)a)$.

The retiree's problem is a standard concave savings problem. The worker's problem is different: the current discrete choice and the option to switch in future periods make the continuation value non-concave. Even when the current branch is fixed, the next-period worker value $V_{t+1}^1$ is itself the upper envelope over feasible future work-retire paths. This is the source of the "secondary kinks" in Iskhakov et al. (2017).

To make this explicit, let $\mathbb{D}$ denote the set of feasible future sequences $\mathbf{d} = \{d_{t+1}, d_{t+2}, \ldots, d_T\}$ consistent with retirement irreversibility. For any fixed sequence $\mathbf{d}$, define the branch-conditioned value

$$
V_t^{\mathbf{d}}(a)
= \max_{(c_k)_{k=t}^{T}}
\left\{
\sum_{k=t}^{T} \beta^{k-t} u(c_k)
- \sum_{k=t}^{T-1} \beta^{k-t} \tau d_{k+1}
\right\},
$$

subject to the budget constraint and the fixed sequence $\mathbf{d}$. The worker's value is then

$$
V_t^1(a) = \max_{\mathbf{d} \in \mathbb{D}} V_t^{\mathbf{d}}(a).
$$

Each $V_t^{\mathbf{d}}$ is concave in assets, but their pointwise maximum need not be. EGM therefore returns candidate points that satisfy branch-specific first-order conditions but may lie below the global upper envelope. FUES is the step that recovers that envelope.

## Euler Equations

The weak inequalities below allow for the occasionally binding asset constraint $a_{t+1} \geq 0$; at interior points they hold with equality.

If the agent chooses $d_{t+1} = 1$ (i.e., they continue as a worker in $t+1$), we can write the time $t$ worker Euler equation as

$$
u'(c_t^1) \geq \beta(1+r) u'(c_{t+1}),
$$

where $c_t^1$ is the time $t$ consumption policy conditional on $d_{t+1} = 1$, while $c_{t+1}$ is the time $t+1$ consumption policy under the optimal continuation choice. On the other hand, if the agent chooses $d_{t+1} = 0$ (i.e., they retire), then the Euler equation is

$$
u'(c_t^0) \geq \beta(1+r) u'(c_{t+1}^0),
$$

where $c_t^0$ and $c_{t+1}^0$ are the consumption policies conditional on $d_{t+1} = 0$ and $d_{t+2} = 0$.

**Functional Euler equations.** It is useful to write the Euler equation in functional form. Let $\sigma_t^d : S \times \{0,1\} \to \mathbb{R}_+$ be the conditional continuation-state policy function for the worker at time $t$ if $d=1$, and for the retiree if $d=0$. The second argument records the discrete choice made for time $t+1$. Time $t$ and $t+1$ policy functions then satisfy

$$
u'\bigl((1+r)a + dy - \sigma_t^d(a, d_{t+1})\bigr) \geq \beta(1+r) u'\bigl((1+r)\sigma_t^d(a, d_{t+1}) + d_{t+1}y - \sigma_{t+1}^{d_{t+1}}(a', d_{t+2})\bigr),
$$

where $a' = \sigma_t^d(a, d_{t+1})$.

On the choice of whether to work or not, the time $t$ worker will choose $d_{t+1} = 1$ if and only if

$$
u\bigl((1+r)a + y - \sigma_t^1(a, 1)\bigr) - \tau + \beta V_{t+1}^1(\sigma_t^1(a, 1)) > u\bigl((1+r)a + y - \sigma_t^1(a, 0)\bigr) + \beta V_{t+1}^0(\sigma_t^1(a, 0)).
$$

Since the discrete choice is itself a function of the state, define a discrete choice policy function $\mathcal{I}_t : S \times \{0,1\} \to \{0,1\}$. Then we have $d_{t+1} = \mathcal{I}_t(a, d)$ and $d_{t+2} = \mathcal{I}_{t+1}(a', d_{t+1})$, where $\mathcal{I}_t$ is evaluated to satisfy the discrete choice condition each period conditional on the $t+1$ value function.

---

## Modular Bellman Form

The formulation above follows the standard presentation in the literature: one Bellman equation with nested discrete and continuous choices. Below we rewrite the same model in staged Bellman form. In the code, each period is decomposed into linked subproblems with their own state spaces and Bellman operators.

An agent lives for $T$ periods. Each period she holds assets $a \geq 0$ and makes two choices: a **discrete** choice $d \in \{\text{work}, \text{retire}\}$ and a **continuous** choice of consumption $c$. Working yields wage income $y$ but costs disutility $\tau$; retirement is absorbing. Assets earn gross return $(1{+}r)$.

The sequence of events within a period is:
- workers and retirees start with beginning-of-period assets $a$ and $a_{\text{ret}}$ respectively
- earn returns and receive income to produce cash-on-hand $w = (1{+}r)a + y$ for workers or $w_{\text{ret}} = (1{+}r)a_{\text{ret}}$ for retirees
- consume $c$, leaving end-of-period savings $b = w - c$ (or $b_{\text{ret}} = w_{\text{ret}} - c$)
- inter-period transition maps $b \to a$ and $b_{\text{ret}} \to a_{\text{ret}}$ for the next period; retirement is absorbing

### Stage decomposition

Each period can be translated to a directed graph of self-contained modular *stages*, following [Carroll (2026)](https://llorracc.github.io/SolvingMicroDSOPs/) and [Carroll and Shanker (2026)](https://bright-forest.github.io/bellman-ddsl/theory/MDP-foundations/). The retirement model has three stages:

1. **`labour_mkt_decision`** (branching) — discrete choice: $\max(\mathrm{v}_{\succ}^{\text{work}} - \tau,\; \mathrm{v}_{\succ}^{\text{retire}})$. Assets $a$ pass through unchanged.
2. **`work_cons`** (continuous, EGM + FUES) — worker consumption: $a \to w \to b$. The continuation value $\mathrm{v}_{\succ}$ is non-concave; FUES recovers the correct envelope.
3. **`retire_cons`** (continuous, EGM) — retiree consumption: $a_{\text{ret}} \to w_{\text{ret}} \to b_{\text{ret}}$. Standard concave problem.

Note that workers arrive at $a$ (into the branching stage); retirees arrive at $a_{\text{ret}}$ (directly into `retire_cons`). If a worker chooses to retire, they become a retiree and move into the retiree consumption problem, `retire_cons`, alongside those who entered the period as a retiree.

### Stage operators

Within each stage, the state space is represented at three information sets: arrival ($\mathsf{X}_{\prec}$), decision ($\mathsf{X}$), and continuation ($\mathsf{X}_{\succ}$). We keep the code's operator labels $\mathbb{B}$ and $\mathbb{I}$: $\mathbb{B}$ is the within-stage Bellman step and $\mathbb{I}$ maps the decision value back to the arrival state.

### `work_cons` — worker consumption (EGM + FUES)

**Decision mover $\mathbb{B}$** (continuation → decision)

Let $\mathrm{v}_{\succ}(b)$ be the continuation value at end-of-period savings $b$, and $\partial\mathrm{v}_{\succ}(b)$ its derivative. The worker's cash-on-hand is $w$ and the budget constraint is $b = w - c$. The decision mover solves:

$$(\mathbb{B}\,\mathrm{v}_{\succ})(w) = \mathrm{v}(w) = \max_c\bigl\{\log(c) + \beta\,\mathrm{v}_{\succ}(w - c)\bigr\}$$

The first-order condition is $1/c = \beta\,\partial\mathrm{v}_{\succ}(b)$.

*EGM.* Given an exogenous grid $\{b_0^{\#},\dots,b_N^{\#}\}$ on the continuation state, the FOC yields optimal consumption $c_i^{\#} = \bigl(\beta\,\partial\mathrm{v}_{\succ}(b_i^{\#})\bigr)^{-1}$ and the budget constraint recovers the endogenous grid $w_i^{\#} = b_i^{\#} + c_i^{\#}$. Each pair $(w_i^{\#},\, c_i^{\#})$ satisfies the FOC, and the corresponding value is $q_i^{\#} = \log(c_i^{\#}) + \beta\,\mathrm{v}_{\succ}(b_i^{\#})$.

*Non-concavity.* The worker's $\mathrm{v}_{\succ}$ is the upper envelope of concave functions, one for each feasible future discrete-choice sequence, and is not itself concave. As a result the endogenous grid $\{w_i^{\#}\}$ may be non-monotone, and the points $(w_i^{\#},\, q_i^{\#})$ define a correspondence rather than a function. The upper-envelope step in FUES recovers the relevant branch of this correspondence.

**Arrival mover $\mathbb{I}$** (decision → arrival)

The arrival transition is $w = (1{+}r)a + y$, so:

$$(\mathbb{I}\mathrm{v})(a) = \mathrm{v}\bigl((1{+}r)a + y\bigr)$$

The chain rule gives $\partial\mathrm{v}_{\prec}(a) = (1{+}r)\,\partial\mathrm{v}(w)$, and the envelope theorem yields $\partial\mathrm{v}(w) = 1/c$.

### `retire_cons` — retiree consumption (EGM, concave)

**Decision mover $\mathbb{B}$** (continuation → decision)

Let $\mathrm{v}_{\succ}(b_{\text{ret}})$ be the continuation value at retiree savings $b_{\text{ret}}$, and $\partial\mathrm{v}_{\succ}(b_{\text{ret}})$ the marginal value. The retiree's cash-on-hand is $w_{\text{ret}}$ and the budget constraint is $b_{\text{ret}} = w_{\text{ret}} - c$. The decision mover is:

$$(\mathbb{B}\mathrm{v}_{\succ})(w_{\text{ret}}) = \mathrm{v}(w_{\text{ret}}) = \max_c\bigl\{\log(c) + \beta\,\mathrm{v}_{\succ}(b_{\text{ret}})\bigr\}$$

such that $b_{\text{ret}} = w_{\text{ret}} - c$. The first-order condition is $1/c = \beta\,\partial\mathrm{v}_{\succ}(b_{\text{ret}})$.

*EGM.* Given a grid on $b_{\text{ret}}$, recover $c_i^{\#} = \bigl(\beta\,\partial\mathrm{v}_{\succ}(b_{\text{ret},i}^{\#})\bigr)^{-1}$ and $w_{\text{ret},i}^{\#} = b_{\text{ret},i}^{\#} + c_i^{\#}$. Here $\mathrm{v}_{\succ}$ is concave (retirement is absorbing), so EGM produces a monotone endogenous grid and no upper-envelope step is needed.

**Arrival mover $\mathbb{I}$** (decision → arrival)

The arrival transition is $w_{\text{ret}} = (1{+}r)\,a_{\text{ret}}$ (no income), so:

$$(\mathbb{I}\mathrm{v})(a_{\text{ret}}) = \mathrm{v}\bigl((1{+}r)\,a_{\text{ret}}\bigr)$$

and $\partial\mathrm{v}_{\prec}(a_{\text{ret}}) = (1{+}r)\,\partial\mathrm{v}(w_{\text{ret}})$.

### `labour_mkt_decision` — discrete branching

**Decision mover $\mathbb{B}$** (continuation → decision)

The branching stage receives the arrival values from the two consumption stages: $\mathrm{v}_{\succ}^{\text{work}}(a)$ from `work_cons` and $\mathrm{v}_{\succ}^{\text{retire}}(a)$ from `retire_cons`. Assets $a$ pass through unchanged (identity transitions). The decision mover is the discrete-choice $\max$:

$$(\mathbb{B}\mathrm{v}_{\succ})(a) = \mathrm{v}(a) = \max\!\bigl(\mathrm{v}_{\succ}^{\text{work}}(a) - \tau,\;\; \mathrm{v}_{\succ}^{\text{retire}}(a)\bigr)$$

**Arrival mover $\mathbb{I}$** (decision → arrival)

Identity: $(\mathbb{I}\mathrm{v})(a) = \mathrm{v}(a)$.

---

> **Sequential form.** Composing the three stage operators and substituting the transitions recovers the traditional sequential recursive Bellman equations. Writing $V_t^1$ for the worker's arrival value and $V_t^0$ for the retiree's:
>
> $$V_t^1(a) = \max_{d}\; Q_t^d(a), \qquad Q_t^{\text{work}}(a) = \max_c \bigl\{ \log(c) - \tau + \beta\, V_{t+1}^1\bigl((1{+}r)a + y - c\bigr) \bigr\}$$
>
> $$Q_t^{\text{retire}}(a) = \max_c \bigl\{ \log(c) + \beta\, V_{t+1}^0\bigl((1{+}r)a - c\bigr) \bigr\}, \qquad V_t^0(a) = \max_c \bigl\{ \log(c) + \beta\, V_{t+1}^0\bigl((1{+}r)a - c\bigr) \bigr\}$$

## Computation


## Solve interface

The primary entry point is `solve_nest` in `solve.py`. It loads the YAML declarations, builds the model, and solves backward over $T$ periods:

```python
from examples.retirement.solve import solve_nest
from examples.retirement.outputs import euler, get_policy

nest, model, stage_ops, waves = solve_nest(
    "examples/retirement/syntax",
    method_switch="FUES",
    draw={"settings": {"grid_size": 3000, "T": 50}},
)

# Euler error (log10)
sigma_work = get_policy(nest, "c", stage="labour_mkt_decision")
print(f"Euler error: {euler(model, sigma_work):.4f}")
```

`solve_nest` exposes the canonical retirement pipeline:

1. **Load syntax** — read the stage YAML, period template, and inter-period connectors from `syntax/`.
2. **Select the upper-envelope method** — `FUES`, `DCEGM`, `RFC`, or `CONSAV`.
3. **Bind settings and calibration** — read numerical settings from `settings.yaml` and economic parameters from `calibration.yaml`.
4. **Build the stage graph** — derive the backward solve order (`waves`) from the period DAG.
5. **Solve backward** — apply the stage operators period by period.

It then solves the three stages in reverse topological order each period:

1. **`retire_cons`** — retiree EGM (standard concave problem, no upper envelope).
2. **`work_cons`** — worker EGM + upper envelope. This is where FUES runs.
3. **`labour_mkt_decision`** — pointwise $\max(\mathrm{v}^{\text{work}} - \tau,\; \mathrm{v}^{\text{retire}})$ to evaluate the discrete choice.

The returned `nest` dict contains the full solution history. Use `get_policy(nest, key, stage=...)` to extract policies and `get_timing(nest)` to extract mean upper-envelope and total solve times.

For stationary models or repeated solves, pass back `model`, `stage_ops`, and `waves` to skip the pipeline:

```python
# First call: full pipeline
nest, model, stage_ops, waves = solve_nest(
    "examples/retirement/syntax",
    method_switch="FUES",
    draw={"settings": {"grid_size": 3000, "T": 50}},
)

# Subsequent calls: reuse model, operators, and graph
nest2, _, _, _ = solve_nest(
    "examples/retirement/syntax",
    method_switch="FUES",
    draw={
        "calibration": {"beta": 0.94},
        "settings": {"grid_size": 5000, "T": 50},
    },
    model=model,
    stage_ops=stage_ops,
    waves=waves,
)
```

## Running locally

`run.py` is a command-line wrapper around `solve_nest`. It loads baseline economic parameters from `syntax/calibration.yaml` and numerical settings from `syntax/settings.yaml`, applies any command-line overrides, solves the model under the available upper-envelope methods, and prints a comparison table. The API names are `FUES`, `RFC`, `DCEGM`, and `CONSAV`; in plots and tables, `DCEGM` and `CONSAV` are displayed under the paper labels MSS and LTM.

The override mechanism is split by role:

- **`--params-override key=value`** overrides economic parameters (e.g. `beta`, `delta`, `y`) from `calibration.yaml`. The code name `delta` corresponds to $\tau$ in the paper notation used above.
- **`--settings-override key=value`** overrides numerical settings (e.g. `grid_size`, `T`, `plot_age`) from `settings.yaml`.
- **`--latex-grids=…`** (runner extra) gives the subset of grid sizes included in **LaTeX** timing/accuracy tables (markdown tables list all sweeps rows).
- **`--override-file path.yml`** loads a sparse YAML file; keys matching `settings.yaml` are treated as settings overrides and the remaining keys as calibration overrides.

Each stage in `syntax/stages/` declares which parameters it consumes. The pipeline binds only the relevant settings and calibration entries to each stage, so unrelated keys are ignored.

Outputs are saved to `--output-dir` and include EGM grid plots, consumption policy plots, and printed timing and Euler-error tables. The table headings follow the paper labels MSS and LTM even though the callable method tags are `DCEGM` and `CONSAV`.

All commands run from the repo root (`FUES/`).

```bash
# Single run (baseline calibration from syntax/)
python -m examples.retirement.run \
    --settings-override grid_size=3000 \
    --output-dir results/retirement

# Override parameters
python -m examples.retirement.run \
    --params-override beta=0.96 \
    --settings-override T=50

# Full timing sweep: Cartesian product of params × settings × methods (YAML @
# files in experiments/retirement/); post-solve policy plots use the largest
# grid_size in the test set
python -m examples.retirement.run --sweep \
    --params-range @experiments/retirement/timing_deltas.yaml \
    --settings-range @experiments/retirement/timing_grids.yaml \
    --methods-range @experiments/retirement/timing_methods.yaml \
    --latex-grids=1000,2000,3000,6000,10000 \
    --sweep-runs 3 \
    --output-dir results/retirement
```

## Replicating paper results (PBS cluster)

The paper's timing and accuracy tables (Tables 1--2) were produced on NCI Gadi (Intel Xeon, single core). To replicate:

```bash
# On Gadi: submit the PBS batch job
cd FUES
source setup/setup.sh
qsub experiments/retirement/retirement_timings.sh
```

This runs the full sweep (15 grid sizes × 4 work-cost values × 4 methods × 3 repetitions) and produces:

- `retirement_timing.md` / `.tex` — timing table (ms per period)
- `retirement_accuracy.md` / `.tex` — Euler error and consumption deviation tables
- EGM grid plots at the configured age

By default the PBS script writes results to `OUTPUT_DIR` (on Gadi this defaults to `/scratch/tp66/$USER/FUES/retirement`) and mirrors logs to `/g/data/tp66/logs/retirement`. The LaTeX tables use the subset of paper grid sizes (1k, 2k, 3k, 6k, 10k) controlled by `LATEX_GRIDS`; the markdown tables report the full sweep used in the [notebook comparison](../notebooks/retirement_fues.ipynb).

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
| `low_delta.yml`    | `delta=0.5` in code (that is, lower $\tau$ in paper notation), $T=50$ |
| `long_horizon.yml` | $\beta=0.96$, $T=50$, tighter lower-bound settings |

```bash
python -m examples.retirement.run --override-file experiments/retirement/params/long_horizon.yml
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

In the displayed benchmarks, FUES is 5--20× faster than MSS and 10--20× faster than RFC, with comparable Euler errors. The broader sweep in the paper and notebook extends the comparison to LTM as well.


## Code structure

```
examples/retirement/
├── run.py                      # CLI entry point (uses solve_nest)
├── syntax/                     # dolo-plus YAML declarations (single source of truth)
│   ├── period.yaml             # Period template (stage list)
│   ├── nest.yaml               # Inter-period connectors
│   ├── calibration.yaml        # r, beta, delta, y, b, smooth_sigma
│   ├── settings.yaml           # grid_size, grid_max_A, T, m_bar
│   └── stages/
│       ├── labour_mkt_decision/  # Branching (max)
│       ├── work_cons/            # Worker EGM + FUES
│       └── retire_cons/          # Retiree EGM
├── model.py                    # RetirementModel (grids, callables)
├── operators.py                # Stage operator factories
├── solve.py                    # Canonical pipeline (solve_nest)
├── benchmark.py                # Timing sweeps (via solve_nest)
├── notebooks/
│   ├── retirement_fues.ipynb   # Interactive walkthrough
│   └── model.md                # Notebook model description
└── outputs/
    ├── diagnostics.py          # Nest accessors, euler, deviation
    ├── plots.py                # Paper + notebook plot functions
    └── tables.py               # LaTeX + Markdown tables
```

See [API Reference](../api/retirement.md) for detailed function signatures.

## References

- Iskhakov, F., Jorgensen, T.H., Rust, J., and Schjerning, B. (2017). "The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks." *Quantitative Economics*, 8(2), 317-365.
- Dobrescu, L.I. and Shanker, A. (2022). "A fast upper envelope scan method for discrete-continuous dynamic programming." [SSRN Working Paper No. 4181302.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4181302)
- Druedahl, J. (2021). "A guide to solve non-convex consumption-saving models." *Computational Economics*, 58, 747-775.
- Dobrescu, L.I. and Shanker, A. (2024). "RFC: A rooftop-cut method for upper envelopes." Working paper.

