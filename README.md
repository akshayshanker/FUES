<p align="center">
  <strong>Fast Upper-Envelope Scan (FUES)</strong>
</p>

<p align="center">
  <a href="https://akshayshanker.github.io/FUES/">Docs</a> ·
  <a href="https://akshayshanker.github.io/FUES/notebooks/retirement_fues/">Notebooks</a> ·
  <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4181302">Paper</a>
</p>

---

FUES recovers the upper envelope of the EGM ([Carroll 2006](https://doi.org/10.1016/j.econlet.2005.09.013)) value correspondence in discrete-continuous problems without requiring monotonicity of the optimal policy or numerical optimisation. FUES can also perform orders of magnitude faster than existing upper-envelope methods.

<p align="center">
  <img src="docs/images/pbs-scaling.png" alt="Upper-envelope scaling: FUES vs MSS, RFC, LTM" width="640">
</p>

This repo ships a unified upper-envelope interface (`uenvelope`) that dispatches to FUES and three benchmark methods: MSS ([Iskhakov et al. 2017](https://doi.org/10.3982/QE643)), LTM ([Druedahl & Jørgensen 2017](https://doi.org/10.1016/j.jedc.2016.11.005)), and RFC ([Dobrescu & Shanker 2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4850746)).

> **Pre-release (v0.6.0dev1)** — Under active development. API may change.
>
> Dobrescu, L.I. and Shanker, A. (2022). "A fast upper envelope scan method for discrete-continuous dynamic programming." [SSRN Working Paper.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4181302)

---

## Install

The installable package is `dcsmm` (contains `fues` and `uenvelope`). Requires Python 3.11+.

### Option 1 — Library only

Install FUES and the upper-envelope interface without cloning the repo. Lets you use `fues` and all other benchmark upper-envelope methods in your own applications.

```bash
pip install git+https://github.com/akshayshanker/FUES.git
```

```python
from dcsmm.fues import FUES
from dcsmm.uenvelope import EGM_UE
```

Runtime dependencies (numba, numpy, scipy, [HARK](https://github.com/econ-ark/HARK), [ConSav](https://github.com/NumEconCopenhagen/ConsumptionSaving)) are installed automatically. See `pyproject.toml` for the full list and version pins.

### Option 2 — With examples

Clone the repo and install with example dependencies (`matplotlib`, `pyyaml`, `seaborn`, [`kikku`](https://github.com/bright-forest/kikku)). Includes everything in Option 1 plus the example models in the repo checkout.

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
pip install ".[examples]"
```

Each example can be run as a simple single solve via `run.py`:

```bash
python examples/retirement/run.py --grid-size 3000
```

See the [retirement example docs](https://akshayshanker.github.io/FUES/examples/retirement/) for CLI arguments, parameter overrides, and outputs. The [interactive notebook](examples/retirement/notebooks/retirement_fues.ipynb) walks through the model step by step.

Formal benchmarking and parameter sweeps are run on an HPC cluster using the PBS scripts in [`experiments/retirement/`](experiments/retirement/). Pre-computed paper results (tables and figures) are in [`replication/`](replication/).

### Option 3 — Developer (editable)

Full setup with editable install, examples, and all dependencies including the dolo-plus compiler. Use this if you are modifying the source.

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
source setup/setup.sh
```

First source creates `.venv` (or `$HOME/venvs/fues` on Gadi), installs `dcsmm[examples]` in editable mode, and verifies the install. Re-source any time to activate; pass `--update` to `git pull` + reinstall.

Contributing? Add pytest + autopep8 on top:

```bash
pip install -e ".[dev]" --no-deps
```

Run a timing sweep (Cartesian product of slot-range axes; see
`experiments/retirement/retirement_timings.sh`):

```bash
python -m examples.retirement.run --sweep \
  --slot-range @experiments/retirement/timing_deltas.yaml \
  --slot-range @experiments/retirement/timing_grids.yaml \
  --slot-range @experiments/retirement/timing_methods.yaml \
  --sweep-runs 3
```

---

## Package layout

| Module | Description |
|--------|-------------|
| `src/dcsmm/fues/` | FUES algorithm + rooftop-cut (RFC) |
| `src/dcsmm/uenvelope/` | Unified dispatch to FUES, MSS, RFC, LTM |

### External methods wrapped by `uenvelope`

| Package | Method | Reference |
|---------|--------|-----------|
| [HARK](https://github.com/econ-ark/HARK) | MSS | [Iskhakov et al. (2017)](https://doi.org/10.3982/QE643) |
| [ConSav](https://github.com/NumEconCopenhagen/ConsumptionSaving) | LTM | [Druedahl & Jørgensen (2017)](https://doi.org/10.1016/j.jedc.2016.11.005) |
| Native | RFC | [Dobrescu & Shanker (2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4850746) |

### Directory structure

```
FUES/
├── src/dcsmm/            # installable package
│   ├── fues/             # FUES + variants
│   └── uenvelope/        # upper-envelope registry
├── examples/
│   ├── retirement/       # retirement choice (+ notebooks/)
│   ├── durables/         # durables with adjustment frictions
│   └── housing_renting/  # discrete housing + capital tax
├── experiments/          # PBS/HPC scripts with param overrides
├── replication/          # paper tables + figures (committed outputs)
├── setup/                # setup.sh (install + activate + env, one script)
└── docs/                 # mkdocs site
```

---

## References

- Carroll, C.D. (2006). "The method of endogenous gridpoints for solving dynamic stochastic optimization problems." *Economics Letters*, 91(3), 312–320.
- Dobrescu, L.I. and Shanker, A. (2022). "A fast upper envelope scan method for discrete-continuous dynamic programming." *SSRN Working Paper No. 4181302*.
- Dobrescu, L.I. and Shanker, A. (2024). "Using Inverse Euler Equations to Solve Multidimensional Discrete-Continuous Dynamic Models." *SSRN Working Paper No. 4850746*.
- Druedahl, J. and Jørgensen, T.H. (2017). "A general endogenous grid method for multi-dimensional models with non-convexities and constraints." *JEDC*, 74, 87–107.
- Druedahl, J. (2021). "A guide on solving non-convex consumption-saving models." *Computational Economics*, 58, 747–775.
- Fella, G. (2014). "A generalized endogenous grid method for non-smooth and non-concave problems." *Review of Economic Dynamics*, 17(2), 329–344.
- Iskhakov, F., Jørgensen, T.H., Rust, J. and Schjerning, B. (2017). "The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks." *Quantitative Economics*, 8(2), 317–365.
