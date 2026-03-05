# Fast Upper-Envelope Scan (FUES)

Implementation of the fast upper-envelope scan (FUES) method for discrete-continuous dynamic programming, as described in:

> Dobrescu, L.I. and Shanker, A. (2026). "A fast upper envelope scan method for discrete-continuous dynamic programming." [SSRN Working Paper.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4181302)

FUES recovers the upper envelope of the EGM ([Carroll 2006](https://doi.org/10.1016/j.econlet.2005.09.013)) value correspondence in problems in discrete-continuous problems. Unlike existing methods (DC-EGM/MSS by [Iskhakov et al. 2017](https://doi.org/10.3982/QE533); LTM by [Druedahl and Jørgensen 2017](https://doi.org/10.1016/j.jedc.2016.11.005); NEGM by [Druedahl 2021](https://doi.org/10.1007/s10614-020-10045-x)), FUES does not require monotonicity of the optimal policy function or numerical optimisation, and scales sub-linearly with grid size.

This repo also provides a unified upper-envelope interface (`uenvelope`) for one-dimensional discrete-continuous EGM problems, with a single entry point for discrete-continuous problems to use FUES, MSS, roof-top-cut (RFC), and CONSAV.

Complete documentation and notebooks are here. 

## Installation

The installable package is called `dcsmm`. Requires Python 3.11+. Each option below is self-contained — pick one.

### Option 1: Library only

Use FUES or the upper-envelope registry in your own models. No repo clone, no examples.

```bash
pip install git+https://github.com/akshayshanker/FUES.git@release-prep
```

```python
from dcsmm.fues import FUES
from dcsmm.uenvelope import EGM_UE
```

This also installs the runtime dependencies: numba, numpy, scipy, [econ-ark](https://github.com/econ-ark/HARK) (DC-EGM), [ConSav](https://github.com/NumEconCopenhagen/ConsumptionSaving) (G2EGM/LTM), and interpolation. See `pyproject.toml` for the full list and version pins.

View notebook demo of interface.here 

### Option 2: With examples

Clone the repo and install with example dependencies (`matplotlib`, `pyyaml`, `seaborn`). Includes everything in Option 1 plus the example models in the repo checkout.

```bash
git clone -b release-prep https://github.com/akshayshanker/FUES.git
cd FUES
pip install ".[examples]"
```

Run the retirement model benchmark:

```bash
python -m examples.retirement.code.benchmarks
```

Example notebooks are here 
### Option 3: Developer (editable)

Full setup with editable install, examples, and all dependencies including the dolo-plus compiler. Use this if you are modifying the source code. Includes everything in Option 2.

```bash
git clone -b release-prep https://github.com/akshayshanker/FUES.git
cd FUES
bash setup/setup_venv.sh
source .venv/bin/activate
```

This creates a local `.venv`, installs `dcsmm` in editable mode, and verifies the install.

Run the benchmark:

```bash
python -m examples.retirement.code.benchmarks
```

## `dcsmm` package structure

### Core modules

- **FUES** (`src/dcsmm/fues/`): Fast Upper-Envelope Scan implementation + rooftop-cut method.
- **Upper-envelope registry** (`src/dcsmm/uenvelope/`): Unified entry point dispatching to FUES, DC-EGM, RFC, or CONSAV.

### Example models

| Model               | Key feature                                                                                                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Retirement choice   | Discrete work/retire + continuous consumption; monotone policy (speed benchmark) a la [Iskhakov et al. (2017)](https://doi.org/10.3982/QE533)                                   |
| Continuous durables | Housing adjustment costs; non-monotone policy (existing methods fail)                                                                                                           |
| Housing-renting     | Discrete housing tenure + non-linear taxation; inaction regions break monotonicity ([Fella (2014)](https://doi.org/10.1016/j.jmoneco.2014.04.008) with added renting and taxes) |

### External upper-envelope methods

In addition to the native FUES and rooftop-cut (RFC) implementations, `uenvelope` provides interfaces for:

| Package                                                          | Method     | Algorithm                        | Reference                                                        |
| ---------------------------------------------------------------- | ---------- | -------------------------------- | ---------------------------------------------------------------- |
| [econ-ark/HARK](https://github.com/econ-ark/HARK)                | DC-EGM/MSS | Monotone segment selection (MSS) | [Iskhakov et al. (2017)](https://doi.org/10.3982/QE533)          |
| [ConSav](https://github.com/NumEconCopenhagen/ConsumptionSaving) | G2EGM      | Local triangulation (LTM)        | [Druedahl and Jørgensen (2017)](https://doi.org/10.1016/j.jedc.2016.11.005) |
| Roof-top-cut                                                     | RFC        |                                  |                                                                  |

### Directory structure

```
FUES/
├── src/dcsmm/            # Installable package
│   ├── fues/             # FUES algorithm + variants
│   └── uenvelope/        # Upper-envelope registry
├── examples/             # Self-contained example models
│   ├── retirement/       # Iskhakov et al. (2017)
│   ├── durables/         # Kaplan & Violante (2014)
│   └── housing_renting/  # Fella (2014)
├── experiments/          # PBS/HPC scripts
├── setup/                # setup_venv.sh, load_env.sh
├── docs/                 # mkdocs documentation site
└── tests/
```

## References

- Carroll, C.D. (2006). "The method of endogenous gridpoints for solving dynamic stochastic optimization problems." *Economics Letters*, 91(3), 312–320.
- Dobrescu, L.I. and Shanker, A. (2026). "A fast upper envelope scan method for discrete-continuous dynamic programming."
- Druedahl, J. and Jørgensen, T.H. (2017). "A general endogenous grid method for multi-dimensional models with non-convexities and constraints." *Journal of Economic Dynamics and Control*, 74, 87–107.
- Druedahl, J. (2021). "A guide on solving non-convex consumption-saving models." *Computational Economics*, 58, 747–775.
- Fella, G. (2014). "A generalized endogenous grid method for non-smooth and non-concave problems." *Review of Economic Dynamics*, 17(2), 329–344.
- Iskhakov, F., Jørgensen, T.H., Rust, J. and Schjerning, B. (2017). "The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks." *Quantitative Economics*, 8(2), 317–365.
- Kaplan, G. and Violante, G.L. (2014). "A model of the consumption response to fiscal stimulus payments." *Econometrica*, 82(4), 1199–1239.
