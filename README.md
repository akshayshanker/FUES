# Fast Upper-Envelope Scan (FUES)

> Dobrescu, L.I. and Shanker, A. (2026). "A fast upper envelope scan method for discrete-continuous dynamic programming." [SSRN Working Paper.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4181302)

FUES recovers the upper envelope of the EGM ([Carroll 2006](https://doi.org/10.1016/j.econlet.2005.09.013)) value correspondence in discrete-continuous problems. Unlike MSS ([Iskhakov et al. 2017](https://doi.org/10.3982/QE643)), LTM ([Druedahl & Jørgensen 2017](https://doi.org/10.1016/j.jedc.2016.11.005)), and NEGM ([Druedahl 2021](https://doi.org/10.1007/s10614-020-10045-x)), it does not require monotonicity of the optimal policy function or numerical optimisation.

The repo also ships a unified upper-envelope interface (`uenvelope`) that dispatches to FUES, MSS, RFC, or CONSAV/LTM through a single call.

**[Docs](https://akshayshanker.github.io/FUES/)** · **[Notebook](https://akshayshanker.github.io/FUES/notebooks/retirement_fues/)** · **[Examples](https://akshayshanker.github.io/FUES/examples/retirement/)**

## Install

Package name is `dcsmm`. Python 3.11+.

### Library only

```bash
pip install git+https://github.com/akshayshanker/FUES.git@release-prep
```

```python
from dcsmm.fues import FUES
from dcsmm.uenvelope import EGM_UE
```

Dependencies: numba, numpy, scipy, [HARK](https://github.com/econ-ark/HARK), [ConSav](https://github.com/NumEconCopenhagen/ConsumptionSaving). Full list in `pyproject.toml`.

### With examples

```bash
git clone -b release-prep https://github.com/akshayshanker/FUES.git
cd FUES
pip install ".[examples]"
python examples/retirement/run.py --grid-size 3000
```

The [notebook](examples/retirement/notebooks/retirement_fues.ipynb) walks through the retirement model step by step.

### Developer

```bash
git clone -b release-prep https://github.com/akshayshanker/FUES.git
cd FUES
bash setup/setup_venv.sh
source .venv/bin/activate
python examples/retirement/run.py --run-timings
```

## Package layout

- `src/dcsmm/fues/` — FUES algorithm + RFC
- `src/dcsmm/uenvelope/` — upper-envelope registry (FUES, MSS, RFC, LTM)

### Example models

| Model | What it tests |
|-------|--------------|
| Retirement choice | Discrete work/retire + continuous consumption; monotone policy ([Iskhakov et al. 2017](https://doi.org/10.3982/QE643)) |
| Continuous durables | Housing adjustment costs; non-monotone policy where MSS/LTM fail |
| Housing-renting | Discrete tenure + capital income tax; inaction regions ([Fella 2014](https://doi.org/10.1016/j.red.2013.07.001)) |

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
├── experiments/          # PBS/HPC scripts, sparse param overrides
├── setup/                # setup_venv.sh, load_env.sh
└── docs/                 # mkdocs site
```

## References

- Carroll, C.D. (2006). "The method of endogenous gridpoints for solving dynamic stochastic optimization problems." *Econ. Letters*, 91(3), 312–320.
- Dobrescu, L.I. and Shanker, A. (2024). "Discrete continuous high dimensional dynamic programming." *SSRN Working Paper No. 4850746*.
- Dobrescu, L.I. and Shanker, A. (2026). "A fast upper envelope scan method for discrete-continuous dynamic programming."
- Druedahl, J. and Jørgensen, T.H. (2017). "A general endogenous grid method for multi-dimensional models with non-convexities and constraints." *JEDC*, 74, 87–107.
- Druedahl, J. (2021). "A guide on solving non-convex consumption-saving models." *Comp. Econ.*, 58, 747–775.
- Fella, G. (2014). "A generalized endogenous grid method for non-smooth and non-concave problems." *RED*, 17(2), 329–344.
- Iskhakov, F., Jørgensen, T.H., Rust, J. and Schjerning, B. (2017). "The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks." *QE*, 8(2), 317–365.
