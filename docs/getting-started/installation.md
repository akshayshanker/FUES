# Installation

The installable package containing `fues` and `uenvelope` modules is `dcsmm`.
Requires Python 3.11+.

## Option 1: Library only

Install FUES and the upper-envelope interface without cloning the repo. This is
the simplest option if you want to use `fues` and the benchmark upper-envelope
methods in your own application.

```bash
pip install git+https://github.com/akshayshanker/FUES.git
```

```python
from dcsmm.fues import FUES
from dcsmm.uenvelope import EGM_UE
```

Runtime dependencies, including `numba`, `numpy`, `scipy`,
[`HARK`](https://github.com/econ-ark/HARK), and
[`ConSav`](https://github.com/NumEconCopenhagen/ConsumptionSaving), are
installed automatically. See `pyproject.toml` for the full list and version
pins.

## Option 2: With examples

Clone the repo and install with example dependencies (`matplotlib`, `pyyaml`,
`seaborn`, [`kikku`](https://github.com/bright-forest/kikku)). This includes
everything in Option 1 plus the example models in the repo checkout.

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
pip install ".[examples]"
```

You can then run a simple example solve:

```bash
python examples/retirement/run.py --grid-size 3000
```

The interactive notebook at `examples/retirement/notebooks/retirement_fues.ipynb`
walks through the retirement model step by step.

## Option 3: Developer (editable)

Full setup with editable install, examples, and all dependencies including the
dolo-plus compiler. Use this if you are modifying the source.

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
bash setup/setup_venv.sh
source .venv/bin/activate
```

This creates a local `.venv`, installs `dcsmm` in editable mode, and verifies
the install.

Run the full timing sweep:

```bash
python examples/retirement/run.py --run-timings
```

## Verify installation

```python
from dcsmm.fues import FUES
from dcsmm.uenvelope import EGM_UE

print("FUES and EGM_UE imported successfully")
```

