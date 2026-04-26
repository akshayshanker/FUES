# Installation

The installable package containing `fues` and `uenvelope` modules is `dcsmm`.
Requires Python 3.11+.

## Option 1: Library only

Install FUES and the upper-envelope interface without cloning the repo. Use
this option if you want `fues` and the comparison methods inside another
application.

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

## Option 2: Repo checkout with examples

Use this option if you also want the benchmark applications, notebooks, and
replication scripts as well as the core library.

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
source setup/setup.sh
```

The setup script installs `dcsmm` in editable mode together with the example
dependencies and activates the environment in your current shell.

You can then run a retirement solve:

```bash
python -m examples.retirement.run --settings-override grid_size=3000
```

Related pages:

- [Quickstart](../start-here/quickstart.md) for the minimal run commands
- [Running Locally](../running-locally.md) for command-line workflows
- [Tutorials](../tutorials/index.md) for notebook walkthroughs

## Option 3: Manual editable install

Use this only if you already manage your own virtual environment and want to
install the package manually rather than through `setup/setup.sh`.

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[examples]"
```

To add the developer extras:

```bash
pip install -e ".[dev]"
```

After pulling new code in a repo checkout managed by the setup script, refresh
the environment with:

```bash
source setup/setup.sh --update
```

## Verify installation

```python
from dcsmm.fues import FUES
from dcsmm.uenvelope import EGM_UE

print("FUES and EGM_UE imported successfully")
```

For batch runs and HPC setup, continue to [Running on PBS / Gadi](../running-on-gadi.md).

