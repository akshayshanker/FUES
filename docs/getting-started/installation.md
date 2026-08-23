# Installation

The installable package containing the `fues` and `uenvelope` modules is
`dcsmm`; it requires Python 3.11 or later. The three options below match
the README and the [Quickstart](../start-here/quickstart.md): Option 1
installs the library alone, Option 2 sets up the example applications
through the project script, and Option 3 builds the same environment as
Option 2 by hand.

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

Since the project is under active development, install inside a virtual
environment (`python3 -m venv .venv && source .venv/bin/activate` first);
this route does not create one.

Runtime dependencies, including `numba`, `numpy`, `scipy`,
[`HARK`](https://github.com/econ-ark/HARK),
[`ConSav`](https://github.com/NumEconCopenhagen/ConsumptionSaving), and
`pykdtree`, are
installed automatically. See `pyproject.toml` for the full list and version
pins.

## Option 2: With examples (setup script)

Use this option if you also want the benchmark applications, notebooks, and
replication scripts as well as the core library.

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
source setup/setup.sh
```

On the first run the script creates the project virtual environment
(`.venv`; `~/venvs/fues` on Gadi), installs `dcsmm` in editable mode
together with the example dependencies, and activates the environment in
your current shell; on later runs it only activates. Pass `--update` to
`git pull` and reinstall.

You can then run a retirement solve:

```bash
python -m examples.retirement.run --slot-override '$draw.grid_size=3000'
```

Related pages:

- [Quickstart](../start-here/quickstart.md) for the minimal run commands
- [Running Locally](../running-locally.md) for command-line workflows

## Option 3: Manual install

This is the sequence `setup/setup.sh` automates, for those who prefer to
run it by hand or already manage their own virtual environment. It
produces the same environment as Option 2.

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[examples]"
pip install lark multipledispatch
pip install --no-deps \
  "dolang @ git+https://github.com/bright-forest/dolang.py.git@92b63c44f44394d511b101cc3ea687505721f97f" \
  "dolo @ git+https://github.com/bright-forest/dolo.git@c899b0176d51f6354b5739a28e61ba45cd286a8b"
```

The last two lines install the pinned dolo-plus compiler that the example
models import. They are deliberately `--no-deps` — the forks' packaging
metadata conflicts with this repo's pins — which is why `[examples]` cannot
pull them in; `setup/setup.sh` runs the same lines for you.

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

For batch runs and HPC setup, continue to [Running on a PBS cluster](../running-on-gadi.md).

