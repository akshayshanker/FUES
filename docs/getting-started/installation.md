# Installation

The installable package containing the `fues` and `uenvelope` modules is
`dcsmm`; it requires Python 3.11 or later. There are two ways to install,
matching the README and the [Quickstart](../start-here/quickstart.md): the
library alone (Option 1), or the full environment for the example applications
(Option 2).

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
environment (`python3 -m venv .venv && source .venv/bin/activate` first); this
route does not create one.

Runtime dependencies, including `numba`, `numpy`, `scipy`,
[`HARK`](https://github.com/econ-ark/HARK),
[`ConSav`](https://github.com/NumEconCopenhagen/ConsumptionSaving), and
`pykdtree`, are installed automatically, and every `EGM_UE` method runs on
this install. See `pyproject.toml` for the full list and version pins.

## Option 2: With examples (setup script)

Use this option if you also want the benchmark applications, notebooks, and
the scripts behind the paper tables.

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
source setup/setup.sh
```

On the first run the script creates the project virtual environment (`.venv`;
`~/venvs/fues` on Gadi), installs everything listed in the next subsection,
verifies the imports, and activates the environment in your current shell; on
later runs it only activates. Pass `--update` to `git pull` and reinstall
after new commits.

You can then run a retirement solve:

```bash
python -m examples.retirement.run --slot-override '$draw.grid_size=3000'
```

### What the script installs

The script automates the following sequence; running it by hand in an
environment you manage yourself produces the same result.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[examples]"
pip install lark multipledispatch
pip install --no-deps \
  "dolang @ git+https://github.com/bright-forest/dolang.py.git@"\
"92b63c44f44394d511b101cc3ea687505721f97f" \
  "dolo @ git+https://github.com/bright-forest/dolo.git@"\
"c899b0176d51f6354b5739a28e61ba45cd286a8b"
```

The `[examples]` extra adds the application dependencies (`kikku`, pinned at a
tagged release, plus plotting and estimation packages) on top of the core
install. The last two lines install the pinned dolo-plus compiler that the
example models import; they are deliberately `--no-deps` — the forks'
packaging metadata conflicts with this repo's pins — which is why the extra
cannot pull them in.

To add the developer extras (pytest, autopep8):

```bash
pip install -e ".[dev]"
```

## Verify the installation

```python
from dcsmm.fues import FUES
from dcsmm.uenvelope import EGM_UE

print("FUES and EGM_UE imported successfully")
```

With the examples environment, the quick test suite gives a deeper check in
about a second:

```bash
pytest tests/test_imports.py tests/test_kikku.py -q
```

## Where next

- [Quickstart](../start-here/quickstart.md) for the minimal library calls
  and the first example run.
- [Running locally](../running-locally.md) for command-line workflows and
  sweeps.
- [Running on a PBS cluster](../running-on-gadi.md) for batch runs and
  HPC setup.
