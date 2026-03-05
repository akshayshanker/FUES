# Fast Upper-Envelope Scan (FUES)

This repo contains the core implementation of the fast upper envelope scan (FUES) method and examples for Dobrescu and Shanker (2026). FUES retrieves the upper envelope of the value correspondence when the Euler equation is inverted (the EGM) in a problem with non-convexities (such as discrete choices). FUES does not require restrictions on monotonicity of the optimal policy function.

The repo also includes a general-purpose upper envelope interface (`uenvelope`) for discrete-continuous EGM problems with single dimensional decisions states; the interface allows a unified entry point for the key alternative `python` implementations of upper envelope methods.

## Installation

The installable package is called `dcsmm`. Requires Python 3.11+.

### Option 1: Library only

Install `dcsmm` (FUES + upper envelope registry) into an existing environment. No examples, no dev tools.

```bash
pip install git+https://github.com/akshayshanker/FUES.git@release-prep
```

```python
from dcsmm.fues import FUES
from dcsmm.uenvelope import EGM_UE
```

### Option 2: With examples

Clone the repo and install with example dependencies (matplotlib, pyyaml, seaborn). Use this to run the benchmark models.

```bash
git clone -b release-prep https://github.com/akshayshanker/FUES.git
cd FUES
pip install ".[examples]"
```

### Option 3: Developer (editable)

Full setup with editable install, dolo-plus compiler, and all dependencies. Use this if you are modifying the source code.

```bash
git clone -b release-prep https://github.com/akshayshanker/FUES.git
cd FUES
bash setup/setup_venv.sh
source .venv/bin/activate
```

This creates a local `.venv`, installs `dcsmm` in editable mode with all dependencies (numba, HARK, consav, dolo-plus), and verifies the install.

## Quick test

```bash
python -m examples.retirement.code.benchmarks
```

Runs the retirement model benchmark (FUES vs DCEGM vs RFC vs CONSAV) and prints Euler errors and timing. Requires Option 2 or 3.

## Quick start

```python
from dcsmm.fues import FUES                    # Main algorithm
from dcsmm.uenvelope import EGM_UE             # Unified UE entry point
```

## Core Modules

- **FUES Algorithm** (`src/dcsmm/fues/`): Fast Upper-Envelope Scan implementation
- **UE Registry** (`src/dcsmm/uenvelope/`): Unified entry point comparing FUES, DCEGM, RFC, CONSAV
- **Example Models**:
  - Retirement choice model (Iskhakov et al. 2017)
  - Continuous durables model
  - Housing-renting model with time-inconsistent preferences

## External Packages

- **[ConSav](https://github.com/NumEconCopenhagen/ConsumptionSaving)**: G2EGM upper envelope (used via `consav.upperenvelope`)
- **[HARK](https://github.com/econ-ark/HARK)**: DCEGM implementation (installed as `econ-ark`)

## Directory Structure

```
FUES/
├── src/dcsmm/            # Installable package
│   ├── fues/             # FUES algorithm + variants
│   └── uenvelope/        # UE engine registry
├── examples/             # Self-contained examples
│   └── retirement/       # Retirement choice model
├── experiments/          # PBS/HPC scripts
├── setup/                # setup_venv.sh, load_env.sh
├── docs/                 # mkdocs documentation site
└── tests/
```

## Notes

### ConSav loading

`consav` requires `EconModel` at import time (`consav/__init__.py` imports
`ModelClass`), but we only use `consav.upperenvelope` which has no such
dependency. To avoid requiring `EconModel` as a dependency, we load the
submodule directly:

```python
import importlib
_consav_ue = importlib.import_module("consav.upperenvelope")
```

This bypasses `consav/__init__.py` entirely. The `consav.upperenvelope`
module only depends on `numpy` and `numba`. See
`src/dcsmm/uenvelope/upperenvelope.py` for the implementation.
