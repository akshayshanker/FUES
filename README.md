# Fast Upper-Envelope Scan (FUES)

Core FUES implementation and examples for Dobrescu and Shanker (2025).

Includes a general-purpose upper envelope class for one-dimensional discrete-continuous EGM problems.

**Work in progress.**

## Installation

```bash
git clone -b release-prep https://github.com/akshayshanker/FUES.git
cd FUES
bash setup/setup_venv.sh
source .venv/bin/activate
```

This creates a local `.venv`, installs `dcsmm` in editable mode with all dependencies (numba, HARK, consav, dolo-plus), and verifies the install. On NCI Gadi it auto-detects and uses `/scratch/tp66/$USER/venvs/dcsmm` instead.

Requires Python 3.11+.

## Quick test

```bash
source .venv/bin/activate
python -m examples.retirement.code.benchmarks
```

This runs the retirement model benchmark (FUES vs DCEGM vs RFC vs CONSAV) and prints Euler errors + timing.

## Quick Start

```python
from dcsmm.fues import FUES                    # Main algorithm
from dcsmm.uenvelope import EGM_UE             # Unified UE entry point
```

## Core Modules

- **FUES Algorithm** (`src/dcsmm/fues/`): Fast Upper-Envelope Scan implementation
- **UE Registry** (`src/dcsmm/uenvelope/`): Unified entry point comparing FUES, DCEGM, RFC, CONSAV
- **Example Models**:
  - Retirement choice model (Ishkakov et al. 2017)
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
│   └── retirement/       # Code, params, plots, tables, run_experiment.py
├── experiments/          # PBS/HPC scripts
├── scripts/              # Developer utilities (setup_venv.sh, etc.)
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
