# Fast Upper-Envelope Scan (FUES)

Core FUES implementation and examples for Dobrescu and Shanker (2025).

Includes a general-purpose upper envelope class for one-dimensional discrete-continuous EGM problems.

**Work in progress.**

## Installation

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
pip install -e ".[examples]"
```

Or on NCI Gadi (creates a venv on scratch):
```bash
bash scripts/setup_venv.sh
```

Requires Python 3.11+.

## Quick Start

```python
from dcsmm.fues import FUES                    # Main algorithm
from dcsmm.uenvelope import EGM_UE             # Unified UE entry point
from dcsmm.fues.helpers.math_funcs import interp_as  # 1D interpolation
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
