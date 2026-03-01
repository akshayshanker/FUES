# Installation

## From source (recommended)

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
pip install -e ".[examples]"
```

This installs `dcsmm` in editable mode with all dependencies for running examples (matplotlib, seaborn, pyyaml).

## Core only

```bash
pip install -e .
```

Core dependencies: NumPy, Numba, SciPy, econ-ark (HARK), ConSav, interpolation, dill.

## On NCI Gadi

```bash
bash scripts/setup_venv.sh
```

This auto-detects the Gadi environment, creates a venv on `/scratch`, and installs everything. Activate with:

```bash
source setup/load_env.sh
```

## Requirements

- Python 3.11+
- NumPy >= 1.23
- Numba (JIT compilation for the FUES scan)
- SciPy

## Optional dependencies

| Extra | Install | Provides |
|-------|---------|----------|
| `[examples]` | `pip install -e ".[examples]"` | matplotlib, seaborn, pyyaml for running examples |
| `[mpi]` | `pip install -e ".[mpi]"` | mpi4py for parallel computation |

## Verify installation

```python
from dcsmm.fues import FUES
print("FUES imported successfully")

from dcsmm.uenvelope import EGM_UE
print("EGM_UE registry imported successfully")
```
