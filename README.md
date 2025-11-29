# Fast Upper-Envelope Scan (FUES)

Core FUES implementation and examples for Dobrescu and Shanker (2025).

Includes a general-purpose upper envelope class for one-dimensional discrete-continuous EGM problems.

**Work in progress.**

## Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS

# Install from GitHub
pip install "git+https://github.com/akshayshanker/FUES.git"

# Or clone and install locally
git clone https://github.com/akshayshanker/FUES.git
cd FUES
pip install .
```

Requires Python 3.11+.

## Core Modules

- **FUES Algorithm**: Fast Upper-Envelope Scan implementation
- **Upper Envelope Comparison**: Benchmarking framework for DCEGM upper envelope algorithms
- **Example Models**:
  - Continuous durables model
  - Retirement choice model (Ishkakov et al. 2017)
  - Housing-renting model with time-inconsistent preferences

## External Packages

- **[ConSav](https://github.com/NumEconCopenhagen/ConsumptionSaving)**: G2EGM implementation
- **[HARK](https://github.com/econ-ark/HARK)**: DCEGM implementation

## Directory Structure

```
├── src/dc_smm/           # Main source code
│   ├── fues/             # FUES algorithm
│   ├── uenvelope/        # Upper envelope comparison framework
│   └── models/           # Model-specific solvers
├── examples/             # Example implementations
│   ├── durables/
│   ├── housing_renting/
│   └── retirement/       # Plotting, tables, benchmarks
├── experiments/          # Experiment runners and configs
│   ├── housing_renting/  # PBS scripts, job configs
│   └── retirement/       # CLI runner, YAML params
├── scripts/              # Utility scripts
│   └── int/              # Interactive session scripts
├── logs/                 # HPC job logs
└── tests/
```