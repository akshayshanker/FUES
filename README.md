# Discrete-Continuous Dynamic Programming

Repository containing solvers and key functions to solve discrete-continuous dynamic programming (DCDP) problems on high-performance clusters.

**Work in progress and incomplete branch.**

## Core Modules

- **Fast Upper-Envelope Scan (FUES)**: Original implementation of the FUES algorithm (serves as replication reference)
- **Upper Envelope Comparison**: Benchmark class to compare performance of different DCEGM upper envelope algorithms in Python
- **Example Solvers**:
  1. Model with continuous durables
  2. Simple retirement model
  3. Housing-renting model with time-inconsistent preferences

## External Upper Envelope Packages

The upper envelope comparison framework includes:
- **[ConsumptionSaving (ConSav)](https://github.com/NumEconCopenhagen/ConsumptionSaving)**: One dimensioanl version of G2EGM implementation
- **[DCEGM](https://github.com/econ-ark/HARK)**: Implementation in HARK (econ-ark)

## Directory Structure

- `docs/`: Documentation files (papers, slides)
- `examples/`: Example model implementations
  - `durables/`: Continuous durables model
  - `housing_renting/`: Housing-renting model with time inconsistency
  - `retirement/`: Retirement choice model
- `experiments/`: Scripts for running experiments and parameter sweeps
  - `durables/`
  - `housing_renting/`
  - `retirement/`
- `results/`: Experiment results (plots, tables)
- `scripts/`: Helper scripts for HPC deployment
  - `int/`: Interactive session scripts
  - `lib/`: Configuration and reference files
  - `pbs/`: PBS job submission scripts
    - `logs/`: HPC execution logs
- `src/`: Main source code
  - `dc_smm/`: Discrete-continuous SMM implementation
    - `fues/`: FUES algorithm implementations
    - `helpers/`: Generic helper functions
    - `models/`: Model-specific solvers ("workhorses")
    - `uenvelope/`: Upper envelope comparison framework
- `tests/`: Test suite