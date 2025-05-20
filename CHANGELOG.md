# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0dev2] - 2025-05-22 – MPI Parameter Sweep

* Fix dumb plotting errors in housing_renting example plots. 
* Convenience scripts for running MPI parameter sweeps for housing renting model


## [0.2.0dev1] - 2025-05-22 – MPI Parameter Sweep

* Clean up no pickling of model classes
* Convenience scripts for running MPI parameter sweeps for housing renting model


## [0.2.0dev0] - 2025-05-21 – Enhanced Parameter Sweep and UE Timing Metrics

* Clean up dependencies with DynX remote packages
* Add `dynx-runner` as a hard dependency
* Added comparison to CONSAV



## [0.1.0a3] - 2025-05-21 – Enhanced Parameter Sweep and UE Timing Metrics

### Added
* Enhanced parameter sweep example with detailed metrics reporting
* Added  UE timing metrics collection and display
* Improved debugging capabilities to diagnose timing issues
* Added dynamic table formatting using the `tabulate` library
* Added support for saving detailed results to CSV files with timestamps

### Changed
* Fixed path resolution issue in `param_sweep.py` to correctly import from examples directory
* Improved metric extraction in CircuitRunner to capture UE timing information
* Enhanced console output with better formatted tables and progress information
* Fixed issues with empty performance tables by ensuring minimum metrics are always present
* Refactored the `enhanced_metric_function` to properly extract timing data from model

## [0.1.0a2] - 2025-05-20 – DynX Runner API Modernization

### Added
* Added support for new DynX v1.6.12 unified `CircuitRunner` and sampler utilities
* Updated housing model examples to use the new API

### Changed
* Refactored `examples/housing_renting/circuit_runner_solving.py` to use the new single-dict `CircuitRunner` constructor
* Replaced manual parameter arrays with `dynx.runner.sampler` helpers for parameter space exploration
* Simplified configuration handling with a unified `base_cfg` dictionary
* Updated parameter path definitions to work with the new sampler interface
* Removed legacy constructor arguments (`epochs_cfgs`, `stage_cfgs`, `conn_cfg`, `param_specs`)
* Streamlined MPI map execution and result handling
* Updated module docstrings to reflect the new API usage

## [0.1.0a1] - 2025-05-13 – Preliminary Public Release

### Added
* First public (alpha) release available on PyPI under the name `dc-smm`.
* Added this `CHANGELOG.md` following the *Keep a Changelog* format and PEP 440 versioning.

### Changed
* **Package layout migrated to a standard *src/* layout**.
  * All source code now lives in `src/`.
* **Namespace renamed** – legacy `FUES.*` imports replaced by the canonical
  `dc_smm.fues.*` (lower-snake-case) path.  A Bowler refactor script
  (`fix_imports.py`) updates downstream code automatically.
* Introduced umbrella namespace package `dc_smm` exposing the two public
  sub-packages `fues` and `uenvelope`.
* Rewritten internal relative imports to absolute ones that honour the new
  package structure.
* Consolidated various ad-hoc upper-envelope implementations into
  `dc_smm.uenvelope.upperenvelope.EGM_UE` with a registry system
  (FUES, DCEGM, RFC, CONSAV, SIMPLE).
* Example scripts updated to use DynX ≥ 1.6.12 canonical import paths:
  `dynx.stagecraft`, `dynx.heptapodx`, `dynx.runner`.
* All path-hacks (`sys.path.append/insert`, `os.chdir`) removed from driver
  scripts in `examples/housing_renting`.
* Added helper module `dc_smm.fues.helpers.math_funcs` with
  `interp_as`, `mask_jumps`, `correct_jumps1d`, etc.
* Replaced bespoke EGM upper-envelope kernels with vectorised implementations
  backed by Numba/JIT wherever feasible.
* `pyproject.toml` modernised to PEP-621:  minimal runtime deps, `build-system`
  section, and SPDX licence expression.

### Removed
* Deprecated shims in `old_deprecated/` (scheduled for deletion in a later
  release).
* Local copies of DynX helper modules – upstream PyPI packages are now the
  single source of truth.

---

See `refactoring_summary.md` for a more granular developer-level breakdown of
refactor steps. 