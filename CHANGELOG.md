# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0dev1] - 2025-06-07 – MPI-enabled `VFI_HDGRID` & root-only workflow

### Added
* **MPI parallelization for `VFI_HDGRID`**
  - New memory-slim MPI implementation that scatters value-function slices to workers instead of broadcasting the full tensor.
  - Workers hold virtually zero memory after each stage, enabling large-scale runs on clusters (e.g. NCI Gadi).
  - Provides bit-for-bit identical results between serial and MPI modes.

* **Two-step baseline workflow**
  - New CLI flags (`--baseline-only`, `--use-baseline`, `--fresh-fast`) for separating expensive HD-grid construction from fast method comparisons.
  - Allows building a baseline once on many cores and reusing it for subsequent fast solver runs on a single core.
  - Leverages CircuitRunner's built-in `save_by_default` and `load_if_exists` functionality.

* **MPI-aware operator factories & solvers**
  - Updated `horses_c.py` and `whisperer.py` to be rank-aware.
  - Workers now receive lightweight stub `Solution` objects for non-MPI stages, preventing deadlocks and memory bloat.
  - No heavy `.sol` objects are ever broadcast back to workers.

### Changed
* **Streamlined terminal value initialization**
  - The `initialize_terminal_values` function in `whisperer.py` now only processes consumption stages (`OWNC` and `RNTC`), eliminating wasteful placeholder grids for housing and tenure stages.
  - Saves 150-300 MB of RAM on large grids and speeds up terminal pass by ~10%.

### Removed
* **Legacy broadcast MPI mode** and `--legacy-bcast` flag.
* **Redundant synchronization calls** (`_sync_perch_solutions`) from `whisperer.py`.
* **Unused utility functions** and imports for a cleaner, more maintainable codebase.
* **Over-engineered baseline I/O** in favor of CircuitRunner's native bundle management.

### Fixed
* **Hash collision bug** where `__runner.mode` was incorrectly included in `param_paths`, preventing fast methods from loading the correct baseline bundle.
* **Deadlocks** caused by workers returning `None` instead of lightweight stubs for non-MPI stages.
* **Unnecessary recomputation** of fast methods when a baseline was loaded.
* **Timing metrics** now correctly captured and displayed in the summary tables.

## [0.2.0 dev4] - 2025-06-09 – **MPI-safe baseline & lean workers**

### Changed
* mpi_run takes in a solver communicator which splits each run across solvers. Only master rank processes the metrics and loading/saving. 
* outputs across mpi and non-mpi runs are not consistent. 
* basic HF vf grid comparison for housing renting model using MPI. (compares to single parameter run in circuit_runner_solving.py)

### Added

* **Root-only metrics path**
  `CircuitRunner.run()` now skips the expensive `metric_fns` block on
  non-root ranks; workers only return lightweight timing info.
  → prevents N× baseline reloads and cuts RAM usage on large jobs.

* **Global MPI helpers** (`_MPI_COMM / _MPI_RANK / _MPI_SIZE`) initialised
  once at import time; used throughout the runner/solver stack to gate code
  that should execute only on rank 0.

### Changed

* **`mpi_map()` rewritten for clarity**

  * Always returns a *pair* `(df, models)` (second element `[]` when models
    are not gathered).
  * Serial code-path untouched; MPI path defers metrics to rank 0.

* **Stage compilation log-level**
  `compile_all_stages()` prints *INFO* messages only when the caller set
  `--verbose`; otherwise it downgrades to *DEBUG* to keep worker logs clean.

* **Config patching** (`patch_cfg`)

  * Consumption stages now carry a cheap `"compute": "SINGLE"` flag for fast
    methods; `"MPI"` is used only for `VFI_HDGRID`.
  * Prevents workers from loading the HD grid bundle when they are solving a
    fast-method row.

* **Solver factory** – workers still solve their share of the VFI grid, but
  the root rank alone runs the post-solve metric evaluation.

### Fixed

* Endless recursion / memory blow-ups when every rank attempted to load the
  baseline bundle to compute metrics.
* `inspect` import was missing in the metric-signature branch – now imported
  once at the top of the guarded section.
* Spurious warnings about missing numerical models on worker ranks
  (initialise/compile order tightened).

### Removed

* Redundant per-rank saving of identical bundles; only rank 0 writes to disk
  (flag `save_by_default` automatically false on workers).

---

*(Previous entries unchanged – see below for full history.)*


## [0.2.0dev3] - 2025-05-22 – Solution Container 

### Changed
* **Breaking**: `perch.sol` is now an instance of `stagecraft.solmaker.Solution`. Old code using dictionary syntax continues to work, but the object is no longer a plain dict.
* Updated all solver operators (`horses_c.py`, `horses_h.py`, `horses_t.py`) to return Solution objects
* Modified `whisperer.py` to handle both Solution objects and legacy dictionaries
* Updated plotting utilities with helper function to access both Solution and dict formats
* Refactored EGM grid storage to use nested structure (unrefined/refined/interpolated layers)

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