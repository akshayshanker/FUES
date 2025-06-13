# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0dev3] - 2025-06-13 – MPI Error Handling & Memory Optimization

### Added
* **Comprehensive MPI warning suppression**
  - Added environment variables to suppress non-fatal MPI collective communication warnings (`LOG_CAT_ML`, `basesmuma`, `ml_discover_hierarchy`)
  - New MPI configuration variables: `OMPI_MCA_coll_ml_priority=0`, `OMPI_MCA_coll_hcoll_enable=0`, and BTL layer warning suppressions
  - Implemented stderr filtering in MPI scripts to remove noise while preserving genuine errors

* **Numba cache management for MPI environments**
  - Added automatic Numba cache clearing before MPI runs to prevent cache corruption
  - Implemented process-specific cache directories (`NUMBA_CACHE_DIR=/tmp/numba_cache_$$`)
  - Added `NUMBA_DISABLE_CACHE=1` and `NUMBA_NUM_THREADS=1` for MPI safety

* **Memory-efficient model processing workflow**
  - Implemented immediate model processing pattern: solve → extract metrics → generate plots → delete model → garbage collect
  - Added per-model memory cleanup with explicit `del model` and `gc.collect()` calls
  - Replaced batch processing with sequential processing to minimize peak memory usage

* **Enhanced logging and error tracking**
  - Added timestamped log files for both stdout and stderr with `tee` command
  - Implemented comprehensive error logging while maintaining screen output visibility
  - Added run completion status reporting with exit codes

### Changed
* **Solve runner workflow optimization**
  - Modified `solve_runner.py` to process each model individually instead of keeping all models in memory
  - Baseline and fast methods now follow identical solve-plot-delete pattern
  - Replaced `mpi_map` batch processing with individual `runner.run()` calls for better memory control
  - Updated metrics collection to use `all_metrics` list instead of DataFrame concatenation

* **MPI script robustness**
  - Enhanced `circuit_run_HR_mpi.sh` with comprehensive error suppression and cache management
  - Added pre-run cache cleaning and post-run status reporting
  - Implemented filtered stderr to separate MPI noise from application errors

### Fixed
* **Numba compilation race conditions**
  - Resolved `KeyError` exceptions in Numba caching system during concurrent MPI compilation
  - Fixed `ReferenceError: underlying object has vanished` errors during object serialization
  - Eliminated cache corruption issues when multiple MPI processes compile identical functions

* **Memory management issues**
  - Fixed memory accumulation when processing multiple models sequentially
  - Resolved potential memory leaks by ensuring proper model cleanup after plotting
  - Eliminated peak memory spikes by processing models one at a time

* **MPI communication noise**
  - Suppressed non-fatal `basesmuma` component warnings that cluttered error logs
  - Filtered out `ml_discover_hierarchy` and collective communication layer warnings
  - Maintained visibility of genuine MPI errors while removing infrastructure noise

### Technical Details
* **Error patterns addressed:**
  - `KeyError: ((Array(int32, 1, 'C', False, aligned=True), ...))` in Numba caching
  - `ReferenceError: underlying object has vanished` during serialization
  - `[LOG_CAT_ML] component basesmuma is not available` MPI warnings
  - Memory exhaustion from keeping multiple large models in memory simultaneously

* **Environment variables added:**
  ```bash
  NUMBA_DISABLE_CACHE=1
  NUMBA_CACHE_DIR=/tmp/numba_cache_$$
  NUMBA_NUM_THREADS=1
  OMPI_MCA_coll_ml_priority=0
  OMPI_MCA_coll_hcoll_enable=0
  OMPI_MCA_btl_base_warn_component_unused=0
  ```

## [0.3.0dev2] - 2025-06-12 – MPI-enabled `VFI_HDGRID` & root-only workflow

### Changed
* Runner metric now specific to each model -- metrics is local rather than being imported from `dynx.runner.metrics.deviations`

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
    methods; `