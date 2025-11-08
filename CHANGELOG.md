# Changelog

All notable changes to this project will be documented in this file.

## [0.5.0dev0] - 2025-08-12 – Multi-GPU Support and FUES Algorithm Cleanup
- [2025-08-16 10:00 AEST] Major refactoring: Removed MPI support from horses_c.py, removed unused F_ownc_cntn_to_dcsn factory, standardized terminology
- [2025-08-17 17:00 AEST] Added DGX A100 support with specialized PBS scripts, GPU kernel optimizations, and log management utilities
- [2025-08-23 11:21 AEST] Fixed numerical stability issues in FUES algorithm for delta != 1 case
- [2025-10-25 16:30 AEST] Fixed ZeroDivisionError in piecewise_gradient_3rd_filtered by adding zero-division protection throughout gradient computation
- [2025-10-25 17:00 AEST] Enhanced _egm_preprocess_core to only add jump constraints for segments with at least 4 points, improving numerical stability
- [2025-10-25 17:30 AEST] Added asset policy monotonicity filter in horses_c.py to remove points where refined asset policy is decreasing
- [2025-10-25 18:00 AEST] Fixed boolean operator error in _egm_preprocess_core (changed 'or' to '|' for array operations)
- [2025-10-25 18:15 AEST] Fixed array allocation in _egm_preprocess_core to correctly account for 2 segments per jump
- [2025-10-27 15:30 AEST] Added skip_egm_plots flag to conditionally skip EGM CSV exports (plot_csv_export.py, solve_runner.py)
- [2025-10-27 16:00 AEST] Implemented conditional EGM grid storage: skip saving EGM grids to Solution object when --skip-egm-plots enabled, reducing memory usage and pickle sizes (horses_c.py, solve_runner.py)
- [2025-10-27 16:35 AEST] Fixed AttributeError in make_housing_model: corrected mc.periods to mc.periods_list for flag injection (solve_runner.py)
- [2025-10-27 16:45 AEST] Added optional asset policy gradient filtering: filter_a_jumps setting removes refined grid points where da/dm exceeds max_a_gradient threshold (horses_c.py, master.yml)
- [2025-11-08 12:00 AEST] Implemented first-order condition (FOC) checks in _egm_preprocess_core to filter constraint points based on economic optimality: only add points that satisfy Kuhn-Tucker conditions with scaled lambda values (horses_common.py, horses_c.py)
- [2025-11-08 12:15 AEST] Modified image saving to always use timestamped directories (images_YYYYMMDD_HHMMSS) to preserve all previous runs instead of overwriting old image files (execution_settings.py, solve_runner.py)
- [2025-11-08 12:30 AEST] Added uc_test function in horses_common.py as a simple test marginal utility for FOC verification, imported into horses_c.py for testing purposes
- [2025-11-08 12:45 AEST] Fixed incorrect double method directory creation: removed redundant method subdirectory since we're already in bundles/hash/METHOD/images_TIMESTAMP/ structure (plots.py, plot_csv_export.py)

### Added
* **Multi-GPU MPI parallelization for housing model**
  - Single-node support for up to 4 GPUs with MPI
  - Multi-node support for scaling across Gadi nodes (8+ GPUs)
  - NUMA-aware CPU binding with 12 cores per MPI rank
  - MPI dispatcher in horses_h.py with GPU detection and fallback
  - MPI driver in horses_c_gpu.py with Allgatherv collectives
  - Shared cache for grid. 

* **PBS scripts for GPU scaling**
  - `run_housing_gpu_mpi.pbs`: Single-node 4 GPU execution
  - `run_housing_gpu_multi_node.pbs`: Multi-node 8 GPU execution  
  - `benchmark_gpu_scaling.pbs`: Automated 1, 2, 4 GPU performance comparison
  - Scripts use same options as single-GPU version for consistency
  - `run_housing_dgxa100_single.pbs`: DGX A100 single GPU job (512GB RAM, 80GB GPU)
  - `run_housing_dgxa100_parallel.pbs`: DGX A100 4-GPU parallel execution
  - `submit_dgxa100_config.sh`: Submit helper accepting multiple configurations
  - `move_logs_to_scratch.sh`: Utility to move all logs to scratch storage

### Changed
* **FUES algorithm cleanup and configurability**
  - Removed 4 unused functions (uniqueEG, linear_interp, seg_intersect, line_intersect_unbounded)
  - Made epsilon parameters configurable (eps_d, eps_sep, eps_fwd_back, parallel_guard)
  - Consolidated duplicated intersection logic into _forced_intersection_twopoint() helper
  - Moved helper functions to src/dc_smm/fues/helpers/math_funcs.py
  - Merged FUES_sep_intersect into main FUES function with return_intersections_separately flag

* **GPU implementation modifications**
  - Added initialize_vfh_from_config() function for MPI initialization
  - Modified V_out calculation in kernel to use formula: V = (Q - (1-delta)*u(c,h)) / delta
  - Implemented C-contiguous array handling for MPI operations
  - Convergence check performed before array swap using allreduce(MAX)
  - Policy gathering made conditional via policy_every parameter
  - Two-pass grid search in vfi_gpu_kernel: coarse then fine search (~25% fewer evaluations)
  - Pre-computed log_H_term for housing utility (avoids redundant calculations)
  - Branchless operations using max() instead of if-statements
  - Immediate memory cleanup after GPU transfers for large arrays

### Architecture
* **MPI implementation structure**
  - MPI logic isolated in solver layer (horses_h.py, horses_c_gpu.py)
  - Whisperer module unchanged - no modifications required
  - 1 MPI rank mapped to 1 GPU
  - Precomputed Allgatherv counts and displacements

## [0.4.0dev11] - 2025-08-11 – FUES Algorithm Stability Improvements

### Enhanced
* **FUES intersection calculations**
  - Rewrote intersection logic to handle near-parallel segments robustly
  - Added forced intersection points that guarantee envelope continuity  
  - Intersection coordinates now strictly bounded within valid intervals
  - Averaging technique reduces numerical drift at segment boundaries

* **Branch detection and continuity**
  - New branch detection checks both gradient thresholds and point proximity
  - Safe extrapolation finds suitable points when direct neighbors unavailable
  - Circular buffer for backward scanning improves memory efficiency
  - Forward scan validates jumps using combined value and gradient criteria

* **Consecutive jump handling**
  - Prevents numerical instabilities from multiple policy jumps in sequence
  - Drops previous jump point when consecutive jump detected
  - Maintains index consistency by removing associated intersections
  - Rule only enforced when current jump passes validation

### Fixed
* **Numerical stability issues**
  - Adjusted epsilon constants for better numerical behavior (EPS_D from 1e-200 to 1e-20)
  - Added parallel line guard (1e-12) for degenerate geometry detection
  - Intersection capacity increased to 2*(N-1) preventing silent truncation
  - Eliminated spurious Euler residuals at policy kinks
  - [2025-08-23] Improved float64 numerical stability for delta != 1 case:
    * Changed EPS_D from 1e-50 to 1e-14 (safe for float64 precision)
    * Increased PARALLEL_GUARD to 1e-10 for better parallel line detection
    * Added explicit float64 dtype enforcement in FUES and egm_preprocess
    * Enhanced uniqueEG() to handle near-duplicate points with tolerance
    * Fixed consumption lower bounds (1e-100 to 1e-10) in horses_common.py

### Performance
* **Memory optimizations**
  - Pre-allocated arrays reduce allocation overhead in hot loops
  - Circular buffer implementation minimizes memory churn
  - Uniform index bookkeeping simplifies maintenance and debugging

## [0.4.0dev10] - 2025-08-03 – GPU Performance Optimizations
  - [2025-08-02 18:38 AEST] Improved CLAUDE.md documentation with better organization, version management discipline, and incorporated feedback from o3pro.
  - [2025-08-03 17:30 AEST] Fixed GPU scaling issue by implementing memory freeing during solve to prevent 193GB+ memory accumulation
  - [2025-08-03 18:15 AEST] Enhanced memory management to completely free periods 2+ while preserving periods 0,1 for Euler error calculation
  - [2025-08-03 18:45 AEST] Fixed Euler error GPU bottleneck by implementing sampling-based calculation for large grids to prevent 100GB+ memory transfers
  - [2025-08-03 15:30 AEST] Created multi-job PBS submission system for running multiple GPU configurations in parallel
  - [2025-08-03 16:00 AEST] Added income process generation script using Fella (2014) parameters for housing model
  - [2025-08-03 16:30 AEST] Verified memory freeing implementation is complete but jobs crashing before benefits visible
  - [2025-08-03 19:00 AEST] Identified issue with FUES algorithm dropping points after policy function jumps - needs intersection fallback when scans fail
  - [2025-08-08 13:37 AEST] Fixed left/right branch assignment in FUES intersection calculation - new branch should be on right (higher e_grid values), old branch on left
  - [2025-08-08 14:15 AEST] Implemented extrapolated segment intersections (extrap_segments_05_08082025_v1) - adds fallback extrapolation when forward/backward scans fail to find bracketing points, ensuring continuous piecewise-linear envelope
  - [2025-08-08 15:34 AEST] Simplified solve_runner.py Phase 1 - extracted configuration management into ConfigurationManager class, reducing main() complexity while maintaining full PBS compatibility
  - [2025-08-12 17:15 AEST] Cleaned up fues.py - removed 4 unused functions (uniqueEG, linear_interp, seg_intersect, line_intersect_unbounded) and made epsilon parameters (eps_d, eps_sep, eps_fwd_back, parallel_guard) configurable as optional function arguments while maintaining backward compatibility 
  - [2025-08-12 18:30 AEST] Consolidated duplicated intersection geometry logic in fues.py - created _forced_intersection_twopoint helper function to eliminate ~150 lines of duplicate code across Cases A, C.1, and C.2, while ensuring all epsilon parameters are properly passed through the function hierarchy
  - [2025-08-12 19:00 AEST] Merged `FUES` and `FUES_sep_intersect` functions in fues.py - consolidated into a single `FUES` function with a `return_intersections_separately` flag for simplified API and improved maintainability.
  - [2025-08-12 19:15 AEST] Cleaned up fues.py formatting - removed redundant comments, excessive blank lines, and obvious inline comments to improve code readability while maintaining functionality
  - [2025-08-12 19:30 AEST] Simplified function signatures in fues.py - refactored _forced_intersection_twopoint and add_intersection_from_pairs_with_sep to accept L and R as tuples instead of 20 individual parameters, improving code clarity
  - [2025-08-12 19:45 AEST] Fixed Numba compilation issue - removed @njit decorator from FUES wrapper function as it's unnecessary (only _scan needs JIT compilation) and was causing return type inconsistency errors
  - [2025-08-12 20:00 AEST] Refactored FUES helpers - moved intersection and circular buffer utilities from fues.py to helpers/math_funcs.py for better code organization and reusability.
  - [2025-08-12 20:15 AEST] Applied PEP8 formatting to fues.py - cleaned up whitespace, fixed spacing around operators, improved line breaks for better readability
  - [2025-08-12 20:30 AEST] Fixed constants handling - moved EPS_D, EPS_SEP, and PARALLEL_GUARD constants from math_funcs.py back to fues.py where they belong, removed default parameter values that used these constants
  - [2025-08-08 15:45 AEST] Renamed ConfigurationManager to ExecutionSettings to distinguish PBS execution settings from model configuration YAML
  - [2025-08-08 16:15 AEST] Implemented clean left/no jump logic (clean_left_no_jump_logic_05_08082025.md) - allows consecutive no-jump left turns while preventing consecutive jumps via demotion, adds jump_now condition to intersection logic, ensures uniform index bookkeeping across all cases
  - [2025-08-08 16:25 AEST] Fixed NameError in solve_runner.py - corrected missed variable rename from cfg_container to model_config in CircuitRunner initialization
  - [2025-08-08 16:40 AEST] Fixed critical FUES implementation errors causing all points to be dropped:
    * Fixed undefined variable 'left_turn' -> 'left_turn_any' in backward_scan_combined call
    * Fixed uninitialized variable 'j' in first iteration (i=0)
    * Added missing 'not_allow_2lefts' parameter to both _scan function calls in FUES and FUES_sep_intersect wrappers
  - [2025-08-08 16:50 AEST] Applied additional FUES fixes from 05pro_fues_dev1_fixes.md:
    * Fixed index update logic in Case C.1 when j is dropped - prev_j now correctly points to k (current tail) instead of dropped j
    * Fixed value-fall state flags - last_turn_left now correctly set to False (value fall is not a geometric turn)
    * Increased intersection capacity from N//2 to 2*(N-1) to prevent silent truncation in pathological cases
  - [2025-08-08 17:05 AEST] Implemented _scan_v2 from right_as_left_no2jumps.md - cleaner, more compact FUES implementation:
    * Single case_id encoding (turn<<1)|jump for simpler branching logic (4 cases: RTNJ, RTJ, LTNJ, LTJ)
    * Different consecutive jump handling: keeps second jump but drops previously jumped-to point j and undoes its intersections
    * Uniform index updates across all cases for better maintainability
    * Intersections only added on jump iterations with robust extrapolation fallback
    * Updated both FUES and FUES_sep_intersect wrappers to use _scan_v2
  - [2025-08-08 17:20 AEST] Applied no_two_jumps.md refinement - only enforce consecutive jump rule when current jump is kept:
    * Removed early unconditional consecutive jump enforcement block
    * RTJ case: only drops previous j when keep_i1 is True (current jump is validated and kept)
    * LTJ case: enforces rule at start since i+1 is always kept by construction
    * Ensures "no two jumps" rule only applies when we're actually accepting the current jump
  - [2025-08-08 18:00 AEST] Implemented strict bracket enforcement for FUES intersections - ensures intersections always lie within (e_j, e_{i+1}):
    * Replaced loose e_min/e_max window check with strict _between_open(intr_x, e_grid[j], e_grid[i+1], EPS_SEP) validation
    * Added _clip_open to clamp intersection x-coordinate into valid interval with safety margin
    * Recompute intersection y-coordinate at clamped x using both line equations and average
    * Applied to all three intersection cases: Case A (right-turn jump), Case C.1 (left turn, j dropped), Case C.2 (left turn, j kept)
    * Prevents spurious off-interval intersections that corrupt envelope geometry on next iteration
  - [2025-08-09 10:00 AEST] Implemented forced intersection points for all kept jumps - eliminates Euler equation residual gaps:
    * Added _force_crossing_inside() function to guarantee valid intersections even for near-parallel lines
    * Implemented adaptive separation min(EPS_SEP, 0.25*interval_length) to handle small intervals
    * Modified all three cases (RTJ, LTJ j-dropped, LTJ j-kept) to use forced intersections
    * Ensures piecewise-linear envelope with explicit kinks at all discrete choice switches
  - [2025-08-09 11:00 AEST] Added comprehensive debug printing for intersection analysis:
    * Added debug parameters to _scan and FUES functions with specific region filtering
    * Prints intersection details including flag, point, indices, liquid savings values, and policies
    * Default debug region set to e: [31.3, 32], v: [6.71, 6.75] for targeted analysis
  - [2025-08-09 11:30 AEST] Replaced complex interpolation with cleaner implementation for non-ConSav methods:
    * Added interp_clean() function with simpler, more robust extrapolation logic
    * Modified horses_c.py to use interp_clean for Q_dcsn and policy interpolation when method != "CONSAV"
    * Addresses suspected interpolation issues causing FUES instabilities
  - [2025-08-09 12:00 AEST] Enhanced forward scan logic in Case A (right-turn jump):
    * Added jump verification when g_1 > g_f_vf_at_idx condition is met
    * Now checks if gradient from i+1 to idx_f exceeds jump threshold (m_bar)
    * Only sets keep_i1=True when both value condition AND jump are confirmed
  - [2025-08-10 10:15 AEST] Refactored `e_grid` to `x_dcsn_hat` in `fues.py` for improved clarity and consistency with paper notation.

### Fixed
* **CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES**
  - Reduced thread block sizes from 1024 to 256-512 threads to avoid GPU resource exhaustion
  - VFI kernel: (8,8,8)=512, Housing Owner: (8,8,4)=256, Housing Renter: (16,16)=256
  - All GPU kernels now launch successfully with HIGH_RES_SETTINGS

### Added
* **GPU-accelerated shock integration**
  - New `shock_integration_kernel` replaces CPU-based np.einsum
  - Automatic GPU dispatch when compute="GPU" and problem size > 1000
  - Expected 5-10x speedup for shock integration operations

* **Development specifications**
  - `multi_gpu_parallel_architecture_03082025.md`: Full 4-GPU + 48-CPU architecture
  - `vfi_hdgrid_gpu_parallel_03082025.md`: Focused VFI-only 4-GPU parallelization

### Changed
* **GPU kernel optimizations**
  - VFI kernel restored to proper 3D parallelization (was 2D with loop)
  - Fixed serialization bottleneck in wealth dimension
  - All kernels now use balanced thread configurations for stability
  - Expected 3-5x speedup from proper parallelization

### Technical Details
* **Resource management:**
  - Complex kernels require fewer threads due to register pressure
  - Trade-off: more kernel launches but successful execution
  - GPU memory usage: ~400MB for test grids, scales linearly

## [0.4.0dev9] - 2025-08-02 – GPU Underutilization Fix

### Fixed
* **GPU underutilization warning for small grids**
  - Fixed "Grid size 1 will likely result in GPU under-utilization" warning
  - Added adaptive thread block sizing when grid dimensions are very small
  - Ensures minimum GPU occupancy by adjusting thread configuration dynamically
  - Affects: horses_h.py (owner/renter choice) and horses_c_gpu.py (VFI solver)
  - Small test grids now launch with better GPU utilization

### Technical Details
* **Adaptive kernel configuration:**
  - Detects when total blocks would be ≤ 2 and reduces thread block size
  - Maintains correctness while improving GPU occupancy for test configurations
  - Example: 1×1 grid now uses 1×1 threads instead of 16×16, avoiding warnings

## [0.4.0dev8] - 2025-08-02 – GPU Kernel Fix and FUES Algorithm Reorganization

### Fixed
* **GPU VFI kernel launch failure**
  - Fixed missing `@cuda.jit` decorator on `calculate_continuation_values_gpu_kernel` function
  - Resolved `CUDA_ERROR_INVALID_VALUE` by converting 3D grid to 2D grid with internal loop
  - Changed from `cuda.grid(3)` to `cuda.grid(2)` to avoid CUDA's Z-dimension limit (65535 blocks)
  - Reduced thread block configuration from (16,16,4) to (16,16) for better compatibility
  - GPU kernel now handles 4+ million grid points without exceeding CUDA limits

### Changed
* **FUES algorithm version reorganization**
  - Renamed `fues_2dev5.py` → `fues.py` as current production version
  - Renamed original `fues.py` → `fues_v0dev.py` (October 2024 paper version)
  - Moved all experimental versions (fues_2dev1-8) to `src/dc_smm/fues/experimental/`
  - Updated all method references: `FUES2DEV5` → `FUES`, `FUES2DEV*` → `FUES`
  - Upper envelope registry updated: `@register("FUES")` for production, `@register("FUES_V0DEV")` for paper version

* **Repository cleanup for public release**
  - Enhanced .gitignore to exclude HPC output files, backup directories, working notes
  - Added `examples/README_OUTPUTS.md` explaining output directory structure
  - Excluded all generated images/results from version control (best practice)
  - Python build artifacts (*.egg-info) now properly ignored

### Technical Details
* **GPU fix details:**
  - Problem: 3D grid with dimensions (250, 250, 64) = 4M points exceeded CUDA Z limit
  - Solution: 2D grid (n_H, n_Y) with internal loop over n_W dimension
  - Maintains same computation pattern while respecting CUDA architecture limits
  
* **Files reorganized:**
  - `src/dc_smm/fues/__init__.py` - Updated imports
  - `src/dc_smm/uenvelope/upperenvelope.py` - Updated engine registrations
  - 11 example/test files updated with new method references
  - Fixed all legacy import paths (dc_smm.fues.legacy.* no longer exists)

* **Repository structure:**
  ```
  src/dc_smm/fues/
  ├── fues.py              # Current production (was fues_2dev5)
  ├── fues_v0dev.py        # Original paper version
  └── experimental/        # All experimental versions
  ```

### Performance Impact
* GPU kernel now successfully launches for high-resolution grids
* Expected 3-5x speedup for VFI GPU solver vs CPU
* Eliminates memory transfer bottleneck by keeping computation on device

## [0.4.0dev7] - 2025-07-31 – Walltime Optimization and Selective Model Loading

### Added
* **Comparison metrics filtering for baseline-only runs**
  - Added `--comparison-metrics` parameter to specify which metrics require baseline loading
  - Automatically skips comparison metrics when running only baseline method to prevent self-comparisons
  - Saves ~45 minutes of unnecessary computation on baseline-only GPU runs

* **Selective model loading for memory efficiency**
  - Added `--load-periods` parameter to load only specific period indices
  - Added `--load-stages` parameter for fine-grained stage filtering per period
  - Reduces loading from 75 to 18 pickle files (76% reduction) for Euler error calculations
  - Integrated with DynX's enhanced load_circuit() function

### Changed
* **Smart metric execution based on method selection**
  - Baseline method now temporarily excludes comparison metrics during its own execution
  - Comparison metrics (dev_c_L2, plot_c_comparison, plot_v_comparison) only run for fast methods
  - Prevents meaningless baseline vs baseline comparisons that always return 0
  - Improves walltime efficiency for GPU baseline computations

* **Updated single-core loading script**
  - Modified `run_housing_single_core.sh` to use selective loading for existing models
  - Added explanatory comments about loading requirements for Euler error
  - Maintains backward compatibility when parameters not specified

### Fixed
* **GPU walltime exceeded errors**
  - Identified that metrics calculation phase was pushing baseline runs over 10-hour limit
  - Baseline solving completed at 9h 13m, but metrics added >47m causing walltime kill
  - Solution: skip unnecessary comparison metrics on baseline-only runs

### Technical Details
* **Files modified:** 
  - `examples/housing_renting/solve_runner.py` - Added comparison metrics filtering and loading options
  - `scripts/pbs/run_housing_single_core.sh` - Added selective loading parameters
  - `examples/housing_renting/helpers/euler_error.py` - Added precompilation function
* **Euler error requirements:** Only needs Period 0 (OWNC stage) and Period 1 (all stages)
* **Performance impact:** Prevents walltime exceeded errors, reduces I/O by 76% when loading models
* **Integration:** Works with DynX v1.7.0 selective loading features

### Performance
* **Euler error precompilation**
  - Added `precompile_euler_error_cpu()` function to warm up Numba JIT cache
  - Eliminates ~30-60 second compilation overhead on first Euler error calculation
  - Automatically runs during initialization when euler_error metric is requested
  - Uses minimal dummy data for fast compilation
  - Fixed utility function expressions to match standard CRRA housing model

* **Metric-specific selective loading for comparison metrics**
  - Comparison metrics now load only Period 0, OWNC stage from baseline (instead of all 5 periods)
  - Reduces baseline loading from 75 to 3 pickle files per comparison (96% reduction)
  - Each fast method saves ~42 seconds on baseline loading for comparisons
  - Total time saved for 4 fast methods: ~168 seconds

## [0.4.0dev6] - 2025-07-26 – FUES Code Cleanup and Optimization

### Changed
* **Refactored FUES scan logic for better code organization**
  - Extracted forward scan logic into dedicated `forward_scan_case_a()` function
  - Combined backward scan and find_backward_same_branch into unified `backward_scan_combined()` function
  - Eliminated nested loops in favor of cleaner function calls while maintaining exact algorithm behavior
  - Removed redundant pre-allocated arrays (g_f_vf, g_f_a, g_m_vf, g_m_a) with on-the-fly computation
  - Memory savings of 4*N floats per scan operation

* **Fixed circular buffer iteration order**
  - Discovered that fues_2dev1 had incorrect backward scan order (oldest to newest instead of newest to oldest)
  - fues_2dev4 correctly implements the intended behavior: selecting the closest (most recent) point
  - Both versions kept for comparison purposes with documented behavioral differences

* **Improved numerical stability for intersection points**
  - Changed intersection point separation from 1e-50 to 1e-8
  - Prevents divide-by-zero errors in numpy gradient calculations
  - Maintains accuracy while avoiding numerical precision issues

### Fixed
* **Index consistency bug**
  - Fixed idx_f being used as both loop counter and grid index
  - Now correctly stores actual grid index: `idx_f = i+2+f`
  - Ensures correct segment selection for intersection calculations

* **Missing circular buffer updates**
  - Added missing `m_head = circ_put(m_buf, m_head, j)` when j is dropped
  - Fixed consecutive left turn handling to properly maintain buffer state
  - Added prev_j tracking for correct j restoration

* **Spurious intersection handling**  
  - Added `added_intersection_last_iter` flag to track intersection creation
  - Remove last intersection on consecutive left turns to avoid spurious points
  - Improved intersection point management for discrete choice switches

### Technical Details
* **Files modified:** 
  - `src/dc_smm/fues/fues_2dev4.py` - Refactored version with correct backward scan
  - `src/dc_smm/fues/fues_2dev1.py` - Original version with backward scan bug (kept for comparison)
  - `src/dc_smm/fues/fues_2dev1_working_backup_dev1.py` - Backup of working version
* **Performance impact:** Reduced memory allocation and improved cache locality
* **Backward compatibility:** Both versions produce valid upper envelopes, just with different point selection in edge cases

## [0.4.0dev5] - 2025-01-28 – Enhanced FUES with Intersection Points

### Added
* **Intersection point tracking in FUES algorithm**
  - Implemented intersection point detection as described in Dobrescu & Shanker (2023) Section 2.1.3
  - Added `add_intersections` parameter to `FUES()` function (default: True) for enhanced accuracy around crossing points
  - Forward scan intersection detection during right-turn jumps identifies where choice-specific value functions cross
  - Backward scan intersection storage during left-turn elimination captures suboptimal point intersections
  - Intersection points include interpolated policy values at crossing locations for complete solution representation

* **Memory-efficient intersection storage**
  - Pre-allocated intersection arrays (10% of grid size) to maintain O(n) complexity
  - Automatic merging of original EGM points with intersection points, sorted by endogenous grid values
  - Configurable intersection tracking with backward compatibility when disabled

### Changed
* **Enhanced `_scan` function with intersection tracking**
  - Added `track_intersections` and `policy_2` parameters for comprehensive intersection detection
  - Consistent return format for all code paths to maintain Numba compatibility
  - Improved boundary checking in forward scan to prevent array index errors

### Technical Details
* **Intersection detection algorithm:**
  ```python
  # Forward scan: detect crossings when jumping to new value function branch
  inter_point = seg_intersect(p1, p2, p3, p4)  # Line-line intersection
  # Interpolate policies at intersection point
  t = (inter_point[0] - e_grid[i+1]) / (e_grid[b_idx] - e_grid[i+1])
  inter_p1[n_inter] = (1-t) * a_prime[i+1] + t * a_prime[b_idx]
  ```
* **Files modified:** `src/dc_smm/fues/fues_2dev1.py`
* **Performance impact:** Minimal overhead when intersections disabled; ~10% memory increase when enabled
* **Accuracy improvement:** Better representation of value function upper envelope around choice-specific crossings

## [0.4.0dev4] - 2025-01-28 – EGM Plotting Fix & Memory Management Enhancements

### Fixed
* **EGM plots generation for all EGM-based methods** 
  - Fixed key prefix mismatch in `plot_egm_grids()` function where plotting code was looking for unprefixed keys (e.g., "0-7") but EGM data was stored with prefixed keys (e.g., "e_0-7", "Q_0-7")
  - EGM plots now generate correctly for FUES2DEV, CONSAV, DCEGM, and other EGM-based methods
  - Updated both unrefined and refined grid access to use proper prefixed key formats
  - Added proper error handling for missing EGM data components

### Changed
* **Plot metrics configuration logic** 
  - Plot metrics are now only included in computation when explicitly requested in `--metrics` list
  - Removed incorrect behavior where `--plots` flag would automatically include plot metrics in computation
  - Improved separation between traditional plot generation (`--plots`) and plot-based metric computation (`--metrics plot_c_comparison`)

### Added
* **Comprehensive debugging for EGM data flow**
  - Added targeted debugging in `plot_egm_grids()` to verify EGM data availability and key formats
  - Enhanced error messages for missing or malformed EGM grid data
  - Created systematic approach for debugging data flow from solution storage to plotting

### Technical Details
* **Key format changes in plots.py:**
  ```python
  # Before (incorrect):
  e_grid_unrefined = egm_data["unrefined"][grid_key]  # Looking for "0-7"
  
  # After (correct):
  prefixed_e_key = f"e_{grid_key}"  # Looking for "e_0-7"
  e_grid_unrefined = unrefined_dict.get(prefixed_e_key)
  ```
* **Files modified:** `examples/housing_renting/helpers/plots.py`, `examples/housing_renting/solve_runner.py`
* **Impact:** Visual validation of endogenous grid method upper envelope refinement process now available for all EGM-based methods

## [0.4.0dev3] - 2025-07-20 – Streamlined Method Configuration & CONSAV Fix

### Added
* **Dynamic baseline method selection** via `--baseline-method` flag with auto-detection based on `--gpu` flag
* **Configurable fast methods** via `--fast-methods` flag (default: FUES2DEV,CONSAV)
* **Automatic baseline inclusion** via `--include-baseline` flag for cleaner single-core workflows
* **Enhanced module docstring** with comprehensive examples for GPU, MPI, and baseline loading workflows

### Changed
* **Method configuration** no longer requires editing source code - all methods configurable via command-line flags
* **Backward compatibility** maintained - existing scripts work without modification

### Fixed
* **CONSAV engine argument handling** - fixed `AttributeError` when `u_func["args"]` expects dictionary format
* **Method selection logic** streamlined to eliminate hardcoded baseline/fast method lists

## [0.4.0dev2] - 2025-07-20 – Metric Accuracy

### Added
* **Professional, publication-quality comparison plots** for policy and value functions via `plot_comparison_factory` in `helpers/metrics.py`.
* **Plots now use proper interpolation**: Both fast and baseline methods are compared on a common grid, matching the logic used in L2 error metrics.
* **Plots are saved in the bundle directory** for each parameter/method, keeping results organized and reproducible.
* **X-axis uses real economic grid values** (e.g., wealth, housing) instead of indices, for interpretability.
* **Error plot features**: Zero reference line, error bars, statistics box (max/mean error), and improved styling for publication-quality output.
* **Docstrings for all metrics and plotting functions** updated to explain interpolation, grid handling, and scientific accuracy.
* **Example usage and configuration** included in docstrings for both plotting and L2 metrics.

### Changed
* **L2 error and plotting metrics** now always compare on a common grid, ensuring scientifically accurate, like-for-like comparisons regardless of discretization.
* **Improved error handling and warnings** for grid mismatches, shape incompatibilities, and extraction failures.
* **All changes are fully integrated with CircuitRunner** and its bundle management system, so plots and metrics are always associated with the correct parameter set.

### Fixed
* **Bugfixes to Euler error metric**: Improved threshold handling and interpolation logic to avoid NaN results and ensure robust error calculation for all methods.
* **Plotting function scope issues**: Fixed closure variable capture and array indexing errors in plotting configuration.
* **Value function extraction**: Now uses correct model attribute names (`vlu` instead of `v`) and solution types for robust extraction.

## [0.4.0] - 2025-06-16 – GPU-Accelerated VFI Solver

### Added
* **GPU-Accelerated VFI Solver (`VFI_HDGRID_GPU`)**
  - Implemented a new solver backend using Numba CUDA to offload the VFI dense grid search to NVIDIA GPUs.
  - The `vfi_gpu_kernel` performs the core computation in parallel across thousands of GPU threads.
  - The `solve_vfi_gpu` host function manages data transfers (CPU↔GPU) and kernel launches.
  - This provides a significant performance increase for high-density baseline calculations, enabling larger and more complex models to be solved within practical time limits.

* **Dynamic and Shared Memory on GPU**
  - The GPU kernel now uses **dynamic shared memory** to dramatically reduce slow global memory access, a key optimization for performance.
  - The launcher calculates the required shared memory size at runtime, allowing the kernel to handle variable-sized grids without hardcoded limits.

* **Unified Solver and Pre-compilation Workflow**
  - `solve_runner.py` is now the single entry point for all workflows (CPU, MPI, and GPU).
  - A new `--precompile` flag intelligently warms up the correct Numba cache (either CPU or GPU) based on the selected method.
  - The framework now automatically uses minimal grid settings during pre-compilation to prevent GPU out-of-memory errors.

* **Robust GPU-Compatible Helper Functions**
  - Created a GPU-safe `interp_gpu` function to perform linear interpolation, as `np.interp` is not supported in CUDA kernels.
  - Implemented a "dispatcher pattern" for utility functions, using an integer ID to select between pre-compiled, static GPU device functions (`u_func_gpu_crra`, etc.). This is the robust solution for handling different functional forms on the GPU.

### Fixed
* **GPU Compilation Errors**: Resolved a series of `TypingError` and `NameError` issues by:
  - Replacing unsupported function calls (`np.interp`, `cuda.lib.isinf`) with GPU-compatible equivalents (`interp_gpu`, `math.isinf`).
  - Correctly handling function namespaces (`math` vs. `np`) inside device code.
  - Eliminating the use of unsupported closures as kernel arguments.
* **GPU Out-of-Memory Errors**: Fixed `CudaAPIError: [700]` by ensuring the pre-compilation step uses a minimal memory footprint.

## [0.3.0dev4] - Planned – Hierarchical MPI Parameter Sweeps

### Planned Features
* **Hierarchical MPI parameter sweep architecture**
  - Two-level MPI communicators: `COMM_TOP` for parameter distribution, `COMM_SOLVER` for intra-node VFI computation
  - Enable scaling to large parameter spaces (e.g., 50 parameter combinations × 45 cores each = 2250 total cores)
  - Each node runs complete baseline+fast workflow for one parameter combination

* **Memory-efficient parameter processing**
  - Apply solve→plot→delete→gc pattern from `solve_runner.py` to parameter sweeps
  - Process each parameter combination sequentially to avoid memory accumulation
  - Immediate model cleanup after plotting and metric extraction

* **DynX Sampler integration for parameter sweeps**
  - Replace manual parameter grid construction with built-in `Cartesian` sampler
  - Canonical column ordering and robust parameter space handling
  - Support for both list (`PATH=v1,v2,v3`) and range (`PATH=min:max:N`) parameter specifications

* **Enhanced bundle management for parameter caching**
  - Hash-based bundle directories for each parameter combination
  - Automatic skip of completed parameter combinations
  - Robust restart capability for interrupted parameter sweeps
  - Method-aware bundle organization (VFI_HDGRID, FUES, CONSAV in separate subdirectories)

### Implementation Strategy
* **Phase 1**: Core architecture with hierarchical MPI and sampler integration
* **Phase 2**: Integration with proven solve_runner patterns and bundle management
* **Phase 3**: CLI enhancement and workflow optimization
* **Migration Path**: Create `param_sweep_v2.py` alongside existing implementation

### Expected Benefits
* **Performance**: 10-100x reduction in peak memory usage for large parameter sweeps
* **Scalability**: Linear scaling to hundreds of parameter combinations across multiple nodes
* **Robustness**: Automatic restart capability and bundle corruption recovery
* **Maintainability**: Code reuse from solve_runner and elimination of manual parameter bookkeeping

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

* **Configurable plotting comparison system**
  - New `plot_comparison_factory()` function in `helpers/metrics.py` creates configurable plotting metrics for comparing fast methods against baseline solutions.
  - Generates difference plots between policy/value functions of different solution methods (e.g., FUES vs VFI_HDGRID).
  - Configurable state-space slicing allows plotting specific indices of multi-dimensional arrays.
  - Supports both consumption policies (`c`) and value functions (`vlu`) with automatic detection of solution attributes.
  - Integrated with CircuitRunner's metric system for seamless workflow integration.
  - Uses existing `_extract_policy()` function for robust data extraction from complex model structures.
  - Memory-efficient design stores baseline model temporarily and cleans up automatically.

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
* **Plot comparison function scope issues** where parameter variables from factory function weren't properly captured in closure.
* **Array indexing errors** in plotting configuration by using 0-indexed bounds instead of array size.
* **Value function extraction** by using correct model attribute names (`vlu` instead of `v`) and solution types.
* **Euler error calculation thresholds** made more flexible and based on model's borrowing constraint to prevent NaN results for FUES methods.

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