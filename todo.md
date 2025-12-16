# TODO List for FUES Project

## Working Log

### 2025-12-16: EGM Loop Memory and Compute Optimizations

**Summary:** Reduced memory footprint and compute overhead in `horses_c.py` EGM/VFI loops.

**Changes Made:**
1. **Removed `policy_a` storage** - Asset policy was stored but never used downstream
   - Removed from `_solve_egm_loop`: allocation, CONSAV branch assignment, return tuple
   - Removed from `_solve_vfi_loop`: discarded with `_`, removed from return
   - Removed from operator functions: unpacking and `sol.policy["a"]` assignments
   - **Savings: ~1.1GB per model**

2. **Conditional `vlu_dcsn` allocation for `delta == 1`**
   - When `delta == 1` (standard discounting): `vlu_dcsn = Q_dcsn` mathematically
   - Skip allocating separate array; return `Q_dcsn` directly
   - **Savings: ~55MB per model for standard discounting**

3. **Skip expensive computations when `delta == 1`**
   - `compute_gradient()` and `u_func()` calls unnecessary when delta=1
   - Simplified to: `lambda_dcsn = uc_today`, `vlu_dcsn = Q_dcsn`
   - **Savings: ~34,000 function calls per model**

4. **EGM grids return `None` instead of empty dicts**
   - When `store_egm_grids=False`, return `None` not `{}`
   - Added downstream `is not None` checks

**Files Modified:** `src/dc_smm/models/housing_renting/horses_c.py`, `CHANGELOG.md`

---

## High Priority

### 📊 NEXT SESSION: Paper Comparison Plots
- [ ] **Create joint comparison plots for paper**
  - **Plot 1: Consumption Error Comparison**
    - Compare consumption policy error of each method (FUES, DCEGM, CONSAV, VFI) vs VFI_HDGRID_GPU baseline
    - Show deviation/error across the state space
  - **Plot 2: Consumption Function Overlay**
    - Plot consumption functions from all methods on same axes
    - Clean academic styling (publication-ready)
  - **Requirements:**
    - Clean, professional academic style (suitable for journal submission)
    - Consistent color scheme across methods
    - Clear legends, axis labels, proper fonts
    - Consider: subplots, insets for detail regions, log-scale for errors
  - **Files:** `examples/housing_renting/helpers/plots.py` or new `paper_figures.py`
  - **Data source:** VFI GPU baseline from `test_0.1-paper-sweep-4`
  - Status: NEXT SESSION

### ⚠️ CRITICAL: Image Files Being Deleted
- [ ] **Debug why image files are disappearing**
  - Problem: Image files are being deleted despite multiple fixes
  - Attempted: Timestamps, fixed directory structure, checked for deletion code
  - Still happening: Files disappear after being created
  - Need to investigate: matplotlib issues, file system issues, external deletion
  - File: `examples/housing_renting/helpers/plots.py`
  - Status: UNRESOLVED as of 2025-11-08

### ⚠️ CONSIDER: Ref Model Grid Size Mismatch in Metrics
- [ ] **Investigate ref model loading - grid vs policy size mismatch**
  - Problem: With current ref model loading scheme, the ref model grid is the same size as the grid in master.yml, yet the policy function has 20000 points
  - This suggests a mismatch between grid construction and policy loading
  - Could be in: ref model construction, bundle loading, or grid extraction in metrics
  - Files to check: `helpers/metrics.py` (managed_model_load, make_policy_dev_metric), `solve_runner.py` (ref model construction)
  - Status: NEEDS INVESTIGATION as of 2025-12-12

### Memory Management & CSV Export
- [ ] **Fix EGM data clearing in low-memory mode**
  - Problem: `cleanup_model()` in `solve_runner.py` clears EGM data when `--low-memory` flag is used, preventing CSV export
  - Solution: Modify `cleanup_model()` to preserve EGM data when `--csv-export` flag is also present
  - File: `examples/housing_renting/solve_runner.py` lines 307-311
  - Workaround: Currently must run without `--low-memory` to get EGM CSV exports

### Performance Optimizations
- [ ] **Implement selective memory cleanup options**
  - Add flags to control what gets cleared (Q, lambda, EGM, etc.)
  - Allow fine-grained control over memory/functionality trade-offs
  - Consider memory profiles for different use cases (cluster vs local, plotting vs metrics)

## Medium Priority

### FUES Algorithm
- [ ] **Make FUES constants configurable as numerical options**
  - Currently hardcoded in `src/dc_smm/fues/fues.py` (lines 17-25):
    - `EPS_D = 1e-14` - Machine epsilon threshold
    - `EPS_SEP = 1e-10` - Separation tolerance for intersections
    - `EPS_fwd_back = 0.5` - Forward/backward scan proximity threshold
    - `PARALLEL_GUARD = 1e-10` - Parallel line detection threshold
  - Already passable via FUES() function args (default to None, falls back to constants)
  - Consider exposing via EGM_UE and model configs for easier tuning

### Configuration Consolidation
- [ ] **Move Euler error settings to main config YAML**
  - Currently hardcoded bounds and sample sizes in `helpers/euler_error.py`
  - Settings to expose: sample_size, grid bounds, tolerance thresholds
  - Should be configurable in `master.yml` under a `metrics` or `diagnostics` section
  - Allows experiment-specific error computation settings without code changes

- [ ] **Make dev_c_log10_mean w_max filter configurable**
  - Currently hardcoded `W_MAX_FILTER = 35.0` in `helpers/metrics.py` (dev_c_log10_mean)
  - Filters consumption comparison to w < 35 to avoid VFI grid boundary extrapolation issues
  - Options:
    1. Make it a YAML-configurable setting (e.g., `metrics.w_max_filter`)
    2. Fix VFI grid extrapolation so filtering isn't needed
  - Related: VFI grid upper bound may need adjustment for proper extrapolation

### Code Quality
- [ ] **Simplify EGM grid attachment in horses_c.py**
  - Current implementation has TODO comment about simplification (lines 138-157)
  - Consider more elegant data structure for EGM grids

### Documentation
- [ ] **Document memory management trade-offs**
  - Create guide for when to use `--low-memory` vs full memory mode
  - Document impact on various features (CSV export, plotting, metrics)

## Low Priority

### Testing
- [ ] **Add tests for CSV export functionality**
  - Ensure EGM data is properly exported in all configurations
  - Test with and without low-memory mode

### Future Enhancements
- [ ] **Implement streaming CSV export**
  - Export data as it's generated rather than at the end
  - Would work better with aggressive memory cleanup

## Completed
- [x] Refactor `_egm_preprocess_core` to conditionally add jump constraints (18/08/2025)
- [x] Diagnose EGM CSV export issue with `--low-memory` flag (18/08/2025)
- [x] Implement FOC checks in `_egm_preprocess_core` to filter constraint points based on economic optimality (08/11/2025)
- [x] Modify image saving to always use timestamped directories to preserve all previous runs (08/11/2025)
- [x] Optimize `_egm_preprocess_core` performance with vectorized FOC checks (~3-5x speedup) (08/11/2025)
- [x] Make KT conditions override configurable via `override_KT_conditions_at_jump` in master.yml (08/11/2025)