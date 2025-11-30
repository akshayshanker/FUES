# TODO List for FUES Project

## High Priority

### ⚠️ CRITICAL: Image Files Being Deleted
- [ ] **Debug why image files are disappearing**
  - Problem: Image files are being deleted despite multiple fixes
  - Attempted: Timestamps, fixed directory structure, checked for deletion code
  - Still happening: Files disappear after being created
  - Need to investigate: matplotlib issues, file system issues, external deletion
  - File: `examples/housing_renting/helpers/plots.py`
  - Status: UNRESOLVED as of 2025-11-08

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