# TODO List for FUES Project


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
- [ ] **Omit `del_a` from FUES return tuple when `del_a` is not supplied**
  - Currently FUES always returns 5 arrays `(e, v, p1, p2, d)` even when `del_a=None` (zeros passthrough)
  - When `del_a` is not provided, return 4 arrays `(e, v, p1, p2)` to keep the interface clean
  - Requires updating all callers that unpack 5 return values
  - File: `src/dcsmm/fues/fues_v0_2dev.py`

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

### Notebook
- [ ] **Update Plotly interactive plot in retirement notebook to handle different grid sizes from overrides**
  - Currently the interactive EGM plot (`nb_plot_egm_interactive`) may not handle grid size changes from `config_overrides`
  - Should adapt axis ranges and data selection when grid size differs from the default
  - File: `examples/retirement/notebooks/retirement_fues.ipynb`, `examples/retirement/outputs/plots.py`

### README
- [ ] **Add PBS runtime comparison plot to README under the intro**
  - Use the PBS cluster scaling plot (Gadi timings) showing FUES vs MSS vs RFC vs LTM
  - Place it right after the intro paragraph, before "Install"
  - Need to generate/commit a static PNG from the PBS results first

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

## Next: Generic `dcsn_to_arvl_mover` in `kikku.asva`

**Context:** `make_egm_1d` in `kikku.asva.egm_1d` already genericizes the `cntn_to_dcsn_mover` (EGM inversion step). The next step is a matching `make_dcsn_to_arvl_1d` that genericizes the interpolation-to-arrival step.

**FUES belongs in `cntn_to_dcsn_mover`** (cleans non-monotone endogenous grid before interpolation). The `dcsn_to_arvl_mover` receives clean, monotone decision-grid quantities.

**Required callables (EGM recipe for `dcsn_to_arvl_mover`):**

| Callable | Signature | Role |
|----------|-----------|------|
| `arrival_transition` | `(exo_grid, fixed_state, params) → arvl_grid` | Compute arrival grid from poststate grid (e.g. `w = R*a + y` for worker, identity for retiree) |
| `constrained_fallback` | `(arvl_pt, v_cntn_floor, fixed_state, params) → (c, v, da)` | Values when borrowing constraint binds |
| `marginal_bellman` | `(c_i, da_i, fixed_state, params) → (dv, ddv)` | Envelope theorem at arrival: `dV[<] = u'(c)`, `ddV = u''(c)*(R-da)` |
| `interp_scheme` | interpolation callable | Numerical interpolation (e.g. `interp_as_3`) |

**Factory signature:**
```python
def make_dcsn_to_arvl_1d(arrival_transition, constrained_fallback,
                          marginal_bellman, interp_scheme, params):
    # returns @njit dcsn_to_arvl_step(endog_grid, c_hat, v_hat, da_hat,
    #                                  exo_grid, v_cntn_floor, fixed_state)
```

**Full stage composition after this:**
```
cntn_to_dcsn_mover:  make_egm_1d(recipe) or make_egm(recipe, K, n_c) [+ FUES if non-convex]
dcsn_to_arvl_mover:  make_dcsn_to_arvl_1d(recipe)
```

**Also fix:** Standardize on 3-output interpolation (`interp_as_3`) for both worker and retiree. Worker currently uses `interp_as_2`.

**Ref:** Matsya consultation on dcsn_to_arvl_mover (15 Mar 2026 session).

## Completed
- [x] Add `make_egm` K-dimensional pure EGM factory to `kikku.asva.egm_1d` (20/03/2026)
- [x] Refactor `_egm_preprocess_core` to conditionally add jump constraints (18/08/2025)
- [x] Diagnose EGM CSV export issue with `--low-memory` flag (18/08/2025)
- [x] Implement FOC checks in `_egm_preprocess_core` to filter constraint points based on economic optimality (08/11/2025)
- [x] Modify image saving to always use timestamped directories to preserve all previous runs (08/11/2025)
- [x] Optimize `_egm_preprocess_core` performance with vectorized FOC checks (~3-5x speedup) (08/11/2025)
- [x] Make KT conditions override configurable via `override_KT_conditions_at_jump` in master.yml (08/11/2025)


## Catch all inbox
- [] 




#### March 2026 sprint todo
- [ ] Retirement model variable rename
    - Rename state variable names in the retirement model code to match the syntax YAML, so the consumed function and EGM grid are interpreted as `m_cntn` etc.
- [ ] `prange` needs to be an option in solve (can turn off in PBS)
- [ ] Make FUES settings part of settings YAML
    - [ ] Separate settings per stage — does this have a devspec?
- [ ] Consolidate environments
    - Devspec: `AI/prompts/28032026/venv_and_setup_refactor.md`
- [-] Periodic MPI restart to reset RSS for memory management* (testing phase)
    - Devspec: `AI/devspecs/28032026/periodic_restart.md`
- [ ] Automatic rootfinding at arrival of successor tenure stage
    - Devspec: `AI/devspecs/28032026/prescan_roots_arvl.md`
- [ ] Modularise estimation, solution, and sweeps (including sweeps across estimations)
    - Initial devspec: `AI/devspecs/28032026/estimation_sweep_runner.md`
    - Needs higher-level thinking at the DDSL level — how does this relate to blocks, inputs, and outputs?
    - But before higher-level thinking, just do a simple batch run?
- [ ] Basic demo notebook for Cobb-Douglas and Seperable environemnts. (need to run CD with lots of w points, see if jumps still there then edit final.)
## Estimation: recorded on 28 Sep 2026, not fixed on the feat/cd-estimation-postprocess branch

- [ ] **Self-generated data solve ignores `--methods-override`** (`estimate.py`
  data-side `solve()` has no `method_switch`): the NEGM recovery sweeps
  generated data with FUES and estimated with NEGM. Owner's decision: the
  data side should use the trial's method. Deferred with self-generated runs.
- [ ] **Self-generated runs with types**: `beta_bar`/`sigma_beta` truths from
  a `theta_true` block, expanded before the data solves. Deferred; the
  driver refuses `types` + `data_source: selfgen` for now.
- [ ] **kikku `_age_group_masks` uses `range(t_lo, t_hi)`**: four of the five
  years of each data bin. Every estimate so far used it; changing it changes
  every result. Decide, then change kikku and re-estimate.
- [ ] **`fit_table.csv` provenance and weights**: kikku's last evaluation on
  rank 0, unweighted contributions. The lifecycle tool's
  `fit_table_at_best.csv` is the weighted fit at `theta_best`; consider
  having the driver write it directly (needs a re-solve at `theta_best` on
  the final segment).
- [ ] **Local serial runs with `mpi4py` installed** get a one-rank
  communicator and `n_samples = 1` unless `--n-samples` is passed; consider
  treating a size-1 communicator as serial in the driver.
- [ ] **`AI/CLAUDE.md`** still describes kikku as the editable local checkout
  at v0.2.0; the `.venv` and `pyproject.toml` now pin `a54619c` from the
  remote.

## Estimation output integrity (found 22 Aug 2026)

- [ ] **Verify `save_nest` on Gadi before the next estimation campaign.**
  In the current local checkout, `save_nest` fails: pickling
  `nest["periods"]` raises `TypeError: cannot pickle 'mappingproxy'`
  (from the dolo stage objects, whose symbol tables are mappingproxy
  views), and `estimate.py` catches the exception and only prints a
  warning. If the same holds on Gadi with the current dolo, estimation
  runs are silently NOT writing `best.nst` / `true.nst` at all.
  - Check a recent run dir under `/g/data/tp66/results/durables/estimation/`
    for `best.nst` (sshfs mount was inactive when this was found).
  - Fix candidates: convert mappingproxy to dict in the nest saver, or
    strip stage objects before pickling (they are rebuildable from the
    registry; the solution arrays are what matter).
  - Found by the durables-docs consistency agent (session 22 Aug 2026);
    the docs (`examples/durables/docs/estimation_outputs.md`) describe
    the intended contract, not verified current behaviour.

## Docs: harmonize the FUES complexity claim (found 22 Aug 2026)

- [ ] The docs state three different complexity claims for FUES:
  `O(n^{1/2})` (docs/index.md and docs/examples/index.md), "a single
  O(n log n) pass" (docs/examples/continuous_housing_model.md), and
  worst-case `O(N)` (docs/api/fues.md — this one is correct as stated).
  The paper (AI/paper/paper.tex:455) claims only that FUES "scales
  sub-linearly in grid size" empirically (envelope step 0.25/0.37/0.82 ms
  at 1k/3k/10k grid points), vs MSS linear and LTM quadratic (formal
  O(K^2) footnote at paper.tex:1208); it never asserts O(n^{1/2}).
  Proposed harmonized wording for the three non-API pages:
  "The scan is a single pass over the (sorted) endogenous grid —
  worst-case linear in grid size, and sub-linear in practice because
  secants are computed only from points on the envelope; MSS scales
  linearly and LTM quadratically."
  Related (same session, examples-pages verification agent): the
  benchmark tables in docs/examples/retirement_choice_model.md do not
  match the current paper draft's numbers (paper.tex:449), and the
  committed rerun paper-results/retirement/2026-03-31/002 diverges from
  the paper's LTM 10k timing (~798 ms vs ~1,360 ms) — reconcile when the
  paper tables are next regenerated.
