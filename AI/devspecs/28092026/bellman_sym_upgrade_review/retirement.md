# Retirement migration review

Reviewed 28 September 2026 against FUES `6d7c3fe` and the current Bellman checkout (`9c332302c` at the final inspection), including the proposed internal development release. This is a source review; no numerical solves were run. The recommendation is to replace construction and binding while preserving FUES numerical routines and outputs. Bellman's existing `applications/retirement-choice` is a useful syntax reference, but has substantive differences from the FUES implementation.

## Preserve and change

| Area | Preserve | Change |
|---|---|---|
| [solve.py](../../../../examples/retirement/solve.py) | `solve_nest` arguments and four-result tuple; backward loop, continuation routing, result dictionaries | Replace dolo factories, private method normalization and graph conversion (lines 24–27, 63–66, 138–139, 314–374) with the released Bellman construction sequence |
| [model.py](../../../../examples/retirement/model.py) | `RetirementModel` attributes, linear asset grid, numerical helper contracts | Replace `.calibration` access with parameter records and numeric `.value`; translate configured settings at construction (149–192) |
| [operators.py](../../../../examples/retirement/solvers/operators.py) | EGM, upper-envelope selection, interpolation, branch comparison and timing | Replace kikku imports (16–17); adapt exported kernel arguments and outputs inside existing factories |
| [syntax](../../../../examples/retirement/syntax) | Occurrence names `work_cons`, `retire_cons`, `labour_mkt_decision`; experiment options | Replace stage YAML with FUES-owned `.bl`; update recipe, trellis and method targets |
| [run.py](../../../../examples/retirement/run.py), [benchmark.py](../../../../examples/retirement/benchmark.py) | Command-line options, sweeps, method ordering, tables and plots | Use the shared FUES command-line replacement; update method-record decoding, including `benchmark.py:114–126` |

The core `src/dcsmm` numerical algorithms need no retirement-specific rewrite.

## Callable and result contracts

The existing EGM factory takes four callbacks: inverse Euler `(dv, fixed_state)`, value `(c, v, fixed_state)`, endogenous grid `(c, savings, fixed_state)`, and curvature `(c, ddv, fixed_state)`. Its result is `(c_hat, v_hat, x_hat, da_hat)`; see [operators.py](../../../../examples/retirement/solvers/operators.py), lines 45–58 and 109–122. These signatures should remain stable.

[Lane E](</Users/akshayshanker/Research/Repos/bellman-ddsl/AI/road-map/AI-todos/briefs-for-agents/internal-dev-0.01-release/kernels-as-functions.md>), lines 51–68, proposes explicit constituent functions with their printed arguments followed by positional parameters. It does not supply maximization, interpolation or differentiation. Named utilities are inner functions of generated kernels, not separately promised exports. Current [kernel queries](</Users/akshayshanker/Research/Repos/bellman-ddsl/packages/bellman/bellman/stage/kernels.py>), lines 64–76, target primitive boxes. Therefore adapt supported inverse-Euler, endogenous-grid and complete value/transition kernels, but retain `u`, `du`, `uc_inv`, `ddu` and curvature helper bodies where no public export exists. Check their equations against the source; do not extract inner functions or invent private export APIs. The existing [helpers](../../../../examples/retirement/model.py), lines 17–33, 59–61 and 87–89, already provide the required Numba functions.

Preserve `{"solutions": [...]}` and each period's `t`, `h`, stage dictionaries, raw worker EGM arrays and timings ([solve.py](../../../../examples/retirement/solve.py), 193–236). [Postprocessing](../../../../examples/retirement/postprocess/diagnostics.py), 6–46, reconstructs calendar order using `t` and excludes the first three backward steps from mean timings. Reusing `model`, `stage_ops` and `waves` bypasses reconstruction (`solve.py:335`); new parameter or grid overrides must not reuse stale bindings silently.

## Pre-existing discrepancies exposed by migration

1. **Terminal continuation.** FUES supplies logarithmic continuation utility, with its derivatives, after the last solved period ([solve.py](../../../../examples/retirement/solve.py), 152–161). Bellman's [bequest](</Users/akshayshanker/Research/Repos/bellman-ddsl/applications/retirement-choice/stages/bequest.bl>), 3–17, instead represents zero terminal continuation and marginal value. Preserve FUES terminal data for the migration baseline; adopting the Bellman example unchanged changes the horizon's economic meaning.

2. **Work cost.** The numerical worker value already subtracts `delta` ([model.py](../../../../examples/retirement/model.py), 51–52; [operators.py](../../../../examples/retirement/solvers/operators.py), 127). Old [choice YAML](../../../../examples/retirement/syntax/stages/labour_mkt_decision/labour_mkt_decision.yaml), 42–45, and Bellman's [choice stage](</Users/akshayshanker/Research/Repos/bellman-ddsl/applications/retirement-choice/stages/ret_choice.bl>), 14–25, place the cost in branch comparison. Charge it exactly once and preserve the stored worker-value convention for downstream readers.

3. **Derivative coordinates.** FUES stores `dv = du(c)` and inserts `R` into inverse Euler (`operators.py:202`, `model.py:47–48`). Bellman's [worker stage](</Users/akshayshanker/Research/Repos/bellman-ddsl/applications/retirement-choice/stages/worker_cons.bl>), 27–35, supplies `R*du(c)` at arrival and uses only `beta` in inverse Euler. Combining these conventions multiplies the return twice. Translate deliberately; second derivatives require the same audit.

4. **Retiree grid coordinates.** FUES forms `(c + savings)/R`, an arrival-asset coordinate (`model.py:83–84`), then interpolates it at `R*arrival_grid` (`operators.py:64–69`, `model.py:125–128`). Bellman's [retiree stage](</Users/akshayshanker/Research/Repos/bellman-ddsl/applications/retirement-choice/stages/retiree_cons.bl>), 13–18 and 33–35, instead forms decision cash `savings + c`. The constrained FUES branch also sets consumption to the asset grid (`operators.py:69`). Resolve these coordinate discrepancies independently; do not conceal a numerical correction inside generated-function adoption.

5. **Actual grids and choice.** FUES uses `linspace(b, grid_max_A, grid_size)` (`model.py:156–162`), rather than the `n_a/a_min/a_max` names in its [methods file](../../../../examples/retirement/syntax/stages/work_cons/work_cons_methods.yml), 34–40. Map existing options explicitly. Hard-choice ties select retirement; positive `smooth_sigma` computes weighted values, not a log-sum expected maximum (`operators.py:189–203`). Preserve absorbing retirement through the two routes in `solve.py:165–172`; do not infer taste-shock semantics from smoothing.

## Validation and simulation scope

Record a small baseline for all four envelope methods, including every stage's values, consumption, derivatives, raw candidates and grid endpoints. Check generated callbacks against manual functions at non-unit returns and changed calibration; test terminal data, budgets, lower bounds, ties, absorbing retirement and unchanged output ordering. Keep numerical corrections separately identified.

The tracked [retirement test](../../../../tests/test_retirement.py), 25–92, is insufficient as mathematical validation. Its [Euler diagnostic](../../../../examples/retirement/postprocess/diagnostics.py), 96–103, computes next assets but interpolates next consumption at current assets, and assumes wage income for the selected work/retirement policy. Add a branch-aware independent diagnostic.

The local [simulation test](../../../../tests/test_simulate_retirement.py), 6–10, depends on `kikku.dynx` and its simulator, but is ignored/untracked under [.gitignore](../../../../.gitignore), 67. Production `run.py:220–226,303–313` advertises simulation yet executes sweeps. The proposed asva simulator is a stub: do not claim production retirement simulation support from a package rename. If simulation is accepted scope, retain or implement a small FUES-local simulator and explicitly track its tests.
