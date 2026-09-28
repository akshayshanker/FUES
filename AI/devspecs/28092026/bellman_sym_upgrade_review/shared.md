# Shared interfaces and review evidence

28 September 2026. This supports the [pre-upgrade specification](../bellman_sym_upgrade.md). Findings describe the inspected local files; proposed implementation and acceptance checks have not been run.

## 1. What was reviewed

| Source | Observed state |
|---|---|
| FUES | Model sources reviewed at `6d7c3fe`; main advanced to `97bc3f5` during review, changing only a batch-submission script's description. Distribution `dcsmm` 0.6.0.dev8. `FUES2026` contains the actual `FUES2026/FUES` repository. |
| Kikku → asva | Sibling kikku checkout `a54619c`, version 0.2.0, is the declared source of the rename to `bellman-ddsl/packages/asva`. Asva succeeds kikku; it is not an unrelated numerical library. |
| Bellman | Main `9c332302c`; distribution metadata 0.1.0. Internal developer release 0.01 is a workstream name, not evidence that its interface is already installed. |
| In-progress FUES estimation | `feat/cd-estimation-postprocess`, worktree `FUES-wt-cd-estimation`, observed commit `7510659` plus changing uncommitted work. Its [specification](../cd_estimation_and_postprocess.md) is also being revised. Do not merge or overwrite that work as part of this review. |
| FUES interpreter | Python 3.12.7; NumPy 2.3.5; Numba 0.64.0; installed kikku 0.2.0, dolo 0.4.9.20, dolang 0.0.21. Neither bellman nor asva installed in this environment. |

Versions came from local git and installed metadata; model-source recheck at 02:00 AEST on 28 September found no changes to the inspected model code. No remote fetch or claim about unpublished remote changes. Do not replace the working FUES or Bellman environments while another task is using them. Record the final accepted Bellman/asva source revisions and full dependency lock when implementation starts; do not install an unrelated package merely because it is named `asva`.

The model reviews trace loading, numerical callbacks, backward passes, simulation, estimation and result readers. Existing result folders and paper tables were not changed. No full model solve, package suite, MPI run or GPU benchmark was performed.

## 2. Current implementation versus the intended release

Current [public exports](/Users/akshayshanker/Research/Repos/bellman-ddsl/packages/bellman/bellman/__init__.py) include `load_recipe`, `elaborate_trellis`, all three binding passes, `assemble_trellis`, `period_joins`, `inter_period_joins`, `forward_order` and `backward_order`. Use these public operations, with one stage version per occurrence. Numerical arrays remain FUES data, separate from the symbolic stage.

[Current assembly](/Users/akshayshanker/Research/Repos/bellman-ddsl/packages/bellman/bellman/factory/trellis.py:56) executes loading, elaboration, calibration, methodization, configuration. The migration must execute every required pass and follow the accepted release's public contract; it must not load a saved solved object instead. Where age-dependent inputs require explicit passes, bind each age's values before realizing its numerical operators.

I ran this read-only probe in Bellman's existing environment:

```python
from bellman import elaborate_stage, load_stage
import numba
s = elaborate_stage(load_stage("applications/portfolio-choice/cons.bl"))
s = s.calibrate({"ρ": 6.0})
p = s.builders.cntn_to_dcsn["Pullback"]
f, k = p.compile_kernel(), p.kernel()
print(hasattr(s, "build"), hasattr(k, "python"), f(3.0, 1.0))
numba.njit(f)(3.0, 1.0)
```

Observed: `False, False, 2.0`; the compiled Python callable has eight closure cells. Numba rejects an f-string in the current interpreter function (`UnsupportedBytecodeError`, [kernels.py:1008](/Users/akshayshanker/Research/Repos/bellman-ddsl/packages/bellman/bellman/stage/kernels.py:1008)). This supports waiting for lane E where generated kernels are used; it is not a defect report against an unimplemented promise. The probe succeeds as an ordinary Python evaluation.

The [kernel brief §§3–6](/Users/akshayshanker/Research/Repos/bellman-ddsl/AI/road-map/AI-todos/briefs-for-agents/internal-dev-0.01-release/kernels-as-functions.md:30) promises explicit trailing parameter arguments, stable source across calibrations and ordinary functions. It nests named utility functions inside the emitted kernel. It does not export every declared function separately, generate derivatives, solve roots, perform expectation or supply a general discrete-choice solver. Its `asva.egm` example establishes a consumption-stage calculation, not the multi-control durables adjuster or housing model.

**Recommendation:** keep existing handwritten Numba helpers and dictionary keys wherever those exports are absent. Bind their inputs from Bellman, and establish equation agreement explicitly. Compile complete explicit kernels where useful; do not create a second expression compiler inside FUES. A small callback adapter may reorder arguments, select a documented output or convert a stated derivative convention. It must not conceal extra discounting, utility terms or economic assumptions. Test the final low-level `asva.egm_1d.make_egm_1d` with FUES's keeper callbacks and a compiled consumer: upstream changes to explicit parameter passing can require an adapter even when the import name survives.

Record the symbolic-to-array mapping explicitly. In durables, `V_cntn` is the expected continuation value; `d_aV_cntn` and `d_hV_cntn` contain β times its asset and housing derivatives, including the transition factors already applied before expectation. Keep the existing arrays while writing mathematically accurate derivative declarations. Retirement's cash derivative and housing's `lambda_` also require the coordinate-specific mappings identified in their reviews. Never identify arrays as ordinary derivatives merely because their old variable names suggest that reading.

## 3. Execution interfaces the release removes

**Command line.** [Asva §4](/Users/akshayshanker/Research/Repos/bellman-ddsl/AI/road-map/AI-todos/briefs-for-agents/internal-dev-0.01-release/asva.md:35) removes `run/cli.py`. Its description refers partly to older behaviour: the live [parser](/Users/akshayshanker/Research/Repos/FUES2026/kikku/kikku/run/cli.py:460) is already a slot-based parser, with no dolo import. Its argument-order merge, `@file` support, Cartesian sweep axes, labels and `RunSpec` construction are reusable. Move this implementation to a proposed `examples/cli.py`, preserve attribution, and import the retained asva `RunSpec`, `TestSpec`, `SimSpec` and `make_test` after verifying their release locations. Retain command syntax; translate model/method payloads at the application boundary. `RunSpec.base_spec` is the real field; `model_dir` and `syntax_dir` are deprecated aliases ([types.py:30](/Users/akshayshanker/Research/Repos/FUES2026/kikku/kikku/run/types.py:30)).

**Simulation.** The existing [simulator](/Users/akshayshanker/Research/Repos/FUES2026/kikku/kikku/asva/simulate.py:138) iterates a NetworkX graph and reads `kind`, branch labels, successors and renames. Simply replacing its import cannot remove the old model infrastructure. Keep its particle slicing, scattering, pre-drawn shocks and stage callbacks in FUES; retain `StageForward`, `BranchingForward` and `draw_shocks` locally if absent from asva. Replace traversal by the period-specific ordering and joins Bellman returns. Construct it once per distinct period structure; verify a changed and terminal period. Generic asva simulation remains unavailable. This FUES-local work is necessary for durables estimation and need not wait for a universal simulator.

**Persistence.** [Old nest I/O](/Users/akshayshanker/Research/Repos/FUES2026/kikku/kikku/run/nest_io.py:32) serializes stage objects and one `inter_conn`; it cannot serve as new trellis persistence. The active durables specification already documents empty `.nst` files and replaces their use. Follow that work's final reconstruction/result contract. Housing also needs its legacy object readers replaced while preserving actual numerical arrays, period indices and table identifiers. Include model-source hashes and complete resolved inputs in new result metadata so an old cached result cannot be mistaken for a migrated one.

## 4. What must be settled before each example is accepted

The default is to preserve executed FUES behaviour and record existing discrepancies as separate corrections. Successful dependency migration does not automatically establish economic correctness. The following issues restrict claims of economic agreement, but an unchanged, named existing defect need not force unrelated numerical corrections into the dependency-change commit. New inability to express or execute a previously supported configuration remains a migration blocker.

| Example | Required resolution before claiming economic agreement |
|---|---|
| Retirement | Decide retiree asset/cash coordinates and terminal derivative convention; preserve the effective terminal utility and one work-cost deduction. Replace the independent validation calculation that evaluates next policy at current assets. |
| Durables | Explicitly record discounted marginals, log-income coordinates, positive borrowing floor and both utility families. Resolve/document utility-shift and terminal-age discrepancies. Check the Cobb–Douglas Euler diagnostic using next period's chosen housing. |
| Housing | Confirm a representable income matrix/finite law, branch connections and terminal closure. Preserve present-bias quantities and effective tax schedules. Settle CPU/GPU utility disagreement before claiming cross-device agreement. |

Housing's existing Bellman application has incomplete matrix calibration and only parsing evidence. The internal release alone does **not** establish a complete housing replacement. First construct a minimal two-state, owner/renter case. If the public factory refuses the needed composition, record the exact refusal and add the narrowly required upstream prerequisite; do not recreate DynX to bypass it. The owner-only recursive mode and each hardware mode need an explicit retained/deferred entry. The overall three-example upgrade is incomplete while a requested supported mode remains deferred.

## 5. Numerical comparisons and performance

Recommended initial tolerances, to be recorded before examining migrated results: for unchanged float64 scalar formulas, `rtol=1e-10, atol=1e-12`; for a preserved whole calculation, maximum scaled value difference `max(abs(new-old)/max(1,abs(old))) <= 1e-8` over valid nodes, with identical finite/infeasible masks. A failure requires an explained formula, arithmetic or algorithm difference; never widen tolerance merely to pass. Handle sentinels separately. Preserve exact tie rules in old/new migration checks: durables chooses adjustment at equality and retirement chooses retirement. Any near-tie arithmetic change needs its own explanation and simulation comparison. For an independent optimizer, compare objective values and policy loss at ties and discontinuities; refine that optimizer before treating it as a reference.

Check raw candidate arrays **before** upper-envelope refinement, selected branches, constrained points and final interpolation separately. An Euler equality is required only in the interior; test feasibility and the appropriate one-sided inequality at a bound. A root of a first-order condition is not proof of a global maximum. The retained adjuster root search is consistent with the distinct inverse-Euler construction described in Dobrescu and Shanker (2024), §6.2 ([primary source text](/Users/akshayshanker/Research/Repos/lit-kb/sources/text/own-work/dobrescu-shanker-2024-inverse-euler-wp-2024-12-07.md:453)); that section explicitly needs root finding when the active state dimension is smaller than the post-state dimension. This is no accuracy certificate for the present implementation.

Measure at least five warmed repeats with the same thread count, grid, horizon, envelope method and parameter values. Compare medians against the legacy run; record construction and first compilation separately, including peak resident memory. The main specification's 10% investigation threshold is a proposed migration criterion, not a measured performance claim. Measure repeated calibration changes as well as repeated identical solves; changing an estimand must change the relevant outputs without reusing stale closures.

For reproducibility, one acceptance notebook per example should show public model construction and the comparisons, with space for observations. Acceptance tests must be tracked: FUES ignores `test_*.py`. These review documents themselves live under the repository's ignored `AI/` tree and are local working documents until deliberately added. No files were staged or committed by this task.
