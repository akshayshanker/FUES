# FUES upgrade to Bellman-SYM

28 September 2026. **Pre-upgrade development specification; no implementation performed.**

## Purpose and scope

Replace dolo-plus/dolang/DynX model construction with Bellman-SYM, and repoint kikku's retained numerical and runner code to its renamed successor, **asva**. Cover retirement, durables and housing/renting. Preserve numerical algorithms, callable signatures, grids, result arrays, command lines and estimation behaviour. A package migration must not silently change the economic model.

The recommended design is **new model construction with the existing numerical routines**. Bellman owns declarations, within-period connections, period occurrences and parameter/method/settings binding. FUES owns numerical solution, simulation and its application-facing command line. Asva supplies the retained numerical factories and runner utilities. A *trellis* is Bellman's collection of period occurrences and their connections; a *kernel* is an explicit numerical function within a stage operation.

Reviewed local revisions and observations are in [shared interfaces and evidence](bellman_sym_upgrade_review/shared.md). Detailed findings: [retirement](bellman_sym_upgrade_review/retirement.md), [durables](bellman_sym_upgrade_review/durables.md), [housing/renting](bellman_sym_upgrade_review/housing_renting.md). The independent durables estimation work remains governed by [its current specification](cd_estimation_and_postprocess.md); rebase this migration on its final merged interfaces.

## Required upstream state

Target the completed [internal developer release plan](/Users/akshayshanker/Research/Repos/bellman-ddsl/AI/road-map/AI-todos/briefs-for-agents/internal-dev-0.01-release/implementation-plan.md), especially:

| Brief | Contract required by FUES |
|---|---|
| [Asva §§1–6](/Users/akshayshanker/Research/Repos/bellman-ddsl/AI/road-map/AI-todos/briefs-for-agents/internal-dev-0.01-release/asva.md) | Numerical factories and runner utilities retained; `dynx`, `run.cli` and `run.nest_io` removed. The generic simulator is deferred. |
| [Stage access](/Users/akshayshanker/Research/Repos/bellman-ddsl/AI/road-map/AI-todos/briefs-for-agents/internal-dev-0.01-release/stage-builder-names.md) | `stage.build.sol.{arvl,dcsn,policy}`, `stage.build.sim.{arvl,dcsn}`, `stage.par[name].value`, `stage.settings[name].value`; method records accessed by target. |
| [Kernels §§2–6](/Users/akshayshanker/Research/Repos/bellman-ddsl/AI/road-map/AI-todos/briefs-for-agents/internal-dev-0.01-release/kernels-as-functions.md) | Explicit arguments followed by parameters; stage-free `builder.kernels()` records; inspectable Numba-compatible functions. No automatic optimization, integration or differentiation. |
| Policy-operation and constraint briefs | Final method targets and the symbolic interpretation of inverse-Euler candidates and binding constraints. They do not implement FUES's constrained root search. |

Verify these contracts against the **merged source**, recording commit IDs. The present checkout does not expose the new stage names or kernel source field. Do not introduce compatibility aliases into Bellman or change its ongoing release brief.

## Implementation

### 1. Record the existing results

Keep the legacy environment available. Record deterministic runs, revisions, dependencies, resolved calibration/settings, grids, seeds, terminal inputs, stage arrays and raw candidates before changing imports. For durables, the reference must follow the estimation branch's final merge and precede this upgrade. Cover retirement's supported envelope methods, both durables registries and housing's CPU variants; record GPU results separately.

Keep numerical corrections separate from the dependency change. The reviews identify existing discrepancies; an unchanged result establishes preservation, not economic correctness. Unresolved coordinate or terminal inconsistencies block claims of source/numerical agreement for the affected mode.

### 2. Author FUES-owned model sources

Keep each example's existing `syntax/` location and occurrence names. Replace stage YAML with `.bl`; update outer period, trellis, recipe, calibration and methods files. Copy Bellman's application files only as starting material: they are not equivalent FUES specifications.

| Example | Required declarations and preservation |
|---|---|
| Retirement | Assets, consumption, work/retire choice and absorbing retirement; existing terminal utility, one work-cost deduction, asset/cash coordinates, marginal scaling and tie rule. Positive smoothing requires its actual probability-weighted objective. |
| Durables | Separable **and** Cobb–Douglas preferences; liquid assets, housing and log-income state; keeper/adjuster choice; borrowing floor, wage/retirement schedule, utility shifts, discounted marginal convention and terminal utility. Preserve adjuster roots and constrained candidates. |
| Housing/renting | All five stages, finite housing choices, income transition matrix, taxes including endpoint conventions, terminal data, and separate present-biased choice value and lifetime value. Preserve configuration variants. |

Classify economic constraints and finite state spaces as model declarations; keep grid resolution, root tolerances and envelope settings numerical. An existing CLI option can retain its spelling while its destination changes. Reject unknown overrides; application-only settings stay outside stage declarations.

### 3. Replace construction and access

Replace old factory imports in `solve.py` and housing's construction entry points with Bellman's public loading, elaboration, methodization, calibration and configuration operations. Use `assemble_trellis` where its full binding sequence and age schedule suffice; otherwise make all passes explicit. Apply age-dependent calibration to individual occurrences; keep terminal-first storage and chronological age mappings explicit.

Read joins and forward/backward order through Bellman's public factory operations. Retain FUES's backward loops and numerical result dictionaries outside symbolic stages. Do not recreate `SymbolicModel`, `Mover` or a second generic stage hierarchy. Small local records for actual numerical grids/results are permitted where housing currently needs them.

Resolve parameter/settings values once when constructing numerical operators, combining their actual owning stages and registry-level inputs. Retain handwritten Numba utility, derivative, curvature and terminal helpers: the release does not promise standalone exports for all of them. Check each against the `.bl` equations. Where a complete explicit kernel replaces a callback cleanly, adapt its argument/output order once and preserve the caller's signature. Never recover utility by cancelling terms in a larger kernel or access private symbolic internals.

Keep symbolic derivatives distinct from scaled numerical arrays: durables' continuation `d_aV/d_hV` include β, while ordinary V does not. Document each conversion and apply β once. Check actual FUES callbacks with the final asva factories and a Numba consumer; adapt changed factory signatures narrowly.

Never reuse calibrated closures after changing inputs. Generated source itself is calibration-independent; bound operators depend on calibration, methods and grids. In-memory functions cannot assume Numba's `cache=True`; retained module functions remain the default.

### 4. Preserve execution and stored results

Move the live kikku slot-based CLI parser into one FUES-owned module, importing retained record types from asva; preserve flags, argument precedence, sweep labels and PBS commands. Translate old method selectors once into final Bellman method-target records. Keep `RunSpec.base_spec`, not deprecated aliases.

Retain durables' particle transitions and outputs, plus local `StageForward`, `BranchingForward` callable records and `draw_shocks` if asva no longer exports them. Replace graph traversal with FUES code reading actual period occurrences and joins. Asva's planned simulator cannot supply this. Retirement's advertised simulation flag has no production implementation; do not claim one by renaming imports.

For housing, replace `Solution` construction and DynX object reads with explicit numerical inputs/results, retaining array axes and CPU/GPU numerical bodies. Preserve runner options and postprocessing; do not require wholesale folder reorganization.

Use the final durables estimation work's result format and reconstruction procedure. Do not pickle Bellman stages or revive `.nst` persistence. Preserve solution-array layouts and existing readable tables; old proprietary objects, if conversion is needed, are read only in the isolated legacy environment.

### 5. Complete each example, then remove dependencies

Order: shared CLI/construction support → retirement → durables → housing CPU → housing GPU verification. Keep `src/dcsmm/fues`, upper-envelope engines, interpolation and root-search algorithms unchanged. Repoint retained kikku numerical/runner imports to asva. Replace housing's DynX imports, including those under `src/dcsmm/models/housing_renting`.

Remove legacy installation lines from `pyproject.toml`, `setup/setup.sh` and documentation only after the affected example passes. Keep Bellman/asva in the examples dependency tier. Validate installation on the target Python/NumPy/Numba combination without changing a shared active environment.

## Acceptance: six items

1. **Dependency removal:** a fresh examples environment without dolo, dolang, kikku or DynX imports all three examples; core FUES still installs independently. CLI/PBS argument parsing and package tests pass. Deliberately track new tests despite the repository's blanket ignore pattern.
2. **Model account:** inspect one executed notebook per example showing construction/binding, timing, constraints, terminal inputs and branches. Classify each discrepancy as resolved, separately tracked pre-existing behaviour, or a migration blocker. Preserved old behaviour can pass dependency-migration acceptance; uncorrected discrepancies cannot support claims of economic agreement.
3. **Numerical interfaces:** scalar/array and Numba checks preserve callback signatures, output ordering and derivative conventions under changed parameters and grids. Unknown methods/settings fail explicitly. For generated callbacks, source text remains stable across calibrations.
4. **Solutions:** compare stage values, policies, raw candidates and finite-value masks with the frozen baseline; independently check budgets, binding constraints and dense objective values. Preserve deterministic tie rules for migration comparisons; use value loss at ties when assessing an independent optimizer. Tolerances and performance procedure are specified in the shared review.
5. **Simulation and estimation:** preserve seeded paths, terminal contributions, moments, parameter sensitivity and result reconstruction. Test merged durables types in bounded serial and grouped-MPI evaluations, including no types, zero spread, empty subsets and original-order pooling. Verify housing present bias/taxes; record CPU/GPU disagreements rather than hiding them in tolerances.
6. **Performance and delivery:** measure construction, compilation, warmed solve, simulation and memory separately on identical settings. Investigate warmed-solve regressions above 10%; no unexplained regression accepted. Update tutorials, setup and affected readers; record remaining hardware-only checks. A mode with outstanding required checks is not complete.

This task produced the specification and supporting reviews; numerical implementation and its acceptance remain future work.
