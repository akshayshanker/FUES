# FUES upgrade to Bellman-SYM — specification, version 2

28 September 2026. **Pre-upgrade development specification; no implementation
performed.** This version replaces
[version 1](bellman_sym_upgrade.md) after a full review of that document, its
five supporting reviews under [`bellman_sym_upgrade_review/`](bellman_sym_upgrade_review/),
and the four upstream briefs it depends on. Version 1 is kept unchanged as the
record of the first draft. Section 12 lists what changed and why.

**Review verdict on version 1: revise before implementation.** The design is
sound and is kept: Bellman-SYM owns the declared model and its bindings, FUES
keeps every numerical algorithm, asva supplies the retained numerical factories
and the runner. Five things had to change. The upstream release version 1
targets is being built now and is not finished, so the specification needs a
gate and a list of what can start before the gate. Housing/renting depends on
upstream capabilities that no lane of the release plans, so it is moved to its
own later phase. The rule for what a `.bl` file may declare was missing, and
without it FUES's numerical conventions would have been written into the model
files. The object that carries model inputs from the bound trellis to the
solvers, the simulator and the estimation driver was described but not named,
and without it each example would derive its inputs its own way. And the
acceptance items were not checkable as written; each now names its oracle, its
quantities and its tolerance.

The review was carried out under the `akshay-ddsl-check` reasoning profile
with four delegated checks whose results are folded in: the state of the
Bellman and asva packages on 28 September, a dependency inventory of FUES
(Appendix A), a check of the citations in the five review documents
(Appendix B), and an architecture review of the ownership cut, the gate, the
scope and the estimation coupling.

Facts in this document were checked on 28 September 2026 against FUES `main`
at `9cafd85`, the FUES estimation branch `feat/cd-estimation-postprocess` at
`679b363` (not yet merged), the kikku checkout at `a54619c`, and bellman-ddsl
`main` at `adec19f70`. Where a fact depends on work not yet merged upstream,
the text says so.

---

## 1 Purpose, benefit and cost

### 1.1 What changes and what does not

FUES's three examples build their models today from dolo-plus YAML stage
files read by the forked `dolo` and `dolang` packages, from `kikku.dynx` for
the period graph, and, in the housing example, from the separate legacy `dynx`
package. The upgrade replaces that construction layer with Bellman-SYM: each
example's model is declared in `.bl` stage files and the outer trellis,
recipe, calibration, methods and settings files that Bellman's factory reads;
FUES reads parameters, settings, method choices, stage order and connections
from the elaborated trellis. Every numerical algorithm in FUES stays as it is:
the endogenous-grid steps, the upper-envelope engines under `src/dcsmm/fues`,
the adjuster root search, the branching and conditioning steps, the
simulators, the estimation criterion and its MPI layout, and every result
array, table and command line.

kikku's retained numerical factories and runner are already renamed asva
upstream (lane A of the release merged at bellman-ddsl `d03f7bff6`). FUES
changes those imports to asva. The `dolang` fork is imported by no FUES
Python file; it is a dependency of `dolo` alone and leaves with it. The
`dynx` package the housing example imports (modules `dynx.stagecraft`,
`dynx.heptapodx`, `dynx.runner`) is a separate legacy package, distinct from
`kikku.dynx`; it is not installed in the FUES environment and is declared
nowhere.

### 1.2 What FUES gains

Version 1 did not say what the upgrade is for. The reviewer's first question
is what the work buys, since a migration that preserves every number and
changes no economics has a cost and needs a stated return. The return is:

1. **Three unmaintained dependencies removed.** The `dolo` and `dolang` forks
   are pinned to fixed commits of two repositories nobody maintains further;
   `kikku.dynx` is deleted upstream; the `dynx` package the housing example
   imports is declared by no dependency list, so housing cannot run from a
   fresh install today (`readme-if-you-are-an-ai.md`, lines 123 to 126). After
   the upgrade FUES depends on `bellman` and `asva`, two packages under active
   development by the same group, and on nothing else for model construction.
2. **One declared model per example, checked by elaboration.** Today the
   economic content of an example is spread over stage YAML, `callables.py`,
   the solver's own reads of `stage.calibration` and `stage.settings`, and
   values inserted into the calibration at run time (`return_grids` in
   `examples/durables/solve.py`, lines 395 to 397). After the upgrade the
   `.bl` files declare states, controls, laws of motion, constraints, utility
   and timing once, Bellman's elaboration refuses an ill-formed declaration
   with a named rule, and FUES reads one resolved record of parameters and
   settings per solve.
3. **A checked correspondence between the declared model and the code.** The
   migration forces every handwritten Numba function in FUES to be compared
   with the equation the `.bl` file declares. That comparison is where the
   pre-existing discrepancies the reviews found (Section 7) become visible and
   recorded instead of implicit.
4. **A route to generated kernels, later.** Once lane E of the release lands,
   the explicit functions a stage declares (utility, marginal utility, budget
   laws) can be compiled from the stage's own text and compared with, or
   substituted for, the handwritten versions. This is optional and is not part
   of the acceptance of this upgrade.

### 1.3 What it costs

The release FUES depends on is estimated upstream at about fourteen working
days from its start on 28 September 2026 (implementation plan §8); lanes A and
T0 are merged, R and N are running, and P, K, C, D, E and W have not started.
FUES work that touches the policy blocks of the `.bl` files, the methods
files or the new access paths cannot usefully start before lanes N, K and C
are merged (Section 3); everything else starts now (Section 4.1). The FUES
work itself is: authoring the model sources for two examples, introducing the
resolved-inputs record and the application settings source, replacing the
construction code in two `solve.py` files, moving the command-line parser into
FUES, rewriting the durables simulator's period traversal, replacing three
imports in the estimation driver, and writing the baselines and comparisons.
Housing is a separate, larger piece of work (Section 4.3).

---

## 2 Terms used in this document

These are Bellman's terms, defined here in plain words so that the rest of the
document can use them without qualification. The authoritative definitions are
in bellman-ddsl `AI/rules/terminology.md`.

- **Stage.** One decision problem within a period, declared in one `.bl` file:
  its states, controls, shocks, laws of motion, constraints and objective. A
  stage has three **perches**: arrival (`arvl`), decision (`dcsn`) and
  continuation (`cntn`), which are the three information sets at which
  functions on the stage are defined.
- **Builder.** The operator a block of stage equations denotes. Backward
  builders construct value functions (from continuation to decision, from
  decision to arrival); forward builders advance a distribution of agents.
  After lane N the backward builders are reached as `stage.build.sol.arvl`,
  `stage.build.sol.dcsn` and `stage.build.sol.policy`, the forward ones as
  `stage.build.sim.arvl` and `stage.build.sim.dcsn`. Today they are reached as
  `stage.builders.<block id>`.
- **Kernel.** The mathematical data one operation of a builder uses: a law of
  motion, a value aggregator such as `V = u(c) + V_cntn`, a shock
  distribution (bellman-ddsl `AI/rules/terminology.md`, row *kernel*). A
  **compiled kernel** is the plain function made from it. After lane E,
  `piece.kernel()` answers a record holding the function's Python source
  (`python`), the parameter names it reads and their bound values, and
  `compile_kernel()` returns a plain function with the kernel's arguments
  followed by its parameters as positional arguments. Today `compile_kernel()`
  returns a closure that interprets the term and cannot be compiled by Numba.
- **Convention map.** This document's name for a FUES function or factor that
  converts between a declared quantity and the array FUES stores for it
  (Section 5.3). The word *adapter* is not used, since upstream it names the
  Rosetta toolkit wrapper and lane E's `egm` adapter.
- **Elaboration.** Parsing and constructing a stage from its text. **Binding
  passes** then attach a calibration (parameter values), methods (which
  numerical method realises which operation) and settings (numerical
  configuration such as grid sizes) to the elaborated stage. Each pass returns
  a new version of the stage; nothing is stored by side effect.
- **Period.** The set of stages an agent passes through in one model period,
  with the **connectors** that map one stage's continuation fields to the next
  stage's arrival fields. **Joins** are the derived connections: within a
  period (`period_joins`) and between adjacent periods (`inter_period_joins`).
- **Trellis.** The whole model: a list of periods, stored terminal period
  first, each holding one **stage occurrence** per stage (a separate object per
  position, so that age-dependent calibration binds to one occurrence and not
  to the shared text), and one inter-period mapping per adjacent pair.
- **Registry and recipe.** The directory holding an application's model
  sources, and the file in it that names the trellis file, the stage files and
  the calibration, methods and settings source lists, with `$slot` names that a
  caller fills at load time. `assemble_trellis(directory, recipe, **fills)`
  runs the five steps in order: load the recipe, elaborate, calibrate,
  methodize, configure.
- **asva.** The renamed kikku: endogenous-grid factories (`asva.egm_1d`),
  compose-and-reapproximate (`asva.compose_interp`), numerical utilities
  (`asva.numerics`), the simulator module (`asva.simulate`, of which the walk
  over periods is retired and raises) and the runner (`asva.run`: sweeps,
  moments, estimation, MPI helpers, result tables, the `RunSpec` records).

Four FUES terms: an **envelope method** is the upper-envelope selection used
after an endogenous-grid step (FUES, DC-EGM, NEGM, RFC, CONSAV as the examples
name them); **raw candidates** are the arrays an endogenous-grid step produces
before upper-envelope refinement; a **factory** is a FUES function that takes
callbacks and grids and returns a configured numerical operator (for example
`make_egm_1d`); a **closure** is such an operator once it has captured the
parameter values of one solve.

---

## 3 Upstream state and the gate

### 3.1 State at bellman-ddsl `adec19f70`

| Lane | What FUES needs from it | State on 28 September 2026 |
|---|---|---|
| A, asva | `asva.egm_1d.make_egm_1d`, `asva.compose_interp.make_compose_interp`, `asva.numerics.{clamp_value, clamp_policy}`, `asva.simulate.{StageForward, BranchingForward, forward_stage, apply_rename, draw_shocks}`, `asva.run.{sweep, SweepResult, RunSpec, TestSpec, format_table, write_results_table, get_comm, is_root, bcast_item, make_moment_fn, moment_names, load_estimation_spec, make_criterion, estimate, NAN_PENALTY, BIG_LOSS}`. Deleted: `kikku.dynx`, `kikku.run.cli` (`parse_cli`), `kikku.run.nest_io` (`save_nest`). `asva.simulate.simulate` and `simulate_period` raise at the call. | **Merged** (`d03f7bff6`). Verified against `packages/asva/asva/` on 28 September. `RunSpec` has `base_spec` and no deprecated aliases; FUES already uses `base_spec`. |
| T0, tests' time | Nothing FUES reads. | Merged (`67e9c97db`). |
| R, housekeeping | The install route (`install-dev.sh`, README) FUES's `setup.sh` will follow; the Python version the release chooses. | Running; rulings RH60 to RH62 recorded. |
| N, builder names | `stage.build.sol.{arvl, dcsn, policy}`, `stage.build.sim.{arvl, dcsn}`, `stage.par[name]` with `.value`, `stage.settings[name]`, `stage.methods[target]`, `denote()`, `diagram()`. The old paths are deleted with no alias. | Running; rulings BN60 to BN63 recorded. Not on `main`: today the stage exposes `builders`, `parameters()`, `settings()`, `denotation()` and `draw()`; `settings()` answers a tuple of rows with keys `name`, `value`, `read at`, `by`, with no item access by name (probe of 28 September). |
| P, card form | Nothing FUES reads. | Not started. |
| K, kinds and methods | The row kinds `Argmax`, `Aggregation`, `Expectation`, `Evaluation`, `Pullback`; `Solution` and `Inversion` retired; `!egm` bound at the policy Evaluation; the `Argmax` row for a joint solve such as the durables adjuster. These fix what a `.bl` file may write for maximisation and for the endogenous-grid method target. | Not started. `Solution` and `Inversion` are still defined (`typing/primitive_definitions.py` lines 117 to 118; `fold/table.py` `KINDS`); the grammar file `syntax/grammars/stage.lark` treats `argmax` and `solve` as ordinary operator heads (`typing/line_unknowns.py` line 43). |
| C, the candidate clause | The reading of an inverse-Euler candidate line with a binding constraint (the durables keeper at the borrowing floor; the retirement worker at the saving floor). | Not started. |
| D, documentation shape | Nothing FUES reads. | Not started. |
| E, kernels as functions | `piece.kernel().python`, `compile_kernel()` as a plain function, `builder.kernels()`; the function `asva.egm_1d.egm` built from a kernels record. Optional for this upgrade (Section 6.4). | Not started. |
| W, documentation update | Nothing FUES reads. | Not started. |
| Release | The tag on `econ-ark/bellman`; a fresh clone installs by the README and `install-dev.sh`; the package suite passes on that install. | Not reached. |

### 3.2 The gate

No FUES policy block in a `.bl` file, no methods file, no call to a lane-N
access path and no change to the construction imports of `solve.py` is
written before all of the following hold, and the FUES session note records
each with its commit or tag:

1. The release tag exists on `econ-ark/bellman`, or, if the owner decides to
   proceed before the tag, the following upstream acceptance items
   (implementation plan §3) are recorded as passed on one named commit of
   bellman-ddsl `main` in a fresh clone: lane N's six items (the new paths);
   lane K's items 3 to 5 (`compile_kernel` refusals; `!egm` bound at
   `policy:Evaluation`; `!root-find` admitting the `Argmax` row); lane C's
   items 4 and 5 (the durables membership at the borrowing floor; the grid
   lines); lane A's items 1, 2 and 5; lane R's item 1 (a fresh clone installs
   by the README). Lane E's six items are required only if generated kernels
   are adopted (Section 6.4).
2. `bellman` and `asva` install into a fresh FUES examples environment from
   the tagged (or named) commit by the documented route, and `import bellman,
   asva` succeeds on the Python, NumPy and Numba versions FUES targets
   (`pyproject.toml`). The two packages are subdirectories of one repository,
   so the pins take the form
   `bellman @ git+https://github.com/econ-ark/bellman@<tag>#subdirectory=packages/bellman`
   and the same for `packages/asva`; asva's optional extra is `mpi`, so the
   present line `kikku[estimation] @ …` (`pyproject.toml` line 54) becomes
   `asva[mpi] @ …`. A lock file under `setup/` records both commits, the
   Python, NumPy, SciPy and Numba versions, and the grammar version Bellman
   reports (`grammar_versions()`, called by the factory at
   `factory/trellis.py` line 79). The release's own environment moves to
   Python 3.13 or 3.14 (decisions record, question 7) while FUES runs 3.12.7;
   whether FUES follows is Decision Q10.
3. The public factory operations FUES will call exist under the names this
   document uses and are exercised by one small probe recorded in
   `AI/working/<date>/`: `load_recipe`, `elaborate_trellis`,
   `calibrate_trellis`, `methodize_trellis`, `configure_trellis`,
   `assemble_trellis`, `period_joins`, `inter_period_joins`, `forward_order`,
   `backward_order` (all exported by `bellman/__init__.py` today, lines 39 to
   64), and the lane-N paths `stage.build.sol.dcsn`, `stage.par[name].value`,
   `stage.settings[name]` (none present today).

The reason for the gate is not caution for its own sake. Lanes K and C change
the grammar of the `.bl` stage file at exactly the rows FUES's models need: the
kind of a maximisation row and the reading of a candidate line at a binding
constraint. A FUES policy block written against today's grammar would be
written twice. The rest of a `.bl` file (parameter declarations, perches,
forward blocks, backward value blocks) is not touched by any lane and may be
drafted and elaborated for refusals with today's package (Section 4.1, item
9).

### 3.3 What starts before the gate

Everything in Phase 0 (Section 4.1) is independent of the release and starts
now: the baselines, the migration register, the resolved-inputs record, the
command-line parser move, the FUES-owned simulator traversal, and the removal of
the three imports asva has already deleted. The last three are forced: the
moment FUES imports from `asva` instead of `kikku`, `parse_cli`, `save_nest`
and `simulate` are gone.

---

## 4 Scope and phasing

### 4.1 Phase 0 — before the gate (starts now)

1. **Merge the estimation branch first.** `feat/cd-estimation-postprocess`
   (`679b363`, fourteen commits over `main`) is not merged. Its interfaces
   (Section 8) are the baseline for durables. Nothing in this upgrade is
   built on the unmerged branch; the branch is merged, then the baseline is
   taken.
2. **Baselines.** For retirement (all four envelope methods) and durables
   (both registries, FUES and NEGM), one recorded run each at a small fixed
   configuration, saved as `.npz` plus a JSON of resolved inputs, under
   `tests/fixtures/migration_baseline/<example>/<config>/`. Contents: Section
   9.1. No housing baseline can be recorded until an environment with the
   `dynx` package exists (Section 4.3); recording one is the first task of
   Phase 3.
3. **The migration register** (Section 5.3), one per example, opened now from
   the reviews' findings (Section 7) and closed before any comparison
   expectation is set.
4. **The command-line parser.** Move the slot-based parser of
   `kikku/run/cli.py` in the kikku checkout at `a54619c` (`parse_cli`, line
   460) into `examples/cli.py` (one FUES-owned module; attribution kept in the
   module docstring), importing `RunSpec`, `TestSpec`, `SimSpec` and
   `make_test` from `asva.run`. Preserve every flag, the argument-order merge,
   `@file` support, the Cartesian sweep axes and the labels. Retirement and
   durables `run.py` import it from there. The test
   `tests/test_method_override.py` (main tree) imports `override_methods` and
   `parse_method_override_str` from `dolo.compiler.spec_factory`; it is
   replaced by a test of the FUES parser's method-selector translation.
5. **The durables simulator's traversal.** `examples/durables/solvers/simulate.py`
   line 21 imports `simulate` and `draw_shocks` from `kikku.asva.simulate`.
   `draw_shocks` survives in asva; `simulate` raises. Write the period
   traversal in FUES (Section 6.5). Its oracle exists: the sha256 digests of
   every simulated array per registry in `tests/test_beta_types.py` (`GOLDEN`),
   recorded from the untouched code.
6. **`save_nest`.** `examples/durables/estimate.py` lines 706 and 943 call
   `kikku.run.nest_io.save_nest`. The estimation specification already records
   that the `.nst` files are empty and not read
   (`cd_estimation_and_postprocess.md`, §5.7). Remove both calls; do not
   replace them.
7. **Private imports.** `examples/durables/postprocess/at_estimates.py` line 27
   and `tests/test_at_estimates.py` line 38 import `_age_group_masks` from the
   runner's moments module. It is private in asva too. Copy the function into
   FUES (it is twenty lines) or request a public export upstream; do not
   import a private name across packages.
8. **Other readers of the deleted modules.** `tests/test_kikku.py` imports
   `kikku.dynx` (`period_to_graph`, `backward_paths`, `forward_order`,
   `load_inter_connector`), `dolo.compiler.model.SymbolicModel` and
   `dolo.compiler.tag_tolerant_yaml`; it tests the old period graph and is
   deleted with it. `tests/test_simulate_retirement.py` (ignored by git)
   imports `kikku.dynx` and the retired simulator walk (Decision Q5). The
   docstrings of `examples/durables/solvers/{branching,adjuster_egm,keeper_egm}.py`
   (lines 193, 272 and 269) name the parameter type
   `dolo.compiler.model.SymbolicModel`; they are reworded when the stage
   argument becomes the FUES resolved record. The notebook
   `examples/durables/notebooks/lifecycle_at_estimates.ipynb` imports
   `load_estimation_spec` from `kikku.run.estimate`; it imports from
   `asva.run` after the change and is re-executed.
9. **Draft `.bl` text outside the policy blocks.** Parameter declarations
   with ASCII names (Section 5.4), perches, laws of motion, shock
   declarations and backward value blocks for retirement and durables may be
   written now and elaborated with today's package to collect refusals; the
   policy blocks and the methods files wait for the gate. The Bellman
   applications are the syntax reference; their economics are not copied
   (Section 7).
10. **The resolved-inputs record and the application settings source**
   (Section 5.4) can be introduced on `main` against the dolo stages first,
   so that the estimation branch's readers and the migration both use them
   (Decision Q9).

### 4.2 Phase 1 and Phase 2 — after the gate

Phase 1 is retirement, Phase 2 is durables (both registries, then estimation).
Retirement first because it has one shock, two branches and no estimation, so
every construction and binding question is met once on the small case. Each
phase ends with its acceptance (Section 9) and with the legacy imports for that
example removed. The dependency lines in `pyproject.toml`, `setup/setup.sh`
and the documentation are removed only when both phases have passed.

### 4.3 Phase 3 — housing/renting, its own specification

Version 1 included housing in the first upgrade. It is moved out for three
reasons, each a fact rather than a preference:

1. FUES's own guide declares the example legacy: it imports the `dynx`
   package that no dependency list declares and cannot run from a fresh
   install (`readme-if-you-are-an-ai.md`, lines 123 to 126; `dynx` is absent
   from the FUES environment, so no baseline of the housing example can be
   recorded today). `AI/CLAUDE.md` marks the example "Do NOT edit" and its
   numerical bodies under `src/dcsmm/models/housing_renting` import `dynx`
   themselves (`horses_c.py` line 54, `horses_h.py` line 42, `horses_t.py`
   line 3), so version 1's instruction to keep `src/dcsmm` unchanged and to
   replace the DynX imports under it cannot both hold.
2. Bellman's housing application is incomplete in ways no lane of the release
   addresses: its calibration holds no income transition matrix because the
   language has no set for a matrix or a list of levels, so `tenure_choice.bl`
   declares `Π` and `z` by stand-in sets, the calibration refuses values for
   them, and the methods file binds `!markov` to produce the chain
   (`applications/housing-renting/calibration.yml`, lines 46 to 54;
   `tenure_choice_methods.yml`, line 13); its trellis has one period with the
   terminal data written in prose (`trellis.yml`, lines 16 to 25); and it has
   no present-biased discounting (`delta_pb`), which FUES's housing example
   implements (`bellman_sym_upgrade_review/housing_renting.md`, "Economic
   differences"). Present bias needs a derivative of the policy in the
   marginal (`horses_c.py` lines 721 to 742), and nothing in Bellman
   differentiates (`kernels-as-functions.md` §6). These are upstream
   prerequisites with no owner yet.
3. The example's CPU and GPU utility functions disagree
   (`log(kappa*(H+iota))` against `log(kappa*H+iota)`; difference 0.596 at
   H = 0 with the standard calibration), and its tax schedule's bracket
   endpoints differ by configuration variant. Those are corrections to decide
   before any migration comparison, and they are independent of Bellman.

Phase 3 therefore gets its own specification once Phases 1 and 2 have passed
and the upstream prerequisites have issue cards in bellman-ddsl. Its likely
steps, for that specification to confirm: (a) housing CPU at `delta_pb = 1`
for one configuration variant, after the income process is either declared as
a two-component shock law with the 49-state Rouwenhorst discretisation as a
setting (the pattern the durables application uses for its AR(1)) or the
matrix-typing limitation is removed upstream; (b) present bias and the tax
variants; (c) GPU, after the utility disagreement is decided. The housing
findings of version 1's review remain the input to that specification and are
summarised in Section 7.3 so that they are not lost. **Decision Q1** asks the
owner to confirm this phasing and, when Phase 3 opens, to lift the "Do NOT
edit" rule for the housing folders.

---

## 5 Ownership and the rule for `.bl` declarations

### 5.1 Who owns what

| Layer | Owner | Contents |
|---|---|---|
| The declared model | Bellman, from FUES-owned `.bl` and outer files under each example's `syntax/` | States, controls, shocks and their laws, laws of motion, constraints, utility, discounting, terminal condition, the stages of a period and their connectors, the trellis of periods, parameter declarations and calibration, method choices, numerical settings. |
| The numerical solution | FUES (`examples/<example>/solvers/`, `src/dcsmm/`) | Grids, endogenous-grid steps, upper envelopes, root search, branching, conditioning, expectation over discretised shocks, result dictionaries and their array layouts. |
| Simulation | FUES (`examples/durables/solvers/simulate.py`) with asva's `StageForward`, `BranchingForward`, `forward_stage`, `draw_shocks` | Particle transitions, pre-drawn shocks, type assignment and pooling, panel outputs. |
| Estimation | FUES (`examples/durables/estimate.py`, `type_groups.py`, `postprocess/`) with `asva.run` | Criterion assembly, MPI layout, manifests, tables, fit at the estimates. |
| Command line | FUES (`examples/cli.py`, each `run.py`) with `asva.run` records | Flags, sweeps, labels, PBS commands. |
| Retained factories | asva | `make_egm_1d`, `make_compose_interp`, clamps, the runner. |

Nothing in FUES reconstructs a stage object, a `SymbolicModel`, a `Mover` or a
second stage hierarchy. Where FUES needs a small record of numerical inputs or
results (grids, arrays, timings), it is a plain FUES record with the arrays in
it, not a symbolic object.

### 5.2 The rule: the `.bl` file declares the economics, not FUES's conventions

Version 1 instructed implementers both to "preserve executed FUES behaviour"
and to author `.bl` files that preserve "marginal scaling", "discounted
marginal convention", "asset/cash coordinates" and "tie rule". Those are
numerical conventions of FUES's arrays, not content of the economic model. If
they are written into the `.bl` files, the model files become wrong in order to
keep the arrays right, and the point of a declared model is lost.

The rule for this upgrade:

1. A `.bl` file declares the economic model in Bellman's conventions, as the
   Bellman applications do: derivatives at the continuation perch are
   undiscounted, the decision stage applies β, the return factor enters the law
   of motion, the terminal stage declares the terminal condition the model
   intends. A `.bl` line is right when an economist reading it recognises the
   model, and wrong otherwise.
2. FUES keeps its arrays exactly as they are. Every conversion between a
   declared quantity and a FUES array is written once in a **convention map**
   (a named function or a documented factor) and recorded as a row of the
   register (Section 5.3). Bit-for-bit preservation of arrays is achieved
   through convention maps, never by editing the declaration. No `.bl` line
   is written to reproduce an array.
3. Where a FUES array disagrees with the declared economics for a reason that
   is not a pure convention (a term charged twice, a terminal condition that
   differs, a bound applied at the wrong coordinate), the disagreement is a
   register row with one of three decisions: preserve for this upgrade and
   correct separately; correct now in a separate commit before the baseline is
   taken; or block the mode until decided. The `.bl` file declares the
   intended economics in every case.
4. Method choices (FUES, NEGM, DC-EGM, RFC, VFI) and settings (grid sizes,
   bounds, tolerances, simulation guards) are bound through the methods and
   settings files, not declared in the stage. A CLI option keeps its spelling
   while its destination becomes a method target or a setting.

Consequence for acceptance: **migration acceptance** means arrays preserved and
the register complete; **economic agreement** is a separate claim, made only
for the modes whose register rows are all *faithful* or *corrected*.

### 5.3 The migration register

One file per example, `examples/<example>/docs/migration_register.md`
(tracked; the `AI/` tree is ignored by git and cannot hold it), opened in
Phase 0 from the reviews' findings and closed before any comparison
expectation is written. One row per declared quantity whose FUES
representation is not the declaration itself, and one row per discrepancy:

| Column | Content |
|---|---|
| Id | `R1`, `D1`, … as in Section 7. |
| Declared quantity and perch | The `.bl` symbol and where it lives (for example the asset derivative of the continuation value at `cntn`). |
| Stored array or callback | Name, module and dictionary key, with axes (for example `d_aV_cntn`, axes `(n_z, n_a, n_h)`). |
| Coordinate | Arrival asset, decision cash, log or level income, and the grid the array is defined on. |
| Scaling and order | The factors applied (β, R, the transition Π) and the order of expectation and scaling, written as an equation. |
| Convention map | The FUES function and line that performs the conversion. |
| Status | One of *faithful* (array equals the declaration), *preserved deviation* (a convention map reproduces the array; the declaration is right), *blocker for economic agreement* (the array disagrees with the declaration for a non-convention reason and is preserved for now), *corrected in commit …*. |
| Decision, owner, date | The owner's initial and date; for a correction, the separate commit. |
| Check | The pointwise test that establishes the row (Section 9.2, C2 and C3). |

Rows known from the reviews, with their expected status: durables
`d_aV_cntn` and `d_hV_cntn` (β and the transition applied before expectation;
`callables.py` lines 170 to 188 of the separable registry), *preserved
deviation*; retirement `dv = du(c)` with `R` inserted in the inverse Euler
step (`operators.py` line 202; `model.py` lines 47 to 48) against Bellman's
`R*du(c)` at arrival, *preserved deviation*; retirement's arrival-asset
coordinate `(c + savings)/R` interpolated at `R × arrival grid` (`model.py`
lines 83 to 84 and 125 to 128; `operators.py` lines 64 to 69) against decision
cash `savings + c`, *preserved deviation*; the durables Tauchen
discretisation's probability floor of 0.001 and renormalisation (`model.py`
lines 73 to 77), a deviation from the declared AR(1) law, *preserved
deviation* with the floor a setting; the `chi` term, the T against T−1
terminal timing and (Phase 3) the CPU/GPU utility disagreement, *blocker for
economic agreement* until decided; housing's `lambda_` multiplied by
`beta*delta*Rfree` at the consumption stage (`horses_c.py` line 535) against
Bellman's return factor in the asset derivative, Phase 3.

### 5.4 The resolved-inputs record, application settings and names

**One record per stage occurrence.** The carrier of model inputs after the
migration is one FUES-owned function per example, proposed as
`resolve_inputs(trellis)` in `examples/<example>/inputs.py`, from a bound
trellis to a frozen record per stage occurrence holding: parameter values by
name, setting values by name, the bound method target per operation, and the
grids built from the settings. It is built once per stage version and never
mutated; every factory, the simulator and the estimation driver receive it by
name. This is what version 1's sentence "resolve parameter/settings values
once" needed and did not name. It replaces three things at once: the reads of
`stage.calibration` and `stage.settings` inside the factories; the
`RetirementModel` proxy of `examples/retirement/model.py`, whose
`__getattr__` (lines 173 to 192) answers any attribute by looking in the
calibration and then the settings, which `AI/CLAUDE.md` ("the calibrated stage
IS the model") forbids as a parallel parameter object; and the stage objects
the durables result and the estimation branch keep in `nest["periods"]` and
read back through `_base_stage(nest).calibration[...]` (`simulate.py` lines
499 to 501). The record keeps the attribute names FUES code uses today, with
no fall-through lookup.

This changes the rule file, and the migration commit updates it:
`AI/CLAUDE.md`'s principle "the calibrated stage IS the model" keeps its
meaning (inputs are derived from the bound stage versions, never from a
separate parameter object) but its example `stage.calibration['beta'] →
callable` becomes `resolve_inputs(trellis)[occurrence].par['beta'] →
callable`, and its chain `calibrate(...) → stage_h → callables_h →
operators_h → solve` gains the record between the stage version and the
callables. The record is not a parallel parameter object: it is the one
derived object, built from the versions and kept in the results in their
place.

**Application-level settings.** `T`, `t0`, `store_cntn`, `warmup_periods`,
`normalisation`, the simulation seed and `N_sim` are read today from an
arbitrary stage (`examples/durables/solve.py` lines 655 to 657 read `T` from
`keeper_cons`'s settings and `t0` from its calibration; `simulate.py` lines
499 to 501 likewise), the pattern `AI/CLAUDE.md` forbids under "stages are
distinct objects". Bellman's recipe refuses a fill that no stage declares and
no registry block admits (`factory/recipe.py`, refusal `fill-names-nothing`),
so `--slot-override T=…` would be refused on a `.bl` registry. Each registry
therefore gets one FUES-owned application settings source (proposed:
`application.yaml` beside the recipe, read by `resolve_inputs`), the command
line routes the existing option spellings to it, and no application setting is
read from a stage occurrence.

**Parameter names.** Lane N rules that `stage.par` is keyed by the name the
file declares and that no ASCII alias exists (`stage-builder-names.md`,
"`stage.par` and `stage.settings`"). FUES's estimation YAML `free:` blocks,
`types.parameters`, `--slot-override`, the PBS scripts and
`check_parameter_names` all use ASCII names (`beta`, `gamma_c`, `sigma_beta`).
Therefore FUES-authored `.bl` files declare ASCII parameter names identical to
today's YAML keys, and no name map exists at the boundary. Bellman's
application files, which use `β`, `γ_c` and `ρ_z`, are syntax references only.
The grammar admits ASCII identifiers (`applications/durables/stages/tenure.bl`
declares `Y` and `y_R`; the housing application declares `a0_T`, `B_T`).

**Reclassification of `b`.** `CLAUDE.md` ("Settings vs Calibration") lists the
borrowing limit `b` among the settings and forbids reading it from
calibration. In the `.bl` file the borrowing limit is the bound of the saving
constraint, which is model content, so `b` becomes a declared parameter bound
by calibration; `a_min`, `h_min`, `w_min`, `n_a`, `n_h`, `n_w`, `fues_*`,
`sim_guard` and the tolerances stay settings; the finite housing set of the
housing example (Phase 3) becomes model content under bellman-ddsl
`AGENTS.md`'s rule that spaces are model, not configuration. The CLI option
`--slot-override b=…` keeps its spelling and binds the calibration slot. The
migration commit rewrites that paragraph of `CLAUDE.md` (Decision Q12).

---

## 6 Design by layer

### 6.1 Model sources

Each example keeps its `syntax/` location and its occurrence names
(`work_cons`, `retire_cons`, `labour_mkt_decision`; `tenure`, `keeper_cons`,
`adjuster_cons`, `bequest`). The stage YAML files are replaced by `.bl` files;
the period, trellis, recipe, calibration, methods and settings files are
written in Bellman's outer formats. Bellman's own applications
(`applications/retirement-choice`, `applications/durables`) are starting
material and syntax references only; Section 7 lists where they differ from
FUES's models, and every difference is resolved by Section 5.2's rule.

Age-dependent calibration is bound per stage occurrence: `calibrate_trellis`
applies the calibration mapping the syntactic trellis holds at each position
(`packages/bellman/bellman/factory/trellis.py`, `calibrate_trellis`). The
Bellman durables application does this for its wage schedule by a function
that runs after `load_recipe` and before the binding verbs and writes each
period's `Y` and `y_R` into that period's `tenure` record from eight
registry-level names (`applications/durables/wage_schedule.py`; 51 periods,
terminal first). FUES's durables wage polynomial and retirement-income
schedule follow the same pattern: a FUES function between loading and binding,
one value per occurrence, and the `age` scalar that `solve.py` lines 447 to
449 insert into each period's calibration is dropped. The trellis is stored
terminal period first; FUES's solvers index by chronological age. The map
between the two is one function in FUES with a test, and no array is stored
twice.

Registry-level inputs that no stage declares (the durables `return_grids`,
the simulation guard) are settings or recipe fills, not calibration values;
`solve.py` lines 395 to 397, which insert `return_grids` into the
calibration, are replaced by a setting read. Application-level values (`T`,
`t0`, `N_sim`, seeds) live in the application settings source of Section 5.4.

The income process is declared as the AR(1) in logs that FUES's own
`tenure.yaml` already states (lines 75 and 79); the Tauchen node count, the
`n_std` width and the 0.001 probability floor are settings, and the floor with
its renormalisation is a register row (Section 5.3).

The two durables registries (`separable`, `cobb_douglas`) each get their own
stage files where the utility differs (the Cobb–Douglas housing-dependent
marginal utilities, their inverse and the terminal indirect utility; the
separable bequest is not a substitute). Files that are identical across the
registries are shared by the recipe, not copied.

### 6.2 Construction and access in `solve.py`

The dolo imports listed in Appendix A are replaced by one construction
sequence per solve. Retirement, which has no age-dependent value, may call
`assemble_trellis(registry, recipe, **fills)`. Durables calls the five public
operations explicitly, `load_recipe`, the wage-schedule write, `elaborate_trellis`,
`calibrate_trellis`, `methodize_trellis`, `configure_trellis`, in that order,
because `assemble_trellis` (`factory/trellis.py` lines 56 to 64) has no hook
between loading and binding; methodize precedes configure because a setting
is read by a method (terminology row *stage version*). The result is the
elaborated, bound trellis; FUES reads from it once, through `resolve_inputs`
(Section 5.4), and stores nothing back into it.

Reads, after lane N, all made inside `resolve_inputs`:

- parameters: `occurrence.par[name].value` (today `callables.py` reads every
  parameter from `keeper_cons`, lines 110 to 129 of the separable registry,
  which cannot recover parameters another stage declares);
- settings: `occurrence.settings[name]` (whether this answers the value or a
  record with `.value` is verified at the gate; version 1 assumed `.value`);
- methods: `occurrence.methods[target]`, translated once into FUES's method
  selection (FUES, NEGM and the others select the same existing routines);
- order and connections: `backward_order`, `forward_order`, `period_joins`,
  `inter_period_joins` from `bellman.factory`.

The private dolo functions FUES calls today (`_normalize_methods`,
`_scheme_for_node`) have no public successor and none is requested; the
method-target records replace them. Nothing private in Bellman is imported.

### 6.3 Parameters, settings and closures

FUES's numerical factories close over parameter values once per solve. That is
kept, with one rule made explicit: a closure is built from the resolved record
of the current solve and is never reused after a calibration, method or grid
changes. Estimation constructs the closures inside the trial (as
`build_criterion` does today), so a changed candidate cannot reuse a stale
closure. Whether generated kernels avoid recompilation across candidates is a
lane-E property and is not claimed here.

### 6.4 Kernels: keep the handwritten functions

The release's kernels are plain functions for explicit constituent functions
only (`kernels-as-functions.md`, §6): no maximisation, root solving,
interpolation, integration or differentiation, and the named utility functions
are inner functions of a kernel, not separate exports. FUES's endogenous-grid
factories take four callbacks (inverse Euler, value, endogenous grid,
curvature; `examples/retirement/solvers/operators.py`, lines 45 to 58) and the
durables adjuster needs a housing-dependent inverse marginal utility and Brent's
root search. Therefore:

- every handwritten Numba function in `model.py` and `callables.py` is kept,
  with its name, signature and dictionary key;
- each is checked against the `.bl` equation it implements, pointwise, at
  non-unit returns and at a changed calibration (Section 9.2), and the check
  is recorded;
- adopting a generated kernel for an explicit function is allowed after lane
  E, one function at a time, after pointwise equality is shown; it changes no
  caller's signature; parameters are passed positionally in the record's
  declared order; compilation happens outside numerical loops; in-memory
  functions are not assumed to support Numba's `cache=True`; the safe order
  of adoption is laws of motion first (pure arithmetic, used by the
  simulator, no convention at stake), then value aggregators and
  arrival-marginal lines, then the explicit inverse-Euler side, endogenous-grid
  and candidate-value lines only after the return-factor rows of the register
  (R3, R4, D4) are closed; utility, its derivatives and its inverses stay
  handwritten, since the release nests them inside kernels and exports none;
- no FUES code extracts an inner function from a generated kernel, cancels
  terms in a larger kernel to recover a utility, or reads a private symbolic
  structure;
- the domain guards the handwritten functions carry (large penalties outside
  positive consumption or housing) are kept, since the plain kernels enforce no
  domain.

### 6.5 Simulation

asva's `simulate` and `simulate_period` are retired upstream and raise; its
`StageForward`, `BranchingForward`, `forward_stage`, `apply_rename` and
`draw_shocks` are complete and are used. The particle helpers `_slice`,
`_scatter` and `_scatter_rec` of kikku's simulator (`kikku/asva/simulate.py`
lines 59 to 86 at `a54619c`) were deleted with the walk and are copied into
FUES with attribution. FUES writes the traversal: a fold over the periods in
chronological order (the reverse of the trellis's storage), the stage steps in
`forward_order`, particles routed across branches by the period's joins and
across periods by `inter_period_joins`, one step per stage by `forward_stage`,
with the pre-drawn shocks keyed `(t, point)` as today. The traversal is
FUES-owned and reads only the order and joins from Bellman; a FUES simulator
with its own timing declaration would be a second source of truth for the
model's timing and is rejected. It does not wait for Bellman's forward record
(the upstream issue card `asva-simulator-over-the-trellis.md`); when that
record arrives, it is adopted only if the digests below still match. The
traversal is written once, with a test on a changed period, on the terminal
period, and one hand-computed two-stage check.

The estimation branch's `simulate_type_subset` and `pool_by_type` (Section 8)
are the entry points; they are kept, and the no-types path continues to route
through the subset path with the full index set, so that the `GOLDEN` digests
of `tests/test_beta_types.py` (sha256 of each array's contiguous bytes with
shape and dtype, at `n_a = n_h = n_w = 30`, `t0 = 55`, `N = 200`, seed 99,
recorded at commit `fb4c757`) are the acceptance oracle for the traversal
rewrite before any Bellman change, and again after. The standard is exact
equality, not a tolerance, because only the traversal changes. Exactness after
the `.bl` migration also requires that every calibration literal in the FUES
`.bl` and calibration files equal today's YAML literal bit for bit; the
baseline JSON of Section 9.1 is the reference.

Retirement's `run.py` advertises a simulation flag with no production
implementation (`run.py` lines 220 to 226 and 303 to 313 execute sweeps);
version 1 correctly says not to claim one. The local
`tests/test_simulate_retirement.py` depends on `kikku.dynx` and is ignored by
git; it is deleted with the dependency unless the owner asks for a retirement
simulator (Decision Q5).

### 6.6 Command line and runner

Section 4.1 item 4. The method-selector strings the old parser accepted
(`stage.target.scheme=TAG`) are translated once, at the application boundary,
into method targets Bellman binds; unknown selectors are refused with the
accepted list. `RunSpec.base_spec` is the field; asva has no aliases.

### 6.7 Persistence and results

No Bellman stage is pickled, and no `.nst` file is written again. Results stay
in FUES's dictionaries and `.npz`/CSV files. The estimation manifest gains the
Bellman and asva commits and the sha256 of every model-source file the recipe
read, so that a result from before the migration cannot be mistaken for one
after it. Housing's proprietary object readers are Phase 3.

---

## 7 Per-example plans and the register rows

The reviews under `bellman_sym_upgrade_review/` are the detailed record; this
section carries what the implementer must act on and keeps the citations. Line
numbers refer to FUES `main` at `6d7c3fe`/`97bc3f5` unless stated; the
citation check in Appendix B records where they have moved.

### 7.1 Retirement (Phase 1)

Preserve: `solve_nest`'s arguments and four-result tuple, the backward loop and
result dictionaries (`solve.py` lines 193 to 236), `RetirementModel`'s
attributes and helper contracts, the EGM factory's four callbacks and result
tuple (`operators.py` lines 45 to 58, 109 to 122), the occurrence names, the
command line, sweeps, tables and plots, and the postprocessing's reconstruction
of calendar order from `t` (`postprocess/diagnostics.py` lines 6 to 46).

Replace: the dolo factory imports and private method calls (`solve.py` lines
24 to 27, 63 to 66, 138 to 139, 314 to 374), `.calibration` reads in
`model.py` (lines 149 to 192), the kikku imports in `operators.py` (lines 16
to 17) with asva, the stage YAML with `.bl`, and `benchmark.py`'s
method-record decoding (lines 114 to 126).

Register entries (all pre-existing; decision column open):

| # | Finding | Where | Bellman's application differs how |
|---|---|---|---|
| R1 | Terminal continuation is logarithmic utility with derivatives after the last solved period. | `solve.py` 152–161 | `bequest.bl` lines 3–17 declare zero terminal continuation. |
| R2 | The work cost `delta` is subtracted in the worker value (`model.py` 51–52; `operators.py` 127) and again placed in the branch comparison by the old choice YAML (lines 42–45). Must be charged once. | as cited | `ret_choice.bl` lines 14–25 place it in the choice. |
| R3 | `dv = du(c)` stored, `R` inserted in the inverse Euler step. | `operators.py` 202; `model.py` 47–48 | `worker_cons.bl` 27–35 supplies `R*du(c)` at arrival and uses β alone. Combining both multiplies by `R` twice. Convention-map row, *preserved deviation*. |
| R4 | Retiree grid formed as `(c + savings)/R` and interpolated at `R × arrival grid`; the constrained branch sets consumption to the asset grid. | `model.py` 83–84, 125–128; `operators.py` 64–69 | `retiree_cons.bl` 13–18, 33–35 use decision cash `savings + c`. Convention-map row, *preserved deviation*. |
| R5 | Grid built as `linspace(b, grid_max_A, grid_size)`; the methods file names `n_a/a_min/a_max`. Ties select retirement; positive `smooth_sigma` is a probability-weighted value, not a log-sum. | `model.py` 156–162; `work_cons_methods.yml` 34–40; `operators.py` 189–203 | Map options explicitly; do not infer taste-shock semantics. |
| R6 | The constrained arrival operator records consumption `w − grid[0]` but evaluates utility at `w`; with log utility, resources 1 and floor 0.2 the difference is 0.2231. | `operators.py` 149 | Hidden by the tiny default floor; compare at a non-negligible floor. |
| R7 | The Euler diagnostic interpolates next consumption at current assets and assumes wage income regardless of branch. | `postprocess/diagnostics.py` 96–103 | Replace with a branch-aware diagnostic before using it as an oracle. |
| R8 | `RetirementModel` answers any attribute by looking in the calibration and then the settings (`__getattr__`, `_cal`), and reads `b` from calibration first. | `model.py` 155, 173–192 | Replaced by the resolved-inputs record (Section 5.4); same attribute names, no fall-through. |

### 7.2 Durables (Phase 2)

Preserve: the numerical bodies of `keeper_egm.py`, `adjuster_egm.py`,
`branching.py`, `conditioning.py`, grid construction (`model.py` lines 48 to
103: Tauchen log-income nodes, probability floor 0.001 and renormalisation,
feasible wealth floor, nonuniform spacing), array order `(z, a, h)`, clamping
and extrapolation, `adjust` on value ties (`branching.py` 118–145),
`solve(...) -> (nest, grids)`, the dictionary names, registry selection,
calibration overlays, numerical overrides, method selection and random draws;
the schedule keeper → adjuster → tenure choice → expectation (`solve.py`
242–275) is FUES's and does not become a generic solver.

Replace: the dolo imports and method normalisation (`solve.py` 8–14, 68–156),
recipe and period construction (604–650), recalibration (214–228), reads of
`stage.calibration`/`stage.settings` in both `callables.py` modules, the
kikku imports (Appendix A), and the `return_grids` insertion (395–397).

Register entries:

| # | Finding | Where | Treatment |
|---|---|---|---|
| D1 | Separable callable utility subtracts `chi`; the keeper value formula and the stage YAML omit it (baseline `chi = 0`). | `callables.py` 24–30, 229–236 | Declare one utility in `.bl`; record which the arrays follow. |
| D2 | Decisions are solved at age T with terminal continuation, but the simulator ends decisions at T−1 and adds terminal utility directly. | `solve.py` 443–455; `simulate.py` 501, 375–393 | Decide the model's last decision age; the `.bl` trellis declares it. |
| D3 | The Cobb–Douglas Euler diagnostic uses carried housing in next-period marginal utility even when the next branch adjusts; `H_adj` is passed but unused. | `simulate.py` 75–86, 153 | Correct the diagnostic before it is an oracle. |
| D4 | Continuation marginals `d_aV_cntn`, `d_hV_cntn` carry β and the transition; `V_cntn` does not. | `callables.py` 170–188; `keeper_egm.py` 356–424 | Convention-map row, *preserved deviation*; β applied once. |
| D5 | Borrowing floor `b = 0.01` in both registries, distinct from the grid's lower bound; the constrained segment is built at saving `b`; the adjuster adds its constrained point and checks the Euler inequality. | `keeper_egm.py` 323–339; `adjuster_egm.py` 432–471 | The `.bl` declares the floor `b` as a parameter (Section 5.4); do not copy Bellman's zero-saving example. |
| D6 | Log-income coordinate and floored transition matrix versus Bellman's level income with a multiplicative transition. | `model.py` 71–80; `tenure.bl` 14–19 | FUES coordinates kept; any translation is a convention map. |
| D7 | The fast keeper path returns `cntn_data = None`; the diagnostic output stores refined arrays, not raw candidates; switching paths changes constrained padding from `max(2, n_a//10)` to `n_a`. | `keeper_egm.py` 252–255, 395 | Add a bounded capture of the fast path's pre-refinement arrays (Section 9.2); never describe refined arrays as raw candidates. |
| D8 | The Tauchen transition matrix is floored at 0.001 and renormalised, a deviation from the declared AR(1) law. | `model.py` 73–77 | *Preserved deviation*; the floor is a setting; the `.bl` declares the law. |
| D9 | `T`, `t0`, `store_cntn` and the simulation horizon are read from `keeper_cons` or from "the base stage". | `solve.py` 655–657; `simulate.py` 499–501 | Moved to the application settings source (Section 5.4). |

### 7.3 Housing/renting (Phase 3 input, not in scope now)

Kept from the review so that nothing is lost: preserve `horses_c.py` (EGM,
envelope preparation, tax candidates, CPU VFI), the array-based discrete-choice
kernels of `horses_h.py`, the tax, gradient and GPU routines; replace the
DynX loader and circuit construction (`solve_runner.py` 265–267, 386–477),
the `mover.model` reads in the numerical factories, and the `Solution` object
storage with a small FUES record holding `vlu`, `Q`, `lambda_`, policies,
grids and timings. Register entries: the marginal conversion (`lambda_` times
`beta*delta*Rfree` at `horses_c.py` 535 against Bellman's return factor in the
asset derivative), present bias (`delta_pb < 1`: choices maximise `Q` while
forwarding lifetime value, with a policy-gradient term; absent in Bellman),
the CPU/GPU utility disagreement (0.5958 at H = 0, 0.0265 at H = 1 with the
standard calibration), the tax-endpoint precedence by variant (0.05 against
0.18852 at assets 6.97), the `_snapped_to_grid` table stored in settings, the
49-state income chain with no Bellman domain, chronological storage against
terminal-first, and the missing tests under `tests/`. "Present-biased choice
value" means the objective `Q` the agent maximises when `delta_pb < 1`, which
differs from the lifetime value `V` that is carried forward; FUES stores both.
The model sources to re-author are the five `config_HR/*/master.yml` files
with their `connections.yml` and five stage files each (38 files). The first
accepted housing migration, when it comes, is `delta_pb = 1`, CPU, one
variant.

### 7.4 The register rows

The seventeen rows above (R1 to R8, D1 to D9) are the opening content of the
two migration registers of Section 5.3, one under `examples/retirement/docs/`
and one under `examples/durables/docs/`. No comparison expectation is written
until every row of the phase's example has a status and a decision. A row
with status *preserved deviation* carries the `.bl` declaration of the
intended economics and the convention map that reproduces the current array;
a row with status *blocker for economic agreement* carries the mode it
blocks.

---

## 8 The estimation branch: what it hands over

`feat/cd-estimation-postprocess` is the durables baseline once merged. One
fact about it decides how the two pieces of work compose: the branch, like
`main`, keeps the calibrated dolo stage objects in the result's period entries
(`nest["periods"][h]["stages"]`) and reads model inputs back from them
(`_base_stage(nest).calibration["t0"]`, `simulate.py` lines 499 to 501; the
type member's `beta` from the solved model, `cd_estimation_and_postprocess.md`
§5.6). After the migration the period entries hold the resolved-inputs record
of Section 5.4 instead, and every reader goes through three named accessors:
the calibration names of the registry (today `resolved_calibration_keys`), a
setting's value by name, and a type member's `beta`. Whether those accessors
are introduced on `main` before the branch merges or in the migration after
it is Decision Q9. The branch's interfaces that this upgrade touches, from
the worktree at `679b363`:

| Interface | Where | What the migration does |
|---|---|---|
| `build_criterion(mod_dir, spec, spec_factory, solver_method, calib_overrides, setting_overrides, N_sim, simulation_seed, comm, *, types_spec=None, group=None) -> (criterion, moment_fn, data_moments, denorm)` | `estimate.py` | Unchanged signature. Inside, the solve is Bellman-constructed; the closures are built per trial as today. |
| `resolved_calibration_keys(mod_dir, spec_factory_name)` | `estimate.py` line 46 imports `dolo.compiler.spec_factory.load/make` | Re-implemented over the elaborated trellis: the union of `par` names over all occurrences plus the registry block. Same return type. |
| `_denormalised_moment_fn` reads `load_syntax` | `estimate.py` line 158 | Replaced by the resolved settings record (Section 6.2). |
| `check_parameter_names(keys, free_names, calib_overrides, types_spec)` | `estimate.py` | Unchanged; its `keys` now come from the trellis. |
| `save_nest` calls | `estimate.py` lines 706, 943 | Removed (Section 4.1 item 6). |
| `beta_types.discretise_types / expand_types / draw_types / pool_by_type`; the manifest record `beta_types` | `solvers/beta_types.py` | Unchanged. `expand_types` produces K calibrations; each becomes one fill of the calibration slot and one bound trellis, so K solved models per candidate as today. The record's `nodes`, `shares`, `implied_mean` keys are the contract `at_estimates.py` reads. The all-agent birth draw and the original-order pooling of `simulate_type_subset` are preserved by the traversal of Section 6.5 without change. |
| `type_groups` (EVAL/PART/FAIL/STOP; `check_divisibility`; `split_type_groups`) | `type_groups.py` | Unchanged; independent of model construction. |
| `simulate_type_subset`, `simulate_lifecycle(..., types=None)` | `solvers/simulate.py` | Traversal rewritten (Section 6.5); signatures unchanged; `GOLDEN` digests hold. |
| `at_estimates.solve_at_estimates -> [(share, nest, grids)]`, `fit_at_estimates` with `penalised` rows, `fit_table_at_best.csv` | `postprocess/at_estimates.py` (`fit_at_estimates` at line 432 of `679b363`) | Unchanged contract. The reconciliation of the fit table with the objective at `theta_best` (to about 1e-9, a NaN simulated moment scored as `NAN_PENALTY`) is a test on the branch (`tests/test_at_estimates.py`) and is the oracle for acceptance item E3 (Section 9.4). One point from the implementation review's correction 2 stands: the function scores a key only if it is present in both the data and the simulated moments, which equals the criterion's rule only because `make_moment_fn` emits every key of the moment list (with NaN where a moment cannot be computed). E3 therefore requires the reconstruction to iterate over the moment list, or to assert that property, so that a key the moment function fails to emit is penalised as the criterion penalises it and not skipped. |
| `estimation_tables.write_estimation_tables(runs, out_dir, fits=())` | `postprocess/estimation_tables.py` | Unchanged; imports `NAN_PENALTY` from `asva.run.estimate`. |
| Manifest fields `spec_factory, solver_method, git_commit, beta_types, max_iter` | `estimate.py` | Gains `bellman_commit`, `asva_commit`, `model_source_sha256` (Section 6.7). |

The criterion's rules are asva's and unchanged: weights `1/data²` where
`|data| ≥ 1` and 1 otherwise (so the same proportional error scores
differently in dollars and in ratios; this is recorded, not claimed to be unit
free), `NAN_PENALTY = 1e6` for an absent or NaN simulated target, `BIG_LOSS =
1e10` for a trial exception; an age group written `[40, 44]` selects ages 40 to
43 (`_age_group_masks`, a known behaviour recorded in `AI/todo.md`), and level
moments are converted to Australian dollars by the existing denormalisation.
Acceptance item E2 compares every contribution before and after the change of
import on one synthetic panel, including a missing and a non-finite moment, so
that the two penalty paths are exercised and distinguished from a successful
evaluation.

---

## 9 Acceptance

Acceptance is per phase. Every check names its oracle. Tests are tracked with
`git add -f` because `.gitignore` line 67 ignores `test_*.py` (Decision Q6
proposes narrowing that rule).

### 9.1 Baseline contents (Phase 0)

For each example and configuration under
`tests/fixtures/migration_baseline/`: the git commit of FUES and of each
dependency; the resolved calibration and settings as JSON; every grid; the
seeds; the terminal inputs; for every stage occurrence the value, policy,
derivative arrays and finite-value masks; the raw candidate arrays and selected
branches (durables via the capture of D7; retirement's EGM arrays); the
simulated panel digests (durables); the moments and criterion value at one
parameter vector (durables); wall times for construction, first compilation
and warmed solve, each separately, five warmed repeats, medians and the thread
count.

### 9.2 Phase 1 and 2 checks, common

| Id | Check | Oracle and tolerance | Reason for the tolerance |
|---|---|---|---|
| C1 | Dependency removal: a fresh examples environment installed from `pyproject.toml` with the `#subdirectory=` pins of Section 3.2 and without `dolo`, `dolang` or `kikku` imports the example's modules; core `dcsmm` installs alone. `dynx` is counted at Phase 3. | `pip install`; `import`; `rg` for the three names finds nothing outside `AI/`, `examples/housing_renting`, `src/dcsmm/models/housing_renting` and history. | None needed. |
| C2 | Scalar functions: each handwritten Numba function against the `.bl` equation it implements, evaluated by an independent scalar formula at three calibrations including a non-unit return. | Absolute difference ≤ 1e-12 on float64 inputs of order one. | Same formula, same arithmetic; only rounding differs. |
| C3 | Convention-map rows: each conversion equation of Section 5.3 holds pointwise on the baseline arrays. | Scaled difference `max(abs(new − old)/max(1, abs(old))) ≤ 1e-12`. | A multiplicative factor applied once. |
| C3a | Construction: two occurrences of one stage text are bound independently (a changed calibration at one age leaves the other unchanged); age-dependent values land on the occurrence of their age; the resolved record equals the values in the baseline JSON. | Exact. | Reads of bound values; no arithmetic. |
| C4 | Solution arrays: every stage array, mask and raw candidate array against the baseline; plus two independent references that do not reuse the production interpolator or transition: one hand-computed terminal identity and one two-state expectation with a nonsymmetric transition matrix. | Identical finite masks; scaled difference ≤ 1e-8 over finite nodes; any larger difference explained by a named formula, operation-order or algorithm change before it is accepted. Tolerance is never widened to pass. The two references: ≤ 1e-12. | Interpolation and summation order may differ at the 1e-10 level; 1e-8 leaves two orders of magnitude for detection. The references are closed-form. |
| C5 | Ties: durables selects adjustment at value equality, retirement selects retirement; checked on constructed ties. | Exact. | A rule, not an approximation. |
| C6 | Construction refuses an unknown method target, an unknown setting and an unknown parameter name, each with the accepted list in the message. | The refusal's rule name and message. | None needed. |
| C7 | Rebinding: solve at θ₀, then θ₁, then θ₀ with common draws; the third equals the first exactly; a parameter-dependent primitive equals its formula at θ₁. | Exact for the repeat; C2 for the primitive. | Establishes that no closure is reused, nothing more. |
| C8 | Performance: construction, first compilation, warmed solve, simulation and peak memory, each separately, five warmed repeats, medians, same thread count, grid and horizon as the baseline. | A warmed-solve median more than 10 % above the baseline is investigated and explained; no unexplained regression is accepted. Construction time may rise (elaboration replaces YAML reads) and is reported, not bounded. | The 10 % figure is a migration criterion the owner set, not a measured claim. |

### 9.3 Phase 1, retirement

| Id | Check |
|---|---|
| R-A | C1 to C8 on the four envelope methods at the baseline configuration. |
| R-B | Register rows R1 to R8 decided; the `.bl` terminal stage declares the intended terminal utility; the work cost appears in exactly one place in the `.bl`; the convention maps for R3 and R4 are tested at `R ≠ 1`; `RetirementModel` is replaced by the resolved record (R8). |
| R-C | Absorbing retirement holds through both routes of `solve.py` lines 165 to 172 on the migrated construction. |
| R-D | One executed notebook, `examples/retirement/notebooks/migration_account.ipynb`, showing the construction, `stage.build.sol.dcsn.denote()` for each stage, the bound parameters, the timing of the two stages and the branch, the terminal inputs, and the C4 comparison table, with room for the owner's observations. |

### 9.4 Phase 2, durables

| Id | Check |
|---|---|
| E1 | C1 to C8 on both registries, FUES and NEGM, at the baseline configuration; the D7 capture is excluded from C8's timings. |
| E2 | Criterion: on one synthetic panel, every moment contribution equal before and after the change from `kikku.run` to `asva.run`, including one missing and one non-finite moment; the three outcomes (success, `NAN_PENALTY`, `BIG_LOSS`) each observed once and reported distinctly in the manifest. |
| E3 | Fit at the estimates: `fit_table_at_best.csv` reconciles with the objective at `theta_best` to ≤ 1e-9 on the fixture runs `tests/fixtures/estimation_run*/`, as `tests/test_at_estimates.py` checks today; a run whose last candidate differs from `theta_best` is included; and a moment list containing one key the moment function does not emit is scored as `NAN_PENALTY` by both the criterion and the reconstruction (Section 8). |
| E4 | Types: `GOLDEN` digests of `tests/test_beta_types.py` hold with no types; with types, zero spread reproduces the single-beta solution; an empty subset is handled; pooling is in original agent order; the grouped-MPI evaluation runs on `size = n_points × K` and refuses any other size on every rank before splitting (`tests/test_type_groups_mpi.py`). |
| E5 | Register rows D1 to D9 decided and recorded; the `.bl` declares the borrowing floor `b` as a parameter, the AR(1) income law in logs, and both utility families; the application settings source holds `T`, `t0`, `N_sim` and the seed, and no stage occurrence is read for them (D9). |
| E6 | One executed notebook, `examples/durables/notebooks/migration_account.ipynb`, as R-D, for one registry, plus the convention-map rows of D4. |

### 9.5 What acceptance does not establish

An unchanged array establishes preservation. It does not establish that the
model is economically right where a register row is "preserve and correct
separately", and no document of this upgrade claims economic agreement for such
a mode. Statistical identification and standard errors are outside this work.

---

## 10 Risks

1. **The upstream schedule moves.** The gate is stated in terms of merged
   lanes and a tag, not dates. If lane E is late, the upgrade proceeds without
   generated kernels (Section 6.4 makes them optional).
2. **Lane N changes a name after this document.** Every access path is verified
   at the gate by the probe of Section 3.2; the names in this document are
   those of the briefs on 28 September.
3. **The grammar refuses a FUES declaration.** Bellman's factory may refuse a
   composition FUES needs (for example the adjuster's joint solve as one
   `Argmax` row, lane K's KM1). The refusal is recorded with its rule name and
   an issue card is opened upstream; FUES does not work around it by
   reintroducing a graph library.
4. **Baselines taken on the wrong branch.** Durables baselines are taken only
   after the estimation branch is merged (Section 4.1 item 1).
5. **A "preserve" decision freezes an error into a paper table.** The register
   makes each such decision explicit and dated; the changelog entry of the
   migration lists them.
6. **FUES's `.bl` files drift from the grammar.** After the release, FUES pins
   the Bellman and asva versions in `pyproject.toml` and `setup/setup.sh`, as
   it pins dolo today, and the FUES `.bl` files are offered to bellman-ddsl as
   conformance inputs so that a later grammar change is tested against them.

---

## 11 Decision register

Each row records a decision only the owner can make, with the recommendation
this review reached and its reason. Rows are closed by the owner's initial and
date.

| # | Decision | Recommendation | Reason | Owner |
|---|---|---|---|---|
| Q1 | Scope of the first accepted upgrade: retirement and durables (Phases 1 and 2), housing in its own later specification. | Phase housing out. | Section 4.3: housing is declared legacy, depends on upstream capabilities no lane plans, and carries two unresolved corrections independent of Bellman. | |
| Q2 | The `.bl` rule: files declare the intended economics in Bellman's conventions; FUES arrays preserved through convention maps; disagreements in the register. | Adopt. | Section 5.2: the alternative writes numerical conventions into the model files. | |
| Q3 | Start the FUES policy blocks and methods files only after the gate of Section 3.2; the rest of the `.bl` text may be drafted now. | Adopt. | Lanes K and C change the rows FUES's models write. | |
| Q4 | Merge `feat/cd-estimation-postprocess` before the durables baseline. | Merge first. | The branch's interfaces are the baseline; building the upgrade on an unmerged branch means comparing against two moving references. | |
| Q5 | Retirement simulation: delete the ignored `tests/test_simulate_retirement.py` with `kikku.dynx`, or write a small FUES retirement simulator. | Delete; no production caller exists. | Section 6.5. | |
| Q6 | Narrow `.gitignore` line 67 (`test_*.py`) to the folders it was meant for, so that new tests need no `git add -f`. | Narrow it. | Two branches have now had to force-add tests. | |
| Q7 | Whether to request public exports upstream for `_age_group_masks` (asva) or copy the function into FUES. | Copy into FUES. | Twenty lines; no cross-package private import. | |
| Q8 | Whether the *preserved deviation* and *blocker* rows of the register are corrected in FUES before or after the upgrade. | After, in one commit per row with its own comparison. | Keeps the migration diff free of numerical changes. | |
| Q9 | Composition order with the estimation branch: (a) merge the branch, then introduce the resolved-inputs record and change its readers in the migration; or (b) introduce the record and the application settings source on `main` first, against the dolo stages, and have both the branch and the migration use them. | (b) if the branch is not merged within the week; otherwise (a). | (b) gives the estimation code one interface change instead of two; (a) is simpler if the merge is imminent. Section 8. | |
| Q10 | Install route and interpreter: pin `bellman` and `asva` by git address with `#subdirectory=` at the release tag (Gadi must reach `econ-ark/bellman`), or vendor wheels; and whether FUES follows the release to Python 3.13 or 3.14 or stays on 3.12. | Git pins with `#subdirectory=`, as FUES pins dolo today; stay on 3.12 until the release's suite is shown to pass on 3.12, then decide. | Section 3.2. The release declares `>=3.12` and runs on the latest interpreter the stack supports (decisions record, question 7). | |
| Q11 | The durables terminal timing the `.bl` declares: decisions solved at age T with terminal continuation (the solver), or decisions ending at T−1 with terminal utility added directly (the simulator). | The owner's; the two disagree today (register row D2) and the `.bl` must state one. | Section 7.2, D2. | |
| Q12 | Reclassify the borrowing limit `b` from a setting to a declared parameter and rewrite the "Settings vs Calibration" paragraph of `CLAUDE.md` accordingly. | Reclassify. | Section 5.4: the bound of the saving constraint is model content. | |

---

## 12 What changed from version 1

1. Added the review verdict, the checked commits and this section.
2. Added Section 1.2 (what FUES gains) and 1.3 (cost). Version 1 had no
   statement of benefit.
3. Added Section 2, definitions in plain words. Version 1 defined only
   *trellis* and *kernel* and used *occurrence*, *perch*, *binding pass*,
   *join*, *recipe*, *elaboration*, *methodization* and *envelope method*
   without definition.
4. Replaced "Required upstream state" with Section 3: the state of each lane at
   `adec19f70`, the gate, and what starts before it. Version 1 said "verify
   against the merged source" with no gate; two of its rows are now met (asva
   merged; `RunSpec` aliases gone) and the rest are not.
5. Moved housing/renting to Phase 3 with its own specification (Section 4.3,
   Decision Q1). Version 1 covered all three examples in one upgrade.
6. Added Section 5.2, the rule that `.bl` files declare the economics and FUES
   preserves arrays through convention maps, with one migration register per
   example (5.3) carrying a status vocabulary. Version 1 asked implementers to
   preserve numerical conventions in the declarations and named no register.
6a. Added Section 5.4: the resolved-inputs record as the one carrier of model
   inputs (replacing the `RetirementModel` fall-through proxy and the stage
   objects kept in results), an application settings source for `T`, `t0`,
   `N_sim` and seeds (today read from an arbitrary stage), the rule that FUES
   `.bl` files declare ASCII parameter names equal to today's YAML keys (lane N
   rules out aliases), and the reclassification of `b` from setting to
   declared parameter with the `CLAUDE.md` sentence it supersedes.
6b. Section 6.2 now requires explicit binding passes for durables, because the
   wage schedule is written between loading and binding and `assemble_trellis`
   has no hook there; methodize before configure; the `age` calibration
   scalar dropped. Section 6.4 gives the safe order for adopting generated
   kernels. Section 6.5 names the three particle helpers asva deleted, the
   exact-equality standard for the simulator and the bit-identical calibration
   literals it presupposes.
6c. Replaced the word "adapter" (reserved upstream for the Rosetta wrapper and
   lane E's `egm`) by "convention map", and corrected the definition of
   *kernel* to the terminology table's (the operator's mathematical data; the
   callable is the *compiled kernel*).
7. Marked the three imports asva has already deleted as forced Phase 0 work
   (`parse_cli`, `save_nest`, `simulate`), with the `GOLDEN` digests as the
   simulator oracle, and listed the other readers of the deleted modules
   (two tests, three docstrings, one notebook) that version 1 did not name.
   Version 1 listed the three as shared support to be done in order with the
   rest.
8. Replaced the six acceptance items with per-phase checks that each name an
   oracle and a tolerance with its reason (Section 9). Version 1 referred to
   "tolerances specified in the shared review", which the implementation
   review found insufficient, and to "raw candidates" that the fast keeper
   path does not produce.
9. Consolidated the reviews' findings into seventeen register rows (Section
   7: R1 to R8, D1 to D9) with a required status and decision per row.
10. Corrected the estimation coupling (Section 8): the fit reconstruction the
    implementation review criticised was fixed on the branch before `679b363`;
    `resolved_calibration_keys` and the `load_syntax` read are named as the
    two dolo dependencies of the estimation driver; the `save_nest` calls are
    named.
11. Added the decision register (Section 11, twelve questions, `Q1` to `Q12`,
    each with a recommendation) and the risks (Section 10). Four of the
    questions come from the architecture review: the composition order with
    the estimation branch (Q9), the install route and interpreter (Q10), the
    durables terminal timing the `.bl` must state (Q11) and the
    reclassification of `b` (Q12).
12. Rewrote the prose in plain English with one instruction per sentence, and
    removed the figurative verbs the register check found in version 1
    ("repoint", "rebase", "silently", "hiding in tolerances", "revive",
    "wholesale", "live").
13. Recorded that `dolang` is imported by no FUES file and leaves with `dolo`;
    version 1 listed it as a construction dependency to replace.

## Appendix A — Legacy imports to replace

From the dependency inventory run on 28 September 2026 over `examples/`,
`src/`, `tests/`, `scripts/` and `benchmarks/` of FUES `main` at `9cafd85`
and of the estimation worktree at `679b363` (worktree preferred for the
estimation files). `scripts/` and `benchmarks/` import none of the four
packages. `dolang` is imported nowhere.

### A.1 dolo

| Module | Names | FUES files |
|---|---|---|
| `dolo.compiler.spec_factory` | `load`, `make` | `examples/durables/solve.py:14`; `examples/retirement/solve.py:27`; worktree `examples/durables/estimate.py:46` (`resolved_calibration_keys`, lines 457 to 472) |
| `dolo.compiler.spec_factory` | `override_methods`, `parse_method_override_str` | `tests/test_method_override.py:13` (main only) |
| `dolo.compiler.stage_factory` | `calibrate` | `examples/durables/solve.py:8` |
| `dolo.compiler.stage_factory` | `load_syntax` | `examples/durables/estimate.py` (main 146; worktree 158) |
| `dolo.compiler.period_factory` | `make`, `period_to_graph`, `load` | `examples/durables/solve.py:9–11`; `examples/retirement/solve.py:24–25` |
| `dolo.compiler.nest_factory` | `backward_paths` | both `solve.py` |
| `dolo.compiler.nest_factory.loader` | `load_inter_connector` | both `solve.py` (retirement at 316) |
| `dolo.compiler.methodization` | `_normalize_methods`, `_scheme_for_node` (private) | both `solve.py` |
| `dolo.compiler.tag_tolerant_yaml` | `load_yaml_tag_tolerant` | `examples/retirement/solve.py:315`; `tests/test_kikku.py:13` |
| `dolo.compiler.model` | `SymbolicModel` | `tests/test_kikku.py:15`; docstring type names only in `examples/durables/solvers/{branching.py:193, adjuster_egm.py:272, keeper_egm.py:269}` |

### A.2 kikku

| Module | Names | FUES files | asva |
|---|---|---|---|
| `kikku.asva.egm_1d` | `make_egm_1d` | `examples/durables/solvers/keeper_egm.py:32`; `examples/retirement/solvers/operators.py:16` | kept |
| `kikku.asva.compose_interp` | `make_compose_interp` | `examples/retirement/solvers/operators.py:17` | kept |
| `kikku.asva.numerics` | `clamp_value`, `clamp_policy` | `keeper_egm.py:31`; `adjuster_egm.py:29` | kept |
| `kikku.asva.simulate` | `simulate`, `draw_shocks` | `examples/durables/solvers/simulate.py:21` | `draw_shocks` kept; `simulate` raises |
| `kikku.asva.simulate` | `StageForward`, `BranchingForward` | `keeper_egm.py:449`; `adjuster_egm.py:851`; `branching.py:306` | kept |
| `kikku.dynx`, `kikku.dynx.graphs` | `period_to_graph`, `backward_paths`, `forward_order`, `load_inter_connector`, `load_syntax`, `instantiate_period` | `tests/test_kikku.py:16–17`; `tests/test_simulate_retirement.py:6, 10` (main only, ignored) | deleted |
| `kikku.run` | `parse_cli` | `examples/durables/run.py:19`; `examples/retirement/run.py:10` | deleted |
| `kikku.run` | `sweep`, `write_results_table` | both `run.py`; `examples/durables/postprocess/writer.py:22` | kept |
| `kikku.run.types` | `RunSpec`, `TestSpec` | both `run.py`; both `postprocess/writer.py`; `examples/retirement/benchmark.py:17` | kept, aliases removed |
| `kikku.run.sweep` | `SweepResult` | both `postprocess/writer.py`; `benchmark.py:16` | kept |
| `kikku.run.metrics` | `format_table` | `examples/durables/postprocess/writer.py:23` | kept |
| `kikku.run.mpi` | `get_comm`, `is_root`, `bcast_item` | `estimate.py:51`; `benchmark.py:15` | kept |
| `kikku.run.estimate` | `load_estimation_spec`, `make_criterion`, `estimate`, `diagnostics`, `BIG_LOSS`, `NAN_PENALTY` | `estimate.py:47–49`; `postprocess/at_estimates.py:26`; `postprocess/estimation_tables.py:21`; `tests/test_at_estimates.py`, `test_cd_estimation_criterion.py`, `test_estimation_specs.py`, `test_estimation_tables.py`; `notebooks/lifecycle_at_estimates.ipynb` | kept |
| `kikku.run.moments` | `make_moment_fn`, `moment_names`, `_age_group_masks` (private) | `estimate.py:50`; `at_estimates.py:27`; `tests/test_at_estimates.py:38` | kept; `_age_group_masks` private |
| `kikku.run.nest_io` | `save_nest` | `estimate.py:706, 943` | deleted |

### A.3 dynx (housing only; Phase 3)

| Module | Names | FUES files |
|---|---|---|
| `dynx.stagecraft.solmaker` | `Solution` | `src/dcsmm/models/housing_renting/horses_{c:54, h:42, t:3}.py`; `examples/housing_renting/whisperer.py:29`; `helpers/plots.py:29`; `helpers/interactive_plots.py:31` |
| `dynx.stagecraft` | `Stage` | `solve_single_model.py:28` |
| `dynx.stagecraft.makemod` | `initialize_model_Circuit`, `compile_all_stages` | `solve_single_model.py:29–32, 37`; `solve_runner.py:267` |
| `dynx.stagecraft.io` | `save_circuit`, `load_circuit`, `load_config` | `solve_single_model.py:33–34`; `solve_runner.py:266, 1510`; `helpers/metrics.py:218` |
| `dynx.heptapodx.core.api` | `initialize_model`, `generate_numerical_model` | `solve_single_model.py:35`; `whisperer.py:1247`; `solve_runner.py:439` |
| `dynx.heptapodx.num.generate` | `compile_num` | `solve_single_model.py:36` |
| `dynx.runner` and submodules | `CircuitRunner`, `write_design_matrix_csv`, `mpi_map`, `load_reference_model`, `get_metric_requirements`, `get_cached_reference_model`, `clear_reference_cache`, `register_baseline_model`, `clear_model_cache` | `solve_runner.py:265, 960, 1295, 1596, 1605`; `helpers/metrics.py:17–26` |
| `dynx_runner` (separate name) | `CircuitRunner` | `examples/housing_renting/model_factory.py:4` |

### A.4 Model sources to re-author

| Example | Files read by the dolo loader today |
|---|---|
| Retirement | 11 files under `examples/retirement/syntax/`: three stage YAML files under `stages/{labour_mkt_decision, retire_cons, work_cons}/`, their three `_methods.yml`, `spec_factory.yaml`, `period.yaml`, `nest.yaml`, `calibration.yaml`, `settings.yaml`. |
| Durables | Two registries under `examples/durables/syntax/{separable, cobb_douglas}/`, each with three stage YAML files and their `_methods.yml`, `spec_factory.yaml` (plus `spec_factory_males.yaml`), `period.yaml`, `nest.yaml`, `calibration.yaml` with `calibration/main.yaml` (and `main_males.yaml`), `settings.yaml` with `settings/default.yaml`, and `callables.py` (Python, kept). 95 files in the worktree including the estimation YAML files. |
| Housing (Phase 3) | Five `config_HR/*/master.yml` with `connections.yml` and `stages/{OWNC, OWNH, RNTC, RNTH, TENU}.yml` each; `income_process_fella2014*.yml`; `generate_income_process.py`. 38 files. |

## Appendix B — Citation check of the version-1 reviews

Thirty-six citations of the five review documents were checked on 28
September 2026 against FUES `main` at `9cafd85` (six commits after the
reviews' `6d7c3fe`; `git diff --stat 6d7c3fe HEAD -- examples/ src/dcsmm/
tests/` is empty, so no cited FUES line has moved), bellman-ddsl `main` at
`adec19f70` (43 commits after the reviews' `9c332302c`; the cited Bellman
files are unchanged), kikku at `a54619c`, and the estimation worktree at
`679b363`. Twenty-nine citations were confirmed, seven point at shifted
lines, none names a missing file. The seven:

| Document | Citation as written | Correct location | Note |
|---|---|---|---|
| `shared.md` | `asva.md:35` (deletion of `run/cli.py`) | line 32 | Line 35 is the deletion of `run/nest_io.py`. |
| `shared.md` | `kernels-as-functions.md:30` (explicit trailing parameters) | line 51 | Line 30 is the heading "3 The text". |
| `shared.md` | `kikku/run/nest_io.py:32` (serialises stages and one `inter_conn`) | lines 56 to 59 | |
| `implementation_and_estimation_review.md` | `keeper_egm.py:55` (fast path returns `cntn_data = None`) | lines 252 to 255 | Line 55 is the definition of `_make_keeper_fast`. |
| `implementation_and_estimation_review.md` | `solution_scheme.md:64` (array axes) | lines 70 to 72 | |
| `implementation_and_estimation_review.md` | `at_estimates.py:395` (fit function) | `fit_at_estimates` at line 432 of `679b363` | The function was rewritten after the review (112 changed lines); see Section 8 for what still stands. |
| `implementation_and_estimation_review.md` | estimation worktree `1ea95ebba2e6` | `679b363` | The branch advanced; the review is dated. |

The review documents themselves are not edited; this table is their erratum.
