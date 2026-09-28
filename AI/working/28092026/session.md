# Session 28 September 2026

Two streams of work, both on the branch `feat/cd-estimation-postprocess`
(worktree `FUES2026/FUES-wt-cd-estimation`, pushed to `origin`): the
Cobb-Douglas estimation with discount-factor types (code, ready to merge), and
the review of the Bellman-SYM upgrade specification (documents only). Nothing
from either stream is on `main` yet. Section 0 is the place to start next
session; Sections 1 to 4 are the record.

---

## 0 Start here next session

### 0.1 State of the repositories (12:20, 28 September)

| Repository | Branch | Commit | Notes |
|---|---|---|---|
| FUES `main` | `main` | `9cafd85` | Three commits since the branch last merged `main` (`97bc3f5`): the PBS agent's `7541b6c` (CHANGELOG: NFS import storm and jobfs staging), `5af809d` (manifest file name carries the job id), `9cafd85` (resubmit script record). Pushed. |
| FUES estimation branch | `feat/cd-estimation-postprocess` | `679b363` + the notes commit made at the end of this session | 14 commits over `main` (incl. one merge of `main` at `97bc3f5`). Pushed. **Not merged.** `main` at `9cafd85` is not an ancestor of the branch, so a fast-forward merge needs one more `git merge main` on the branch first (Decision E1). |
| kikku checkout | `main` | `a54619c` | Frozen. Upstream it is now `asva` inside bellman-ddsl. |
| bellman-ddsl | `main` | `adec19f70` | Internal developer release 0.01 being built: lanes A (asva) and T0 merged; R and N running; P, K, C, D, E, W not started. |

The `AI/` tree is ignored by git (`.gitignore` line 83) and `test_*.py` is
ignored (line 67). Everything under `AI/` that is on the branch was added with
`git add -f`; the six new test files likewise.

### 0.2 Decisions on the estimation branch (E1 to E9)

Each row: what has to be decided, the options, my recommendation, and what
waits on it. Nothing here is decided until you say so.

| # | Decision | Options | Recommendation | Waits on it |
|---|---|---|---|---|
| E1 | **Merge the branch into `main` and push.** | (a) Merge now: in the worktree `git merge main` (expect a small conflict in `CHANGELOG.md`, both sides added entries), run the suite, then `git checkout main && git merge --ff-only feat/cd-estimation-postprocess && git push`. (b) Wait. | (a). Nothing on `main` conflicts with the branch's code; the PBS agent's commits touch the resubmit script and CHANGELOG only. The Bellman-SYM spec (Q4) also asks for the merge before any durables baseline. | The Gadi Cobb-Douglas runs (they need the branch's specs and PBS scripts); the Bellman-SYM Phase 0 baseline. |
| E2 | **Which Gadi jobs to submit, in which order, and the budget.** Scripts: `benchmarks/durables/estimation/data-estimation/cobb_douglas/females/run_{xlarge,xxl}_{egm,negm}.pbs`, `.../cobb_douglas/males/run_xxl_{egm,negm}_males.pbs`, `.../cobb_douglas/females/run_large_egm_types.pbs`, `.../separable/females/run_large_egm_types.pbs`. The xlarge scripts request 2,080 cores, 5 h, `normalsr`; the types scripts 4,160 cores, 19,200 GB, 5 h. | (a) Small MPI job first to prove the updated environment (one xlarge EGM), then CD females EGM and NEGM, then males, then the types job. (b) Everything at once. | (a). The types path has run locally under MPICH only, never on Gadi; the first Gadi run of it should be the smallest available. Check `nci_account` for the tp66 balance before the 4,160-core job (5 h × 4,160 cores at the `normalsr` rate). | Paper tables for Cobb-Douglas; the types estimates. |
| E3 | **Age-group convention.** kikku's `_age_group_masks` uses `range(t_lo, t_hi)`, four of the five years of each data bin. Every estimate so far used it. | (a) Keep for the current runs; change and re-estimate everything before the paper's final tables. (b) Fix now (in asva/kikku, or by a FUES-side mask) and re-estimate now. | (a). Changing it changes every result and every comparison with the earlier separable estimates. | The final re-estimation. Recorded in `AI/todo.md`. |
| E4 | **`fit_table.csv` provenance.** kikku writes the last cross-entropy evaluation on rank 0, unweighted. The weighted fit at `theta_best` is `fit_table_at_best.csv` from `at_estimates.py` (reconciles with the objective to about 1e-9). | (a) Keep: run `at_estimates.py` on every finished run and feed `estimation_tables.py --fit`. (b) Have the driver re-solve at `theta_best` on the final segment and write the weighted table itself. | (a) now; (b) later if the extra re-solve is acceptable inside the job. | The fit tables in the paper. |
| E5 | **Self-generated (recovery) runs with types, and the data-side method fix.** Deferred by you today. The driver refuses `types` + `data_source: selfgen`; the data-side solve ignores `--methods-override` (NEGM recovery sweeps generated data with FUES). | (a) Schedule after the data runs, with the agreed `theta_true` rule (truths for `beta_bar`/`sigma_beta` expanded before the data solves). (b) Drop. | (a), after E2's runs finish. | Recovery evidence for the types estimator. |
| E6 | **Local serial runs with `mpi4py` installed** get a one-rank communicator and `n_samples = 1` unless `--n-samples` is passed. | (a) Treat a size-1 communicator as serial in the driver (use the spec's `n_samples`). (b) Leave; document. | (a); a few lines in `estimate.py`. | Nothing; convenience. |
| E7 | **`AI/CLAUDE.md`** still says kikku is installed editable from the local checkout at v0.2.0; the `.venv` and `pyproject.toml` pin `a54619c` from the remote, and the next dependency is asva. | (a) Update the paragraph now. (b) Update with the Bellman-SYM migration. | (a); one paragraph. | Nothing. |
| E8 | **`.gitignore` `test_*.py`** (also Q6 of the Bellman-SYM spec). Six new tests were force-added on this branch; the earlier branch did the same. | (a) Narrow the rule to the folders it was meant for. (b) Keep force-adding. | (a). | Nothing; hygiene. |
| E9 | **Two table tools.** `scripts/collect_estimation_results.py` (PBS agent, on `main`) walks every `est_*` folder and the PBS logs into one CSV/Markdown table with service units and exit status; `examples/durables/postprocess/estimation_tables.py` (branch) writes the paper's parameter table (best with standard errors) and per-run fit tables in Markdown and LaTeX. | (a) Keep both with the division stated in the docs: the collector is the run inventory, the tables tool is the paper's tables. (b) Fold one into the other. | (a). | Nothing; documentation. |

Settled today, no decision left: the Cobb-Douglas targets are the estimation
export (`moments_data.csv`, copied from separable; the male age-8 housing
value is 264,077, not the plotting export's 63,529); `moments.csv` deleted;
`theta` free in every Cobb-Douglas spec; equal type shares with a spread
parameter; fourteen Cobb-Douglas spec files fine; raw Gadi results committed
locally; the branch's PBS scripts follow the committed format.

### 0.3 Decisions on the Bellman-SYM upgrade specification (Q1 to Q12)

The full register with reasons is Section 11 of
`AI/devspecs/28092026/bellman_sym_upgrade_v2.md`. One line each:

| # | Decision | Recommendation |
|---|---|---|
| Q1 | Scope of the first upgrade: retirement and durables only; housing/renting in its own later specification; lift the "Do NOT edit" rule on the housing folders when that opens. | Phase housing out. |
| Q2 | `.bl` files declare the economics in Bellman's conventions; FUES arrays preserved through convention maps recorded in one migration register per example. | Adopt. |
| Q3 | Policy blocks and methods files wait for the upstream gate; the rest of the `.bl` text may be drafted now. | Adopt. |
| Q4 | Merge the estimation branch before the durables baseline. | Merge first (= E1). |
| Q5 | Retirement simulation: delete the ignored `tests/test_simulate_retirement.py` with `kikku.dynx`, or write a FUES retirement simulator. | Delete. |
| Q6 | Narrow `.gitignore` `test_*.py`. | Narrow (= E8). |
| Q7 | `_age_group_masks` (private in asva): copy into FUES or request a public export. | Copy. |
| Q8 | Correct the *preserved deviation* and *blocker* register rows before or after the upgrade. | After, one commit per row. |
| Q9 | Composition order: introduce the resolved-inputs record and the application settings source on `main` first (against the dolo stages) so the estimation code changes interface once, or merge the branch and change its readers in the migration. | Record first if the merge is not within the week; otherwise merge first. |
| Q10 | Install route (git pins with `#subdirectory=packages/bellman`, `packages/asva`; Gadi must reach `econ-ark/bellman`) and whether FUES follows the release to Python 3.13/3.14 or stays on 3.12. | Git pins; stay on 3.12 until the release's suite passes on it. |
| Q11 | Durables terminal timing the `.bl` declares: decisions at age T with terminal continuation (the solver) or ending at T−1 with terminal utility added directly (the simulator). The two disagree today. | Yours; the `.bl` must state one. |
| Q12 | Reclassify the borrowing limit `b` from a setting to a declared parameter; rewrite the "Settings vs Calibration" paragraph of `CLAUDE.md`. | Reclassify. |

Nothing in the upgrade starts before the gate of the spec's Section 3.2
except the Phase 0 list of its Section 4.1 (baselines, register, parser move,
simulator traversal, the three deleted imports, draft `.bl` text outside the
policy blocks). Re-check the upstream state before doing anything:

```bash
git -C ~/Research/Repos/bellman-ddsl log --oneline -5
sed -n 1,40p ~/Research/Repos/bellman-ddsl/AI/road-map/AI-todos/briefs-for-agents/internal-dev-0.01-release/lanes/log.md
ls ~/Research/Repos/bellman-ddsl/AI/road-map/AI-todos/briefs-for-agents/internal-dev-0.01-release/lanes/*.md   # a lane file appears when its lane has a build record
grep -rn "def build\b\|class Build\|def denote\|def kernels" ~/Research/Repos/bellman-ddsl/packages/bellman/bellman/stage/ | head   # lane N and E landmarks
```

### 0.4 Immediate next actions, in order

1. Decide E1; if (a): merge `main` into the branch, run the suite (Section
   0.5), fast-forward `main`, push.
2. On Gadi (never through the sshfs mount for git): `cd ~/dev/fues.dev/FUES
   && git pull` once by hand, then `source setup/setup.sh --update`, then the
   first job of E2.
3. When a Cobb-Douglas run finishes: `scripts/pull_gadi_results.sh
   --estimation`, then `at_estimates.py` and `estimation_tables.py` on it
   (commands in 0.5).
4. Answer Q1 to Q12, or the subset needed to open Phase 0 of the upgrade
   (Q1, Q2, Q4, Q9 are enough to start; Q11 and Q12 are needed before the
   durables `.bl` is written).
5. Update `AI/CLAUDE.md` (E7) and `.gitignore` (E8) if agreed.

### 0.5 Commands

The virtual environment lives in the main tree; the worktree has none.

```bash
# Tests on the branch (MPI tests use the MPICH mpiexec installed in the venv)
cd ~/Research/Repos/FUES2026/FUES-wt-cd-estimation
PYTHONPATH=$PWD:$PWD/src ../FUES/.venv/bin/python -m pytest -q tests/

# A bounded local estimate (serial; --n-samples needed when mpi4py is installed, see E6)
../FUES/.venv/bin/python -m examples.durables.estimate --mod cobb_douglas \
  --spec baseline.yaml --N-sim 2000 --n-samples 8 --n-elite 2 --max-iter 2 \
  --settings-override n_a=30 n_h=30 n_w=30 --local-results /tmp/fues-est

# The same with four types under MPI (size must be a multiple of K = 4)
../FUES/.venv/bin/mpiexec -n 8 ../FUES/.venv/bin/python -m mpi4py -m examples.durables.estimate \
  --mod separable --spec baseline_large_egm_types.yaml --N-sim 2000 \
  --n-samples 2 --n-elite 1 --max-iter 1 --settings-override n_a=30 n_h=30 n_w=30 \
  --local-results /tmp/fues-est-types

# Lifecycle profiles and the weighted fit at theta_best (refuses to write inside the Gadi checkout)
../FUES/.venv/bin/python -m examples.durables.postprocess.at_estimates \
  tests/fixtures/estimation_run/est_20260928_014054 --out /tmp/fues-at-estimates

# Parameter table across runs and a fit table per run (Markdown and LaTeX)
../FUES/.venv/bin/python -m examples.durables.postprocess.estimation_tables \
  --run fixture=tests/fixtures/estimation_run/est_20260928_014054 \
  --fit fixture=tests/fixtures/estimation_run/est_20260928_014054 --out /tmp/fues-est-tables

# Pull finished Gadi estimation folders to the local results tree
scripts/pull_gadi_results.sh --estimation
```

Failure-injection hook for the MPI protocol: `FUES_TYPES_FORCE_FAIL=rank<r>`,
`root`, `estimate` or `all` (a warning is printed at import when it is set).

---

## 1 What was done

1. **Dependency pins and `setup.sh`** (on `main`, commit `e14a6a7`, pushed).
   dolang pin moved from `92b63c4` (a February commit on `phase1.1_0.1`) to
   the fork `master` `97c3378`; kikku from tag `v0.2.0` to `main` `a54619c`
   (docs-only difference); dolo unchanged at `c899b01` (already master).
   `sympy` added to core dependencies: `econ-ark` 0.17.2 imports it without
   declaring it, and a fresh install failed at `import HARK`. `setup.sh
   --update` fixed: bash reads a sourced file before running it, so the
   reinstall lines after `git pull` used the pre-pull pins; the pull now
   re-sources the pulled copy, a failed pull aborts the update, and the
   installed commits are printed. Gadi needs one manual `git pull` before its
   first `--update` after this change.
   The FUES `.venv` was repaired (its `dolo.pth` pointed at a deleted
   directory) and moved to the same pins; kikku in the `.venv` is now the
   remote commit, not the editable local checkout.

2. **Devspec** `AI/devspecs/28092026/cd_estimation_and_postprocess.md`:
   Cobb-Douglas estimation, discount-factor types, lifecycle-at-estimates
   tool, estimation tables, pull-script flag. Two independent reviews
   (architecture; estimation correctness), both "revise", all findings folded
   in; owner decisions recorded in its Section 11. Self-generated (recovery)
   runs deferred by the owner.

3. **Implementation** on `feat/cd-estimation-postprocess` (plan
   `AI/prompts/28092026/cd_estimation_implementation_plan.md`), by parallel
   subagents in waves (cheaper models for routine parts and tests, every diff
   reviewed before commit). Fourteen commits over `main` at `679b363`,
   including one merge of `main` (`97bc3f5`) and a whole-branch review with
   eleven findings folded in. Suite before the merge of `main`: 157 passed, 6
   pre-existing skips (self-generated specs have no data file); the result of
   the final run at `679b363` is recorded in Section 4.

4. **Review of the Bellman-SYM upgrade specification** (Section 3), with
   version 2 of the specification written.

### 1.1 What is on the branch (file map)

| Area | Files | Purpose |
|---|---|---|
| Estimation driver | `examples/durables/estimate.py` | `build_criterion(mod_dir, spec, spec_factory, solver_method, calib_overrides, setting_overrides, N_sim, simulation_seed, comm, *, types_spec=None, group=None) -> (criterion, moment_fn, data_moments, denorm)`; `resolved_calibration_keys`; `check_parameter_names` (refuses unknown free names, mapped names not in `free`, targets that are free); `load_types_spec` (`n >= 2`); `check_types_data_source` (refuses types + selfgen); `types_summary`; manifest written into the results folder (`spec_factory`, `solver_method`, `git_commit`, `beta_types`, `max_iter`); every trial failure logged; exit 3 when all evaluations fail. |
| Layered MPI | `examples/durables/type_groups.py` | `check_divisibility` (`size % (n_points × K) == 0`, on every rank before any split); `split_type_groups` (`group_comm = comm.Split(rank // K)`, `roots_comm` over group roots); `make_group_trial`; `worker_loop`; the EVAL/PART/FAIL/STOP protocol; the root computes member 0 inside try/except, always gathers, raises after the gather; STOP in `finally`; workers return `(None, None, all_failed)`. Hook `FUES_TYPES_FORCE_FAIL`. |
| Types | `examples/durables/solvers/beta_types.py` | `discretise_types(location, spread, n, transform='logit'|'log'|'identity')` (K equiprobable normal quantiles, equal shares, algebraic form `b/(b+(1-b)e^{σz})`, exact at zero spread); `discretise_beta`; `expand_types(theta, types_spec)` (K calibrations; comonotonic members across parameters); `draw_types(N, shares, seed)` (birth draw with `SeedSequence(seed, spawn_key=(1,))`); `pool_by_type(parts, type_idx, N, member_values) -> type_idx, beta arrays in original agent order`. |
| Simulator | `examples/durables/solvers/simulate.py` | `_take_agents`; `simulate_type_subset(nest, grids, agent_idx, N, seed, ...)`; `simulate_lifecycle(..., types=None)`; the no-types path routes through the subset path with the full index set and is bit-identical to the old code (sha256 digests in `tests/test_beta_types.py`). |
| Post-estimation | `examples/durables/postprocess/at_estimates.py` | `load_run`, `types_spec_for_run` (from the run's `beta_types` record, cross-checked with the spec file), `solve_at_estimates -> [(share, nest, grids)]`, `simulate_at_estimates`, `data_points_by_age`, `fit_at_estimates` (penalised NaN moments scored as `NAN_PENALTY`), `plot_lifecycle_at_estimates`, `write_outputs`; CLI with the Gadi guard (`/scratch/tp66` exists → `--out` required). |
| Tables | `examples/durables/postprocess/estimation_tables.py` | `PARAMETER_ORDER`, `LATEX_SYMBOLS`, `read_run`, `parameter_table` (best (se), four decimals for `beta`/`beta_bar`, three otherwise), `fit_table` (reads `fit_table_at_best.csv` when present, else `fit_table.csv` with a caption), `write_estimation_tables(runs, out_dir, fits=())`; exact values in CSV, rounded for display. |
| Notebook | `examples/durables/notebooks/lifecycle_at_estimates.ipynb` (+ `docs/notebooks/` symlink, `mkdocs.yml` nav) | Same functions as the tool. |
| Specs | `examples/durables/syntax/cobb_douglas/estimation/*.yaml` (9 baselines), `moments_data.csv`, `calibration/main_males.yaml`, `spec_factory_males.yaml`; `separable/estimation/baseline_large_egm_types.yaml` | Cobb-Douglas on the separable targets; `theta` free; males overlay; K = 4 types specs in both registries. `moments.csv` deleted. |
| PBS | `benchmarks/durables/estimation/data-estimation/cobb_douglas/{females,males}/*.pbs`, `separable/females/run_large_egm_types.pbs` | Regenerated from the final separable scripts (environment-overridable defaults, repo-root walk-up). |
| Scripts, docs | `scripts/pull_gadi_results.sh --estimation`; `examples/durables/docs/estimation_outputs.md`; `examples/durables/README.md`; `docs/running-on-gadi.md`; `benchmarks/durables/estimation/README.md`; `CHANGELOG.md` (Unreleased) | |
| Tests (force-added) | `tests/test_estimation_specs.py`, `test_cd_estimation_criterion.py`, `test_beta_types.py`, `test_type_groups_mpi.py`, `test_at_estimates.py`, `test_estimation_tables.py`; fixtures `tests/fixtures/estimation_run/est_20260928_014054/`, `tests/fixtures/estimation_run_types/est_20260928_022152/` | |

### 1.2 How the parameter-draws design generalises

The `types` block is not specific to `beta`: `types: {n, parameters: {name:
{location, spread, transform}}}`. Member k takes the k-th equiprobable
quantile of every listed parameter (comonotonic members). Adding
heterogeneity in another parameter is one more entry under `parameters`; the
MPI layout (one rank per member within a group of K, one group per candidate)
and the pooling are unchanged. Per-rank solve time within a group varied by at
most a factor 1.04 at the 30³ grid (8 ranks, K = 2) and did not visibly depend
on `beta`.

## 2 Findings worth remembering

- The old Cobb-Douglas estimation spec listed `gamma_c` as free; the
  Cobb-Douglas model has `rho`, and `solve()` ignored the unknown override
  silently. The driver now refuses unknown names at start-up.
- `save_nest` fails on dolo stage objects (`mappingproxy` pickle) after
  opening the file, so every `best.nst` on Gadi is a zero-byte file. Upstream
  asva has deleted `nest_io` altogether; the two `save_nest` calls in
  `estimate.py` (lines 706 and 943) go with the Bellman-SYM migration.
- `fit_table.csv` is kikku's last evaluation on rank 0, not the evaluation at
  `theta_best`, and its `contribution` is unweighted. `fit_table_at_best.csv`
  from the lifecycle tool is the weighted fit at the estimates; on the saved
  test run it reconciles with the reported objective to 8.5e-10.
- The self-generated data solve ignores `--methods-override`: the existing
  NEGM recovery sweeps generated data with FUES. Not changed (selfgen
  deferred).
- kikku's age-group masks use four of the five years of each data bin.
- Tauchen at zero persistence fixes the type shares (2.3/47.7/47.7/2.3
  percent for K = 4); the types use K equiprobable quantiles with equal
  shares instead, in the algebraic form `b/(b+(1-b)e^{sz})` (exact at zero
  spread).
- `make_initial_particles` seeds `default_rng(seed + 1)`; a second stream on
  the same seed ties agent i's type to agent 2i+1's income state (numpy draws
  bounded integers from 32-bit halves of the same words). The birth draw uses
  `SeedSequence(seed, spawn_key=(1,))`.
- With `mpi4py` installed, a plain `python -m examples.durables.estimate`
  run has a one-rank communicator and one candidate per iteration unless
  `--n-samples` is given (pre-existing rule `n_samples = comm size`).
- The `/usr/local` Open MPI is x86_64 only; local MPI tests use MPICH 5.0.1
  installed into the venv (`pip install mpich`, launcher `.venv/bin/mpiexec`).
- `cobb_douglas/estimation/moments.csv` was the Eggsandbaskets plotting
  export; `moments_data.csv` (both registries) is the estimation export. The
  two differ materially in one male target (housing wealth, age group 8:
  63,529 vs 264,077); the owner settled on the estimation export.

## 3 Review of the Bellman-SYM upgrade specification (later in the day)

Reviewed `AI/devspecs/28092026/bellman_sym_upgrade.md` with its five
supporting reviews and the four upstream briefs, under the `akshay-ddsl-check`
reasoning profile, with four delegated read-only checks (Bellman/asva state;
FUES dependency inventory; citation and register check of the review
documents; architecture review by `bellman-architect`). Verdict on version 1:
revise before implementation. Wrote
`AI/devspecs/28092026/bellman_sym_upgrade_v2.md` (version 1 kept unchanged;
its Section 12 lists every change and why).

Facts established on 28 September: bellman-ddsl `main` at `adec19f70`; lanes
A (asva) and T0 merged, R and N running, P, K, C, D, E, W not started; the
lane-N paths (`stage.build`, `stage.par`, item access on `settings`,
`denote()`, `diagram()`), `kernel().python` and `builder.kernels()` absent;
`Solution` and `Inversion` still defined. `packages/asva` exists with
`asva.run` and `asva.simulate` (`simulate` and `simulate_period` raise;
`StageForward`, `BranchingForward`, `forward_stage`, `apply_rename`,
`draw_shocks` kept; `_slice`, `_scatter`, `_scatter_rec` deleted).
`RunSpec` in asva has no deprecated aliases. FUES casualties of the rename:
`kikku.run.parse_cli` (both `run.py`), `kikku.run.nest_io.save_nest`
(`estimate.py` ×2), `kikku.asva.simulate.simulate` (durables simulator),
`tests/test_kikku.py`, `tests/test_method_override.py`,
`tests/test_simulate_retirement.py`. `dolang` is imported by no FUES file.
`dynx` (housing) is not installed and declared nowhere. Citation check of the
five review documents: 36 checked, 29 confirmed, 7 with shifted lines, none
missing (Appendix B of version 2).

Main changes in version 2: an explicit upstream gate with the acceptance items
it requires and the pin form (`#subdirectory=packages/...`); housing/renting
moved to a later phase with its own specification; the rule that `.bl` files
declare the economics in Bellman's conventions and FUES arrays are preserved
through convention maps recorded in one migration register per example (with
statuses faithful / preserved deviation / blocker / corrected); the
resolved-inputs record as the one carrier of model inputs (replacing the
`RetirementModel` fall-through proxy and the stage objects kept in results);
an application settings source for `T`, `t0`, `N_sim`, seeds; ASCII parameter
names equal to today's YAML keys; `b` reclassified from setting to declared
parameter; explicit binding passes for durables (wage schedule written between
loading and binding); per-phase acceptance checks each with an oracle and a
tolerance; twelve owner decisions Q1 to Q12 (Section 0.3 above).

Two things noticed outside the task: `AI/CLAUDE.md` names
`AI/design-principles.md`, which does not exist; and the implementation
review's point about `fit_at_estimates` is mostly superseded on the branch,
with one residue carried into the spec's check E3 (the reconstruction scores
only keys present in both dictionaries, which equals the criterion's rule only
because the moment function emits every key of the moment list).

## 4 Suite state at the end of the session

At `679b363`, run with the main tree's `.venv` and the worktree on
`PYTHONPATH` (the command in 0.5): **223 passed, 8 skipped, 1 warning, 16
subtests passed in 6 min 7 s**. The count is higher than the 157 recorded
before the merge of `main` because that merge brought the PBS agent's tests
in. All eight skips are one case, `tests/test_estimation_specs.py:57: no
data file`: the self-generated specs, which have no data file (six before
this branch, eight with the Cobb-Douglas registry's two).

## Gadi, after merge

```bash
cd ~/dev/fues.dev/FUES && git pull          # once by hand: the old setup.sh runs first
source setup/setup.sh --update
qsub benchmarks/durables/estimation/data-estimation/cobb_douglas/females/run_xlarge_egm.pbs
qsub benchmarks/durables/estimation/data-estimation/separable/females/run_large_egm_types.pbs   # 4160 cores, after E2
```

## Open (all in Section 0)

- E1 to E9 (estimation branch) and Q1 to Q12 (Bellman-SYM upgrade).
- Recorded and not fixed, in `AI/todo.md` ("Estimation: recorded on 28 Sep
  2026"): selfgen data-side method; selfgen with types; age-group convention;
  `fit_table.csv` provenance; size-1 communicator; `AI/CLAUDE.md` kikku note.
