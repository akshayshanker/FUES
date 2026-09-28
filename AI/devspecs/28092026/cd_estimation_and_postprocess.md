# Devspec: Cobb-Douglas estimation, discount-factor types, and post-estimation tools

Date: 28 September 2026. Status: revised after two independent reviews
(architecture, estimation correctness; both "revise", all findings folded in
or listed in Section 11); awaiting the owner's approval before the
implementation plan is written.
Branch: `feat/cd-estimation-postprocess`, developed in a separate git worktree
(`FUES2026/FUES-wt-cd-estimation`) so that the main working tree, where another
agent is editing the PBS scripts, is not touched.

## 1. Purpose

Four things the durables example cannot do today, and this work adds:

1. Repeat the SMM estimation for the Cobb-Douglas utility registry
   (`examples/durables/syntax/cobb_douglas`) on the same data, targets, sample
   sizes and seeds as the separable registry, for females and males.
2. Estimate with permanent heterogeneity in the discount factor: K types of
   `beta`, drawn once per agent from a distribution with a mean and a standard
   deviation that are free parameters, each type solved on its own MPI rank
   inside a group that evaluates one candidate parameter vector (the
   Eggsandbaskets layering), and the K simulated sub-populations pooled before
   the moments are computed. Available to both registries.
3. Draw lifecycle profiles (consumption, housing, financial assets, adjustment
   rate by age) at the estimated parameters of any finished run, with the data
   moments by age group drawn on top, from a notebook or the command line.
4. Turn the JSON and CSV files an estimation writes into tables for the docs and
   the paper: a parameter table across runs and a moment-fit table per run.

A fifth, small item makes these usable: a switch on
`scripts/pull_gadi_results.sh` that copies the estimation result files from
Gadi to this machine.

## 2. Facts this spec rests on (all verified on 28 Sep 2026)

Each fact below was checked by reading the file or running the command named.

- The estimation driver `examples/durables/estimate.py` selects the registry
  with `--mod`, the spec with `--spec`, and the calibration chain with
  `--spec-factory`. It has no code specific to the separable registry. Results
  go to `<results_dir>/<mod_name>/<spec_name>/est_<run_id>/`.
- The Cobb-Douglas registry received the same grammar migrations as the
  separable one (commits `57a77c1`, 21 Jun 2026, and `2dd7026`, 22 Aug 2026).
  At the current dolo `c899b01`, dolang `97c3378` and kikku `a54619c` it solves,
  simulates, and gives all 70 target moments without NaN at `t0=20`
  (`/tmp/fues-verify/cd_dryrun.py`).
- Its calibration (`cobb_douglas/calibration/main.yaml`) has `rho`, `alpha`,
  `beta`, `tau`, `theta`, `d_ubar`; it has no `gamma_c` or `gamma_h`. Its
  estimation spec `cobb_douglas/estimation/baseline.yaml` lists `gamma_c` as a
  free parameter. Passing `gamma_c` as a calibration override changes nothing:
  simulated moments at `gamma_c` 3.0, 5.0 and absent are identical to the last
  digit, while `rho` 2.0 against 3.0 changes them by up to 0.41
  (`/tmp/fues-verify/cd_probe2.py`). No error is raised.
- The Cobb-Douglas spec is otherwise older in form than the separable one: a
  single `seed` key (kikku still accepts it as a fallback), `init` values, a
  `theta_true` block, `n_samples: 48`, `tol: 1e-3`, `results_dir` relative to
  the repo, and targets on `fin_assets` columns. The separable specs target
  `a_tot` columns, use `sampling_seed`/`simulation_seed`, `tol: 1e-6` for the
  large and xlarge variants, and write to `/g/data/tp66/results/...`.
- The two `moments_data.csv` files hold the same data: nine age-group rows,
  and the Cobb-Douglas copy (139 columns) equals the separable copy cell for
  cell on every shared column; the separable copy has four extra columns
  (`av_a_tot_14_0/1`, `sd_a_tot_14_0/1`) that its specs target. Both descend
  from the Eggsandbaskets estimation export
  `AI/temp/eggs-bash-minimal-latest/settings/Eggs3.0_dev_ultra_lite/moments_data.csv`.
  Cobb-Douglas also carries `moments.csv`, a copy of the plotting export of
  the same vintage (`moments_data_plot.csv` there) that no spec reads: 756
  cells differ only by a missing leading zero, 122 by last-digit rounding, 46
  are `0` where the estimation export has `NA` (columns durables never uses),
  and 3 values differ materially. One of the three is a male target,
  `av_wealth_real_14_1` at age group 8: 63,529 in the plotting export against
  264,077 in the estimation export. The male estimations run so far used
  264,077, and the owner has decided (28 Sep 2026) that the estimation export,
  the file on Gadi, is the data; the plotting export is deleted.
- The separable EGM and NEGM spec files are identical once comments are
  stripped; the method is chosen by the PBS script's `METHODS` line
  (`--methods-override adjuster_cons.upper_env.upper_envelope=NEGM`).
- The separable male variant is `calibration/main_males.yaml` (only `phi_w`,
  `sigma_w`, `lambdas`) plus `spec_factory_males.yaml` (chain
  `[calibration/main, calibration/main_males, $draw]`). Both are
  utility-independent. The two registries' `spec_factory.yaml` files differ
  only in their header comment.
- Per-solve cost is the same in both registries: 4.9 s against 4.8 s at 100³
  over 50 periods; 17 s at 300³; 61 s at 600³ (the estimation grid), plus 1.6 s
  to simulate 10,000 agents, all single-threaded on this Mac.
- `kikku.run.nest_io.save_nest` raises `TypeError: cannot pickle
  'mappingproxy' object` on a solved nest after opening the output file, so
  `estimate.py` (which catches the error and prints a warning) leaves a
  zero-byte `best.nst` that `load_nest` rejects with `EOFError`; the same
  holds for `true.nst`. Results folders on Gadi therefore contain empty
  `.nst` files, not usable ones (verified on the check-L reference run). The notebook
  snippet in `examples/durables/docs/estimation_outputs.md` that loads
  `best.nst` cannot work. This is the open item "Estimation output integrity"
  in `AI/todo.md`.
- `manifest.json` (grid, `N_sim`, seeds, free parameters) is written to
  scratch only; `summary.json` in the results directory records the estimates,
  loss, convergence and `calib_overrides` but not the grid or the spec factory.
- `plot_lifecycle(sim_data, euler, nest, output_dir)` in
  `examples/durables/postprocess/plots.py` draws the 2×2 lifecycle figure; the
  `euler` argument is unused. `generate_cohort_table(sim_data, t0, T, norm)` in
  `tables.py` computes cohort means and standard deviations.
  `_md_table(rows, cols, caption, params)` and `_tex_table(...)` in `tables.py`
  write the committed paper tables.
- `examples/durables/run.py` hard-codes `base_spec="examples/durables/syntax/separable"`
  and cannot select a spec factory; kikku's `parse_cli` has no model-directory
  flag. The command line therefore cannot run Cobb-Douglas or the male
  calibration.
- The moment keys in `fit_table.csv` and in `load_estimation_spec(...)["data_moments"]`
  are `<csv column>__age<g>` (kikku `_flatten_moments_csv`).
- `.gitignore` blankets `test_*.py`; new tests need `git add -f`. Notebooks are
  jupytext-paired (`formats = "ipynb,myst"`) and the `.md` pair is ignored.
- kikku's `estimate()` distributes candidates over the ranks of the one
  communicator it is given (`mpi_map`: `scatter_items`, evaluate,
  `gather_results`) and treats the trial as opaque; it contains no
  inner-communicator or group logic, and `kikku.run.mpi` has seven helpers and
  no split. The durables driver already performs one `Split` (by sweep point)
  and hands the sub-communicator to `estimate()`. `_safe_criterion` in kikku
  turns an exception inside the criterion into the penalty loss.
- Eggsandbaskets (`AI/temp/Eggsandbaskets3.0/eggsandbaskets/lifecycle_model.py:559-568`)
  models the discount factor as `x = ln(1/beta - 1)` following a
  `quantecon.tauchen(n, rho_beta, sigma_beta, mu=...)` chain with mean
  `ln(1/beta_bar - 1)`, mapped back by `beta = 1/(1 + exp(x))`, and draws each
  simulated agent's `beta` index at the first period
  (`tseries_generator.py:276`). Its two-layer communicator split is
  `smm.py:76 gen_communicators`.
- `quantecon` is in the `[examples]` extra; the project venv has 0.7.2 and a
  fresh install today gets 0.12.0; both expose `tauchen(n, rho, sigma, mu=0.0,
  n_std=3)`. At `rho = 0` its shares are fixed by `n` and `n_std` alone
  (K = 4: 2.3, 47.7, 47.7, 2.3 percent for any `sigma`), verified in the venv.
  `scipy.stats.norm` is available through the core `scipy` dependency. The
  Open MPI 5.0.7 at `/usr/local/bin` on this Mac is x86_64 only and cannot
  launch the venv's arm64 Python with `mpi4py`; local MPI tests therefore use
  MPICH 5.0.1 installed into the venv from PyPI (`pip install mpich`,
  launcher `.venv/bin/mpiexec`) with `mpi4py` 4.1.2, verified on two ranks
  on 28 Sep 2026. Gadi keeps Open MPI; the layering code uses only standard
  MPI calls.
- `make_initial_particles` seeds its own generator with `seed + 1`
  (`solvers/simulate.py:423`) and draws the initial income state from it.
- `fit_table.csv` is not evaluated at `theta_best`: `diagnostics()` uses
  `result.sim_moments`, which kikku sets from the criterion's last successful
  evaluation on its rank 0 (the last candidate of the final iteration in
  serial, rank 0's candidate in MPI). `objective` in `summary.json` is at
  `theta_best`, so the two files are not mutually consistent; the existing
  `estimation_outputs.md:100-104` already says so.
- The `contribution` column of the estimation's `fit_table.csv` is the
  unweighted squared residual: `estimate.py` calls kikku's `diagnostics()`
  without the weights the loss applies (`1/data^2` when `|data| >= 1`), so
  the file's `contribution_pct` is not the share of the reported loss (found
  by Task C, 28 Sep 2026; 42 of 70 rows differ). Both new tools recompute the
  weighted contribution.
- The self-generated data solve in `estimate.py:177-186` passes no
  `method_switch`, while the trial solve does; with `--methods-override
  ...=NEGM` the data are generated with the YAML default (FUES) and every
  trial uses NEGM, so the existing NEGM recovery sweeps measure recovery
  plus the FUES-NEGM discrepancy.
- kikku's `make_criterion` closure catches any exception from the trial and
  returns the penalty loss, printing only the first three per process; a run
  in which every evaluation fails reports `converged = True`, objective
  `1e10`, and the driver exits 0 (probe by the correctness reviewer).
- kikku's `_age_group_masks` uses `range(t_lo, t_hi)`, so the age group
  `[40, 44]` covers rows 40 to 43, four of the five years of the data bin;
  `generate_cohort_table` in `tables.py` uses five-row bins. Two conventions
  coexist.
- kikku clamps `n_elite` to `n_samples` (`estimate.py:386`), so a small test
  needs no `n_elite` override for correctness; the manifest records the
  unclamped YAML value.
- `load_syntax` reads the top-level `calibration.yaml`; `solve()` resolves
  `calibration/main` (and the male overlay) through the spec factory. The two
  files are identical today in both registries, by coincidence, not by rule.
- `simulate_lifecycle` draws its shocks from `draw_shocks(..., rng_seed=seed)`
  and its initial conditions from `make_initial_particles(..., seed=seed)`;
  the panels it returns are `a, h, c, y, z_idx, discrete, a_nxt, h_nxt` of
  shape `(T, N)` plus `(N,)` utility statistics, and `sim_data["solutions"]`
  is not part of them.

## 3. Decisions already taken by the owner

- Numerical settings stay separate per registry. `settings.yaml`,
  `settings/default.yaml` and `calibration/main.yaml` are not edited in either
  registry. The differences are listed in Appendix A for the record only.
- kikku is not being worked on. `save_nest` is not fixed; the lifecycle tool
  re-solves at the estimates instead (one minute at the estimation grid).
- The PBS scripts under `benchmarks/durables/estimation/` are owned by another
  agent and are not edited here. Section 9 records what this work needs from
  them.
- Work happens on a branch in a separate worktree; `main` keeps working
  throughout.
- Discount-factor heterogeneity is part of this spec (not a follow-on). It is
  parameterised as in Eggsandbaskets: a mean `beta_bar` and a standard
  deviation `sigma_beta` of `x = ln(1/beta - 1)`, discretised with
  `quantecon.tauchen`, mapped back by `beta = 1/(1 + exp(x))`. Persistence is
  removed (`rho = 0`): a type is drawn once per agent and kept for life.
- Each candidate parameter vector is evaluated by a group of K MPI ranks, one
  type per rank (the layered communicator), not by K solves in sequence on one
  rank.
- In the simulation every agent draws its own beta type at the start of life
  as an initial condition, with the same shares as the discretised
  distribution; agents are not allotted to fixed sub-population sizes.

### 3.1 Everything that runs today keeps running unchanged

This is a hard requirement, checked by tests (Section 7, checks F, H and L):

- An estimation spec without a `types` block takes exactly today's code path
  in `estimate.py`: no communicator split, `n_samples` equal to the
  communicator size, the same candidate draws for the same seeds, the same
  output files. The existing PBS scripts and the existing separable specs are
  therefore unaffected.
- `simulate_lifecycle(nest, grids, N, seed)` without the new argument returns
  arrays bit-identical to today's.
- `examples/durables/run.py`, the retirement example, and the four existing
  notebooks are not modified; the two durables notebooks are executed end to
  end on the branch as part of acceptance.
- No file under `src/dcsmm` changes.

## 4. Scope

In scope: the files listed in Section 6, their tests, and the documentation
that describes them. Out of scope: PBS scripts; kikku; `settings.yaml` and
`calibration/main.yaml` in both registries; `examples/housing_renting`;
fixing `save_nest`; any change to the existing separable estimation specs
(a new types spec is added beside them, Section 5.6); a general
`--model-dir` option for `run.py` (the notebook route covers what the command
line cannot, and adding CLI surface is a separate decision); heterogeneity in
any parameter other than `beta`; a persistent (AR(1)) beta process; and, by
the owner's decision of 28 Sep 2026, everything to do with self-generated
(recovery) runs: no Cobb-Douglas `selfgen_*` specs, no recovery table, no
types under `data_source: selfgen`, and no change to the existing
self-generated data path in `estimate.py` (its known defects are recorded in
Section 2 and `AI/todo.md` for a later spec).

## 5. Design

### 5.1 Component A: Cobb-Douglas estimation files

Mirror the separable estimation folder name for name, substituting the free
parameter block. Each Cobb-Douglas file is the separable file with:

- `gamma_c` and `gamma_h` removed and `rho` added with bounds `[1.2, 6.0]`
  (the bounds the separable specs use for `gamma_c`, and the range the
  stale Cobb-Douglas spec already used; `rho` plays the same role, the
  curvature on the composite good);
- `beta`, `alpha`, `tau` bounds unchanged; `theta` free in every
  Cobb-Douglas spec with bounds `[0.5, 3.0]` (owner's decision, 28 Sep 2026:
  the Cobb-Douglas calibration notes that the separable value 1.3498 needs
  recalibration for this utility, so it is estimated rather than fixed; the
  separable specs are unchanged and keep `theta` fixed outside
  `baseline.yaml`);
- targets, `data_file`, age groups, `method_options`, paths and save options
  copied verbatim;
- the header comment naming the registry.

`alpha` is a Cobb-Douglas share here, not the separable weight on
non-durable utility; the bounds `[0.3, 0.95]` are kept because they are
admissible for a share, and the spec header says so.

Files created under `examples/durables/syntax/cobb_douglas/`:

| File | From separable | Change |
|---|---|---|
| `estimation/baseline.yaml` (replaces the stale file) | `baseline.yaml` | free block; header |
| `estimation/baseline_large_egm.yaml`, `_negm.yaml` | same names | free block; header |
| `estimation/baseline_large_egm_males.yaml`, `_negm_males.yaml` | same names | free block; header |
| `estimation/baseline_xlarge_egm.yaml`, `_negm.yaml`, `_egm_males.yaml` | same names | free block; header |
| `estimation/moments_data.csv` (replaces) | copy of separable file | adds the four `a_tot` columns; every other cell is already identical |
| `calibration/main_males.yaml` | copy | header comment |
| `spec_factory_males.yaml` | copy | header comment |

The files are generated by a throwaway script kept outside the repository
(`/tmp`), so the substitution is mechanical and identical across the eight
YAMLs (the six separable `selfgen_*` specs are not mirrored: self-generated
runs are deferred, Section 4); the generated files are what gets committed
and reviewed.

Decided by the owner (28 Sep 2026): `cobb_douglas/estimation/moments.csv`,
the unused copy of the plotting export (Section 2), is deleted.

### 5.2 Component A2: two small changes to `examples/durables/estimate.py`

1. Write `manifest.json` to the results directory as well as to scratch, and
   add two fields: `spec_factory` (the `--spec-factory` value) and
   `solver_method` (the upper-envelope tag from `--methods-override`, or
   `null`). Scratch is purged on Gadi; without this the grid, `N_sim`, seeds and
   spec factory of a finished run are lost, and the lifecycle tool has to be
   told them by hand.
2. Add `--n-samples` (applied only when there is no MPI communicator; in MPI
   runs `n_samples` is already forced to the communicator size) and
   `--n-elite` (applied in both modes, so that a small MPI test can keep
   `n_elite` at or below its candidate count). Purpose: a serial end-to-end
   short end-to-end run of the real driver on a small grid, which also produces a genuine
   results folder for the tests of Components B and C, and the MPI equivalence
   test J. PBS scripts do not use these flags.

3. A start-up guard at the existing `load_syntax` call: every name in
   `calib_overrides` (from `--params-override` and sweep points) and every
   name in `free:` must be a key of the calibration the solver will use (the
   spec factory's resolved chain, so the male overlay counts), except the
   names a `types` block maps, which must not be; otherwise the driver stops
   and lists the offending names. Today `solve()` accepts unknown calibration
   keys silently, which is how the Cobb-Douglas spec estimated a parameter
   the model does not have; the static check A catches stale spec files, this
   guard catches a command-line or sweep-key mistake at run time.
4. `build_criterion(mod_dir, spec, spec_factory, solver_method,
   calib_overrides, setting_overrides, N_sim, simulation_seed, comm)` is
   factored out of `_run_single_estimation`, which then calls it. The
   behaviour is unchanged (check L); the purpose is that tests B and I call
   the driver's own criterion rather than rebuilding a copy of the pipeline.
5. Failures are never silent and never counted as success. kikku's criterion
   closure turns any exception into the penalty loss and prints only the
   first three per process; the driver therefore logs every failure itself
   (rank, candidate, message, flushed) before the exception reaches kikku,
   counts them on the criterion, prints the total after `estimate()`, and
   exits non-zero when `result.objective` is the penalty value (every
   evaluation failed) even though kikku reports convergence.
For a spec without a `types` block nothing else in the driver changes, and
its outputs are the same files plus `manifest.json` in the results folder;
the additions for types are in Section 5.6.

### 5.3 Component B: lifecycle profiles at the estimates

New module `examples/durables/postprocess/at_estimates.py`. Pure functions,
disk and plotting at the boundary, following the existing `postprocess/`
layout:

```python
def load_run(run_dir) -> dict
    # Reads summary.json (required), theta_*.json, manifest.json (optional).
    # Returns: theta_best, theta_mean, theta_se, objective, converged, n_iter,
    # calib_overrides, registry (from manifest["mod"] basename, else the
    # results layout <mod>/<spec>/est_<id>), spec_name, spec_factory
    # (manifest, else "spec_factory_males.yaml" when spec_name ends in
    # "_males", else "spec_factory.yaml"), grid (manifest or None), N_sim,
    # simulation_seed, solver_method (manifest or None), beta_types (the
    # block from summary.json when the run used types, else None).

def solve_at_estimates(run, registry_root, grid=None, method_switch=None) -> list[(share, nest, grids)]
    # Without types: one entry, share 1.0, from
    # solve(str(registry_root / run["registry"]),
    #       spec_factory_name=run["spec_factory"], method_switch=...,
    #       draw={"calibration": {**run["calib_overrides"], **run["theta_best"]},
    #             "settings": grid or run["grid"]}, verbose=False, strip_solved=True).
    # With types: expand_types(run["theta_best"], run["beta_types"]) and one
    # solve per node. Raises ValueError when no grid is known (no manifest and
    # none given).

def simulate_at_estimates(run, solved, N=None, seed=None) -> sim_data
    # One entry: simulate_lifecycle(nest, grids, N=N or run N_sim, seed=...).
    # Several: simulate_lifecycle(nest_0, grids_0, ..., beta_types=(nodes,
    # shares, nests, grids)), the same birth draw as the estimation (Section 5.6).

def data_points_by_age(registry_dir, spec_name) -> dict[str, list[tuple[float, float]]]
    # From load_estimation_spec: for each target with stat == "mean" and var in
    # {c, a, h}, pairs (age position, value in AUD) for every age group
    # present in data_moments. The age position is the mean of the rows kikku's
    # _age_group_masks assigns to the group, so data and model use one rule.

def simulated_moments_at_estimates(sim_data, moment_spec, sett) -> dict[str, float]
    # kikku.run.moments.make_moment_fn(moment_spec) applied to the pooled
    # panels, then the same level-prefix denormalisation as estimate.py
    # (times 1/normalisation for means, SDs and conditional means). The
    # estimation's own engine, so the numbers are on the fit table's footing.

def fit_at_estimates(run, sim_moments, data_moments) -> list[dict]
    # Rows moment, data, simulated, residual, contribution, contribution_pct,
    # with contribution = w (sim - data)^2 and w = 1/data^2 when |data| >= 1
    # else 1 (kikku's rule). Written as fit_table_at_best.csv. When the grid,
    # t0, N_sim, seed, spec factory and method match the run, the contributions
    # sum to summary.json["objective"]; the tool prints both numbers.

def plot_lifecycle_at_estimates(sim_data, nest, data_points, out_dir) -> Path
    # The 2x2 figure of plot_lifecycle (mean, 25-75 band), in AUD, with the
    # data points drawn as markers on the c, h and a panels; by-type means
    # as thin lines when types are present.
```

Command line: `python -m examples.durables.postprocess.at_estimates RUN_DIR
[--grid 600] [--n-sim 10000] [--seed 99] [--out DIR]`. Output:
`<out>/lifecycle_at_estimates.png`, `<out>/fit_table_at_best.csv` and
`<out>/cohort_means.csv` (age group, variable, data, simulated).

Why the tool recomputes the fit: the `fit_table.csv` an estimation writes is
kikku's last evaluation on its rank 0, not the evaluation at `theta_best`
(Section 2), so it does not reconcile with the reported loss. The re-solve at
the estimates is the only place the fit at `theta_best` exists.

Notebook `examples/durables/notebooks/lifecycle_at_estimates.ipynb`: one
parameters cell (`RUN_DIR`, `GRID`, `N_SIM`), then calls to the functions
above, then the figure and the cohort table. Symlinked from `docs/notebooks/`
and added to the `mkdocs.yml` notebook list, like the existing four.

### 5.4 Component C: estimation tables

New module `examples/durables/postprocess/estimation_tables.py`:

```python
def parameter_table(runs: list[tuple[str, dict]], fmt="md") -> str
    # Rows: parameters in the fixed order beta, beta_bar, sigma_beta, alpha,
    # gamma_c, gamma_h, rho, tau, theta (only those present in at least one
    # run; any other free parameter follows in alphabetical order). One column
    # per run label. Cell: "best (se)" with 3 decimals (4 for beta, beta_bar).
    # Footer rows: loss, converged, iterations, n_samples, grid (from manifest
    # when present, else "n/a"), and for a types run the implied beta nodes
    # and shares at theta_best.

def fit_table(run_dir, fmt="md", top_n=None) -> str
    # From fit_table_at_best.csv when Component B has written it, else from
    # the estimation's fit_table.csv with the caption stating that the
    # simulated column is the last CE evaluation, not theta_best. Columns:
    # moment, data, simulated, residual, contribution %. Sorted by
    # contribution, descending; top_n truncates.

def write_estimation_tables(runs, out_dir) -> list[Path]
    # Writes parameters.md/.tex and fit_<label>.md/.tex.
```

Symbols in the LaTeX output: `\beta, \alpha, \gamma_c, \gamma_h, \rho, \tau,
\theta`; plain names in Markdown. Formatting reuses `_md_table` and
`_tex_table` from `tables.py` where their column formatting fits; where it does
not (string cells such as "0.945 (0.003)"), a small writer in the same style is
added to `estimation_tables.py` rather than changing `tables.py`.

Command line: `python -m examples.durables.postprocess.estimation_tables
--run LABEL=RUN_DIR ... --out DIR`. Default `--out`:
`paper-results/durables/estimation/<YYYY-MM-DD>/`.

### 5.5 Component D: fetch estimation results

`scripts/pull_gadi_results.sh --estimation`: a fourth rsync from
`gadi:/g/data/tp66/results/durables/estimation/` into
`paper-results/durables/estimation/raw/`, including directories, `*.json` and
`*.csv`, excluding `*.nst` and `*.pkl`. The existing default behaviour is
unchanged when the flag is absent. Decided by the owner (28 Sep 2026): the
raw files (a few kilobytes per run, the primary evidence for the paper's
estimates) are committed on the branch under `paper-results/`.

### 5.6 Component E: discount-factor types in the estimation

#### The type distribution

Two free parameters replace the single `beta`: a location `beta_bar` and a
spread `sigma_beta`, both on the Eggsandbaskets scale `x = ln(1/beta - 1)`,
so that every type's `beta` stays inside (0, 1). With
`x_bar = ln(1/beta_bar - 1)`, the K nodes are

```python
z_k     = norm.ppf((k + 0.5) / K)            # k = 0..K-1: K equiprobable quantiles
x_k     = x_bar + sigma_beta * z_k
beta_k  = 1.0 / (1.0 + np.exp(x_k))           # sorted increasing in beta
share_k = 1.0 / K
```

written directly with the normal distribution function (`scipy.stats.norm`),
with no Markov-chain machinery. In code the last line is evaluated in the
algebraically identical form `beta_bar / (beta_bar + (1 - beta_bar) *
exp(sigma_beta * z_k))`, which returns `beta_bar` exactly at zero spread;
the literal form is off by one unit in the last place for about 42 percent
of `beta_bar` values (implementation note from Task E1, 28 Sep 2026). Why not Tauchen at zero persistence, which is
what "Eggsandbaskets without the AR(1)" would literally be: at `rho = 0` the
Tauchen bins have fixed standardised edges, so the shares do not depend on
`sigma_beta` or `beta_bar` at all (K = 4 gives 2.3, 47.7, 47.7 and 2.3
percent whatever the spread; verified against quantecon 0.7.2). The
estimator would then be moving the positions of two dominant types and two
tail types carrying 230 agents each in a 10,000-agent simulation. With
equiprobable quantiles every type carries the same share and `sigma_beta`
acts through all of them; `sigma_beta = 0` gives K identical types equal to
`beta_bar` with no special case; `n_std` truncation does not arise. Decided
by the owner (28 Sep 2026): equal shares, K = 4; `sigma_beta` remains a free
parameter (it sets how far apart the four values sit; the shares are fixed).

Interpretation and scale: `beta_bar` is the location of the distribution
(the median type's discount factor), not the population mean; the mean is
reported in the tables from the nodes and shares. `sigma_beta` is in logit
units: around `beta_bar = 0.94` with K = 4, `sigma_beta = 0.05` spreads the
outer types about 0.3 percentage points of `beta` either side, and 0.30
about 2 percentage points (the outer quantile of four sits at 1.15 standard
deviations). Bounds: `beta_bar` `[0.88, 0.99]`, `sigma_beta` `[0.0, 0.5]`.

In the estimation YAML:

```yaml
estimation:
  free:
    beta_bar:   {bounds: [0.88, 0.99]}
    sigma_beta: {bounds: [0.0, 0.5]}
    alpha:      {bounds: [0.3, 0.95]}
    ...
  types:
    n: 4                          # K members
    parameters:
      beta:
        location: beta_bar        # free parameters that drive the nodes
        spread: sigma_beta
        transform: logit          # logit for (0,1) parameters; log for positive; identity
```

The block is written for any number of heterogeneous parameters, not for
`beta` alone (owner's request, 28 Sep 2026): member k takes the k-th
equiprobable quantile of every listed parameter at once (comonotonic types,
as Eggsandbaskets does with `beta` and `alpha`), each parameter on its own
transform scale. Heterogeneity in a second parameter is one more entry
under `parameters:` and no code change. A product of independent mixtures
(K members per parameter) would be a new `rule:` key, not a rewrite of this
machinery, and is not built now.

`beta_bar` and `sigma_beta` never reach the solver: `expand_types(theta,
types_spec)` returns K records `(share_k, calibration_k)`, each the rest of
`theta` plus one entry per listed parameter (`beta: beta_k`). The pooled
panels carry `type_idx` and one `(N,)` array per listed parameter. Rules
enforced at start-up in `estimate.py`: a listed parameter may not appear in
`free:` (error, not a silent override); the names a `types` block maps may
not be calibration keys (they would shadow a model parameter); and no name
mapped by `types` may reach `solve()` through `calib_overrides`. A spec
without a `types` block behaves exactly as today. One new spec per registry
is added beside the existing ones, `baseline_large_egm_types.yaml` (K = 4,
1040 candidates per iteration), so the existing specs and their results are
untouched; further sizes are copies.

A `types` block together with `data_source: selfgen` is refused at start-up
with a clear message: the true values of `beta_bar` and `sigma_beta` would
have no defined source (they are not calibration keys, and a value passed as
a calibration override is ignored by `solve()`, the defect Section 2
documents for `gamma_c`). Self-generated runs with types are a later spec.

#### The simulation: every agent draws a beta at birth

`simulate_lifecycle` gains an optional argument `types`: a list of K records
`(share, nest, grids)`, the same shape `solve_at_estimates` returns
(Section 5.3). Each type's `beta` is read from its own solved model
(`_base_stage(nest).calibration["beta"]`), never passed beside it, so the
calibrated stage stays the only carrier of parameters. Before the lifecycle
walk, one uniform draw `u_i` per agent is taken from a stream that no other
draw uses, `np.random.default_rng(np.random.SeedSequence(seed,
spawn_key=(1,)))`, documented next to the two existing seeds. `seed + 1` is
not available: `make_initial_particles` already seeds `default_rng(seed + 1)`
(`solvers/simulate.py:423`) and a second generator on the same seed would
make the type draw dependent on the initial income states. Agent i's type is
`min(searchsorted(cumsum(share), u_i, side="right"), K - 1)`, which with
equal shares is `floor(K u_i)`. Because the shares do not depend on theta,
the assignment is the same for every candidate (common random numbers) and
is drawn once per simulation. Agents of type k are then walked with type k's
policies; the panels come back in the agents' original order. When types are
present two `(N,)` arrays are added, `beta_type` (the index) and `beta` (the
value); when they are absent nothing is added and the output is bit-identical
to today's. The `(T, N)` panels and `(N,)` utility statistics keep their keys
and shapes, so `make_moment_fn`, the Euler evaluators and `plot_lifecycle`
work unchanged.

Implementation note. A new function `simulate_type_subset(nest_k, grids_k,
agent_idx, N, seed)` in `simulate.py` runs the existing machinery for the
agents in `agent_idx` only: it draws the shocks and the initial particles for
all N agents exactly as `simulate_lifecycle` does today (so the draws for a
given agent are the same whatever its type), takes the subset, walks it with
`simulate` from kikku, converts the records with `_records_to_panels`, and
computes the utility statistics with that type's callables. It returns the
subset's panels together with `agent_idx`. `pool_by_type(parts, type_idx, N)`
scatters K such parts back into full arrays in the agents' original order,
both the `(T, N)` panels and the `(N,)` utility statistics (`npv_utility`,
`n_adj_periods` and the rest). `simulate_lifecycle(..., types=...)` is then
the birth draw, K calls of `simulate_type_subset`, and one `pool_by_type`;
on the MPI side each rank calls `simulate_type_subset` for its own type and
the group root calls `pool_by_type` on the gathered parts. Calling
`simulate_type_subset` with the full index set must reproduce
`simulate_lifecycle` bit for bit (check H).

Assumption to confirm during implementation: `make_initial_particles` does
not depend on `beta`, so the initial conditions drawn from any type's nest are
the same. Check H compares the particles from two nests that differ only in
`beta`; if they differ, the particles are drawn once from the type-0 nest and
the spec is amended to say so.

#### The MPI layering (Eggsandbaskets `gen_communicators`, two layers)

Within the communicator that today reaches `_run_single_estimation` (the world,
or the sweep point's `sub_comm`), of size W with W a multiple of K:

```python
group_id   = rank // K                                   # W/K groups of K ranks
group_comm = comm.Split(group_id, rank)                  # layer 2: one type per rank
roots_comm = comm.Split(0 if group_comm.rank == 0 else MPI.UNDEFINED, rank)
                                                         # layer 1: one rank per candidate
```

Divisibility is checked identically on every rank before any `Split`, and
the check is joint: the world size must be a multiple of `n_points × K`
(sweep points times types), because the sweep colouring
`world_rank * n_points // world_size` gives sub-communicators of unequal size
otherwise, and a check that fails on one sub-communicator while the others
proceed hangs the survivors at the world-level `allreduce`. On failure every
rank raises the same message together.

Only group roots call `estimate(criterion, ..., comm=roots_comm)`, so kikku
sees W/K candidates per iteration and `n_samples` is set to `roots_comm.size`
(on roots only: on workers `roots_comm` is `MPI.COMM_NULL`). Inside the
root's `trial(theta)`:

1. `group_comm.bcast(("EVAL", theta))`;
2. every rank in the group, the root included, solves its own type (root:
   k = 0; workers: k = 1..K-1) inside a `try/except` and simulates only the
   agents whose birth draw is k (the draw is deterministic given `seed`, so
   all ranks agree on it); an exception becomes a `("FAIL", message)` part;
3. `parts = group_comm.gather(part)`: every rank always reaches the gather,
   the root included, so a failure on the root cannot leave its workers
   blocked in a collective ("collective" here means an MPI call every member
   of the communicator must make; one absent member hangs the rest);
4. after the gather the root raises if any part is a failure marker, logging
   each message with the candidate; kikku's `_safe_criterion` turns the
   exception into the penalty loss. Otherwise the root pools the parts and
   returns the panels.

Workers loop on `group_comm.bcast(None)`: `"EVAL"` runs steps 2 and 3;
`"STOP"` exits. Roots call `estimate()` inside `try/finally` and broadcast
`"STOP"` in the `finally`, so an exception inside `estimate()` itself (a
checkpoint write error, for instance) still releases the workers. Order
within `_run_single_estimation`: both layers are created at entry, but the
worker loop starts only after the last collective on `comm` that precedes
`estimate()` today (the self-generated data broadcast and the resume-state
broadcast), otherwise the root's broadcast on `comm` waits for workers that
are blocked on `group_comm`. Workers return `(None, None)` from
`_run_single_estimation`, skipping the lines that read `result`, and all
ranks then reach the existing `is_final` broadcasts, the restart-loop exit
(code 42 and the world `Barrier`) and, in sweep mode, the `allreduce` and
`gather`, all of which already run on `comm`. Serial mode (`comm is None`)
runs the K solves in sequence inside `trial`.

The existing self-generated data path is untouched (Section 4); with a
`types` block it is refused before any communicator is split.

Cost: per-iteration wall time is unchanged, because every rank still performs
one solve per candidate; cores rise K-fold for the same number of candidates
(K = 4, 1040 candidates: 4160 cores, 40 Sapphire Rapids nodes). Memory per
rank is unchanged (one nest).

#### Outputs

`theta_best.json` and the other theta files carry `beta_bar` and
`sigma_beta`. `summary.json` and `manifest.json` gain a `beta_types` block
with `n`, the node values, the shares and the implied population mean of
`beta` at `theta_best`; `manifest.json` also records the git commit of the
checkout. The lifecycle tool (Component B) expands the types, re-solves the K
models at the estimates, simulates with the same birth draw, and can plot by
type; the parameter table (Component C) shows `beta_bar` and `sigma_beta` as
rows and the nodes, shares and implied mean as a footer line.

A mixture can be expressed only through an estimation spec; `run.py` and the
notebooks solve one calibration at a time and are not extended (Section 4).
`n` (K) is a discretisation setting of the estimator, which is why it lives
under `estimation:` in the spec and not in `settings.yaml`.

#### Where the code lives

- `examples/durables/solvers/beta_types.py`: `discretise_beta(beta_bar,
  sigma_beta, n)` (the logit rule; `discretise_types(location, spread, n,
  transform)` generalises it to the log and identity transforms),
  `expand_types(theta, types_spec)`, `draw_types(N, shares, seed)`,
  `pool_by_type(parts, type_idx, N, member_values)` where `member_values`
  maps each heterogeneous parameter name to its K values and yields one
  `(N,)` array per name plus `type_idx`. Pure functions, no MPI.
- `examples/durables/type_groups.py` (beside `estimate.py`, since it is
  run orchestration rather than a stage operator or a simulator):
  `split_type_groups(comm, K, n_points)`, `make_group_trial(...)` (the root
  side), `worker_loop(...)`, the `EVAL`, `STOP`, `FAIL` message forms. The
  only file that touches communicators beyond what `estimate.py` already
  does.
- `examples/durables/solvers/simulate.py`: `simulate_type_subset` and the
  `beta_types` argument of `simulate_lifecycle`.
- `examples/durables/estimate.py`: read the `types` block, create the layers,
  route roots and workers, record `beta_types` in the outputs.

### 5.7 Where output goes (nothing new under the Gadi home directory)

Estimation output on Gadi is unchanged for every registry: results under
`/g/data/tp66/results/durables/estimation/`, checkpoints and the job's
working directory under `/scratch/tp66/$USER/durables_est/`, the numba cache
under `$PBS_JOBFS`, and `--local-results none` in every PBS script, the new
ones included. The new `manifest.json` is written into the same `/g/data`
results folder. The only files this work adds to the repository, and so to
the Gadi checkout under home when it pulls the branch, are the committed raw
snapshots (about 25 KB per run) and the tests' saved example outputs (about
50 KB in total). The two post-processing tools (Components B and C) default
to an output folder inside the repository and are meant for the Mac; when
`/scratch/tp66` exists (the Gadi detection `setup.sh` uses) they refuse to
run without an explicit `--out`, so they cannot write into the Gadi checkout
by accident.

## 6. Files

Created:

- `examples/durables/syntax/cobb_douglas/estimation/*.yaml` (14 files, Section 5.1)
- `examples/durables/syntax/cobb_douglas/calibration/main_males.yaml`
- `examples/durables/syntax/cobb_douglas/spec_factory_males.yaml`
- `examples/durables/postprocess/at_estimates.py`
- `examples/durables/postprocess/estimation_tables.py`
- `examples/durables/solvers/beta_types.py`, `examples/durables/type_groups.py` (Section 5.6)
- `examples/durables/syntax/separable/estimation/baseline_large_egm_types.yaml`
  and `examples/durables/syntax/cobb_douglas/estimation/baseline_large_egm_types.yaml`
- `examples/durables/notebooks/lifecycle_at_estimates.ipynb` (+ `docs/notebooks/` symlink)
- `tests/test_estimation_specs.py`, `tests/test_cd_estimation_criterion.py`,
  `tests/test_at_estimates.py`, `tests/test_estimation_tables.py`,
  `tests/test_beta_types.py`, `tests/test_type_groups_mpi.py`
- `tests/fixtures/estimation_run/…` and `tests/fixtures/estimation_run_types/…`
  (the small JSON/CSV outputs of the short end-to-end runs, Section 7; kept as saved example outputs)

Modified:

- `examples/durables/syntax/cobb_douglas/estimation/moments_data.csv` (replaced)
- `examples/durables/estimate.py` (Sections 5.2 and 5.6)
- `examples/durables/solvers/simulate.py` (`simulate_type_subset`; the optional `beta_types` argument, Section 5.6)
- `scripts/pull_gadi_results.sh` (Section 5.5)
- `examples/durables/docs/estimation_outputs.md`: state that `best.nst` and
  `true.nst` are not written at present and why; replace the `load_nest`
  snippet with the re-solve route; document `manifest.json` in results and the
  two new tools
- `examples/durables/README.md`: Cobb-Douglas estimation, the notebook, the tables
- `docs/running-on-gadi.md`: how a Cobb-Douglas job is submitted, once the PBS
  scripts expose the variables (Section 9)
- `mkdocs.yml`: notebook entry
- `CHANGELOG.md`: entry under `Unreleased`
- `AI/working/28092026/session.md` and `AI/INDEX.md` (working-note convention)

Deleted: `cobb_douglas/estimation/moments.csv`.

## 7. Acceptance checks and tests

Each check names its oracle. Tests are written first and seen to fail.

A. Spec consistency (`tests/test_estimation_specs.py`, no solve, under 2 s).
For every registry in `examples/durables/syntax/*` and every
`estimation/*.yaml` in it: `load_estimation_spec` succeeds; every free
parameter name is a key of the calibration the solver will use (oracle: the
spec factory's resolved calibration,
`make_spec(load_spec(registry/spec_factory*.yaml), registry_dir=...)`, for
both the female and the male factory; not the top-level `calibration.yaml`,
which matches only by coincidence today), except the names a `types` block
maps through `location` and `spread`, which must not be calibration keys
(they would otherwise shadow a model parameter) and whose target (`beta`)
must be one and must not itself be in `free:`; for specs with
`data_source: precomputed`, every target key appears in `data_moments` for at
least one age group (self-generated specs have no data file and skip this
part); for `cobb_douglas`, `rho` is free and `gamma_c`, `gamma_h` are not.
This test fails on the current tree (the stale `gamma_c`) and passes after
Component A.

B. Cobb-Douglas criterion (`tests/test_cd_estimation_criterion.py`, about 30 s).
Using the driver's own `build_criterion` (Section 5.2), at
`n_a=n_h=n_w=30`, `t0=20`, `N=500`, for `spec_factory.yaml` and
`spec_factory_males.yaml`: one criterion evaluation is finite and no target
moment is NaN; simulated moments at `rho=2.0` and `rho=3.0` differ (max
absolute difference above 1e-3); and passing an unknown calibration name such
as `gamma_c` through the driver's override path raises with that name in the
message (the guard of Section 5.2). Oracle: the resolved calibration defines
which names are parameters.

C. Serial end-to-end run of the real driver (manual step, recorded in the
session note; its outputs become the saved example outputs the tests read): `python -m
examples.durables.estimate --mod syntax/cobb_douglas --spec
baseline_large_egm.yaml --settings-override n_a=30 n_h=30 n_w=30
--params-override t0=20 --N-sim 500 --n-samples 6 --n-elite 2 --max-iter 2
--scratch /tmp/... --results /tmp/... --local-results none`. Expected: exits
0; `summary.json`, `theta_best.json`, `theta_se.json`, `fit_table.csv`,
`convergence.csv` and `manifest.json` present in the results folder;
`manifest.json` has `spec_factory` and `solver_method`.

D. Lifecycle tool (`tests/test_at_estimates.py`). On the saved example outputs:
`load_run` returns the estimates, registry `cobb_douglas`, spec factory
`spec_factory.yaml`, grid `{30,30,30}`; `solve_at_estimates` plus
`simulate_at_estimates` (N=200) returns panels of shape `(70, 200)`;
`simulated_moments_at_estimates` returns one finite value per target and age
group inside the horizon, in AUD (between 1e2 and 1e7 given normalisation
1e-5); the figure file and `fit_table_at_best.csv` are written to
`tmp_path`. Oracle for the means: recomputed directly from `sim_data["c"]`
with `numpy` over exactly the rows kikku's `_age_group_masks` returns for
each group (so the test pins the four-of-five-years convention rather than
assuming five), times 1/normalisation. On the saved example outputs, whose
grid, seed, `N_sim` and spec factory the manifest records, the contributions
in `fit_table_at_best.csv` sum to `summary.json["objective"]` within 1e-8:
the re-solve reproduces the estimation's own evaluation at `theta_best`.

E. Tables (`tests/test_estimation_tables.py`). On the saved example outputs:
`parameter_table` (Markdown) has one row per free parameter, each cell
matching `^-?\d+\.\d{3,4} \(\d+\.\d+\)$` or `n/a` (four decimals for `beta`
and `beta_bar`, three otherwise, asserted per row), and each cell parsed back
equals `theta_best.json` and `theta_se.json` at the printed precision; the
LaTeX version starts with `\begin{table}` and contains `\rho`; in
`fit_table` every `contribution` equals `w (simulated - data)^2` recomputed
from the `data` and `simulated` columns with `w = 1/data^2` when
`|data| >= 1` else 1, at 1e-9 (the sum-to-100 property alone is a tautology
of kikku's `diagnostics`).

F. Existing suite unchanged: `pytest tests/` still 29 passed, 16 subtests,
plus the new tests. Documentation commands in the touched pages executed.

G. Type distribution (`tests/test_beta_types.py`, no solve). Oracles are
closed forms written in the test, not the function under test:
`discretise_beta(0.94, 0.2, 4)` equals `1/(1+exp(x_bar + 0.2 z))` with
`z = (-1.1503, -0.3186, 0.3186, 1.1503)` (the four equiprobable normal
quantiles, written as numbers) and shares `(0.25, 0.25, 0.25, 0.25)`; at
`sigma_beta = 0` every node equals `beta_bar` exactly and at `1e-9` to 1e-9;
nodes are increasing and inside (0, 1) over a grid of `(beta_bar,
sigma_beta)` covering the bounds; `expand_types` returns K records each
containing every other free parameter and `beta = beta_k`, and no `beta_bar`
or `sigma_beta`; `draw_types(N=10000, shares, seed)` gives type frequencies
within 0.02 of the shares, identical output on a second call, and a
chi-square test of independence between `beta_type` and the initial `z_idx`
from `make_initial_particles(seed)`, for the same agent and for the
neighbouring agents `2i` and `2i+1`, that does not reject at the 1 percent
level (the collision that `seed + 1` would have produced is the failure this
guards against; because `Generator.choice` consumes 32-bit halves of the
same words that `Generator.random` consumes whole, that collision ties agent
i's type to agent `2i+1`'s income state, chi-square 15,000 on 9 degrees of
freedom, while the same-agent pairing shows nothing; the test asserts the
cross-agent dependence when the draw is deliberately taken from
`default_rng(seed + 1)`); `pool_by_type` on
hand-made parts, including `(N,)` statistics, restores each agent's row and
marks `beta_type` correctly (oracle: the parts were built from a known
assignment).

H. Simulation with types is bit-identical without them
(`tests/test_beta_types.py`). The reference is a saved digest, not the
function being modified: before `simulate.py` is touched, the sha256 of the
concatenated panels of `simulate_lifecycle` at a 30³ grid, `N = 200`,
`seed = 99`, `t0 = 55`, for both registries, is recorded from `main` into the
test as a literal, together with each key's dtype. After the change,
`simulate_lifecycle(nest, grids, N, seed)`, `simulate_lifecycle(..., types=
[(1.0, nest, grids)])`, and `pool_by_type([simulate_type_subset(nest, grids,
arange(N), N, seed)], ...)` all reproduce that digest (comparison with
`np.array_equal(..., equal_nan=True)`, since rows before `t0` hold NaN, and
dtype equality for `z_idx` and `discrete`). `make_initial_particles` gives
identical particles from two nests that differ only in `beta` (the assumption
in Section 5.6). With K = 2 at a 30³ grid, the two types' consumption paths
differ (max absolute difference above 1e-6) and the pooled `beta` array takes
exactly the two node values.

I. Refusal of types with self-generated data (`tests/test_estimation_specs.py`):
a spec carrying both a `types` block and `data_source: selfgen` makes the
driver stop at start-up with a message naming both keys, before any solve.

J. Layered communicator (`tests/test_type_groups_mpi.py`, skipped with a
recorded reason when `mpi4py` is absent; run here after `pip install mpi4py`
into the project venv against the Open MPI 5.0.7 at `/usr/local/bin`, and once
on Gadi as a small job): `mpiexec -n 4 python -m mpi4py -m
examples.durables.estimate --mod syntax/separable --spec
baseline_large_egm_types.yaml --settings-override n_a=30 n_h=30 n_w=30
--params-override t0=55 --N-sim 500 --max-iter 2 ...` with K = 2 (two groups
of two ranks, so two candidates per iteration) produces `theta_best.json`,
`convergence.csv` and `summary.json` equal, to 1e-12, to the serial run with
`--n-samples 2` at the same seeds (the same candidate draws are evaluated to
the same losses). Three further runs, each forced through an environment
variable read only by the test, must finish with exit code 0, the failure
message in the log, and the penalty visible in `convergence.csv` as
`elite_mean_loss` at iteration 0 equal to `(best_loss + 1e10)/2` exactly
(the file has no per-candidate column; with two candidates and `n_elite`
2 this identity is the only trace), with no deadlock: (a) a worker raises
inside its solve on its first evaluation; (b) the group root raises inside
its own solve; (c) an exception is raised outside any solve, inside
`estimate()` on the roots, and the workers still receive `STOP`. A fourth
run in which every rank fails on every evaluation must exit non-zero with
the failure total printed, although kikku reports convergence (Section 5.2
item 5). A restart run (`--max-iter-this-run 1`,
exit code 42, then `--resume`) gives a `convergence.csv` equal to the
single-segment run. A sweep-with-types run on the data (a two-point `sweep:` block over a
calibration parameter such as `tau`, K = 2, 8 ranks: two sub-communicators
of two groups of two) completes and writes a sweep summary with one row per
point. Per-rank
solve times are logged and reported in the session note (the load-imbalance
question of Section 10).

K. Short end-to-end run with types (manual; its outputs are the second set of saved example outputs for B and C):
`python -m examples.durables.estimate --mod syntax/separable --spec
baseline_large_egm_types.yaml --settings-override n_a=30 n_h=30 n_w=30
--params-override t0=20 --N-sim 500 --n-samples 6 --n-elite 2 --max-iter 2
...`; `summary.json` carries `beta_types` with 4 nodes and shares; the
lifecycle tool re-solves the 4 types and writes a figure with by-type means;
`parameter_table` shows `beta_bar` and `sigma_beta` rows and the node footer.

L. Single-beta path unchanged (`tests/test_estimation_specs.py` and a manual
step). Automated: a spec without `types` never calls `split_type_groups`
(asserted through a counter the test installs on the module). The comparison
with `main` cannot use the new flags, which `main` does not have, so it runs
a copy of the spec placed under `/tmp` with `n_samples: 6`, `n_elite: 2`,
`max_iter: 2` and an absolute `data_file` path, passed as `--spec
/tmp/<file>.yaml` (an absolute `--spec` resolves as such), on `main` and on
the branch; `convergence.csv` and `theta_best.json` must be identical (same
seeds, same draws). Manual, recorded in
the session note with timings: `jupyter nbconvert --execute --to notebook
--output /tmp/... examples/durables/notebooks/durables_fues.ipynb` and the
same for `durables_fues_separable.ipynb` run to completion on the branch;
`python -m examples.retirement.run` and `python -m examples.durables.run
--slot-override '$draw.t0=55'` run to completion.

## 8. Order of work

1. Worktree and branch; baseline suite green.
2. Component A files by generator script; test A written to fail first, then made to pass.
3. Component A2 driver changes; short end-to-end run C; its outputs saved for the tests.
4. Component E pure functions (`beta_types.py`), tests G and H; the
   `beta_types` argument in `simulate_lifecycle`; the `types` YAML block in
   `estimate.py`; serial trial with types; test I.
5. Component E layering (`type_groups.py`), `mpi4py` installed locally, test J;
   short end-to-end run K; its outputs saved for the tests.
6. Component B module and tests D (including the types example outputs); notebook;
   docs symlink and nav.
7. Component C module and tests E (including the types example outputs).
8. Component D script flag (tested against a local directory laid out like
   Gadi's; the Gadi transfer itself needs the owner's SSH key).
9. Documentation and changelog; session note.
10. Whole-branch review (econ-code-critic), then hand to the owner for the
    merge after the PBS work lands, and one small Gadi job for test J.

## 9. PBS scripts (owned elsewhere; state as of 28 Sep 2026, 00:50)

The PBS work has landed on `main` (commits `6579acd`, `e0fe46b`, `2439b1d`).
Its convention, documented in `benchmarks/durables/estimation/README.md`, is
one script per registry, calibration overlay, size and upper-envelope method,
with `MOD`, `SPEC`, `SPEC_FACTORY`, `METHODS`, `GRID`, `N_SIM` fixed in the
header, grouped as `data-estimation/<registry>/<females|males>/` and
`selfgen/`; the environment-variable script `run_estimation.pbs` is retired
to `generic/old/`. Scripts find the repo root by walking up to
`pyproject.toml`, run from a scratch working directory with
`--local-results none`, disable core dumps on every rank, and use the
restart loop (3 iterations per segment at grid 600, where rank memory grows
by about 1.8 GB per iteration; at most 4.6 GB per rank on `normalsr`).

What this spec therefore needs on the PBS side, following that convention
rather than the environment-variable form proposed earlier:

- `data-estimation/cobb_douglas/females/` and `.../males/`: copies of the
  separable scripts with `MOD="syntax/cobb_douglas"`, the matching `SPEC`, and
  a distinct `#PBS -N`. Resources unchanged (same grid, same per-rank memory).
- For a types spec, one script with `ncpus` equal to K times the candidates
  per iteration (4 × 1040 = 4160 for `baseline_large_egm_types.yaml`, K = 4)
  and memory scaled with the node count; per-rank memory is unchanged because
  each rank holds one nest. The driver stops with a clear message when the
  communicator size is not a multiple of K.

Decided by the owner (28 Sep 2026): this branch writes these scripts, in the
format of the committed ones (same header block, repo-root discovery,
scratch working directory, core-dump and collective settings, restart loop,
and a row in the README table), and does not alter any existing script.
`docs/running-on-gadi.md` will describe the resulting layout.

## 10. Risks

- Deadlock in the layered communicator if a rank exits a collective early.
  Mitigated by the single message protocol (`EVAL`/`STOP`/`FAIL`), the failure
  marker, and test J's forced-failure runs. Any exception inside a worker's
  solve is caught at the worker's outer loop, turned into `FAIL`, and still
  participates in the gather.
- A silent wrong number, the 28 March 2026 incident class: a failure marker
  that is turned into the penalty must be logged by the root that received
  it, with the candidate and the message (each root sees only its own
  candidates, and kikku's progress output carries only the best and
  elite-mean losses, so there is no central count); the tests assert the log
  line.
- Load imbalance inside a group: ranks wait for the slowest type's solve. The
  grids are the same for every type, so no large difference is expected, but
  this has not been measured; test J records per-rank solve times, and the
  result goes into the session note.
- Identification of `sigma_beta` from age-group means, SDs and correlations
  is not guaranteed, and the natural check, a self-generated sweep over
  `sigma_beta`, is deferred with the rest of the self-generated work. Until it
  is run, the standard error of `sigma_beta` and the shape of the loss along
  it (from the CE history) are the only evidence; the tables report both.
- `sigma_beta = 0` makes the K types identical and the model equal to the
  single-beta case; with the quantile rule this is an ordinary point of the
  parameter space with no division by zero, and check G covers it exactly.
- The node rule is a modelling choice with consequences for what
  `sigma_beta` identifies (Section 5.6); it is the first open decision in
  Section 11, and the types YAML header states the rule and the shares.

- Merge overlap with the PBS branch: none in code files; possible in
  `CHANGELOG.md` and `docs/running-on-gadi.md`. Resolved at merge time.
- The male target keys (`_14_1`) and the `a_tot` columns are taken as they
  stand in the separable specs; if those are later revised, the Cobb-Douglas
  copies must follow. The consistency test catches missing columns, not
  economic mismatches. The male housing-wealth mean at age group 8 differs
  between the two Eggsandbaskets exports (264,077 in the estimation export
  used on Gadi, 63,529 in the plotting export); the owner has settled on the
  estimation export. The male profile is erratic across age groups, which
  bears on how precisely the male estimates can be read.
- The saved example outputs come from a 30³ grid and 2 CE iterations, so its estimates are
  meaningless as economics; the tests treat them as file-format examples only.
- `rho` bounds `[1.2, 6.0]` are inherited, not chosen for this utility; the
  owner may wish to narrow them after the first Cobb-Douglas run.
- Re-solving at the estimates reproduces the estimation's own evaluation at
  `theta_best` (and so the reported loss) only if grid, `t0`, `N_sim`, seed,
  spec factory and method match the run; it never reproduces the estimation's
  `fit_table.csv`, which is a different evaluation (Section 2). For runs
  finished before Component A2, the manifest is on scratch only; the tool
  takes the settings as arguments and prints what it used.
- Age groups: kikku's masks take four of the five years of each data bin
  (Section 2). Every estimate so far, separable included, was fitted under
  that convention, and kikku is frozen, so this spec keeps it and makes the
  overlay and the checks follow the same rows rather than correcting it. It is
  recorded for the owner because a later correction changes every result.
- Two inherited points about the copied specs, recorded with the numbers
  rather than decided here: the Cobb-Douglas `calibration/main.yaml`
  (line 29) says `theta` "will need recalibration for CD", yet the large and
  xlarge specs, like their separable models, fix it at 1.3498 and only
  `baseline.yaml` frees it; and the `a_tot` targets are far larger than the
  `fin_assets` columns the old Cobb-Douglas spec used (age group 5, females:
  `av_a_tot_14_0` 222,961 against `av_fin_assets_14_0` 18,431, with housing
  `av_wealth_real_14_0` 299,932), so `var: a` is matched to a total-assets
  stock rather than to liquid financial assets.

## 11. Open decisions for the owner

1. Node rule for the beta types (Section 5.6). Proposed: K equiprobable
   quantiles, share 1/K each, so that `sigma_beta` moves every type and the
   zero-spread case is ordinary. The alternative, Tauchen at zero
   persistence, is the literal "Eggsandbaskets without the AR(1)" but fixes
   the shares at 2.3/47.7/47.7/2.3 percent for K = 4, leaving `sigma_beta` to
   move only node positions in what is close to a two-type model. Also K
   itself: 4 is the working assumption.
2. Agreed in principle, implementation deferred with self-generated runs
   (28 Sep 2026): in a self-generated run the spec's `theta_true` block lists
   the true value of every estimated parameter, `beta_bar` and `sigma_beta`
   included, and the data are generated from those, expanded into types the
   same way as the trials.
3. Fourteen or seven Cobb-Douglas spec files (Section 5.1). The EGM and NEGM
   files are identical apart from comments because the method comes from the
   PBS `METHODS` line. Proposed: keep the fourteen, because the PBS scripts
   name the spec files and the new layout keeps one script per method, so
   parity of names avoids a mapping between the two conventions.
4. Commit the raw estimation JSON and CSV pulled from Gadi under
   `paper-results/durables/estimation/raw/` (proposed: yes; a few kilobytes
   per run, and the primary evidence for the paper's estimates).
5. Who writes the Cobb-Douglas and types PBS scripts under the new
   `data-estimation/<registry>/` layout: the PBS owner, or this branch once
   the PBS work is confirmed finished (Section 9).
6. Confirm the male housing-wealth cell at age group 8 with the data pipeline
   (Section 2, Section 10).
7. (Deferred with self-generated runs, 28 Sep 2026. The data-side solve
   ignores the method override, Section 2; recorded in `AI/todo.md`.)
8. `theta` in the Cobb-Douglas specs: fixed at the separable value 1.3498 as
   copied, or freed, given the note in the Cobb-Douglas calibration that it
   needs recalibration (Section 10).

Decided already: `beta` in `free:` together with a `types` block naming it is
an error (Section 5.6); `moments.csv` is deleted (Section 5.1).

## Appendix A: numerical-setting and calibration differences (not changed)

| Key | separable | cobb_douglas |
|---|---|---|
| `h_max` | 10.0 | 7.5 |
| `w_min` | 1e-10 | 1e-05 |
| `w_max` | 12.5 | 12.0 |
| `fues_lb` | 4 | 5 |
| `fues_eps_d` | 1e-05 | 1e-08 |
| `sim_guard` | 0.02 | 0.01 |
| `util_floor` | -1e18 | -1e10 |
| `ce_burn_in` | 10 | absent (code default 0) |
| `phi_w` | 0.82 | 0.8170411 |
| `sigma_w` | 0.11 | 0.1117012 |
| utility parameters | `gamma_c`, `gamma_h`, `kappa`, `sigma`, `chi`, `K` | `rho`, `d_ubar` |
