# Estimation output files

## Directory structure

Results are organised by mod (the utility specification) and spec
(the estimation configuration); gender variants are separate specs
combined with spec_factory overlays:

```
/g/data/tp66/results/durables/estimation/
  <mod_name>/                          e.g. separable, cobb_douglas
    <spec_name>/                       e.g. baseline_large_egm, baseline_large_egm_males
      est_<timestamp>/                 single estimation run
        theta_best.json
        theta_mean.json
        theta_se.json
        summary.json
        manifest.json                  grid, seeds, spec factory, method
        fit_table.csv
        convergence.csv
        best.nst                       attempted save; a zero-byte file
        true.nst                       selfgen only; a zero-byte file

    <sweep_spec_name>/                 e.g. selfgen_sweep_gamma_c_egm
      gamma_c=1.5/                     one sweep point
        est_<timestamp>/
          theta_best.json, summary.json, best.nst, true.nst, ...
      gamma_c=1.83/
        est_<timestamp>/
          ...
      gamma_c=3.5/
        est_<timestamp>/
          ...
      sweep_summary_<timestamp>.csv    all points in one table

Note: sweep point folders (gamma_c=1.5/, etc.) are NOT timestamped.
Multiple runs of the same sweep accumulate est_<timestamp>/ dirs
within each point folder. Use the sweep_summary timestamp or the
est_ timestamp to identify which run is which.
```

Scratch mirrors this structure with checkpoints:

```
/scratch/tp66/<user>/durables_est/
  <mod_name>/<spec_name>/est_<timestamp>/
    state.pkl                          CE checkpoint (updated each iteration)
    manifest.json                      run metadata
    best.nst                           same as results
    true.nst                           selfgen only
```

## File descriptions

### theta_best.json

This file records the parameter vector with the lowest SMM loss
across all CE iterations.

```json
{
  "alpha": 0.710,
  "beta": 0.959,
  "gamma_c": 5.965,
  "gamma_h": 1.959,
  "tau": 0.014
}
```

### theta_mean.json

This file records the elite-weighted mean parameter vector at
convergence. It is typically close to `theta_best` but smoother,
because it averages over the top-N candidates.

### theta_se.json

This file records standard errors from the diagonal of the elite
covariance matrix at convergence. They measure the dispersion of the
elite set, not classical sampling variation, and should be read as
CE uncertainty rather than asymptotic standard errors.

### summary.json

This file collects all of the above together with convergence
information:

```json
{
  "theta_best": { ... },
  "theta_mean": { ... },
  "theta_se": { ... },
  "objective": 37.50,
  "converged": true,
  "n_iter": 42,
  "sweep_point": null,
  "calib_overrides": {"t0": 20}
}
```

A types run adds a `beta_types` block to this file (see
[Runs with discount-factor types](#runs-with-discount-factor-types)).

### fit_table.csv

The `simulated` column is the last successful cross-entropy evaluation
on rank 0 (the last candidate of the final iteration in serial, or
rank 0's candidate under MPI). It is not the evaluation at
`theta_best`. The loss in `summary.json` is at `theta_best`, so the
two files do not describe the same parameter vector.

The `contribution` column is the unweighted squared residual. The
driver calls kikku's `diagnostics()` without the weights the criterion
uses (`1/data^2` when `|data| >= 1`, otherwise 1), so
`contribution_pct` is the share of the sum of those unweighted
squares, not the share of the reported loss.

The fit at the estimates, with the criterion's weights, is
`fit_table_at_best.csv` from the lifecycle tool in
[Post-estimation tools](#post-estimation-tools).

| Column | Description |
|--------|-------------|
| `moment` | Moment key (e.g. `av_a_tot_14_0__age5`) |
| `data` | Empirical data moment (AUD or dimensionless) |
| `simulated` | Last CE evaluation on rank 0 (denormalised to AUD) |
| `residual` | simulated − data |
| `contribution` | Unweighted squared residual |
| `contribution_pct` | Share of the sum of unweighted squared residuals |

### convergence.csv

This file records the CE iteration trace:

| Column | Description |
|--------|-------------|
| `iter` | Iteration number (0-indexed, continuous across restarts) |
| `best_loss` | Best loss found up to this iteration |
| `elite_mean_loss` | Mean loss of the elite set this iteration |

### best.nst and true.nst

`kikku.run.nest_io.save_nest` raises
`TypeError: cannot pickle 'mappingproxy' object` on a solved dolo
stage after it has already opened the output path. The driver catches
the exception and prints a warning, so a zero-byte `.nst` file is
left in the results folder. `load_nest` then rejects that file with
`EOFError`. The same sequence produces `true.nst` on a self-generated
run. Results folders on Gadi therefore contain empty `.nst` files.

To obtain policies at the estimates, re-solve. See
[Loading a finished run](#loading-a-finished-run). The re-solve takes
about one minute at the estimation grid of 600 points on each of
assets, housing and income.

### state.pkl (scratch only)

This is the CE optimizer checkpoint, updated each iteration. It
contains:

| Field | Description |
|-------|-------------|
| `means` | Elite-weighted mean parameter vector |
| `cov` | Elite covariance matrix |
| `best_theta` | Best parameter vector found so far |
| `best_loss` | Best loss found so far |
| `it` | Iteration number (0-indexed) |
| `history` | Full convergence trace (list of dicts) |
| `elite_mean_loss_prev` | Previous iteration's elite mean (for tol check) |
| `rng_state` | RNG state for reproducible sampling on resume |

The `--resume` flag reads this file to continue an estimation after
a restart.

### sweep_summary_<timestamp>.csv

The table has one row per sweep point. Its columns include:

- `true_<param>`: the sweep grid value used for data generation
- `<param>`: the estimated value
- `objective`, `converged`, `n_iter`

Example for a gamma_c sweep:
```
true_t0, true_gamma_c, alpha, beta, gamma_c, gamma_h, tau, objective, converged, n_iter
20,      1.5,          0.700, 0.945, 1.4998,  1.500,  0.120, 0.001,   True,      19
20,      3.5,          0.699, 0.944, 3.5001,  1.501,  0.119, 0.002,   True,      22
```

### manifest.json

Written at job start into both the scratch folder and the results
folder. Scratch is purged on Gadi; the results copy keeps the grid,
`N_sim`, seeds, spec factory and method with the estimates.

```json
{
  "mod": "/path/to/syntax/separable",
  "spec": "/path/to/baseline_large_egm.yaml",
  "spec_factory": "spec_factory.yaml",
  "solver_method": null,
  "run_id": "20260328_165846",
  "n_samples": 1040,
  "n_elite": 20,
  "max_iter": 200,
  "grid": {"n_a": 600, "n_h": 600, "n_w": 600},
  "N_sim": 10000,
  "simulation_seed": 99,
  "sampling_seed": 42,
  "free_params": ["alpha", "beta", "gamma_c", "gamma_h", "tau"],
  "git_commit": "744584a507b91ee993d639899ce36785254429b0",
  "beta_types": null
}
```

`spec_factory` is the `--spec-factory` value. `solver_method` is the
upper-envelope tag from `--methods-override` (for example `NEGM`), or
`null` when the YAML default is used. `git_commit` is
`git rev-parse HEAD` of the checkout; the field is `null` if git is
absent or the command fails. `beta_types` is `null` on a single-beta
run; on a types run it holds the type distribution at `theta_best`
(see [Runs with discount-factor types](#runs-with-discount-factor-types)).

## Loading a finished run

Re-solve at `theta_best` through
`examples.durables.postprocess.at_estimates`:

```python
from examples.durables.postprocess.at_estimates import (
    load_run, solve_at_estimates, simulate_at_estimates,
)

run = load_run("tests/fixtures/estimation_run/est_20260928_014054")
solved = solve_at_estimates(run, "examples/durables/syntax")
sim_data = simulate_at_estimates(run, solved)
```

`load_run` reads `summary.json` (required) and, when present,
`theta_*.json` and `manifest.json`. `solve_at_estimates` re-solves the
registry at `theta_best` on the manifest grid. `simulate_at_estimates`
walks the lifecycle with the run's `N_sim` and `simulation_seed`.
At the estimation grid of 600 the re-solve takes about one minute.

The same path from the command line, writing the figure and the two
CSV files:

```
python -m examples.durables.postprocess.at_estimates \
  tests/fixtures/estimation_run/est_20260928_014054 \
  --out /tmp/fues-at-estimates
```

The notebook `examples/durables/notebooks/lifecycle_at_estimates.ipynb`
calls the same functions. Flags `--grid`, `--n-sim` and `--seed`
override the manifest. On Gadi the tool refuses to run unless `--out`
points outside the repository checkout (see
[Post-estimation tools](#post-estimation-tools)).

## Runs with discount-factor types

A spec with an `estimation.types` block replaces the single free
`beta` by a location `beta_bar` (bounds `[0.88, 0.99]`) and a spread
`sigma_beta` (bounds `[0.0, 0.5]`). Both are defined on the logit scale
`x = ln(1/beta - 1)`. With K types the nodes sit at the equiprobable
normal quantiles of that distribution and each type carries share
`1/K`. `beta_bar` is the discount factor of the median type, not the
population mean. The files
`syntax/separable/estimation/baseline_large_egm_types.yaml` and
`syntax/cobb_douglas/estimation/baseline_large_egm_types.yaml` set
K = 4.

```yaml
estimation:
  free:
    beta_bar:   {bounds: [0.88, 0.99]}
    sigma_beta: {bounds: [0.0, 0.5]}
    ...
  types:
    n: 4
    parameters:
      beta:
        location: beta_bar
        spread: sigma_beta
        transform: logit
```

`beta_bar` and `sigma_beta` never reach the solver. Each type is
solved at its own `beta`. In the simulation every agent draws a type
once at birth from those shares; the panels then carry `type_idx`
(the type index, shape `(N,)`) and `beta` (that agent's discount
factor, shape `(N,)`).

On a types run, `summary.json` and `manifest.json` gain a
`beta_types` block: `n`, `shares`, and under `parameters` one entry per
heterogeneous parameter with its `location` and `spread` names, its
`transform`, its `nodes` at `theta_best` and its `implied_mean` (the
share-weighted mean of the nodes); for convenience `nodes` and
`implied_mean` of `beta` are repeated at the top level. In the manifest
the block is complete only once the final segment has run.

Under MPI the communicator size must be a multiple of the number of
types times the number of sweep points (types alone when there is no
sweep). The check runs on every rank before any communicator is split
and before any folder is created; the message reads, for example,
`communicator size 6 is not a multiple of n_points x K = 1 x 4 = 4`. A
types job at K = 4 and 1,040 candidates per iteration requests 4,160
cores. Local serial runs: with `mpi4py` installed, a plain
`python -m examples.durables.estimate` run has a one-rank communicator
and therefore one candidate per iteration unless `--n-samples` is given.

A `types` block together with `data_source: selfgen` is refused at
start-up. Self-generated runs with types are not part of this work.

## Post-estimation tools

Two command-line tools read a finished `est_*` folder. Both default
to a path under `paper-results/` inside the repository. When
`/scratch/tp66` exists (the Gadi detection `setup.sh` uses) they
refuse to write without an explicit `--out`, so they cannot write
into the Gadi checkout by accident.

### Lifecycle profiles at the estimates

`python -m examples.durables.postprocess.at_estimates` re-solves at
`theta_best`, simulates, and writes three files. Against the saved
example run:

```
python -m examples.durables.postprocess.at_estimates \
  tests/fixtures/estimation_run/est_20260928_014054 \
  --out /tmp/fues-at-estimates
```

The command wrote `lifecycle_at_estimates.png` (mean consumption,
housing, financial assets and the adjustment rate by age, with data
markers on the consumption, housing and assets panels),
`fit_table_at_best.csv` and `cohort_means.csv`. The printed line was
`sum of contributions: 40.02548629  objective: 40.02548628915387`.

`fit_table_at_best.csv` is the moment fit at the estimates. Its
`contribution` column uses the criterion's weights (`1/data^2` when
`|data| >= 1`, otherwise 1). When the grid, `t0`, `N_sim`, seed, spec
factory and method match the run, the contributions sum to
`summary.json["objective"]`.

The optional flags are `--grid` (cubic size), `--n-sim`, `--seed`
and `--registry-root` (default `examples/durables/syntax`). The
notebook `examples/durables/notebooks/lifecycle_at_estimates.ipynb`
calls the same functions.

### Parameter and moment-fit tables

`python -m examples.durables.postprocess.estimation_tables` writes a
parameter table across runs and, when `--fit` is given, a moment-fit
table per run. Against the same fixture:

```
python -m examples.durables.postprocess.estimation_tables \
  --run fixture=tests/fixtures/estimation_run/est_20260928_014054 \
  --fit fixture=tests/fixtures/estimation_run/est_20260928_014054 \
  --out /tmp/fues-est-tables
```

The command wrote `parameters.md`, `parameters.tex`,
`fit_fixture.md` and `fit_fixture.tex`. The parameter cells are
`best (se)` (four decimals for `beta` and `beta_bar`, three
otherwise). The fit table is read from `fit_table_at_best.csv` when
that file is present, otherwise from `fit_table.csv` with a caption
stating that the simulated column is the last cross-entropy
evaluation. In either case the displayed contribution is recomputed
with the criterion's weights.

## Periodic restart

Long-running estimations use a PBS restart loop that kills and restarts
`mpiexec` every K CE iterations to reset memory (K=3 in the standard
PBS scripts, 25 in the hugemem variants). The restart is
transparent to the estimation:

- `state.pkl` is checkpointed at each iteration
- `--resume` and `--run-id` together ensure that all segments use
  the same directory
- Iteration numbering is continuous across restarts
- Final results are written only when the run converges or reaches
  max_iter
- The driver attempts to write `best.nst` on the final segment only,
  and `true.nst` (selfgen) at the start of every segment; both writes
  fail as described under `best.nst` and `true.nst`
