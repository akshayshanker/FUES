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
        fit_table.csv
        convergence.csv
        best.nst                       solved model at theta_best (stripped)
        true.nst                       selfgen only: model at true params (full)

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

### fit_table.csv

This table compares data and simulated moments at (approximately)
`theta_best`; the simulated moments are the root rank's cached
evaluation from the last CE trial, not a re-solve at `theta_best`:

| Column | Description |
|--------|-------------|
| `moment` | Moment key (e.g. `av_a_tot_14_0__age5`) |
| `data` | Empirical data moment (AUD or dimensionless) |
| `simulated` | Model simulated moment (denormalised to AUD) |
| `residual` | simulated - data |
| `contribution` | Weighted squared residual (share of total loss) |
| `contribution_pct` | Percentage of total loss |

### convergence.csv

This file records the CE iteration trace:

| Column | Description |
|--------|-------------|
| `iter` | Iteration number (0-indexed, continuous across restarts) |
| `best_loss` | Best loss found up to this iteration |
| `elite_mean_loss` | Mean loss of the elite set this iteration |

### best.nst

This is the pickled nest object from the root rank's last CE trial
evaluation — a final-iteration candidate near `theta_best`, while
the metadata records `theta_best` itself. It contains the solved
model:

- `periods`: list of period dicts (`{stages, connectors}` with
  calibrated dolo+ stage objects; one per age)
- `solutions`: list of solution dicts per period (stripped — policies only)
- `inter_conn`: state renaming across periods
- `metadata`: `{theta_best, objective, n_iter}`

**Stripped** means that value functions (`V`) and most marginal
values are removed; the arrays kept are `keeper_cons/dcsn/c`,
`adjuster_cons/dcsn/{c, h_choice}`, `tenure/dcsn/adj`, and
`tenure/arvl/d_hV`. These are sufficient for simulation and moment
computation, but not for value function plots.

Load it with:
```python
from kikku.run.nest_io import load_nest
nest = load_nest('path/to/best.nst')
```

### true.nst (selfgen only)

This is the pickled nest at the YAML calibration defaults plus any
`--params-override` or sweep-point values — the "true" parameters
used to generate the selfgen data. The file is **full** (unstripped),
so it includes all value functions and marginal values; use it to
compare estimated against true policies and value functions.

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

### manifest.json (scratch only)

This file records run metadata at job start:

```json
{
  "mod": "/path/to/syntax/separable",
  "spec": "/path/to/baseline_large_egm.yaml",
  "run_id": "20260328_165846",
  "n_samples": 1040,
  "n_elite": 20,
  "max_iter": 200,
  "grid": {"n_a": 600, "n_h": 600, "n_w": 600},
  "N_sim": 10000,
  "free_params": ["alpha", "beta", "gamma_c", "gamma_h", "tau"]
}
```

## Loading results in notebooks

```python
import json, os
from kikku.run.nest_io import load_nest
from dolo.compiler.period_factory import period_to_graph
from examples.durables.model import make_grids

GADI = os.path.expanduser('~/gadi/g/data/tp66/results/durables/estimation')
run = os.path.join(GADI, 'separable', 'baseline_large_egm', 'est_20260328_165846')

# Load estimates
with open(os.path.join(run, 'summary.json')) as f:
    summary = json.load(f)

# Load solved model
nest = load_nest(os.path.join(run, 'best.nst'))

# Rebuild what load_nest does not carry: grids and graph
stage0 = nest['periods'][0]['stages']['keeper_cons']
grids = make_grids(stage0.calibration, stage0.settings)
nest['graph'] = period_to_graph(nest['periods'][0])

# Simulate the loaded policies
from examples.durables.solvers.simulate import simulate_lifecycle
sim_data = simulate_lifecycle(nest, grids, N=10000, seed=99)
```

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
- `best.nst` is saved on the final segment only; `true.nst` (selfgen)
  is re-generated and re-saved at the start of every segment
