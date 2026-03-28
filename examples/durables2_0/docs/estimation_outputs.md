# Estimation output files

## Directory structure

Results are organised by mod (gender/utility) and spec (estimation config):

```
/g/data/tp66/results/durables2_0/estimation/
  <mod_name>/                          e.g. separable, separable_males
    <spec_name>/                       e.g. baseline_large_egm
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

The parameter vector with the lowest SMM loss across all CE iterations.

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

The elite-weighted mean parameter vector at convergence. Typically
close to `theta_best` but smoother (averaged over the top-N candidates).

### theta_se.json

Standard errors from the diagonal of the elite covariance matrix at
convergence. Measures the dispersion of the elite set, not classical
standard errors — interpret as CE uncertainty, not asymptotic SE.

### summary.json

All of the above plus convergence info:

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

Per-moment fit comparison at `theta_best`:

| Column | Description |
|--------|-------------|
| `moment` | Moment key (e.g. `av_a_tot_14_0__age5`) |
| `data` | Empirical data moment (AUD or dimensionless) |
| `simulated` | Model simulated moment (denormalised to AUD) |
| `residual` | simulated - data |
| `contribution` | Weighted squared residual (share of total loss) |
| `contribution_pct` | Percentage of total loss |

### convergence.csv

CE iteration trace:

| Column | Description |
|--------|-------------|
| `iter` | Iteration number (0-indexed, continuous across restarts) |
| `best_loss` | Best loss found up to this iteration |
| `elite_mean_loss` | Mean loss of the elite set this iteration |

### best.nst

Pickled nest object at `theta_best`. Contains the solved model:

- `periods`: list of calibrated dolo+ stage objects (one per age)
- `solutions`: list of solution dicts per period (stripped — policies only)
- `inter_conn`: state renaming across periods
- `metadata`: `{theta_best, objective, n_iter}`

**Stripped** means value functions (`V`) and most marginal values are
removed. Kept: `keeper_cons/dcsn/c`, `adjuster_cons/dcsn/{c, h_choice}`,
`tenure/dcsn/adj`, `tenure/arvl/d_hV`. Sufficient for simulation and
moment computation. Not sufficient for value function plots.

Load with:
```python
from kikku.run.nest_io import load_nest
nest = load_nest('path/to/best.nst')
```

### true.nst (selfgen only)

Pickled nest at the YAML calibration defaults (the "true" parameters
used to generate the selfgen data). **Full** (unstripped) — includes
all value functions and marginal values. Use for comparing estimated
vs true policies and value functions.

### state.pkl (scratch only)

CE optimizer checkpoint, updated each iteration. Contains:

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

Used by `--resume` to continue estimation after a restart.

### sweep_summary_<timestamp>.csv

One row per sweep point. Columns include:

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

Run metadata recorded at job start:

```json
{
  "mod": "/path/to/mod/separable",
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

GADI = os.path.expanduser('~/gadi/g/data/tp66/results/durables2_0/estimation')
run = os.path.join(GADI, 'separable', 'baseline_large_egm', 'est_20260328_165846')

# Load estimates
with open(os.path.join(run, 'summary.json')) as f:
    summary = json.load(f)

# Load solved model
nest = load_nest(os.path.join(run, 'best.nst'))

# Simulate at theta_best
from examples.durables2_0.horses.simulate import simulate_lifecycle
sim_data = simulate_lifecycle(nest, grids, N=10000, seed=99)
```

## Periodic restart

Long-running estimations use a PBS restart loop that kills and restarts
`mpiexec` every 10 CE iterations to reset memory. The restart is
transparent:

- `state.pkl` is checkpointed each iteration
- `--resume` + `--run-id` ensure all segments use the same directory
- Iteration numbering is continuous across restarts
- Final results are written only when converged or max_iter reached
- `best.nst` and `true.nst` are saved on the final segment only
