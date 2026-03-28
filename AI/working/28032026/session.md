# Session notes — 27-28 March 2026

## Summary

Major push on the SMM estimation pipeline: memory fixes, sweep support,
loss function correctness, and results infrastructure. Two days of
debugging a chain of issues that prevented the CE estimator from
producing valid results on Gadi.

## Bugs found and fixed (chronological)

### 1. NaN data moments propagating through loss (kikku)
**Commit**: `dc01dfb` (kikku main)

`make_criterion` built `data_vec` from all keys in `data_moments`,
including NaN entries from selfgen (age groups outside t0 range).
NaN in `data_vec` → NaN weights → NaN loss → `elite_mean=nan`.

**Fix**: filter NaN entries before building `data_vec`:
```python
keys = [k for k in all_keys if not _is_nan_float(data_moments[k])]
```

### 2. Selfgen data at wrong t0 (estimate.py)
**Commit**: `a7cf3dd`

Selfgen solve didn't pass `calib_overrides`, so data generated at
`t0=40` (YAML default) while trial uses `t0=20`. Data moments for
age groups 1-4 were NaN.

**Fix**: pass `calib_overrides` to selfgen solve call.

### 3. Memory leak — solution arrays not freed (estimate.py + kikku)
**Commits**: `5c6a2f7`, `28ada15` (kikku), `34e0e6e`

Each `solve()` produces ~1.8 GB of solution arrays at 300-grid.
Objects lingered because nothing explicitly freed them between
evaluations. 520 ranks × growing memory → OOM.

**Fix chain**:
- `del nest, grids` after simulation in trial function
- `del panels` after moment extraction in kikku criterion
- `gc.collect()` after each CE iteration
- Post-solve stripping of V/marginal arrays not needed for simulation

### 4. Memory OOM at 600-grid — incremental nest stripping
**Commit**: `c2513bc`

At 600-grid, each solution period has 15 arrays × 20 MB = 300 MB.
30 periods = 9 GB per rank. 1040 ranks = 9.4 TB, exceeding 4.8 TB PBS.

**Fix**: `strip_solved` option in `accrete_and_solve()`. Strips period
`h-2` after period `h` solves (keeps `h-1` alive for `vlu_cntn`).
Reduces peak from 9 GB to ~3.2 GB per rank. Default `False` so
notebooks keep full solutions.

### 5. Key mismatch — all CSV columns matched against model (estimate.py)
**Commits**: `0244a17`, `7d73db2`

`_flatten_moments_csv` loads ALL 130+ CSV columns (1069 keys across
9 age groups). Model only produces ~100 keys. The ~900 unmatched
keys each got `NAN_PENALTY = 1e6` → loss ≈ 1e9.

**First attempt**: filter using `get_moment_names(moment_spec)` — but
this function uses dummy panels with hardcoded `T=10`, while age groups
need rows 20-64. All masks empty → 0 keys → filter removed everything.

**Fix**: build target key set from YAML `targets:` section directly.
Each target has a `key` field matching CSV column names.

### 6. Selfgen data moments filtered to 0 keys
**Commit**: `78d0fdc`

The key filter ran for ALL data sources. Selfgen data uses model keys
(e.g. `mean_c__age5`) not CSV keys (e.g. `av_consumption2_14_0__age5`).
Filtering selfgen against CSV-derived target prefixes removed everything.

**Fix**: only filter when `data_source == 'precomputed'`.

### 7. Scale mismatch — model units vs AUD
**Commits**: `919b44d`, `257ccd2`

CSV data moments are in AUD (~18,000 for consumption). Model simulates
in normalised units (normalisation = 1e-5, so consumption ≈ 0.18).
Direct comparison gives huge residuals.

**First attempt**: normalise data DOWN to model units. But this broke
the loss weighting: consumption (0.18 model units, |data| < 1) got
absolute weight while housing (2.05, |data| > 1) got relative weight.

**Final fix**: denormalise MODEL moments UP to AUD via a wrapper around
`moment_fn`. Data stays in natural units. The relative deviation
weighting (`1/data²` for `|data| >= 1`) now correctly treats:
- All means/SDs: relative (all >> 1 in AUD)
- All correlations/autocorrelations: absolute (|data| < 1)

## Features added

### Sweep estimation via MPI comm splitting
**Commit**: `547585e`

`sweep:` block in estimation YAML → MPI `comm.Split` → independent
CE estimation per sweep point. Single PBS job, ranks divided evenly.

```yaml
estimation:
  sweep:
    sigma_w: [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.25]
```

Results under `<spec_name>/<sweep_var>=<value>/est_<timestamp>/`.
`sweep_summary.csv` aggregates all points. Multi-dim sweeps via
cartesian product. Backward compatible (no sweep block = single run).

### Estimation results notebook
**Commit**: `c00d1bc`

`estimation_fit.ipynb`: loads theta_best from a results dir, solves +
simulates, plots model vs data moments by age group (means, SDs,
correlations, autocorrelations), convergence, lifecycle profiles,
loss decomposition bar chart. Points at Gadi mount by default.

### Results organisation
- Results saved under `<spec_name>/est_<timestamp>/` (not flat)
- Local copy to `experiments/durables/estimation/results/` (on /home/)
- `--local-results` CLI flag to override path or disable

### Estimation YAML inventory

| YAML | Data | Method | Sweep |
|------|------|--------|-------|
| baseline | precomputed | EGM | - |
| baseline_large_egm | precomputed | EGM | - |
| baseline_large_negm | precomputed | NEGM | - |
| selfgen_large_egm | selfgen | EGM | - |
| selfgen_large_negm | selfgen | NEGM | - |
| selfgen_sweep_sigma_w_egm | selfgen | EGM | sigma_w x7 |
| selfgen_sweep_sigma_w_negm | selfgen | NEGM | sigma_w x7 |
| selfgen_sweep_gamma_c_egm | selfgen | EGM | gamma_c x10 |
| selfgen_sweep_gamma_c_negm | selfgen | NEGM | gamma_c x10 |

All have `max_iter: 200`. Each has a matching PBS script.

## Gadi results (from completed runs before latest fixes)

### Selfgen recovery (working)
- `selfgen_large_egm`: converged in 11 iters, loss=3e-07
- All 5 params recovered to 4+ decimal places

### Selfgen sweeps (broken — 0 matched moments)
- All sweep runs on Gadi used old code with `get_moment_names` filter
- Got `Matched data moments: 0` → loss=0 → meaningless convergence in 2 iters
- Need `git pull` and resubmit

### Precomputed baseline (broken — 0 matched moments)
- Same `get_moment_names` bug → 0 matched data moments
- Need `git pull` and resubmit

## Devspecs written

1. `AI/devspecs/27032026/incremental_nest_stripping.md` — memory analysis
   and implementation plan (now implemented)
2. `AI/devspecs/27032026/batch_estimation_sweep.md` — sweep design with
   MPI comm splitting (now implemented)

## Key lessons

1. **The loss function must see data and model in the same units.**
   Denormalise model → AUD is cleaner than normalise data → model units
   because the weighting thresholds are designed for natural-scale data.

2. **`get_moment_names()` uses hardcoded T=10 in its dummy panels.**
   Don't use it for filtering — build key sets from the YAML spec directly.

3. **Key filtering must be data-source-aware.** Selfgen keys are model
   keys; precomputed keys are CSV column names. They intersect through
   the `targets:` section but only when both use the same key pattern.

4. **Memory during backward induction is the bottleneck**, not simulation.
   The nest holds all 30 periods' solutions but only period h-1 is needed
   for the next solve step. Incremental stripping cuts peak by ~70%.

## Next steps

- `git pull` on Gadi and resubmit all jobs
- Verify precomputed baseline converges with the denorm + filter fixes
- Check sweep results are now meaningful (non-zero loss, varying theta)
- Notebook: run with a converged precomputed result to verify plots

## 28 March (continued session)

### Memory leak investigation

Deployed per-phase RSS diagnostics (`FUES_MEM_DIAG=1`) to isolate
the ~75 MB/rank/iter memory growth.

**Key finding**: `delta_eval=73MB`, `delta_bcast=0`, `delta_gc=0`.
The growth is entirely in the solve+simulate phase. mpi4py is clean.

**Per-step breakdown inside solve()**:
- `load_syntax`: +0 MB
- `instantiate_period`: +0-1 MB (scattered, dolo SymbolicModel objects)
- `make_grids`: +40 MB ← **this was recreated every CE iteration**
- `accrete_and_solve`: +2245 MB (expected, backward induction peak)

**Root cause**: `ru_maxrss` is a high-water mark. Transient allocations
(grids, stage objects) that aren't freed before the next solve's peak
ratchet the high-water mark up.

**Fix**: cache grids once before the CE loop, pass via
`solve(..., grids=_cached_grids)`. Saves ~40 MB/iter.

**Remaining ~35 MB/iter**: from per-call recreation of callables,
operator closures, and dolo stage objects. These overlap with the
solve peak. Not a true leak — just high-water mark ratcheting.

**Things ruled out**: NRT leak (tested locally — 0 outstanding),
mpi4py (delta_bcast=0), pymalloc (PYTHONMALLOC=malloc), glibc
fragmentation (MALLOC_MMAP_THRESHOLD_), numba parallel runtime
(workqueue), dolo/dolang caches (none found).

### Estimation results

**Selfgen recovery** (earlier runs): converges in 11-16 iters.
All 5 params recovered to 4+ decimal places.

**Precomputed baseline**: converges to loss ~37.4 after 20-30 iters.
Parameters hitting bounds (gamma_c → 6.0, tau → 0.01). The model
doesn't fit AUD empirical data well at any parameter combination.
But pipeline is working correctly.

### Features delivered today

1. **Grid caching** — grids built once, reused across CE iterations
2. **Memory diagnostics** — `FUES_MEM_DIAG=1` env var, rank 0 only
3. **RSS per CE iteration** — `[mem]` lines in output
4. **PBS env vars** — MALLOC_MMAP_THRESHOLD_, PYTHONMALLOC,
   NUMBA_THREADING_LAYER in all PBS scripts
5. **XLarge specs** — 2080-rank estimation (20 nodes)
6. **Noise fraction** — 10% uniform draws per CE iteration
7. **Sweep devspec** — estimation sweep runner design
8. **Prompt template** — new estimation round template
9. **Gadi skill update** — comp-econ-aeons pbs-patterns.md expanded
10. **EGM vs NEGM notebook** — selfgen recovery comparison
11. **Setup refactor prompt** — venv strategy design document
12. **update_and_activate.sh** — quick Gadi refresh script

### Files changed

See git log for full commit history. Major files:
- `examples/durables2_0/estimate.py` — grid caching, diagnostics, sweep
- `examples/durables2_0/solve.py` — diagnostics, strip_solved
- `kikku/kikku/run/estimate.py` — RSS logging, noise_fraction, malloc_trim
- `experiments/durables/estimation/*.pbs` — env vars, xlarge specs
- `examples/durables2_0/notebooks/estimation_*.ipynb` — results UI
- `AI/devspecs/28032026/` — estimation sweep runner devspec
- `AI/working/28032026/` — session notes, memory analysis

## Diagnostic workflow for memory issues

### Always-on: [mem] line per CE iteration (kikku)

Every CE iteration prints rank 0's RSS at three phases:
```
[mem] iter=5 eval=2928MB bcast=2928MB gc=2928MB delta_eval=70MB delta_bcast=0MB delta_gc=0MB
```

- `delta_eval` > 0: growth during solve+simulate (model code)
- `delta_bcast` > 0: growth during MPI broadcast (mpi4py)
- `delta_gc` < 0: memory reclaimed by gc.collect + malloc_trim

This is always printed (no env var needed). Lives in kikku `_cross_entropy_minimize`.

### On demand: per-step [solve] diagnostics

Set `FUES_MEM_DIAG=1` in the PBS script to enable:
```
[solve] load_syntax: +0MB
[solve] instantiate: +1MB
[solve] make_grids: +40MB
[solve] accrete_and_solve: +2245MB
```

Shows which step inside solve() contributes to the RSS peak.
Only prints from MPI rank 0.

To enable: uncomment `export FUES_MEM_DIAG=1` in the PBS script.
Code lives in `examples/durables2_0/solve.py`.

### Workflow to diagnose a memory issue

1. Submit job, check `nqstat_anu <jobid>` for RSS growth
2. Read `[mem]` lines via `qcat -o <jobid>` — identify which phase grows
3. If `delta_eval` is the culprit: enable `FUES_MEM_DIAG=1`, resubmit
4. Read `[solve]` lines — identify which step (load_syntax, instantiate,
   make_grids, accrete_and_solve) contributes most
5. If needed: add more granular diagnostics inside the offending step

### Grid caching

Grids are built once before the CE loop and reused:
```python
_cached_grids = _make_grids(_pre_cal, _pre_sett)
# ...
nest, grids = solve(..., grids=_cached_grids)
```

This is safe because:
- `solve()` already accepts `grids=` parameter (existing API)
- Grids are read-only (used for interpolation, never mutated)
- Grid structure depends on settings only, not on theta
- To rebuild: pass `grids=None` (notebooks, CLI default)

### Environment variables in PBS scripts

```bash
export MALLOC_MMAP_THRESHOLD_=65536     # large arrays use mmap
export MALLOC_TRIM_THRESHOLD_=0         # aggressive sbrk trim
export PYTHONMALLOC=malloc              # bypass pymalloc arenas
export NUMBA_THREADING_LAYER=workqueue  # lightweight parallel backend
export NUMBA_NUM_THREADS=1              # single-threaded per MPI rank
# export FUES_MEM_DIAG=1               # uncomment for per-step diagnostics
```
