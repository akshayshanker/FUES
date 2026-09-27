# Durables estimation jobs (Gadi)

PBS scripts for the cross-entropy SMM estimation of the durables model,
grouped by purpose:

| Folder | Contents |
|---|---|
| `selfgen/` | Parameter-recovery sweeps: data are generated from the model at known parameters, then re-estimated (γ_c sweep at 5,200 cores; σ_w sweep at 1,040 cores; EGM and NEGM upper envelopes; female and male calibration overlays) |
| `data-estimation/separable/females/` | Estimation on the female moments with the default calibration (`calibration/main.yaml`): xlarge (2,080 samples) and xxl (4,160 samples), EGM and NEGM |
| `data-estimation/separable/males/` | Estimation on the male moments (`spec_factory_males.yaml`): xxl, EGM and NEGM |
| `generic/` | `run_estimation.pbs`: one run, configured through `MOD`, `SPEC`, `GRID`, `N_SIM` environment variables |
| `batches/` | Convenience submitters for the usual job sets (run from the repo root) |

Submit any script with `qsub` from anywhere inside the repository, or with
`scripts/run_pbs.sh`, which lists and confirms before submitting and matches
substrings against the path (`scripts/run_pbs.sh selfgen egm`).

## Conventions shared by every script

- **Repo root** is found by walking up from the submission directory to
  `pyproject.toml`, so the scripts may live at any depth.
- **Working directory** is `/scratch/tp66/$USER/durables_est/jobs/<jobid>`,
  not the repository: the repo is imported through `PYTHONPATH`. Core files
  or any other stray output therefore land on scratch, never on the 10 GiB
  home quota. `--local-results none` disables the copy of results into the
  repository for the same reason.
- **Core dumps are disabled on every rank** by launching through
  `bash -c 'ulimit -c 0; exec python3 …'`; a plain `ulimit` in the job script
  does not reach the ranks that OpenMPI's daemons start on other nodes.
- **Collectives**: `OMPI_MCA_coll=^hcoll,ucc`. On 23 Sep 2026 (job 179615514)
  146 ranks died with SIGBUS inside UCC's shared-memory collectives as nodes
  ran out of memory, leaving 134 core files in the repository.
- **Restart loop**: rank memory grows every CE iteration (about 1.8 GB per
  iteration at grid 600, 60 MB at grid 300), so each script runs
  `ITERS_PER_RESTART` iterations per process, exits with code 42 after
  checkpointing, and restarts until the estimator converges or the walltime
  budget (`WALL_SECONDS`, which must equal the `#PBS -l walltime`) is spent.
  Checkpoints live under `--scratch`; results under
  `/g/data/tp66/results/durables/estimation/`.
- **Exit status** is that of the last `mpiexec` segment (42 counts as
  success: budget spent, checkpoint saved). Logs go to
  `/g/data/tp66/logs/durables/<jobid>.gadi-pbs.{OU,ER}`.

## Scripts

| Script | Queue | ncpus | mem | walltime | Spec | Iters/segment | Upper envelope |
|---|---|---|---|---|---|---|---|
| `data-estimation/separable/females/run_xlarge_egm.pbs` | normalsr | 2080 | 9600GB | 5:00:00 | baseline_xlarge_egm.yaml | 3 | FUES (EGM) |
| `data-estimation/separable/females/run_xlarge_negm.pbs` | normalsr | 2080 | 9600GB | 5:00:00 | baseline_xlarge_negm.yaml | 3 | NEGM |
| `data-estimation/separable/females/run_xxl_egm.pbs` | normalsr | 4160 | 19200GB | 5:00:00 | baseline_xlarge_egm.yaml | 3 | FUES (EGM) |
| `data-estimation/separable/females/run_xxl_negm.pbs` | normalsr | 4160 | 19200GB | 6:00:00 | baseline_xlarge_negm.yaml | 3 | NEGM |
| `data-estimation/separable/males/run_xxl_egm_males.pbs` | normalsr | 4160 | 19200GB | 6:00:00 | baseline_xlarge_egm_males.yaml | 3 | FUES (EGM) |
| `data-estimation/separable/males/run_xxl_negm_males.pbs` | normalsr | 4160 | 19200GB | 7:00:00 | baseline_xlarge_negm_males.yaml | 3 | NEGM |
| `selfgen/run_selfgen_sweep_gamma_c_egm.pbs` | normalsr | 5200 | 24000GB | 5:00:00 | selfgen_sweep_gamma_c_egm.yaml | 10 | FUES (EGM) |
| `selfgen/run_selfgen_sweep_gamma_c_egm_males.pbs` | normalsr | 5200 | 24000GB | 5:00:00 | selfgen_sweep_gamma_c_egm.yaml | 10 | FUES (EGM) |
| `selfgen/run_selfgen_sweep_gamma_c_negm.pbs` | normalsr | 5200 | 24000GB | 5:00:00 | selfgen_sweep_gamma_c_negm.yaml | 10 | NEGM |
| `selfgen/run_selfgen_sweep_gamma_c_negm_males.pbs` | normalsr | 5200 | 24000GB | 5:00:00 | selfgen_sweep_gamma_c_negm.yaml | 10 | NEGM |
| `selfgen/run_selfgen_sweep_sigma_w_egm.pbs` | normalsr | 1040 | 4800GB | 5:00:00 | selfgen_sweep_sigma_w_egm.yaml | 10 | FUES (EGM) |
| `selfgen/run_selfgen_sweep_sigma_w_negm.pbs` | normalsr | 1040 | 4800GB | 5:00:00 | selfgen_sweep_sigma_w_negm.yaml | 10 | NEGM |

The NEGM scripts use the same estimation spec as their EGM twin (the
`*_negm*.yaml` files differ only in their title line) and select NEGM with
`--methods-override adjuster_cons.upper_env.upper_envelope=NEGM`; the
separate spec name keeps EGM and NEGM results in separate folders.

## Variants through `qsub -v`

Every active script reads `MOD`, `SPEC`, `SPEC_FACTORY`, `GRID` and `N_SIM`
from the environment before falling back to its defaults, so one script
serves a family of runs:

```bash
qsub -N xxlEGM_s7 -v SPEC=baseline_xlarge_egm_seed7.yaml \
     benchmarks/durables/estimation/data-estimation/separable/females/run_xxl_egm.pbs
```

Results are filed under the spec's name, so a variant needs its own spec
file (a copy with the changed line and a title noting the change):
`baseline_xlarge_*_seed{7,11,99,123,2026}.yaml` change only
`sampling_seed`; `baseline_xlarge_egm*_nsim20k.yaml` are the base specs under
a new name for `N_SIM=20000` runs; `selfgen_sweep_gamma_c_low_{egm,negm}.yaml`
sweep the five lowest γ_c values with 1,040 CE samples per point.
`batches/submit_tier1.sh` and `submit_tier2.sh` are the 28 Sep 2026 run
plan (NEGM at xxl, CE-seed robustness, low-γ_c recovery, N_SIM sensitivity).

## Robustness campaign, 28–30 Sep 2026

Dimensions and where each lives:

| Dimension | Values | How it is set | Spec name carries |
|---|---|---|---|
| CE sampling seed | 42 (base), 7, 123, 2026, 11, 99 | `sampling_seed` inside the spec | `_seed<n>` |
| Upper envelope | EGM (FUES), NEGM | `--methods-override …=NEGM` in the script | `_egm` / `_negm` |
| Sex | females (base calibration), males | script (`spec_factory_males.yaml`, `_14_1` moments) | `_males` |
| Income grid `N_wage` | 4 (base), 6, 8 | `EXTRA_SETTINGS=N_wage=6`; needs 56 ranks/node (`NRANKS=2080,PPR=7`) | `_nwage<n>` |
| Housing ceiling `h_max` | 10 (base), 8, 12.5, 15 | `EXTRA_SETTINGS=h_max=15;w_max=17.5` (`w_max` keeps its 2.5 gap) | `_hmax<x>` |
| Simulated agents `N_SIM` | 10000 (base), 20000 | `N_SIM=20000` | `_nsim20k` |
| CE samples per point (recovery) | 520 (base), 1,040 | ranks per sweep point | `selfgen_sweep_gamma_c_low_*` |

Seed, sex and method variants run at the xxl size (4,160 samples); the
settings variants at the xlarge size (2,080 samples), whose base results are
30.953188 (EGM) and 30.926550 (NEGM) for females.

**Recovering a result.** Every job writes
`<results>/<mod>/<spec>/manifest_est_<run id>.json` (job id, git commit,
spec, every override, ranks, walltime, purpose) and appends a line to
`<results>/manifests/runs.csv`; the estimation itself writes `summary.json`
(θ, SE, objective, converged, iterations) into `est_<run id>/`. The PBS log is
`/g/data/tp66/logs/durables/<job id>.gadi-pbs.OU`. `scripts/collect_estimation_results.py`
joins all of these into one CSV/Markdown table with the variant fields as
columns (run it on Gadi, or on the Mac against the sshfs mount with
`--results ~/gadi/g/data/tp66/results/durables/estimation --logs ~/gadi/g/data/tp66/logs/durables`).

## Retired scripts (`old/`)

Kept for reference; `scripts/run_pbs.sh` skips them. Retired on 28 Sep 2026
after reading every job log since 30 Jun 2026:

| Script | Queue | ncpus | mem | walltime | Spec | Iters/segment | Upper envelope |
|---|---|---|---|---|---|---|---|
| `data-estimation/separable/females/old/run_large_egm.pbs` | normalsr | 1040 | 4800GB | 5:00:00 | baseline_large_egm.yaml | 3 | FUES (EGM) |
| `data-estimation/separable/females/old/run_large_egm_hugemem.pbs` | hugemem | 192 | 5000GB | 5:00:00 | baseline_large_egm.yaml | 25 | FUES (EGM) |
| `data-estimation/separable/females/old/run_large_negm.pbs` | normalsr | 1040 | 4800GB | 5:00:00 | baseline_large_negm.yaml | 3 | NEGM |
| `data-estimation/separable/females/old/run_large_negm_hugemem.pbs` | hugemem | 192 | 5000GB | 5:00:00 | baseline_large_negm.yaml | 25 | NEGM |
| `data-estimation/separable/females/old/run_xxl_egm_normal.pbs` | normal | 3840 | 15200GB | 5:00:00 | baseline_xlarge_egm.yaml | 3 | FUES (EGM) |
| `data-estimation/separable/males/old/run_large_egm_males.pbs` | normalsr | 1040 | 4800GB | 10:00:00 | baseline_large_egm_males.yaml | 3 | FUES (EGM) |
| `data-estimation/separable/males/old/run_large_egm_males_hugemem.pbs` | hugemem | 192 | 5000GB | 5:00:00 | baseline_large_egm_males.yaml | 25 | FUES (EGM) |
| `data-estimation/separable/males/old/run_large_negm_males.pbs` | normalsr | 1040 | 4800GB | 5:00:00 | baseline_large_negm_males.yaml | 3 | NEGM |
| `data-estimation/separable/males/old/run_large_negm_males_hugemem.pbs` | hugemem | 192 | 5000GB | 5:00:00 | baseline_large_negm_males.yaml | 25 | NEGM |
| `data-estimation/separable/males/old/run_xxl_egm_males_normal.pbs` | normal | 3840 | 15200GB | 5:00:00 | baseline_xlarge_egm_males.yaml | 3 | FUES (EGM) |
| `generic/old/run_estimation.pbs` | expresssr | 520 | 2400GB | 1:00:00 | `$SPEC` (default baseline.yaml) | none | FUES (EGM) |

- `*/old/run_large_*.pbs` (1,040 normalsr cores or 192 hugemem cores):
  superseded by the xlarge and xxl runs, which sample 2,080 and 4,160 draws
  per iteration against 1,040 and 192. The male variants never converged in
  their 5 h budget (0 of 11 runs); the hugemem ones were the cheapest per
  iteration (27 SU) but hugemem jobs of 144–192 cores are capped at 5 h.
- `*/old/run_xxl_egm*_normal.pbs`: 0 converged in 14 two-hour runs. The
  Cascade Lake nodes fit only 36 grid-600 ranks per 48-core node, so a
  quarter of the charged cores idle, and with `n_samples` equal to the rank
  count they run a 2,880-sample estimator that cannot reproduce the
  4,160-sample xxl results.
- `generic/old/run_estimation.pbs`: no restart loop, so memory growth ends
  it near iteration 44, and `expresssr` charges three times the normalsr rate.

Every converged run since 10 Sep 2026 reproduced the earlier loss to six
decimals (the estimator is seeded), so re-running an unchanged spec only
confirms a result already under `/g/data/tp66/results/durables/estimation/`.

Memory per rank on `normalsr` is at most 4.6 GB (480 GB per 104-core node);
the 5,200-core sweeps therefore use 10-iteration segments and the grid-600
runs 3-iteration segments. The xxl EGM runs converge in 3.4–4.4 h (65 and
72 iterations); the male EGM script has 6 h because one 5-hour run stopped
at 69 of 72 iterations, and the NEGM scripts 6 h and 7 h because NEGM took
26 per cent more iterations than EGM at the xlarge size. Walltime is
charged as used.
