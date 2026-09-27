# Durables estimation jobs (Gadi)

PBS scripts for the cross-entropy SMM estimation of the durables model,
grouped by purpose:

| Folder | Contents |
|---|---|
| `selfgen/` | Parameter-recovery sweeps: data are generated from the model at known parameters, then re-estimated (γ_c sweep at 5,200 cores; σ_w sweep at 1,040 cores; EGM and NEGM upper envelopes; female and male calibration overlays) |
| `data-estimation/separable/females/` | Estimation on the female moments with the default calibration (`calibration/main.yaml`) |
| `data-estimation/separable/males/` | Estimation on the male moments (`spec_factory_males.yaml`) |
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
| `data-estimation/separable/females/run_large_egm.pbs` | normalsr | 1040 | 4800GB | 5:00:00 | baseline_large_egm.yaml | 3 | FUES (EGM) |
| `data-estimation/separable/females/run_large_egm_hugemem.pbs` | hugemem | 192 | 5000GB | 5:00:00 | baseline_large_egm.yaml | 25 | FUES (EGM) |
| `data-estimation/separable/females/run_large_negm.pbs` | normalsr | 1040 | 4800GB | 5:00:00 | baseline_large_negm.yaml | 3 | NEGM |
| `data-estimation/separable/females/run_large_negm_hugemem.pbs` | hugemem | 192 | 5000GB | 5:00:00 | baseline_large_negm.yaml | 25 | NEGM |
| `data-estimation/separable/females/run_xlarge_egm.pbs` | normalsr | 2080 | 9600GB | 5:00:00 | baseline_xlarge_egm.yaml | 3 | FUES (EGM) |
| `data-estimation/separable/females/run_xlarge_negm.pbs` | normalsr | 2080 | 9600GB | 5:00:00 | baseline_xlarge_negm.yaml | 3 | NEGM |
| `data-estimation/separable/females/run_xxl_egm.pbs` | normalsr | 4160 | 19200GB | 5:00:00 | baseline_xlarge_egm.yaml | 3 | FUES (EGM) |
| `data-estimation/separable/females/run_xxl_egm_normal.pbs` | normal | 3840 | 15200GB | 5:00:00 | baseline_xlarge_egm.yaml | 3 | FUES (EGM) |
| `data-estimation/separable/males/run_large_egm_males.pbs` | normalsr | 1040 | 4800GB | 10:00:00 | baseline_large_egm_males.yaml | 3 | FUES (EGM) |
| `data-estimation/separable/males/run_large_negm_males.pbs` | normalsr | 1040 | 4800GB | 5:00:00 | baseline_large_negm_males.yaml | 3 | NEGM |
| `data-estimation/separable/males/run_xxl_egm_males.pbs` | normalsr | 4160 | 19200GB | 5:00:00 | baseline_xlarge_egm_males.yaml | 3 | FUES (EGM) |
| `data-estimation/separable/males/run_xxl_egm_males_normal.pbs` | normal | 3840 | 15200GB | 5:00:00 | baseline_xlarge_egm_males.yaml | 3 | FUES (EGM) |
| `selfgen/run_selfgen_sweep_gamma_c_egm.pbs` | normalsr | 5200 | 24000GB | 5:00:00 | selfgen_sweep_gamma_c_egm.yaml | 10 | FUES (EGM) |
| `selfgen/run_selfgen_sweep_gamma_c_egm_males.pbs` | normalsr | 5200 | 24000GB | 5:00:00 | selfgen_sweep_gamma_c_egm.yaml | 10 | FUES (EGM) |
| `selfgen/run_selfgen_sweep_gamma_c_negm.pbs` | normalsr | 5200 | 24000GB | 5:00:00 | selfgen_sweep_gamma_c_negm.yaml | 10 | NEGM |
| `selfgen/run_selfgen_sweep_gamma_c_negm_males.pbs` | normalsr | 5200 | 24000GB | 5:00:00 | selfgen_sweep_gamma_c_negm.yaml | 10 | NEGM |
| `selfgen/run_selfgen_sweep_sigma_w_egm.pbs` | normalsr | 1040 | 4800GB | 5:00:00 | selfgen_sweep_sigma_w_egm.yaml | 10 | FUES (EGM) |
| `selfgen/run_selfgen_sweep_sigma_w_negm.pbs` | normalsr | 1040 | 4800GB | 5:00:00 | selfgen_sweep_sigma_w_negm.yaml | 10 | NEGM |

## Retired scripts (`old/`)

Kept for reference; `scripts/run_pbs.sh` skips them. Retired on 28 Sep 2026
after reading every job log since 30 Jun 2026:

| Script | Queue | ncpus | mem | walltime | Spec | Iters/segment | Upper envelope |
|---|---|---|---|---|---|---|---|
| `data-estimation/separable/males/old/run_large_egm_males_hugemem.pbs` | hugemem | 192 | 5000GB | 5:00:00 | baseline_large_egm_males.yaml | 25 | FUES (EGM) |
| `data-estimation/separable/males/old/run_large_negm_males_hugemem.pbs` | hugemem | 192 | 5000GB | 5:00:00 | baseline_large_negm_males.yaml | 25 | NEGM |
| `generic/old/run_estimation.pbs` | expresssr | 520 | 2400GB | 1:00:00 | `$SPEC` (default baseline.yaml) | none | FUES (EGM) |

- `males/old/run_large_*_males_hugemem.pbs`: 0 converged in 8 runs. Hugemem
  jobs of 144–192 cores are capped at 5 h, which allows 100 iterations at
  25 per segment, and the male estimation needs more than that.
- `generic/old/run_estimation.pbs`: no restart loop, so memory growth ends
  it near iteration 44, and `expresssr` charges three times the normalsr rate.

Every converged run since 10 Sep 2026 reproduced the earlier loss to six
decimals (the estimator is seeded), so re-running an unchanged spec only
confirms a result already under `/g/data/tp66/results/durables/estimation/`.

Memory per rank on `normalsr` is at most 4.6 GB (480 GB per 104-core node);
the 5,200-core sweeps therefore use 10-iteration segments and the grid-600
runs 3-iteration segments. The `_normal` variants underpopulate Cascade Lake
nodes (36 ranks per node) to reach 5.3 GB per rank, and need the full 5 h.
