# Running on a PBS cluster

This guide applies to any cluster that uses PBS (OpenPBS, PBS Pro, or
Altair PBS) for job scheduling. Cluster-specific details — queue names,
filesystem paths, project codes, module versions — are exposed as shell
variables so you can adapt them to your site.

## Site-specific variables

Set these once (e.g. in `~/.bashrc` or a `site.env` file you source at
the top of each PBS script):

```bash
export PROJECT=your_project_code             # allocation / project id for #PBS -P
export SCRATCH_DIR=/scratch/$PROJECT/$USER   # fast, ephemeral (purged periodically)
export STORAGE_DIR=/storage/$PROJECT         # persistent (logs, estimation outputs)
export FUES_VENV=$HOME/venvs/fues            # venv location (on a filesystem that
                                             # handles concurrent MPI reads — often NFS)
```

Rule of thumb:

| Filesystem | Use for | Why |
|---|---|---|
| Home (`$HOME`) | Venv, code | Persistent, small quota; no concurrent-read pressure |
| Scratch (`$SCRATCH_DIR`) | Run outputs, numba cache | Fast, ephemeral, large |
| Project storage (`$STORAGE_DIR`) | Logs, final results you want to keep | Persistent, tracked against project quota |

**Do not write large outputs to `$HOME`** — home quotas are typically small
(~10 GiB). Outputs go to scratch; things you want to keep are archived
from scratch to project storage.

## Environment

One venv for everything, one script:

```bash
# Edit the module loads inside setup/setup.sh if your cluster uses
# different Python / MPI module names, then:
source setup/setup.sh
```

`setup/setup.sh` is the only script. Behaviour:

- First run: creates `$FUES_VENV` (default `$HOME/venvs/fues`), installs
  the scientific stack (numpy, numba, scipy), `dcsmm[examples]` (FUES +
  HARK + ConSav + kikku + matplotlib + seaborn + …) in editable mode, the
  `bright-forest/dolo` fork at the pinned `phase1.1_0.1` branch (for
  `dolo.compiler.spec_factory`), and `mpi4py` built from source against
  the loaded MPI. Runs verification imports that exit non-zero if
  anything is missing.
- Subsequent runs: just activates the venv and sets runtime env vars
  (`NUMBA_NUM_THREADS=1`, `NUMBA_CACHE_DIR=…`, `MPLBACKEND=Agg`).
- `source setup/setup.sh --update`: `git pull` + reinstall
  `dcsmm[examples]` and `kikku[estimation]` `--no-deps` (keeps the pinned
  scientific stack intact).

Full rebuild when dependencies change enough to warrant it:

```bash
rm -rf "$FUES_VENV"        # or ~/.venv on local
source setup/setup.sh
```

Interactive PBS session on Gadi (so you can run a `source setup/setup.sh`
inside a compute node before submitting MPI jobs):

```bash
qsub -I -q expresssr -P "$PROJECT" -l ncpus=1,mem=5GB,walltime=01:00:00,storage=scratch/"$PROJECT",wd
```

## Storage layout

| What | Where | Purge? |
|---|---|---|
| Run outputs (plots, tables) | `$SCRATCH_DIR/FUES/<model>/YYYY-MM-DD/NNN/` | Yes (cluster policy) |
| PBS logs | `$STORAGE_DIR/logs/<model>/` | No |
| Estimation results (kept) | `$STORAGE_DIR/results/<model>/estimation/` | No |
| Numba cache | `$SCRATCH_DIR/numba_cache/` or `$PBS_JOBFS` | Job-scoped |
| Code checkout | `$HOME/.../FUES/` | No |
| Venv | `$FUES_VENV` | No |

## PBS script structure

Every script in `experiments/<model>/` follows this pattern:

```bash
#!/bin/bash
#PBS -P <your-project-code>
#PBS -q <queue-name>
#PBS -l ncpus=N,mem=XGB,walltime=HH:MM:SS
#PBS -l storage=scratch/<project>+<persistent-storage-alias>
#PBS -l wd
#PBS -o <absolute-path-to-log-dir>/
#PBS -e <absolute-path-to-log-dir>/

set -euo pipefail

# Load your cluster's Python + MPI modules. Names vary across sites.
module purge
module load python3/3.12.1
module load openmpi/4.1.5

source "${FUES_VENV:-$HOME/venvs/fues}/bin/activate"

export OMP_NUM_THREADS=1                 # one BLAS/OMP thread per MPI rank
export NUMBA_NUM_THREADS=1               # same for Numba
export NUMBA_CACHE_DIR="${SCRATCH_DIR}/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

cd "$PBS_O_WORKDIR"
python3 -m examples.<model>.run \
    --output-dir "${SCRATCH_DIR}/FUES/<model>" \
    <model-specific flags>
```

Key points:
- `#PBS -l storage=...` must list every filesystem the job touches. Sites
  that isolate filesystems per project will refuse access otherwise.
- `#PBS -l wd` means the job starts in the directory where it was
  submitted (`$PBS_O_WORKDIR`); relative paths in your command work as
  they do interactively.
- Pin threads (`OMP_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`) when you run
  MPI — otherwise each rank's BLAS will oversubscribe the cores your
  MPI process is already using, and throughput drops sharply.

## MPI jobs

For MPI runs (estimation sweeps, large parameter sweeps), use
NUMA-aware placement so each rank sits on its own core within a NUMA
domain:

```bash
#PBS -q <large-parallel-queue>
#PBS -l ncpus=<total-cores>

# ppr:<ranks>:numa — <ranks> processes per NUMA domain.
# Set <ranks> to (cores-per-node / NUMA-domains-per-node).
mpiexec -n "$PBS_NCPUS" \
    --map-by ppr:<ranks-per-numa>:numa \
    python3 -u -m mpi4py -m examples.<model>.estimate \
    --output-dir "${SCRATCH_DIR}/FUES/<model>" \
    <flags>
```

To figure out your cluster's NUMA layout, run `lscpu` or `numactl -H`
on a compute node (usually from an interactive job). Example: 8 NUMA
domains × 13 cores = 104 cores per node → `ppr:13:numa`.

`python3 -m mpi4py` ensures uncaught exceptions on any rank abort the
whole job cleanly instead of leaking into a hang.

## Queues

Queue names and limits are cluster-specific. Check your site:

```bash
qstat -Q            # list available queues
qstat -Qf <queue>   # walltime/core limits, cost, node type
```

General guidance (adapt to your site's naming):

| Class | Use for |
|---|---|
| Express / short | Quick correctness tests; single-core runs |
| Normal / standard | Long batch runs, estimation, large sweeps |
| Fat / high-memory | Jobs whose working set exceeds a standard node's memory |
| GPU | Only if you've rewritten an operator for CUDA (not the case here) |

Pick the queue whose walltime and core count match your job — you're
billed for the full allocation regardless of whether you use it.

## Monitoring

```bash
qstat -u $USER                    # your jobs' status
qstat -f <jobid>                  # full resource use + exit codes
tail -f <log-dir>/*.o<jobid>      # stdout as it streams
tail -f <log-dir>/*.e<jobid>      # stderr as it streams
```

Some sites ship extra tools: `nqstat_anu` (Gadi), `squeue`-like aliases,
or Grafana dashboards. Check `which`-style commands or your site docs.

## Examples

All scripts are in `experiments/<model>/`.

### Durables

```bash
# Compare FUES vs NEGM
qsub experiments/durables/run_durables.pbs

# Paper table: grid × tau × method sweep
qsub experiments/durables/run_durables_tests.pbs

# Estimation (MPI)
qsub experiments/durables/estimation/run_large_egm.pbs
```

### Retirement

```bash
# Timing benchmark
qsub experiments/retirement/retirement_timings.sh

# Single solve + plots
qsub experiments/retirement/run_retirement_single_core.sh
```

Before submitting a large job, do a 2-rank smoke test on a login node
or an interactive job:

```bash
source "$FUES_VENV/bin/activate"
mpiexec -n 2 python3 -m mpi4py -m examples.durables.run \
    --sweep \
    --slot-range '[{"draw":{"n_a":60,"tau":0.12}},{"draw":{"n_a":80,"tau":0.12}}]' \
    --slot-range '[{"method_switch":"FUES"},{"method_switch":"NEGM"}]' \
    --sweep-runs 1 --simulate --n-sim 500 --seed 42 \
    --slot-override '$draw.t0=60' \
    --output-dir "${SCRATCH_DIR}/FUES/durables_smoke"
```

## Common issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'dolo.compiler.spec_factory'` | Venv has vanilla PyPI dolo. `rm -rf "$FUES_VENV"; source setup/setup.sh` to rebuild. |
| `ModuleNotFoundError: No module named 'seaborn'` (or similar) | `pip install -e ".[examples]"` in the venv. |
| `make_egm_1d() missing argument` (or other kikku API mismatch) | Reinstall kikku: `pip install --force-reinstall --no-deps "kikku @ git+https://github.com/bright-forest/kikku.git"`. |
| PBS log says permission denied on a filesystem | That filesystem isn't listed in `#PBS -l storage=...`. Add it. |
| `$HOME` quota full | Delete unused venvs and caches: `rm -rf ~/venvs/old_*`, `rm -rf ~/.cache/pip`. |
| Numba `ReferenceError: underlying object has vanished` | Stale cache. `rm -rf "$NUMBA_CACHE_DIR"/*` and resubmit. |
| MPI job hangs at startup | Often a filesystem-isolation issue on Lustre. Put the venv on NFS (home) rather than scratch; Lustre struggles with hundreds of concurrent `import` reads. |
| Ranks silently produce different results | Check that the driver seeds every per-rank RNG. For `--simulate`, `--seed N` is deterministic across ranks; other randomness needs per-rank care. |
