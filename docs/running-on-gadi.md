# Running on Gadi

NCI Gadi HPC. All examples use the same venv, output structure, and log paths.

## Environment

One venv for everything: `$HOME/venvs/fues`.

```bash
# First-time setup
module load python3/3.12.1
python3 -m venv ~/venvs/fues
source ~/venvs/fues/bin/activate
pip install -e ".[examples,mpi]"
pip install "kikku @ git+https://github.com/bright-forest/kikku.git"
```

Update after pulling changes:

```bash
source ~/venvs/fues/bin/activate
pip install --force-reinstall --no-deps "kikku @ git+https://github.com/bright-forest/kikku.git"
```

## Storage layout

| What | Where | Purge? |
|------|-------|--------|
| **Results** (plots, tables) | `/scratch/tp66/$USER/FUES/<model>/YYYY-MM-DD/NNN/` | 100-day |
| **PBS logs** | `/g/data/tp66/logs/<model>/` | No |
| **Estimation results** | `/g/data/tp66/results/<model>/estimation/` | No |
| **Numba cache** | `/scratch/tp66/$USER/numba_cache/` or `$PBS_JOBFS` | Job-scoped |
| **Code** | `~/dev/fues.dev/FUES/` | No |
| **Venv** | `~/venvs/fues/` | No |

**Do NOT write large outputs to `$HOME`** — 10 GiB quota.

## PBS scripts

All scripts are in `experiments/<model>/`.

### Durables

```bash
# Compare FUES vs NEGM (2 cores)
qsub experiments/durables/run_durables.pbs

# Paper table: grid × tau × method sweep (4 cores)
qsub experiments/durables/run_durables_tests.pbs

# Estimation (1040 cores, normalsr)
qsub experiments/durables/estimation/run_large_egm.pbs
```

### Retirement

```bash
# Timing benchmark (1 core, expresssr)
qsub experiments/retirement/retirement_timings.sh

# Single solve + plots (1 core)
qsub experiments/retirement/run_retirement_single_core.sh
```

## PBS script structure

Every PBS script follows the same pattern:

```bash
#!/bin/bash
#PBS -P tp66
#PBS -q expresssr
#PBS -l ncpus=N,mem=XGB,walltime=HH:MM:SS
#PBS -l storage=scratch/tp66+gdata/tp66
#PBS -l wd
#PBS -o /g/data/tp66/logs/<model>/
#PBS -e /g/data/tp66/logs/<model>/

module purge
module load python3/3.12.1

FUES_VENV="${FUES_VENV:-$HOME/venvs/fues}"
source "$FUES_VENV/bin/activate"

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/scratch/tp66/$USER/numba_cache

cd "$REPO_ROOT"

python3 -m examples.<model>.run \
    --output-dir /scratch/tp66/$USER/FUES/<model> \
    ...
```

Key points:
- **One venv**: `$HOME/venvs/fues` everywhere
- **Logs → gdata**: persistent, not purged
- **Results → scratch**: large, ephemeral (100-day purge)
- **`storage=scratch/tp66+gdata/tp66`**: required for both scratch and gdata access
- **Thread pinning**: `OMP_NUM_THREADS=1` prevents oversubscription

## MPI

For MPI jobs (estimation), use `mpiexec` with NUMA-aware placement:

```bash
#PBS -q normalsr
#PBS -l ncpus=1040

mpiexec -n $PBS_NCPUS \
    --map-by ppr:13:numa \
    python3 -u -m mpi4py -m examples.durables.estimate \
    --output-dir /scratch/tp66/$USER/FUES/durables \
    ...
```

Sapphire Rapids: 8 NUMA domains × 13 cores = 104 cores/node.

## Queues

| Queue | Cores/node | Cost (SU) | Max walltime | Use for |
|-------|-----------|-----------|-------------|---------|
| `expresssr` | 104 | 6.0 | 24h | Quick tests, single-core runs |
| `normalsr` | 104 | 2.0 | 48h | Estimation, large sweeps |
| `normal` | 48 | 1.0 | 48h | Cost-sensitive batch runs |

## Monitoring

```bash
# Job status
qstat -u $USER

# Live CPU/memory
nqstat_anu <jobid>

# Check output while running
tail -f /g/data/tp66/logs/durables/*.o<jobid>
```

## Common issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'seaborn'` | `pip install -e ".[examples]"` in the fues venv |
| `ModuleNotFoundError: No module named 'HARK'` | `pip install econ-ark` |
| `make_egm_1d() missing argument` | Reinstall kikku: `pip install --force-reinstall "kikku @ git+https://..."` |
| PBS log says "permission denied" on gdata | Add `+gdata/tp66` to `#PBS -l storage` |
| `$HOME` quota full | Delete old venvs (`~/venvs/dcsmm`, `~/venvs/fues_public`), clear `~/.cache/pip` |
| Numba `ReferenceError: underlying object has vanished` | Clear Numba cache: `rm -rf /scratch/tp66/$USER/numba_cache/*` |
