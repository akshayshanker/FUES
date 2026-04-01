# Periodic MPI restart for memory management

## Problem

Each CE iteration grows rank RSS by ~70 MB due to transient Python
object lifetimes overlapping with the next solve's peak (high-water
mark ratcheting). At 1040 ranks × 70 MB × 30 iters = 2.2 TB growth,
jobs exceed the PBS memory allocation.

## Solution

The PBS script runs a bash loop around `mpiexec`. Every K iterations
(default K=10), Python exits cleanly, the OS reclaims all memory, and
a new `mpiexec` resumes from the last checkpoint.

```
PBS job starts
│
├─ Segment 1: mpiexec ... --max-iter-this-run 10 --run-id $RUN_ID
│   ├─ CE iters 0-9
│   ├─ Checkpoint saved (state.pkl)
│   ├─ All ranks exit with code 42
│   └─ RSS reset to 0
│
├─ Segment 2: mpiexec ... --max-iter-this-run 10 --run-id $RUN_ID --resume
│   ├─ Load state.pkl → resume from iter 10
│   ├─ CE iters 10-19
│   ├─ Checkpoint saved
│   ├─ All ranks exit with code 42
│   └─ RSS reset to 0
│
├─ ... (repeat until converged or max_iter reached)
│
└─ Final segment: converged → exit code 0 → write results
```

## PBS script structure

```bash
ITERS_PER_RESTART=10
MAX_RESTARTS=20
RUN_ID=$(date +%Y%m%d_%H%M%S)

RESTART_NUM=0
RESUME_FLAG=""

while [ $RESTART_NUM -lt $MAX_RESTARTS ]; do
    RESTART_NUM=$((RESTART_NUM + 1))

    mpiexec -n $PBS_NCPUS ... \
        --max-iter-this-run $ITERS_PER_RESTART \
        --run-id $RUN_ID \
        $RESUME_FLAG

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 42 ]; then
        break               # converged or error
    fi

    RESUME_FLAG="--resume"  # subsequent segments resume
done
```

## Exit codes

| Code | Meaning | PBS loop action |
|------|---------|-----------------|
| 0 | Converged or max_iter reached | Break, results written |
| 42 | Restart needed (iter budget exhausted) | Continue loop, add --resume |
| Other | Error | Break |

## How ranks synchronise

### Single estimation (no sweep)

All ranks participate in one CE loop. The CE optimizer broadcasts
convergence status after each iteration via `bcast_item`. When the
iter budget is exhausted:

1. Root computes `is_final` (converged? max_iter reached?)
2. `is_final` is broadcast to all ranks
3. If not final: all ranks hit `Barrier()` then `sys.exit(42)` together
4. If final: root writes results, all ranks exit 0

The `Barrier()` ensures no rank exits before others — prevents
`MPI_ABORT` from rank disagreement.

### Sweep estimation (communicator splitting)

The world communicator is split into sub-communicators, one per sweep
point. Each sub-comm runs an independent CE loop with its own
checkpoint.

When the iter budget is exhausted:

1. Each rank computes `is_final` for its sub-comm
2. `is_final` is broadcast within each sub-comm (so all ranks in a
   sub-comm agree)
3. `allreduce(MIN)` across the world communicator: if ANY sub-comm
   is not final, `all_final = False`
4. If not all final: all ranks hit `Barrier()` then `sys.exit(42)`
5. If all final: gather results, write sweep summary, exit 0

This means: if 9 of 10 sweep points converged but 1 didn't, ALL 10
restart. The 9 converged points re-converge immediately on the next
segment (tol check passes on iter 1). This wastes ~1 solve per
converged point per restart but avoids complex partial-communicator
management.

**Idle ranks**: within a restart segment, fast-converging sub-comms
finish their K iterations before slow ones. Those ranks sit idle in
`mpi_map` waiting for the slowest sub-comm's ranks to complete their
evaluations. This is inherent to the flat MPI topology — all ranks
must participate in the world-level `allreduce` and `Barrier` at the
end of the segment.

## Checkpoint contract

`state.pkl` contains everything needed for seamless resume:

| Field | Purpose |
|-------|---------|
| `means` | Elite mean — initialises next iteration's MVN draws |
| `cov` | Elite covariance — shapes the sampling distribution |
| `best_theta` | Best parameter vector found so far |
| `best_loss` | Best loss value |
| `it` | Global iteration counter (continuous across restarts) |
| `history` | Full convergence trace (appended, not reset) |
| `elite_mean_loss_prev` | For tol convergence check across restart boundary |
| `rng_state` | Numpy RNG state for reproducible sampling |

The checkpoint is written atomically (temp file + rename) to prevent
corruption if the job is killed mid-write.

## Run ID

The `--run-id` flag ensures all restart segments use the same results
directory. Without it, each segment generates a new timestamp and
creates a new directory — the resume can't find the previous
segment's checkpoint.

The PBS script generates `RUN_ID` once before the loop and passes it
to every segment. The Python code uses `args.run_id` if provided,
otherwise falls back to `datetime.now()`.

## Memory budget

With K=10 iters per restart:

| | Base RSS | Growth (10 iters) | Peak | Allocation |
|---|----------|-------------------|------|------------|
| Large (1040 ranks) | ~2.6 TB | 0.7 TB | ~3.3 TB | 4.8 TB |
| XLarge (2080 ranks) | ~5.2 TB | 1.4 TB | ~6.6 TB | 9.6 TB |
| Sweep (5200 ranks) | ~13 TB | 3.5 TB | ~16.5 TB | 24 TB |

All well within PBS allocations. Without restart, the large jobs
would OOM at ~30 iterations.

## Configuring

| Parameter | Where | Default | Description |
|-----------|-------|---------|-------------|
| `ITERS_PER_RESTART` | PBS script | 10 | CE iterations per restart segment |
| `MAX_RESTARTS` | PBS script | 20 | Max restart segments (failsafe) |
| `max_iter` | estimation YAML | 200 | Global iteration limit |
| `--max-iter-this-run` | CLI (from PBS) | None | Per-segment limit |
| `--run-id` | CLI (from PBS) | auto | Shared run ID across segments |
| `--resume` | CLI (from PBS) | False | Resume from checkpoint |
