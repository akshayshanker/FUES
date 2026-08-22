# Periodic MPI restart for memory management

## Problem

Each CE iteration increases each rank's resident memory (RSS) by
roughly 70 MB, because transient Python objects overlap with the
next solve's peak allocation and ratchet the high-water mark upward.
At 1040 ranks × 70 MB × 30 iterations = 2.2 TB of growth, jobs
exceed the PBS memory allocation.

## Solution

The PBS script runs a bash loop around `mpiexec`. Every K iterations
(K=3 in the standard PBS scripts, 25 in the hugemem variants),
Python exits cleanly, the OS reclaims all memory, and a new
`mpiexec` resumes from the last checkpoint. The loop is
time-bounded: it stops before the PBS walltime is exhausted rather
than after a fixed number of restarts.

```
PBS job starts
│
├─ Segment 1: mpiexec ... --max-iter-this-run K --run-id $RUN_ID
│   ├─ CE iters 0..K-1
│   ├─ Checkpoint saved (state.pkl)
│   ├─ All ranks exit with code 42
│   └─ RSS reset to 0
│
├─ Segment 2: mpiexec ... --max-iter-this-run K --run-id $RUN_ID --resume
│   ├─ Load state.pkl → resume from iter K
│   ├─ CE iters K..2K-1
│   ├─ Checkpoint saved
│   ├─ All ranks exit with code 42
│   └─ RSS reset to 0
│
├─ ... (repeat until converged, error, or walltime budget nearly spent)
│
└─ Final segment: converged → exit code 0 → write results
```

## PBS script structure

```bash
ITERS_PER_RESTART=3             # 25 in the hugemem variants
WALL_SECONDS=18000              # must match #PBS -l walltime
SAFETY_BUFFER=600               # reserve 10 min for checkpoint + log move
DEADLINE=$(( $(date +%s) + WALL_SECONDS - SAFETY_BUFFER ))
RUN_ID=$(date +%Y%m%d_%H%M%S)

RESTART_NUM=0
RESUME_FLAG=""
LAST_SEG_DUR=0

while : ; do
    REMAINING=$(( DEADLINE - $(date +%s) ))
    if [ $REMAINING -le 0 ]; then
        break               # walltime budget exhausted
    fi
    if [ $RESTART_NUM -gt 0 ] && [ $REMAINING -lt $LAST_SEG_DUR ]; then
        break               # not enough time left for another segment
    fi
    RESTART_NUM=$((RESTART_NUM + 1))
    SEG_START=$(date +%s)

    mpiexec -n $PBS_NCPUS ... \
        --max-iter 100000 \
        --max-iter-this-run $ITERS_PER_RESTART \
        --run-id $RUN_ID \
        $RESUME_FLAG

    EXIT_CODE=$?
    LAST_SEG_DUR=$(( $(date +%s) - SEG_START ))

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
iteration budget is exhausted:

1. The root rank computes `is_final` (has the run converged, or has
   max_iter been reached?)
2. `is_final` is broadcast to all ranks
3. If not final, all ranks call `Barrier()` and then `sys.exit(42)`
   together
4. If final, the root rank writes results and all ranks exit with
   code 0

The `Barrier()` ensures that no rank exits before the others, which
prevents an `MPI_ABORT` caused by rank disagreement.

### Sweep estimation (communicator splitting)

The world communicator is split into sub-communicators, one per sweep
point. Each sub-comm runs an independent CE loop with its own
checkpoint.

When the iteration budget is exhausted:

1. Each rank computes `is_final` for its sub-comm
2. `is_final` is broadcast within each sub-comm (so all ranks in a
   sub-comm agree)
3. An `allreduce(MIN)` across the world communicator sets
   `all_final = False` if any sub-comm is not final
4. If not all final, all ranks call `Barrier()` and then
   `sys.exit(42)`
5. If all final, the ranks gather results, write the sweep summary,
   and exit with code 0

If 9 of 10 sweep points have converged but 1 has not, all 10
restart. The 9 converged points re-converge immediately on the next
segment, because the tolerance check passes on the first iteration.
Each restart therefore wastes roughly one solve per converged point,
but this avoids the complexity of partial-communicator management.

**Idle ranks**: within a restart segment, fast-converging sub-comms
finish their K iterations before slow ones. Those ranks sit idle in
`mpi_map` waiting for the slowest sub-comm's ranks to complete their
evaluations. This is inherent to the flat MPI topology — all ranks
must participate in the world-level `allreduce` and `Barrier` at the
end of the segment.

## Checkpoint contract

`state.pkl` contains everything needed to resume the run:

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

The checkpoint is written atomically (to a temporary file that is
then renamed) to prevent corruption if the job is killed mid-write.

## Run ID

The `--run-id` flag ensures all restart segments use the same results
directory. Without it, each segment generates a new timestamp and
creates a new directory, and the resume cannot find the previous
segment's checkpoint.

The PBS script generates `RUN_ID` once before the loop and passes it
to every segment. The Python code uses `args.run_id` if provided,
otherwise falls back to `datetime.now()`.

## Memory budget

The table below is illustrative and assumes K=10 iterations per
restart; the deployed scripts use K=3 (or K=25 on hugemem), so the
per-segment growth is smaller:

| | Base RSS | Growth (10 iters) | Peak | Allocation |
|---|----------|-------------------|------|------------|
| Large (1040 ranks) | ~2.6 TB | 0.7 TB | ~3.3 TB | 4.8 TB |
| XLarge (2080 ranks) | ~5.2 TB | 1.4 TB | ~6.6 TB | 9.6 TB |
| Sweep (5200 ranks) | ~13 TB | 3.5 TB | ~16.5 TB | 24 TB |

All of these peaks sit well within the PBS allocations. Without the
restart loop, the large jobs would run out of memory at roughly 30
iterations.

## Configuring

| Parameter | Where | Default | Description |
|-----------|-------|---------|-------------|
| `ITERS_PER_RESTART` | PBS script | 3 (25 in hugemem scripts) | CE iterations per restart segment |
| `WALL_SECONDS` | PBS script | matches `#PBS -l walltime` | Walltime budget for the restart loop |
| `SAFETY_BUFFER` | PBS script | 600 | Seconds reserved for final checkpoint + log move |
| `max_iter` | estimation YAML | 200 | Global iteration limit |
| `--max-iter` | CLI (from PBS) | None | Overrides YAML `max_iter`; PBS scripts pass 100000 so walltime binds |
| `--max-iter-this-run` | CLI (from PBS) | None | Per-segment limit |
| `--run-id` | CLI (from PBS) | auto | Shared run ID across segments |
| `--resume` | CLI (from PBS) | False | Resume from checkpoint |
