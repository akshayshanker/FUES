# Devspec: Periodic MPI restart to reset RSS

**Date**: 28 March 2026
**Status**: Design
**Priority**: High — RSS growth of ~70 MB/iter/rank causes OOM at ~30 iterations

## Problem

Each CE iteration leaks ~70 MB per rank due to high-water mark RSS
ratcheting from overlapping Python object lifetimes (load_syntax,
instantiate_period, make_callables, solve). This is not a true memory
leak — objects are freed — but the peak RSS (ru_maxrss) grows because
transient allocations from one phase overlap with the next phase's peak.

At 1040 ranks × 70 MB/iter × 30 iters = 2.2 TB growth on top of a
2.6 TB base = 4.8 TB, exactly the PBS limit. Jobs that need 50+
iterations OOM.

## Solution: periodic restart

Kill the `mpiexec` process every K iterations (e.g. K=10), then
restart it. Python exits, OS reclaims ALL memory. On restart, the CE
resumes from the last checkpoint.

```
PBS bash loop:
  iter 1-10:  mpiexec python3 ... --max-iter-this-run 10
  iter 11-20: mpiexec python3 ... --resume --max-iter-this-run 10
  iter 21-30: mpiexec python3 ... --resume --max-iter-this-run 10
  ...until converged or total max_iter reached
```

## Checkpoint contract

The CE loop already saves `state.pkl` each iteration:

```python
state = {
    "means": means,        # dict[str, float] — elite mean
    "cov": cov,            # np.ndarray — elite covariance
    "best_theta": best_theta,  # dict[str, float]
    "best_loss": best_loss,    # float
    "it": it,                  # int — 0-indexed iteration number
}
```

### What must be added to state.pkl for seamless resume

```python
state = {
    # Existing:
    "means": means,
    "cov": cov,
    "best_theta": best_theta,
    "best_loss": best_loss,
    "it": it,
    # New — required for seamless continuation:
    "history": history,              # list[dict] — full convergence trace
    "elite_mean_loss_prev": elite_mean_loss_prev,  # float — for tol check
    "rng_state": rng.bit_generator.state,  # for reproducible sampling
}
```

### Why each field matters

- **`it`**: the iteration counter. On resume, start at `it + 1`. The
  total `max_iter` from the YAML is the global limit (e.g. 200), not
  per-restart. If `it + 1 >= max_iter`, don't restart.

- **`history`**: the convergence trace (means, cov, best_loss,
  elite_mean_loss per iteration). Without this, the final results
  file has a gap in the convergence CSV. The resumed run must
  **append** to the existing history, not start fresh.

- **`elite_mean_loss_prev`**: the tolerance convergence check compares
  consecutive iterations: `abs(prev - current) < tol`. Without this,
  the first iteration after restart can't check convergence.

- **`rng_state`**: the CE parameter sampling RNG. Without this,
  restart draws different candidates than a non-restarted run would.
  For reproducibility, save and restore the RNG state. (Optional —
  results will still be valid without it, just not bit-identical.)

## Implementation

### 1. kikku CE loop (`_cross_entropy_minimize`)

Add `resume_state` parameter:

```python
def _cross_entropy_minimize(
    criterion, param_spec, options, comm, verbose,
    resume_state=None,   # NEW: loaded state.pkl dict, or None
):
```

On startup:
```python
if resume_state is not None:
    means = resume_state["means"]
    cov = np.asarray(resume_state["cov"])
    best_theta = resume_state["best_theta"]
    best_loss = resume_state["best_loss"]
    start_iter = resume_state["it"] + 1
    history = resume_state.get("history", [])
    elite_mean_loss_prev = resume_state.get("elite_mean_loss_prev")
    rng_state = resume_state.get("rng_state")
    if rng_state is not None:
        rng.bit_generator.state = rng_state
else:
    start_iter = 0
    history = []
    elite_mean_loss_prev = None
```

The main loop becomes:
```python
for it in range(start_iter, max_iter):
    ...
```

Save expanded state each iteration:
```python
state = {
    "means": means,
    "cov": cov,
    "best_theta": best_theta,
    "best_loss": best_loss,
    "it": it,
    "history": history,
    "elite_mean_loss_prev": elite_mean_loss_prev,
    "rng_state": rng.bit_generator.state,
}
```

### 2. estimate.py

Add `--resume` CLI flag:

```python
parser.add_argument('--resume', action='store_true',
    help='Resume from latest checkpoint in scratch dir')
```

When `--resume`:
```python
# Find the latest checkpoint
checkpoint_path = find_latest_checkpoint(scratch_dir, spec_name)
with open(checkpoint_path, 'rb') as f:
    resume_state = pickle.load(f)
# Use the SAME run_id as the original run
run_id = extract_run_id(checkpoint_path)
```

Pass to estimate:
```python
result = estimate(
    criterion, param_spec,
    method=spec['method'],
    method_options=method_options,
    comm=comm,
    verbose=is_root(comm),
    resume_state=resume_state,   # NEW
)
```

### 3. PBS script

```bash
#!/bin/bash
#PBS -P tp66
#PBS -q normalsr
#PBS -N est_lgEGM
#PBS -l ncpus=1040
#PBS -l mem=4800GB
#PBS -l walltime=5:00:00
...

ITERS_PER_RESTART=10
MAX_ITER=200  # from YAML, but enforced here too
CONVERGED=0
RESTART_COUNT=0

while [ $CONVERGED -eq 0 ] && [ $RESTART_COUNT -lt 20 ]; do
    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "=== Restart $RESTART_COUNT at $(date) ==="

    if [ $RESTART_COUNT -eq 1 ]; then
        RESUME_FLAG=""
    else
        RESUME_FLAG="--resume"
    fi

    mpiexec -n $PBS_NCPUS \
        --map-by ppr:13:numa \
        python3 -u -m mpi4py -m examples.durables2_0.estimate \
            --mod "$MOD" \
            --spec "$SPEC" \
            --scratch /scratch/tp66/$USER/durables_est \
            --results /g/data/tp66/results/durables2_0/estimation \
            --setting-override n_a=$GRID n_h=$GRID n_w=$GRID \
            --calib-override t0=20 \
            --N-sim $N_SIM \
            --max-iter-this-run $ITERS_PER_RESTART \
            $RESUME_FLAG

    EXIT_CODE=$?

    # Exit code 0 = converged or max_iter reached
    # Exit code 42 = restart requested (not converged, more iters needed)
    if [ $EXIT_CODE -ne 42 ]; then
        CONVERGED=1
    fi

    echo "=== Restart $RESTART_COUNT finished, exit=$EXIT_CODE ==="
done

echo "Finished at $(date), $RESTART_COUNT restarts"
```

### 4. Exit codes

| Code | Meaning | PBS action |
|------|---------|------------|
| 0 | Converged or max_iter reached | Stop loop, write results |
| 42 | Restart needed (iter budget for this run exhausted) | Loop continues |
| 1 | Error | Stop loop |

In `estimate.py`:
```python
import sys

if result.converged or (start_iter + iters_this_run >= max_iter):
    # Final: write results
    ...
    sys.exit(0)
else:
    # More iterations needed, checkpoint already saved
    if is_root(comm):
        print(f"Restart requested: completed iter {result.n_iter}, "
              f"best_loss={result.objective:.6f}")
    sys.exit(42)
```

### 5. `--max-iter-this-run` CLI flag

```python
parser.add_argument('--max-iter-this-run', type=int, default=None,
    help='Max iterations for this restart segment. '
         'Overrides YAML max_iter for this run only.')
```

When set, the CE loop runs at most this many iterations, then exits
with code 42 if not converged. The YAML `max_iter` remains the global
limit.

## Seamless continuation guarantees

1. **Iteration numbering**: continuous across restarts. If restart 1
   runs iters 0-9, restart 2 starts at iter 10. The convergence CSV
   has no gaps.

2. **History**: accumulated across restarts. state.pkl includes the
   full history list. The final results file has the complete trace.

3. **RNG**: restored from checkpoint. Restart 2 draws the same
   candidates as a non-restarted run would at iter 10. (If RNG state
   isn't saved, draws differ but estimation is still valid.)

4. **Convergence check**: `elite_mean_loss_prev` is saved. The tol
   check works across restart boundaries.

5. **Results directory**: same `run_id` across all restarts. The
   final results overwrite (not duplicate) the checkpoint.

6. **Idempotent**: if a restart is killed mid-iteration, the
   checkpoint from the last completed iteration is still valid.
   Re-running `--resume` picks up from there.

## Memory effect

Each restart resets RSS to the base footprint (~2.6 TB for 1040 ranks).
With K=10:
- Peak per segment: 2.6 + 0.07 × 10 = 3.3 TB (well under 4.8 TB)
- 20 restarts × 10 iters = 200 iterations capacity
- Restart overhead: ~60s for numba JIT warmup per restart

## Files to change

| File | Change |
|------|--------|
| `kikku/kikku/run/estimate.py` | `resume_state` param, expanded state.pkl, `start_iter` |
| `examples/durables2_0/estimate.py` | `--resume`, `--max-iter-this-run`, exit codes, checkpoint loading |
| `experiments/durables/estimation/*.pbs` | Bash restart loop |

## Open questions

1. **Numba warmup**: each restart pays ~60s for JIT compilation on
   the first solve. With K=10 and 20 restarts, that's ~20 min overhead
   on a 2-hour job. Acceptable?

2. **K value**: 10 iters per restart means ~700 MB growth before reset.
   Could do K=20 (~1.4 GB growth) for fewer restarts but tighter memory.
   Make it configurable in PBS (`ITERS_PER_RESTART=10`).

3. **Exit code 42**: PBS Pro may interpret non-zero exit codes as
   failures for job dependency chains. Use a file flag instead?
   e.g. touch `$SCRATCH/.restart_needed` vs `$SCRATCH/.converged`.

4. **Multiple specs**: if running sweeps (comm split), each sub-comm
   may converge at different iterations. The restart loop would need
   to wait for ALL sub-comms. Simpler to disable restart for sweeps
   initially.

---

## Detailed implementation plan

### Context for the implementing agent

This is a cross-entropy (CE) SMM estimation loop that runs on NCI Gadi
via MPI. The codebase has two repos:

- **FUES** (`examples/durables2_0/estimate.py`): the estimation entry
  point. Installed editable — changes take effect on `git pull`.
- **kikku** (`kikku/kikku/run/estimate.py`): the CE optimizer and MPI
  helpers. Installed from GitHub — changes need `pip install --force-reinstall`.

The CE loop is in kikku's `_cross_entropy_minimize`. The entry point
is FUES's `estimate.py` which calls `kikku.run.estimate.estimate()`.

PBS scripts in `experiments/durables/estimation/` invoke the entry point
via `mpiexec python3 -u -m mpi4py -m examples.durables2_0.estimate`.

### Step-by-step implementation

#### Step 1: Expand state.pkl in kikku

**File**: `kikku/kikku/run/estimate.py`
**Function**: `_cross_entropy_minimize`, line ~380-385

Current checkpoint save (inside the `if checkpoint_dir:` block):
```python
state = {"means": means, "cov": cov, "best_theta": best_theta, "best_loss": best_loss, "it": it}
```

**Change to**:
```python
state = {
    "means": means,
    "cov": cov,
    "best_theta": best_theta,
    "best_loss": best_loss,
    "it": it,
    "history": history,
    "elite_mean_loss_prev": elite_mean_loss_prev,
    "rng_state": rng.bit_generator.state,
}
```

This is backward compatible — old state.pkl files just won't have the
new keys, and the resume code uses `.get()` with defaults.

#### Step 2: Add resume_state parameter to _cross_entropy_minimize

**File**: `kikku/kikku/run/estimate.py`
**Function**: `_cross_entropy_minimize`

Change signature from:
```python
def _cross_entropy_minimize(
    criterion, param_spec, options, comm, verbose,
) -> EstimationResult:
```

To:
```python
def _cross_entropy_minimize(
    criterion, param_spec, options, comm, verbose,
    resume_state=None,
) -> EstimationResult:
```

Replace the initialization block (lines ~319-333):
```python
    means: dict[str, float] | None = None
    cov: np.ndarray | None = None
    rng = np.random.default_rng(sampling_seed)
    history: list[dict[str, Any]] = []
    best_theta: dict[str, float] = {
        n: 0.5 * (float(param_spec[n]["bounds"][0]) + float(param_spec[n]["bounds"][1]))
        for n in names
    }
    best_loss = BIG_LOSS
    converged = False
    elite_mean_loss_prev: float | None = None
    rss_post_gc_prev = 0

    for it in range(max_iter):
```

With:
```python
    rng = np.random.default_rng(sampling_seed)

    if resume_state is not None:
        means = resume_state["means"]
        cov = np.asarray(resume_state["cov"])
        best_theta = dict(resume_state["best_theta"])
        best_loss = float(resume_state["best_loss"])
        start_iter = int(resume_state["it"]) + 1
        history = list(resume_state.get("history", []))
        elite_mean_loss_prev = resume_state.get("elite_mean_loss_prev")
        rng_state = resume_state.get("rng_state")
        if rng_state is not None:
            rng.bit_generator.state = rng_state
        if is_root(comm) and verbose:
            print(f"[ce] Resuming from iter {start_iter}, "
                  f"best_loss={best_loss:.6f}")
    else:
        means = None
        cov = None
        best_theta = {
            n: 0.5 * (float(param_spec[n]["bounds"][0]) + float(param_spec[n]["bounds"][1]))
            for n in names
        }
        best_loss = BIG_LOSS
        start_iter = 0
        history = []
        elite_mean_loss_prev = None

    converged = False
    rss_post_gc_prev = 0

    for it in range(start_iter, max_iter):
```

**CRITICAL**: the loop range changes from `range(max_iter)` to
`range(start_iter, max_iter)`. The `it` variable is the GLOBAL
iteration number, not relative to this restart segment.

#### Step 3: Add max_iter_this_run support

In the same function, after reading options:
```python
    max_iter = int(options.get("max_iter", 50))
    max_iter_this_run = int(options.get("max_iter_this_run", max_iter))
```

In the loop, after the convergence check and before the gc.collect():
```python
        # Check if this restart segment's budget is exhausted
        iters_done_this_run = it - start_iter + 1
        if iters_done_this_run >= max_iter_this_run and not converged:
            if is_root(comm) and verbose:
                print(f"[ce] Restart budget exhausted ({iters_done_this_run} iters). "
                      f"Checkpoint saved at iter {it + 1}.")
            break
```

This must go AFTER the checkpoint save (so state.pkl is current) and
AFTER the `converged = bcast_item(...)` line (so all ranks agree).

#### Step 4: Wire resume_state through estimate()

**File**: `kikku/kikku/run/estimate.py`
**Function**: `estimate`

Change:
```python
def estimate(
    criterion, param_spec, method="cross-entropy",
    method_options=None, comm=None, verbose=True,
) -> EstimationResult:
```

To:
```python
def estimate(
    criterion, param_spec, method="cross-entropy",
    method_options=None, comm=None, verbose=True,
    resume_state=None,
) -> EstimationResult:
```

And pass it through:
```python
    if method in ("cross-entropy", "ce", "cross_entropy"):
        return _cross_entropy_minimize(criterion, param_spec, opts, comm, verbose,
                                       resume_state=resume_state)
```

#### Step 5: Add --resume and --max-iter-this-run to estimate.py

**File**: `examples/durables2_0/estimate.py`
**Function**: `main()` argument parser

Add two new arguments:
```python
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from latest checkpoint in scratch dir')
    parser.add_argument(
        '--max-iter-this-run', type=int, default=None,
        help='Max CE iterations for this restart segment. '
             'Exits with code 42 when exhausted (not converged).')
```

#### Step 6: Implement resume logic in _run_single_estimation

**File**: `examples/durables2_0/estimate.py`
**Function**: `_run_single_estimation`

After building the criterion and before calling `estimate()`, add:

```python
    # --- Resume from checkpoint if --resume ---
    resume_state = None
    if args.resume:
        state_path = os.path.join(scratch_run, 'state.pkl')
        if os.path.exists(state_path):
            import pickle as _pkl
            with open(state_path, 'rb') as f:
                resume_state = _pkl.load(f)
            if is_root(comm):
                print(f"  Resuming from checkpoint: iter {resume_state['it'] + 1}")
        elif is_root(comm):
            print(f"  WARNING: --resume but no state.pkl found. Starting fresh.")
```

**CRITICAL for resume**: the `run_id` must be the SAME as the original
run so results go to the same directory. When `--resume`, extract the
run_id from the checkpoint path instead of generating a new timestamp.

Currently `run_id` is generated at the top of `_run_single_estimation`:
```python
    run_id = ...  # passed from main()
```

For resume, the `main()` function must find the LATEST existing run_id
under `scratch_dir/spec_name/` and reuse it:

```python
    if args.resume:
        # Find latest run dir in scratch
        spec_scratch = os.path.join(scratch_dir, spec_name)
        if os.path.isdir(spec_scratch):
            run_dirs = sorted([d for d in os.listdir(spec_scratch)
                              if d.startswith('est_') and
                              os.path.isfile(os.path.join(spec_scratch, d, 'state.pkl'))])
            if run_dirs:
                run_id = run_dirs[-1].replace('est_', '')
                if is_root(world_comm):
                    print(f"  Resuming run_id: {run_id}")
```

Pass `max_iter_this_run` to the CE options:
```python
    if args.max_iter_this_run is not None:
        method_options['max_iter_this_run'] = args.max_iter_this_run
```

Pass `resume_state` to `estimate()`:
```python
    result = estimate(
        criterion, param_spec,
        method=spec['method'],
        method_options=method_options,
        comm=comm,
        verbose=is_root(comm),
        resume_state=resume_state,
    )
```

#### Step 7: Exit codes

After the estimation completes and results are saved:

```python
    # Determine exit code for restart loop
    if result.converged:
        sys.exit(0)  # converged — stop restarting
    elif args.max_iter_this_run is not None:
        # Check if we've hit the global max_iter
        global_iter = result.n_iter  # total iters including all restarts
        global_max = int(method_options.get('max_iter', 200))
        if global_iter >= global_max:
            sys.exit(0)  # hit global limit — stop
        else:
            sys.exit(42)  # more iters needed — restart
    else:
        sys.exit(0)  # no restart mode — normal exit
```

**Note**: `result.n_iter` must reflect the GLOBAL iteration count
(including prior restarts), not the count for this segment. This is
automatic if history is accumulated across restarts — `result.n_iter
= len(result.history)`.

#### Step 8: PBS bash restart loop

**File**: `experiments/durables/estimation/run_large_egm.pbs`

Replace the single `mpiexec` call with a restart loop:

```bash
ITERS_PER_RESTART=10
MAX_RESTARTS=20

RESTART_NUM=0
RESUME_FLAG=""

while [ $RESTART_NUM -lt $MAX_RESTARTS ]; do
    RESTART_NUM=$((RESTART_NUM + 1))
    echo ""
    echo "=== Restart segment $RESTART_NUM at $(date) ==="

    mpiexec -n $PBS_NCPUS \
        --map-by ppr:13:numa \
        python3 -u -m mpi4py -m examples.durables2_0.estimate \
            --mod "$MOD" \
            --spec "$SPEC" \
            --scratch /scratch/tp66/$USER/durables_est \
            --results /g/data/tp66/results/durables2_0/estimation \
            --setting-override n_a=$GRID n_h=$GRID n_w=$GRID \
            --calib-override t0=20 \
            --N-sim $N_SIM \
            --max-iter-this-run $ITERS_PER_RESTART \
            $RESUME_FLAG

    EXIT_CODE=$?
    echo "=== Segment $RESTART_NUM exit code: $EXIT_CODE ==="

    if [ $EXIT_CODE -ne 42 ]; then
        echo "Estimation complete (converged or max_iter reached)."
        break
    fi

    # After first segment, all subsequent segments resume
    RESUME_FLAG="--resume"
done

echo "Total restart segments: $RESTART_NUM"
echo "Finished at $(date)"
```

#### Step 9: Results writing — only on final segment

Currently `_run_single_estimation` writes results (theta_best.json,
fit_table.csv, etc.) after every call. With restart, intermediate
segments should NOT write final results — only the checkpoint.

Add a check:
```python
    is_final = result.converged or (args.max_iter_this_run is None) or \
               (result.n_iter >= int(method_options.get('max_iter', 200)))

    if is_root(comm) and is_final:
        # Write final results (theta_best.json, fit_table.csv, etc.)
        ...
    elif is_root(comm):
        # Intermediate segment — just confirm checkpoint
        print(f"  Checkpoint saved. Will resume on next restart.")
```

### Testing plan

1. **Local test (serial, small grid)**: run 5 iters, kill, resume,
   verify iteration counter continues and convergence CSV has no gaps.

2. **Gadi test (small job)**: submit with `ITERS_PER_RESTART=3`,
   `max_iter=10` in YAML. Should do 4 restart segments. Check:
   - Iteration numbering is 1-10 in output
   - convergence.csv has 10 rows
   - state.pkl after each segment has correct `it` value
   - Final results match a non-restarted run (same seed, same iters)

3. **Production**: `ITERS_PER_RESTART=10`, `max_iter=200`. Should
   survive 200 iterations with ~3.3 TB peak per segment.

### Commit plan

1. **kikku commit 1**: expand state.pkl + add resume_state parameter
2. **FUES commit 1**: add --resume + --max-iter-this-run + exit codes
3. **FUES commit 2**: update PBS scripts with restart loop
4. **Test locally**, then deploy to Gadi
