# Agent spec: MPI + Python memory growth investigation

## Objective

Investigate and resolve the memory growth pattern observed in MPI Python
jobs running cross-entropy SMM estimation on NCI Gadi (PBS Pro, Linux,
Sapphire Rapids nodes, 480 GB/node).

## The problem

Each MPI rank runs a Python process that repeatedly:
1. Solves a dynamic programming model (backward induction, ~50 periods)
2. Simulates 10,000 agents forward (50 periods)
3. Computes scalar moments from simulation panels
4. Returns a scalar loss

Each iteration allocates ~1.5-2 GB of numpy arrays per rank, then frees
them via `del` + `gc.collect()`. Despite explicit cleanup, **RSS grows
monotonically across iterations** at ~50 MB/iter/rank. After 30-50
iterations, 520-1040 MPI ranks collectively exhaust the PBS memory
allocation (2.4-4.8 TB) and get killed.

Jobs that converge quickly (11-16 iterations) complete fine. Jobs needing
30+ iterations OOM.

## Environment

- Python 3.12.1
- numpy (latest), numba (latest), mpi4py
- OpenMPI 4.1.5
- Linux (Gadi: Rocky Linux 8, kernel 4.18)
- PBS Pro scheduler
- Each rank is an independent Python process (no shared memory)
- 104 cores per node (Sapphire Rapids), 480 GB RAM per node
- 5-10 nodes per job (520-1040 ranks)

## What we've already tried

1. **`del` + `gc.collect()`** after each solve/simulate — helps but
   doesn't stop the growth
2. **Incremental nest stripping** — drop unused solution arrays during
   the backward loop (reduces peak per solve by ~70%)
3. **`ctypes.CDLL(None).malloc_trim(0)`** after `gc.collect()` — just
   added, not yet tested on Gadi
4. **Skip post-convergence re-evaluation** — avoids one extra
   solve+simulate after convergence

## Observed memory patterns

| Run type | Ranks | Grid | Iters | Mem used | Mem alloc | Outcome |
|----------|-------|------|-------|----------|-----------|---------|
| selfgen 600-grid | 1040 | 600 | 13 | 3.34 TB | 4.69 TB | OK |
| selfgen 600-grid | 1040 | 600 | 16 | 3.59 TB | 4.69 TB | OK |
| precomputed 300-grid | 520 | 300 | 53 | 2.34 TB | 2.34 TB | KILLED |
| precomputed 600-grid | 1040 | 600 | 31 | 4.64 TB | 4.69 TB | OOM |
| precomputed 600-grid | 1040 | 600 | 32 | 4.67 TB | 4.69 TB | KILLED |
| sweep 300-grid | 1040 | 300 | 214 | 2.42 TB | 4.69 TB | OK (104 ranks/sub) |

Key observations:
- Memory grows linearly with iterations
- Base footprint (~2.5 TB for 1040 ranks) accounts for Python + numba JIT
- Growth rate ~50 MB/iter total across all ranks (~50 KB/iter/rank)
- Short runs (selfgen, 11-16 iters) complete fine
- Long runs (precomputed, 30-53 iters) hit the limit
- Sweeps with sub-communicators (104 ranks each) stay within budget even
  at 200+ iterations

## Questions to investigate

### 1. glibc malloc fragmentation
- Does `malloc_trim(0)` actually reclaim memory on this kernel version?
- Are there better alternatives (`MALLOC_TRIM_THRESHOLD_`, `M_TRIM_THRESHOLD`)?
- Should we set `MALLOC_MMAP_THRESHOLD_` to force large allocations to
  use mmap (which is returned to OS immediately on free)?
- Does jemalloc or tcmalloc perform better than glibc malloc for this
  allocation pattern?
- Relevant env vars: `MALLOC_TRIM_THRESHOLD_=-1`, `MALLOC_MMAP_THRESHOLD_=65536`

### 2. Python memory allocator (pymalloc)
- Is pymalloc holding arenas that prevent glibc from releasing pages?
- Does `PYTHONMALLOC=malloc` (bypass pymalloc, use raw malloc) help?
- What about `PYTHONMALLOC=debug` for leak detection?

### 3. numpy allocation patterns
- numpy uses `PyMem_RawMalloc` for arrays > 512 bytes (bypasses pymalloc)
- Smaller arrays and array headers go through pymalloc
- Does numpy's internal free list cache prevent page release?
- `numpy.core.multiarray._set_madvise_hugepage(0)` — does this help?

### 4. numba JIT memory
- numba compiles functions on first call and caches in memory
- Each rank compiles independently (no shared JIT cache across MPI ranks)
- Does numba leak memory across calls to the same compiled function?
- Is the LLVM compilation memory freed after the first call?

### 5. mpi4py serialization
- `comm.bcast` pickles objects on root, unpickles on all ranks
- `comm.gather` pickles results from all ranks
- Does pickle buffer memory accumulate?
- Should we use `comm.Bcast` (uppercase, buffer-based) instead of
  `comm.bcast` (lowercase, pickle-based) for large objects?

### 6. dolo model objects
- Each `solve()` call loads and instantiates dolo SymbolicModel objects
- These contain compiled expression trees and cached evaluators
- Do they accumulate across calls even though the same model is reloaded?
- Module-level `_callables_cache` prevents reloading but may hold
  references to old model objects

### 7. Circular references
- dolo stages reference their parent period, which references stages
- `gc.collect()` handles circular references but only if there are no
  `__del__` methods in the cycle
- Are there any `__del__` methods in the dolo/kikku object graph?

## Code locations

- CE loop: `kikku/kikku/run/estimate.py` lines 315-381
- Trial function: `examples/durables2_0/estimate.py` `_run_single_estimation.trial()`
- Solve: `examples/durables2_0/solve.py` `solve()` → `accrete_and_solve()`
- Simulate: `examples/durables2_0/horses/simulate.py` `simulate_lifecycle()`
- Nest stripping: `solve.py` `_strip_old_solution()`
- MPI helpers: `kikku/kikku/run/mpi.py`
- Criterion closure: `kikku/kikku/run/estimate.py` `make_criterion()`

## Search terms for GitHub issues

- `numpy memory leak mpi` / `numpy RSS growth`
- `python malloc_trim mpi` / `glibc malloc fragmentation python`
- `numba memory leak repeated calls`
- `mpi4py memory growth pickle bcast`
- `python gc.collect not releasing memory`
- `MALLOC_MMAP_THRESHOLD numpy`
- `jemalloc python numpy`
- `pymalloc arena fragmentation`

## Expected deliverables

1. Root cause analysis: which component (glibc, pymalloc, numpy, numba,
   mpi4py, dolo) is responsible for the ~50 KB/iter/rank growth
2. Concrete fix: env vars, code changes, or allocator swaps that
   stabilise RSS across iterations
3. Verification approach: how to measure RSS per rank per iteration
   (e.g. `resource.getrusage(RUSAGE_SELF).ru_maxrss` logged each iter)
4. Any known issues in numpy/numba/mpi4py GitHub repos that match
   this pattern

## Constraints

- Cannot change the MPI topology (flat, one rank per core)
- Cannot reduce the number of iterations (precomputed estimation needs
  30-50+ to converge)
- Can change malloc implementation (jemalloc, tcmalloc) if needed
- Can set env vars in PBS scripts
- Python 3.12 on Gadi (cannot change version easily)
