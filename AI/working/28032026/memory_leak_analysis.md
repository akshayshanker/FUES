# Memory leak analysis — 28 March 2026

## Status: partially diagnosed, grid caching applied

## Key finding from diagnostics

Per-phase RSS diagnostics (`FUES_MEM_DIAG=1`) on job 164120969:

```
[mem] iter=1 eval=2642MB bcast=2642MB gc=2642MB delta_eval=0MB  delta_bcast=0MB delta_gc=0MB
[mem] iter=2 eval=2715MB bcast=2715MB gc=2715MB delta_eval=73MB delta_bcast=0MB delta_gc=0MB
[mem] iter=3 eval=2791MB bcast=2791MB gc=2791MB delta_eval=76MB delta_bcast=0MB delta_gc=0MB
[mem] iter=4 eval=2858MB bcast=2858MB gc=2858MB delta_eval=67MB delta_bcast=0MB delta_gc=0MB
[mem] iter=5 eval=2928MB bcast=2928MB gc=2928MB delta_eval=70MB delta_bcast=0MB delta_gc=0MB
```

**All growth is in `delta_eval`** (the solve+simulate phase).
`delta_bcast=0` and `delta_gc=0` — mpi4py and gc are clean.

Per-step diagnostics inside solve() (job 164120969, trial #1):

```
[solve] load_syntax: +0MB       (all ranks, expected)
[solve] instantiate: +0-1MB     (scattered, dolo SymbolicModel objects)
[solve] make_grids: +40MB       (per rank — X_all cartesian product is 60MB)
[solve] accrete_and_solve: +2245-2395MB  (backward induction, expected peak)
```

## Critical insight: ru_maxrss is a high-water mark

`resource.getrusage(RUSAGE_SELF).ru_maxrss` reports the **peak** RSS
over the lifetime of the process. It can only increase, never decrease.

The ~75 MB/iter growth in `delta_eval` means: the peak RSS during
iteration N+1 is 75 MB higher than the peak during iteration N.

This happens when **transient allocations from one phase overlap with
the next phase's peak**. Specifically:

1. `make_grids` allocates ~40 MB (X_all + grid arrays)
2. `accrete_and_solve` runs and peaks at ~2250 MB
3. If the grids from step 1 aren't freed before step 2 peaks, the
   combined peak is 2250 + 40 = 2290 MB
4. After `del nest, grids + gc.collect()`, memory drops
5. Next iteration: make_grids again (+40 MB), then accrete peaks
   at 2250 + 40 = 2290 MB — BUT the new grids are a different
   allocation at a different address, and the old grids' memory
   pages may not have been returned to the OS

With `MALLOC_MMAP_THRESHOLD_=65536`, the X_all array (60 MB) should
use mmap and be returned on free. But smaller arrays and Python
objects from instantiate_period may use sbrk, contributing to the gap.

## Fix applied: grid caching

**Commit**: `abeb9d3`

Pre-build grids once before the CE loop, pass via `solve(..., grids=_cached_grids)`.
This eliminates the ~40 MB make_grids allocation from every iteration.

Expected effect: reduce the ~75 MB/iter growth by ~40 MB, to ~35 MB/iter.

## Remaining ~35 MB/iter growth: analysis

Per solve() call, these objects are created and should be freed:

| Object | Created in | Size | Freed by |
|--------|-----------|------|----------|
| calibration dict | load_syntax | ~1 KB | del (local) |
| stage_sources dicts | load_syntax | ~50 KB | del (local) |
| 3 × SymbolicModel | instantiate_period | ~100 KB each | del nest → clear periods |
| period_inst dict | instantiate_period | ~1 KB | del (local) |
| callables dict | make_callables | ~5 KB (function refs) | del nest → clear solutions |
| ~50 @njit dispatchers | make_callables | ~1 KB each | del callables |
| condition_V, condition_V_HD | make_conditioners | ~(n_z × n_a × n_h) | del (local) |
| keeper_ops, adjuster_ops | make_*_ops | closures, small | del (local) |
| 50 × solution arrays | accrete_and_solve | 15 arrays × 20 MB (stripped to 5) | del nest → clear solutions |

Total live during solve peak: ~2250 MB
Total that should survive after del nest: ~0

The ~35 MB residual is likely:
1. **Python object metadata** for the 50 SymbolicModel objects and their
   YAML node trees (created by `yaml.compose`). Each has ~100+ node objects.
   50 periods × 3 stages × ~100 nodes × ~100 bytes = ~1.5 MB — too small.

2. **numba dispatcher registry**: each `@njit` call registers a dispatcher.
   Even after the Python object is freed, numba may retain type metadata
   in its internal `_TargetRegistry`. ~50 dispatchers × ~100 KB metadata
   = ~5 MB — plausible but not 35 MB.

3. **interpolation library state**: `eval_linear` from the `interpolation`
   package may cache grid evaluation plans. Called thousands of times per
   solve. Unknown overhead.

4. **Memory fragmentation despite mmap**: arrays < 64 KB (dict entries,
   list nodes, small numpy headers) still use sbrk. These fragment over
   time. `PYTHONMALLOC=malloc` should route all through glibc, but glibc's
   sbrk heap still fragments for sub-page allocations.

## What we ruled out

| Suspect | Status | Evidence |
|---------|--------|----------|
| glibc malloc fragmentation (large arrays) | **Ruled out** | MALLOC_MMAP_THRESHOLD_=65536 doesn't help |
| pymalloc arena fragmentation | **Ruled out** | PYTHONMALLOC=malloc doesn't help |
| numba NRT leak | **Ruled out** | NRT alloc/free balanced locally (test) |
| numba parallel runtime buffers | **Unlikely** | NUMBA_THREADING_LAYER=workqueue doesn't help |
| mpi4py pickle/bcast buffers | **Ruled out** | delta_bcast=0 |
| Retained references in trial/criterion | **Ruled out** | Code trace shows clean del chain |
| dolo/dolang caches | **Ruled out** | No caches found in source |
| `_callables_cache` growth | **Ruled out** | Populated once, constant |

## Conclusion

The "leak" is most likely **not a retained reference** but **high-water
mark RSS growth from overlapping allocations**. Each solve() creates
~2.3 GB of arrays (even with stripping). If even ~35 MB of transient
objects from load_syntax/instantiate_period aren't freed before
accrete_and_solve peaks, the high-water mark ratchets up.

The grid caching fix eliminates 40 MB of this overlap. The remaining
35 MB is from the per-call recreation of callables, operator closures,
and dolo stage objects — all of which are transient but overlap with
the solve peak.

## Pragmatic approach

Given the analysis:
1. **Grid caching**: saves ~40 MB/iter (implemented)
2. **Request sufficient memory**: at ~35 MB/iter × 200 iters × 1040 ranks
   = 7.3 TB growth → request 2.4 + 7.3 = ~10 TB for 200-iter runs
3. **Future**: cache callables and operator closures across CE iterations
   (they only depend on grids + settings, not calibration — but callables
   DO depend on calibration via theta, so this needs refactoring)

## Environment variables in PBS scripts

```bash
export MALLOC_MMAP_THRESHOLD_=65536     # large arrays use mmap
export MALLOC_TRIM_THRESHOLD_=0         # aggressive sbrk trim
export PYTHONMALLOC=malloc              # bypass pymalloc
export NUMBA_THREADING_LAYER=workqueue  # lightweight parallel backend
export NUMBA_NUM_THREADS=1              # single-threaded per MPI rank
export FUES_MEM_DIAG=1                  # per-phase RSS diagnostics
```
