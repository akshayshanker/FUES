# Devspec: Estimation sweep runner

**Date**: 28 March 2026
**Status**: Design

## Problem statement

We want to run K independent estimations (each a full CE-SMM loop),
where each estimation fixes one or more calibration parameters at
different values. For example: estimate 5 free parameters at each
of 10 wage mean values.

Each estimation is itself a parallel job — the CE loop uses MPI to
distribute n_draws candidate evaluations across ranks. So the
architecture is:

```
Outer: K sweep points (10 wage means)
  Inner: CE loop (n_draws parameter draws per iteration, distributed via MPI)
```

The outer loop must split MPI ranks into K groups. Each group runs
an independent CE estimation with its own sub-communicator.

## Current implementation

`estimate.py` already implements this via `comm.Split`:

```python
color = world_rank * n_points // world_size
sub_comm = world.Split(color, world_rank)
# Each sub_comm runs independent CE with n_samples = sub_comm.size
```

This works but has design issues:

1. **n_samples per point = total_ranks / K** — with 1040 ranks and
   10 points, each estimation gets 104 draws per iteration. That's
   fewer than a single-point run (1040 draws). Impacts convergence.

2. **All points run simultaneously** — good for throughput, but
   you can't inspect intermediate results from one point while
   others are running.

3. **The sweep logic lives in estimate.py** — mixed with the
   single-estimation code, making both harder to read.

## Proposed design

### Separate the runner from the estimator

Two distinct components:

```
estimate.py            — runs ONE estimation (no sweep logic)
estimate_sweep.py      — runner that invokes estimate for each sweep point
```

`estimate_sweep.py` is the outer loop. It:
1. Reads the sweep grid from the YAML
2. Splits the world communicator into K sub-communicators
3. For each sub-comm, calls the single-estimation function with
   the sweep point merged into calib_overrides
4. Gathers results from all sub-comms on world rank 0
5. Writes the sweep summary

`estimate.py` stays clean — it takes a comm and runs one CE loop.
No sweep awareness.

### MPI topology

```
PBS job: 1040 ranks
Sweep: 10 wage means

Option A: Equal split
  sub_comm_0 (104 ranks) → wage_mean=1.0, CE with 104 draws
  sub_comm_1 (104 ranks) → wage_mean=1.5, CE with 104 draws
  ...
  sub_comm_9 (104 ranks) → wage_mean=5.5, CE with 104 draws

Option B: Unequal split (if some points are harder)
  sub_comm_0 (200 ranks) → wage_mean=1.0
  sub_comm_1 (80 ranks)  → wage_mean=1.5
  ...
  (configured in YAML)

Option C: Sequential (one point at a time, full power)
  iter 1: all 1040 ranks → wage_mean=1.0
  iter 2: all 1040 ranks → wage_mean=1.5
  ...
  (slower wall time, but each point gets maximum draws)
```

**Recommendation**: Option A (equal split) as the default, with
Option C available via a `sweep_mode: sequential` flag.

### YAML syntax

```yaml
estimation:
  free: { ... }
  method: cross-entropy
  method_options:
    n_samples: 1040    # total ranks (split in parallel mode)
    n_elite: 20
    max_iter: 200
    ...

  # Sweep definition
  sweep:
    grid:
      wage_mean: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    mode: parallel     # parallel (comm split) | sequential (one at a time)
```

When `mode: parallel` (default):
- n_samples per point = total_ranks / n_points
- All points run simultaneously

When `mode: sequential`:
- n_samples per point = total_ranks
- Points run one after another
- Total wall time = K × single-point wall time

### Results directory

```
estimation/results/<spec_name>/
  sweep_20260328_140000/             # timestamped sweep run
    wage_mean=1.0/
      theta_best.json
      summary.json
      fit_table.csv
      convergence.csv
    wage_mean=1.5/
      ...
    sweep_summary.csv                # all points in one table
    sweep_config.json                # grid, mode, timestamps
```

The sweep gets a single timestamped directory. Each point is a
subdirectory. The summary CSV sits at the sweep level.

### Implementation: estimate_sweep.py

```python
def sweep_main():
    world = get_comm()
    # Load spec, extract sweep block
    spec = load_estimation_spec(spec_path)
    sweep_grid = build_sweep_grid(spec['sweep']['grid'])
    mode = spec['sweep'].get('mode', 'parallel')

    if mode == 'parallel':
        # Split communicator
        n_points = len(sweep_grid)
        color = world.Get_rank() * n_points // world.Get_size()
        sub_comm = world.Split(color, world.Get_rank())
        my_point = sweep_grid[color]

        # Run single estimation on sub_comm
        result = run_single_estimation(
            ..., calib_overrides={**base_calib, **my_point},
            comm=sub_comm)

        # Gather on world root
        all_results = world.gather(result if is_root(sub_comm) else None, root=0)

    elif mode == 'sequential':
        results = []
        for point in sweep_grid:
            result = run_single_estimation(
                ..., calib_overrides={**base_calib, **point},
                comm=world)
            if is_root(world):
                results.append(result)

    # Write summary
    if is_root(world):
        write_sweep_summary(results, sweep_dir)
```

### PBS scripts

No change needed for parallel mode — same `mpiexec` command.
For sequential mode, you might want longer walltime since it
runs K × single-point time.

The PBS script calls `estimate_sweep.py` instead of `estimate.py`
when the YAML has a `sweep:` block. Or `estimate.py` detects the
sweep block and dispatches internally (current approach).

### Discipline

Each sweep round = one YAML + one PBS:
- YAML specifies: free params, CE options, sweep grid, sweep mode
- PBS specifies: total ranks, memory, walltime
- Results land in one timestamped sweep directory

## Open questions

1. **Should estimate.py dispatch internally or should there be a
   separate estimate_sweep.py?** Internal dispatch is simpler
   (one entry point), but mixes concerns. Separate file is cleaner
   but adds another module to maintain.

2. **Multi-dimensional sweeps**: `grid: {sigma_w: [3 vals], phi_w: [3 vals]}`
   = 9 points. With 1040 ranks, that's 115 per point. Feasible but
   tight. Should we warn if ranks/point < n_elite?

3. **Checkpointing**: in sequential mode, if point 5 of 10 OOMs,
   you lose all progress. Should each point checkpoint independently?
   (Currently yes — state.pkl per point.)

4. **Resume**: if a sweep was interrupted, can we skip already-completed
   points? Check for existing summary.json in each point dir.

5. **Mixed sweep + non-sweep params**: what if the sweep variable
   is also a free parameter in the estimation? Currently the gamma_c
   sweep removes gamma_c from free params. Should the runner
   automatically exclude swept params from the free set?
