# Changelog

The full version history is in
[`CHANGELOG.md`](https://github.com/akshayshanker/FUES/blob/main/CHANGELOG.md)
in the repository root. The current pre-release is summarised below.

## 0.6.0dev4 · 2026-04-15

Unified install and examples polish.

**Install.** Three tiers: `pip install -e .` for FUES plus the
upper-envelope benchmarks (DCEGM/MSS, LTM, RFC); `pip install -e
".[examples]"` for everything needed to run the shipped models and
notebooks; `pip install -e ".[dev]"` adds `pytest` and `autopep8`. HARK
and ConSav are now in the core install because they are the benchmark
targets. Four previous setup scripts are consolidated into one:

```bash
source setup/setup.sh            # install if needed, then activate
source setup/setup.sh --update   # git pull, refresh install, activate
```

**FUES algorithm.** `FUES_jit` provides a Numba-compatible entry point
for use inside `@njit` loops; the existing `FUES` entry is unchanged.
Two new helpers appear in `dcsmm.fues.helpers`: `interp_as_3` (fused
three-output interpolation) and `interp2d_nonuniform` (2D bilinear on
uneven grids). The `uenvelope` registry recognises `"MSS"` as an alias
for `"DCEGM"` — same method, two names in the literature. HARK,
ConSav, and pykdtree are now optional at import time: the methods
that rely on them raise a clear error if the backend is missing,
rather than breaking the whole package.

**Durables.** Terminal utility is now included in the NPV used for
welfare comparisons (previously silently dropped). 2D interpolation
uses a library routine that handles non-uniform grids; the `grid_phi`
setting controls grid packing. `ce_burn_in` skips initial periods in
the certainty-equivalent welfare calculation, to remove
starting-distribution artefacts. `init_dispersion` optionally draws
initial wealth and housing log-normally rather than from point
masses. FUES guard parameters (tolerances, extrapolation, clamp
factors) are now exposed in `settings.yaml`.

**Retirement.** On clusters the post-sweep solve-and-plot block runs
on rank 0 only, avoiding the race condition where every rank redid
the work and raced on plot writes. Plot helpers are loaded lazily, as
in durables.

**Docs.** Install sections rewritten around the three-tier layout and
the single setup script. Paper-faithful documentation added for the
retirement and continuous-housing models. Notebook cohort tables
render correctly on Material.

---

Older entries live in
[`CHANGELOG.md`](https://github.com/akshayshanker/FUES/blob/main/CHANGELOG.md).
