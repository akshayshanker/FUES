# Durables model

A lifecycle model of consumption and a durable (housing) stock with
adjustment frictions (Dobrescu and Shanker). One period contains three
stages: the tenure decision (keep or adjust the durable), keeper
consumption (one-dimensional endogenous grid per income–housing slice),
and adjuster consumption (a two-dimensional problem solved by inverse
Euler equations). The discrete keep/adjust choice makes the problem
non-concave; the adjuster stage refines the raw endogenous-grid output
with FUES (NEGM is available as a benchmark).

Two specifications are under `syntax/`: `separable` (baseline) and
`cobb_douglas`. Each model spec. has its own own `callables.py`.

`estimate.py` is the entry point for the simulated-moments estimation on a pbs cluster.
The Cobb-Douglas estimation specs under `syntax/cobb_douglas/estimation/` use
the same data, targets and sample sizes as the separable specs; the free
parameters put `rho` in place of `gamma_c` and `gamma_h`, and they estimate
`theta`. Each registry also has `baseline_large_egm_types.yaml`, a four-type
discount-factor spec. After a run, `notebooks/lifecycle_at_estimates.ipynb`
and `python -m examples.durables.postprocess.at_estimates` re-solve at the
estimates; `python -m examples.durables.postprocess.estimation_tables`
writes the parameter and moment-fit tables.

Run a single solve from the repo root:

    python -m examples.durables.run

Cluster scripts are in `benchmarks/durables/`; developer notes
(solution layout, estimation outputs) are in `docs/`. The layout follows
`examples/README.md`.
