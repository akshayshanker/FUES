# Durables model

A lifecycle model of consumption and a durable (housing) stock with
adjustment frictions (Dobrescu and Shanker). One period contains three
stages: the tenure decision (keep or adjust the durable), keeper
consumption (one-dimensional endogenous grid per income–housing slice),
and adjuster consumption (a two-dimensional problem solved by inverse
Euler equations). The discrete keep/adjust choice makes the problem
non-concave; the adjuster stage refines the raw endogenous-grid output
with FUES (NEGM is available as a benchmark).

Two parameterisations live under `syntax/`: `separable` (baseline) and
`cobb_douglas`, each a complete registry with its own `callables.py`.
`estimate.py` drives the simulated-moments estimation on the cluster.

Run a single solve from the repo root:

    python -m examples.durables.run

Cluster scripts are in `benchmarks/durables/`; developer notes
(solution layout, estimation outputs) are in `docs/`. The layout follows
`examples/README.md`.
