# Retirement choice model

A lifecycle consumption–saving model with a discrete retirement decision
(Iskhakov, Jørgensen, Rust and Schjerning 2017). One period contains
three stages: the labour-market decision (work or retire), worker
consumption, and retiree consumption. The worker's discrete choice makes
the value correspondence non-concave, so the consumption stage applies an
upper-envelope method to the endogenous-grid output — FUES by default,
with RFC, DCEGM, and the ConSav envelope as benchmarks.

Run a single solve from the repo root (defaults compare all four
upper-envelope methods):

    python -m examples.retirement.run --slot-override '$draw.settings.grid_size=3000'

The timing sweep behind the paper tables is documented in the repo-root
README; cluster scripts are in `benchmarks/retirement/`. The layout
follows `examples/README.md`; the user-facing documentation page is
`docs/examples/retirement_choice_model.md`.
