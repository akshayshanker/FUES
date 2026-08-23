# Example models — standard layout

Every example is a self-contained model application. The model is written
as a sequence of *stages* — the within-period decision problems that
together make up one period — defined in YAML. Two supporting packages do
the mechanical work: `dcsmm` (this repo's installable library, containing
the FUES upper-envelope algorithm) and `kikku` (which reads the YAML model
definition, assembles the backward-induction problem, and provides the
command line). Each example follows one standard layout, so a reader who
has understood one example can navigate all of them.

```
examples/<model>/
├── README.md        # what the model is, its stages, how to run it
├── run.py           # command-line script; run from the repo root:
│                    #   python -m examples.<model>.run
├── solve.py         # backward induction: builds each period's stages
│                    #   from the YAML, links them into a period, solves
├── model.py         # grid construction and model-level helpers; every
│                    #   parameter comes from the calibrated stage objects
├── benchmark.py     # solution-time and accuracy measurement  (optional)
├── estimate.py      # estimation driver (SMM etc.)            (optional)
├── __init__.py
├── syntax/          # the model definition, and the single place
│                    #   parameters live: calibration, numerical settings,
│                    #   and one folder per stage under stages/<stage>/
│                    #   (stage YAML + methods YAML)
├── solvers/         # the stage operators of the backward pass,
│                    #   one maker per stage
├── postprocess/     # everything after the solve: tables, figures,
│                    #   diagnostics
├── notebooks/       # interactive walkthroughs
└── docs/            # example-local developer notes; user-facing pages
                     #   live in the repo-root docs/examples/ (the website)
```

## Conventions

- **`syntax/` is the single source of truth.** Parameters live in the
  calibrated stage objects built from these files; grids, equation
  functions, and operators are all derived from them — there is no
  separate parameter object. An example with one parameterisation keeps
  the files directly in `syntax/` (retirement); an example with several
  keeps one registry folder per parameterisation, each with its own
  `callables.py` (durables: `syntax/separable/`, `syntax/cobb_douglas/`).
  Variant files inside a registry (alternative calibrations, estimation
  targets) are allowed.
- **`solvers/` holds the backward-pass operators.** Each maker takes
  `(callables, grids, stage)` and receives the stage object it belongs
  to. An example's own forward simulator, when it has one, lives here
  as well (durables `solvers/simulate.py`); retirement simulates through
  the library instead.
- **`postprocess/` comes after the solve.** The backward pass never
  imports from it. One deliberate exception: forward simulation may call
  the Euler-error diagnostics that live there, since both run after the
  model is solved.
- **Root files run the model, nothing else.** Run outputs go to
  `--output-dir` (default: the gitignored `results/`); committed paper
  outputs go to repo-root `paper-results/`; cluster job scripts live in
  repo-root `benchmarks/<model>/` — never inside the example.
- **Tests live in the repo-level `tests/`**, named `test_<model>*.py`.
  An example directory contains no test files.

## Status

| Example | Conforms | Notes |
|---------|----------|-------|
| `retirement/` | yes | reference implementation; single parameterisation |
| `durables/` | yes | several registries under `syntax/`; adds `estimate.py` |
| `housing_renting/` | no (legacy) | predates the standard; do not edit |
| `old/` | local only | gitignored scratch; absent from a fresh clone |

`_template/` is a copyable skeleton: the directory structure with a
one-line README in each folder and no code. To start a new example, copy
it, rename it, and write the `syntax/` registry first — the YAML defines
the model; the Python only runs it.
