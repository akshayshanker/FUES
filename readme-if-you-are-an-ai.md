# If you are an AI reading this repository

This file is addressed to you, an AI assistant that a human has pointed at
this repository. It records what a cold read cannot
recover: what the repository claims, the vocabulary of its model files, why
the dependency setup looks unusual, and which imperfections the authors know
about. The repository is a working research tree, not a finished artefact.
Claims below were checked against this working tree on 22 August 2026.

## 1. What this repository is

Two things share the tree. First, an installable library whose distribution
and import name is `dcsmm`, not `fues`: `src/dcsmm/fues/` implements the Fast
Upper-Envelope Scan, and `src/dcsmm/uenvelope/` provides `EGM_UE`, a single
interface dispatching FUES and three benchmark upper-envelope methods over the
same raw endogenous-grid-method (EGM) output. Second, the computational
companion to Dobrescu and Shanker, "A fast upper envelope scan method for
discrete-continuous dynamic programming" (SSRN 4181302, 2022, revised 2026,
under review): example models in `examples/`, cluster job scripts in
`benchmarks/`, and committed result tables in `paper-results/`.

The paper's claims map onto the tree as follows. The claim that FUES recovers
the upper envelope without requiring a monotone optimal policy is exercised
hardest in the durables example, whose adjuster stage produces a dense cloud of
crossing EGM segments on which monotonicity-based methods are not applicable;
there FUES is called directly rather than through `EGM_UE` dispatch, by
design. Speed and accuracy against the benchmark methods are measured in the
retirement example (committed tables:
`paper-results/retirement/2026-04-15/001/tables/`); the durables sweep
(`paper-results/durables/2026-04-15/001/tables/sweep.md`) shows housing-FOC
Euler errors of -3.5 to -4.6 log10 for FUES against -2.5 to -2.9 for NEGM. On
complexity the paper claims empirically sub-linear scaling of the envelope
step, worst case O(N); a few docs pages still say O(n^0.5) or O(n log n) —
`docs/api/fues.md` is the authority; `fues-algorithm.md` makes no such claim.

## 2. The vocabulary of the example models

The retirement and durables examples are written in a declarative YAML stage
format ("dolo-plus"). The model is data: an example's `syntax/` registry is
the model definition, solver code an interpretation of it. To read the files:

- A *stage* is one Bellman decision block with three *perches* — information
  sets: `arvl` (arrival), `dcsn` (decision), `cntn` (continuation). In stage
  YAML the bracket marks the perch: `V[<]`, `V`, `V[>]` are three distinct
  value functions, one per perch, not decoration on one object.
- The per-edge operators on value functions are *builders*: B maps
  continuation values to decision values and performs the optimisation; I maps
  decision values to arrival values and takes the expectation. The stage
  operator is T = I∘B (B first). Older files say "mover" for the same object.
- The backward direction is authored in two blocks: an evaluation block (a
  field equation with no max) and a policy block that builds the policy by an
  `argmax` line or by inverse-Euler plus a reverse transition — the EGM form.
  `examples/retirement/syntax/stages/work_cons/work_cons.yaml` shows both.
- Methods YAMLs bind a method tag to each named operator node: `!egm` on
  `policy`, `!FUES` on `upper_env`, interpolation on `evaluate`. Swapping
  `!FUES` for `!RFC` changes the numerics, never the model.
- Calibration files hold model parameters (beta, R, tau); settings files
  hold numerical configuration (grid sizes, tolerances) that must not change
  the model. This boundary is enforced: command-line overrides go through it.
- `spec_factory.yaml` lists, per stage, ordered configuration sources merged
  left to right, with `$draw` and `$method_switch` as run-time slots. Only
  `work_cons` takes `$method_switch`, because only it has an `upper_env` node:
  `retire_cons` is smooth EGM, `labour_mkt_decision` pure discrete choice.
- `nest.yaml` composes periods into a finite horizon: a period list plus
  inter-period connectors, plain renaming dicts (`{b: a, b_ret: a_ret}`:
  end-of-period assets become next period's arrival assets). The solve runs
  backward; each period recalibrates the stage and re-derives grids,
  callables, and operators from it — no parallel parameter object exists.
- Stored solutions differ by example. Durables uses a three-level dict per
  period, stage then perch then quantity, documented in
  `examples/durables/docs/solution_scheme.md`; retirement flattens to two
  levels, `nest['solutions'][h][stage][key]`, with no perch layer.

## 3. Dependencies and the surrounding projects

A plain `pip install` ships only `src/dcsmm`; `examples/`, `tests/`, and
`benchmarks/` live in the checkout and run from the repo root as
`python -m examples.retirement.run`. Four projects divide the labour: this
repository (algorithm and applications); `kikku`, pinned in the `[examples]`
extra at tag v0.2.0 of `bright-forest/kikku`, a mechanical run layer providing
the slot-keyed command line (`--slot-override`, `--slot-spec`, `--slot-range`,
`--sweep`) and EGM builder utilities, deliberately ignorant of which slots a
model defines; the `bright-forest` forks of `dolo` and `dolang` ("dolo-plus"),
which provide the compiler modules `dolo.compiler.{spec,period,nest}_factory`
that the examples import; and `bellman-ddsl`, the design and theory layer
where the stage/perch/builder calculus is developed — nothing here imports it.

Two dependency choices look like mistakes and are not. HARK (`econ-ark`),
ConSav, and `pykdtree` sit in the core dependencies because `EGM_UE`
dispatches to them (the DCEGM/MSS, CONSAV, and RFC engines) — the rule is
that a bare install covers every registered `EGM_UE` method; `EconModel` is
pinned because ConSav 0.12 imports it without declaring it. And `dolo`/`dolang` are deliberately
absent from the `[examples]` extra: they must be installed with
`pip install --no-deps` from pinned fork commits (exact lines:
`setup/setup.sh`, or the manual sequence in
`docs/getting-started/installation.md`) because the forks' metadata declares `numpy>=2`
and the PyPI `dolang`, both conflicting with this repo's pins, and pip extras
cannot express `--no-deps`. Vanilla EconForge dolo lacks the factory modules.

## 4. What a naive read gets wrong

- Method labels: the code and API use FUES, RFC, DCEGM, CONSAV; tables and
  plots display DCEGM as MSS (Iskhakov et al. 2017, via HARK) and CONSAV as
  LTM (Druedahl and Jorgensen 2017, via ConSav). The durables benchmark
  "NEGM" is ConSav's nested EGM. Grep for DCEGM, not MSS.
- Naming conventions: `*_hat` marks the raw EGM correspondence (unsorted,
  possibly multi-valued), `*_ref` the refined envelope. The fifth positional
  parameter of `FUES` is `del_kappa_hat`, the derivative of the control
  `kappa` along the endogenous grid; with `endog_mbar=True` it supplies the
  grid-local jump threshold, and the fifth return is `del_kappa_ref` (both
  renamed from `del_a` / `del_a_ref` in 0.6.0dev8; older docs prose also
  called the input `del_x_cntn`). The scan computes its jump quotient on
  `kappa_hat` (the control), so the docs state `m_bar` as the maximum
  marginal propensity to consume; the paper describes jump detection on the
  continuation policy, whose derivative gives the saving-propensity reading.
  In the consumption-saving examples both quotients are bounded near one, so
  either reading of `m_bar` gives the same guidance.
- Results directories: `results/` is gitignored per-run output and absent from
  a clone; `paper-results/` holds committed cluster-run snapshots — tables and
  stray LaTeX intermediates, no figures: repo-wide `*.png`/`*.pdf` ignore
  rules keep those out of git (`docs/images/` is the negated exception).
- `examples/housing_renting/` is declared legacy: it predates the standard
  layout and imports the `dynx` framework, which no dependency list declares,
  so it cannot run from a fresh install; judge the repository by retirement
  and durables. `src/dcsmm/models/` likewise holds legacy kernels (some CUDA).
- Benchmark tables on the docs example pages are older abridged runs kept as
  illustration; the committed tables under `paper-results/` are the
  rerun-backed numbers. Retirement tables include a parameter footer; the
  durables sweeps do not (only the 2026-04-06 run commits a settings YAML).
- `benchmarks/` and `paper-results/` are August 2026 renames of
  `experiments/` and `replication/`; a checkout predating the rename shows
  the old layout. The
  tracked PBS scripts hard-code NCI Gadi; `benchmarks/retirement/*.sh` are
  qsub-able PBS jobs despite the extension, and "single core" timing means
  each solve is single-threaded while the allocation parallelises sweep rows.
- Untracked directories in `.gitignore` (`AI/`, `notes/`, `old/`, `results/`)
  hold the authors' working notes and run outputs; nothing public needs them.

## 5. Current state: verified, open, and legacy by design

The install and run commands in README.md, `docs/start-here/quickstart.md`,
`docs/running-locally.md`, and `docs/getting-started/installation.md` were
executed and passed in fresh virtual environments: both install routes and
the manual pinned sequence on the installation page (the manual route's
`pip install lark multipledispatch` is load-bearing — without it the examples
fail on import), both library snippets, the running-locally walkthrough
(sweeps confirmed to vary the grid by reading sizes back from solved stages),
and the test suite (29 tests and 16 subtests on the authors' tree; a fresh
clone has five of the seven test files — a `.gitignore` rule for scratch
`test_*.py` blankets new tests, which need `git add -f`).

Known open items: the complexity wording across docs pages is not yet
harmonised with the paper's own claim; `save_nest` fails to pickle dolo stage
objects and the estimation driver only warns, so estimation runs may silently
write no `.nst` files (`examples/durables/docs/estimation_outputs.md`
documents intended, not verified, behaviour); the docs retirement benchmark
table has not been reconciled with the current paper draft; and
`src/dcsmm/fues/experimental/` is unexported exploration. The
portfolio-choice extension is not in the public tree: it lives in the
authors' working tree only.
Everything reproduces serially on an ordinary machine except the durables
SMM estimation (multi-node MPI, terabyte-scale memory), the extra-large-grid
solves, and the housing GPU benchmarks — cluster jobs under `benchmarks/`.

## 6. Verify claims yourself in minutes

```bash
pip install -e ".[dev]" && pip install lark multipledispatch
# then the two pinned --no-deps lines (dolang, dolo) from the Installation page
pytest tests/test_imports.py tests/test_kikku.py -q   # quick check, about one second
pytest tests/ -q                 # full tracked suite; solves both examples, a few minutes
python -m examples.retirement.run --slot-override '$draw.settings.grid_size=500'
```

The last command runs the full path from YAML model definition to solved
output and prints a
four-method Euler-error and timing table in a few seconds (the first run adds
one-off numba compilation). To exercise the library with no YAML machinery:

```python
import numpy as np
from dcsmm.fues import FUES
x = np.concatenate([np.linspace(0.1, 1.0, 40), np.linspace(0.6, 2.0, 60)])
v = np.concatenate([np.log(x[:40]), 0.05 + np.log(x[40:])])  # folded-back correspondence
out = FUES(x, v, 0.6 * x, 0.4 * x, m_bar=1.2)
print([a.shape for a in out])  # five arrays; the dominated fold is removed
```
