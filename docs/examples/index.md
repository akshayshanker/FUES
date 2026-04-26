# Applications

The paper demonstrates FUES on three applications, each chosen to expose a
distinct failure mode of existing upper-envelope methods. Each page states the
economic model, records the notation used in the codebase, and points to the
associated notebook and command-line workflow.

| Application | Paper § | What makes it hard | Competing methods |
|---|---|---|---|
| [Retirement choice](retirement_choice_model.md) | 2.1 | Monotone policies — a clean speed benchmark | MSS · LTM · RFC · FUES |
| [Continuous housing](continuous_housing_model.md) | 2.2 | **Non-monotone housing policy** — MSS and LTM do not apply | NEGM · FUES |
| [Housing-renting](housing-renting.md) | 2.3 | Inaction regions from kinked tax schedule — draft | NEGM · FUES |

## Headline numbers

On the retirement benchmark, FUES delivers the upper envelope in $O(n^{1/2})$
time. With $10{,}000$ grid points it completes in roughly $0.8$ ms against
roughly $800$ ms for LTM and $8$ ms for MSS (Intel Xeon, single-core). See
the [retirement notebook](../notebooks/retirement_fues.ipynb) for the full
scaling sweep.

On the continuous-housing benchmark, the adjuster policy is non-monotone and
neither MSS nor LTM can be applied. Against NEGM — the standard fallback in
this class — FUES yields roughly two orders of magnitude lower Euler error on
the adjuster FOC at comparable runtimes. See the
[Cobb–Douglas notebook](../notebooks/durables_fues.ipynb) for the comparison.

## How to run the applications

Both applications share a single command-line interface. Typical entry points:

```bash
python -m examples.retirement.run --settings-override grid_size=3000
python -m examples.durables.run
```

For the full set of command-line options see
[running locally](../running-locally.md). For the batch and cluster workflow —
sweeps, MPI estimation, paper tables — see
[running on PBS / Gadi](../running-on-gadi.md).

Committed paper outputs (tables and figures in the form reproduced in the
paper) live under
[`replication/`](https://github.com/akshayshanker/FUES/tree/main/replication)
in the repository.
