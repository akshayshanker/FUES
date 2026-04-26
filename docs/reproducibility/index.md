# Reproducibility

FUES is both a library and a research-code repository. This section collects the
workflows used to generate benchmark outputs and paper artifacts.

## Where things live

| Folder | Contents |
|---|---|
| `examples/<model>/` | Model code and `run.py` entry point |
| `experiments/<model>/` | PBS scripts and sweep drivers used for paper tables |
| `results/<model>/` | Local run outputs (dated, auto-incremented) |
| `replication/<model>/` | Committed tables and figures for a paper-ready run |

## Reading order

1. [Quickstart](../start-here/quickstart.md) — the shortest path to a first solve.
2. [Running locally](../running-locally.md) — single solves, comparisons, small sweeps.
3. [Applications](../examples/index.md) — model-level context and benchmark commands.
4. [Running on PBS / Gadi](../running-on-gadi.md) — batch, MPI, paper-scale sweeps.
