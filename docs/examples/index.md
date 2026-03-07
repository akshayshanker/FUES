# Examples

Three benchmark applications from the paper. Each page covers the model, how to run it, PBS replication instructions, and links to interactive notebooks.

| Application | Paper section | Key feature | Notebook |
|-------------|--------------|-------------|----------|
| [Retirement choice](retirement.md) | Section 2.1 | Monotone policy; speed benchmark | [notebook](../notebooks/retirement_fues.ipynb) |
| [Housing investment](housing.md) | Section 2.2 | Non-monotone policy; MSS/LTM fail | — |
| [Housing-renting](housing-renting.md) | Section 2.3 | Inaction regions; tax kinks | — |

All examples use the canonical pipeline (`solve_nest`) and can be run locally or on PBS. Paper results are in [`replication/`](https://github.com/akshayshanker/FUES/tree/release-prep/replication).
