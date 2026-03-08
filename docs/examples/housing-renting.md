# Housing-Renting Model

!!! note "Coming soon"
    This page will document the discrete housing choice model (Paper Section 2.3).

Extended version of the discrete housing choice model in [Fella (2014)](https://doi.org/10.1016/j.red.2013.07.001) with renting and a non-smooth capital income tax schedule. The tax kinks create behavioral inaction regions that break strict monotonicity of the savings policy.

## Running

```bash
python examples/housing_renting/run.py
```

## PBS replication

```bash
qsub experiments/housing_renting/run_housing_single_core.sh
```

---

*(c) Akshay Shanker*
