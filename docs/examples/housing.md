# Durables Model

!!! note "Coming soon"
    This page will document the durables model with adjustment frictions (Paper Section 2.2).

Two-asset model with liquid financial assets and illiquid housing. Adjusting housing requires paying a transaction cost \(\tau H'\). The discrete choice of whether to adjust creates a non-monotone endogenous grid where MSS and LTM cannot be applied. FUES and RFC are the applicable methods.

See [Dobrescu, Shanker, and Yogo (2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4850746) for the model specification.

## Running

```bash
python examples/durables/durables.py
```

## PBS replication

```bash
qsub experiments/durables/durables_timings.sh
```

