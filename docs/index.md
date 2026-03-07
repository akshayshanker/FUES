# FUES: Fast Upper-Envelope Scan

!!! warning "Under development — do not cite"
    This documentation and codebase are under active development. Please do not cite until a stable release is published.

**A general method for computing the upper envelope of EGM value correspondences in discrete-continuous dynamic programming.**

> Dobrescu, L.I. and Shanker, A. (2026). "A fast upper envelope scan method for discrete-continuous dynamic programming." [SSRN.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4181302)

FUES scans the sorted endogenous grid in a single pass, detecting sub-optimal points by the conjunction of a policy jump and a concave right turn in the value correspondence. It does not require monotonicity of the policy function or numerical optimisation, and is orders of magnitude faster than existing methods.

See [How FUES Works](algorithm/fues-algorithm.md) for the full algorithm.

## At a glance

```python
from dcsmm.fues import FUES

e_clean, v_clean, c_clean, a_clean, _ = FUES(
    e_grid, vlu, c_hat, a_hat, del_a, m_bar=1.2,
)
```

Or use the unified registry to compare methods:

```python
from dcsmm.uenvelope import EGM_UE

refined, _, _ = EGM_UE(
    e_grid, vlu, v_cntn, c_hat, a_hat, target_grid,
    uc_func, u_func, ue_method="FUES",  # or "DCEGM", "RFC", "CONSAV"
)
```

## Quick links

| | |
|---|---|
| **[Installation](getting-started/installation.md)** | pip install, editable, Gadi |
| **[Quick Start](getting-started/quickstart.md)** | Drop-in API usage for FUES and EGM_UE |
| **[Examples](examples/index.md)** | Retirement, housing, durables — with notebooks |
| **[API Reference](api/fues.md)** | Full signatures and docstrings |
| **[Algorithm](algorithm/fues-algorithm.md)** | How FUES works, forward/backward scans |
