# FUES: Fast Upper-Envelope Scan

!!! warning "Under development — do not cite"
    This documentation and codebase are under active development. Please do not cite until a stable release is published.

**A general method for computing the upper envelope of EGM value correspondences in discrete-continuous dynamic programming.**

> Dobrescu, L.I. and Shanker, A. (2026). "A fast upper envelope scan method for discrete-continuous dynamic programming." [SSRN.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4181302)

FUES scans the sorted endogenous grid in a single pass, detecting sub-optimal points by the conjunction of a policy jump and a concave right turn in the value correspondence. Unlike existing upper envelope methods, it does not require monotonicity of the policy function or numerical optimisation. It is also orders of magnitude faster.

See [How FUES Works](algorithm/fues-algorithm.md) for the full algorithm.

## At a glance

Suppose we invert an Euler equation to recover the raw endogenous decision grid `x_dcsn_hat`, the value correspondence `v_hat` (the current value evaluated at those endogenous grid points), the policy correspondence `kappa_hat`, and the continuation or post-decision object `x_cntn_hat`.

```python
from dcsmm.fues import FUES

x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, _ = FUES(
    x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat, m_bar=1.01,
)
```

The refined upper-envelope objects are `x_dcsn_ref`, `v_ref`, `kappa_ref`, and `x_cntn_ref`.

Or use the unified registry to compare methods:

```python
from dcsmm.uenvelope import EGM_UE

refined, _, _ = EGM_UE(
    x_dcsn_hat=x_dcsn_hat,
    qf_hat=v_hat,              # legacy API name; read as raw value correspondence
    v_cntn_hat=v_cntn,
    kappa_hat=kappa_hat,
    X_cntn=x_cntn_hat,         # legacy API name; read as continuation state
    X_dcsn=X_dcsn,
    uc_func_partial=uc_func,   # only used by "CONSAV"
    u_func=u_func,
    ue_method="FUES",          # or "DCEGM", "RFC", "CONSAV"
)
```

`refined` is a dict of refined objects. 
## Quick links

| | |
|---|---|
| **[Installation](getting-started/installation.md)** | pip install, editable, Gadi |
| **[Quick Start](getting-started/quickstart.md)** | Drop-in API usage for FUES and EGM_UE |
| **[Examples](examples/index.md)** | Retirement, housing, durables — with notebooks |
| **[API Reference](api/fues.md)** | Full signatures and docstrings |
| **[Algorithm](algorithm/fues-algorithm.md)** | How FUES works, forward/backward scans |

---

*(c) Akshay Shanker*
