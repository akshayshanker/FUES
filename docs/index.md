# FUES: Fast Upper-Envelope Scan

!!! warning "Pre-release (v0.6.0dev1)"
    Under active development. The API and docs may change.

**A general method for computing the upper envelope of EGM value correspondences in discrete-continuous dynamic programming.**

> Dobrescu, L.I. and Shanker, A. (2022). "A fast upper envelope scan method for discrete-continuous dynamic programming." [SSRN.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4181302)

FUES scans the endogenous grid in a single $O(n^{1/2})$ pass, detecting sub-optimal points as the conjunction of a policy jump and a concave right turn in the value correspondence. 

Unlike existing upper envelope methods, FUES does not require monotonicity of the policy function or numerical optimisation. It is also orders of magnitude faster.

See [How FUES Works](algorithm/fues-algorithm.md) for the full algorithm.

## At a glance

Suppose we invert an Euler equation and recover the raw endogenous decision grid `x_dcsn_hat`, the value correspondence `v_hat` (the current value evaluated at those endogenous grid points), the policy correspondence `kappa_hat`, and the continuation or post-decision object `x_cntn_hat`.

The `FUES` operator can be applied like this:

```python
from dcsmm.fues import FUES

x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, _ = FUES(
    x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat, m_bar=1.01,
)
```

the refined (optimal) upper-envelope objects are the endogenous decision state grid points `x_dcsn_ref`, the value function `v_ref`, the optimal policy `kappa_ref`, and the optimal post-state `x_cntn_ref`.

The `dcsmm` package also ships an upper envelope interface that dispatches to the main benchmark algorithms in `python`. 

```python
from dcsmm.uenvelope import EGM_UE

refined, _, _ = EGM_UE(
    x_dcsn_hat=x_dcsn_hat,
    v_hat=v_hat,
    v_cntn_hat=v_cntn,
    kappa_hat=kappa_hat,
    x_cntn_hat=x_cntn_hat,
    X_dcsn=X_dcsn,
    uc_func_partial=uc_func,   # only used by "CONSAV"
    u_func=u_func,
    ue_method="FUES",          # or "DCEGM", "RFC", "CONSAV"
)
```

`refined` is a dict of refined objects. 

Note: DCEGM (MSS in the paper) and CONSAV (LTM in the paper) will only work if the optimal policy function is strictly monotone. See the [durables example](examples/housing.md) and the [housing-renting example](examples/housing-renting.md).

## Quick links

| | |
|---|---|
| **[Installation](getting-started/installation.md)** | pip install, editable, Gadi |
| **[Quick Start](getting-started/quickstart.md)** | Drop-in API usage for FUES and EGM_UE |
| **[Examples](examples/index.md)** | Retirement, housing, durables — with notebooks |
| **[API Reference](api/fues.md)** | Full signatures and docstrings |
| **[Algorithm](algorithm/fues-algorithm.md)** | How FUES works, forward/backward scans |

