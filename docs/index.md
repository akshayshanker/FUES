---
hide:
  - toc
---

# FUES: Fast Upper Envelope Scan

!!! warning "Pre-release (v0.6.0dev7)"
    Under active development. The API and documentation may change.

Paper: Dobrescu, L.I. and Shanker, A. (2022, revised 2026). "A fast upper envelope scan method for discrete–continuous dynamic programming." [SSRN 4181302](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4181302).

FUES recovers the upper envelope of the EGM ([Carroll 2006](https://doi.org/10.1016/j.econlet.2005.09.013)) value correspondence in discrete–continuous problems. It scans the endogenous grid in a single sub-linear pass, and identifies sub-optimal points as the conjunction of a discontinuous jump in the continuation policy and a concave right turn in the value correspondence. It imposes no monotonicity on the optimal policy and requires no numerical optimisation. See [How FUES works](algorithm/fues-algorithm.md) for the derivation.

## Minimal use

After your EGM step produces arrays for the raw endogenous-grid correspondence, pass them to `FUES`:

```python
from dcsmm.fues import FUES

x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, _ = FUES(
    x_dcsn_hat,   # raw endogenous decision grid (N,) — unsorted is fine
    v_hat,        # raw value correspondence (N,)
    kappa_hat,    # primary control, e.g. consumption (N,)
    x_cntn_hat,   # raw continuation / post-decision state (N,)
    del_x_cntn,   # auxiliary jump-detection series, e.g. d x_cntn / d x_dcsn
    m_bar=1.2,    # jump threshold (approx max marginal propensity to save)
    LB=4,         # look-back buffer for forward/backward scans
)
```

The returned arrays contain only the upper-envelope points. Convention: `*_hat` for raw correspondence, `*_ref` for refined upper-envelope objects. For a cell-by-cell walkthrough of exactly these calls on a small worked problem, see the [EGM / FUES walkthrough notebook](notebooks/egm_fues_transparent.ipynb).

**Setting `m_bar`.** Use the maximum marginal propensity to save in your model. For log utility with $\beta R < 1$, values in the range $1.0$–$1.2$ work well. Setting `endog_mbar=True` lets FUES compute a grid-local threshold from `del_x_cntn`.

To compare FUES against MSS, RFC, and LTM on the same problem, use the unified `EGM_UE` interface:

```python
from dcsmm.uenvelope import EGM_UE

refined, raw, interpolated = EGM_UE(
    x_dcsn_hat=x_dcsn_hat,
    v_hat=v_hat,
    v_cntn_hat=v_cntn,
    kappa_hat=kappa_hat,
    x_cntn_hat=x_cntn_hat,
    X_dcsn=X_dcsn,
    uc_func_partial=uc_func,    # u'(c); used to compute lambda_ref
    u_func=u_func,
    method_switch="FUES",         # or "DCEGM", "RFC", "CONSAV"
    m_bar=1.2,
)
```

All methods return the same dict schema. `DCEGM` (MSS in the paper) and `CONSAV` (LTM in the paper) require a strictly monotone optimal policy; `FUES` and `RFC` do not. See the [durables application](examples/continuous_housing_model.md) for the main non-monotone benchmark.

See the [Core API](api/fues.md) for full parameter documentation.
