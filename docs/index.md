# FUES: Fast Upper-Envelope Scan

!!! warning "Pre-release (v0.6.0dev1)"
    Under active development. The API and docs may change.

**A general method for computing the upper envelope of EGM value correspondences in discrete-continuous dynamic programming.**

> Dobrescu, L.I. and Shanker, A. (2022). "A fast upper envelope scan method for discrete-continuous dynamic programming." [SSRN.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4181302)

FUES scans the endogenous grid in a single $O(n^{1/2})$ pass, detecting sub-optimal points as the conjunction of a policy jump and a concave right turn in the value correspondence.

Unlike existing upper envelope methods, FUES does not require monotonicity of the policy function or numerical optimisation. It is also orders of magnitude faster.

See [How FUES Works](algorithm/fues-algorithm.md) for the full algorithm.

## At a glance

After your EGM step produces arrays for the raw endogenous-grid correspondence, pass them to `FUES`:

```python
from dcsmm.fues import FUES

x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, _ = FUES(
    x_dcsn_hat,   # raw endogenous decision grid (N,) — unsorted is fine
    v_hat,        # raw value correspondence (N,)
    kappa_hat,    # raw control correspondence, e.g. consumption (N,)
    x_cntn_hat,   # raw continuation / post-decision state (N,)
    del_x_cntn,   # auxiliary jump-detection series, e.g. d x_cntn / d x_dcsn
    m_bar=1.2,    # jump detection threshold (approx max MPC)
    LB=4,         # look-back buffer for forward/backward scans
)
```

The returned arrays contain only the upper-envelope points. A useful naming convention:

- `*_hat` = raw / unrefined EGM correspondence
- `*_ref` = refined upper-envelope objects
- `X_dcsn` = target decision grid for interpolation

**Setting `m_bar`**: use the maximum marginal propensity to consume in your model. For log utility, `m_bar` in the range 1.0--1.2 works well. If unsure, set `endog_mbar=True` to let FUES compute it from `del_x_cntn`.

## Upper-envelope registry

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
    uc_func_partial=uc_func,               # u'(c); only used by "CONSAV"
    u_func=u_func,
    ue_method="FUES",                       # or "DCEGM", "RFC", "CONSAV"
    m_bar=1.2,
)

# Results
refined["x_dcsn_ref"]   # refined endogenous grid
refined["v_dcsn_ref"]   # refined values
refined["kappa_ref"]    # refined control / consumption
refined["x_cntn_ref"]   # refined continuation state
refined["ue_time"]      # wall-clock time (seconds)
```

Switch `ue_method` to `"DCEGM"`, `"RFC"`, or `"CONSAV"` to run the same problem with a different algorithm. All methods return the same dict structure.

Note: DCEGM (MSS in the paper) and CONSAV (LTM in the paper) require a strictly monotone optimal policy. See the [durables example](examples/housing.md) and the [housing-renting example](examples/housing-renting.md).

See [API Reference](api/fues.md) for full parameter documentation.

## Try the notebook

For a complete worked example solving the [Iskhakov et al. (2017)](https://doi.org/10.3982/QE643) retirement model:

```bash
jupyter lab examples/retirement/notebooks/retirement_fues.ipynb
```

Or view the [rendered notebook](notebooks/retirement_fues.ipynb) in the docs.

## Quick links

| | |
|---|---|
| **[Installation](getting-started/installation.md)** | pip install, editable, Gadi |
| **[Examples](examples/index.md)** | Retirement, housing, durables — with notebooks |
| **[API Reference](api/fues.md)** | Full signatures and docstrings |
| **[Algorithm](algorithm/fues-algorithm.md)** | How FUES works, forward/backward scans |
