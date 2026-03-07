# Quick Start

## Drop-in FUES

After your EGM step produces arrays of endogenous grid points, values, and policies, pass them to FUES:

```python
from dcsmm.fues import FUES

e_clean, v_clean, c_clean, a_clean, _ = FUES(
    e_grid,     # endogenous grid (N,) — unsorted is fine
    vlu,        # value at each point (N,)
    c_hat,      # consumption policy (N,)
    a_hat,      # next-period assets (N,)
    del_a,      # policy gradient da'/de (N,)
    m_bar=1.2,  # jump detection threshold (≈ max MPC)
    LB=4,       # look-back buffer for forward/backward scans
)
```

The returned arrays contain only the optimal (upper-envelope) points. Interpolate `(e_clean, c_clean)` onto your target grid.

**Setting `m_bar`**: use the maximum marginal propensity to consume in your model. For log utility, `m_bar ≈ 1.0`–`1.2` works well. If unsure, set `endog_mbar=True` to let FUES compute it from `del_a`.

## Upper-envelope registry

To compare FUES against MSS, RFC, and LTM on the same problem, use the unified `EGM_UE` interface:

```python
from dcsmm.uenvelope import EGM_UE

refined, raw, interpolated = EGM_UE(
    x_dcsn_hat=e_grid,
    qf_hat=vlu,
    v_cntn_hat=v_continuation,
    kappa_hat=c_hat,
    X_cntn=a_hat,
    X_dcsn=target_grid,
    uc_func_partial=marginal_utility,   # u'(c)
    u_func={"func": utility_fn, "args": {}},
    ue_method="FUES",                   # or "DCEGM", "RFC", "CONSAV"
    m_bar=1.2,
)

# Results
refined["x_dcsn_ref"]   # refined endogenous grid
refined["v_dcsn_ref"]   # refined values
refined["kappa_ref"]    # refined consumption
refined["ue_time"]      # wall-clock time (seconds)
```

Switch `ue_method` to `"DCEGM"`, `"RFC"`, or `"CONSAV"` to run the same problem with a different algorithm. All methods return the same dict structure.

See [API Reference](../api/fues.md) for full parameter documentation.

## Try the notebook

For a complete worked example solving the [Iskhakov et al. (2017)](https://doi.org/10.3982/QE643) retirement model:

```bash
jupyter lab examples/retirement/notebooks/retirement_fues.ipynb
```

Or view the [rendered notebook](../notebooks/retirement_fues.ipynb) in the docs.
