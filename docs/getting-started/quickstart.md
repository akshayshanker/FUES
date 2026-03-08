# Quick Start

## Drop-in FUES

After your EGM step produces arrays for the raw endogenous-grid correspondence, pass them to `FUES`:

```python
from dcsmm.fues import FUES

x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, _ = FUES(
    x_dcsn_hat,   # raw endogenous decision grid (N,) — unsorted is fine
    v_hat,        # raw value correspondence (N,)
    kappa_hat,    # raw control correspondence, e.g. consumption (N,)
    x_cntn_hat,   # raw continuation / post-decision state (N,)
    del_x_cntn,   # auxiliary jump-detection series, e.g. d x_cntn / d x_dcsn
    m_bar=1.2,  # jump detection threshold (≈ max MPC)
    LB=4,       # look-back buffer for forward/backward scans (optional, default =)
)
```

The returned arrays contain only the upper-envelope points. A useful reading convention is:

- `*_hat` = raw / unrefined EGM correspondence
- `*_ref` = refined upper-envelope objects
- `X_dcsn` = target decision grid for interpolation

**Setting `m_bar`**: use the maximum marginal propensity to consume in your model. For log utility, `m_bar ≈ 1.0`–`1.2` works well. If unsure, set `endog_mbar=True` to let FUES compute it from `del_a`.

**details of other options here**


## Upper-envelope registry

To compare FUES against MSS, RFC, and LTM on the same problem, use the unified `EGM_UE` interface:

```python
from dcsmm.uenvelope import EGM_UE

refined, raw, interpolated = EGM_UE(
    x_dcsn_hat=x_dcsn_hat,
    qf_hat=v_hat,                    # legacy API name; read as value correspondence
    v_cntn_hat=v_continuation,
    kappa_hat=kappa_hat,
    X_cntn=x_cntn_hat,               # legacy API name; read as continuation state
    X_dcsn=X_dcsn,
    uc_func_partial=marginal_utility,   # u'(c)
    u_func={"func": utility_fn, "args": {}},
    ue_method="FUES",                   # or "DCEGM", "RFC", "CONSAV"
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

## Notation note

Across the docs we now recommend:

- `v_hat` rather than `q_hat` for the raw value object in paper-facing notation
- `x_cntn_hat` in code and `\hat{x}_e` in math for the continuation / post-decision object
- `\hat{x}_e \equiv \hat{x}'` when transitioning from the current paper notation

That lets the documentation stay close to the current FUES paper while remaining readable from a Bellman-DDSL perspective.

See [API Reference](../api/fues.md) for full parameter documentation.

## Try the notebook

For a complete worked example solving the [Iskhakov et al. (2017)](https://doi.org/10.3982/QE643) retirement model:

```bash
jupyter lab examples/retirement/notebooks/retirement_fues.ipynb
```

Or view the [rendered notebook](../notebooks/retirement_fues.ipynb) in the docs.

---

*(c) Akshay Shanker*
