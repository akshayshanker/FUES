# `EGM_UE` — Upper Envelope Registry

```python
from dcsmm.uenvelope import EGM_UE
```

A unified entry point for all upper envelope algorithms. Wraps FUES, DC-EGM, RFC, and CONSAV behind a common interface for benchmarking and interchangeable use.

## Signature

```python
EGM_UE(
    x_dcsn_hat,         # endogenous grid
    qf_hat,             # value correspondence
    v_cntn_hat,         # continuation value (for CONSAV)
    kappa_hat,           # consumption policy
    X_cntn,              # next-period assets / continuation state
    X_dcsn,              # target evaluation grid
    uc_func_partial,     # marginal utility function u'(c)
    u_func,              # utility dict {"func": u, "args": {...}}
    ue_method="FUES",    # method name
    m_bar=1.0,
    lb=4,
    rfc_radius=0.75,
    rfc_n_iter=20,
    interpolate=False,
    include_intersections=True,
    ue_kwargs=None,
)
```

## Returns

```python
(refined, raw, interpolated)
```

- `refined` — dict with keys `x_dcsn_ref`, `v_dcsn_ref`, `kappa_ref`, `x_cntn_ref`, `lambda_ref`, `ue_time`
- `raw` — dict with the original (unrefined) inputs
- `interpolated` — dict with values interpolated onto `X_dcsn` (if `interpolate=True`)

## Available methods

| Method | `ue_method` | Source | Requirements |
|--------|-------------|--------|-------------|
| FUES | `"FUES"` | Dobrescu and Shanker (2025) | — |
| DC-EGM | `"DCEGM"` | Iskhakov et al. (2017) | `econ-ark` |
| RFC | `"RFC"` | Dobrescu and Shanker (2024) | `pykdtree` |
| CONSAV | `"CONSAV"` | Druedahl (2021) | `ConSav` |
| FUES v0 | `"FUES_V0DEV"` | Original paper version | — |
| Simple | `"SIMPLE"` | Monotonicity filter | — |

## Example: method comparison

```python
from dcsmm.uenvelope import EGM_UE
import numpy as np

# After EGM produces e_grid, vlu, c_hat, a_hat ...
eval_grid = np.linspace(0.01, 500, 3000)

for method in ["FUES", "DCEGM", "RFC", "CONSAV"]:
    refined, _, _ = EGM_UE(
        e_grid, vlu, v_continuation, c_hat, a_hat,
        eval_grid, du, {"func": u, "args": {}},
        ue_method=method, m_bar=1.2,
    )
    print(f"{method:8s}: {refined['ue_time']*1000:.2f} ms, "
          f"{len(refined['x_dcsn_ref'])} points retained")
```

## Registering custom engines

```python
from dcsmm.uenvelope.upperenvelope import register

@register("MY_METHOD")
def my_engine(x_dcsn_hat, qf_hat, kappa_hat, X_cntn, *,
              uc_func_partial, **kwargs):
    # ... your upper envelope logic ...
    return {
        "x_dcsn_ref": x_refined,
        "v_dcsn_ref": v_refined,
        "kappa_ref": c_refined,
        "x_cntn_ref": a_refined,
        "lambda_ref": uc_func_partial(c_refined),
    }
```

Then use `ue_method="MY_METHOD"` in `EGM_UE`.
