# Quick Start

## Using FUES directly

FUES takes the raw output of an EGM step — endogenous grid, values, and policies — and returns the refined arrays with sub-optimal points removed.

```python
import numpy as np
from dcsmm.fues import FUES

# Suppose EGM has produced:
#   e_grid  : endogenous grid points (N,)
#   vlu     : value at each point (N,)
#   c_hat   : consumption policy (N,)
#   a_hat   : next-period assets / secondary policy (N,)
#   del_a   : policy gradient da'/de (N,)

e_clean, v_clean, c_clean, a_clean, d_clean = FUES(
    e_grid, vlu, c_hat, a_hat, del_a,
    m_bar=1.2,    # jump detection threshold
    LB=4,         # look-back buffer length
)
```

The returned arrays contain only the optimal points. Interpolate over `(e_clean, v_clean)` and `(e_clean, c_clean)` to evaluate the value function and policy on any target grid.

## Using the upper envelope registry

To compare FUES against other methods (DC-EGM, RFC, CONSAV) on the same problem:

```python
from dcsmm.uenvelope import EGM_UE

refined, raw, interpolated = EGM_UE(
    x_dcsn_hat=e_grid,
    qf_hat=vlu,
    v_cntn_hat=v_continuation,
    kappa_hat=c_hat,
    X_cntn=a_hat,
    X_dcsn=target_grid,
    uc_func_partial=marginal_utility,
    u_func={"func": utility_fn, "args": {}},
    ue_method="FUES",    # or "DCEGM", "RFC", "CONSAV"
    m_bar=1.2,
)

# refined["x_dcsn_ref"]  — refined endogenous grid
# refined["v_dcsn_ref"]  — refined values
# refined["kappa_ref"]   — refined consumption
# refined["x_cntn_ref"]  — refined next-period assets
```

## Running the retirement example

```bash
python examples/retirement/run_experiment.py \
    --params params/baseline.yml \
    --grid-size 3000
```

This solves the Iskhakov et al. (2017) retirement model with all four upper envelope methods and prints timing and Euler error comparisons.
