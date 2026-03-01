# `FUES` — Fast Upper-Envelope Scan

::: dcsmm.fues.fues.FUES

```python
from dcsmm.fues import FUES
```

## Signature

```python
FUES(
    e_grid, vlu, policy_1, policy_2, del_a,
    b=1e-10, m_bar=1.0, LB=4,
    endog_mbar=False, padding_mbar=0.0,
    include_intersections=True,
    return_intersections_separately=False,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `e_grid` | ndarray (N,) | — | Endogenous decision grid. Internally sorted ascending. |
| `vlu` | ndarray (N,) | — | Value at each grid point. |
| `policy_1` | ndarray (N,) | — | Primary policy (e.g. consumption). |
| `policy_2` | ndarray (N,) | — | Secondary policy (e.g. next-period assets). Used for jump classification. |
| `del_a` | ndarray (N,) | — | Policy gradient series for endogenous jump thresholds. |
| `m_bar` | float | 1.0 | Jump detection threshold. Set to the maximum marginal propensity to save, or slightly above. |
| `LB` | int | 4 | Look-back/forward buffer length for forward and backward scans near crossing points. |
| `endog_mbar` | bool | False | If True, compute an endogenous jump threshold at each grid point using `del_a` and `padding_mbar`. |
| `padding_mbar` | float | 0.0 | Additional padding added to the endogenous threshold when `endog_mbar=True`. |
| `include_intersections` | bool | True | If True, interpolate crossing points at retained jumps and include them in the output. |
| `return_intersections_separately` | bool | False | If True, return intersections as a separate tuple (see Returns). |

## Returns

**Default** (`return_intersections_separately=False`):

```python
(e_kept, v_kept, p1_kept, p2_kept, d_kept)
```

All arrays contain only the retained (optimal) points, with crossing-point interpolants merged in if `include_intersections=True`.

**With** `return_intersections_separately=True`:

```python
(fues_result, intersections)
```

where `fues_result = (e_kept, v_kept, p1_kept, p2_kept, d_kept)` contains only the scan-retained points, and `intersections = (inter_e, inter_v, inter_p1, inter_p2, inter_d)` contains the interpolated crossing points separately.

## Example

```python
import numpy as np
from dcsmm.fues import FUES

# Simulate EGM output for a simple problem
N = 500
a_grid = np.linspace(0.01, 100, N)

# ... (run your EGM step to produce e_grid, vlu, c_hat, a_hat, del_a)

# Apply FUES
e_clean, v_clean, c_clean, a_clean, d_clean = FUES(
    e_grid, vlu, c_hat, a_hat, del_a,
    m_bar=1.2,
    LB=4,
)

print(f"Input: {N} points, Output: {len(e_clean)} points")
```

## Implementation notes

- The core scan is Numba JIT-compiled (`@njit`) for performance
- Input arrays are sorted internally by `e_grid` — no pre-sorting required
- The scan operates in \(O(N)\) time with a fixed look-back window of size `LB`
- Sub-optimal points are identified by the conjunction of a policy jump (exceeding `m_bar`) and a concave right turn in the value correspondence
- Intersection points are computed via two-point linear interpolation between the last retained point and the first point on the new branch
