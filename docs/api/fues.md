# API Reference — Core Library

## `FUES` — Fast Upper-Envelope Scan

```python
from dcsmm.fues import FUES
```

### Signature

```python
FUES(
    e_grid, vlu, policy_1, policy_2, del_a,
    b=1e-10, m_bar=1.0, LB=4,
    endog_mbar=False, padding_mbar=0.0,
    include_intersections=True,
    return_intersections_separately=False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `e_grid` | ndarray (N,) | — | Endogenous decision grid. Internally sorted ascending. |
| `vlu` | ndarray (N,) | — | Value at each grid point. |
| `policy_1` | ndarray (N,) | — | Primary policy (e.g. consumption). |
| `policy_2` | ndarray (N,) | — | Secondary policy (e.g. next-period assets). Used for jump classification. |
| `del_a` | ndarray (N,) | — | Policy gradient series for endogenous jump thresholds. |
| `m_bar` | float | 1.0 | Jump detection threshold. Set to the maximum marginal propensity to save, or slightly above. |
| `LB` | int | 4 | Look-back/forward buffer length for forward and backward scans. |
| `endog_mbar` | bool | False | If True, compute endogenous jump threshold using `del_a`. |
| `padding_mbar` | float | 0.0 | Additional padding for the endogenous threshold. |
| `include_intersections` | bool | True | Interpolate crossing points at retained jumps. |
| `return_intersections_separately` | bool | False | Return intersections as a separate tuple. |

### Returns

**Default** (`return_intersections_separately=False`):

```python
(e_kept, v_kept, p1_kept, p2_kept, d_kept)
```

**With** `return_intersections_separately=True`:

```python
(fues_result, intersections)
```

### Recommended notation

The current implementation keeps some legacy parameter names for API stability.
For documentation and paper alignment, the preferred interpretation is:

| API name | Recommended code meaning | Math notation |
|---|---|---|
| `e_grid` | `x_dcsn_hat` | `\hat{x}` or `\hat{x}_v` |
| `vlu` | `v_hat` | `\hat{v}` or `\hat{v}_v` |
| `policy_1` | `kappa_hat` | `\hat{c}` in consumption-saving applications |
| `policy_2` | `x_cntn_hat` | `\hat{x}_e`, with `\hat{x}_e \equiv \hat{x}'` as the transition from the paper's current notation |
| `*_ref` outputs | refined upper-envelope objects | refined counterparts of the above |

This keeps the docs close to the current paper while making the continuation / post-decision object easier to read from a Bellman-DDSL perspective.

### Implementation notes

- Core scan is `@njit` (Numba JIT-compiled)
- Input arrays sorted internally — no pre-sorting required
- \(O(N)\) time with fixed look-back window of size `LB`
- Sub-optimal = policy jump **and** concave right turn
- Crossing points computed via two-point linear interpolation

## `EGM_UE` — Upper Envelope Registry

```python
from dcsmm.uenvelope import EGM_UE
```

Unified entry point for all upper envelope algorithms. Wraps FUES, MSS, RFC, and LTM behind a common interface.

### Signature

```python
EGM_UE(
    x_dcsn_hat, qf_hat, v_cntn_hat, kappa_hat,
    X_cntn, X_dcsn, uc_func_partial, u_func,
    ue_method="FUES", m_bar=1.0, lb=4,
    rfc_radius=0.75, rfc_n_iter=20,
    interpolate=False, include_intersections=True,
    ue_kwargs=None,
)
```

### Returns

```python
(refined, raw, interpolated)
```

- `refined` — dict: `x_dcsn_ref`, `v_dcsn_ref`, `kappa_ref`, `x_cntn_ref`, `lambda_ref`, `ue_time`
- `raw` — dict: original inputs
- `interpolated` — dict: values on `X_dcsn` (if `interpolate=True`)

### Naming note

`EGM_UE` currently uses `qf_hat` and `X_cntn` in the API. Read these as:

- `qf_hat` = raw value correspondence, with preferred paper-facing notation `v_hat`
- `X_cntn` = raw continuation / post-decision state, with preferred descriptive name `x_cntn_hat`
- `X_dcsn` = target decision grid used for interpolation and comparison

### Available methods

| `ue_method` | Algorithm | Source |
|-------------|-----------|--------|
| `"FUES"` | Fast Upper-Envelope Scan | Dobrescu & Shanker (2022) |
| `"DCEGM"` | Monotone segment selection (MSS) | Iskhakov et al. (2017), via [HARK](https://github.com/econ-ark/HARK) |
| `"RFC"` | Rooftop-cut | Dobrescu & Shanker (2024) |
| `"CONSAV"` | Local triangulation (LTM) | Druedahl (2021), via [ConSav](https://github.com/NumEconCopenhagen/ConsumptionSaving) |
| `"FUES_V0DEV"` | Original paper FUES | — |
| `"SIMPLE"` | Monotonicity filter | — |

### Registering custom engines

```python
from dcsmm.uenvelope.upperenvelope import register

@register("MY_METHOD")
def my_engine(x_dcsn_hat, qf_hat, kappa_hat, X_cntn, *,
              uc_func_partial, **kwargs):
    return {
        "x_dcsn_ref": ..., "v_dcsn_ref": ...,
        "kappa_ref": ..., "x_cntn_ref": ...,
        "lambda_ref": uc_func_partial(...),
    }
```

## Helpers

```python
from dcsmm.fues.helpers.math_funcs import interp_as, interp_as_scalar
```

### `interp_as` — 1D array interpolation

```python
interp_as(xp, yp, x, extrap=False)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `xp` | ndarray (M,) | Grid points (sorted ascending) |
| `yp` | ndarray (M,) | Values at grid points |
| `x` | ndarray (N,) | Evaluation points |
| `extrap` | bool | Extrapolate beyond grid bounds (default: False, clamp) |

Returns ndarray (N,). Numba JIT-compiled.

### `interp_as_scalar` — 1D scalar interpolation

```python
interp_as_scalar(xp, yp, x)
```

Same as `interp_as` for a single float `x`. Numba JIT-compiled.

### `correct_jumps1d` — jump correction

```python
correct_jumps1d(values, grid, threshold, policy_dict)
```

Detects and corrects spurious jumps in interpolated functions by checking gradient against threshold and re-interpolating. Numba JIT-compiled.

### Convention

All 1D interpolation in `dcsmm` uses `interp_as` / `interp_as_scalar`. Do not use `np.interp` directly.

