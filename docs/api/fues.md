# API Reference — Core Library

## `FUES` — Fast Upper-Envelope Scan

```python
from dcsmm.fues import FUES
```

### Signature

```python
FUES(
    x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat, del_kappa_hat=None,
    m_bar=1.0, LB=4,
    endog_mbar=False, padding_mbar=0.0,
    include_intersections=True,
    return_intersections_separately=False,
    single_intersection=False,
    no_double_jumps=True,
    disable_jump_checks=False,
    assume_sorted=False,
    eps_d=None, eps_sep=None, eps_fwd_back=None, parallel_guard=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x_dcsn_hat` | ndarray (N,) | — | Endogenous decision grid. Internally |
|  |  |  | sorted ascending. |
| `v_hat` | ndarray (N,) | — | Value at each grid point. |
| `kappa_hat` | ndarray (N,) | — | Primary policy (e.g. consumption). |
| `x_cntn_hat` | ndarray (N,) | — | Secondary policy (e.g. next-period |
|  |  |  | assets). Used in the double-jump |
|  |  |  | post-clean and carried through |
|  |  |  | intersections; the scan itself |
|  |  |  | classifies jumps on `kappa_hat`. |
| `del_kappa_hat` | ndarray (N,) | None | Derivative of the control `kappa` |
|  |  |  | along the endogenous grid (d kappa |
|  |  |  | / d x_dcsn); supplies the |
|  |  |  | grid-local jump threshold. |
|  |  |  | Required when `endog_mbar=True`. |
| `m_bar` | float | 1.0 | Jump threshold on `kappa_hat` difference |
|  |  |  | quotient. Set to max control slope along a branch |
|  |  |  | (max MPC in consumption models), or slightly above. |
| `LB` | int | 4 | Look-back/forward scan buffer length |
| `endog_mbar` | bool | False | If True, compute endogenous jump threshold |
|  |  |  | using `del_kappa_hat`. |
| `padding_mbar` | float | 0.0 | Additional padding for the endogenous |
|  |  |  | threshold. |
| `include_intersections` | bool | True | Interpolate crossing points at |
|  |  |  | retained jumps. |
| `return_intersections_separately` | bool | False | Return intersections as |
|  |  |  | a separate tuple. |
| `single_intersection` | bool | False | Create only one intersection per |
|  |  |  | crossing. |
| `no_double_jumps` | bool | True | Suppress consecutive double jumps in the |
|  |  |  | scan. |
| `disable_jump_checks` | bool | False | Override jump validity checks. |
| `assume_sorted` | bool | False | Skip the internal ascending sort of the |
|  |  |  | input arrays. |
| `eps_d` | float | None | Minimum grid-point separation (None uses the |
|  |  |  | module default). |
| `eps_sep` | float | None | Min separation for intersections (None → |
|  |  |  | default) |
| `eps_fwd_back` | float | None | Proximity threshold for forward/backward |
|  |  |  | scans (None uses the module default). |
| `parallel_guard` | float | None | Guard against near-parallel segments |
|  |  |  | (None uses the module default). |

### Returns

**Default** (`return_intersections_separately=False`):

```python
(x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, del_kappa_ref)
```

**With** `return_intersections_separately=True`:

```python
(fues_result, intersections)
```

### Recommended notation

The current implementation uses the recommended parameter names directly
(earlier releases used legacy names). The interpretation is:

| API name | Legacy name | Math notation |
|---|---|---|
| `x_dcsn_hat` | `e_grid` | `\hat{x}` or `\hat{x}_v` |
| `v_hat` | `vlu` | `\hat{v}` or `\hat{v}_v` |
| `kappa_hat` | `policy_1` | `\hat{c}` in consumption-saving applications |
| `x_cntn_hat` | `policy_2` | `\hat{x}_e`, with `\hat{x}_e \equiv \hat{x}'` |
|  |  | as the transition from the paper's current |
|  |  | notation |
| `del_kappa_hat` | `del_a` | `d\hat{\kappa}/d\hat{x}` — derivative of the |
|  |  | control along the endogenous grid, used for |
|  |  | the endogenous jump threshold |
| `*_ref` outputs | — | refined counterparts of the above |

This keeps the docs close to the current paper while making the continuation /
post-decision object easier to read from a Bellman-DDSL perspective.

### Implementation notes

- Core scan is `@njit` (Numba JIT-compiled)
- Input arrays sorted internally — no pre-sorting required (use
  `assume_sorted=True` to skip the sort)
- \(O(N)\) time with fixed look-back window of size `LB`
- Sub-optimal = policy jump **and** concave right turn
- Crossing points computed via two-point linear interpolation

## `EGM_UE` — Upper Envelope Registry

```python
from dcsmm.uenvelope import EGM_UE
```

Unified entry point for all upper envelope algorithms. Wraps FUES, MSS, RFC,
and LTM behind a common interface.

### Signature

```python
EGM_UE(
    x_dcsn_hat, v_hat, v_cntn_hat, kappa_hat,
    x_cntn_hat, X_dcsn, uc_func_partial, u_func,
    method_switch=None, m_bar=1.0, lb=4,
    rfc_radius=0.75, rfc_n_iter=20,
    interpolate=False, include_intersections=True,
    ue_kwargs=None,
)
```

When `method_switch` is omitted (None) the engine defaults to `"FUES"`. The
deprecated keyword ``ue_method`` is still accepted as an alias of
``method_switch`` (not both at once).

### Returns

```python
(refined, raw, interpolated)
```

- `refined` — dict: `x_dcsn_ref`, `v_dcsn_ref`, `kappa_ref`, `x_cntn_ref`,
  `lambda_ref`, `ue_time`
- `raw` — dict: original inputs
- `interpolated` — dict: values on `X_dcsn` (if `interpolate=True`)

### Naming note

- `X_dcsn` = target decision grid used for interpolation and comparison

### Available methods

| `method_switch` | Algorithm | Source |
|-------------|-----------|--------|
| `"FUES"` | Fast Upper-Envelope Scan | Dobrescu & Shanker (2022) |
| `"DCEGM"` (alias `"MSS"`) | Monotone segment selection | Iskhakov et al. |
|  | (MSS) |  |
|  |  | (2017), via [HARK](https://github.com/econ-ark/HARK) |
| `"RFC"` | Rooftop-cut | Dobrescu & Shanker (2024) |
| `"CONSAV"` | Local triangulation (LTM) | Druedahl (2021), via |
|  |  | [ConSav](https://github.com/NumEconCopenhagen/ConsumptionSaving) |
| `"FUES_V0DEV"` | Original paper FUES | — |
| `"FUES_V0_1DEV"` | FUES v0.1dev baseline | — |
| `"FUES_V0_2DEV"` | FUES v0.2dev (same engine as `"FUES"`) | — |
| `"SIMPLE"` | Monotonicity filter | — |

### Registering custom engines

```python
from dcsmm.uenvelope.upperenvelope import register

@register("MY_METHOD")
def my_engine(x_dcsn_hat, v_hat, kappa_hat, x_cntn_hat, *,
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
interp_as_scalar(xp, yp, x, extrap=False)
```

Same as `interp_as` for a single float `x`. Numba JIT-compiled.

### `correct_jumps1d` — jump correction

```python
correct_jumps1d(data, x, gradient_jump_threshold, policy_value_funcs)
```

Detects and corrects spurious jumps in interpolated functions by checking
gradient against threshold and re-interpolating. `policy_value_funcs` is a
dict of aligned 1D arrays corrected the same way (pass a `numba.typed.Dict`
— the function is Numba JIT-compiled). Returns `(corrected_data,
corrected_policy_value_funcs)`.

### Convention

All 1D interpolation in `dcsmm` uses `interp_as` / `interp_as_scalar`. Do not
use `np.interp` directly.

