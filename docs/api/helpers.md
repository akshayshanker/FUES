# Helpers

```python
from dcsmm.fues.helpers.math_funcs import interp_as, interp_as_scalar
```

## `interp_as` — 1D array interpolation

Linear interpolation on an irregular grid. All 1D interpolation in `dcsmm` should use this function.

```python
interp_as(xp, yp, x, extrap=False)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `xp` | ndarray (M,) | Grid points (must be sorted ascending) |
| `yp` | ndarray (M,) | Values at grid points |
| `x` | ndarray (N,) | Points at which to evaluate |
| `extrap` | bool | If True, extrapolate beyond grid bounds. Default: False (clamp). |

**Returns:** ndarray (N,) — interpolated values at `x`.

Numba JIT-compiled.

## `interp_as_scalar` — 1D scalar interpolation

Same as `interp_as` but for a single evaluation point.

```python
interp_as_scalar(xp, yp, x)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `xp` | ndarray (M,) | Grid points |
| `yp` | ndarray (M,) | Values |
| `x` | float | Single evaluation point |

**Returns:** float.

Numba JIT-compiled.

## `correct_jumps1d` — jump correction

Detects and corrects spurious jumps in interpolated value/policy functions by checking the gradient against a threshold and replacing jump regions via local re-interpolation.

```python
correct_jumps1d(values, grid, threshold, policy_dict)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `values` | ndarray (N,) | Value function on `grid` |
| `grid` | ndarray (N,) | Evaluation grid |
| `threshold` | float | Maximum allowed gradient magnitude |
| `policy_dict` | numba typed Dict | String-keyed dict of policy arrays to correct alongside values |

**Returns:** `(corrected_values, corrected_policies)`.

Numba JIT-compiled.

## Usage convention

All 1D interpolation in `dcsmm` uses `interp_as` / `interp_as_scalar`. Do not use `np.interp` or `interpolation.interp` directly. If interpolation performance needs optimising, do it inside these functions.
