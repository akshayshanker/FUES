# Modularize retirement operators into separate movers

## Goal

Split each monolithic stage solver in `operators.py` into two separate movers following bellman calculus:

- **Decision mover** $\mathbb{B}$: continuation → decision (EGM inversion + upper envelope)
- **Arrival mover** $\mathbb{I}$: decision → arrival (compose with transition, re-approximate on Cartesian grid)

Build a generic "compose-and-interpolate" horse in `kikku.asva` for the arrival mover.

## Current state

Each stage has one monolithic solver (e.g. `solver_worker_stage`) that combines both movers. The methods YAML already declares them separately:

```yaml
# work_cons_methods.yml
methods:
  - on: cntn_to_dcsn_mover
    schemes:
      - scheme: bellman_backward
        method: !egm
      - scheme: interpolation
        method: {grid: endogenous, method: !Linear}
      - scheme: upper_envelope
        method: !FUES
        settings:
          m_bar: m_bar

  - on: dcsn_to_arvl_mover
    schemes:
      - scheme: interpolation
        method: {grid: !Cartesian, method: !Linear}
        settings:
          orders: [n_a]
          bounds: [[a_min, a_max]]
```

## Proposed split

### Worker decision mover (`dcsn_mover_work`)

The EGM + upper envelope step. Returns results on the endogenous grid.

```python
def dcsn_mover_work(dv_cntn, ddv_cntn, v_cntn, grid):
    """cntn_to_dcsn_mover for work_cons.

    Uses make_egm_1d from kikku.asva with WORKER_EGM_FNS.
    Interpolation is on endogenous grid (per methods YAML).
    Upper envelope via FUES (per methods YAML).
    """
    c_hat, v_hat, x_dcsn_hat, del_a = _invert_euler(
        dv_cntn, ddv_cntn, v_cntn, grid, 0.0)

    refined, _, _ = egm_ue_global(
        x_dcsn_hat, v_hat,
        beta * v_cntn - delta, c_hat,
        grid, grid,
        du, {"func": u, "args": {}},
        ue_method=ue_method, m_bar=m_bar, lb=10,
    )

    x_dcsn = refined["x_dcsn_ref"]
    v_dcsn = refined["v_dcsn_ref"]
    c_dcsn = refined["kappa_ref"]
    dela_dcsn = np.zeros_like(refined["x_cntn_ref"])

    return x_dcsn, v_dcsn, c_dcsn, dela_dcsn
```

### Worker arrival mover (`arvl_mover_work`)

Composes the decision-perch function with the arrival-to-decision transition $g(a) = (1{+}r)a + y$ and re-approximates on the Cartesian arrival grid. Uses a generic horse from `kikku.asva`.

```python
def arvl_mover_work(x_dcsn, v_dcsn, c_dcsn, dela_dcsn, grid):
    """dcsn_to_arvl_mover for work_cons.

    Applies g: a → w = R*a + y, interpolates decision-perch
    values onto the mapped grid (Cartesian + Linear per methods YAML).
    Patches constrained region (application-specific).
    """
    v_arvl, c_arvl = _compose_worker(x_dcsn, v_dcsn, c_dcsn, grid)

    # Constrained region (application-specific, not in horse)
    w_grid = R * grid + y
    constrained = np.where(w_grid < x_dcsn[0])
    c_arvl[constrained] = w_grid[constrained] - grid[0]
    v_arvl[constrained] = u(w_grid[constrained]) + beta * v_cntn[0] - delta

    da_arvl = np.zeros_like(v_arvl)
    return v_arvl, c_arvl, da_arvl
```

### Composed stage operator (for solve.py)

The top-level operator returned in the `{'work_cons': ...}` dict composes both movers internally. This way `solve.py` does not need to change:

```python
def work_cons_stage(dv_cntn, ddv_cntn, v_cntn, grid):
    x_dcsn, v_dcsn, c_dcsn, dela_dcsn = dcsn_mover_work(...)
    v_arvl, c_arvl, da_arvl = arvl_mover_work(x_dcsn, v_dcsn, ...)
    return (v_arvl, c_arvl, da_arvl, ue_time,
            c_hat, v_hat, x_dcsn, dela_dcsn)
```

### Retiree — same pattern

Split `solver_retiree_stage` into `dcsn_mover_ret` and `arvl_mover_ret`. The retiree's transition is $g(a_{\text{ret}}) = (1{+}r)a_{\text{ret}}$ (no income). The EGM step has no upper envelope (concave problem).

**Action required:** add a `dcsn_to_arvl_mover` block to `retire_cons_methods.yml` (currently missing — the Cartesian interpolation spec sits under `cntn_to_dcsn_mover`, conflating two movers).

## Generic horse in `kikku.asva`

### `make_compose_interp`

A generic compose-and-reapproximate factory. Takes the arrival-to-decision transition $g$ and an interpolation function, returns a callable that maps decision-perch arrays onto an arrival grid.

```python
# kikku/asva/compose_interp.py

def make_compose_interp(g_transition, interp_fn):
    """Build a compose-and-reapproximate callable.

    Parameters
    ----------
    g_transition : callable
        Maps arrival grid to decision grid: x_dcsn = g(x_arvl).
    interp_fn : callable
        Interpolation function with signature
        interp_fn(x_src, *y_src, x_target) → (*y_target,).

    Returns
    -------
    callable
        mover(x_dcsn_src, *y_dcsn_arrays, x_arvl) → (*y_arvl_arrays,)
    """
```

The transition callables are defined manually in `model.py` for now (later, these will be extracted automatically from the `arvl_to_dcsn_transition` equation in the stage syntax):

```python
# model.py
@njit(cache=True)
def g_arvl_to_dcsn_worker(a, params):
    """arvl_to_dcsn_transition: w = (1+r)*a + y."""
    R, y = params[1], params[3]  # params = [beta, R, delta, y]
    return R * a + y

@njit(cache=True)
def g_arvl_to_dcsn_retiree(a_ret, params):
    """arvl_to_dcsn_transition: w_ret = (1+r)*a_ret."""
    R = params[1]
    return R * a_ret
```

The retirement operator module constructs the horse using these callables:

```python
_compose_worker  = make_compose_interp(g_arvl_to_dcsn_worker, interp_as_2)
_compose_retiree = make_compose_interp(g_arvl_to_dcsn_retiree, interp_as_3)
```

### What stays outside the horse

- **Constrained region** handling (application-specific boundary conditions)
- **Marginal value** computation (envelope theorem: $\partial v = 1/c$)
- **Chain rule** factors ($(1{+}r)$ from the transition)
- **Diagnostics** (pre-UE points, UE timing)

These remain in the application-level arvl_mover closures.

### Validation

At operator construction time, the operator factory should validate that the methods YAML specifies the expected interpolation:

```python
arvl_spec = work_cons_methods['dcsn_to_arvl_mover']
assert arvl_spec.interp_method == 'Linear'
assert arvl_spec.grid_type == 'Cartesian'
```

## Prerequisite: rename variables to perch-based names

The current code uses legacy names (`egrid`, `vf_work`, `x_dcsn_ref`, `kappa_ref`, `q_hat`, etc.) that mix FUES output names with ad-hoc labels. With the mover split, all decision-perch quantities should use consistent `x_dcsn`, `v_dcsn`, `c_dcsn`, `dela_dcsn` naming — the endogenous grid is a method choice declared in the YAML (`interpolation: {grid: endogenous}`), not a property of the variable name.

| Current name | New name | Meaning |
|---|---|---|
| `refined["x_dcsn_ref"]`, `egrid1`, `egrid` | `x_dcsn` | decision-perch grid (endogenous, post-UE) |
| `refined["v_dcsn_ref"]`, `v_ref`, `vf_work` | `v_dcsn` | decision-perch value |
| `refined["kappa_ref"]`, `c_ref`, `sigma` | `c_dcsn` | decision-perch consumption |
| `refined["x_cntn_ref"]` | `x_cntn` | continuation-perch grid |
| `x_dcsn_hat`, `c_hat`, `v_hat`, `del_a` | `x_dcsn_hat`, `c_dcsn_hat`, `v_dcsn_hat`, `dela_dcsn_hat` | pre-UE (raw EGM output) |
| `q_hat` | `v_dcsn_hat` | same (was "Q-value") |
| `da_pre_ue` | `dela_dcsn_hat` | pre-UE concavity |

Files to update:
- `operators.py` — output variables and return values
- `solve.py` — solution dict keys in `_run_work_cons`, `_run_retire_cons`
- `outputs/plots.py` — `get_policy` key arguments, plot variable names
- `outputs/diagnostics.py` — any key lookups
- `notebooks/retirement_fues.ipynb` — cell references to `egrid`, `q_hat`, etc.

This rename should be done as a separate commit before the mover split, so the diff is clean.

## Checklist

0. **Rename** all legacy variable names to perch-based names (separate commit)
1. Add `g_arvl_to_dcsn_worker` and `g_arvl_to_dcsn_retiree` callables to `model.py`
2. Build `kikku.asva.compose_interp` with `make_compose_interp`
3. Split `solver_worker_stage` → `dcsn_mover_work` + `arvl_mover_work`
4. Split `solver_retiree_stage` → `dcsn_mover_ret` + `arvl_mover_ret`
5. Compose both into the stage operators returned by `make_stage_operators`
6. Add `dcsn_to_arvl_mover` block to `retire_cons_methods.yml`
7. Validate methods YAML at construction time
8. Run `test_retirement.py` — results must be identical
