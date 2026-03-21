# Solution Syntax: Durables Model

The solution dict returned by `solve_period` follows a
declarative convention aligned with the DDSL stage structure.
**This syntax is specific to this model** — other models
will have different stages, perches, and value/policy
structures. The DDSL is declarative: the solution syntax
is derived from the YAML stage declarations, not imposed
by the framework.

## Per-period solution

Each period's solution is a dict keyed by stage name,
with sub-dicts for values and policies at each perch:

```python
sol = {
    't': int,          # calendar time (age)
    'h': int,          # distance from terminal

    'keeper_cons': {
        'pol':      {'c': ndarray, 'a_nxt': ndarray},
        'vlu_dcsn': {'V': ndarray},
    },

    'adjuster_cons': {
        'pol':      {'c': ndarray, 'a_nxt': ndarray,
                     'h_nxt': ndarray},
        'vlu_dcsn': {'V': ndarray},
    },

    'tenure': {
        'pol':      {'c': ndarray, 'a_nxt': ndarray,
                     'h_nxt': ndarray, 'd': ndarray},
        'vlu_dcsn': {'V': ndarray,
                     'dV': {'a': ndarray, 'h': ndarray}},
        'vlu_arvl': {'V': ndarray,
                     'dV': {'a': ndarray, 'h': ndarray}},
        'dV_h_hd':      ndarray or None,
        'dV_h_hd_arvl': ndarray or None,
    },

    'solve_time': float,
}
```

## Naming convention

### Values: `vlu_{perch}`

A value dict carries `V` and optionally `dV` (a dict of
partial derivatives keyed by the state variable name):

```python
vlu_cntn = {         # continuation perch (input)
    'V':  ndarray,   # E_z[V]
    'dV': {
        'a': ndarray,  # E_z[d_a V]
        'h': ndarray,  # E_z[d_h V]
    },
}

vlu_dcsn = {         # decision perch (stage output)
    'V':  ndarray,   # V (value function)
    'dV': {
        'a': ndarray,  # d_a V (marginal value w.r.t. a)
        'h': ndarray,  # d_h V (marginal value w.r.t. h)
    },
}

vlu_arvl = {         # arrival perch (E_z conditioned)
    'V':  ndarray,   # E_z[V]
    'dV': {
        'a': ndarray,  # E_z[d_a V]
        'h': ndarray,  # E_z[d_h V]
    },
}
```

The `dV` sub-dict keys match the state variable names
from the YAML `values_marginal` declarations:

```yaml
values_marginal:
    d_{a}V[<]: '@in R'    # -> dV['a']
    d_{h}V[<]: '@in R'    # -> dV['h']
```

### Policies: `pol`

A flat dict of policy arrays keyed by YAML control /
poststate names:

```python
pol = {
    'c':     ndarray,   # consumption (control)
    'a_nxt': ndarray,   # savings (poststate)
    'h_nxt': ndarray,   # housing choice (poststate)
    'd':     ndarray,   # discrete choice (control)
}
```

Not all stages produce all keys:

| Stage | `pol` keys |
|-------|-----------|
| `keeper_cons` | `c`, `a_nxt` |
| `adjuster_cons` | `c`, `a_nxt`, `h_nxt` |
| `tenure` | `c`, `a_nxt`, `h_nxt`, `d` |

### Perch labels

| Label | Meaning | When used |
|-------|---------|-----------|
| `cntn` | Continuation perch | Input to `dcsn_mover` (from twister) |
| `dcsn` | Decision perch | Output of `dcsn_mover` |
| `arvl` | Arrival perch | Output of `arvl_mover` (E_z conditioned) |

### Inter-period wiring (twister)

The twister maps `vlu_arvl` from period h to `vlu_cntn`
for period h+1. For this model the rename is identity
(variable names match across periods):

```python
vlu_cntn_{h+1} = sol_h['tenure']['vlu_arvl']
```

## Why this syntax is model-specific

The DDSL is declarative: the YAML stage declarations
determine what values and policies exist. Different
models will have:

- Different state variables (hence different `dV` keys)
- Different controls (hence different `pol` keys)
- Different stages and wave orderings
- Different perch structures (some stages may lack
  `vlu_arvl` if there are no exogenous shocks)

The `vlu_{perch}` + `pol` convention is general, but
the concrete keys inside each dict are derived from the
model's YAML syntax. There is no fixed schema — the
solution syntax is an emergent property of the
declarations.

## Example: accessing solution data

```python
from examples.durables2_0.solve import solve

nest, model, ops, waves = solve(
    'examples/durables2_0/syntax')

# Period at distance h=5 from terminal
sol = nest['solutions'][5]

# Branching stage value function
V = sol['tenure']['vlu_dcsn']['V']

# Marginal value d_a V
dV_a = sol['tenure']['vlu_dcsn']['dV']['a']

# Consumption policy (after discrete choice)
c = sol['tenure']['pol']['c']

# Keeper-specific consumption
c_keep = sol['keeper_cons']['pol']['c']

# E_z-conditioned continuation for next period
vlu_cntn = sol['tenure']['vlu_arvl']
```

## Comparison with retirement model

The retirement model uses the same `vlu` + `pol`
convention but with different contents:

| | Durables | Retirement |
|---|---------|-----------|
| State vars | `a`, `h` | `a` |
| `dV` keys | `{'a', 'h'}` | `{'a'}` (stored as `ddv`) |
| Stages | keeper, adjuster, tenure | retire_cons, work_cons, labour_mkt_decision |
| Branching `arvl_mover` | E_z conditioning | None (no shocks) |
| `pol` keys | `c, a_nxt, h_nxt, d` | `c, a_nxt, d` |
```
