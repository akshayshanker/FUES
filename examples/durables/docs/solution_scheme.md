# Solution Scheme: Durables Model

A finite-horizon lifecycle model of household consumption, saving,
and durable housing. Each period the household draws stochastic
income y(z, age) from an age-tenure polynomial with AR(1) shock z,
holds financial assets a and housing stock h, and makes two linked
decisions: (1) whether to adjust housing (tenure / adjustment-decision
stage), and (2) how much to consume and save (consumption stage).
Adjusters liquidate housing at market price R_H·(1−δ)·h and choose
new housing h_choice at transaction cost (1+τ)·h_choice. Keepers
retain depreciated housing (1−δ)·h. The model is solved by backward
induction with EGM (Endogenous Grid Method) and FUES upper envelopes.

## Perch definitions

A "perch" is a point in the information timeline of a single stage.

| Label | DDSL name | DP concept | YAML bracket |
|---|---|---|---|
| `arvl` | arrival | post-expectation, pre-optimization | `[<]` |
| `dcsn` | decision | post-optimization | unmarked |
| `cntn` | continuation | post-transition, pre-expectation | `[>]` |

**Reading rule**: `V[<]` is the arrival value, `V` is the decision
value, `V[>]` is the continuation value. The perch tag is carried
by the dict key (`'arvl'`, `'dcsn'`, `'cntn'`), not the variable
name.

| Stage | Perches stored |
|---|---|
| `keeper_cons` | `dcsn`, `cntn` (if `store_cntn`) |
| `adjuster_cons` | `dcsn`, `cntn` (if `store_cntn`) |
| `tenure` | `dcsn`, `arvl` |

## Notation: `d_xV` convention

Marginal values are named `d_xV` where `x` is the differentiation
variable: `d_aV` reads as ∂V/∂a. This maps from the YAML
`d_{x}V` by dropping the braces. The convention avoids `dV_da`
(which inverts the fraction) and keeps a uniform prefix so that
derivative keys sort together.

## Grids, exogenous, and solver data

Created once by `make_grids(cp)` and shared across all periods.

```python
grids = {
    'a':          ndarray,   # financial assets (n_a,)
    'h':          ndarray,   # housing stock (n_h,)
    'h_choice':   ndarray,   # adjuster housing choice grid (n_he,)
    'we':         ndarray,   # total wealth (n_w,)
    'z':          ndarray,   # exogenous shock values (n_z,)
    'Pi':         ndarray,   # Markov transition matrix (n_z, n_z)
    'X_all':      ndarray,   # state-space index tuples (n_states, 3)
    'UGgrid_all': tuple,     # UCGrid tuple for 2D interpolation on (a, h)
}
```

Endogenous arrays (e.g. `m_endog`) are solver output and live in
the per-period solution dict, not in `grids`.

### Which grid do dcsn quantities live on?

Depends on the stage and the EGM scheme:

| Stage | dcsn grid | Why |
|---|---|---|
| `keeper_cons` | `grids['a']` per `h` slice | EGM inverts onto asset grid; 3D arrays `(n_z, n_a, n_h)` |
| `adjuster_cons` | `grids['we']` | EGM inverts onto 1D wealth grid; 2D arrays `(n_z, n_w)` |
| `tenure` | `grids['a']` × `grids['h']` | state space is (a, h); 3D arrays `(n_z, n_a, n_h)` |

## Working value bundle

Flat dict passed between operators during the backward solve.
Carries value + marginal values at a single perch.

```python
vlu_cntn = {
    'V':    ndarray,   # value function
    'd_aV': ndarray,   # ∂V/∂a
    'd_hV': ndarray,   # ∂V/∂h
}
```

These keys use **tenure-state variable names** (`a`, `h`) because
the working bundle flows from tenure's arrival perch. Consuming
stages (keeper, adjuster) interpret `d_aV` as their `∂V[>]/∂a_nxt`
— the conceptual rename is implicit. When `store_cntn=True`, the
stored solution renames to poststate symbols (`d_a_nxtV`,
`d_h_nxtV`) for clarity.

## What gets stored

**Principle**: states and poststates are discretized on scheme grids
(or endogenous grids from EGM). Grid arrays are NOT duplicated in
the solution — they are already in `grids`.

The solution stores only:
- **Values** (`V`) and **marginal values** (`d_wV`, `d_aV`, ...)
- **Controls** (`c`, `h_choice`, `adj`) — agent choices
- **Endogenous grids** (`m_endog`) — when EGM creates a non-uniform
  grid not in `grids`
- **Timing** (`solve_time`, `keeper_ms`, `adj_ms`, `discrete_ms`)

Poststates like `a_nxt` are the exogenous grid for EGM and are
not stored.

## Stored solution (per period)

Three-level dict: **stage → perch → quantity**.

The `store_cntn` flag (from `settings.yaml`) controls whether
continuation-perch data is stored. When `False` (default), only
`dcsn` and `arvl` perches are stored. When `True`, raw EGM and
FUES-refined data are added under `cntn`.

```python
sol_h = {
    'h': int,              # period index (0 = terminal)
    't': int,              # age

    'keeper_cons': {
        'dcsn': {
            'V':    ndarray,       # (n_z, n_a, n_h) value on asset grid
            'd_wV': ndarray,       # (n_z, n_a, n_h) ∂V/∂w_keep = d_c u(c)
            'd_hV': ndarray,       # (n_z, n_a, n_h) ∂V/∂h_keep (= phi_keep)
            'c':    ndarray,       # (n_z, n_a, n_h) consumption
        },
        # cntn: present only when store_cntn=True
        'cntn': {
            'V':        ndarray,               # (n_z, n_a, n_h) continuation value
            'd_a_nxtV': ndarray,               # (n_z, n_a, n_h) renamed from d_aV
            'd_h_nxtV': ndarray,               # (n_z, n_a, n_h) renamed from d_hV
            'c':        dict[(iz,ih), ndarray], # FUES-refined c per (z,h) slice
            'm_endog':  dict[(iz,ih), ndarray], # FUES-refined endogenous grid
        },
    },

    'adjuster_cons': {
        'dcsn': {
            'V':        ndarray,   # (n_z, n_w) value on wealth grid
            'd_wV':     ndarray,   # (n_z, n_w) ∂V/∂m = d_c u(c)
            'c':        ndarray,   # (n_z, n_w) consumption
            'h_choice': ndarray,   # (n_z, n_w) housing choice
        },
        # cntn: present only when store_cntn=True
        'cntn': {
            'c':          ndarray,   # (n_z, n_he, egm_n) raw EGM consumption
            'm_endog':    ndarray,   # (n_z, n_he, egm_n) raw endogenous wealth
            'a_nxt_eval': ndarray,   # (n_z, n_he, egm_n) poststate a at EGM points
            'h_nxt_eval': ndarray,   # (n_z, n_he, egm_n) poststate h at EGM points
            '_refined':   dict,      # per-z FUES-refined arrays (see below)
        },
    },

    'tenure': {
        'dcsn': {
            'V':    ndarray,       # (n_z, n_a, n_h) V after max(keep, adjust)
            'd_aV': ndarray,       # (n_z, n_a, n_h) ∂V/∂a (chain rule)
            'd_hV': ndarray,       # (n_z, n_a, n_h) ∂V/∂h (chain rule)
            'adj':  ndarray,       # (n_z, n_a, n_h) adjustment indicator (1=adjust)
        },
        'arvl': {
            'V':    ndarray,       # (n_z, n_a, n_h) E_z[V]
            'd_aV': ndarray,       # (n_z, n_a, n_h) E_z[∂V/∂a]
            'd_hV': ndarray,       # (n_z, n_a, n_h) E_z[∂V/∂h]
        },
    },

    'solve_time':   float,     # total period solve time (seconds)
    'keeper_ms':    float,     # keeper stage time (ms)
    'adj_ms':       float,     # adjuster stage time (ms)
    'discrete_ms':  float,     # tenure branching time (ms)
}
```

### Notes on specific fields

**`keeper_cons.dcsn.d_hV`** is the composite housing marginal
`phi_keep = d_h u(h_keep) + beta · ∂V[>]/∂h_nxt`. It combines the
within-period marginal utility of housing with the discounted
shadow value of housing carried forward. In the code this variable
is named `phi_keep`.

**`keeper_cons.cntn.c` and `m_endog`** are tuple-keyed dicts
`{(iz, ih): ndarray}`, one 1D array per (z, h) slice. This is
because the keeper EGM + FUES operates per-slice; the arrays have
variable length after FUES pruning.

**`adjuster_cons.cntn._refined`** is a solver-diagnostic dict keyed
by shock index `iz`. Each entry contains post-FUES arrays:
`{'m_endog', 'vf', 'a_nxt_eval', 'h_nxt_eval'}`.

**`dcsn_derived`** (planned, not yet implemented): tenure would
assemble branch-selected policies (`c`, `h` from the chosen branch)
into a `dcsn_derived` sub-dict. Currently not produced by the
backward solve.

## Naming rules

### Values

| YAML | Code key | Perch |
|---|---|---|
| `V[<]` | `'V'` | `arvl` |
| `V` (unmarked) | `'V'` | `dcsn` |
| `V[>]` | `'V'` | `cntn` |

### Marginal values

Each stage uses its own YAML variable names:

| Stage | YAML | Code key | Perch |
|---|---|---|---|
| keeper_cons | `d_{w}V` | `'d_wV'` | dcsn |
| keeper_cons | `d_{h}V` | `'d_hV'` | dcsn |
| keeper_cons | `d_{a_nxt}V[>]` | `'d_a_nxtV'` | cntn (stored) |
| keeper_cons | `d_{h_nxt}V[>]` | `'d_h_nxtV'` | cntn (stored) |
| adjuster_cons | `d_{w}V` | `'d_wV'` | dcsn |
| tenure | `d_{a}V` | `'d_aV'` | dcsn |
| tenure | `d_{h}V` | `'d_hV'` | dcsn |
| tenure | `d_{a}V[<]` | `'d_aV'` | arvl |
| tenure | `d_{h}V[<]` | `'d_hV'` | arvl |

Rule: `d_{x}V` → `d_xV`. Drop braces, keep prefix.

**Working bundle vs stored**: the working bundle uses `d_aV`/`d_hV`
(tenure-state names). When stored at keeper's `cntn` perch, these
are renamed to `d_a_nxtV`/`d_h_nxtV` (poststate names) for clarity.

### Controls (stored)

| Stage | YAML name | Code key | Perch |
|---|---|---|---|
| keeper_cons | `c` | `'c'` | dcsn |
| adjuster_cons | `c` | `'c'` | dcsn |
| adjuster_cons | `h_choice` | `'h_choice'` | dcsn |
| tenure | `adj` | `'adj'` | dcsn |

### Poststates (NOT stored — they are the grid)

| Stage | YAML name | Grid |
|---|---|---|
| keeper_cons | `a_nxt` | `grids['a']` |
| keeper_cons | `h_nxt` | `grids['h']` (= h_keep, pass-through) |
| adjuster_cons | `a_nxt` | `grids['a']` |
| adjuster_cons | `h_nxt` | `grids['h_choice']` (= h_choice) |

### Evaluation domain convention

Every solver stores at `cntn` the **evaluation domain** on which
its raw outputs are defined. The test:

> *Can the domain be recovered from `grids` alone?*
> If yes → omit. If no → store.

| Case | Domain | Store? |
|---|---|---|
| 1D EGM (keeper) | = `grids['a']` | No |
| Partial EGM (adjuster) | ⊂ `grids['a']` × `grids['h_choice']` | Yes (`a_nxt_eval`, `h_nxt_eval`) |
| FUES upper envelope | ⊂ `grids['a']` (pruned) | Yes (inside tuple-keyed dicts) |

The adjuster's evaluation domain is a **paired** set of points,
not a Cartesian product:

```python
# (a_nxt_eval[iz, ihp, k], h_nxt_eval[iz, ihp, k])
# for iz = 0..n_z-1, ihp = 0..n_he-1, k = 0..egm_n-1
```

This is the solver-constructed submanifold of `grids['a']` ×
`grids['h_choice']` where the partial EGM inversion is well-defined
(Dobrescu & Shanker).

## Mover I/O contracts

### keeper_cons.cntn_to_dcsn_mover

Inverts the Euler equation on the continuation grid, constructs
the endogenous cash-on-hand grid, applies the FUES upper envelope,
and interpolates to the scheme grid.

```
INPUT:  vlu_cntn = {V, d_aV, d_hV}     shape (n_z, n_a, n_h)
OUTPUT: (A_keep, C_keep, V_keep, dVw_keep, phi_keep, cntn_data)
        → stored as dcsn: {V, d_wV, d_hV, c}
```

Keeper does not have a separate `dcsn_to_arvl_mover` — the YAML
defines it as identity (pass-through). The keeper's dcsn arrays
are passed directly to the tenure branching mover.

### adjuster_cons.cntn_to_dcsn_mover

Solves the dual InvEuler (consumption and housing FOCs
simultaneously), root-finds along the housing Euler residual for
each h_choice grid point, constructs a 1D endogenous wealth grid,
applies FUES, and interpolates to the wealth grid.

```
INPUT:  vlu_cntn = {V, d_aV, d_hV}     shape (n_z, n_a, n_h)
OUTPUT: (A_adj, C_adj, H_adj, V_adj, dVw_adj, cntn_data)
        → stored as dcsn: {V, d_wV, c, h_choice}
```

Like keeper, the adjuster's `dcsn_to_arvl_mover` is identity
and implicit in the code.

### tenure.cntn_to_dcsn_mover (branching)

Computes branch transitions (keep: w_keep = R·a + y(z),
h_keep = (1−δ)·h; adjust: w_adj = R·a + R_H·(1−δ)·h + y(z)),
evaluates keeper and adjuster values at transition points by
interpolation, takes the pointwise max, and applies the
MarginalBellman chain rule.

```
INPUT:  vlu_cntn, grids,
        keeper arrays  (A_keep, C_keep, V_keep, dVw_keep, phi_keep),
        adjuster arrays (A_adj, C_adj, H_adj, V_adj, dVw_adj)
OUTPUT: ({V, d_aV, d_hV}, {adj})
        → stored as dcsn: {V, d_aV, d_hV, adj}
```

### tenure.dcsn_to_arvl_mover (E_z)

Conditions on the current shock z by computing expectations over
tomorrow's shock z' using the Markov transition matrix Pi.

```
INPUT:  {V, d_aV, d_hV}            shape (n_z, n_a, n_h)
OUTPUT: {V, d_aV, d_hV}            shape (n_z, n_a, n_h)
        → stored as arvl: {V, d_aV, d_hV}
```

## Raw EGM outputs in cntn

EGM inverts the Euler equation on the continuation grid. The raw
outputs are **continuation-measurable** and belong in `cntn`.

`c[>]` denotes "an endogenous-grid representation of the policy,
not a semantic claim that the decision is chosen at cntn."

```
cntn_to_dcsn_mover (EGM + FUES)
├── INPUT:  d_aV[>] at cntn       ← continuation marginal
├── STEP 1: InvEuler → ĉ(a') = c[>]  ← continuation-measurable
├── STEP 2: Reverse transition → m̂   ← continuation-measurable
├── STEP 3: FUES + interpolation      ← representation transform
└── OUTPUT: c*(w), V(w) at dcsn       ← decision-measurable
```

In standard 1D EGM (keeper) the exogenous grid is `grids['a']`
— no need to store it.

In partial EGM (adjuster), the exogenous grid is a chosen subset
of the poststate space. Store the evaluation points in `cntn` as
`a_nxt_eval`, `h_nxt_eval`.

## Inter-period wiring (twister)

The twister maps tenure's `arvl` to the next period's working
bundle. In this model, the state space `(a, h)` is period-invariant,
so the twister is a **pure identity pass-through**:

```python
def _apply_twister(prev_sol, twister):
    return prev_sol['tenure']['arvl']
```

The working bundle retains tenure-state names (`d_aV`, `d_hV`)
throughout. Consuming stages interpret these as their continuation
marginals (`∂V[>]/∂a_nxt`, `∂V[>]/∂h_nxt`) without a physical
key rename.

## Example: accessing solution data

```python
from examples.durables.solve import solve

nest, cp, grids, callables, settings = solve(
    'examples/durables/syntax')

sol = nest['solutions'][5]

V          = sol['tenure']['dcsn']['V']
d_aV       = sol['tenure']['dcsn']['d_aV']
adj        = sol['tenure']['dcsn']['adj']
c_keep     = sol['keeper_cons']['dcsn']['c']
d_wV_keep  = sol['keeper_cons']['dcsn']['d_wV']
phi_keep   = sol['keeper_cons']['dcsn']['d_hV']
c_adj      = sol['adjuster_cons']['dcsn']['c']
h_adj      = sol['adjuster_cons']['dcsn']['h_choice']
vlu_arvl   = sol['tenure']['arvl']
```

## Migration from current code

| Current (old) | New |
|---|---|
| `vlu_cntn['V']` | `vlu_cntn['V']` |
| `vlu_cntn['dV']['a']` | `vlu_cntn['d_aV']` |
| `vlu_cntn['dV']['h']` | `vlu_cntn['d_hV']` |
| `sol['tenure']['vlu_dcsn']['V']` | `sol['tenure']['dcsn']['V']` |
| `sol['tenure']['vlu_dcsn']['dV']['a']` | `sol['tenure']['dcsn']['d_aV']` |
| `sol['tenure']['vlu_arvl']` | `sol['tenure']['arvl']` |
| `sol['tenure']['pol']['c']` | (planned: `sol['tenure']['dcsn_derived']['c']`) |
| `sol['keeper_cons']['C']` | `sol['keeper_cons']['dcsn']['c']` |
| `sol['keeper_cons']['A']` | not stored (poststate = `grids['a']`) |

---

*Design sources: Matsya review (22 Mar 2026), DDSL spec
(PROVISIONAL), ConSav / HARK flat-key conventions.*
