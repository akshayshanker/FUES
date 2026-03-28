# Devspec: Pre-scan adjuster FOC roots in the tenure arvl mover

**Date:** 2026-03-28
**Status:** Draft
**Scope:** `conditioning.py`, `branching.py`, `adjuster_egm.py`, `solve.py`

## 1. Problem

The adjuster's `cntn_to_dcsn_egm` spends most of its time in `inv_euler_h_residual`, which evaluates the housing FOC residual at every `a_grid` point (300) for every `(ihp, iz)` pair (300 × 4 = 1200). That is **360K residual evaluations** with **strided memory access** (`dv_a_cntn[iz, :, ihp]` has stride `n_h × 8 = 2400` bytes along the a-axis). This dominates the FUES adjuster's runtime and makes it slower than NEGM.

The non-root-finding work in the adjuster (FUES + interp + recovery) is essentially free. The bottleneck is the scan itself.

## 2. Key insight

The tenure's `arvl_mover` (the `condition_V` function in `conditioning.py`) already visits every `(i_a, i_h)` grid point to compute `E_z[d_aV]` and `E_z[d_hV]`. These post-expectation arrays are exactly what the adjuster later scans for sign changes. If we piggyback the housing FOC sign-change detection onto this existing pass — or a lightweight post-pass on the warm arrays — the adjuster receives pre-computed root locations and skips the 360K-evaluation scan entirely.

## 3. The housing FOC residual

From the formal MDP (Stage 2, FOC-h):

```
R(a', h') = (1+τ) · ∂_a v_succ(a', h') − ∂_h v_succ(a', h') − ∂_h u(c, h')
```

where `c = (α / ∂_a v_succ)^(1/γ_c)` from the asset Euler inversion (FOC-c).

### Separable utility (current default)

`∂_h u` does **not** depend on c:

```
∂_h u(c, h) = (1−α) · κ^(1−γ_h) · h^(−γ_h)
```

So the residual simplifies to:

```
R(a', h') = (1+τ) · d_aV[iz, j, ihp] − d_hV[iz, j, ihp] − threshold[ihp]
```

where `threshold[ihp] = (1−α) · κ^(1−γ_h) · h_choice_grid[ihp]^(−γ_h)` is **constant along the a-axis**. Sign changes in R along j (for fixed iz, ihp) depend only on the difference `(1+τ) · d_aV − d_hV` crossing a per-ihp threshold.

### Cobb-Douglas utility

`∂_h u(c, h)` depends on c, which depends on `d_aV`. The residual is nonlinear in `d_aV`. Sign-change detection still works, but the residual evaluation calls `invEuler_foc_h_residual(d_aV, d_hV, h_choice)` — the same callable the adjuster currently uses.

## 4. Data flow (current)

```
condition_V(V, d_aV, d_hV)              ← runs inside tenure.arvl_mover
    │  loop (i_a prange, i_h inner):
    │    new_d_aV[:, i_a, i_h] = Pi @ d_aV[:, i_a, i_h]
    │    new_d_hV[:, i_a, i_h] = Pi @ d_hV[:, i_a, i_h]
    │    new_V[:, i_a, i_h]    = Pi @ V[:, i_a, i_h]
    ▼
vlu_cntn = {"V", "d_aV", "d_hV"}       ← identity pass to adjuster
    │
    ▼
adjuster.cntn_to_dcsn_egm(d_aV, d_hV, V)
    loop (ihp, iz):
      dv_a_1d = d_aV[iz, :, ihp]       ← strided (stride n_h)
      for j in range(n_a):             ← 300 evals per (ihp, iz)
        resid[j] = foc_h(dv_a_1d[j], dv_h_1d[j], h_choice)
      find_roots(resid, ...)
```

## 5. Proposed data flow

```
condition_V(V, d_aV, d_hV)              ← unchanged (parallel)
    │
    ▼
new_d_aV, new_d_hV, new_V              ← warm in L2/L3
    │
    ▼
_prescan_adjuster_roots(                ← NEW: sequential post-pass
    new_d_aV, new_d_hV, new_V,
    h_choice_grid, invEuler_foc_h_residual, ...)
    │
    │  loop (iz outer, j middle, ihp inner):
    │    read new_d_aV[iz, j, ihp]      ← contiguous (ihp last axis)
    │    read new_d_hV[iz, j, ihp]      ← contiguous
    │    read new_V[iz, j, ihp]         ← contiguous
    │    evaluate residual, carry prev_r[ihp]
    │    on sign change: compute root a_p, c, m, v from carried values
    │
    │  Separately per (iz, ihp): add borrowing-constraint point
    │
    ▼
precomp = {
  "m_cntn_raw":     (n_z, n_he, egm_n),
  "v_cntn_raw":     (n_z, n_he, egm_n),
  "a_nxt_cntn":     (n_z, n_he, egm_n),
  "h_choice_cntn":  (n_z, n_he, egm_n),
}
    │  passed via vlu_cntn["_adjuster_egm_roots"]
    ▼
adjuster.cntn_to_dcsn_egm(...)
    │  if precomp available: SKIP root scan, use precomp directly
    │  else: fall back to original full scan
    ▼
fues_refine(precomp arrays, m_grid)     ← unchanged
```

## 6. The pre-scan function

### 6.1 Location

New `@njit` function in `conditioning.py` (or a new `prescan.py` alongside it). Called from `branching.py::arvl_mover` right after `condition_V`.

### 6.2 Signature

```python
@njit
def prescan_adjuster_roots(
    d_aV,              # (n_z, n_a, n_h) — post-conditioning, C-order
    d_hV,              # (n_z, n_a, n_h)
    V,                 # (n_z, n_a, n_h)
    a_grid,            # (n_a,)
    h_choice_grid,     # (n_h,)  — same as h_grid for the adjuster
    invEuler_foc_h_residual,   # callable(dv_a, dv_h, h) → float
    du_c_inv_fn,       # callable(dv_a, h) → c
    u_fn,              # callable(c, h) → float
    bellman_discount,   # callable(v) → float
    housing_cost,       # callable(h) → float
    b,                 # float — borrowing limit
    egm_n,             # int — max roots per (ihp, iz)
):
    """Detect housing FOC sign changes and compute roots inline.

    Scans (iz, j, ihp) with j outer / ihp inner so reads of
    d_aV[iz, j, ihp] hit the contiguous last axis.  Carry buffers
    prev_r, prev_dv_a, prev_v (~7 KB) stay in L1.
    """
```

### 6.3 Loop structure

```
for iz in range(n_z):
    # Carry buffers — all (n_h,), ~7 KB, hot in L1
    prev_r    = empty(n_h)
    prev_dv_a = empty(n_h)
    prev_v    = empty(n_h)
    root_n    = zeros(n_h, int64)

    # Seed at j=0
    for ihp in range(n_h):                      # contiguous
        prev_dv_a[ihp] = d_aV[iz, 0, ihp]
        prev_v[ihp]    = V[iz, 0, ihp]
        prev_r[ihp]    = foc_h_resid(d_aV[iz, 0, ihp],
                                      d_hV[iz, 0, ihp],
                                      h_choice_grid[ihp])

    # Scan j=1…n_a-1
    for j in range(1, n_a):                     # sequential along a
        for ihp in range(n_h):                  # contiguous reads
            c_dv_a = d_aV[iz, j, ihp]
            c_dv_h = d_hV[iz, j, ihp]
            c_v    = V[iz, j, ihp]
            c_r    = foc_h_resid(c_dv_a, c_dv_h, h_choice_grid[ihp])

            if prev_r[ihp] * c_r < 0:
                # Root between j-1 and j — compute inline
                t = -prev_r[ihp] / (c_r - prev_r[ihp])
                a_p = a_grid[j-1] + t * (a_grid[j] - a_grid[j-1])
                dv_a_root = prev_dv_a[ihp] + t * (c_dv_a - prev_dv_a[ihp])
                c_val = du_c_inv_fn(dv_a_root, h_choice_grid[ihp])
                v_root = prev_v[ihp] + t * (c_v - prev_v[ihp])
                # Store m, a, v, h directly
                ...

            prev_r[ihp]    = c_r
            prev_dv_a[ihp] = c_dv_a
            prev_v[ihp]    = c_v
```

### 6.4 Memory access

| Access | Pattern | Cost |
|--------|---------|------|
| `d_aV[iz, j, ihp]` inner ihp | **contiguous** (last axis) | L1 hit (prefetch) |
| `d_hV[iz, j, ihp]` inner ihp | **contiguous** | L1 hit |
| `V[iz, j, ihp]` inner ihp | **contiguous** | L1 hit |
| `prev_r[ihp]` | contiguous, 2.4 KB | L1 resident |
| `prev_dv_a[ihp]` | contiguous, 2.4 KB | L1 resident |
| `prev_v[ihp]` | contiguous, 2.4 KB | L1 resident |
| Output arrays (at sign changes only) | infrequent | negligible |

**Hot working set:** ~17 KB → fits in L1 (32 KB).

### 6.5 Borrowing constraint

After the main scan, a per-(iz, ihp) cleanup loop adds the constraint point at `a' = b`. This requires `interp_as_scalar` calls on `d_aV[iz, :, ihp]` and `d_hV[iz, :, ihp]` (strided), plus optional `brentq` for CD utility. Only ~1200 calls total — negligible.

## 7. Why this differs from the fused scan in the adjuster

The fused scan **inside** `cntn_to_dcsn_egm` was slower because:

1. **Changed the adjuster's JIT-optimised code path.** The original `inv_euler_h_residual` → `find_roots_piecewise_linear` pipeline was well-optimised by numba as a clean per-(ihp, iz) function. Replacing it with a fused scan changed the function's shape, register allocation, and branch structure.

2. **The adjuster's non-root work is free.** FUES + interp + recovery is fast. Making the root scan "free" by fusing it INTO the adjuster just added complexity to the adjuster without actually saving wall time.

The arvl-mover approach is different:

- The pre-scan is a **separate function** — it doesn't change the adjuster's code path at all.
- The adjuster receives pre-computed root arrays and does **zero root scanning** — it goes straight to FUES.
- The pre-scan runs on **warm cache lines** (data freshly written by `condition_V`).
- The pre-scan's cost is amortised: it produces roots for BOTH separable and CD utility without the adjuster needing to know.

## 8. Interface changes

### 8.1 `conditioning.py` or new `prescan.py`

Add `prescan_adjuster_roots()` as described in §6.

### 8.2 `branching.py::arvl_mover`

```python
def arvl_mover(vlu_dcsn):
    Ev, Edv_a, Edv_h = condition_V(
        vlu_dcsn["V"], vlu_dcsn["d_aV"], vlu_dcsn["d_hV"])

    # Pre-scan adjuster FOC roots on the warm arrays
    precomp = prescan_adjuster_roots(
        Edv_a, Edv_h, Ev,
        a_grid, h_choice_grid,
        invEuler_foc_h_residual, du_c_inv_fn,
        u_fn, bellman_discount, housing_cost,
        b, egm_n)

    return {"V": Ev, "d_aV": Edv_a, "d_hV": Edv_h,
            "_adjuster_egm_roots": precomp}
```

### 8.3 `adjuster_egm.py::cntn_to_dcsn_egm`

Check for pre-computed roots. If present, skip the root scan:

```python
def dcsn_mover(vlu_cntn, grids):
    precomp = vlu_cntn.get("_adjuster_egm_roots")
    if precomp is not None:
        # Use pre-scanned roots — skip inv_euler_h_residual entirely
        m_cntn_raw  = precomp["m_cntn_raw"]
        v_cntn_raw  = precomp["v_cntn_raw"]
        a_nxt_cntn  = precomp["a_nxt_cntn"]
        h_choice_cntn = precomp["h_choice_cntn"]
    else:
        # Fallback: original full scan
        m_cntn_raw, v_cntn_raw, a_nxt_cntn, h_choice_cntn = \
            cntn_to_dcsn_egm(vlu_cntn["d_aV"], vlu_cntn["d_hV"], vlu_cntn["V"])

    # FUES + interp (unchanged)
    a_nxt, c, h_choice, V, refined = fues_refine(...)
    ...
```

### 8.4 `solve.py`

The `vlu_cntn` dict now carries the optional `_adjuster_egm_roots` key. No changes to `solve_period` needed — the key passes through the identity inter-period connector automatically.

## 9. Callables dependency

The pre-scan needs access to `invEuler_foc_h_residual`, `du_c_inv_fn`, `u_fn`, `bellman_discount`, `housing_cost` — these are adjuster-specific callables. They must be passed to the tenure's `arvl_mover` at construction time. This is a new dependency:

```python
# In make_tenure_ops (branching.py), receive adjuster callables
def make_tenure_ops(callables, grids, stage, adj_callables=None):
    ...
    if adj_callables is not None:
        # Build prescan with adjuster's equation primitives
        ...
```

Alternatively, the pre-scan could be constructed in `solve.py::solve_period` after both keeper and adjuster are built, and injected into the tenure arvl_mover.

## 10. DDSL architectural notes (from matsya RAG)

### Stage isolation is preserved

Matsya confirms: stages must be fully isolated — they communicate exclusively through declared `values` / `values_marginal` at perch boundaries. The pre-scan is a **solver-layer optimization**, not a stage-level operation. It operates on the declared interface (d_aV, d_hV) produced by the tenure arvl_mover and consumed by the adjuster. No stage's equations are exposed to another stage. The `_adjuster_egm_roots` key is advisory metadata attached to the solution dict — the adjuster falls back to its full scan if absent.

### Connector scaling is already applied

The tenure `_tenure_dcsn_kernel` applies chain-rule factors (`marginal_a_fn = βR·dvw`, `marginal_h_adj_fn = βR_H(1−δ)·dvw`). After `condition_V`, the output arrays already include the full discount + chain-rule scaling. The pre-scan operates on these final arrays — no additional scaling needed.

### The residual is "essentially free" for separable utility

Matsya confirms: for separable utility, `∂_h u` depends only on `h'` (not on `c`), so `R(a', h') = (1+τ)·d_aV − d_hV − threshold(h')` is pure array arithmetic on the post-conditioning arrays. Cost: O(n_a × n_h × n_z) elementwise ops.

## 11. Open questions

1. **Should `condition_V` itself be restructured?** Currently `prange` over `i_a` prevents carrying sign-change state across a-grid points. Option A: keep `condition_V` parallel + separate sequential pre-scan. Option B: restructure to `i_h` outer, `i_a` inner (loses parallelism on conditioning but fuses the two passes).

2. **CD utility.** The pre-scan calls `invEuler_foc_h_residual` which for CD utility involves `du_c_inv_fn(dv_a, h)` — a power function. Verify this doesn't add excessive per-point cost in the inner loop.

3. **Grid identity.** The pre-scan assumes `h_choice_grid` ≡ `h_grid` (the tenure/keeper housing grid). Verify this is always true in the current model.

4. **Separable simplification.** For separable utility, the residual is `(1+τ)·d_aV − d_hV − const(h)`. The `const(h)` term can be pre-computed per ihp. The inner loop becomes a single subtraction + sign check per (j, ihp) — potentially vectorisable.

## 11. Expected performance

| Component | Before | After |
|-----------|--------|-------|
| Root scan in adjuster | 360K strided evals × ~110 cycles | **0** (uses precomp) |
| Pre-scan in arvl mover | 0 | 360K contiguous evals × ~14 cycles |
| Borrowing constraint | ~1200 brentq | ~1200 brentq (unchanged) |
| FUES + interp | ~2 ms | ~2 ms (unchanged) |
| **Net change** | | **~5–8× on root-finding portion** |

The pre-scan does the same 360K evaluations but with contiguous access (~14 cycles/eval vs ~110 cycles/eval strided). The adjuster's `cntn_to_dcsn_egm` becomes a trivial passthrough to `fues_refine`.
