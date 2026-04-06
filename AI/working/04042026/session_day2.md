# Session Working Note — 5-6 April 2026

## Summary

Continued debugging CE gap between EGM(FUES) and NEGM(FUES). Made major
infrastructure changes (consav.interp_2d, non-uniform grids, grid_phi,
marginal recomputation). Diagnosed but did not fully resolve a persistent
~0.07% CE deficit for FUES relative to NEGM despite FUES having 1-2
orders better Euler errors.

---

## Changes implemented

### 1. consav.linear_interp.interp_2d replaces eval_linear

All 2D interpolation now uses consav's `interp_2d` which supports
non-uniform rectilinear grids. Removed all `eval_linear`/`UCGrid`
dependencies from horses/. Files: branching.py, adjuster_egm.py,
keeper_egm.py, simulate.py.

Numba readonly array issue: `consav.interp_2d` rejects readonly arrays
from njit closures. Workaround: NEGM objectives use our own
`interp2d_nonuniform` (2 binary searches, proper bilinear). Other call
sites use `consav.interp_2d` directly.

### 2. Non-uniform wealth grid via grid_phi

`model.py`: added `nonlinspace(x_min, x_max, n, phi)` — Druedahl
convention. `grid_phi > 1` packs points near x_min. Applied to
`we_grid` only (a/h grids stay uniform for now since `interp_2d`
handles non-uniform).

### 3. a_min, h_min, w_min from settings (not hardcoded b)

**Bug found:** `make_grids` was using `b` for ALL grid lower bounds,
ignoring `a_min`, `h_min`, `w_min` settings. Fixed to read each
independently. Also added auto-floor:
`w_min >= R*a_min + R_H*(1-delta)*h_min`.

### 4. Terminal utility in CE computation

The simulation NPV was missing the terminal utility `term_u(w_T)`.
Added `term_u(R*a_nxt + R_H*(1-delta)*h_nxt)` at the last period,
discounted by `beta^(T-t_ce_start)`.

### 5. NEGM h_hi fix

`h_hi = (wealth - b) / fac_housing` (was `wealth / fac_housing + b`).
Correct budget constraint upper bound on housing choice.

### 6. NEGM -inf elimination

Added `V[iz, iw] = -1e18` guard when optimizer returns -inf or NaN.
Also handle degenerate search intervals (h_hi <= h_lo).

### 7. Marginal recomputation in tenure kernel

Tenure kernel computes `dvw_k = du_c(c_k_interp, h_keep)` and
`dvw_a = du_c(c_adj_interp, h_adj)` from interpolated consumption
instead of interpolating pre-computed du_c directly. More accurate
because c^(-gamma) is highly nonlinear.

### 8. Extrapolation slope cap

`interp_as` and `interp_as_scalar`: cap extrapolation slopes at 1e8
to prevent wild values from steep edge intervals.

### 9. Simulation guards (Druedahl convention)

`sim_guard = 0.02` (on assets), `util_floor = -1e18` (effectively
disabled). Matches Druedahl's `euler_cutoff = 0.02`.

---

## Parameter audit (verified 6 Apr)

All settings reach the stage correctly for both FUES and NEGM:
- b=0.01, a_min=0.01, h_min=0.01, w_min=0.02
- grid_phi=1.14, fues_lb=4
- extrap_policy=0, correct_jumps=0, clamp_max_factor=1.1
- sim_guard=0.02, util_floor=-1e18, ce_burn_in=10
- Intersections: keeper=1, adj=1 (both on)
- No spurious fallbacks to wrong defaults

Simulation verified: `c_sim == c_grid` at all tested agents when
using correct w_adj (including income). The simulation faithfully
reproduces the solved policy.

---

## CE gap diagnosis

### Observed (300 grids, separable, tau=0.12)

| Metric | EGM(FUES) | NEGM(FUES) |
|--------|-----------|------------|
| CE utility | 17,063 | 17,068 |
| Mean disc. utility | -33.165 | -33.140 |
| Euler c (all) | -6.01 | -5.73 |
| Euler h (adj) | -4.18 | -2.86 |
| Adj rate | 14.8% | 15.1% |

### Quantile analysis of discounted utility

FUES wins at bottom 1% (+1.44 per agent). NEGM wins at every other
quantile by 0.01-0.12. The gap is systematic across the entire
distribution, not from outliers.

### Value function comparison

FUES V > NEGM V at 270/300 wealth grid points (averaged across ages/z).
NEGM wins at 30 low-wealth points (w < 0.16).

### Flow utility comparison

At the grid level: FUES u(c,h) < NEGM u(c,h) by 0.001-0.014 at
w=0.15-0.33. Despite V being higher. This means FUES has higher
continuation values but lower current utility — agents over-save.

### Tenure decision comparison

0.29% of grid states (105k/36M) have different adjust/keep decisions.
FUES adjusts at 67k states where NEGM keeps. FUES keeps at 38k states
where NEGM adjusts. But in simulation, NEGM adjusts MORE (15.1% vs
14.8%) because the FUES-keep/NEGM-adjust states are at positions
agents actually visit (a≈2.5, h≈6.2).

### Root cause hypothesis: upper envelope selection bias

The FUES upper envelope takes MAX over noisy EGM candidates. Each
candidate's V includes interpolation error from V_cntn. The MAX of
noisy estimates is biased upward (winner's curse / order statistics
bias). This inflates FUES continuation values, causing over-saving
and lower flow utility.

NEGM golden-section search can only underestimate the max (finite
tolerance), giving slight downward bias. Opposite directions:
- FUES V biased UP → over-save → lower u → lower CE
- NEGM V biased DOWN → under-save → higher u → higher CE

### Alternative hypothesis: constraint point discretisation

FUES constraint points are evaluated at discrete h_choice grid points.
NEGM optimises h continuously. At low wealth (w=0.15-0.25), the
nearest grid h is suboptimal, producing lower V and u.

---

## Files changed (day 2)

| File | Changes |
|------|---------|
| `model.py` | nonlinspace, a_min/h_min/w_min from settings, grid_phi |
| `adjuster_egm.py` | consav.interp_2d (NEGM), interp2d_nonuniform (EGM), h_hi fix, -inf guard, clamp_fac |
| `branching.py` | interp_2d, marginal recomputation from c, removed eval_linear/xto |
| `keeper_egm.py` | interp_2d for sim forward |
| `simulate.py` | interp_2d, terminal utility in NPV, removed eval_linear/xto |
| `math_funcs.py` | interp2d_nonuniform (proper bilinear), extrap slope cap |
| All settings.yaml | grid_phi, clamp_max_factor, sim_guard=0.02, util_floor=-1e18 |

---

## Next steps

1. Grid convergence test: does the CE gap shrink at 600/900 grids?
   If yes, it's interpolation error. If not, structural bias.
2. Test with fewer h_choice points to check the selection bias
   hypothesis (fewer candidates → less bias → should help FUES CE).
3. Consider bias correction for the FUES upper envelope.
4. Compare FUES and NEGM policies (c, h, a) at specific grid points
   to understand the over-saving pattern.
