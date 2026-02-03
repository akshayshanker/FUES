# CLAUDE.md - Project Instructions for Claude Code

## CRITICAL: File System Warning

**DO NOT SAVE ANYTHING TO HOME DRIVE!**

This project runs on NCI Gadi HPC cluster. The home drive has very limited quota.
Saving files to the home drive will overload it and DELETE user files.

All outputs MUST go to `/scratch/tp66/{user}/` only.

## Project Structure

- `src/dc_smm/fues/` - FUES algorithm implementation
- `examples/durable-cons/` - Durable consumption model example
- `experiments/durables-cons/` - Experiment scripts for HPC

## Running on Gadi

Scripts are submitted via PBS and run on compute nodes. Files are synced from local to Gadi.
You CANNOT run tests locally - the code runs on the HPC cluster.

## Key Files

- `post_decision.py` - Computes post-decision value functions w, q, and q_d
- `fues_utils.py` - FUES upper envelope algorithm
- `egm.py` - EGM solver using FUES
- `negm.py` - Nested EGM solver (reference implementation)

## Optimization Notes

### lambda_d_keep optimization (Dec 2025)

For the EGM method, the shadow value of durables `q_d` requires computing:
- Keepers: `λ_d = u'_d(c, n) + q_d(p, n, a)`
- Adjusters: `λ_d = u'_c(c, d) * (1-τ)`

**Before**: In `post_decision.compute_wq`, for each shock (Nshocks iterations):
- Interpolate `c_keep_plus` at (p', n', m')
- Compute `a' = m' - c'` with clamping
- Interpolate `q_d_plus` at (p', n', a')
- Evaluate `marg_u_d_plus`

**After**:
- Pre-compute `lambda_d_keep = u'_d(c, n) + q_d(p, n, a)` in `negm.solve_keep`
- In `post_decision.compute_wq`, just interpolate `lambda_d_keep[t+1]` once per shock

**Result**: w computation reduced from ~4.5s to ~2.5s per period (grid 150x150x300).

## Coding Rules

### 1D Interpolation

For 1D interpolation, always use functions from `dc_smm.fues.helpers.math_funcs`:
- `interp_as(xp, yp, x)` - array version, `x` is a 1D array
- `interp_as_scalar(xp, yp, x)` - scalar version, `x` is a float

Do NOT use `interp` from `interpolation` or `np.interp` directly. Any optimizations to 1D interpolation should be done in these functions.

