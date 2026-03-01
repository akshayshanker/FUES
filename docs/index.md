# FUES: Fast Upper-Envelope Scan

!!! warning "Under development — do not cite"
    This documentation and codebase are under active development. Please do not cite until a stable release is published.

**A general method for computing the upper envelope of EGM value correspondences in discrete-continuous dynamic programming.**

Dobrescu and Shanker (2025), "A fast upper envelope scan method for discrete-continuous dynamic programming."

## The problem

The endogenous grid method (EGM, Carroll 2006) solves dynamic programming problems by analytically inverting the Euler equation, avoiding costly numerical optimisation. However, when a model features both discrete and continuous choices, the value function is the upper envelope of multiple concave functions — one for each future sequence of discrete choices. The EGM step produces a *value correspondence* that contains sub-optimal points. These must be removed to recover the true value function.

Existing methods for computing the upper envelope either require monotonicity of the policy function (DC-EGM, Iskhakov et al. 2017; Fella 2014) or rely on numerical optimisation combined with EGM (NEGM, Druedahl 2021). FUES removes both requirements.

## How FUES works

FUES exploits a geometric property of the EGM output: when the endogenous grid points and their associated values are sorted in ascending order, **sub-optimal points create concave right turns** in the value correspondence, while **optimal points create convex left turns**.

<!-- Interactive FUES scan animation (temporarily disabled) -->

The algorithm scans the sorted endogenous grid in a single pass:

1. Compute the endogenous grid, value correspondence, and policy via standard EGM
2. Sort all sequences by the endogenous grid points in ascending order
3. Scan from left to right, computing local secant slopes \(g_i\) and \(g_{i+1}\)
4. At each point, check if the policy "jumps" (exceeds threshold \(\bar{M}\)) **and** a concave right turn is made (\(g_{i+1} < g_i\))
5. If both conditions hold, the point is sub-optimal — remove it
6. If a convex left turn occurs at a jump (\(g_{i+1} > g_i\)), the point lies after a crossing of two choice-specific value functions — retain it and (optionally) interpolate the crossing point

The jump detection threshold \(\bar{M}\) arises naturally from economics: within a single choice-specific value function, the policy gradient is bounded by the maximum marginal propensity to save. Jumps exceeding this bound signal a switch between different future discrete choice sequences.

FUES also incorporates forward and backward scans around crossing points to improve accuracy when the grid is coarse. See [How FUES Works](algorithm/fues-algorithm.md) for the full algorithm description.

### Complexity

| Method | Time | Monotone policy? | Gradient info? |
|--------|------|-----------------|----------------|
| **FUES** | \(O(N)\) | No | No |
| DC-EGM | \(O(N \log N)\) | Yes | No |
| RFC | \(O(Nk)\) | No | Yes |
| NEGM | \(O(N \cdot \text{opt})\) | N/A | N/A |

## Installation

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
pip install -e ".[examples]"
```

Requires Python 3.11+ with NumPy, Numba, and SciPy. The `[examples]` extra adds matplotlib, seaborn, pyyaml for running the benchmark examples.

## Quick start

### Standalone FUES

```python
from dcsmm.fues import FUES

# After standard EGM produces:
#   e_grid: endogenous grid points (unsorted)
#   vlu:    value correspondence at each grid point
#   c_hat:  consumption policy at each grid point
#   a_hat:  asset policy (next-period assets) at each grid point
#   del_a:  policy gradient (da'/de)

e_clean, v_clean, c_clean, a_clean, d_clean = FUES(
    e_grid, vlu, c_hat, a_hat, del_a,
    m_bar=1.2,   # jump detection threshold
    LB=4,        # look-back buffer length
)

# e_clean, v_clean, c_clean, a_clean are the refined arrays
# with sub-optimal points removed
```

**Parameters:**
- `e_grid` — endogenous decision grid (N,)
- `vlu` — value at each grid point (N,)
- `c_hat` — primary policy, e.g. consumption (N,)
- `a_hat` — secondary policy, e.g. next-period assets (N,)
- `del_a` — policy gradient series for jump classification (N,)
- `m_bar` — jump detection threshold (default 1.0). Set to the maximum marginal propensity to consume, or slightly above
- `LB` — look-back/forward buffer length (default 4)

**Returns:** `(e_kept, v_kept, p1_kept, p2_kept, d_kept)` — the retained grid points and their associated values, policies, and gradients.

### Upper envelope registry (EGM_UE)

For comparing multiple upper envelope methods on the same problem:

```python
from dcsmm.uenvelope import EGM_UE

refined, raw, interpolated = EGM_UE(
    x_dcsn_hat=e_grid,       # endogenous grid
    qf_hat=vlu,              # value correspondence
    v_cntn_hat=v_continuation, # continuation value (for CONSAV)
    kappa_hat=c_hat,          # consumption policy
    X_cntn=a_hat,             # asset policy
    X_dcsn=eval_grid,         # target evaluation grid
    uc_func_partial=du,       # marginal utility function
    u_func={"func": u, "args": {}},  # utility (for CONSAV)
    ue_method="FUES",         # "FUES", "DCEGM", "RFC", or "CONSAV"
    m_bar=1.2,
)
```

Available methods:
- `"FUES"` — Fast Upper-Envelope Scan (Dobrescu and Shanker, 2025)
- `"DCEGM"` — DC-EGM (Iskhakov et al., 2017), via [HARK](https://github.com/econ-ark/HARK)
- `"RFC"` — Rooftop-Cut (Dobrescu and Shanker, 2024), requires `pykdtree`
- `"CONSAV"` — G2EGM upper envelope (Druedahl, 2021), via [ConSav](https://github.com/NumEconCopenhagen/ConsumptionSaving)

## Benchmark results

### Retirement choice model (Ishkakov et al. 2017)

Finite-horizon model with discrete retirement choice and continuous consumption-savings. Parameters: T=50, beta=0.96, r=0.02, y=20, log utility.

**Upper envelope computation time (ms per period):**

| Grid |  delta | RFC | FUES | DC-EGM |
|------|--------|-----|------|--------|
|  500 |   0.25 | 1.2 |  0.1 |    0.4 |
|  500 |   1.00 | 1.4 |  0.1 |    0.7 |
| 1000 |   0.25 | 2.9 |  0.2 |    0.6 |
| 1000 |   1.00 | 3.1 |  0.2 |    2.2 |
| 2000 |   0.25 | 5.6 |  0.4 |    1.3 |
| 2000 |   1.00 | 7.4 |  0.4 |    4.8 |
| 3000 |   0.25 | 8.4 |  0.6 |    2.0 |
| 3000 |   1.00 | 12.8 | 0.6 |    6.6 |

FUES is 5-20x faster than DC-EGM and RFC across all grid sizes and parameter values. Euler equation errors are comparable across methods.

**Key observations:**
- FUES timing is stable across delta values; DC-EGM degrades as the endogenous grid becomes more irregular
- FUES does not require monotonicity of the policy function
- With taste shocks (logit smoothing), the endogenous grid becomes non-monotone; DC-EGM slows by an order of magnitude while FUES remains stable

### Continuous housing investment (Dobrescu et al. 2024)

Two-asset model with liquid financial assets and illiquid housing. The discrete choice of whether to adjust housing creates a non-monotone endogenous grid where DC-EGM cannot be applied. FUES and RFC are the only applicable upper envelope methods. FUES is faster than RFC with comparable accuracy.

## Running the examples

### Retirement model

```bash
# Activate environment
source setup/load_env.sh   # on Gadi
# or: source .venv/bin/activate  # locally

# Single run with baseline parameters
python examples/retirement/run_experiment.py \
    --params params/baseline.yml --grid-size 3000

# Full timing sweep (all methods, grid sizes, delta values)
python examples/retirement/run_experiment.py \
    --params params/baseline.yml --run-timings
```

Output: timing and accuracy tables (LaTeX + Markdown) and EGM grid plots in the output directory.

## Package structure

```
src/dcsmm/
  fues/
    fues.py         # FUES algorithm (Numba JIT-compiled)
    fues_v0dev.py   # Original paper version
    dcegm.py        # DC-EGM wrapper (via HARK)
    rfc_simple.py   # RFC wrapper (via pykdtree)
    helpers/
      math_funcs.py # 1D interpolation, jump correction
  uenvelope/
    upperenvelope.py  # Unified UE registry (EGM_UE entry point)
```

## References

- Dobrescu, L.I. and Shanker, A. (2025). "A fast upper envelope scan method for discrete-continuous dynamic programming."
- Carroll, C.D. (2006). "The method of endogenous gridpoints for solving dynamic stochastic optimization problems." *Economics Letters*, 91(3), 312-320.
- Iskhakov, F., Jorgensen, T.H., Rust, J., and Schjerning, B. (2017). "The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks." *Quantitative Economics*, 8(2), 317-365.
- Druedahl, J. (2021). "A guide on solving non-convex consumption-saving models." *Computational Economics*, 58, 747-775.
- Fella, G. (2014). "A generalized endogenous grid method for non-smooth and non-concave problems." *Review of Economic Dynamics*, 17(2), 329-344.
