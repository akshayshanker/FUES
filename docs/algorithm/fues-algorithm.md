# How FUES Works

## The problem in one picture

In discrete-continuous models, EGM produces candidate solution points on a non-uniform endogenous grid. Some are optimal (on the upper envelope), others are sub-optimal (below it). FUES identifies and removes the sub-optimal points in a single linear scan.

<div class="diagram-container">
  <img src="../images/fues-scan.svg" alt="FUES scan diagram" />
</div>

---

## Why EGM produces sub-optimal points

!!! abstract "Setup"
    Consider a standard Bellman equation with a discrete choice \(d \in \{0, 1\}\) and a continuous choice \(c\):

    \[
    V_t(a) = \max_{c,\, d} \left\{ u(c) + \beta V_{t+1}^d(a') \right\}
    \]

    The continuation value \(V_{t+1}^d\) depends on the future discrete choice sequence. Each sequence yields a **concave** value function. The true \(V_t\) is the **upper envelope** of these concave functions.

When we invert the Euler equation via EGM, we get a set of points \((\hat{x}_i, \hat{v}_i)\) — but some of these points lie on *inferior* choice-specific value functions. They satisfy the first-order conditions but are not globally optimal.

!!! tip "The key insight"
    Sub-optimal points create **concave right turns** in the sorted value correspondence. Optimal crossing points create **convex left turns**. FUES exploits this geometry.

---

## The scan

Sort the EGM output by the endogenous grid \(\hat{x}_i\) in ascending order. Walk through the points computing secant slopes between consecutive triples:

\[
g_i = \frac{\hat{v}_i - \hat{v}_{i-1}}{\hat{x}_i - \hat{x}_{i-1}}, \qquad
g_{i+1} = \frac{\hat{v}_{i+1} - \hat{v}_i}{\hat{x}_{i+1} - \hat{x}_i}
\]

=== "Right turn: remove"

    If \(g_{i+1} < g_i\) (concave) **and** the policy jumps by more than \(\bar{M}\):

    - The point lies on a sub-optimal choice-specific value function
    - **Remove it** from the grid

=== "Left turn: keep"

    If \(g_{i+1} > g_i\) (convex) at a policy jump:

    - The point lies after a crossing of two value functions
    - **Keep it** and optionally interpolate the crossing location

---

## Jump detection

!!! info "What is a jump?"
    A "jump" occurs when the policy gradient between adjacent points exceeds a threshold \(\bar{M}\):

    \[
    \left| \frac{\hat{x}'_{i+1} - \hat{x}'_i}{\hat{x}_{i+1} - \hat{x}_i} \right| > \bar{M}
    \]

    Within a single choice-specific value function, the policy is smooth and its gradient is bounded. A jump signals a transition between different discrete choice sequences.

**Setting \(\bar{M}\):** Use the maximum marginal propensity to save in your model. For standard consumption-savings problems, values between 1.0 and 2.0 work well. Results are not sensitive to the exact choice.

Alternatively, set `endog_mbar=True` to compute \(\bar{M}\) endogenously at each grid point.

---

## The algorithm

!!! example "FUES (Box 1, Dobrescu and Shanker 2025)"

    1. Compute \(\hat{\mathbb{X}}_t\), \(\hat{\mathbb{V}}_t\), \(\hat{\mathbb{X}}'_t\) using standard EGM
    2. Set jump detection threshold \(\bar{M}\)
    3. Sort all sequences by the endogenous grid \(\hat{\mathbb{X}}_t\)
    4. Start from point \(i = 2\). Compute secant slopes \(g_i\) and \(g_{i+1}\)
    5. If \(|\text{policy jump}| > \bar{M}\) **and** \(g_{i+1} < g_i\) (right turn):
        - Remove point \(i+1\) from all grids
    6. Otherwise: advance \(i = i + 1\)
    7. Repeat until all points scanned

### Forward and backward scans

Near crossing points, the basic left/right turn test can be imprecise. FUES refines the classification using a look-back buffer of size `LB` (default 4):

- **Forward scan**: after a tentative removal, check that the next retained point is truly on a different (dominant) branch
- **Backward scan**: before removing a point at a right turn, verify that the preceding point is not itself sub-optimal

These scans handle edge cases where two value functions cross very close to a grid point.

---

## Complexity comparison

| Method | Time | Monotone policy? | Gradient info? | Note |
|--------|------|:---:|:---:|------|
| **FUES** | \(O(N)\) | -- | -- | Single scan, fixed look-back |
| DC-EGM | \(O(N \log N)\) | Required | -- | Segment detection + interpolation |
| RFC | \(O(Nk)\) | -- | Required | Nearest-neighbour search |
| NEGM | \(O(N \cdot \text{opt})\) | N/A | N/A | Numerical optimisation per point |

!!! success "FUES advantages"
    - **No monotonicity assumption** on the policy function
    - **No gradient information** required (unlike RFC)
    - **Linear time** with a small constant
    - **Simple to implement** — a single scan loop
    - **Compatible with Numba, JAX, and GPU** vectorisation

---

## References

- Dobrescu, L.I. and Shanker, A. (2025). "A fast upper envelope scan method for discrete-continuous dynamic programming."
- Carroll, C.D. (2006). "The method of endogenous gridpoints for solving dynamic stochastic optimization problems." *Economics Letters*, 91(3).
- Iskhakov, F. et al. (2017). "The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks." *Quantitative Economics*, 8(2).
- Druedahl, J. (2021). "A guide on solving non-convex consumption-saving models." *Computational Economics*, 58.
- Fella, G. (2014). "A generalized endogenous grid method for non-smooth and non-concave problems." *Review of Economic Dynamics*, 17(2).
