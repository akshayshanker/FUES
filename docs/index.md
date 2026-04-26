---
hide:
  - toc
---

# FUES: Fast Upper Envelope Scan

!!! warning "Pre-release (v0.6.0dev1)"
    Under active development. The API and documentation may change.

<div class="paper-card" markdown>
<p class="citation">Working paper · SSRN 4181302 · 2022, revised 2026</p>
<p class="title">A fast upper envelope scan method for discrete–continuous dynamic programming</p>
<p class="authors">Loretti I. Dobrescu &nbsp;·&nbsp; Akshay Shanker</p>

<p class="abstract">
We develop a general <em>fast upper envelope scan</em> (FUES) method for solving
stochastic dynamic programs with discrete–continuous choices. Existing
endogenous grid method (EGM) implementations rely on monotonicity of the optimal
policy — an assumption frequently violated in economic applications — which
forces researchers to impose restrictive structure or revert to slower
value-function iteration. FUES provides an upper-envelope operator that
accommodates arbitrary policy mappings while retaining the computational
advantages of EGM. Under standard conditions we prove that the method recovers
the correct upper envelope. On benchmark applications we document substantial
speed and accuracy gains relative to MSS, LTM, RFC, and NEGM.
</p>

<p class="keywords">
<strong>Keywords.</strong> Discrete–continuous choice · non-convex optimisation ·
Euler equation · endogenous grid method · stochastic dynamic programming.
&nbsp;<strong>JEL.</strong> C13, C63, D91.
</p>
</div>

FUES recovers the upper envelope of the EGM ([Carroll 2006](https://doi.org/10.1016/j.econlet.2005.09.013)) value correspondence in discrete–continuous problems. It scans the endogenous grid in a single $O(n^{1/2})$ pass, and identifies sub-optimal points as the conjunction of a discontinuous jump in the continuation policy and a concave right turn in the value correspondence. It imposes no monotonicity on the optimal policy and requires no numerical optimisation. See [How FUES works](algorithm/fues-algorithm.md) for the derivation.

## Documentation guide

- **Library use.** Start with [Installation](getting-started/installation.md), then the [Core API](api/fues.md).
- **Algorithm.** Read [How FUES works](algorithm/fues-algorithm.md), then the [transparent EGM / FUES notebook](notebooks/egm_fues_transparent.ipynb).
- **Benchmarks and replication.** Start with the [Quickstart](start-here/quickstart.md), then see [Applications](examples/index.md), [Running locally](running-locally.md), and [Running on PBS / Gadi](running-on-gadi.md).

## Minimal use

After your EGM step produces arrays for the raw endogenous-grid correspondence, pass them to `FUES`:

```python
from dcsmm.fues import FUES

x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, _ = FUES(
    x_dcsn_hat,   # raw endogenous decision grid (N,) — unsorted is fine
    v_hat,        # raw value correspondence (N,)
    kappa_hat,    # primary control, e.g. consumption (N,)
    x_cntn_hat,   # raw continuation / post-decision state (N,)
    del_x_cntn,   # auxiliary jump-detection series, e.g. d x_cntn / d x_dcsn
    m_bar=1.2,    # jump threshold (approx max marginal propensity to save)
    LB=4,         # look-back buffer for forward/backward scans
)
```

The returned arrays contain only the upper-envelope points. Convention: `*_hat` for raw correspondence, `*_ref` for refined upper-envelope objects.

**Setting `m_bar`.** Use the maximum marginal propensity to consume in your model. For log utility with $\beta R < 1$, values in the range $1.0$–$1.2$ work well. Setting `endog_mbar=True` lets FUES compute a grid-local threshold from `del_x_cntn`.

## Upper-envelope registry

To compare FUES against MSS, RFC, and LTM on the same problem, use the unified `EGM_UE` interface:

```python
from dcsmm.uenvelope import EGM_UE

refined, raw, interpolated = EGM_UE(
    x_dcsn_hat=x_dcsn_hat,
    v_hat=v_hat,
    v_cntn_hat=v_cntn,
    kappa_hat=kappa_hat,
    x_cntn_hat=x_cntn_hat,
    X_dcsn=X_dcsn,
    uc_func_partial=uc_func,    # u'(c); only used by "CONSAV"
    u_func=u_func,
    method_switch="FUES",         # or "DCEGM", "RFC", "CONSAV"
    m_bar=1.2,
)
```

All methods return the same dict schema. `DCEGM` (MSS in the paper) and `CONSAV` (LTM in the paper) require a strictly monotone optimal policy; `FUES` and `RFC` do not. See the [durables application](examples/continuous_housing_model.md) for the main non-monotone benchmark.

See the [Core API](api/fues.md) for full parameter documentation.
