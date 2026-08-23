---
hide:
  - toc
---

# FUES: Fast Upper Envelope Scan

!!! warning "Pre-release (v0.6.0dev8)"
    Under active development. The API and documentation may change.

Paper: Dobrescu, L.I. and Shanker, A. (2022, revised 2026). "A fast upper
envelope scan method for discrete–continuous dynamic programming." [SSRN
4181302](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4181302).

FUES recovers the upper envelope of the EGM ([Carroll
2006](https://doi.org/10.1016/j.econlet.2005.09.013)) value correspondence in
discrete–continuous dynamic programming problems. It scans the endogenous
grid in a single sub-linear pass, and identifies sub-optimal points as the
conjunction of a discontinuous jump in the continuation policy and a concave
right turn in the value correspondence. It imposes no monotonicity on the
optimal policy and requires no numerical optimisation. See [How FUES
works](algorithm/fues-algorithm.md) for the derivation.

To use `FUES` or `EGM_UE` in your own model, start with the
[Quickstart](start-here/quickstart.md) and a plain `pip install`. You can then
call `FUES` directly on your EGM output, or use the `EGM_UE` interface to run
the alternative upper-envelope methods on the same problem.
