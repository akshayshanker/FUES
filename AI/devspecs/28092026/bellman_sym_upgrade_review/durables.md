# Durables: preserving the numerical implementation

Reviewed 28 September 2026. This is a source review for the proposed upgrade,
not an implementation or a claim that numerical comparisons have passed.
Both separable and Cobb–Douglas registries are required. The running
estimation work must be integrated before its final interfaces become the
upgrade's baseline.

## Preserve and replace

Preserve the numerical bodies of `keeper_egm.py`, `adjuster_egm.py`,
`branching.py`, `conditioning.py`, grid construction, interpolation, root
search, upper-envelope selection, simulation panels, and estimation moments.
Preserve `solve(...) -> (nest, grids)`, solution dictionary names and array
ordering, registry selection, calibration overlays, numerical overrides,
method selection, and random draws.

The current construction depends directly on Dolo+: imports and method
normalisation in [solve.py](../../../../examples/durables/solve.py), lines
8–14, 68–156; recipe and period construction at 604–650; recalibration at
214–228. Replace these with Bellman-SYM construction and public queries.
The numerical schedule at 242–275 already explicitly executes keeper,
adjuster, tenure choice, then expectation. It need not become a generic
Bellman-SYM solver.

At the construction functions, replace reads of `stage.calibration`,
`stage.settings` and old method dictionaries with explicit resolved inputs.
Do not reproduce the Dolo stage class. One model-input extraction should
serve the solver, simulation and estimation, including registry-level names
that no individual stage declares. Current
[callables.py](../../../../examples/durables/syntax/separable/callables.py),
lines 110–129, reads every parameter from `keeper_cons`; the Bellman example
declares returns and income parameters on tenure and terminal parameters on
bequest, so reading one stage cannot recover the complete calibration.
`return_grids` is currently inserted into calibration by `solve.py:395–397`;
pass it as numerical configuration instead.

The old `upper_env` method target and private Dolo method parsers must retire.
The [new method brief](/Users/akshayshanker/Research/Repos/bellman-ddsl/AI/road-map/AI-todos/briefs-for-agents/internal-dev-0.01-release/inv-Euler-functional-operation.md),
§3 and acceptance item 3, places the endogenous-grid representation at policy
Evaluation and the adjuster's computing method at its joint solution.
FUES/NEGM selection must still select the same existing numerical routines.

## Keep the handwritten Numba functions

The [kernel brief](/Users/akshayshanker/Research/Repos/bellman-ddsl/AI/road-map/AI-todos/briefs-for-agents/internal-dev-0.01-release/kernels-as-functions.md),
§§1–5, expressly permits users to retain their own implementations. Its
compiled functions expose explicit block calculations; named utility
functions are nested inside those functions. It promises neither standalone
utility/marginal exports nor a compiled root solver, maximisation or
expectation. Durables consumes all of these separately, including the
housing-dependent inverse marginal utility of Cobb–Douglas.

Keep both existing `callables.py` modules and their dictionary keys and
signatures. Change parameter acquisition, then check their formulas against
the new stage files. Optional adoption of exported transition or value
kernels can follow verified pointwise equivalence; extracting nested
functions or constructing a new general expression compiler is unnecessary.
Handwritten numerical guards must remain explicit: the present functions
return large penalties outside positive consumption/housing domains, whereas
the new plain kernels do not enforce those domains.

If exported kernels are used, pass parameters in the record's declared
positional order. Compile and bind outside numerical loops. In-memory
kernels do not support Numba `cache=True`; persistent caching requires
writing and importing unchanged source, as §4 specifies. Existing closures
bake parameters once per solve, so retaining them does not promise that
estimation avoids recompilation at every candidate.

## Economic and numerical differences to preserve

The Bellman [durables stage files](/Users/akshayshanker/Research/Repos/bellman-ddsl/applications/durables/stages/adjuster_cons.bl)
are useful templates, not exact replacements for FUES.

| Item | Required treatment |
| --- | --- |
| Income state | FUES [model.py](../../../../examples/durables/model.py):71–80 stores Tauchen log-income nodes, floors transition probabilities at 0.001 and renormalises rows. Bellman `tenure.bl:14–19` uses a positive income level with a multiplicative transition. Retain FUES coordinates, transition matrix, wage polynomial, normalisation and retirement income; any translation between log and level belongs in an explicit adapter. |
| Discounting | FUES separable `callables.py:170–188` already includes β in continuation marginals; `keeper_egm.py:356–424` consumes them without another β. Bellman `tenure.bl:34–43` supplies undiscounted derivatives and `keeper_cons.bl:41` applies β. Preserve the existing array meaning and apply β exactly once. |
| Borrowing limit | Both registries default to `b=0.01`, distinct from grid lower bounds. `keeper_egm.py:323–339` constructs the constrained segment at saving b; `adjuster_egm.py:432–471` adds its constrained point and checks the Euler inequality. New clauses must state this actual bound, not copy the Bellman zero-saving example. |
| Cobb–Douglas | [CD callables](../../../../examples/durables/syntax/cobb_douglas/callables.py):27–106 define housing-dependent marginal utilities, their inverse, and a distinct terminal indirect utility. Add corresponding stage sources; do not substitute the separable bequest. The constrained housing condition uses Brent's root search, versus a closed form for separable utility. |
| Grids and ties | Preserve all grids, the feasible wealth floor, nonuniform spacing and probability flooring (`model.py:48–103`), array order `(z,a,h)`, clamping, extrapolation, and `adjust` on value ties (`branching.py:118–145`). These affect policies and simulated moments. |

Three existing inconsistencies need separate records rather than silent
correction during the upgrade. First, separable callable utility subtracts
`chi`, while its keeper value formula and stage YAML omit it
(`callables.py:24–30,229–236`; baseline `chi=0`). Second, `solve.py:443–455`
solves decisions at age T using terminal continuation, while
[simulate.py](../../../../examples/durables/solvers/simulate.py):501 ends
decisions at T−1 and 375–393 adds terminal utility directly. Third,
`simulate.py:75–86` uses carried housing in next-period marginal utility even
when the next branch adjusts; the supplied `H_adj` at line 153 is unused
there. That diagnostic matters for Cobb–Douglas. Numerical preservation and
economic corrections must be distinguishable in the acceptance record.

## Estimation and verification

The active [estimation brief](../cd_estimation_and_postprocess.md), §5.6,
now specifies equal-probability normal quantiles, configurable transformed
parameters, permanent type assignment and pooling in original agent order.
The separate worktree still contains intermediate beta-specific code and
uncommitted driver changes. Use its final approved and merged interfaces;
do not freeze today's transient `types` shape or stale introductory wording.
Retain independent random streams, per-type calibration, subgroup simulation,
pooled moments, male overlays and result manifests. Replace its additional
Dolo recipe/name-validation imports and `load_syntax` use with the same
resolved inputs used by `solve()`.

Acceptance should cover six comparisons: both registries' scalar functions
against authored formulas; grids and discounted derivatives; constrained
and interior raw candidate arrays; tiny FUES and NEGM solutions including
branch choices; identical seeded panels and type pooling; and one bounded
estimation evaluation including moments, parameter-name refusals and
manifest contents. Compare actual old/new arrays, not only broad statistics.
The existing [simulation test](../../../../tests/test_durables_simulate.py)
checks shapes and loose ranges; it is insufficient to establish migration
equivalence. Record the chosen treatment of the three existing
inconsistencies before setting comparison expectations.
