# Plan v2: Portfolio choice in the durables model → financial-guidance demo notebook

**Audience:** financial-services postgraduate (investor demo).
**Anchor:** Econ-ARK × T. Rowe Price "Life-Cycle Modeling is (Almost) Ready for Prime Time"
(Carroll et al.) — estimated structural lifecycle models as the engine for personalized
saving / portfolio / housing guidance, risk attitude first personalization axis.

**Gate-1 status:** reviewed by Matsya (session `port-housing-design`, defects D1–D11) +
three adversarial critics (DDSL-architecture, economics/numerics, investor-audience).
This v2 incorporates all confirmed findings. Verdicts: revise → addressed below.

---

## 1. Model brief (unchanged from v1 except naming + terminal fix)

- Extend FUES durables lifecycle (consumption + housing keep/adjust) with a risky-share
  choice `varsigma ∈ [0,1]` (NOT `alpha` — taken by utility weight; NOT `b` for the
  post-return balance — taken by settings lower bound; the poststate is `a_r`).
- States `(a, h, z)`; existing income shock `eps[>]` (tenure); NEW return shock
  `eta[>] ~ N(0,1)`, post-decision, in the new port stage.
- **Terminal value fix (Matsya D1 + arch critic, independently confirmed):** with the
  port stage owning returns, the calibration override `R: 1.0` must NOT reach the
  terminal condition (`w = R*a + ...` in callables.py would silently value terminal
  savings pre-return). Terminal keeps its own deterministic risk-free return
  `R_term = R_f` — the port-able ς̄-fixed special case. Also retire `r: 0.045 → 0.0`
  (derived duplicate of R).

## 2. DDSL design (final; see port.yaml artifact)

```
period:   port  →  tenure  →  { keeper_cons | adjuster_cons }  →  (a_nxt, h_nxt) → twister
          ς, η       j, eps              c  (+ h_choice)
backward waves:  [keeper, adjuster] → [tenure] → [port]
```

- Placement START of period (Matsya-canonical; unfolds the existing folded ς̄=0 return).
- Wiring `port → tenure: {a_r: a, h_p: h}`; `z` threads as solver slice (repo
  convention — tenure's `z_kp/z_adj` are likewise never wired).
- Port exports all three channels (`V[<]`, `d_aV[<]` by Danskin, `d_hV[<]`) — the
  upstream EGM/FUES chain is unbroken; FUES stays in the two cons stages; port itself
  needs no envelope (direct max over ς).
- **Honest diff (arch critic):** one new stage YAML + one `port_methods.yml` + one
  spec_factory entry + calibration override. The production dcsmm horse for the port
  stage is a separate work item — the demo does not claim the production solver runs
  the extended YAML today.
- New params: `R_f = 1.02`, `mu_r = ln(1.06) − sigma_r²/2`, `sigma_r = 0.18`
  (pending econ-critic calibration review).

## 3. Demo-twin numerics (self-contained in the notebook)

Mirrors the stage decomposition one function per stage with the same exports and
backward waves. **Econ-critic amendments (all adopted):**

- **Units fix (critical):** normalize demo income to mean working income = 1.0
  (production units at c≈0.5 make MU_c differ ×45 across γ types → housing story
  inverts). Ladder in normalized-income units; κ retuned by pilot with acceptance
  test: ownership (rung ≥ 1.0) at age 50 within 40–80% for ALL three types.
- **Corner-saturation fix (critical):** pilot the no-housing twin BEFORE building;
  acceptance: Growth ς < 1 by age 55 and ≥ 25pp Growth–Conservative spread at 65.
  Growth γ = 3.0 (not 2.5); pension replacement 30% of final working trend; if
  Growth still saturates, σ_r → 0.20 (never cut the premium).
- Income: Rouwenhorst **7** states (φ=0.82, σ=0.11). Return quadrature GH-7 with an
  11-node robustness row for the Conservative type.
- ς grid step **0.025** (41 pts); explicit tie-break ς=0 at a≈0; glide paths and
  corner diagnostics conditional on a > 0.05 (sim_guard convention).
- `a` grid 80 pts, a_min=1e-3, **a_max=30** (normalized units), denser near the
  constraint; verify < 1% of simulated agent-periods in top decile of grid.
- Ladder closure requires δ=0 (baseline is 0) — stated in the notebook.
- Diagnostics reframed: ς monotone in a **away from purchase thresholds**; the
  de-risking dip approaching a purchase is the **down-payment effect** — plotted
  and narrated as a feature of joint guidance.
- Do NOT equalize β·R across types; add ONE matched-wealth glide-path figure
  (policy at pooled median wealth path) beside the unconditional one; disclose
  Growth γ is chosen (not an LCPT estimate) and 4.65 was estimated jointly with a
  bequest motive.
- Port-stage kink honesty (for §2 as well): exports valid a.e.; ς-argmax jumps and
  GH-node boundary crossings create secondary kinks; FUES upstream prunes both
  true and quadrature-artifact kinks; more η nodes shrink artifacts.

## 4. Notebook (deliverable) — RESTRUCTURED per investor critic: hook → payoff → how

`examples/durables/notebooks/portfolio_advice/portfolio_housing_advice.ipynb`

1. **The pitch + money exhibit first.** One paragraph: households hold a house AND a
   portfolio; guidance that ignores the house mis-prices risk capacity. Then
   immediately: one guidance illustration (age 42, renter, Balanced profile) next to
   the glide-path chart with the **S&P Target-Date overlay (mandatory — data verified
   readable; embedded as a hardcoded table with source citation)**.
2. **Results in full:** glide paths by age × 3 risk profiles vs S&P benchmark;
   housing entry/upgrade by age × profile; wealth accumulation; a **value-of-guidance
   number** (certainty-equivalent gain of model-consistent vs benchmark glide path,
   in bp/yr and $ at retirement); **one validation chart** (model homeownership by
   age vs published data) for calibration credibility.
3. **Guidance illustrations** (NOT "advice" — regulatory reframe): function
   `illustrate_plan(age, wealth, housing, profile)`; positioned as a decision-support
   engine for licensed advisers; one explicit paragraph on the regulatory pathway.
4. **How it works** (plain language; jargon renames): "decision steps in the year"
   (stages), "the plan is computed by working backwards from retirement" (Bellman),
   "decision blueprint / audit map" (semantic graph — rendered figure from Matsya's
   node/edge tables), the one-file-per-decision-step point as unit economics of
   adding product features. Disclose: no mortgage/leverage yet (first roadmap item),
   gross of fees and taxes, cautious profiles also save more here (risk attitude and
   patience travel together in this model; separating them is a planned refinement).
5. **From demo to production:** the estimated-model credibility story (SMM pipeline
   on NCI Gadi exists today, LCPT estimates anchor the profiles), production solver
   (FUES upper envelopes) with a concrete solve-time number, compliance/auditability
   via the decision blueprint.
6. Summary + references (LCPT, Carroll 2006 EGM, Dobrescu–Shanker FUES,
   Iskhakov et al., S&P glide-path source).

Risk profiles anchored to LCPT structural estimates: Conservative γ≈8 (plain LCP),
Balanced γ≈4.65 (warm-glow bequest), Growth γ≈2.5–3; bridged in one sentence to
practice ("these correspond to the risk-tolerance bands advisers already use").

## 5. Waterfall gates

- **Gate 1 (plan): DONE** — Matsya D1–D11 + 3 critics; this v2.
- **Gate 2 (graphs): DONE** — Matsya produced stage-level (12-edge) and period-level
  (6-edge, 3 backward waves) node/edge tables; render with matplotlib in notebook.
- **Gate 3 (notebook):** build → execute → adversarial round (econ-tribe readability,
  numerics correctness, investor-fit re-check vs this plan) → fix → re-execute → deliver.

## 6. Confirmed-defect ledger (what reviews changed)

| Finding | Source | Resolution |
|---|---|---|
| Terminal value breaks under R:1.0 | Matsya D1 + arch critic | `R_term = R_f` terminal scope |
| `b` name taken (settings lower bound) | arch critic | poststate `a_r` |
| Minimal-diff undercount | arch critic | + methods YAML + spec_factory entry, stated honestly |
| `r` calibration duplicate | arch critic | retire `r → 0.0` with R |
| z threading unstated | Matsya D3 | solver-slice convention, documented |
| exogenous/states/space declaration hygiene | Matsya D5–D7 | port.yaml matches tenure.yaml conventions |
| Sell buried; advice overclaim; no benchmark; no validation | investor critic | §4 restructure above |
| Jargon | investor critic | rename table in §4.4 |
