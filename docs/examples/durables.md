# Durables Model

> **Notation: stages and perches.** We use the multi-stage Bellman notation from [Carroll (2026)](https://llorracc.github.io/SolvingMicroDSOPs/) and [Carroll and Shanker (2026)](https://bright-forest.github.io/bellman-ddsl/theory/MDP-foundations/). Each period is decomposed into modular **stages** — self-contained Bellman sub-problems. Within each stage, variables are organised by **perches**, which represent information sets:
>
> - **Arrival** (subscript $\prec$): the entering state of the stage. A variable indexed by $\prec$ is predetermined — its value is known before any shocks or decisions *within this stage* have been made. For instance, $a_{\prec}$ is the adjuster's financial assets on arrival.
>
> - **Decision** (subscript $\sim$ or unmarked): the information set at which the agent optimises. A variable with no perch index is decision-measurable — the agent can condition on it when choosing. For instance, $m$ is total wealth available to the adjuster. When disambiguation is needed, we write the subscript $\sim$ explicitly.
>
> - **Continuation** (subscript $\succ$): the outgoing state of the stage. A variable indexed by $\succ$ is determined only after all decisions and shocks within the stage have resolved. For instance, $a_{\succ}$ is the adjuster's financial assets after consumption, housing, and savings choices are made, and $\mathrm{v}_{\succ}$ is the continuation value.
>
> **When are perch indices needed?** Each variable has a **native perch** determined by its role (arrival states live at $\prec$, decision states and controls at $\sim$, post-decision states at $\succ$). If a variable name is unique to one role, the perch can be inferred and the index omitted — e.g. $k$ for arriving capital, $m$ for market resources at the decision point, and $a$ for end-of-stage assets need no subscripts. When the *same* algebraic symbol appears at different perches (as $H$ does in this notebook), perch indices are essential to distinguish $H_{\prec}$ (housing entering the stage) from $H$ (housing at the decision point) from $H_{\succ}$ (housing leaving the stage). When a variable passes through unchanged via an identity transition (e.g. $a = a_{\prec}$), the bare symbol is measurable at both perches simultaneously.


A household lives for $T$ periods, holding financial assets $a \ge 0$ and housing (or durables) $H \ge 0$. Each period it draws an IID income innovation $\varepsilon_{\succ} \sim N(0,1)$ and updates the persistent income state via $\log z_{\succ} = \rho_z \log z + \sigma_z \varepsilon_{\succ}$, earns income $\mathrm{y}(z_{\succ})$, and makes a discrete `tenure` choice $d \in \{0,1\}$ to keep or adjust their housing stock, followed by a non-durable consumption choice (and a durable choice in the case of the adjuster). Financial assets earn gross return $(1+r)$; adjusting the durables stock requires paying a proportional transaction cost $\tau H_{\succ}$.

In the simple exposition below, we assume house price is constant and there is no home depreciation. 

### Stage decomposition

Each period decomposes into three self-contained stages solved in reverse order, following [Carroll (2026)](https://llorracc.github.io/SolvingMicroDSOPs/) and [Carroll and Shanker (2026)](https://bright-forest.github.io/bellman-ddsl/theory/MDP-foundations/). Proceeding forward from the start of a period, we have:

1. **`tenure`** (branching + $\mathbb{E}_{\varepsilon_{\succ}}$) — discrete choice: $\max(\mathrm{v}_{\mathrm{kp},\succ},\, \mathrm{v}_{\mathrm{adj},\succ})$, then draw IID innovation $\varepsilon_{\succ}$, update income $z_{\succ}$, realize asset returns and liquidate durables on the adjuster branch.
	1. Continuation keeper: $w_{\mathrm{kp}} = Ra + \mathrm{y}(z_{\succ})$
	2. Continuation adjuster: $w_{\mathrm{adj}} = Ra + \mathrm{y}(z_{\succ}) + R_H(1-\delta)H$.
2. **`keeper_cons`** (1D EGM + FUES) — keeper consumption choice on cash-on-hand $w_{\mathrm{kp}}$, with housing passing through.
3. **`adjuster_cons`** (2D partial EGM + FUES) — adjuster joint choice of consumption and new housing on total wealth $w_{\mathrm{adj}}$.

Let's go through stage by stage. 

---

<details style="border-left:3px solid #1565c0;padding:8px 16px;margin:12px 0;background:rgba(21,101,192,0.04);border-radius:4px;">
<summary style="cursor:pointer;font-weight:600;font-size:0.95em;"><code>tenure</code> — discrete branching + income shock</summary>

| Perch | State variables | Description |
| :--- | :--- | :--- |
| Arrival $(\prec)$ | $a,\; H,\; z$ | Assets, housing, previous income realisation |
| Decision | $a,\; H,\; z$ | Identity transition; agent chooses branch |
| Continuation — **kp** $(\succ)$ | $w_{\mathrm{kp}},\; H_{\mathrm{kp}},\; z_{\succ}$ | Cash-on-hand, housing, new income |
| Continuation — **adj** $(\succ)$ | $w_{\mathrm{adj}},\; z_{\succ}$ | Total liquid wealth, new income |

**Arrival → decision (identity).** The transition is an identity: $a = a_{\prec},\; H = H_{\prec},\; z = z_{\prec}$. Since arrival and decision states coincide, bare symbols are measurable at both perches. The previous income realisation $z$ is known at decision; the IID innovation $\varepsilon_{\succ}$ has not yet been drawn.

**Decision → continuation (two branches).** The income innovation $\varepsilon_{\succ} \sim N(0,1)$ resolves at the continuation edge, producing the new income state $z_{\succ} = z^{\rho_z}\exp(\sigma_z\,\varepsilon_{\succ})$. Conditional on $z_{\succ}$, each branch maps $(a, H, z_{\succ})$ to a distinct continuation space.

*Keep branch* — housing depreciates and is retained:

$$w_{\mathrm{kp}} = R\,a + \mathrm{y}(z_{\succ}), \qquad H_{\mathrm{kp}} = (1-\delta)\,H.$$

*Adjust branch* — housing is liquidated at resale price $R_H(1-\delta)H$ and folded into liquid wealth:

$$w_{\mathrm{adj}} = R\,a + R_H\,(1-\delta)\,H + \mathrm{y}(z_{\succ}).$$

**Backward mover $\mathbb{B}$ (cntn → dcsn).** &ensp; Conditional on $z_{\succ}$, the agent picks the better branch:

$$\mathrm{v}(a,\, H,\, z_{\succ}) = \max\!\bigl(\,\mathrm{v}_{\mathrm{kp},\succ}(w_{\mathrm{kp}},\, H_{\mathrm{kp}}),\;\; \mathrm{v}_{\mathrm{adj},\succ}(w_{\mathrm{adj}})\,\bigr).$$

**Marginal-value propagation.** Write $\mathbb{1}_{\mathrm{adj}}$ for the branch indicator. The envelope conditions at the decision perch are:

$$\partial_a \mathrm{v} = \mathbb{1}_{\mathrm{adj}}\; R\;\partial_{w_{\mathrm{adj}}} \mathrm{v}_{\mathrm{adj},\succ} + (1-\mathbb{1}_{\mathrm{adj}})\; R\;\partial_{w_{\mathrm{kp}}} \mathrm{v}_{\mathrm{kp},\succ},$$

$$\partial_H \mathrm{v} = \mathbb{1}_{\mathrm{adj}}\; R_H(1-\delta)\;\partial_{w_{\mathrm{adj}}} \mathrm{v}_{\mathrm{adj},\succ} + (1-\mathbb{1}_{\mathrm{adj}})\;(1-\delta)\;\partial_{H_{\mathrm{kp}}} \mathrm{v}_{\mathrm{kp},\succ}.$$

**Arrival mover $\mathbb{I}$ (dcsn → arvl).** &ensp; Integrate over the IID innovation $\varepsilon_{\succ}$, with $z_{\succ} = z^{\rho_z}\exp(\sigma_z\,\varepsilon_{\succ})$:

$$\mathrm{v}_{\prec}(a,H,z) = \mathbb{E}_{\varepsilon_{\succ}}\!\bigl[\mathrm{v}(a,H,z_{\succ})\bigr], \qquad \partial_a \mathrm{v}_{\prec} = \mathbb{E}_{\varepsilon_{\succ}}\!\bigl[\partial_a \mathrm{v}\bigr], \qquad \partial_H \mathrm{v}_{\prec} = \mathbb{E}_{\varepsilon_{\succ}}\!\bigl[\partial_H \mathrm{v}\bigr].$$

</details>

<details style="border-left:3px solid #00897b;padding:8px 16px;margin:12px 0;background:rgba(0,137,123,0.04);border-radius:4px;">
<summary style="cursor:pointer;font-weight:600;font-size:0.95em;"><code>keeper_cons</code> — keeper consumption (1D EGM + FUES)</summary>

| Perch | Variables | Description |
| :--- | :--- | :--- |
| Arrival $(\prec)$ | $w_{\prec},\, H_{\prec}$ | Cash-on-hand, inherited (fixed) housing |
| Decision | $w,\, H$ | Same — no transition before optimisation |
| Continuation $(\succ)$ | $a_{\succ},\, H_{\succ} = H$ | Post-savings assets; housing passes through |

**Decision mover $\mathbb{B}$** &ensp; (continuation → decision)

The keeper optimises over consumption alone:

$$\mathrm{v}_{\mathrm{kp}}(w, H) = \max_{c \ge 0}\;\bigl\{\mathrm{u}(c, H) + \beta\,\mathrm{v}_{\succ}(w - c,\; H)\bigr\},$$

subject to $a_{\succ} = w - c \ge 0$. Housing passes through unchanged: $H_{\succ} = H$.

The FOC is $\mathrm{u}_1(c, H) = \partial_a \mathrm{v}_{\succ}(a_{\succ}, H)$.

*EGM step.* &ensp; Fix $a_{\succ}$ on the exogenous grid. The financial Euler inverts analytically:

$$c_{\succ} = \mathrm{u}_1^{-1}\!\bigl(\partial_a \mathrm{v}_{\succ}(a_{\succ}, H)\bigr), \qquad w_{\succ} = a_{\succ} + c_{\succ}.$$

This yields the endogenous cash-on-hand grid. FUES selects the upper envelope from the pooled constrained and unconstrained EGM points.

*Non-concavity.* &ensp; The continuation $\mathrm{v}_{\succ}(a_{\succ}, H)$ is next period's tenure-stage arrival value. The branching $\max$ introduces primary kinks at the keep/adjust switching threshold; future switching thresholds propagate backward as *secondary kinks* — the DC-EGM mechanism of [Iskhakov et al. (2017)](https://doi.org/10.3982/QE643). As a result $\partial_a \mathrm{v}_{\succ}(\cdot, H)$ is non-monotone and the EGM-constructed $w_{\succ}(a_{\succ})$ mapping has crossing segments.

**Arrival mover $\mathbb{I}$** &ensp; (decision → arrival)

The arrival transition is an identity ($w = w_{\prec},\; H = H_{\prec}$). The marginal values follow from the envelope theorem:

$$\partial_w \mathrm{v}_{\mathrm{kp}}(w, H) = \mathrm{u}_1(c^\star, H), \qquad \partial_H \mathrm{v}_{\mathrm{kp}}(w, H) = \mathrm{u}_2(c^\star, H) + \beta\,\partial_H \mathrm{v}_{\succ}(a_{\succ}, H).$$

The marginal $\partial_H \mathrm{v}_{\mathrm{kp}}$ — housing service value plus the continuation shadow value — is the quantity $\Theta$ in the paper's stopping-time representation.

</details>

<details style="border-left:3px solid #7c4dff;padding:8px 16px;margin:12px 0;background:rgba(124,77,255,0.04);border-radius:4px;">
<summary style="cursor:pointer;font-weight:600;font-size:0.95em;"><code>adjuster_cons</code> — adjuster consumption (2D partial EGM + FUES)</summary>

| Perch | Variables | Description |
| :--- | :--- | :--- |
| Arrival $(\prec)$ | $m_{\prec}$ | Total liquidated wealth from tenure stage |
| Decision | $m$ | Same scalar total wealth |
| Continuation $(\succ)$ | $a_{\succ},\, H_{\succ}$ | New savings and new housing |

**Decision mover $\mathbb{B}$** &ensp; (continuation → decision)

The adjuster jointly chooses consumption and new housing:

$$\mathrm{v}_{\mathrm{adj}}(m) = \max_{c,\, H_{\succ} \ge 0}\;\bigl\{\mathrm{u}(c, H_{\succ}) + \beta\,\mathrm{v}_{\succ}(m - c - (1+\tau)\,H_{\succ},\; H_{\succ})\bigr\},$$

subject to $a_{\succ} = m - c - (1+\tau)\,H_{\succ} \ge 0$. The two FOCs are:

$$\mathrm{u}_1(c, H_{\succ}) = \partial_a \mathrm{v}_{\succ}, \qquad (1+\tau)\,\mathrm{u}_1(c, H_{\succ}) = \partial_H \mathrm{v}_{\succ}.$$

The housing FOC equates the marginal cost $(1+\tau)\,\mathrm{u}_1$ to the marginal benefit $\mathrm{u}_2 + \beta\,\partial_H \mathrm{v}_{\succ}$.

*Partial EGM.* &ensp; Fix $H_{\succ}$ on an exogenous grid. For each $\hat{H}_i$: invert the financial Euler to get $c_{\succ}(a_{\succ}, \hat{H}_i)$, recover the endogenous wealth $m_{\succ} = a_{\succ} + c_{\succ} + (1+\tau)\,\hat{H}_i$, and evaluate the Bellman value $\hat{\mathrm{v}}$. Pool all candidates across the full $(a_{\succ}, \hat{H}_i)$ grid and apply FUES to recover the optimal $(c^\star, H_{\succ}^\star)$ as functions of $m$.

*Multiple roots.* &ensp; The mapping $a_{\succ} \mapsto m_{\succ}$ is **non-monotone** for two compounding reasons: (i) secondary kinks from future tenure switches make $\partial_a \mathrm{v}_{\succ}(\cdot, H)$ non-monotone; (ii) intertemporal income effects between $H_{\succ}$ and $a_{\succ}$ — housing is simultaneously a consumption good and an asset — produce discontinuous drops in housing investment at tenure-switch thresholds. The housing policy $H_{\succ}^\star(m)$ exhibits upward and downward jumps. MSS and LTM require locally isolated monotone segments and cannot resolve this structure. FUES scans the full cloud in a single $O(n \log n)$ pass.

**Arrival mover $\mathbb{I}$** &ensp; (decision → arrival)

The arrival transition is an identity ($m = m_{\prec}$). The envelope theorem gives:

$$\partial_m \mathrm{v}_{\mathrm{adj}}(m) = \mathrm{u}_1(c^\star, H_{\succ}^\star).$$

</details>