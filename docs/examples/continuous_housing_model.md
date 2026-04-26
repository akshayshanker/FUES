# Application 2: Continuous Housing Investment with Frictions

*Source: Dobrescu and Shanker (2026), Section 2.2. The paper is the canonical reference for the economic model; this page records the notation and staged Bellman representation used in the codebase.*

## Notation

| Symbol | Meaning |
|--------|---------|
| $a_t$ | Financial (liquid) assets at time $t$ |
| $H_t$ | Housing (illiquid) assets at time $t$ |
| $c_t$ | Non-housing consumption at time $t$ |
| $y_t$ | Stochastic wage at time $t$ (Markov process) |
| $d_t \in \{0,1\}$ | Housing adjustment indicator ($1$ = adjust, $0$ = keep) |
| $r$ | Rate of return on financial assets |
| $\tau > 0$ | Proportional housing adjustment cost |
| $\beta$ | Discount factor |
| $x_t = (1+r)a_t + y_t + d_t H_t$ | Total wealth at time $t$ |
| $u : \mathbb{R}_+ \times \mathbb{R}_+ \to \mathbb{R} \cup \{-\infty\}$ | Per-period utility over consumption and housing |
| $\theta : \mathbb{R}_+ \to \mathbb{R} \cup \{-\infty\}$ | Bequest function |
| $V_t(a, H, y)$ | Value function |
| $\sigma_t^{a,d}$, $\sigma_t^{H,d}$ | Choice-specific policy functions for assets and housing |
| $\mathcal{I}_t(a, H, y)$ | Discrete choice policy function |
| $\Theta_t(a, H, y)$ | Continuation marginal value of housing if not adjusted |
| $\iota$ | Random stopping time: next period when $d_\iota = 1$ |

## 1. Model environment

Households hold non-negative financial assets $a_t$ and non-negative housing $H_t$. Financial assets earn a rate of return $r$; housing earns no return. Liquid assets adjust frictionlessly, while adjusting housing to a new level $H_{t+1}$ costs $\tau H_{t+1}$, $\tau > 0$. In period $t \in \{0, 1, \ldots, T\}$, the household consumes $c_t$, chooses $a_{t+1}$, and selects a discrete adjustment indicator $d_t \in \{0, 1\}$: when $d_t = 0$ housing is held fixed, $H_{t+1} = H_t$; when $d_t = 1$ a new $H_{t+1}$ is chosen subject to the adjustment cost. Income $y_t$ is exogenous and follows a Markov process.

The period budget constraint is

$$
(1+r)a_t + y_t + d_t H_t \geq a_{t+1} + c_t + d_t (1+\tau) H_{t+1},
$$

where $x_t = (1+r)a_t + y_t + d_t H_t$ is total available wealth. The household lives through $t = 0, \ldots, T$ and values terminal bequests via $\theta : \mathbb{R}_+ \to \mathbb{R} \cup \{-\infty\}$. Per-period utility $u(c_t, H_{t+1})$ is concave, jointly differentiable, and increasing in non-housing consumption and end-of-period housing (Yogo, 2016). The sequential problem is

$$
V_0(a_0, H_0, y_0)
= \max_{(c_t, a_{t+1}, H_{t+1}, d_t)_{t=0}^{T}}
\left\{
\sum_{t=0}^{T} \beta^t \mathbb{E}\, u(c_t, H_{t+1})
\;+\;
\mathbb{E}\, \beta^{T+1} \theta\bigl((1+r)a_{T+1} + H_{T+1}\bigr)
\right\},
$$

subject to the budget constraint, $H_t \geq 0$, $a_t \geq 0$, and given $(a_0, H_0, y_0)$. The terminal value is $V_{T+1}(a_{T+1}, H_{T+1}) = \theta\bigl((1+r)a_{T+1} + H_{T+1}\bigr)$, and for $t \leq T$ the Bellman equation is

$$
V_t(a, H, y) = \max_{a', H', d} \left\{ u(c, H') + \beta \mathbb{E}_y V_{t+1}(a', H', y') \right\},
$$

with primes denoting continuation states satisfying the budget constraint, $c = y + (1+r)a + dH - a' - d(1+\tau)H'$, and $H' = H$ if $d = 0$.

## 2. Euler equations

The weak inequalities below allow for the occasionally binding non-negativity constraint on liquid assets.

There is one Euler equation for each state. The financial-asset Euler equation is

$$
u_1(c_t, H_{t+1}) \geq \beta(1+r)\, \mathbb{E}_t\, u_1(c_{t+1}, H_{t+2}),
$$

with $u_1$ and $u_2$ denoting partial derivatives of $u$ with respect to its first and second arguments. When $d_t = 1$, the housing Euler equation is

$$
(1+\tau)\, u_1(c_t, H_{t+1}) \geq \underbrace{\mathbb{E}_t \sum_{k=t}^{\iota-1} \beta^{k-t}\, u_2(c_k, H_{k+1})}_{\text{marginal value of the housing-services stream}} + \underbrace{\mathbb{E}_t\, \beta^{\iota-t}\, u_1(c_\iota, H_{\iota+1})}_{\text{marginal value of liquidating housing at time } \iota}.
$$

The financial Euler equation is standard. The housing Euler equation features a random stopping time $\iota$, defined as the next period with $d_\iota = 1$. The shadow value of housing is the discounted expected marginal utility at the next liquidation date plus the intervening stream of housing-services utilities.

**Functional form.** The recursive solution implies measurable policy functions $\sigma_t^a$, $\sigma_t^H$, and $\mathcal{I}_t$, with branch-specific versions $\sigma_t^{a,d}$ and $\sigma_t^{H,d}$. On the adjust branch at time $t$,

$$
c = (1+r)a + H + y - \sigma_t^{a,1}(a, H, y) - (1+\tau)\sigma_t^{H,1}(a, H, y),
$$

with

$$
a' = \sigma_t^{a,1}(a, H, y), \qquad H' = \sigma_t^{H,1}(a, H, y).
$$

Let $d_{t+1} = \mathcal{I}_{t+1}(a', H', y')$. Next-period consumption is then determined by the same budget constraint on the realized branch:

$$
c' = (1+r)a' + d_{t+1} H' + y' - \sigma_{t+1}^{a,d_{t+1}}(a', H', y') - d_{t+1}(1+\tau)\sigma_{t+1}^{H,d_{t+1}}(a', H', y').
$$

Define

$$
H'' =
\begin{cases}
\sigma_{t+1}^{H,1}(a', H', y'), & d_{t+1} = 1, \\
H', & d_{t+1} = 0 .
\end{cases}
$$

Substituting the policies into the housing Euler equation yields

$$
(1+\tau)\, u_1(c, H') \geq u_2(c, H') + \beta \mathbb{E}_y \left\{ \mathcal{I}_{t+1}(a', H', y')\, u_1(c', H'') \right\} + \beta \mathbb{E}_y \left\{ (1 - \mathcal{I}_{t+1}(a', H', y')) \bigl[ \Theta_{t+1}(a', H', y') + u_2(c', H') \bigr] \right\}.
$$

The multiplier $\Theta_t$ collects the continuation marginal value of housing when the household does not adjust:

$$
\Theta_t(a, H, y) = \beta \mathbb{E}_y \left\{ \mathcal{I}_{t+1}(a', H', y')\, u_1(c', H'') \right\} + \beta \mathbb{E}_y \left\{ (1 - \mathcal{I}_{t+1}(a', H', y')) \bigl[ \Theta_{t+1}(a', H', y') + u_2(c', H') \bigr] \right\}.
$$

The functional financial Euler equation is

$$
u_1(c, H') \geq \beta(1+r)\, \mathbb{E}_y\, u_1(c', H''),
$$

with $c$, $c'$, and $H''$ defined as above.

## 3. Modular Bellman form

The formulation above is the standard one: one Bellman equation with a discrete choice embedded in the budget constraint. We now rewrite the same model in staged Bellman form. In the code, each period is split into linked subproblems with their own state spaces and Bellman operators.

> **Timing notation.** We use the multi-stage Bellman notation of [Carroll (2026)](https://llorracc.github.io/SolvingMicroDSOPs/) and [Carroll and Shanker (2026)](https://bright-forest.github.io/bellman-ddsl/theory/MDP-foundations/). Each stage has three information sets:
>
> - **Arrival** ($\prec$): the state on entering the stage.
> - **Decision** (unmarked): the information set on which the agent chooses.
> - **Continuation** ($\succ$): the outgoing state after the within-stage choice and transition.
>
> To keep the notation readable, when a consumption-stage formula is written only in $w$, $H$, or $m$, it should be read pointwise in a fixed income state $z$. In the implementation, the corresponding value and policy arrays still carry a full $z$ index.

A household lives for $T$ periods, holding financial assets $a \ge 0$ and housing $H \ge 0$. Each period it draws an i.i.d. innovation $\varepsilon_{\succ} \sim N(0,1)$, updates persistent income via $\log z_{\succ} = \rho_z \log z + \sigma_z \varepsilon_{\succ}$, earns income $\mathrm{y}(z_{\succ})$, and makes a discrete tenure choice $d \in \{0, 1\}$ — keep or adjust housing — followed by a non-durable consumption choice (plus a durable choice under the adjuster branch). Financial assets earn gross return $R = 1+r$; adjusting housing costs $\tau H_{\succ}$. The stage equations below keep $\delta$ (housing depreciation) and $R_H$ (housing gross return) as free parameters; the simplest case of the main text sets $\delta = 0$ and $R_H = 1$.

### Stage decomposition

Each period decomposes into three stages, solved in reverse. Ordered forward:

1. **`tenure`** (branching + $\mathbb{E}_{\varepsilon_{\succ}}$) — discrete choice $\max(\mathrm{v}_{\mathrm{kp}, \succ}, \mathrm{v}_{\mathrm{adj}, \succ})$; income innovation $\varepsilon_{\succ}$ draws; asset returns realised; durables liquidated on the adjuster branch.
   - Keeper continuation state: $w_{\mathrm{kp}} = Ra + \mathrm{y}(z_{\succ})$.
   - Adjuster continuation state: $w_{\mathrm{adj}} = Ra + \mathrm{y}(z_{\succ}) + R_H(1-\delta)H$.
2. **`keeper_cons`** (1D EGM + FUES) — consumption choice on cash-on-hand $w_{\mathrm{kp}}$; housing passes through.
3. **`adjuster_cons`** (2D partial EGM + FUES) — joint choice of consumption and new housing on total wealth $w_{\mathrm{adj}}$.

In `keeper_cons` and `adjuster_cons`, the Bellman equations below are written conditional on the realized income state $z_{\succ}$. Economically this is a one-dimensional EGM problem for each $(z, H)$ slice of the keeper problem and a one-dimensional wealth problem for each $z$ slice of the adjuster problem.

<details style="border-left:3px solid #1565c0;padding:8px 16px;margin:12px 0;background:rgba(21,101,192,0.04);border-radius:4px;">
<summary style="cursor:pointer;font-weight:600;font-size:0.95em;"><code>tenure</code> — discrete branching + income shock</summary>

| Perch | State variables | Description |
| :--- | :--- | :--- |
| Arrival $(\prec)$ | $a,\; H,\; z$ | Assets, housing, previous income realisation |
| Decision | $a,\; H,\; z$ | Identity transition; agent chooses branch |
| Continuation — **kp** $(\succ)$ | $w_{\mathrm{kp}},\; H_{\mathrm{kp}},\; z_{\succ}$ | Cash-on-hand, housing, new income |
| Continuation — **adj** $(\succ)$ | $w_{\mathrm{adj}},\; z_{\succ}$ | Total liquid wealth, new income |

**Arrival → decision (identity).** &ensp; $a = a_{\prec},\; H = H_{\prec},\; z = z_{\prec}$. The previous income realisation $z$ is known at decision; the innovation $\varepsilon_{\succ}$ has not yet resolved.

**Decision → continuation (two branches).** &ensp; The innovation $\varepsilon_{\succ} \sim N(0,1)$ resolves on the continuation edge, producing $z_{\succ} = z^{\rho_z}\exp(\sigma_z\,\varepsilon_{\succ})$. Conditional on $z_{\succ}$, each branch maps $(a, H, z_{\succ})$ to a distinct continuation space.

*Keep branch* — housing depreciates and is retained:

$$w_{\mathrm{kp}} = R\,a + \mathrm{y}(z_{\succ}), \qquad H_{\mathrm{kp}} = (1-\delta)\,H.$$

*Adjust branch* — housing is liquidated at resale value $R_H(1-\delta)H$ and folded into liquid wealth:

$$w_{\mathrm{adj}} = R\,a + R_H(1-\delta)\,H + \mathrm{y}(z_{\succ}).$$

**Bellman step $\mathbb{B}$** &ensp; Conditional on $z_{\succ}$, pick the better branch:

$$\mathrm{v}(a, H, z_{\succ}) = \max\bigl(\,\mathrm{v}_{\mathrm{kp},\succ}(w_{\mathrm{kp}}, H_{\mathrm{kp}}),\;\; \mathrm{v}_{\mathrm{adj},\succ}(w_{\mathrm{adj}})\,\bigr).$$

**Marginal-value propagation.** &ensp; Writing $\mathbb{1}_{\mathrm{adj}}$ for the branch indicator, the envelope conditions at decision are

$$\partial_a \mathrm{v} = \mathbb{1}_{\mathrm{adj}}\, R\, \partial_{w_{\mathrm{adj}}} \mathrm{v}_{\mathrm{adj},\succ} + (1 - \mathbb{1}_{\mathrm{adj}})\, R\, \partial_{w_{\mathrm{kp}}} \mathrm{v}_{\mathrm{kp},\succ},$$

$$\partial_H \mathrm{v} = \mathbb{1}_{\mathrm{adj}}\, R_H(1-\delta)\, \partial_{w_{\mathrm{adj}}} \mathrm{v}_{\mathrm{adj},\succ} + (1 - \mathbb{1}_{\mathrm{adj}})\, (1-\delta)\, \partial_{H_{\mathrm{kp}}} \mathrm{v}_{\mathrm{kp},\succ}.$$

**Expectation step $\mathbb{I}$** &ensp; Integrate over $\varepsilon_{\succ}$ with $z_{\succ} = z^{\rho_z}\exp(\sigma_z\,\varepsilon_{\succ})$:

$$\mathrm{v}_{\prec}(a, H, z) = \mathbb{E}_{\varepsilon_{\succ}}\!\bigl[\mathrm{v}(a, H, z_{\succ})\bigr], \qquad \partial_a \mathrm{v}_{\prec} = \mathbb{E}_{\varepsilon_{\succ}}\!\bigl[\partial_a \mathrm{v}\bigr], \qquad \partial_H \mathrm{v}_{\prec} = \mathbb{E}_{\varepsilon_{\succ}}\!\bigl[\partial_H \mathrm{v}\bigr].$$

</details>

<details style="border-left:3px solid #00897b;padding:8px 16px;margin:12px 0;background:rgba(0,137,123,0.04);border-radius:4px;">
<summary style="cursor:pointer;font-weight:600;font-size:0.95em;"><code>keeper_cons</code> — keeper consumption (1D EGM + FUES)</summary>

| Perch | Variables | Description |
| :--- | :--- | :--- |
| Arrival $(\prec)$ | $w_{\prec},\, H_{\prec}$ | Cash-on-hand, inherited (fixed) housing |
| Decision | $w,\, H$ | Identity transition; no within-stage shock before optimisation |
| Continuation $(\succ)$ | $a_{\succ},\, H_{\succ} = H$ | Post-savings assets; housing passes through |

**Bellman step $\mathbb{B}$** &ensp; The keeper optimises over consumption alone:

$$\mathrm{v}_{\mathrm{kp}}(w, H) = \max_{c \ge 0}\;\bigl\{\mathrm{u}(c, H) + \beta\,\mathrm{v}_{\succ}(w - c,\; H)\bigr\},$$

subject to $a_{\succ} = w - c \ge 0$ and $H_{\succ} = H$. The FOC is

$$
\mathrm{u}_1(c, H) = \beta\, \partial_a \mathrm{v}_{\succ}(a_{\succ}, H).
$$

*EGM step.* &ensp; Fix $a_{\succ}$ on the exogenous grid. The financial Euler inverts analytically:

$$
c_{\succ} = \mathrm{u}_1^{-1}\!\bigl(\beta\, \partial_a \mathrm{v}_{\succ}(a_{\succ}, H)\bigr), \qquad w_{\succ} = a_{\succ} + c_{\succ},
$$

which yields the endogenous cash-on-hand grid. FUES selects the upper envelope from the pooled constrained and unconstrained EGM points.

*Non-concavity.* &ensp; The continuation $\mathrm{v}_{\succ}(a_{\succ}, H)$ is next period's tenure-stage arrival value. The branching $\max$ introduces primary kinks at the keep/adjust switching threshold; future switching thresholds propagate backward as *secondary kinks* — the DC-EGM mechanism of [Iskhakov et al. (2017)](https://doi.org/10.3982/QE643). $\partial_a \mathrm{v}_{\succ}(\cdot, H)$ is non-monotone, and the EGM-constructed $w_{\succ}(a_{\succ})$ mapping has crossing segments.

**Return to arrival $\mathbb{I}$** &ensp; The arrival transition is identity ($w = w_{\prec}, H = H_{\prec}$). The envelope theorem gives

$$\partial_w \mathrm{v}_{\mathrm{kp}}(w, H) = \mathrm{u}_1(c^\star, H), \qquad \partial_H \mathrm{v}_{\mathrm{kp}}(w, H) = \mathrm{u}_2(c^\star, H) + \beta\, \partial_H \mathrm{v}_{\succ}(a_{\succ}, H).$$

The marginal $\partial_H \mathrm{v}_{\mathrm{kp}}$ — housing-services value plus the continuation shadow value — is the quantity $\Theta$ in §2's stopping-time representation.

</details>

<details style="border-left:3px solid #7c4dff;padding:8px 16px;margin:12px 0;background:rgba(124,77,255,0.04);border-radius:4px;">
<summary style="cursor:pointer;font-weight:600;font-size:0.95em;"><code>adjuster_cons</code> — adjuster consumption (2D partial EGM + FUES)</summary>

| Perch | Variables | Description |
| :--- | :--- | :--- |
| Arrival $(\prec)$ | $m_{\prec}$ | Total liquidated wealth from the tenure stage |
| Decision | $m$ | Same scalar total wealth |
| Continuation $(\succ)$ | $a_{\succ},\, H_{\succ}$ | New savings and new housing |

**Bellman step $\mathbb{B}$** &ensp; The adjuster jointly chooses consumption and new housing:

$$\mathrm{v}_{\mathrm{adj}}(m) = \max_{c,\, H_{\succ} \ge 0}\;\bigl\{\mathrm{u}(c, H_{\succ}) + \beta\,\mathrm{v}_{\succ}(m - c - (1+\tau)\,H_{\succ},\; H_{\succ})\bigr\},$$

subject to $a_{\succ} = m - c - (1+\tau)\, H_{\succ} \ge 0$. The two FOCs are

$$
\mathrm{u}_1(c, H_{\succ}) = \beta\, \partial_a \mathrm{v}_{\succ}, \qquad
(1+\tau)\, \mathrm{u}_1(c, H_{\succ}) = \mathrm{u}_2(c, H_{\succ}) + \beta\, \partial_H \mathrm{v}_{\succ}.
$$

The housing FOC equates the marginal cost $(1+\tau)\, \mathrm{u}_1$ to the marginal benefit $\mathrm{u}_2 + \beta\, \partial_H \mathrm{v}_{\succ}$.

*Partial EGM.* &ensp; Fix $H_{\succ}$ on an exogenous grid. For each $\hat{H}_i$: invert the financial Euler to obtain $c_{\succ}(a_{\succ}, \hat{H}_i)$, recover the endogenous wealth $m_{\succ} = a_{\succ} + c_{\succ} + (1+\tau)\, \hat{H}_i$, and evaluate the Bellman value $\hat{\mathrm{v}}$. Pool candidates across the full $(a_{\succ}, \hat{H}_i)$ grid and apply FUES to recover $(c^\star, H_{\succ}^\star)$ as functions of $m$.

*Multiple roots.* &ensp; The mapping $a_{\succ} \mapsto m_{\succ}$ is **non-monotone** for two compounding reasons: (i) secondary kinks from future tenure switches make $\partial_a \mathrm{v}_{\succ}(\cdot, H)$ non-monotone; (ii) housing is simultaneously a consumption good and an asset, and the resulting intertemporal income effects produce discontinuous drops in housing investment at tenure-switch thresholds. The housing policy $H_{\succ}^\star(m)$ exhibits upward and downward jumps. MSS and LTM require locally isolated monotone segments and cannot resolve this structure. FUES scans the full cloud in a single $O(n \log n)$ pass.

**Return to arrival $\mathbb{I}$** &ensp; The arrival transition is identity ($m = m_{\prec}$). The envelope theorem gives

$$\partial_m \mathrm{v}_{\mathrm{adj}}(m) = \mathrm{u}_1(c^\star, H_{\succ}^\star).$$

</details>
