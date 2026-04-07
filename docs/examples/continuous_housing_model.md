# Application 2: Continuous Housing Investment with Frictions

*Source: Dobrescu and Shanker (2026), Section 2.2. Related to Kaplan and Violante (2014), Yogo (2016).*

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

## Model Environment

Let (non-negative) financial assets be denoted by $a_t$, and (non-negative) housing assets be denoted by $H_t$. Consider financial assets earn a rate of return $r$, and assume for simplicity that housing earns no returns. Investments can be made in and out of the stock of $a_t$ without friction, while adjusting housing to a value $H_{t+1}$ requires a payment of $\tau H_{t+1}$, with $\tau > 0$. In each period $t$, with $t = t_0, 1, \ldots, T$, the agent consumes non-housing goods $c_t$, and invests a total of $H_{t+1}$ and $a_{t+1}$ in housing and financial assets, respectively. The agent also makes a discrete choice $d_t$, where investment in and out of housing assets can only be made if $d_t = 1$; otherwise if $d_t = 0$ then $H_{t+1} = H_t$. Finally, in each period, the agent earns a stochastic wage $y_t$ and we assume $(y_t)_{t=0}^T$ is a Markov process.

The following constraints will hold for each $t$: first, total investments and consumption cannot exceed total available wealth each period

$$
(1+r)a_t + y_t + d_t H_t \geq a_{t+1} + c_t + d_t(1+\tau)H_{t+1},
$$

where total wealth at time $t$ is denoted by $x_t = (1+r)a_t + y_t + d_t H_t$.

Second, in terms of payoffs, the agent lives up to time $T$, after which they die and value the bequest they leave behind according to a function $\theta : \mathbb{R}_+ \to \mathbb{R} \cup \{-\infty\}$. Per-period utility is given by a real-valued function $u : \mathbb{R}_+ \times \mathbb{R}_+ \to \mathbb{R} \cup \{-\infty\}$ as $u(c_t, H_{t+1})$, where $u$ is a concave, jointly differentiable, increasing function of non-housing consumption $c_t$ and end-of-period housing $H_{t+1}$ (Yogo, 2016). Formally, the agent's dynamic optimization problem becomes

$$
V_0(a_0, H_0, y_0) = \max_{(a_t, H_t, d_t)_{t=0}^{T+1}} \left\{ \sum_{t=0}^{T} \beta^t \mathbb{E}\, u(c_t, H_{t+1}) + \mathbb{E}\, \beta^{T+1} \theta\bigl((1+r)a_{T+1} + H_{T+1}\bigr) \right\},
$$

such that the budget constraint holds, $H_t \geq 0$ and $a_t \geq 0$ for each $t$, $a_0$, $H_0$ and $y_0$ are given, and the expectation is taken over the wage process $(y_t)_{t=0}^T$.

The $T+1$ value function is given by the bequest value $V_{T+1}(a_{T+1}, H_{T+1}) = \theta((1+r)a_{T+1} + H_{T+1})$. For each $t \leq T$, the sequential problem implies the recursive Bellman equation

$$
V_t(a, H, y) = \max_{a', H', d} \left\{ u(c, H') + \beta \mathbb{E}_y V_{t+1}(a', H', y') \right\},
$$

where the prime notation indicates continuation state values satisfying the budget constraint, $c = y_t + (1+r)a + dH - a' - d(1+\tau)H'$, $H' = H$ if $d = 0$, and expectations are now conditional on the realization $y$ of the time $t$ wage $y_t$.

## Euler Equations

The problem will feature two Euler equations, one for each state. For periods prior to the terminal one, the Euler equation for the financial assets is

$$
u_1(c_t, H_{t+1}) \geq \beta(1+r) \mathbb{E}_t\, u_1(c_{t+1}, H_{t+2}),
$$

where we have used the subscript "1" to refer to the first partial derivative of $u$, and will use the subscript "2" to refer to the second partial derivative of $u$. If $d_t = 1$, the Euler equation for housing is

$$
(1+\tau)\, u_1(c_t, H_{t+1}) \geq \underbrace{\mathbb{E}_t \sum_{k=t}^{\iota-1} \beta^{k-t} u_2(c_k, H_{k+1})}_{\text{Marginal value of the housing services stream}} + \underbrace{\mathbb{E}_t\, \beta^{\iota-t} u_1(c_\iota, H_{\iota+1})}_{\text{Marginal value of liquidating housing at time } \iota}.
$$

The intuition of the Euler equation for financial assets is standard. The Euler equation for housing, however, features a stochastic time subscript $\iota$, defined as the next period when $d_\iota = 1$. Since the next time housing is adjusted will be stochastic, $\iota$ becomes a random stopping time. The Euler equation for housing then tells us that the shadow value (price) of investment (or withdrawal) from the stock of housing assets is given by the discounted expected value of housing *when housing is next liquidated*, along with the stream of housing services provided up to the time of liquidation.

**Functional Euler equations.** Since the solution sequence for the problem is recursive, there exist measurable functions $\sigma_t^a$, $\sigma_t^H$ and $\mathcal{I}_t$ such that $H_{t+1} = \sigma_t^H(a_t, H_t, y_t)$, $a_{t+1} = \sigma_t^a(a_t, H_t, y_t)$ and $d_t = \mathcal{I}_t(a_t, H_t, y_t)$ for each $t$. Let $\sigma_t^{a,d}$ and $\sigma_t^{H,d}$ be the choice-specific policy functions conditional on the time $t$ discrete choice $d \in \{0,1\}$. Inserting the policy functions back into the Euler equations yields the following functional Euler equation for housing:

$$
u_1(c, H')(1+\tau) \geq u_2(c, H') + \beta \mathbb{E}_y \left\{ \mathcal{I}_{t+1}(a', H', y')\, u_1(c', H'') \right\} + \beta \mathbb{E}_y \left\{ (1 - \mathcal{I}_{t+1}(a', H', y')) \left[ \Theta_{t+1}(a', H', y') + u_2(c', H') \right] \right\},
$$

where

$$
c' = (1+r)\sigma_t^a(a, H, y) + \sigma_t^H(a, H, y) + y' - \sigma_{t+1}^a(a', H', y') - \sigma_{t+1}^H(a', H', y')(1 + d_{t+1}\tau),
$$

$$
c = (1+r)a + H + y - \sigma_t^{a,1}(a, H, y) - \sigma_t^{H,1}(a, H, y)(1+\tau),
$$

$$
H'' = \sigma_{t+1}^{H,1}(a', H', y'),
$$

$a' = \sigma_t^{a,1}(a, H, y)$, $H' = \sigma_t^{H,1}(a, H, y)$, and $\Theta_{t+1}$ is a multiplier denoting the continuation marginal value of housing if time $t+1$ housing stock is not adjusted. For a set of time $t+1$ recursive policy functions, we can compute $\Theta_t$ as a function of the states as follows:

$$
\Theta_t(a, H, y) = \beta \mathbb{E}_y \left\{ \mathcal{I}_{t+1}(a', H', y')\, u_1(c', H'') \right\} + \beta \mathbb{E}_y \left\{ (1 - \mathcal{I}_{t+1}(a', H', y')) \left[ \Theta_{t+1}(a', H', y') + u_2(c', H') \right] \right\}.
$$

The functional Euler equation for the financial assets then becomes

$$
u_1(c, H') \geq \beta(1+r) \mathbb{E}_y\, u_1(c', H''),
$$

with $c$, $c'$, $H'$ defined analogously, and where if the time $t$ discrete choice is not to adjust, then $d = 0$.

---

## Modular Bellman Form

The formulation above follows the standard presentation in the literature — a single Bellman equation with a discrete choice indicator $d$ embedded in the budget constraint. Below we rewrite the same model in **modular Bellman form**: each period is decomposed into self-contained stages, each with its own state space, value function, and optimality condition. This decomposition is the basis for the stage-based computational pipeline.

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

### `tenure` — discrete branching + income shock

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

**Backward mover $\mathbb{B}$ (cntn → dcsn).** Conditional on $z_{\succ}$, the agent picks the better branch:

$$\mathrm{v}(a,\, H,\, z_{\succ}) = \max\!\bigl(\,\mathrm{v}_{\mathrm{kp},\succ}(w_{\mathrm{kp}},\, H_{\mathrm{kp}}),\;\; \mathrm{v}_{\mathrm{adj},\succ}(w_{\mathrm{adj}})\,\bigr).$$

**Marginal-value propagation.** Write $\mathbb{1}_{\mathrm{adj}}$ for the branch indicator. The envelope conditions at the decision perch are:

$$\partial_a \mathrm{v} = \mathbb{1}_{\mathrm{adj}}\; R\;\partial_{w_{\mathrm{adj}}} \mathrm{v}_{\mathrm{adj},\succ} + (1-\mathbb{1}_{\mathrm{adj}})\; R\;\partial_{w_{\mathrm{kp}}} \mathrm{v}_{\mathrm{kp},\succ},$$

$$\partial_H \mathrm{v} = \mathbb{1}_{\mathrm{adj}}\; R_H(1-\delta)\;\partial_{w_{\mathrm{adj}}} \mathrm{v}_{\mathrm{adj},\succ} + (1-\mathbb{1}_{\mathrm{adj}})\;(1-\delta)\;\partial_{H_{\mathrm{kp}}} \mathrm{v}_{\mathrm{kp},\succ}.$$

**Arrival mover $\mathbb{I}$ (dcsn → arvl).** Integrate over the IID innovation $\varepsilon_{\succ}$, with $z_{\succ} = z^{\rho_z}\exp(\sigma_z\,\varepsilon_{\succ})$:

$$\mathrm{v}_{\prec}(a,H,z) = \mathbb{E}_{\varepsilon_{\succ}}\!\bigl[\mathrm{v}(a,H,z_{\succ})\bigr], \qquad \partial_a \mathrm{v}_{\prec} = \mathbb{E}_{\varepsilon_{\succ}}\!\bigl[\partial_a \mathrm{v}\bigr], \qquad \partial_H \mathrm{v}_{\prec} = \mathbb{E}_{\varepsilon_{\succ}}\!\bigl[\partial_H \mathrm{v}\bigr].$$

### `keeper_cons` — keeper consumption (1D EGM + FUES)

| Perch | Variables | Description |
| :--- | :--- | :--- |
| Arrival $(\prec)$ | $w_{\prec},\, H_{\prec}$ | Cash-on-hand, inherited (fixed) housing |
| Decision | $w,\, H$ | Same — no transition before optimisation |
| Continuation $(\succ)$ | $a_{\succ},\, H_{\succ} = H$ | Post-savings assets; housing passes through |

**Decision mover $\mathbb{B}$** (continuation → decision)

The keeper optimises over consumption alone:

$$\mathrm{v}_{\mathrm{kp}}(w, H) = \max_{c \ge 0}\;\bigl\{\mathrm{u}(c, H) + \beta\,\mathrm{v}_{\succ}(w - c,\; H)\bigr\},$$

subject to $a_{\succ} = w - c \ge 0$. Housing passes through unchanged: $H_{\succ} = H$.

The FOC is $\mathrm{u}_1(c, H) = \partial_a \mathrm{v}_{\succ}(a_{\succ}, H)$.

*EGM step.* Fix $a_{\succ}$ on the exogenous grid. The financial Euler inverts analytically:

$$c_{\succ} = \mathrm{u}_1^{-1}\!\bigl(\partial_a \mathrm{v}_{\succ}(a_{\succ}, H)\bigr), \qquad w_{\succ} = a_{\succ} + c_{\succ}.$$

This yields the endogenous cash-on-hand grid. FUES selects the upper envelope from the pooled constrained and unconstrained EGM points.

*Non-concavity.* The continuation $\mathrm{v}_{\succ}(a_{\succ}, H)$ is next period's tenure-stage arrival value. The branching $\max$ introduces primary kinks at the keep/adjust switching threshold; future switching thresholds propagate backward as *secondary kinks* — the DC-EGM mechanism of Iskhakov et al. (2017). As a result $\partial_a \mathrm{v}_{\succ}(\cdot, H)$ is non-monotone and the EGM-constructed $w_{\succ}(a_{\succ})$ mapping has crossing segments.

**Arrival mover $\mathbb{I}$** (decision → arrival)

The arrival transition is an identity ($w = w_{\prec},\; H = H_{\prec}$). The marginal values follow from the envelope theorem:

$$\partial_w \mathrm{v}_{\mathrm{kp}}(w, H) = \mathrm{u}_1(c^\star, H), \qquad \partial_H \mathrm{v}_{\mathrm{kp}}(w, H) = \mathrm{u}_2(c^\star, H) + \beta\,\partial_H \mathrm{v}_{\succ}(a_{\succ}, H).$$

The marginal $\partial_H \mathrm{v}_{\mathrm{kp}}$ — housing service value plus the continuation shadow value — is the quantity $\Theta$ in the paper's stopping-time representation.

### `adjuster_cons` — adjuster consumption (2D partial EGM + FUES)

| Perch | Variables | Description |
| :--- | :--- | :--- |
| Arrival $(\prec)$ | $m_{\prec}$ | Total liquidated wealth from tenure stage |
| Decision | $m$ | Same scalar total wealth |
| Continuation $(\succ)$ | $a_{\succ},\, H_{\succ}$ | New savings and new housing |

**Decision mover $\mathbb{B}$** (continuation → decision)

The adjuster jointly chooses consumption and new housing:

$$\mathrm{v}_{\mathrm{adj}}(m) = \max_{c,\, H_{\succ} \ge 0}\;\bigl\{\mathrm{u}(c, H_{\succ}) + \beta\,\mathrm{v}_{\succ}(m - c - (1+\tau)\,H_{\succ},\; H_{\succ})\bigr\},$$

subject to $a_{\succ} = m - c - (1+\tau)\,H_{\succ} \ge 0$. The two FOCs are:

$$\mathrm{u}_1(c, H_{\succ}) = \partial_a \mathrm{v}_{\succ}, \qquad (1+\tau)\,\mathrm{u}_1(c, H_{\succ}) = \mathrm{u}_2(c, H_{\succ}) + \beta\,\partial_H \mathrm{v}_{\succ}.$$

The housing FOC equates the marginal cost $(1+\tau)\,\mathrm{u}_1$ to the marginal benefit $\mathrm{u}_2 + \beta\,\partial_H \mathrm{v}_{\succ}$.

*Partial EGM.* Fix $H_{\succ}$ on an exogenous grid. For each $\hat{H}_i$: invert the financial Euler to get $c_{\succ}(a_{\succ}, \hat{H}_i)$, recover the endogenous wealth $m_{\succ} = a_{\succ} + c_{\succ} + (1+\tau)\,\hat{H}_i$, and evaluate the Bellman value $\hat{\mathrm{v}}$. Pool all candidates across the full $(a_{\succ}, \hat{H}_i)$ grid and apply FUES to recover the optimal $(c^\star, H_{\succ}^\star)$ as functions of $m$.

*Multiple roots.* The mapping $a_{\succ} \mapsto m_{\succ}$ is **non-monotone** for two compounding reasons: (i) secondary kinks from future tenure switches make $\partial_a \mathrm{v}_{\succ}(\cdot, H)$ non-monotone; (ii) intertemporal income effects between $H_{\succ}$ and $a_{\succ}$ — housing is simultaneously a consumption good and an asset — produce discontinuous drops in housing investment at tenure-switch thresholds. The housing policy $H_{\succ}^\star(m)$ exhibits upward and downward jumps. MSS and LTM require locally isolated monotone segments and cannot resolve this structure. FUES scans the full cloud in a single $O(n \log n)$ pass.

**Arrival mover $\mathbb{I}$** (decision → arrival)

The arrival transition is an identity ($m = m_{\prec}$). The envelope theorem gives:

$$\partial_m \mathrm{v}_{\mathrm{adj}}(m) = \mathrm{u}_1(c^\star, H_{\succ}^\star).$$
