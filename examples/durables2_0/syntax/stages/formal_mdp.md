======================================================================
Query: Translate each of these three dolo-plus stage YAMLs to a formal MDP writeup. For each stage, produce: (1) state space, (2) action space, (3) transitions, (4) Bellman equation with perch structure, (5) first-order conditions / EGM representation.

=== STAGE 1: keeper_cons ===
name: KeeperConsumption

symbols:
  spaces:
    Xw: '@def R+'          # cash-on-hand domain
    Xh: '@def R+'          # housing stock domain
    Xa: '@def R+'          # financial savings domain

  prestate:
    w_keep: '@in Xw'       # cash-on-hand from branching stage
    h_keep: '@in Xh'       # housing stock from branching stage

  states:
    w_keep: '@in Xw'       # cash-on-hand (pass-through)
    h_keep: '@in Xh'       # housing stock (conditioning variable, no choice)

  poststates:
    a_nxt: '@in Xa'        # end-of-period financial savings
    h_nxt: '@in Xh'        # end-of-period housing = (1 - delta) * h_keep

  controls:
    c: '@in R+'

  values:
    V[<]: '@in R'
    V: '@in R'
    V[>]: '@in R'          # V[>]: R^2 -> R (depends on a_nxt, h_nxt)

  values_marginal:
    d_{w}V[<]: '@in R'
    d_{h}V[<]: '@in R'
    d_{w}V: '@in R'
    d_{h}V: '@in R'
    d_{a_nxt}V[>]: '@in R+'     # partial_{a_nxt} v[>]
    d_{h_nxt}V[>]: '@in R+'     # partial_{h_nxt} v[>]

  functions:
    u: 'c, h -> alpha * c^(1 - gamma_c) / (1 - gamma_c) + (1 - alpha) * (kappa * h)^(1 - gamma_h) / (1 - gamma_h)'
    d_{c}u: 'c -> alpha * c^(-gamma_c)'
    d_{h}u: 'h -> (1 - alpha) * kappa^(1 - gamma_h) * h^(-gamma_h)'

  parameters:
    R: '@in R+'
    beta: '@in (0,1)'
    alpha: '@in (0,1)'
    delta: '@in [0,1]'
    gamma_c: '@in R+'
    gamma_h: '@in R+'
    kappa: '@in R+'

equations:
  arvl_to_dcsn_transition: |
    w_keep = w_keep[<]
    h_keep = h_keep[<]

  dcsn_to_cntn_transition: |
    a_nxt = w_keep - c
    h_nxt = (1 - delta) * h_keep

  cntn_to_dcsn_mover:
    Bellman: |
      V = max_{c}(u(c, (1 - delta) * h_keep) + beta * V[>])
    InvEuler: |
      c[>] = (alpha / (beta * d_{a_nxt}V[>]))^(1 / gamma_c)
    cntn_to_dcsn_transition: |
      w_keep[>] = a_nxt + c[>]
    MarginalBellman: |
      d_{w}V = d_{c}u(c)
      d_{h}V = d_{h}u((1 - delta) * h_keep) * (1 - delta) + beta * d_{h_nxt}V[>] * (1 - delta)

  dcsn_to_arvl_mover:
    Bellman: |
      V[<] = V
    MarginalBellman: |
      d_{w}V[<] = d_{w}V
      d_{h}V[<] = d_{h}V

=== STAGE 2: adjuster_cons ===
name: AdjusterConsumption

symbols:
  spaces:
    Xw: '@def R+'          # total wealth domain
    Xa: '@def R+'          # financial savings domain
    Xh: '@def R+'          # housing stock domain

  prestate:
    w_adj: '@in Xw'        # total wealth from branching stage

  states:
    m: '@in Xw'            # total wealth (decision perch)

  poststates:
    a_nxt: '@in Xa'        # end-of-period financial savings
    h_nxt: '@in Xh'        # end-of-period housing stock

  controls:
    c: '@in R+'            # consumption
    h_choice: '@in R+'     # housing choice (adjuster picks freely)

  values:
    V[<]: '@in R'
    V: '@in R'
    V[>]: '@in R'

  values_marginal:
    d_{w}V[<]: '@in R'
    d_{w}V: '@in R'
    d_{a_nxt}V[>]: '@in R+'      # partial_{a_nxt} v[>]
    d_{h_nxt}V[>]: '@in R+'      # partial_{h_nxt} v[>]

  functions:
    u: 'c, h -> alpha * c^(1 - gamma_c) / (1 - gamma_c) + (1 - alpha) * (kappa * h)^(1 - gamma_h) / (1 - gamma_h)'
    d_{c}u: 'c -> alpha * c^(-gamma_c)'
    d_{h}u: 'h -> (1 - alpha) * kappa^(1 - gamma_h) * h^(-gamma_h)'

  parameters:
    alpha: '@in (0,1)'
    beta: '@in (0,1)'
    R: '@in R+'
    tau: '@in R+'
    gamma_c: '@in R+'
    gamma_h: '@in R+'
    kappa: '@in R+'

equations:
  arvl_to_dcsn_transition: |
    m = w_adj

  dcsn_to_cntn_transition: |
    a_nxt = m - c - (1 + tau) * h_choice
    h_nxt = h_choice

  cntn_to_dcsn_mover:
    Bellman: |
      V = max_{c, h_choice}(u(c, h_choice) + beta * V[>])

    InvEuler: |
      solve_{c[>], h_choice[>]}{d_{c}u(c[>]) - beta * d_{a_nxt}V[>] = 0, (1 + tau) * d_{c}u(c[>]) - d_{h}u(h_choice[>]) - beta * d_{h_nxt}V[>] = 0}

    cntn_to_dcsn_transition: |
      m[>] = a_nxt + c[>] + (1 + tau) * h_choice[>]

    MarginalBellman: |
      d_{w}V = d_{c}u(c)

  dcsn_to_arvl_mover:
    Bellman: |
      V[<] = V
    MarginalBellman: |
      d_{w}V[<] = d_{w}V

=== STAGE 3: tenure (branching) ===
name: AdjustmentDecision
kind: branching
branch_control: agent

symbols:
  spaces:
    Xa: '@def R+'          # financial assets domain
    Xh: '@def R+'          # housing stock domain
    Xw: '@def R+'          # total wealth / cash-on-hand domain

  prestate:
    a: '@in Xa'            # financial assets (arrival)
    h: '@in Xh'            # housing stock (arrival)

  states:
    a: '@in Xa'            # financial assets (pass-through)
    h: '@in Xh'            # housing stock (pass-through)

  exogenous:
    z: '@dist MarkovChain(Pi)'

  poststates:
    keep:
      w_keep: '@in Xw'    # keeper cash-on-hand = R * a_d + y(z)
      h_keep: '@in Xh'    # housing stock passed to keeper = h_d
    adjust:
      w_adj: '@in Xw'     # total wealth = R * a_d + R_H * (1 - delta) * h_d + y(z)

  controls:
    d: '@in {keep, adjust}'

  values:
    V[<]: '@in R'
    V: '@in R'
    V[>]:
      keep: '@in R'
      adjust: '@in R'

  values_marginal:
    d_{a}V[<]: '@in R'          # partial_a v at arrival
    d_{h}V[<]: '@in R'          # partial_h v at arrival
    d_{a}V: '@in R'             # partial_a v at decision
    d_{h}V: '@in R'             # partial_h v at decision
    d_{w_keep}V[>]: '@in R'     # keep branch: partial_{w_keep} v[>]
    d_{h_keep}V[>]: '@in R'     # keep branch: partial_{h_keep} v[>]
    d_{w_adj}V[>]: '@in R'      # adjust branch: partial_{w_adj} v[>]
  
  parameters:
    R: '@in R+'            # gross financial return (1 + r)
    R_H: '@in R+'          # gross housing return (1 + r_H)
    delta: '@in [0,1]'     # housing depreciation rate
    y: '@in R+'            # income (deterministic given z, age)

equations:
  arvl_to_dcsn_transition: |
    a = a[<]
    h = h[<]

  dcsn_to_cntn_transition:
    keep: |
      w_keep[>] = R * a + y(z)
      h_keep[>] = h
    adjust: |
      w_adj[>] = R * a + R_H * (1 - delta) * h + y(z)

  cntn_to_dcsn_mover:
    Bellman: |
      V = max_{d}(V[>][keep], V[>][adjust])
    MarginalBellman:
      keep: |
        d_{a}V = R * d_{w_keep}V[>]
        d_{h}V = d_{h_keep}V[>]
      adjust: |
        d_{a}V = R * d_{w_adj}V[>]
        d_{h}V = R_H * (1 - delta) * d_{w_adj}V[>]

  dcsn_to_arvl_mover:
    Bellman: |
      V[<] = E_{z}(V)
    MarginalBellman: |
      d_{a}V[<] = E_{z}(d_{a}V)
      d_{h}V[<] = E_{z}(d_{h}V)
======================================================================

# Formal MDP Writeups for the Housing Adjustment Model

Below I translate each of the three stage YAMLs into a formal MDP document, following the Sargent–Stachurski expository style and the DDSL perch-based notation.

---

## Stage 1 — Keeper Consumption

### 1. Rosetta Stone

| DDSL Abstract | This Stage |
|:---|:---|
| Stage name | `KeeperConsumption` |
| Arrival perch $x_{\prec}$ | $(w_{\text{keep}}, h_{\text{keep}}) \in X_w \times X_h$ |
| Decision perch $x$ | $(w_{\text{keep}}, h_{\text{keep}}) \in X_w \times X_h$ |
| Continuation perch $x_{\succ}$ | $(a_{\text{nxt}}, h_{\text{nxt}}) \in X_a \times X_h$ |
| Controls | $c \in \mathbb{R}_+$ |
| Exogenous shocks | — (none) |
| Value functions | $\mathrm{v}_{\prec},\; \mathrm{v},\; \mathrm{v}_{\succ}$ |
| Marginal values | $\partial_w \mathrm{v}_{\prec},\; \partial_h \mathrm{v}_{\prec},\; \partial_w \mathrm{v},\; \partial_h \mathrm{v},\; \partial_{a} \mathrm{v}_{\succ},\; \partial_{h} \mathrm{v}_{\succ}$ |
| Arrival → Decision mover $\mathbb{I}$ | Identity (no pre-decision shock) |
| Continuation → Decision mover $\mathbb{B}$ | EGM-amenable maximization |

### 2. Model

**State and action spaces.** The keeper enters with cash-on-hand $w_{\text{keep}} \in X_w \coloneqq \mathbb{R}_+$ and a housing stock $h_{\text{keep}} \in X_h \coloneqq \mathbb{R}_+$ that was chosen at a prior branching stage. The arrival-to-decision transition is the identity:

$$
\mathrm{g}_{\prec\circ}(w_{\text{keep}}, h_{\text{keep}}) = (w_{\text{keep}}, h_{\text{keep}}).
$$

The single control is non-durable consumption $c \in \mathbb{R}_+$.

**Preferences.** The per-period utility function is additively separable in consumption and housing services:

$$
u(c, h) \coloneqq \alpha \frac{c^{1-\gamma_c}}{1-\gamma_c} + (1-\alpha) \frac{(\kappa h)^{1-\gamma_h}}{1-\gamma_h},
$$

where $\alpha \in (0,1)$ governs the consumption weight, $\gamma_c > 0$ and $\gamma_h > 0$ are curvature parameters, and $\kappa > 0$ scales housing services. The keeper does not choose housing; housing services flow from the depreciated stock $(1-\delta)\,h_{\text{keep}}$.

**Transitions.** The decision-to-continuation transition $\mathrm{g}_{\circ\succ}$ maps:

$$
a_{\text{nxt}} = w_{\text{keep}} - c, \qquad h_{\text{nxt}} = (1-\delta)\,h_{\text{keep}},
$$

where $\delta \in [0,1]$ is the housing depreciation rate. The budget constraint requires $c \leq w_{\text{keep}}$, ensuring $a_{\text{nxt}} \geq 0$.

### 3. Bellman Equation

The stage has three perches. Since there are no pre-decision shocks, the arrival mover $\mathbb{I}$ is trivial.

**Continuation → Decision mover $\mathbb{B}$ (optimize).** Given a continuation value function $\mathrm{v}_{\succ}: X_a \times X_h \to \mathbb{R}$, the decision-perch value is

$$
\mathrm{v}(w_{\text{keep}},\, h_{\text{keep}}) = \max_{c \,\geq\, 0} \Bigl\{ u\bigl(c,\, (1-\delta)\,h_{\text{keep}}\bigr) + \beta\, \mathrm{v}_{\succ}\bigl(w_{\text{keep}} - c,\; (1-\delta)\,h_{\text{keep}}\bigr) \Bigr\}.
$$

**Decision → Arrival mover $\mathbb{I}$ (identity).** In the absence of pre-decision shocks:

$$
\mathrm{v}_{\prec}(w_{\text{keep}},\, h_{\text{keep}}) = \mathrm{v}(w_{\text{keep}},\, h_{\text{keep}}).
$$

The full backward operator is $\mathbb{T} = \mathbb{I} \circ \mathbb{B}$.

### 4. First-Order Conditions and EGM Representation

**First-order condition.** Interior solutions satisfy

$$
\partial_c\, u(c,\, (1-\delta)h_{\text{keep}}) = \beta\, \partial_{a_{\text{nxt}}}\, \mathrm{v}_{\succ}(a_{\text{nxt}},\, h_{\text{nxt}}),
$$

which, given the functional form, becomes

$$
\alpha\, c^{-\gamma_c} = \beta\, \partial_{a_{\text{nxt}}}\, \mathrm{v}_{\succ}. \tag{FOC}
$$

**Inverse Euler (EGM).** Fix a grid point $(a_{\text{nxt}}, h_{\text{nxt}})$ in the continuation space. The inverse Euler equation recovers optimal consumption as a function of continuation-perch quantities:

$$
c_{\succ} = \left( \frac{\alpha}{\beta\, \partial_{a_{\text{nxt}}}\, \mathrm{v}_{\succ}} \right)^{1/\gamma_c}.
$$

The subscript on $c_{\succ}$ denotes that this is a continuation-measurable object (written `c[>]` in YAML). The reverse transition recovers the endogenous grid for $w_{\text{keep}}$:

$$
w_{\text{keep},\succ} = a_{\text{nxt}} + c_{\succ}.
$$

**Marginal value functions (envelope conditions).** The marginal values at the decision perch are:

$$
\partial_w\, \mathrm{v}(w_{\text{keep}},\, h_{\text{keep}}) = \alpha\, c^{*-\gamma_c},
$$

$$
\partial_h\, \mathrm{v}(w_{\text{keep}},\, h_{\text{keep}}) = (1-\alpha)\,\kappa^{1-\gamma_h}\,\bigl((1-\delta)\,h_{\text{keep}}\bigr)^{-\gamma_h}\,(1-\delta) \;+\; \beta\, \partial_{h_{\text{nxt}}}\, \mathrm{v}_{\succ}\!\bigl(a_{\text{nxt}},\, h_{\text{nxt}}\bigr)\,(1-\delta).
$$

The first term is the direct marginal utility of housing services; the second is the continuation capital-gains channel, both scaled by the chain-rule factor $(1-\delta)$ from the depreciation law $h_{\text{nxt}} = (1-\delta)h_{\text{keep}}$.

Since $\mathbb{I}$ is the identity, $\partial_w\, \mathrm{v}_{\prec} = \partial_w\, \mathrm{v}$ and $\partial_h\, \mathrm{v}_{\prec} = \partial_h\, \mathrm{v}$.

### 5. Forward Operator (Population Dynamics)

Let $\mu$ denote the measure over $(w_{\text{keep}}, h_{\text{keep}})$ at the arrival perch. Since the arrival-to-decision transition is the identity, the decision-perch measure is $\mu$ itself. Given the policy $c^*(w_{\text{keep}}, h_{\text{keep}})$, the continuation-perch measure $\mu_{\succ}$ over $(a_{\text{nxt}}, h_{\text{nxt}})$ is the pushforward:

$$
\mu_{\succ} = \mathrm{g}_{\circ\succ}(\cdot\,; c^*)\,\#\, \mu,
$$

where $\mathrm{g}_{\circ\succ}(w,h; c^*) = (w - c^*(w,h),\; (1-\delta)h)$.

### 6. Calibration

| Symbol | Domain | Description |
|:---|:---|:---|
| $R$ | $\mathbb{R}_+$ | Gross financial return |
| $\beta$ | $(0,1)$ | Discount factor |
| $\alpha$ | $(0,1)$ | Consumption weight in utility |
| $\delta$ | $[0,1]$ | Housing depreciation rate |
| $\gamma_c$ | $\mathbb{R}_+$ | Consumption CRRA curvature |
| $\gamma_h$ | $\mathbb{R}_+$ | Housing CRRA curvature |
| $\kappa$ | $\mathbb{R}_+$ | Housing services scaling |

---

## Stage 2 — Adjuster Consumption

### 1. Rosetta Stone

| DDSL Abstract | This Stage |
|:---|:---|
| Stage name | `AdjusterConsumption` |
| Arrival perch $x_{\prec}$ | $w_{\text{adj}} \in X_w$ |
| Decision perch $x$ | $m \in X_w$ |
| Continuation perch $x_{\succ}$ | $(a_{\text{nxt}}, h_{\text{nxt}}) \in X_a \times X_h$ |
| Controls | $(c, h_{\text{choice}}) \in \mathbb{R}_+^2$ |
| Exogenous shocks | — (none) |
| Value functions | $\mathrm{v}_{\prec},\; \mathrm{v},\; \mathrm{v}_{\succ}$ |
| Marginal values | $\partial_w \mathrm{v}_{\prec},\; \partial_w \mathrm{v},\; \partial_a \mathrm{v}_{\succ},\; \partial_h \mathrm{v}_{\succ}$ |
| Arrival → Decision mover $\mathbb{I}$ | Identity |
| Continuation → Decision mover $\mathbb{B}$ | Two-control optimization (system EGM) |

### 2. Model

**State and action spaces.** The adjuster enters with total wealth $w_{\text{adj}} \in X_w \coloneqq \mathbb{R}_+$, which bundles financial assets and the liquidation value of old housing (net of transaction costs, computed at the branching stage). The arrival-to-decision transition relabels:

$$
\mathrm{g}_{\prec\circ}(w_{\text{adj}}) = m \coloneqq w_{\text{adj}}.
$$

The agent jointly chooses consumption $c \in \mathbb{R}_+$ and a new housing stock $h_{\text{choice}} \in \mathbb{R}_+$.

**Preferences.** The same additively separable utility as Stage 1:

$$
u(c, h) \coloneqq \alpha \frac{c^{1-\gamma_c}}{1-\gamma_c} + (1-\alpha) \frac{(\kappa h)^{1-\gamma_h}}{1-\gamma_h}.
$$

Crucially, the adjuster enjoys housing services from the *newly chosen* stock $h_{\text{choice}}$, not from a depreciated inherited stock.

**Transitions.** The decision-to-continuation transition $\mathrm{g}_{\circ\succ}$ maps:

$$
a_{\text{nxt}} = m - c - (1+\tau)\,h_{\text{choice}}, \qquad h_{\text{nxt}} = h_{\text{choice}},
$$

where $\tau \geq 0$ is a proportional transaction cost on housing purchases. The budget constraint $c + (1+\tau)\,h_{\text{choice}} \leq m$ ensures $a_{\text{nxt}} \geq 0$.

### 3. Bellman Equation

**Continuation → Decision mover $\mathbb{B}$ (optimize).** Given $\mathrm{v}_{\succ}: X_a \times X_h \to \mathbb{R}$,

$$
\mathrm{v}(m) = \max_{c,\, h_{\text{choice}} \,\geq\, 0} \Bigl\{ u(c,\, h_{\text{choice}}) + \beta\, \mathrm{v}_{\succ}\bigl(m - c - (1+\tau)\,h_{\text{choice}},\; h_{\text{choice}}\bigr) \Bigr\}.
$$

**Decision → Arrival mover $\mathbb{I}$ (identity).**

$$
\mathrm{v}_{\prec}(w_{\text{adj}}) = \mathrm{v}(w_{\text{adj}}).
$$

### 4. First-Order Conditions and EGM Representation

**First-order conditions.** Interior solutions satisfy a system of two equations:

$$
\partial_c\, u(c, h_{\text{choice}}) = \beta\, \partial_{a_{\text{nxt}}}\, \mathrm{v}_{\succ}(a_{\text{nxt}}, h_{\text{nxt}}), \tag{FOC-c}
$$

$$
\partial_h\, u(c, h_{\text{choice}}) + \beta\, \partial_{h_{\text{nxt}}}\, \mathrm{v}_{\succ}(a_{\text{nxt}}, h_{\text{nxt}}) = (1+\tau)\,\beta\, \partial_{a_{\text{nxt}}}\, \mathrm{v}_{\succ}(a_{\text{nxt}}, h_{\text{nxt}}). \tag{FOC-h}
$$

The first is the standard Euler equation for consumption. The second equates the total marginal benefit of housing — direct utility plus continuation capital gains — to the marginal cost of housing, which is $(1+\tau)$ units of forgone financial savings.

**Interpretation.** Substituting (FOC-c) into (FOC-h) eliminates $\mathrm{v}_{\succ}$ from the cost side:

$$
(1+\tau)\,\partial_c\, u(c, h_{\text{choice}}) = \partial_h\, u(c, h_{\text{choice}}) + \beta\, \partial_{h_{\text{nxt}}}\, \mathrm{v}_{\succ}.
$$

This intratemporal condition balances the marginal rate of substitution between consumption and housing against the relative price $(1+\tau)$, adjusted for the continuation value of housing capital.

**System EGM (Inverse Euler).** Fix a continuation grid point $(a_{\text{nxt}}, h_{\text{nxt}})$. Read off $\partial_{a_{\text{nxt}}}\,\mathrm{v}_{\succ}$ and $\partial_{h_{\text{nxt}}}\,\mathrm{v}_{\succ}$. The YAML specifies a simultaneous solve for the continuation-measurable controls $(c_{\succ}, h_{\text{choice},\succ})$:

$$
\begin{cases}
\alpha\, c_{\succ}^{-\gamma_c} - \beta\, \partial_{a_{\text{nxt}}}\, \mathrm{v}_{\succ} = 0, \\[6pt]
(1+\tau)\,\alpha\, c_{\succ}^{-\gamma_c} - (1-\alpha)\,\kappa^{1-\gamma_h}\, h_{\text{choice},\succ}^{-\gamma_h} - \beta\, \partial_{h_{\text{nxt}}}\, \mathrm{v}_{\succ} = 0.
\end{cases}
$$

From the first equation:

$$
c_{\succ} = \left( \frac{\alpha}{\beta\, \partial_{a_{\text{nxt}}}\, \mathrm{v}_{\succ}} \right)^{1/\gamma_c}.
$$

Substituting into the second and solving for $h_{\text{choice},\succ}$:

$$
h_{\text{choice},\succ} = \left( \frac{(1-\alpha)\,\kappa^{1-\gamma_h}}{(1+\tau)\,\alpha\, c_{\succ}^{-\gamma_c} - \beta\, \partial_{h_{\text{nxt}}}\, \mathrm{v}_{\succ}} \right)^{1/\gamma_h}.
$$

The reverse transition recovers the endogenous grid for total wealth:

$$
m_{\succ} = a_{\text{nxt}} + c_{\succ} + (1+\tau)\, h_{\text{choice},\succ}.
$$

**Note.** The two-control system EGM is structurally more complex than single-control EGM: the system may not admit a closed-form sequential solution in all parameterizations, and the `solve_{...}{...}` syntax in the YAML reflects this by declaring a simultaneous root-finding problem rather than a single inversion.

**Marginal value (envelope).**

$$
\partial_w\, \mathrm{v}(m) = \alpha\, c^{*-\gamma_c},
$$

which, by the identity mover, also gives $\partial_w\, \mathrm{v}_{\prec}(w_{\text{adj}}) = \partial_w\, \mathrm{v}(w_{\text{adj}})$.

### 5. Forward Operator (Population Dynamics)

Let $\mu$ denote the measure over $w_{\text{adj}}$ at the arrival perch. Under the joint policy $(c^*(m), h^*(m))$, the continuation-perch measure $\mu_{\succ}$ over $(a_{\text{nxt}}, h_{\text{nxt}})$ is:

$$
\mu_{\succ} = \mathrm{g}_{\circ\succ}(\cdot\,; c^*, h^*)\,\#\, \mu,
$$

where $\mathrm{g}_{\circ\succ}(m; c^*, h^*) = \bigl(m - c^*(m) - (1+\tau)h^*(m),\; h^*(m)\bigr)$.

### 6. Calibration

| Symbol | Domain | Description |
|:---|:---|:---|
| $\alpha$ | $(0,1)$ | Consumption weight in utility |
| $\beta$ | $(0,1)$ | Discount factor |
| $R$ | $\mathbb{R}_+$ | Gross financial return |
| $\tau$ | $\mathbb{R}_+$ | Proportional housing transaction cost |
| $\gamma_c$ | $\mathbb{R}_+$ | Consumption CRRA curvature |
| $\gamma_h$ | $\mathbb{R}_+$ | Housing CRRA curvature |
| $\kappa$ | $\mathbb{R}_+$ | Housing services scaling |

---

## Stage 3 — Adjustment Decision (Branching)

### 1. Rosetta Stone

| DDSL Abstract | This Stage |
|:---|:---|
| Stage name | `AdjustmentDecision` |
| Kind | `branching`, `branch_control: agent` |
| Arrival perch $x_{\prec}$ | $(a, h) \in X_a \times X_h$ |
| Decision perch $x$ | $(a, h, z) \in X_a \times X_h \times Z$ |
| Continuation perch $x_{\succ}$ | Coproduct: $X_{\succ} = X_{\succ}^{\text{keep}} \sqcup X_{\succ}^{\text{adj}}$ |
| $\quad$ keep branch | $(w_{\text{keep}}, h_{\text{keep}}) \in X_w \times X_h$ |
| $\quad$ adjust branch | $w_{\text{adj}} \in X_w$ |
| Controls | $d \in \{\text{keep}, \text{adjust}\}$ (discrete branch selector) |
| Exogenous shocks | $z \sim \text{MarkovChain}(\Pi)$ |
| Aggregator | $\mathcal{A}_{\text{agent}} = \max$ |
| Value functions | $\mathrm{v}_{\prec},\; \mathrm{v},\; \mathrm{v}_{\succ}^{\text{keep}},\; \mathrm{v}_{\succ}^{\text{adj}}$ |
| Marginal values | $\partial_a \mathrm{v}_{\prec},\; \partial_h \mathrm{v}_{\prec},\; \partial_a \mathrm{v},\; \partial_h \mathrm{v}$; branch-indexed continuation marginals |
| Arrival → Decision mover $\mathbb{I}$ | Expectation over Markov shock $z$ |
| Continuation → Decision mover $\mathbb{B}$ | Discrete max over branch values |

### 2. Model

**State and action spaces.** The agent enters with financial assets $a \in X_a \coloneqq \mathbb{R}_+$ and housing stock $h \in X_h \coloneqq \mathbb{R}_+$. At the arrival perch, a Markov income shock $z$ is drawn from a transition matrix $\Pi$. The arrival-to-decision transition is the identity on $(a,h)$ augmented by the shock resolution:

$$
\mathrm{g}_{\prec\circ}(a, h; z) = (a, h, z).
$$

The discrete control $d \in \{\text{keep}, \text{adjust}\}$ determines which successor stage the agent enters:

- **Keep**: the agent retains the current housing stock.
- **Adjust**: the agent sells housing at gross return $R_H(1-\delta)$ per unit, folding proceeds into total wealth, and enters the adjuster stage where a new housing stock is chosen.

**Transitions (branch-indexed).** The decision-to-continuation transition is a *family* indexed by the branch label $d$:

$$
\mathrm{g}_{\circ\succ}^{\text{keep}}(a, h, z) = \bigl(\underbrace{R\,a + y(z)}_{w_{\text{keep}}},\; \underbrace{h\vphantom{R_H}}_{h_{\text{keep}}}\bigr),
$$

$$
\mathrm{g}_{\circ\succ}^{\text{adj}}(a, h, z) = \underbrace{R\,a + R_H\,(1-\delta)\,h + y(z)}_{w_{\text{adj}}}.
$$

The keep branch passes the housing stock through unchanged. The adjust branch liquidates the (depreciated) housing stock at gross price $R_H$ and bundles all resources into a scalar wealth variable $w_{\text{adj}}$.

### 3. Bellman Equation

The stage has three perches with a non-trivial arrival mover (expectation over $z$) and a branching continuation mover (discrete max).

**Continuation → Decision mover $\mathbb{B}$ (discrete max).** Let $\mathrm{v}_{\succ}^{\text{keep}}: X_w \times X_h \to \mathbb{R}$ and $\mathrm{v}_{\succ}^{\text{adj}}: X_w \to \mathbb{R}$ be the continuation value functions from the keeper and adjuster stages, respectively. The decision-perch value is:

$$
\mathrm{v}(a, h, z) = \max\Bigl\{\;\mathrm{v}_{\succ}^{\text{keep}}\!\bigl(R\,a + y(z),\; h\bigr),\;\; \mathrm{v}_{\succ}^{\text{adj}}\!\bigl(R\,a + R_H(1-\delta)\,h + y(z)\bigr)\;\Bigr\}.
$$

The optimal branch is:

$$
d^*(a, h, z) = \begin{cases} \text{keep} & \text{if } \mathrm{v}_{\succ}^{\text{keep}}(w_{\text{keep}}, h) \geq \mathrm{v}_{\succ}^{\text{adj}}(w_{\text{adj}}), \\ \text{adjust} & \text{otherwise.} \end{cases}
$$

**Decision → Arrival mover $\mathbb{I}$ (expectation).** The arrival-perch value integrates over the Markov shock:

$$
\mathrm{v}_{\prec}(a, h) = \mathbb{E}_z\!\bigl[\mathrm{v}(a, h, z)\bigr] = \sum_{z'} \Pi(z' \mid z)\; \mathrm{v}(a, h, z').
$$

The full backward operator is $\mathbb{T} = \mathbb{I} \circ \mathbb{B}$.

### 4. Marginal Value Functions

Since the branching stage involves no continuous optimization (the control $d$ is discrete), there are no first-order conditions or EGM inversion. Instead, marginal values propagate through the transitions by the chain rule, conditional on the chosen branch.

**Decision-perch marginal values (branch-conditional).** At the *keep* branch:

$$
\partial_a\, \mathrm{v}\big|_{\text{keep}} = R\, \partial_{w_{\text{keep}}}\, \mathrm{v}_{\succ}^{\text{keep}}(w_{\text{keep}}, h_{\text{keep}}), \qquad \partial_h\, \mathrm{v}\big|_{\text{keep}} = \partial_{h_{\text{keep}}}\, \mathrm{v}_{\succ}^{\text{keep}}(w_{\text{keep}}, h_{\text{keep}}).
$$

At the *adjust* branch:

$$
\partial_a\, \mathrm{v}\big|_{\text{adj}} = R\, \partial_{w_{\text{adj}}}\, \mathrm{v}_{\succ}^{\text{adj}}(w_{\text{adj}}), \qquad \partial_h\, \mathrm{v}\big|_{\text{adj}} = R_H\,(1-\delta)\, \partial_{w_{\text{adj}}}\, \mathrm{v}_{\succ}^{\text{adj}}(w_{\text{adj}}).
$$

The decision-perch marginal is selected by the optimal branch: $\partial_a\, \mathrm{v}(a,h,z) = \partial_a\, \mathrm{v}\big|_{d^*(a,h,z)}$, and likewise for $\partial_h\, \mathrm{v}$.

**Arrival-perch marginal values (expectations).**

$$
\partial_a\, \mathrm{v}_{\prec}(a, h) = \mathbb{E}_z\!\bigl[\partial_a\, \mathrm{v}(a, h, z)\bigr], \qquad \partial_h\, \mathrm{v}_{\prec}(a, h) = \mathbb{E}_z\!\bigl[\partial_h\, \mathrm{v}(a, h, z)\bigr].
$$

These arrival marginals are what the successor period's keeper and adjuster stages receive as $\partial_{a_{\text{nxt}}}\,\mathrm{v}_{\succ}$ and $\partial_{h_{\text{nxt}}}\,\mathrm{v}_{\succ}$, closing the backward-iteration loop.

### 5. Forward Operator (Population Dynamics)

Let $\mu$ denote the population measure over $(a, h)$ at the arrival perch.

**Shock realization.** The decision-perch measure is $\mu \otimes \Pi$, i.e., each $(a,h)$ agent draws $z$ from the Markov transition.

**Branch routing.** The policy $d^*(a, h, z)$ partitions the decision-perch population:

$$
\mu^{\text{keep}}_{\succ}(B) = \int \mathbf{1}\bigl[\mathrm{g}_{\circ\succ}^{\text{keep}}(a,h,z) \in B\bigr]\, \mathbf{1}\bigl[d^*(a,h,z) = \text{keep}\bigr]\; d(\mu \otimes \Pi)(a,h,z),
$$

$$
\mu^{\text{adj}}_{\succ}(B) = \int \mathbf{1}\bigl[\mathrm{g}_{\circ\succ}^{\text{adj}}(a,h,z) \in B\bigr]\, \mathbf{1}\bigl[d^*(a,h,z) = \text{adjust}\bigr]\; d(\mu \otimes \Pi)(a,h,z).
$$

The measure splits: keepers proceed to `KeeperConsumption` with state $(w_{\text{keep}}, h_{\text{keep}})$; adjusters proceed to `AdjusterConsumption` with state $w_{\text{adj}}$.

### 6. Calibration

| Symbol | Domain | Description |
|:---|:---|:---|
| $R$ | $\mathbb{R}_+$ | Gross financial return |
| $R_H$ | $\mathbb{R}_+$ | Gross housing return |
| $\delta$ | $[0,1]$ | Housing depreciation rate |
| $y(\cdot)$ | $\mathbb{R}_+$ | Income function (deterministic given $z$, age) |
| $\Pi$ | Stochastic matrix | Markov transition for income shock $z$ |

---

## Period-Level Composition

The three stages compose into a single period via branching:

$$
\mathbb{T}_{\text{period}} = \mathbb{T}_{\text{tenure}} \circ \mathcal{A}_{\text{agent}}\!\bigl(\mathbb{T}_{\text{keeper}},\; \mathbb{T}_{\text{adjuster}}\bigr),
$$

where $\mathcal{A}_{\text{agent}} = \max$ is the agent-controlled branching aggregator. The wiring is:

- `AdjustmentDecision` (branching) feeds its **keep** continuation perch $(w_{\text{keep}}, h_{\text{keep}})$ to the arrival perch of `KeeperConsumption`.
- `AdjustmentDecision` feeds its **adjust** continuation perch $w_{\text{adj}}$ to the arrival perch of `AdjusterConsumption`.
- Both consumption stages produce continuation states $(a_{\text{nxt}}, h_{\text{nxt}})$ that, after period closing (e.g., returns, income realization), feed back as next-period arrival states $(a, h)$ for `AdjustmentDecision`.

**Backward iteration** proceeds: solve keeper and adjuster stages for their continuation value functions $\mathrm{v}_{\succ}^{\text{keep}}$ and $\mathrm{v}_{\succ}^{\text{adj}}$; then evaluate the branching stage's discrete max and expectation to produce $\mathrm{v}_{\prec}(a,h)$ and its marginals.

======================================================================
Sources (10 chunks):
  [0.739] project/repos/bellman-ddsl/docs/examples/housing_owner.md
  [0.735] project/repos/bellman-ddsl/docs/examples/housing_owner.md
  [0.730] project/repos/bellman-ddsl/docs/examples/cons-port-stages/cons_stage.md
  [0.721] project/repos/bellman-ddsl/docs/examples/housing_owner.md
  [0.715] project/repos/bellman-ddsl/docs/examples/housing_renting/stages/tenure_choice.md
  [0.713] project/repos/bellman-ddsl/docs/examples/housing_owner.md
  [0.713] project/repos/bellman-ddsl/docs/examples/housing_owner.md
  [0.709] project/repos/bellman-ddsl/docs/examples/housing_owner.md
  [0.708] project/repos/bellman-ddsl/docs/examples/index.md
  [0.706] project/repos/bellman-ddsl/docs/examples/housing_owner.md
