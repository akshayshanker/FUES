## 1. Model

An agent lives for $T$ periods. Each period she holds assets $a \geq 0$ and makes two choices: a **discrete** choice $d \in \{\text{work}, \text{retire}\}$ and a **continuous** choice of consumption $c$. Working yields wage income $y$ but costs disutility $\delta$; retirement is absorbing. Assets earn gross return $(1{+}r)$.

The sequence of events within a period are
- workers and retirees start with beginning-of-period assets $a$ and $a_{\text{ret}}$ respectively
- earn returns and receive income to produce cash-on-hand $w = (1{+}r)a + y$ for workers or $w_{\text{ret}} = (1{+}r)a_{\text{ret}}$ for retirees
- consumes $c$, leaving end-of-period savings $b = w - c$ (or $b_{\text{ret}} = w_{\text{ret}} - c$)
- inter-period transition maps $b \to a$ and $b_{\text{ret}} \to a_{\text{ret}}$ for the next period; retirement is absorbing.
### Stage decomposition

Each period can be translated to a directed graph of self-contained modular *stages*, following [Carroll (2026)](https://llorracc.github.io/SolvingMicroDSOPs/); see [Carroll and Shanker (2026)](https://bright-forest.github.io/bellman-ddsl/theory/MDP-foundations/) for the formal framework. The retirement model has three stages:

1. **`labour_mkt_decision`** (branching) — discrete choice: $\max(\mathrm{v}_{\succ}^{\text{work}} - \delta,\; \mathrm{v}_{\succ}^{\text{retire}})$. Assets $a$ pass through unchanged.
2. **`work_cons`** (continuous, EGM + FUES) — worker consumption: $a \to w \to b$. The continuation value $\mathrm{v}_{\succ}$ is non-concave; FUES recovers the correct envelope.
3. **`retire_cons`** (continuous, EGM) — retiree consumption: $a_{\text{ret}} \to w_{\text{ret}} \to b_{\text{ret}}$. Standard concave problem.

Note that workers arrive at $a$ (into the branching stage); retirees arrive at $a_{\text{ret}}$ (directly into `retire_cons`). If a worker chooses to retire, they become a retiree and move into retiree consumption problem,  `retire_cons`, as those who entered the period as a retiree. 


<div markdown="0">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 280" style="max-width:680px;width:100%;">
  <defs>
    <style>
      .dag-edge   { fill: none; stroke: currentColor; stroke-width: 1.2px; stroke-linecap: round; }
      .dag-arrow  { fill: currentColor; }
      .dag-perch  { fill: var(--md-default-bg-color, #fff); stroke: currentColor; stroke-width: 1.2px; }
      .dag-stg    { fill: currentColor; stroke: none; }
      .dag-bound  { fill: none; stroke: currentColor; stroke-width: 0.8px; stroke-dasharray: 4 3; opacity: 0.25; }
      .dag-field  { font-family: Georgia, 'Times New Roman', serif; font-style: italic; font-size: 15px; fill: currentColor; }
      .dag-sub    { font-size: 10.5px; }
      .dag-ptag   { font-family: Georgia, 'Times New Roman', serif; font-style: italic; font-size: 9.5px; fill: currentColor; opacity: 0.40; }
      .dag-slabel { font-family: 'Inter', 'Helvetica Neue', sans-serif; font-size: 11px; fill: currentColor; }
      .dag-branch { font-family: 'Inter', 'Helvetica Neue', sans-serif; font-size: 10px; fill: currentColor; font-style: italic; }
      .dag-period { font-family: 'Inter', 'Helvetica Neue', sans-serif; font-size: 9.5px; fill: currentColor; opacity: 0.30; font-style: italic; }
      .dag-legend { font-family: 'Inter', 'Helvetica Neue', sans-serif; font-size: 8px; fill: currentColor; opacity: 0.45; }
    </style>
    <marker id="dag-ah" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,3 L0,6 L1.5,3 Z" class="dag-arrow"/>
    </marker>
  </defs>
  <rect class="dag-bound" x="14" y="14" width="652" height="252" rx="8"/>
  <text class="dag-period" x="26" y="30">LifecyclePeriod</text>
  <circle class="dag-stg" cx="548" cy="27" r="3.5"/>
  <text class="dag-legend" x="555" y="30">stage</text>
  <circle class="dag-perch" cx="600" cy="27" r="3.5"/>
  <text class="dag-legend" x="607" y="30">perch field</text>
  <circle class="dag-perch" cx="68" cy="105" r="6"/>
  <text class="dag-field" x="68" y="88" text-anchor="middle">a</text>
  <circle class="dag-perch" cx="68" cy="210" r="6"/>
  <text class="dag-field" x="68" y="198" text-anchor="middle">a<tspan class="dag-sub" dy="3">ret</tspan></text>
  <circle class="dag-stg" cx="220" cy="105" r="7"/>
  <text class="dag-slabel" x="220" y="82" text-anchor="middle">labour_mkt_decision</text>
  <circle class="dag-stg" cx="415" cy="65" r="7"/>
  <text class="dag-slabel" x="415" y="48" text-anchor="middle">work_cons</text>
  <circle class="dag-stg" cx="415" cy="192" r="7"/>
  <text class="dag-slabel" x="415" y="222" text-anchor="middle">retire_cons</text>
  <circle class="dag-perch" cx="575" cy="65" r="6"/>
  <text class="dag-field" x="575" y="48" text-anchor="middle">b</text>
  <circle class="dag-perch" cx="575" cy="192" r="6"/>
  <text class="dag-field" x="575" y="180" text-anchor="middle">b<tspan class="dag-sub" dy="3">ret</tspan></text>
  <line class="dag-edge" x1="75" y1="105" x2="212" y2="105" marker-end="url(#dag-ah)"/>
  <path class="dag-edge" d="M228,102 C265,82 360,65 407,65" marker-end="url(#dag-ah)"/>
  <text class="dag-branch" x="310" y="72">work</text>
  <path class="dag-edge" d="M228,109 C268,140 355,182 407,192" marker-end="url(#dag-ah)"/>
  <text class="dag-branch" x="298" y="165">retire</text>
  <path class="dag-edge" d="M75,210 C185,210 335,198 407,193" marker-end="url(#dag-ah)"/>
  <line class="dag-edge" x1="423" y1="65" x2="568" y2="65" marker-end="url(#dag-ah)"/>
  <line class="dag-edge" x1="423" y1="192" x2="568" y2="192" marker-end="url(#dag-ah)"/>
</svg>
</div>

### Stage operators

Within each each stage, the state space is represented at three nodes: arrival ($\mathsf{X}_{\prec}$), decision ($\mathsf{X}$), and continuation ($\mathsf{X}_{\succ}$). Solving proceeds backward: given a continuation-value function $\mathrm{v}_{\succ}$ on $\mathsf{X}_{\succ}$, the decision mover $\mathbb{B}$ produces the decision-value function $\mathrm{v}$ on $\mathsf{X}$, and the arrival mover $\mathbb{I}$ passes $\mathrm{v}$ back to $\mathsf{X}_{\prec}$. Throughout, $\partial\mathrm{v}$ denotes the derivative of $\mathrm{v}$ with respect to the stage's own state variable.

The term mover here refers to operations that move from one node to the next (either forward or backward). The mathematical aspect of the mover is simply a functional operator -- but the mover captures an object that may also contain computational representations of how the operator is implemented on the computer. 

<details style="border-left:3px solid #7c4dff;padding:8px 16px;margin:12px 0;background:rgba(124,77,255,0.04);border-radius:4px;">
<summary style="cursor:pointer;font-weight:600;font-size:0.95em;"><code>work_cons</code> — worker consumption (EGM + FUES)</summary>

**Decision mover $\mathbb{B}$** &ensp; (continuation $\to$ decision)

Let $\mathrm{v}_{\succ}(b)$ be the continuation value at end-of-period savings $b$, and $\partial\mathrm{v}_{\succ}(b)$ its derivative. The worker's cash-on-hand is $w$ and the budget constraint is $b = w - c$. The decision mover solves:

$$(\mathbb{B}\,\mathrm{v}_{\succ})(w) = \mathrm{v}(w) = \max_c\bigl\{\log(c) + \beta\,\mathrm{v}_{\succ}(w - c)\bigr\}$$

The first-order condition is $1/c = \beta\,\partial\mathrm{v}_{\succ}(b)$.

*EGM.* &ensp; Given an exogenous grid $\{b_0^{\#},\dots,b_N^{\#}\}$ on the continuation state, the FOC yields optimal consumption $c_i^{\#} = \bigl(\beta\,\partial\mathrm{v}_{\succ}(b_i^{\#})\bigr)^{-1}$ and the budget constraint recovers the endogenous grid $w_i^{\#} = b_i^{\#} + c_i^{\#}$. Each pair $(w_i^{\#},\, c_i^{\#})$ satisfies the FOC, and the corresponding value is $q_i^{\#} = \log(c_i^{\#}) + \beta\,\mathrm{v}_{\succ}(b_i^{\#})$.

*Non-concavity.* &ensp; The worker's $\mathrm{v}_{\succ}$ is the upper envelope of concave functions (one for each future discrete-choice sequence) and is not itself concave. As a result the endogenous grid $\{w_i^{\#}\}$ may be non-monotone, and the points $(w_i^{\#},\, q_i^{\#})$ define a correspondence rather than a function. An upper-envelope algorithm recovers the monotone upper envelope of $\{(w_i^{\#},\, q_i^{\#})\}$, thereby approximating $\mathrm{v}$. This is where FUES comes in.

**Arrival mover $\mathbb{I}$** &ensp; (decision $\to$ arrival)

The arrival transition is $w = (1{+}r)a + y$, so:

$$(\mathbb{I}\mathrm{v})(a) = \mathrm{v}\bigl((1{+}r)a + y\bigr)$$

The chain rule gives $\partial\mathrm{v}_{\prec}(a) = (1{+}r)\,\partial\mathrm{v}(w)$, and the envelope theorem yields $\partial\mathrm{v}(w) = 1/c$.

</details>

<details style="border-left:3px solid #00897b;padding:8px 16px;margin:12px 0;background:rgba(0,137,123,0.04);border-radius:4px;">
<summary style="cursor:pointer;font-weight:600;font-size:0.95em;"><code>retire_cons</code> — retiree consumption (EGM, concave)</summary>

**Decision mover $\mathbb{B}$** &ensp; (continuation $\to$ decision)

Let $\mathrm{v}_{\succ}(b_{\text{ret}})$ be the continuation value at retiree savings $b_{\text{ret}}$, and $\partial\mathrm{v}_{\succ}(b_{\text{ret}})$ the marginal value. The retiree's cash-on-hand is $w_{\text{ret}}$ and the budget constraint is $b_{\text{ret}} = w_{\text{ret}} - c$. The decision mover is:

$$(\mathbb{B}\mathrm{v}_{\succ})(w_{\text{ret}}) = \mathrm{v}(w_{\text{ret}}) = \max_c\bigl\{\log(c) + \beta\,\mathrm{v}_{\succ}(b_{\text{ret}})\bigr\}$$

such that $b_{\text{ret}} = w_{\text{ret}} - c$. The first-order condition is $1/c = \beta\,\partial\mathrm{v}_{\succ}(b_{\text{ret}})$.

*EGM.* &ensp; Given a grid on $b_{\text{ret}}$, recover $c_i^{\#} = \bigl(\beta\,\partial\mathrm{v}_{\succ}(b_{\text{ret},i}^{\#})\bigr)^{-1}$ and $w_{\text{ret},i}^{\#} = b_{\text{ret},i}^{\#} + c_i^{\#}$. Here $\mathrm{v}_{\succ}$ is concave (retirement is absorbing), so EGM produces a monotone endogenous grid and no upper-envelope step is needed.

**Arrival mover $\mathbb{I}$** &ensp; (decision $\to$ arrival)

The arrival transition is $w_{\text{ret}} = (1{+}r)\,a_{\text{ret}}$ (no income), so:

$$(\mathbb{I}\mathrm{v})(a_{\text{ret}}) = \mathrm{v}\bigl((1{+}r)\,a_{\text{ret}}\bigr)$$

and $\partial\mathrm{v}_{\prec}(a_{\text{ret}}) = (1{+}r)\,\partial\mathrm{v}(w_{\text{ret}})$.

</details>

<details style="border-left:3px solid #1565c0;padding:8px 16px;margin:12px 0;background:rgba(21,101,192,0.04);border-radius:4px;">
<summary style="cursor:pointer;font-weight:600;font-size:0.95em;"><code>labour_mkt_decision</code> — discrete branching</summary>

**Decision mover $\mathbb{B}$** &ensp; (continuation $\to$ decision)

The branching stage receives the arrival values from the two consumption stages: $\mathrm{v}_{\succ}^{\text{work}}(a)$ from `work_cons` and $\mathrm{v}_{\succ}^{\text{retire}}(a)$ from `retire_cons`. Assets $a$ pass through unchanged (identity transitions). The decision mover is the discrete-choice $\max$:

$$(\mathbb{B}\mathrm{v}_{\succ})(a) = \mathrm{v}(a) = \max\!\bigl(\mathrm{v}_{\succ}^{\text{work}}(a) - \delta,\;\; \mathrm{v}_{\succ}^{\text{retire}}(a)\bigr)$$

**Arrival mover $\mathbb{I}$** &ensp; (decision $\to$ arrival)

Identity: $(\mathbb{I}\mathrm{v})(a) = \mathrm{v}(a)$.

</details>

---

> **Sequential form.** &ensp; Composing the three stage operators and substituting the transitions recovers the traditional sequential recursive Bellman equations. Writing $V_t^1$ for the worker's arrival value and $V_t^0$ for the retiree's:
>
> $$V_t^1(a) = \max_{d}\; Q_t^d(a), \qquad Q_t^{\text{work}}(a) = \max_c \bigl\{ \log(c) - \delta + \beta\, V_{t+1}^1\bigl((1{+}r)a + y - c\bigr) \bigr\}$$
>
> $$Q_t^{\text{retire}}(a) = \max_c \bigl\{ \log(c) + \beta\, V_{t+1}^0\bigl((1{+}r)a - c\bigr) \bigr\}, \qquad V_t^0(a) = \max_c \bigl\{ \log(c) + \beta\, V_{t+1}^0\bigl((1{+}r)a - c\bigr) \bigr\}$$
>
>