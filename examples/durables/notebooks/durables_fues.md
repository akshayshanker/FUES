---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Durable and non-durable assets with adjustment frictions: EGM(FUES) vs NEGM(FUES)

**Paper**: Dobrescu and Shanker (2022), [*A fast upper envelope scan method for discrete-continuous dynamic programming*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4181302)

We solve a lifecycle model (Application 2 in Dobrescu and Shanker, 2022) in which an agent holds a liquid financial asset and an illiquid durable asset subject to adjustment frictions. The durable provides flow utility — it can be thought of as housing. Each period the agent decides whether to adjust the durable stock, then chooses non-durable consumption and savings. We compare two solution methods:

- **EGM(FUES)** inverts the Euler equations for both the adjuster and the keeper. The keeper's problem is a standard 1D discrete-continuous EGM with FUES. For the adjuster, each durable choice $H_{\succ}$ maps to multiple continuation assets $a_{\succ}$ via root-finding on the housing FOC and analytical inversion of the consumption FOC ([Dobrescu and Shanker, 2022](https://doi.org/10.2139/ssrn.4181660)).
- **NEGM(FUES)** nests the keeper's 1D EGM solution inside a golden-section search over the adjuster's durable choice $H_{\succ}$ ([Druedahl, 2021](https://doi.org/10.1016/j.jedc.2021.104107)).

Both methods use FUES for the keeper's upper envelope; they differ in how the **adjuster stage** recovers the housing policy.

| Method | Adjuster stage | UE (keeper) | UE (adjuster) |
|--------|----------------|:-----------:|:-------------:|
| **EGM(FUES)** | EGM: find all roots of housing FOC, invert consumption FOC analytically | FUES | FUES |
| **NEGM(FUES)** | Golden-section search over $H_{\succ}$, with $a_{\succ}$ from keeper's optimal policy (via EGM + FUES) | FUES | — |

The keeper stage and the tenure stage (discrete adjust-or-keep choice + income shock) are identical under both methods. The comparison isolates the accuracy benefit of EGM inversion for the adjuster, where non-monotone policies and multiple locally optimal candidates create a cloud of crossing value-function segments. MSS ([Iskhakov et al., 2017](https://doi.org/10.3982/QE643)) requires monotone segments separated by identifiable crossing points. LTM ([Druedahl, 2021](https://doi.org/10.1016/j.jedc.2021.104107)) requires the endogenous-to-exogenous mapping to be one-to-one within each segment. FUES imposes neither requirement.

```{code-cell} ipython3
import warnings
warnings.filterwarnings('ignore', message='.*IProgress.*')

import numpy as np
import sys, os, time

REPO_ROOT = os.path.abspath('../../..')
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
os.environ['FUES_RETURN_GRIDS'] = '1'

from examples.durables.solve import solve
from examples.durables.horses.simulate import (
    simulate_lifecycle, evaluate_euler_c, evaluate_euler_h,
)
from examples.durables.outputs import (
    setup_nb_style,
    FilteredStdout, get_timing,
    print_solve_summary, build_comparison_row,
    nb_plot_adjuster_comparison, nb_plot_adjuster_egm,
    nb_plot_keeper_egm, nb_plot_keeper_policy,
    plot_euler_histogram, plot_lifecycle,
    compute_euler_stats,
    generate_vertical_comparison,
    generate_cohort_table,
)
from IPython.display import Markdown

setup_nb_style()
SYNTAX = 'examples/durables/mod/cobb_douglas'
print('Ready')
```

## 2. Solve

We use the Cobb-Douglas utility specification $\mathrm{u}(c, H_{\succ}) = \bigl(c^\alpha\,(H_{\succ} + \bar{d})^{1-\alpha}\bigr)^{1-\rho}\!/(1{-}\rho)$ following [Druedahl (2021)](https://doi.org/10.1016/j.jedc.2021.104107). The income process is AR(1) in logs: $\log z_{\succ} = \rho_z \log z + \sigma_z\,\varepsilon_{\succ}$, $\varepsilon_{\succ} \sim N(0,1)$, discretised to `N_wage` nodes via Tauchen's method. Parameters and grid settings are loaded from YAML; `setting_overrides` and `calib_overrides` edit these at runtime before instantiating a solvable model.

The syntax `_mo = {('adjuster_cons', 'cntn_to_dcsn_mover', 'upper_envelope'): method}` is used to pass the `upper_envelope` method name to the `cntn_to_dcsn_mover` of the `adjuster_cons` stage each period. Parameters and settings are applied "globally" across all stages in the model.  

```{code-cell} ipython3
# Parameters and settings
OVERRIDES = dict(
    setting_overrides={
        'store_cntn': 1, 'N_wage': 4,
        'n_w': 600, 'n_a': 600, 'n_h': 600, 'T': 70,
    },
    calib_overrides={
        't0': 20, 'tau': 0.07,
        'sigma_w': 0.11, 'phi_w': 0.86,
        'beta': 0.89, 'R': 1.04, 'rho': 2.5,
    },
)

labels = {'FUES': 'EGM(FUES)', 'NEGM': 'NEGM(FUES)'}
results = {}
real_stdout = sys.stdout

# Solve under each method
for method in ['FUES', 'NEGM']:
    print(f'{labels[method]}', flush=True)
    t0 = time.time()
    sys.stdout = FilteredStdout(real_stdout)
    try:
        _mo = {
            ('adjuster_cons',
             'cntn_to_dcsn_mover',
             'upper_envelope'): method,
        }
        nest, grids = solve(
            SYNTAX, method_overrides=_mo,
            verbose=False, progress='bar',
            **OVERRIDES)
    finally:
        sys.stdout = real_stdout
    elapsed = time.time() - t0
    timing = get_timing(nest)
    results[method] = {
        'nest': nest, 'grids': grids,
        'timing': timing, 'elapsed': elapsed,
    }

print_solve_summary(results)
```

## 3. Adjuster policies

We now plot adjuster financial assets $a_{\succ}$ and housing choice $H_{\succ}$ as functions of market resources $w_{\text{adj}}$ for the adjuster. First start with the model solved with EGM and FUES for both the adjuster and the keeper.

```{code-cell} ipython3
fig = nb_plot_adjuster_comparison(
    results, results['FUES']['grids'],
    plot_t=[35, 55, 65], methods_filter=['FUES'], xlim =12)
```

Now we use nested EGM (with FUES for the keeper problem only). Golden-section search replaces the EGM in the adjuster stage.

```{code-cell} ipython3
fig = nb_plot_adjuster_comparison(
    results, results['NEGM']['grids'],
    plot_t=[35, 55, 65], methods_filter=['NEGM'], xlim =12)
```

## 4. Adjuster EGM grid

The EGM over the $H_{\succ}$ grid produces an endogenous wealth grid where **multiple housing choices map to the same wealth level**. The left panel shows raw EGM candidates for financial assets $a_{\succ}$; the middle panel shows housing $H_{\succ}$; the right panel shows the value correspondence $\mathrm{v}(\hat{m})$. Each scatter point is one $(H_{\succ}, a_{\succ})$ root of the coupled FOCs. FUES scans this dense cloud to recover the upper envelope in a single pass. MSS and LTM require locally isolated points on the exogenous grid to separate unique segments before interpolating over each of them; because of multiple endogenous grid for each exogenous grid point, neither method can identify these segments.

```{code-cell} ipython3
fig = nb_plot_adjuster_egm(
    results['FUES']['nest'],
    results['FUES']['grids'], plot_t=35, xlim =8)
```

Interactive versions of the EGM grid (zoom/pan to inspect dense crossing regions). Requires Plotly; skipped gracefully in static environments.

```{code-cell} ipython3
# ── Interactive plotly (zoom/pan to inspect dense crossings) ──
try:
    from examples.durables.outputs import nb_plot_adjuster_egm_interactive
    fig_a, fig_h, fig_v = nb_plot_adjuster_egm_interactive(
        results['FUES']['nest'], results['FUES']['grids'], plot_t=35)
    fig_a.show()
    fig_h.show()
    fig_v.show()
except Exception as e:
    print(f'Interactive plots skipped: {e}')
```

## 5. Keeper policies

Keeper consumption $c$ and savings $a_{\succ}$ as functions of cash-on-hand $w_{\mathrm{kp}}$, for a fixed housing level $H$. The keeper has a single control (consumption); housing passes through unchanged. (Recall that for the keeper, MSS *and* LTM can be applied to recover the upper envelope since each exogenous grid point maps to a unique endogenous grid point *and* the continuation-state policy only jumps  upwards)

```{code-cell} ipython3
fig = nb_plot_keeper_policy(results, results['FUES']['grids'],
                           plot_t=[35, 55, 65], methods_filter=['FUES'])
```

NEGM(FUES) keeper — same 1D EGM + FUES kernel, so policies should be nearly identical to EGM(FUES).

```{code-cell} ipython3
fig = nb_plot_keeper_policy(results, results['NEGM']['grids'],
                           plot_t=[35, 55, 65], methods_filter=['NEGM'])
```

## 6. Euler equation errors

We forward-simulate 10,000 agents and evaluate two Euler residuals — one for the liquid asset and one for the durable — measured as $\log_{10}$ relative error ($-4$ = four digits of accuracy). We assume agents start with an average of $\$ 50,000$ in cash-on-hand.

**Consumption FOC** (all agents): $\partial_c\,\mathrm{u}(c, H_{\succ}) = \partial_a\,\mathrm{v}_{\succ}(a_{\succ}, H_{\succ})$, where $\mathrm{v}_{\succ}$ is the continuation value. Since the keeper stage is identical under both methods, keeper errors are similar.

**Housing FOC** (adjusters only): $(1{+}\tau)\,\partial_c\,\mathrm{u}(c, H_{\succ}) = \partial_H\,\mathrm{v}_{\succ}(a_{\succ}, H_{\succ})$. The factor $(1+\tau)$ reflects the proportional adjustment cost. This is where the methods diverge: EGM(FUES) inverts this FOC via root-finding and applies FUES to the resulting candidates; NEGM(FUES) maximises the adjuster objective numerically via golden-section search, nesting the keeper policy.

```{code-cell} ipython3
euler_results = {}

for method in ['FUES', 'NEGM']:
    r = results[method]
    sim_data = simulate_lifecycle(
        r['nest'], r['grids'],
        N=10_000, seed=41, init_dispersion=0.11)
    euler_c = evaluate_euler_c(
        sim_data, r['nest'], r['grids'])
    euler_h = evaluate_euler_h(
        sim_data, r['nest'], r['grids'])
    euler_results[method] = {
        'euler_c': euler_c,
        'euler_h': euler_h,
        'euler': euler_c,
        'stats_c': compute_euler_stats(
            euler_c, sim_data['discrete']),
        'stats_h': compute_euler_stats(
            euler_h, sim_data['discrete']),
        'stats': compute_euler_stats(
            euler_c, sim_data['discrete']),
        'sim_data': sim_data,
    }

print('Simulation complete (N = 10,000)')
```

Left: keeper consumption FOC (methods nearly identical). Right: adjuster housing FOC, where EGM(FUES) gains roughly between one and two orders of magnitude in accuracy over NEGM(FUES).

```{code-cell} ipython3
fig = plot_euler_histogram(euler_results)
```

### Welfare distribution

Per-agent certainty-equivalent utility at the start of the lifecycle, computed from the discounted NPV of flow utilities over the lifecycle.

```{code-cell} ipython3
# Per-agent CE utility histogram
import matplotlib.pyplot as plt

_st = results['FUES']['nest']['periods'][0][
    'stages']['keeper_cons']
_rho = float(_st.calibration.get(
    'gamma_c', _st.calibration.get('rho', 2.0)))
_norm = 1.0 / float(_st.settings['normalisation'])
_labels = {'FUES': 'EGM(FUES)', 'NEGM': 'NEGM(FUES)'}
_colors = {'FUES': '#4361ee', 'NEGM': '#e07c3e'}

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for method in ['FUES', 'NEGM']:
    npv = euler_results[method]['sim_data'][
        'npv_utility']
    valid = npv[np.isfinite(npv) & (npv != 0)]
    inner = (1.0 - _rho) * valid
    ce = np.where(
        inner > 0,
        inner ** (1.0 / (1.0 - _rho)),
        np.nan) * _norm
    ce = ce[np.isfinite(ce)]
    ax.hist(
        ce, bins=80, alpha=0.5,
        label=_labels[method],
        color=_colors[method])
    ax.axvline(
        np.median(ce), color=_colors[method],
        ls='--', lw=1.2,
        label=f'{_labels[method]} median:'
              f' {np.median(ce):,.0f}')

ax.set_xlabel('CE utility (per agent)')
ax.set_ylabel('Count')
ax.set_title(
    'Per-agent certainty-equivalent utility')
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
```

## 7. Comparison of errors and aggregates

The advantage of EGM FUES in the adjuster problem here is one to two orders of magnitude in accuracy over NEGM(FUES). Because FUES is already used for the keeper problem, the overall improvement in speed is modest (and will depend on the choice of numerical optimization used in the adjuster stage).

```{code-cell} ipython3
rows = [build_comparison_row(m, results, euler_results)
        for m in ['FUES', 'NEGM']]
Markdown(generate_vertical_comparison(
    rows, caption='EGM(FUES) vs NEGM(FUES)'))
```

## 8. Lifecycle profiles

Mean consumption, financial assets, and housing over the lifecycle. Both methods produce nearly identical aggregate profiles.

+++

The difference between EGM and NEGM is more substantial towards the end of the lifecycle, where adjustment rates are higher and secondary kinks are stronger. 

```{code-cell} ipython3
from IPython.display import Image, display

for method in ['FUES', 'NEGM']:
    er = euler_results[method]
    r = results[method]
    plot_lifecycle(er['sim_data'], er['euler'], r['nest'],
                   output_dir=f'_nb_plots/{method}')
    path = f'_nb_plots/{method}/simulation/lifecycle.png'
    if os.path.exists(path):
        print(f'{labels[method]}')
        display(Image(path))
```

Simulation moments by 5-year age cohort (mean and SD of consumption, financial assets, and housing, in dollars).

```{code-cell} ipython3
_st0 = results['FUES']['nest']['periods'][0]['stages']['keeper_cons']
_t0 = int(_st0.calibration['t0'])
_T = int(_st0.settings['T'])
_norm = 1.0 / float(_st0.settings['normalisation'])
labels = {'FUES': 'EGM(FUES)', 'NEGM': 'NEGM(FUES)'}

for method in ['FUES', 'NEGM']:
    sd = euler_results[method]['sim_data']
    tbl = generate_cohort_table(sd, _t0, _T, _norm)
    display(Markdown(f'**{labels[method]}**\n\n{tbl}'))
```

---

*Source: `examples/durables/` — Dobrescu and Shanker (2026), Application 2*
