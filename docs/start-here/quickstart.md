# Quickstart

To use the upper envelope in your own
model, start with the minimal use sections below. This just installs the FUES algorithm as 
and the upper envelop wrapper `EGM_UE` as a single `pip install`. Without any 
additional dependencies required  by the applications. 

To run the applications, jump to
[running the applications](#3-clone-and-set-up-the-environment).

## 1. Minimal use

Install the library alone:

```bash
pip install git+https://github.com/akshayshanker/FUES.git
```

After your EGM step produces arrays for the raw endogenous-grid
correspondence, pass them to `FUES`:

```python
from dcsmm.fues import FUES

x_dcsn_ref, v_ref, kappa_ref, x_cntn_ref, _ = FUES(
    x_dcsn_hat,   # raw endogenous decision grid (N,) — unsorted is fine
    v_hat,        # raw value correspondence (N,)
    kappa_hat,    # primary control, e.g. consumption (N,)
    x_cntn_hat,   # raw continuation / post-decision state (N,)
    del_kappa_hat,  # jump-detection series: derivative of the control kappa
    m_bar=1.2,    # jump threshold (approx max marginal propensity to save)
    LB=4,         # look-back buffer for forward/backward scans
)
```

The returned arrays contain only the upper-envelope points. Convention:
`*_hat` for raw correspondence, `*_ref` for refined upper-envelope objects.
For a cell-by-cell walkthrough of exactly these calls on a small worked
problem — every EGM step written out in raw NumPy — see the
[transparent EGM / FUES notebook](../notebooks/egm_fues_transparent.ipynb).

**Setting `m_bar`.** Use the maximum marginal propensity to save in your
model. For instance, in a consumption saving model with log utility and
$\beta R < 1$, set `m_bar` to $(1.0 + E{-2})$. Higher values also work and
will remove less points in coarser grids. In the limit, as the grid size
grows, **any `m_bar` above the maximum MPC is guaranteed to recover the
true upper envelope.** Setting `endog_mbar=True` lets FUES compute the
"true" grid-local `m_bar` from `del_kappa_hat`, but one needs to pass
through the derivative of the control variable `del_kappa_hat`.

> Importantly, note that FUES detects jumps in the control variable
> `kappa_hat`, not the continuation state `x_cntn_hat`. The `x_cntn_hat`
> is simply cleaned, and a refined continuation-state set of points is
> returned.

For the full documentation of the `FUES` function, see the [Core API](../api/fues.md).

### 1.1 The upper-envelope registry

To run FUES against MSS, RFC, and LTM on the same raw EGM output, use the
unified `EGM_UE` interface:

```python
from dcsmm.uenvelope import EGM_UE

refined, raw, interpolated = EGM_UE(
    x_dcsn_hat=x_dcsn_hat,
    v_hat=v_hat,
    v_cntn_hat=v_cntn,          # raw continuation value; only used by "CONSAV"
    kappa_hat=kappa_hat,
    x_cntn_hat=x_cntn_hat,
    X_dcsn=X_dcsn,
    uc_func_partial=uc_func,    # u'(c); used to compute lambda_ref
    u_func=u_func,
    method_switch="FUES",       # or "DCEGM", "RFC", "CONSAV"
    m_bar=1.2,
)
```

All methods return the same dict schema. `DCEGM` (MSS in the paper) and
`CONSAV` (LTM in the paper) require a strictly monotone optimal policy;
`FUES` and `RFC` do not. `RFC` needs the optional `pykdtree` package
(`pip install pykdtree`; included in the `[examples]` extra). 

See the
[durables application](../examples/continuous_housing_model.md) for the
main non-monotone benchmark. 

## 2. Clone and set up the environment

To run the benchmark examples, clone the repository and use the project setup script from a fresh
checkout:

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
source setup/setup.sh
```

This creates or reuses the project virtual environment, installs the example
dependencies, and activates the environment in your current shell.

The applications run three ways: through their notebooks, from the command
line, and as PBS cluster jobs. The notebooks, in `examples/*/notebooks/` and
rendered on the [docs site](https://akshayshanker.github.io/FUES/), are the
easiest way to experiment once you have cloned the repo; the sections below
cover the command line.

## 4. Run one example solve

The retirement model is the smaller benchmark and is convenient for a first
run:

```bash
python -m examples.retirement.run \
    --slot-override '$draw.grid_size=3000' \
    --output-dir results/retirement
```

The durables model is the main non-monotone application:

```bash
python -m examples.durables.run \
    --output-dir results/durables
```

## 5. Output locations

Runs write dated output folders under the model-specific results directory, for
example:

```text
results/retirement/YYYY-MM-DD/NNN/
results/durables/YYYY-MM-DD/NNN/
```

These folders contain the run's plots and tables. The durables runner writes `tables/sweep.md` (and `comparison.md` in compare mode); the retirement runner prints its method table to the console and writes plots, filling `tables/` only for timing sweeps.

## 6. Related pages

- [How FUES Works](../algorithm/fues-algorithm.md) for the algorithm.
- [Applications](../examples/index.md) for the benchmark model pages.
- [Transparent EGM / FUES](../notebooks/egm_fues_transparent.ipynb) for a stripped-down notebook walkthrough.
- [Running Locally](../running-locally.md) and [Running on a PBS cluster](../running-on-gadi.md) for sweeps and cluster jobs.
