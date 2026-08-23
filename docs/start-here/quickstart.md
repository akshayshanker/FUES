# Quickstart

To use the upper envelope in your own model, start with the minimal-use
sections below: a single `pip install` provides the FUES algorithm and the
upper-envelope interface `EGM_UE`, without any of the additional
dependencies the applications require.

To run the applications, jump to
[running the applications](#2-clone-and-set-up-the-environment).

## 1. Installing FUES and EGM_UE only

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
    m_bar=1.2,    # jump threshold (approx max marginal propensity to consume)
    LB=4,         # look-back buffer for forward/backward scans
)
```

The returned arrays contain only the upper-envelope points. Convention:
`*_hat` for raw correspondence, `*_ref` for refined upper-envelope objects.
For a cell-by-cell walkthrough of exactly these calls on a small worked
problem — every EGM step written out in raw NumPy — see the
[transparent EGM / FUES notebook](../notebooks/egm_fues_transparent.ipynb).

**Setting `m_bar`.** Use the maximum marginal propensity to consume in
your model. For instance, in a consumption-saving model with log utility
and $\beta R < 1$, set `m_bar` to $1.0 + 10^{-2}$. Higher values also work
and remove fewer points on coarser grids; in the limit, as the grid size
grows, **any `m_bar` above the maximum MPC is guaranteed to recover the
true upper envelope.** Setting `endog_mbar=True` instead lets FUES compute
a grid-local threshold, in which case the derivative of the control must
be supplied as `del_kappa_hat`.

> FUES detects jumps in the control variable `kappa_hat`, not in the
> continuation state `x_cntn_hat`; the continuation series is only
> cleaned, and a refined set of continuation points is returned.

For the full documentation of the `FUES` function, see the
[Core API](../api/fues.md).

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
`FUES` and `RFC` do not. All methods work on a bare install: every
`EGM_UE` engine's dependencies, including `pykdtree` for `RFC`, are core
dependencies.

See the
[durables application](../examples/continuous_housing_model.md) for the
main non-monotone benchmark.

## 2. Clone and install to run applications and edit source

To run the benchmark examples, clone the repository and run the setup
script:

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
source setup/setup.sh
```

On the first run this creates the project virtual environment (`.venv`),
installs `dcsmm` (editable) with the application dependencies and the
pinned dolo-plus compiler, verifies the install, and activates the
environment; on later runs it only activates (`--update` refreshes). The
equivalent manual pip sequence is in
[Installation](../getting-started/installation.md).

The applications run three ways: through their notebooks, from the command
line, and as PBS cluster jobs. The notebooks, in `examples/*/notebooks/` and
rendered on the [docs site](https://akshayshanker.github.io/FUES/), are the
easiest way to experiment once you have cloned the repo; the sections below
cover the command line. Start with the simple EGM/FUES walkthrough notebook: [egm_fues_transparent.ipynb](../notebooks/egm_fues_transparent.ipynb).

## 3. CLI runs of applications

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

Each run creates a dated, auto-numbered folder inside the directory passed
to `--output-dir` — so the two commands above write to:

```text
results/retirement/YYYY-MM-DD/NNN/
results/durables/YYYY-MM-DD/NNN/
```

Omitting `--output-dir` gives the same locations, since `results/<model>/`
is each runner's default.

Details of how to run PBS cluster jobs are in the
[Running on a PBS cluster](../running-on-gadi.md) page.

