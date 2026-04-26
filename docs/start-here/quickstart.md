# Quickstart

This page lists the minimal commands required to run the benchmark examples from
a fresh checkout.

## 1. Clone and set up the environment

From the repo root, use the project setup script:

```bash
git clone https://github.com/akshayshanker/FUES.git
cd FUES
source setup/setup.sh
```

This creates or reuses the project virtual environment, installs the example
dependencies, and activates the environment in your current shell.

## 2. Run one example solve

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

## 3. Know where outputs go

Runs write dated output folders under the model-specific results directory, for
example:

```text
results/retirement/YYYY-MM-DD/NNN/
results/durables/YYYY-MM-DD/NNN/
```

These folders contain plots, tables, and summaries for the run.

## 4. Related pages

- [How FUES Works](../algorithm/fues-algorithm.md) for the algorithm.
- [Applications](../examples/index.md) for the benchmark model pages.
- [Transparent EGM / FUES](../notebooks/egm_fues_transparent.ipynb) for a stripped-down notebook walkthrough.
- [Running Locally](../running-locally.md) and [Running on PBS / Gadi](../running-on-gadi.md) for sweeps and cluster jobs.
