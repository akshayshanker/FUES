# Example Outputs

This directory structure is designed to generate outputs when you run the examples. 
Generated files (images, results, etc.) are excluded from version control to keep 
the repository clean and encourage reproducibility.

## Directory Structure

When you run the examples, the following directories will be created:

```
examples/
├── housing_renting/
│   ├── images/           # Generated plots (excluded from git)
│   ├── results/          # Numerical results (excluded from git)
│   └── ...
├── retirement/
│   ├── results/
│   │   └── plots/        # Generated plots (excluded from git)
│   └── ...
└── durables/
    └── results/          # Generated outputs (excluded from git)
```

## Running Examples

### Retirement

From the repo root:

```bash
# Baseline single run (solve + plots)
PYTHONPATH=".:src" python examples/retirement/run_experiment.py \
    --grid-size 3000 --output-dir results/retirement

# With a specific params file
PYTHONPATH=".:src" python examples/retirement/run_experiment.py \
    --params params/sigma05.yml --grid-size 3000 --output-dir results/retirement

# Full timing sweep (FUES vs DCEGM vs RFC vs CONSAV)
PYTHONPATH=".:src" python examples/retirement/run_experiment.py \
    --run-timings --sweep-grids 500,1000,2000,3000,10000 \
    --sweep-deltas 0.25,0.5,1,2 --output-dir results/retirement
```

All commands assume you run from the repo root (`FUES/`). The `--output-dir` path is relative to cwd.

Outputs go to `--output-dir`:
- `plots/` — consumption policy, EGM grids, DCEGM comparison (PNG)
- `tables/` — timing and accuracy tables (LaTeX + Markdown, when `--run-timings`)

Parameter files live in `examples/retirement/params/` (baseline, sigma05, high_beta, low_delta, long_horizon).

### Housing/Renting

```bash
cd examples/housing_renting
python solve_runner.py --method FUES
```

### Durables

```bash
cd examples/durables
python durables_plot.py
```

## Sample Outputs

For sample outputs and expected results, please see:
- Documentation: [docs/examples_output_guide.md](../docs/examples_output_guide.md)
- Online gallery: [link to project website/gallery if available]

## Why Outputs Are Excluded

1. **Reproducibility**: Ensures users can verify results by running code
2. **Repository Size**: Keeps the repo lightweight and fast to clone
3. **Version Control**: Avoids binary file conflicts and large diffs
4. **Flexibility**: Users can generate outputs with their own parameters

## Sharing Results

If you need to share results:
1. Use the project's data repository (if available)
2. Upload to a service like Zenodo or Figshare
3. Include in supplementary materials for publications