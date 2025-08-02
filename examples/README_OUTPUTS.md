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

To generate the outputs:

```bash
# Housing/Renting example
cd examples/housing_renting
python solve_runner.py --method FUES

# Retirement example  
cd examples/retirement
python retirement_plot.py

# Durables example
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