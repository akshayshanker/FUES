# New estimation round

Create a new estimation round: YAML spec(s) + PBS script(s) + qsub commands.

## Instructions for the AI

Read the reference estimation YAML and PBS script listed below. Create new
files that match the reference but with the changes specified. Follow the
project convention: each estimation round = one YAML + one PBS script.

- YAMLs go in: `examples/durables2_0/mod/{syntax}/estimation/`
- PBS scripts go in: `experiments/durables/estimation/`
- Name convention: `{name}_{method}.yaml` and `run_{name}_{method}.pbs`
- Copy all moment specification, age groups, targets from the reference
- Only change what is listed under "Changes from reference"

After creating the files, print the qsub commands.

---

## Fill out below

### Reference

```
Reference YAML: examples/durables2_0/mod/separable/estimation/baseline_large_egm.yaml
Reference PBS:  experiments/durables/estimation/run_large_egm.pbs
```

### New round name

```
Name: _______________
```

### Methods (create one YAML + PBS per method)

```
- [ ] EGM   (default, no --method flag)
- [ ] NEGM  (--method NEGM)
```

### Changes from reference

<!-- List ONLY what differs from the reference. Everything else is copied. -->
<!-- Delete any sections you don't need. Add more as needed. -->

**CE options** (method_options):
```yaml
# Examples — delete lines you don't change:
n_samples: ___
n_elite: ___
max_iter: ___
tol: ___
noise_fraction: ___
```

**Grid**:
```
GRID: ___
```

**Free parameters** (if different from reference):
```yaml
# Only list if adding/removing/changing bounds. Otherwise inherits from reference.
```

**Sweep** (if any):
```yaml
# Example:
# sweep:
#   sigma_w: [0.05, 0.10, 0.15, 0.20]
```

**Data source**:
```
- [ ] precomputed
- [ ] selfgen      (recovery test from YAML calibration)
```

<!-- If precomputed, specify the data file and its location.
     The file must exist in the mod's estimation/ directory.
     If it doesn't exist, copy it there or give the source path. -->

**Data file** (precomputed only):
```
File:   moments_data.csv
Source: examples/durables2_0/mod/separable/estimation/moments_data.csv
Notes:  ___  (e.g. "Eggsandbaskets wave 14 females", "custom moments from ...")
```

<!-- Data file format: wide CSV, first column = age group index (1-9),
     remaining columns = moment names. Values in AUD for levels,
     dimensionless for correlations/autocorrelations. NAs allowed.
     The estimation code denormalises model moments to AUD for comparison. -->

**Calibration overrides** (--calib-override in PBS):
```
t0=20
```

**N_sim**:
```
N_SIM: ___
```

### PBS resources

```
Queue:    normalsr
Nodes:    ___
NCPUs:    ___  (nodes x 104)
Memory:   ___  GB
Walltime: 05:00:00
```

### Output paths

```
Scratch: /scratch/tp66/{user}/durables_est
Results: /g/data/tp66/results/durables2_0/estimation
```

### Syntax directory

```
Mod: mod/separable
```

---

## Expected output

1. Created files:
   - `examples/durables2_0/mod/{syntax}/estimation/{name}_{method}.yaml`
   - `experiments/durables/estimation/run_{name}_{method}.pbs`
   (one pair per method checked above)

2. Print the qsub commands:
   ```
   qsub experiments/durables/estimation/run_{name}_egm.pbs
   qsub experiments/durables/estimation/run_{name}_negm.pbs
   ```

3. Summary table:
   | File | Draws | Grid | Elite | Nodes | Data | Sweep |
   |------|-------|------|-------|-------|------|-------|
