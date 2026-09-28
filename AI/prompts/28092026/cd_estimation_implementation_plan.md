# Cobb-Douglas estimation, beta types, post-estimation tools: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Each task below is executed by a fresh `econ-model-maker` (or `generalPurpose` for scripts and docs) subagent that reads the spec and this plan, works only in the files its task lists, writes its tests first and sees them fail, and returns the eleven-heading report of the `econ-model-maker` skill. Agents do not commit; the orchestrator reviews and commits after each wave. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the durables SMM estimation runnable for the Cobb-Douglas registry, add permanent discount-factor types (K equiprobable quantiles, `beta_bar`/`sigma_beta`, one MPI rank per type inside a group per candidate), and add the lifecycle-at-estimates tool and the estimation tables, without changing any result that exists today.

**Architecture:** Pure functions for the type distribution and panel pooling (`solvers/beta_types.py`); a per-type subset simulator in `solvers/simulate.py`; one MPI-orchestration module beside the driver (`type_groups.py`); small, guarded additions to `estimate.py`; two post-processing modules that read the results folder and re-solve at the estimates. Composition stays visible: `birth draw → K × simulate_type_subset → pool_by_type`; `roots → estimate(); workers → worker_loop()`.

**Tech stack:** Python 3.12, numpy, scipy, numba (unchanged), quantecon not needed for types, mpi4py + Open MPI 5.0.7 for the MPI tests, pytest, jupytext.

**Spec:** `/Users/akshayshanker/Research/Repos/FUES2026/FUES/AI/devspecs/28092026/cd_estimation_and_postprocess.md` (read it first; this plan argues from it).

## Global constraints (from the spec; every task inherits them)

- Work only in the worktree `/Users/akshayshanker/Research/Repos/FUES2026/FUES-wt-cd-estimation`, branch `feat/cd-estimation-postprocess`. Never touch `/Users/akshayshanker/Research/Repos/FUES2026/FUES` (the main tree) except to read `AI/`.
- Interpreter: `/Users/akshayshanker/Research/Repos/FUES2026/FUES/.venv/bin/python` (the main tree's venv; `dcsmm` is installed editable from the main tree's `src/`, which nobody changes; `examples` resolves from the worktree because commands run from the worktree root).
- Environment for every command: `NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg NUMBA_CACHE_DIR=/tmp/numba-cache-<task letter>`.
- Do not edit: `settings.yaml`, `settings/default.yaml`, `calibration/main.yaml` in either registry; anything under `src/dcsmm`; `examples/housing_renting`; kikku; any existing PBS script; any existing separable estimation spec; `run.py`; the existing notebooks.
- Everything that runs today keeps running unchanged (spec 3.1). A spec without `types` takes today's code path exactly.
- Read every name you use from the source (`rg`, reading the installed kikku in the venv); never from memory. Rename history is in the main tree's `CLAUDE.md`.
- New test files need `git add -f` (the orchestrator does this); never add `AI/`, `results/`, `_nb_plots/`.
- Register: plain English in comments, docstrings and docs; no colloquialisms.
- Do not commit. Do not push. Report per the `econ-model-maker` skill's eleven headings.

## Review focus (uncovered inputs most likely to bite; each has a pinned test in the owning task)

1. A candidate whose solve raises on a **root** rank inside a group must still reach the gather (Task E2, test J-b).
2. A `types` spec run on a communicator whose size is not a multiple of `n_points × K` must fail on every rank together, before any `Split` (Task E2).
3. `sigma_beta = 0.0` exactly (a clipped candidate) must give K identical types, not NaN (Task E1, test G).
4. Runs finished before this branch have no `manifest.json` in results: the lifecycle tool must take grid, `t0`, `N_sim`, seed, spec factory as arguments and print what it used (Task B).
5. `summary.json` from a run with types must carry `beta_types`; a run without must not, and the tables must render both (Tasks E2, C).

---

## Wave 0 (orchestrator): worktree, baseline, golden references, mpi4py

- [ ] `git -C FUES worktree add ../FUES-wt-cd-estimation -b feat/cd-estimation-postprocess main`
- [ ] Baseline: `pytest tests/ -q -p no:cacheprovider` in the worktree → 29 passed, 16 subtests.
- [ ] Golden digest for check H (before `simulate.py` changes): for both registries, at `n_a=n_h=n_w=30`, `t0=55`, `N=200`, `seed=99`, compute sha256 of `np.ascontiguousarray(sim_data[k]).tobytes()` for each key in sorted order plus dtypes; save to `/tmp/fues-verify/golden_sim_digest.json`. Task E1 copies the literals into its test.
- [ ] Check-L reference: run the driver on the base commit with `/tmp/fues-verify/checkL_spec.yaml` (copy of `separable/estimation/baseline_large_egm.yaml` with `n_samples: 6`, `n_elite: 2`, `max_iter: 2`, absolute `data_file`), `--settings-override n_a=30 n_h=30 n_w=30 --params-override t0=20 --N-sim 500 --scratch /tmp/fues-verify/checkL_main/scratch --results /tmp/fues-verify/checkL_main/results --local-results none`; keep the outputs.
- [ ] `pip install mpi4py` into the venv (background), verify `mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"`.

## Wave 1 (parallel): Tasks A, A2, E1, D, P

### Task A: Cobb-Douglas estimation files and the spec-consistency test

**Files:**
- Create (generated): `examples/durables/syntax/cobb_douglas/estimation/{baseline,baseline_large_egm,baseline_large_negm,baseline_large_egm_males,baseline_large_negm_males,baseline_xlarge_egm,baseline_xlarge_negm,baseline_xlarge_egm_males}.yaml` (the first replaces the stale file)
- Replace: `examples/durables/syntax/cobb_douglas/estimation/moments_data.csv` with a byte copy of `examples/durables/syntax/separable/estimation/moments_data.csv`
- Delete: `examples/durables/syntax/cobb_douglas/estimation/moments.csv`
- Create: `examples/durables/syntax/cobb_douglas/calibration/main_males.yaml` (copy of the separable file; header comment says Cobb-Douglas), `examples/durables/syntax/cobb_douglas/spec_factory_males.yaml` (copy; header says Cobb-Douglas)
- Create: `tests/test_estimation_specs.py`
- Generator script: `/tmp/fues-verify/gen_cd_specs.py` (outside the repo)

**Interfaces:** none consumed; produces the eight spec files that Tasks E2, B, C and P name.

- [ ] Step 1: write `tests/test_estimation_specs.py` (fails on the current tree because of the stale `gamma_c`):

```python
"""Estimation specs agree with the calibration the solver actually uses."""
import os, sys
from pathlib import Path
import pytest, yaml
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from kikku.run.estimate import load_estimation_spec
from examples.durables.solve import load_spec, make_spec   # read solve.py for the exact names

SYNTAX = REPO / "examples" / "durables" / "syntax"
CASES = [(reg, spec) for reg in sorted(p for p in SYNTAX.iterdir() if p.is_dir())
         for spec in sorted((reg / "estimation").glob("*.yaml"))]

def resolved_calibration_keys(registry, spec_factory_name):
    recipe = load_spec(str(registry / spec_factory_name))
    spec = make_spec(recipe, registry_dir=str(registry))
    # read examples/durables/solve.py to see how the per-stage calibration is reached from `spec`
    ...  # return the set of calibration keys of the first stage
    
def types_block(raw):
    return (raw.get("estimation") or {}).get("types") or {}

@pytest.mark.parametrize("registry,spec_path", CASES, ids=lambda p: p.name if isinstance(p, Path) else str(p))
def test_free_parameters_are_model_parameters(registry, spec_path):
    spec = load_estimation_spec(str(spec_path))
    raw = yaml.safe_load(spec_path.read_text())
    factory = "spec_factory_males.yaml" if spec_path.stem.endswith("_males") else "spec_factory.yaml"
    keys = resolved_calibration_keys(registry, factory)
    mapped = set()
    for target, blk in types_block(raw).items():
        assert target in keys and target not in spec["free"]
        mapped |= {blk["location"], blk["spread"]}
        assert not (mapped & keys)
    for name in spec["free"]:
        if name in mapped: continue
        assert name in keys, f"{spec_path.name}: {name} is not a parameter of {registry.name}"

@pytest.mark.parametrize("registry,spec_path", CASES, ids=...)
def test_targets_exist_in_data(registry, spec_path):
    spec = load_estimation_spec(str(spec_path))
    if spec["moment_spec"].get("data_source") != "precomputed": pytest.skip("no data file")
    keys = set(spec["data_moments"])
    for t in spec["moment_spec"].get("targets") or []:
        assert any(k == t["key"] or k.startswith(t["key"] + "__") for k in keys), t["key"]

def test_cobb_douglas_uses_rho():
    for p in (SYNTAX / "cobb_douglas" / "estimation").glob("baseline*.yaml"):
        free = load_estimation_spec(str(p))["free"]
        assert "rho" in free and "gamma_c" not in free and "gamma_h" not in free and "theta" in free, p.name
```

- [ ] Step 2: run it → FAIL on `cobb_douglas/estimation/baseline.yaml` (`gamma_c`).
- [ ] Step 3: write `/tmp/fues-verify/gen_cd_specs.py`: for each of the eight separable files, read the text; in the `free:` block drop the `gamma_c` and `gamma_h` entries, insert `rho:\n      bounds: [1.2, 6.0]` after `alpha`, ensure `theta:\n      bounds: [0.5, 3.0]` is present (add after `tau` if absent); rewrite the leading comment block to name the Cobb-Douglas registry, say that `alpha` is the Cobb-Douglas share and `theta` is estimated because the Cobb-Douglas calibration marks it for recalibration; write to the Cobb-Douglas folder. Copy the CSV and the two male-overlay files. Delete `moments.csv`.
- [ ] Step 4: run the test → PASS for all specs in both registries; run `pytest tests/ -q` → 29 + new pass.
- [ ] Step 5: one criterion evaluation on `cobb_douglas` with `baseline_large_egm_males.yaml` and `spec_factory_males.yaml` at 30³, `t0=20`, `N=500` (reuse the pattern of `/tmp/fues-verify/cd_dryrun.py`) → finite loss, 0/70 NaN. Quote the numbers in the report.

### Task A2: driver changes in `examples/durables/estimate.py`

**Files:** Modify `examples/durables/estimate.py`. Create `tests/test_cd_estimation_criterion.py`. Read first: `estimate.py` in full, `solve.py` lines 520–640, kikku `run/estimate.py` (`_safe_criterion`, `make_criterion`, `BIG_LOSS`, `_cross_entropy_minimize` options), kikku `run/mpi.py`.

**Interfaces produced (Task E2 depends on these exact names):**
```python
def build_criterion(mod_dir, spec, spec_factory, solver_method, calib_overrides,
                    setting_overrides, N_sim, simulation_seed, comm, *,
                    types_spec=None, group=None):
    """Return (criterion, moment_fn, data_moments, denorm). Without types this is
    today's trial: solve → simulate_lifecycle → panels. `types_spec` and `group`
    are accepted now and used by Task E2; with types_spec given and group None the
    trial solves the K types in sequence (serial mode)."""
def check_parameter_names(resolved_calibration_keys, free_names, calib_overrides, types_spec) -> None:
    """Raise ValueError listing offending names (spec 5.2 item 3)."""
def load_types_spec(est_yaml) -> dict | None
```

- [ ] Step 1: tests first, in `tests/test_cd_estimation_criterion.py`: (a) `build_criterion` on `cobb_douglas`, `baseline_large_egm.yaml`, female and male factories, 30³, `t0=20`, `N=500`: loss finite, no NaN target; (b) moments at `rho=2.0` vs `rho=3.0` differ by more than 1e-3 (call the criterion's trial + moment_fn directly); (c) `check_parameter_names` raises `ValueError` naming `gamma_c` when it is in `calib_overrides` for `cobb_douglas`, and passes for `separable`; (d) `load_types_spec` returns None for the existing specs; (e) refusal: a spec dict with `types` and `data_source: selfgen` makes the start-up validation raise with both keys in the message. Run → FAIL (names undefined).
- [ ] Step 2: implement: factor `build_criterion` out of `_run_single_estimation` (behaviour-preserving; the closure `_last_nest` handling stays inside); add the guard (resolve the calibration through `load_spec`/`make_spec` exactly as `solve()` does, for the spec factory in use); `--n-samples` (serial only) and `--n-elite` (both); write `manifest.json` to `results_run` too with `spec_factory`, `solver_method`, `git_commit` (from `git rev-parse HEAD` in the repo root, `None` on failure), `beta_types: None`; failure logging: wrap the trial so any exception is printed (`rank`, `theta`, message, `flush=True`) and counted on `criterion.n_failures` before re-raising; after `estimate()`, on the root, print the total and, when `result.objective >= BIG_LOSS` (import the constant from kikku), print "every evaluation failed" and make the process exit with code 3 on all ranks (broadcast the decision on `comm`); refuse `types` + selfgen at start-up.
- [ ] Step 3: run the new tests → PASS; run `pytest tests/ -q` → all pass.
- [ ] Step 4: end-to-end run C (spec 7-C) on `cobb_douglas`: `python -m examples.durables.estimate --mod syntax/cobb_douglas --spec baseline_large_egm.yaml --settings-override n_a=30 n_h=30 n_w=30 --params-override t0=20 --N-sim 500 --n-samples 6 --n-elite 2 --max-iter 2 --scratch /tmp/fues-verify/runC/scratch --results /tmp/fues-verify/runC/results --local-results none`; copy the `est_*` folder (JSON and CSV only) to `tests/fixtures/estimation_run/`.
- [ ] Step 5: check L: rerun the orchestrator's `/tmp/fues-verify/checkL_spec.yaml` command on the branch (without the new flags) into `/tmp/fues-verify/checkL_branch/`; `convergence.csv` and `theta_best.json` must be byte-identical to `/tmp/fues-verify/checkL_main/`. Quote the diff (empty).

### Task E1: type distribution, subset simulator, pooling

**Files:** Create `examples/durables/solvers/beta_types.py`, `tests/test_beta_types.py`. Modify `examples/durables/solvers/simulate.py` (add `simulate_type_subset`; add `types=None` to `simulate_lifecycle`). Read first: `simulate.py` in full, kikku `asva/simulate.py` (`draw_shocks`, `simulate`), spec 5.6.

**Interfaces produced:**
```python
# beta_types.py
def discretise_beta(beta_bar: float, sigma_beta: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """nodes (increasing, in (0,1)) and shares (all 1/n). x_k = ln(1/beta_bar - 1) + sigma_beta * norm.ppf((k+0.5)/n); beta = 1/(1+exp(x))."""
def expand_types(theta: dict, types_spec: dict) -> list[tuple[float, dict]]:
    """[(share_k, {**theta_without_location_and_spread, 'beta': beta_k})]"""
def draw_types(N: int, shares, seed: int) -> np.ndarray:
    """int64 (N,): rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)));
    u = rng.random(N); idx = np.searchsorted(np.cumsum(shares), u, side='right'); return np.minimum(idx, len(shares)-1)"""
def pool_by_type(parts: list[tuple[np.ndarray, dict]], type_idx: np.ndarray, betas, N: int) -> dict:
    """parts: [(agent_idx, sim_data_subset)] one per type, in type order. Scatter every (T, n_k) array to (T, N)
    and every (n_k,) array to (N,) at agent_idx; add 'beta_type' (N,) int64 and 'beta' (N,) float64."""
# simulate.py
def simulate_type_subset(nest, grids, agent_idx, N, seed, use_empirical_init=False, init_dispersion=0.0, init_gender='male') -> dict:
    """Draw shocks and initial particles for ALL N agents exactly as simulate_lifecycle does, take the subset
    agent_idx, walk it with this nest's policies, return the subset's panels and utility statistics."""
def simulate_lifecycle(nest, grids, N=10000, seed=42, use_empirical_init=False, init_dispersion=0.0,
                       init_gender='male', types=None) -> dict:
    """types: list of (share, nest_k, grids_k). None → today's behaviour, bit for bit."""
```

- [ ] Step 1: tests first (`tests/test_beta_types.py`): G as in spec 7-G with closed-form literals (`z = (-1.1503493804, -0.3186393639, 0.3186393639, 1.1503493804)`); the chi-square independence test between `draw_types` and `make_initial_particles(seed)['z_idx']`, and its deliberate-failure twin using `default_rng(seed+1).random(N)`; `pool_by_type` on hand-made parts including an `(n_k,)` statistic. H: the golden digest literals from `/tmp/fues-verify/golden_sim_digest.json` for both registries; `simulate_lifecycle`, `simulate_lifecycle(types=[(1.0, nest, grids)])` and `pool_by_type([(arange(N), simulate_type_subset(...))], zeros(N), [beta], N)` all reproduce the digest (compare arrays with `np.array_equal(..., equal_nan=True)` and dtypes); `make_initial_particles` equal across two nests differing only in `beta`; K = 2 types differ in consumption and `beta` takes the two node values. Run → FAIL.
- [ ] Step 2: implement `beta_types.py`; implement `simulate_type_subset` by extracting the body of `simulate_lifecycle` so that shocks and particles are drawn for all N and then indexed (read how kikku's `simulate` consumes `draws` and `initial_particles`; slice every per-agent array on axis 1 or 0 as its shape dictates); make `simulate_lifecycle` call it with the full index set when `types is None` **only if that reproduces the digest bit for bit; otherwise keep today's body untouched and add the types branch beside it**.
- [ ] Step 3: run tests → PASS; `pytest tests/ -q` → all pass (`test_durables_simulate.py` unchanged).
- [ ] Step 4: report the chi-square statistics and the digests.

### Task D: pull-script flag

**Files:** Modify `scripts/pull_gadi_results.sh`. Test: a local rehearsal with `GADI_HOST=localhost`-style substitution is not possible; instead add `--dry-run` passthrough to rsync when `DRY_RUN=1`, and rehearse against a local directory laid out like Gadi's via `rsync` from a path (the script's source becomes `${GADI_HOST}:${...}`; with `GADI_HOST=""`... keep it simple: factor the estimation rsync into a function taking a source argument and test the function with a local source).

- [ ] Step 1: add `--estimation` flag: when present, an additional rsync from `${GADI_HOST}:/g/data/tp66/results/durables/estimation/` into `${REPO_ROOT}/paper-results/durables/estimation/raw/` with `--include='*/' --include='*.json' --include='*.csv' --exclude='*'`, no deletion of local files; the default path (no flag) unchanged; `--help` text updated.
- [ ] Step 2: `bash -n`; rehearse the function against `/tmp/fues-verify/fake_gdata/estimation/separable/x/est_1/{summary.json,fit_table.csv,best.nst}` → only the JSON and CSV arrive.

### Task P: PBS scripts for Cobb-Douglas and types, in the committed format

**Files:** Create `benchmarks/durables/estimation/data-estimation/cobb_douglas/females/{run_large_egm,run_large_egm_hugemem,run_large_negm,run_large_negm_hugemem,run_xlarge_egm,run_xlarge_negm,run_xxl_egm}.pbs` and `.../cobb_douglas/males/{run_large_egm_males,run_large_negm_males,run_xxl_egm_males}.pbs` as copies of the separable scripts with `MOD="syntax/cobb_douglas"` and `#PBS -N est_cd_<same suffix>`; create `.../separable/females/run_large_egm_types.pbs` and `.../cobb_douglas/females/run_large_egm_types.pbs` from `run_large_egm.pbs` with `SPEC="baseline_large_egm_types.yaml"`, `#PBS -l ncpus=4160`, `#PBS -l mem=19200GB`, `#PBS -l jobfs=1600GB`, `#PBS -N est_<reg>_lgEGMtypes`, and a comment line stating "4 beta types × 1040 candidates; the driver refuses a core count that is not a multiple of 4". Add rows to `benchmarks/durables/estimation/README.md` tables in the same format. Do not modify any existing script.

- [ ] Step 1: `diff` each new script against its source: only the `MOD`, `SPEC`, `-N`, resource lines and the comment differ. `bash -n` each.

## Wave 2: Task E2 (after A2 and E1)

### Task E2: layered communicator, types in the driver, MPI tests

**Files:** Create `examples/durables/type_groups.py`, `tests/test_type_groups_mpi.py`, `examples/durables/syntax/separable/estimation/baseline_large_egm_types.yaml`, `examples/durables/syntax/cobb_douglas/estimation/baseline_large_egm_types.yaml`. Modify `examples/durables/estimate.py` (types block, layering, `beta_types` in outputs). Read first: spec 5.6 in full, `estimate.py` after Task A2, `beta_types.py`, `simulate.py` after Task E1, kikku `run/estimate.py` and `run/mpi.py`, Eggsandbaskets `AI/temp/Eggsandbaskets3.0/eggsandbaskets/smm.py:76-128` (main tree) for the split pattern.

**Interfaces produced:**
```python
# type_groups.py
EVAL, STOP, FAIL = "EVAL", "STOP", "FAIL"
def check_divisibility(comm, K, n_points) -> None      # raises the same ValueError on every rank
def split_type_groups(comm, K) -> tuple[group_comm, roots_comm | None]   # roots_comm is None on workers
def make_group_trial(group_comm, K, solve_type_k, pool, log) -> callable   # root side: theta -> panels
def worker_loop(group_comm, solve_type_k) -> None                          # workers: until STOP
```
`solve_type_k(theta_k_calib, k, agent_idx)` is built in `estimate.py` from `solve`, `expand_types`, `draw_types` and `simulate_type_subset`. Test hook: if the environment variable `FUES_TYPES_FORCE_FAIL` equals `"rank<r>"`, `"root"`, `"estimate"` or `"all"`, the named failure is raised at the named place (documented in the module docstring as a test hook, nowhere else).

- [ ] Step 1: the two types specs: copies of `baseline_large_egm.yaml` per registry with `beta` replaced by `beta_bar: {bounds: [0.88, 0.99]}` and `sigma_beta: {bounds: [0.0, 0.5]}` in `free:`, a `types: {beta: {n: 4, location: beta_bar, spread: sigma_beta}}` block, and a header stating the node rule ("four types at the normal quantiles 12.5, 37.5, 62.5, 87.5 percent, equal shares; sigma_beta in logit units"). `tests/test_estimation_specs.py` must pass on them.
- [ ] Step 2: tests first (`tests/test_type_groups_mpi.py`, `pytest.importorskip("mpi4py")` with the recorded reason): J as in spec 7-J: the 4-rank run vs the serial `--n-samples 2` run (byte-identical `theta_best.json`, `convergence.csv`; `summary.json` equal after dropping `run_id`-dependent fields); forced failures (a) `rank1`, (b) `root`, (c) `estimate` → exit 0, log line present, `elite_mean_loss_0 == (best_loss_0 + 1e10)/2`; (d) `all` → exit code 3; restart run equality; the 8-rank sweep-with-types run over `tau`; per-rank solve times printed. Each MPI run is a `subprocess.run([mpiexec, "-n", n, python, "-m", "mpi4py", "-m", "examples.durables.estimate", ...], env=...)` from the worktree root with a 20-minute timeout, where `mpiexec` is `/Users/akshayshanker/Research/Repos/FUES2026/FUES/.venv/bin/mpiexec` (arm64 MPICH 5.0.1 installed into the venv; the x86_64 Open MPI at `/usr/local/bin` cannot launch the venv's Python). The test locates it as `Path(sys.executable).parent / "mpiexec"` and skips with a recorded reason when it is absent. Run → FAIL.
- [ ] Step 3: implement `type_groups.py` and wire it in `estimate.py`: parse `types`; refuse with selfgen; `check_divisibility` before any split; in `_run_single_estimation`, after the data-moments broadcast and the resume broadcast, split; roots build the criterion with `build_criterion(..., types_spec=..., group=(group_comm, K))` and call `estimate(comm=roots_comm)` inside `try/finally` that broadcasts `STOP`; workers run `worker_loop` and return `(None, None)`; serial mode solves K types in sequence; `beta_types` (n, nodes, shares, implied mean) written into `summary.json` and `manifest.json`.
- [ ] Step 4: run the MPI tests → PASS; `pytest tests/ -q` → all pass.
- [ ] Step 5: end-to-end run K (spec 7-K) in serial with types on `separable`, copy JSON/CSV to `tests/fixtures/estimation_run_types/`.

## Wave 3 (parallel, after E2): Tasks B, C, Doc

### Task B: lifecycle at the estimates

**Files:** Create `examples/durables/postprocess/at_estimates.py`, `examples/durables/notebooks/lifecycle_at_estimates.ipynb` (build it with jupytext from a `.py` script kept in `/tmp`, then delete the script), `docs/notebooks/lifecycle_at_estimates.ipynb` (relative symlink like the existing four), `tests/test_at_estimates.py`. Modify `mkdocs.yml` (one nav line). Interfaces: exactly spec 5.3; `solve_at_estimates` returns `[(share, nest, grids)]`; with `beta_types` in `summary.json` it re-solves K models via `expand_types`. Tests: spec 7-D on both saved example outputs, with the contributions-sum-to-objective check on each (they were produced at 30³, `t0=20`, `N=500`, seed 99; the manifest records it).

Gadi guard (Tasks B and C alike): if `os.path.isdir("/scratch/tp66")` and no
`--out` was given, exit with a message asking for `--out` under scratch; the
tools never write inside the repository on Gadi (spec 5.7). Test: monkeypatch
`os.path.isdir` and assert `SystemExit`.

### Task C: estimation tables

**Files:** Create `examples/durables/postprocess/estimation_tables.py`, `tests/test_estimation_tables.py`. Interfaces: spec 5.4 minus `sweep_recovery_table`. Row order list at module level: `beta, beta_bar, sigma_beta, alpha, gamma_c, gamma_h, rho, tau, theta`, then any others alphabetically; LaTeX symbols `\beta, \bar\beta, \sigma_\beta, \alpha, \gamma_c, \gamma_h, \rho, \tau, \theta`. Footer for types runs: nodes, shares, implied mean. Tests: spec 7-E. Reuse `_md_table`/`_tex_table` from `tables.py` where they fit; otherwise a small writer in the same style inside the new module.

### Task Doc: documentation and changelog

**Files:** Modify `examples/durables/docs/estimation_outputs.md` (best.nst not written and why; manifest in results; the two tools; fit_table provenance), `examples/durables/README.md`, `docs/running-on-gadi.md` (the new PBS layout, Cobb-Douglas and types scripts), `mkdocs.yml` if Task B did not, `CHANGELOG.md` (Unreleased entry listing every change, including the data-side method defect as a recorded issue). Plain English; verify every command quoted by running it.

## Wave 4 (orchestrator)

- [ ] Whole-branch review by `econ-code-critic`; fix findings.
- [ ] Manual checks L: `jupyter nbconvert --execute` on the two durables notebooks; `python -m examples.retirement.run`; `python -m examples.durables.run --slot-override '$draw.t0=55'`.
- [ ] Session note `AI/working/28092026/session.md` (main tree) and `AI/INDEX.md` link; `AI/todo.md` entries (data-side method defect; age-group four-of-five convention; selfgen with types deferred).
- [ ] Push the branch; report to the owner with the merge instruction and the Gadi commands.
