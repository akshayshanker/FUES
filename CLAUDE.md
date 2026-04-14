# CLAUDE.md — Project Instructions for FUES

## Critical: Avoid rename/refactor breakage

This codebase runs on **two environments** (local Mac + NCI Gadi). Every
rename propagates through imports, CLI entry points, PBS scripts, and
kikku's RunSpec attributes. Before renaming any function, class, or
attribute:

1. **Grep the entire repo** for every reference — not just the file you're
   editing. Include `run.py`, `solve.py`, tests, PBS scripts, notebooks.
2. **Check kikku's actual attribute names** — `RunSpec` now uses
   `model_dir` (the `syntax_dir` alias is deprecated as of kikku's latest
   cli.py). Always verify against the installed kikku source:
   `grep -n 'attr_name' $(python3 -c "import kikku.run.cli; print(kikku.run.cli.__file__)")`
3. **Test the CLI entry point** after any change to run.py:
   `python3 -c "from examples.durables.run import main; print('OK')"`
4. **Push and verify on Gadi** before submitting PBS jobs. Check the Gadi
   mount at `/Users/akshayshanker/gadi/home/141/as3442/dev/fues.dev/FUES/`
   to confirm the file matches.

### Known rename history (do not repeat these mistakes)

| Old name | New name | Where | Notes |
|----------|----------|-------|-------|
| `solve_block` | `solve` | `examples/durables/solve.py` | run.py imported the old name |
| `_solver_config` | `_solve_overrides` | `examples/durables/run.py` | Returns 3-tuple, not dict |
| `run.syntax_dir` | `run.model_dir` | kikku `RunSpec` | kikku reversed course — `syntax_dir` is now the deprecated alias |
| `examples/durables/syntax` | `examples/durables/mod/separable` | directory path | Tests and precompile had stale refs |
| `examples.durables.simulate` | `examples.durables.horses.simulate` | import path | Test file had stale import |

## Settings vs Calibration

- **Settings** (`settings.yaml`): numerical/grid parameters — `b`, `a_min`,
  `h_min`, `w_min`, `n_a`, `n_h`, `n_w`, `fues_*`, `sim_guard`, etc.
- **Calibration** (`calibration.yaml`): economic parameters — `beta`, `R`,
  `tau`, `alpha`, `gamma_c`, etc.

**Read settings from `sett`, calibration from `cal`.** Never read `b` from
calibration — it's in settings. The code default `calibration.get("b", 1e-8)`
is a legacy fallback that should not be relied on.

## Grid construction (`model.py`)

- `a_grid`, `h_grid` use `a_min`, `h_min` from settings (not `b`)
- `we_grid` uses `w_min` from settings with auto-floor `R*a_min + R_H*h_min`
- `grid_phi > 1` enables non-uniform grids (denser at the bottom)
- `UGgrid_all` (UCGrid) is kept for backward compat but NOT used for
  interpolation — all 2D interp uses `consav.linear_interp.interp_2d` or
  `interp2d_nonuniform`

## Interpolation

- **1D**: `interp_as`, `interp_as_scalar`, `interp_as_3` from `dcsmm.fues.helpers.math_funcs`
- **2D**: `consav.linear_interp.interp_2d` (fast, supports non-uniform grids)
  - Fallback: `interp2d_nonuniform` from math_funcs for numba readonly array issues
- **No `eval_linear`** — removed from all horses code. Do not reintroduce.
- `extrap_policy` setting controls extrapolation (0=clamp, 1=linear)

## Simulation

- NPV utility includes terminal utility `term_u(w_T)` at the last period
- `sim_guard` excludes agent-periods with `a < threshold` (Druedahl convention)
- `ce_burn_in` skips first N periods from NPV accumulation
- `init_dispersion > 0` enables log-normal initial condition heterogeneity

## FUES settings (all in settings.yaml, declared in stage symbols)

`fues_eps_d`, `fues_eps_sep`, `fues_eps_fwd_back`, `fues_parallel_guard`,
`extrap_policy`, `clamp_max_factor`, `correct_jumps`, `n_sections` —
all must be declared in `symbols.settings` of the stage YAML and
referenced in the methods YAML `settings:` block.

## Gadi deployment

- Gadi repo: `/home/141/as3442/dev/fues.dev/FUES/`
- Gadi venv: `~/venvs/fues/`
- Gadi scratch results: `/scratch/tp66/as3442/FUES/durables/`
- Mount point: `/Users/akshayshanker/gadi/`
- **Always push to GitHub before pulling on Gadi.** Do not attempt
  git operations via sshfs — they create lock files.
- After pulling on Gadi, the venv may need `pip install -e .` to pick up
  new dependencies (e.g. `consav`, `quantecon` added to core deps).
- PBS scripts live in `experiments/durables/` but are NOT tracked in git.
  Use `scripts/pull_gadi_results.sh` to snapshot them into replication/.
