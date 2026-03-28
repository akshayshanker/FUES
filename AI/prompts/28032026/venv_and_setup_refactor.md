# Prompt: Venv and setup strategy refactor

**Date**: 28 March 2026
**Status**: Design review — do NOT implement until reviewed

## Task

Review the full repo structure, working notes history, prompts history,
and current setup scripts. Then propose an optimised venv and setup
strategy that:
1. Works for all audiences
2. Is honest about what external users can/can't install
3. Consolidates the overlapping setup scripts
4. Updates the README and docs accordingly

## Context

Before proposing changes, the AI should read:

### Repo structure
- `pyproject.toml` — package metadata, optional deps, extras
- `setup/` — all setup scripts (setup_venv.sh, setup_durables_est.sh,
  load_env.sh, update_and_activate.sh, interactive_session.sh)
- `README.md` — current install instructions
- `docs/getting-started/installation.md` — current docs install page
- `examples/readme_examples.md` — example run instructions
- `.gitignore` — what's excluded (kikku/, AI/, etc.)

### Working notes (for lessons learned)
- `AI/working/28032026/session.md` — today's session (estimation pipeline,
  Gadi debugging, memory issues, PBS patterns)
- Recent working notes in `AI/working/` for any prior setup discussions

### Prompts history
- `AI/prompts/` — any prior prompts about installation or setup

### External dependencies
- `kikku` — bright-forest/kikku (private GitHub)
- `dolo` — bright-forest/dolo@phase1.1_0.1 (private GitHub)
- `dolang` — bright-forest/dolang.py@phase1.1_0.1 (private GitHub)
- These are required for the durables DDSL pipeline (examples + estimation)
- External users CANNOT install these via pip

## Current state (problems)

### Three audiences, muddled install paths

| Audience | What they need | Current path | Works? |
|----------|---------------|--------------|--------|
| FUES library user | Just `dcsmm.fues` | `pip install git+...` | YES |
| Example/notebook runner | dcsmm + kikku + dolo + matplotlib | `pip install ".[examples]"` | NO (private deps) |
| Gadi estimation | dcsmm + kikku + dolo + mpi4py | `bash setup/setup_durables_est.sh` | YES |

### Overlapping setup scripts

| Script | Creates venv at | Profile | Gadi-only? |
|--------|----------------|---------|------------|
| `setup_venv.sh` | `$HOME/venvs/fues` (Gadi) or `.venv` (local) | examples or durables-est | No (detects) |
| `setup_durables_est.sh` | `$HOME/venvs/fues` | durables-est | Yes |
| `load_env.sh` | — (activation only) | — | No (detects) |
| `update_and_activate.sh` | — (activation + pull + reinstall) | — | Yes |

`setup_venv.sh` and `setup_durables_est.sh` both write to `$HOME/venvs/fues`.
`load_env.sh` looks for `/scratch/tp66/$USER/venvs/dcsmm` — a path that
no setup script creates.

### pyproject.toml extras include private GitHub URLs

```toml
[project.optional-dependencies]
durables-est = [..., "kikku[estimation] @ git+https://github.com/bright-forest/kikku.git"]
examples = [..., "kikku @ git+https://github.com/bright-forest/kikku.git"]
```

These fail for anyone without access to bright-forest org. But the
setup scripts install these deps separately (working around the issue).

## Proposed design

### Two audiences that need venvs

1. **Laptop (full)**: you, running notebooks, solving models, plotting,
   maybe tests. Needs everything: dcsmm + kikku + dolo + matplotlib +
   HARK + ConSav + quantecon.

2. **Gadi (estimation)**: lean for MPI. dcsmm + kikku + dolo + mpi4py +
   quantecon. No matplotlib, no HARK, no ConSav.

A third audience (external FUES user) just does `pip install` and
doesn't use any setup script.

### One setup script, two modes

```bash
bash setup/setup_venv.sh              # laptop: full
bash setup/setup_venv.sh estimation   # Gadi: lean
```

The script:
1. Detects Gadi (checks `/scratch/tp66`)
2. Creates venv at `.venv` (local) or `~/venvs/fues` (Gadi)
3. Installs the right pyproject.toml extras
4. Installs dolo + dolang from GitHub with `--no-deps`
5. On Gadi: builds mpi4py from source
6. Verifies imports

### Kill redundant scripts

| Script | Action |
|--------|--------|
| `setup_venv.sh` | Keep — consolidate into single entry point |
| `setup_durables_est.sh` | DELETE — merged into `setup_venv.sh estimation` |
| `load_env.sh` | DELETE — just use `source .venv/bin/activate` or `source ~/venvs/fues/bin/activate` |
| `update_and_activate.sh` | KEEP — quick refresh for Gadi |
| `interactive_session.sh` | KEEP — PBS interactive session |

### Fix pyproject.toml extras

Remove private GitHub URLs from pyproject.toml. Let the setup script
handle private deps:

```toml
[project.optional-dependencies]
full = [
    "matplotlib", "seaborn", "tqdm", "dill>=0.3.6",
    "econ-ark", "ConSav", "pykdtree>=1.3.12",
    "pyyaml", "quantecon", "pytest", "autopep8",
]
estimation = [
    "pyyaml", "quantecon", "packaging",
]
```

kikku, dolo, dolang are installed by the setup script, not pip.

### Update README

```markdown
## Installation

### FUES algorithm only (external users)
pip install git+https://github.com/akshayshanker/FUES.git

### Full development setup (laptop)
git clone https://github.com/akshayshanker/FUES.git
cd FUES
bash setup/setup_venv.sh
source .venv/bin/activate

### Estimation on NCI Gadi (HPC)
git clone https://github.com/akshayshanker/FUES.git
cd FUES
bash setup/setup_venv.sh estimation
source ~/venvs/fues/bin/activate
```

No `pip install ".[examples]"` as a standalone option — it requires
private GitHub repos that external users can't access.

### Update docs/getting-started/installation.md

Mirror the README structure. Add a section on updating:
```bash
# Update code + deps on Gadi:
source setup/update_and_activate.sh
```

## Questions to resolve before implementing

1. Should `[full]` include mpi4py? It won't install on macOS without
   homebrew OpenMPI. Options:
   - Include it, document the homebrew prereq
   - Exclude it, let `setup_venv.sh` install it only on Gadi
   - Make it a separate extra: `pip install ".[full,mpi]"`

2. Should the retirement and housing-renting examples work without
   kikku/dolo? Currently retirement uses HARK directly (no DDSL).
   If so, there's an intermediate tier: `pip install ".[examples-lite]"`
   that has HARK + ConSav but not kikku/dolo.

3. Where should dolo/dolang version pins live? Currently hardcoded in
   setup scripts (`@phase1.1_0.1`). Should they be in a requirements
   file or a config?

4. Should `update_and_activate.sh` pull a specific branch or the
   current branch? Currently hardcodes `durables-ddsl-phase2`.

5. Should we add a `Makefile` or `just` recipes as the top-level
   interface instead of shell scripts?

## Deliverables

1. Consolidated `setup/setup_venv.sh` with `estimation` mode
2. Deleted `setup_durables_est.sh` and `load_env.sh`
3. Updated `pyproject.toml` (no private GitHub URLs in extras)
4. Updated `README.md` (three clear install paths)
5. Updated `docs/getting-started/installation.md`
6. Updated `update_and_activate.sh` (pull current branch, not hardcoded)
