#!/bin/bash
# ==========================================================================
#  Pull latest durables results from Gadi and snapshot into paper-results/
#
#  What it does:
#    1. rsync latest results from scratch/tp66/$USER/FUES/durables/ on Gadi
#    2. rsync the PBS scripts and mod/ settings as-is from Gadi home
#    3. Create a dated snapshot in paper-results/durables/YYYY-MM-DD/NNN/
#    4. Copy tables, settings snapshot, and PBS snapshot into the snapshot dir
#    5. With --estimation: rsync estimation JSON and CSV from
#       /g/data/tp66/results/durables/estimation/ into
#       paper-results/durables/estimation/raw/ (nest and pickle files stay on Gadi)
#
#  Usage (from repo root, locally):
#    bash scripts/pull_gadi_results.sh
#    bash scripts/pull_gadi_results.sh --estimation
#
#  Requirements:
#    - SSH config has a 'gadi' host entry (e.g. as3442@gadi.nci.org.au)
#    - Gadi results are at /scratch/tp66/$GADI_USER/FUES/durables/
#    - Gadi repo is at /home/141/as3442/dev/fues.dev/FUES/
# ==========================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Configuration ---
GADI_HOST="${GADI_HOST:-gadi}"
GADI_USER="${GADI_USER:-as3442}"
GADI_REPO="/home/141/${GADI_USER}/dev/fues.dev/FUES"
GADI_SCRATCH="/scratch/tp66/${GADI_USER}/FUES/durables"

LOCAL_STAGING="${REPO_ROOT}/_gadi_staging"
REPLICATION_BASE="${REPO_ROOT}/paper-results/durables"

# Copy only directories, JSON and CSV from SOURCE into DEST.
# SOURCE may be a local path or a host:path. DEST is a local directory.
# Nest files (*.nst) and pickle files (*.pkl) are excluded. Local files
# already in DEST are left in place (no --delete).
rsync_estimation() {
    local SOURCE="$1"
    local DEST="$2"
    mkdir -p "${DEST}"
    rsync -avz --progress \
        "${SOURCE}" \
        "${DEST}" \
        --include='*/' \
        --include='*.json' \
        --include='*.csv' \
        --exclude='*'
}

# When this file is sourced, stop after defining the function so a rehearsal
# can call rsync_estimation with a local SOURCE and DEST.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    return 0
fi

PULL_ESTIMATION=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --estimation)
            PULL_ESTIMATION=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: bash scripts/pull_gadi_results.sh [--estimation]" >&2
            exit 1
            ;;
    esac
done

# --- Step 1: Pull latest results from scratch ---
echo "=== Step 1: Pull results from Gadi scratch ==="
mkdir -p "${LOCAL_STAGING}/results"
rsync -avz --progress \
    "${GADI_HOST}:${GADI_SCRATCH}/" \
    "${LOCAL_STAGING}/results/" \
    --exclude='*.npy' \
    --exclude='__pycache__' \
    --include='*/' \
    --include='*.tex' \
    --include='*.md' \
    --include='*.csv' \
    --include='*.json' \
    --include='*.png' \
    --include='*.pdf' \
    --exclude='*'

# --- Step 2: Snapshot PBS scripts from Gadi home ---
echo ""
echo "=== Step 2: Snapshot PBS scripts from Gadi ==="
mkdir -p "${LOCAL_STAGING}/pbs"
rsync -avz --progress \
    "${GADI_HOST}:${GADI_REPO}/benchmarks/durables/" \
    "${LOCAL_STAGING}/pbs/" \
    --include='*.pbs' \
    --include='*.sh' \
    --include='*/' \
    --exclude='*'

# --- Step 3: Snapshot mod/ settings from Gadi home ---
echo ""
echo "=== Step 3: Snapshot settings from Gadi ==="
mkdir -p "${LOCAL_STAGING}/mod"
rsync -avz --progress \
    "${GADI_HOST}:${GADI_REPO}/examples/durables/syntax/" \
    "${LOCAL_STAGING}/mod/" \
    --include='*.yaml' \
    --include='*.yml' \
    --include='*/' \
    --exclude='*'

# --- Step 4: Create dated snapshot ---
echo ""
echo "=== Step 4: Create replication snapshot ==="
TODAY=$(date +%Y-%m-%d)
SNAP_BASE="${REPLICATION_BASE}/${TODAY}"
mkdir -p "${SNAP_BASE}"

# Find next run number
N=1
while [[ -d "${SNAP_BASE}/$(printf '%03d' $N)" ]]; do
    N=$((N + 1))
done
SNAP_DIR="${SNAP_BASE}/$(printf '%03d' $N)"
mkdir -p "${SNAP_DIR}"

echo "Snapshot: ${SNAP_DIR}"

# Copy results
if [[ -d "${LOCAL_STAGING}/results" ]]; then
    # Find the latest dated run in results
    LATEST_RUN=$(ls -td "${LOCAL_STAGING}/results/"*/ 2>/dev/null | head -1)
    if [[ -n "${LATEST_RUN}" ]]; then
        cp -r "${LATEST_RUN}" "${SNAP_DIR}/results"
        echo "  Copied results from $(basename ${LATEST_RUN})"
    fi
    # Also copy any top-level tables
    find "${LOCAL_STAGING}/results" -maxdepth 2 -name '*.tex' -o -name '*.md' -o -name '*.csv' | while read f; do
        mkdir -p "${SNAP_DIR}/tables"
        cp "$f" "${SNAP_DIR}/tables/"
    done
fi

# Copy PBS snapshot
if [[ -d "${LOCAL_STAGING}/pbs" ]]; then
    cp -r "${LOCAL_STAGING}/pbs" "${SNAP_DIR}/pbs_snapshot"
    echo "  Copied PBS scripts snapshot"
fi

# Copy settings snapshot
if [[ -d "${LOCAL_STAGING}/mod" ]]; then
    cp -r "${LOCAL_STAGING}/mod" "${SNAP_DIR}/mod_snapshot"
    echo "  Copied mod/ settings snapshot"
fi

# Write metadata
cat > "${SNAP_DIR}/README.md" << HEREDOC
# Replication snapshot: ${TODAY}/$(printf '%03d' $N)

Pulled from Gadi at $(date).

- **Results**: \`${GADI_SCRATCH}/\`
- **PBS scripts**: \`${GADI_REPO}/benchmarks/durables/\`
- **Settings**: \`${GADI_REPO}/examples/durables/syntax/\`
- **Git branch**: $(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')
- **Git commit**: $(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')
HEREDOC

echo ""
echo "=== Done ==="
echo "Snapshot: ${SNAP_DIR}"
echo "Contents:"
find "${SNAP_DIR}" -type f | sort | sed 's|^|  |'

# --- Optional: estimation JSON and CSV from /g/data ---
if [[ "${PULL_ESTIMATION}" -eq 1 ]]; then
    echo ""
    echo "=== Step 5: Pull estimation results from Gadi ==="
    rsync_estimation \
        "${GADI_HOST}:/g/data/tp66/results/durables/estimation/" \
        "${REPO_ROOT}/paper-results/durables/estimation/raw/"
fi

# --- Cleanup staging ---
rm -rf "${LOCAL_STAGING}"
echo ""
echo "Staging dir cleaned up."
