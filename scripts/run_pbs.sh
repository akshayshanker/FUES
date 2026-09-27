#!/bin/bash
# run_pbs.sh — submit durables estimation PBS jobs to Gadi (lives in scripts/).
#
# Run on Gadi (needs qsub). Resolves the repo root from the script's location,
# so it works no matter where you call it from.
#
# Usage:
#   ./scripts/run_pbs.sh                       # list + submit ALL *.pbs under benchmarks/durables/estimation
#   ./scripts/run_pbs.sh -y                    # submit ALL, skip the confirm prompt
#   ./scripts/run_pbs.sh males                 # only files whose path contains 'males'
#   ./scripts/run_pbs.sh selfgen egm           # paths matching ALL given substrings (selfgen AND egm)
#   ./scripts/run_pbs.sh run_large_egm_males.pbs run_xlarge_egm.pbs   # specific files (any subfolder)
#
# Scripts live in subfolders (selfgen/, data-estimation/separable/{females,males}/,
# generic/); patterns are matched against the path relative to estimation/.
#
# These jobs are expensive (each large run is ~10k SU, xlarge ~21k, sweeps ~52k),
# hence the confirmation. Use -y only when you mean it.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # scripts/ -> repo root
EST_DIR="$ROOT/benchmarks/durables/estimation"
cd "$ROOT"

command -v qsub >/dev/null 2>&1 || { echo "ERROR: qsub not found — run this on Gadi." >&2; exit 1; }
[ -d "$EST_DIR" ] || { echo "ERROR: $EST_DIR not found." >&2; exit 1; }

ASSUME_YES=0
PATTERNS=()
EXPLICIT=()
for a in "$@"; do
    case "$a" in
        -y|--yes) ASSUME_YES=1 ;;
        *.pbs)
            if   [ -f "$a" ];           then EXPLICIT+=("$a")
            elif [ -f "$EST_DIR/$a" ];  then EXPLICIT+=("$EST_DIR/$a")
            else
                # bare file name: look for it in any subfolder
                found="$(find "$EST_DIR" -name "$(basename "$a")" -type f | head -1)"
                if [ -n "$found" ]; then EXPLICIT+=("$found")
                else echo "WARN: not found, skipping: $a" >&2; fi
            fi ;;
        *) PATTERNS+=("$a") ;;
    esac
done

JOBS=()
if [ "${#EXPLICIT[@]}" -gt 0 ]; then
    JOBS=("${EXPLICIT[@]}")
else
    while IFS= read -r f; do
        rel="${f#"$EST_DIR"/}"          # e.g. selfgen/run_selfgen_sweep_gamma_c_egm.pbs
        keep=1
        if [ "${#PATTERNS[@]}" -gt 0 ]; then
            for p in "${PATTERNS[@]}"; do
                case "$rel" in *"$p"*) ;; *) keep=0 ;; esac
            done
        fi
        [ "$keep" -eq 1 ] && JOBS+=("$f")
    done < <(find "$EST_DIR" -name '*.pbs' -type f | sort)
fi

[ "${#JOBS[@]}" -gt 0 ] || { echo "No matching .pbs jobs under $EST_DIR" >&2; exit 1; }

echo "Will submit ${#JOBS[@]} job(s) from benchmarks/durables/estimation:"
for j in "${JOBS[@]}"; do echo "  ${j#"$EST_DIR"/}"; done
echo

if [ "$ASSUME_YES" -ne 1 ]; then
    read -r -p "Submit these ${#JOBS[@]} job(s)? [y/N] " ans
    case "$ans" in y|Y|yes|YES) ;; *) echo "aborted."; exit 0 ;; esac
fi

for j in "${JOBS[@]}"; do
    id="$(qsub "$j")"
    echo "  $(basename "$j") -> $id"
done
echo "submitted ${#JOBS[@]} job(s)."
