#!/bin/bash
# submit_tier1.sh — run plan of 28 Sep 2026, tier 1 (about 560k SU, 15 jobs):
#   * NEGM at the xxl (4,160-sample) size for both sexes — new results
#   * CE-seed robustness: the four xxl estimates with sampling_seed 7, 123, 2026
#     (every run so far used seed 42 and reproduced itself exactly)
#   * sigma_w NEGM recovery sweep under the restart loop
# Run from the repo root on Gadi. Each job is an existing script with SPEC (and
# the job name) overridden through qsub -v / -N; results land in folders named
# after the spec, so variants never overwrite each other.
set -euo pipefail
D=benchmarks/durables/estimation/data-estimation/separable
S=benchmarks/durables/estimation/selfgen

sub() {   # sub <job name> <script> [VAR=value[,VAR=value]]
    local name=$1 script=$2 vars=${3:-} id
    if [ -n "$vars" ]; then id=$(qsub -N "$name" -v "$vars" "$script"); else id=$(qsub -N "$name" "$script"); fi
    printf '  %-15s %-52s %s\n' "$name" "${vars:-(script defaults)}" "$id"
}

echo "Tier 1 (2026-09-28): NEGM xxl, seed robustness, sigma_w NEGM"
sub xxlNEGM      "$D/females/run_xxl_negm.pbs"
sub xxlNEGMm     "$D/males/run_xxl_negm_males.pbs"
for seed in 7 123 2026; do
    sub "xxlEGM_s$seed"   "$D/females/run_xxl_egm.pbs"       "SPEC=baseline_xlarge_egm_seed$seed.yaml"
    sub "xxlNEGM_s$seed"  "$D/females/run_xxl_negm.pbs"      "SPEC=baseline_xlarge_negm_seed$seed.yaml"
    sub "xxlEGMm_s$seed"  "$D/males/run_xxl_egm_males.pbs"   "SPEC=baseline_xlarge_egm_males_seed$seed.yaml"
    sub "xxlNEGMm_s$seed" "$D/males/run_xxl_negm_males.pbs"  "SPEC=baseline_xlarge_negm_males_seed$seed.yaml"
done
sub sw_sigW_NEGM "$S/run_selfgen_sweep_sigma_w_negm.pbs"
echo "15 jobs submitted. Watch with: qstat -u \$USER"
