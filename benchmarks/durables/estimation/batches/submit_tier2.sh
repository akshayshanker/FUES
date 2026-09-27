#!/bin/bash
# submit_tier2.sh — run plan of 28 Sep 2026, tier 2 (about 200k SU, 6 jobs):
#   * low-gamma_c recovery sweeps with 1,040 CE samples per point (double):
#     does more sampling fix recovery below gamma_c ~ 3.3?
#   * xxl EGM, both sexes, with N_SIM=20000: simulation-noise sensitivity
#   * two further CE seeds (11, 99) for xxl EGM, both sexes
# Run from the repo root on Gadi once tier 1 is flowing.
set -euo pipefail
D=benchmarks/durables/estimation/data-estimation/separable
S=benchmarks/durables/estimation/selfgen

sub() {   # sub <job name> <script> [VAR=value[,VAR=value]]
    local name=$1 script=$2 vars=${3:-} id
    if [ -n "$vars" ]; then id=$(qsub -N "$name" -v "$vars" "$script"); else id=$(qsub -N "$name" "$script"); fi
    printf '  %-15s %-52s %s\n' "$name" "${vars:-(script defaults)}" "$id"
}

echo "Tier 2 (2026-09-28): low-gamma_c sweeps, N_SIM sensitivity, seeds 11 and 99"
sub sw_gamClo_EGM  "$S/run_selfgen_sweep_gamma_c_egm.pbs"   "SPEC=selfgen_sweep_gamma_c_low_egm.yaml"
sub sw_gamClo_NEGM "$S/run_selfgen_sweep_gamma_c_negm.pbs"  "SPEC=selfgen_sweep_gamma_c_low_negm.yaml"
sub xxlEGM_N20k    "$D/females/run_xxl_egm.pbs"      "SPEC=baseline_xlarge_egm_nsim20k.yaml,N_SIM=20000"
sub xxlEGMm_N20k   "$D/males/run_xxl_egm_males.pbs"  "SPEC=baseline_xlarge_egm_males_nsim20k.yaml,N_SIM=20000"
for seed in 11 99; do
    sub "xxlEGM_s$seed"  "$D/females/run_xxl_egm.pbs"      "SPEC=baseline_xlarge_egm_seed$seed.yaml"
    sub "xxlEGMm_s$seed" "$D/males/run_xxl_egm_males.pbs"  "SPEC=baseline_xlarge_egm_males_seed$seed.yaml"
done
echo "6 jobs submitted. Watch with: qstat -u \$USER"
