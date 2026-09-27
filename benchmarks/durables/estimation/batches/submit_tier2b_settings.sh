#!/bin/bash
# submit_tier2b_settings.sh — run plan of 28 Sep 2026, tier 2b (about 230k SU, 10 jobs):
# robustness of the female xlarge (2,080-sample) estimates to numerical settings,
# EGM and NEGM:
#   * h_max in {8, 12.5, 15} (default 10), with w_max moved to keep w_max - h_max = 2.5
#   * N_wage in {6, 8} Tauchen income points (default 4). N_wage is a state
#     variable, so rank memory grows with it: these run 2,080 ranks on 38 nodes
#     (56 ranks/node, ppr 7) with shorter segments; qsub -l overrides the header.
# Results are filed under the variant spec's name; each run writes a manifest.
set -euo pipefail
D=benchmarks/durables/estimation/data-estimation/separable/females

sub() {   # sub <job name> <script> <VAR=value,...> [extra qsub args...]
    local name=$1 script=$2 vars=$3; shift 3
    local id; id=$(qsub -N "$name" -v "$vars" "$@" "$script")
    printf '  %-15s %-70s %s\n' "$name" "$vars" "$id"
}
LOWDENSITY=(-l ncpus=3952,mem=18240GB,jobfs=1520GB)   # 38 nodes x 104 cores; 2,080 ranks at 56/node

echo "Tier 2b (2026-09-28): settings robustness, females, xlarge sample size"
for m in egm negm; do
    M=${m^^}
    sub "xl${M}_hmax8"   "$D/run_xlarge_$m.pbs" "SPEC=baseline_xlarge_${m}_hmax8.yaml,EXTRA_SETTINGS=h_max=8;w_max=10.5,PURPOSE=settings-robustness-h_max=8"
    sub "xl${M}_hmax125" "$D/run_xlarge_$m.pbs" "SPEC=baseline_xlarge_${m}_hmax12p5.yaml,EXTRA_SETTINGS=h_max=12.5;w_max=15,PURPOSE=settings-robustness-h_max=12.5"
    sub "xl${M}_hmax15"  "$D/run_xlarge_$m.pbs" "SPEC=baseline_xlarge_${m}_hmax15.yaml,EXTRA_SETTINGS=h_max=15;w_max=17.5,PURPOSE=settings-robustness-h_max=15"
    sub "xl${M}_nw6"     "$D/run_xlarge_$m.pbs" "SPEC=baseline_xlarge_${m}_nwage6.yaml,EXTRA_SETTINGS=N_wage=6,NRANKS=2080,PPR=7,ITERS_PER_RESTART=2,PURPOSE=settings-robustness-N_wage=6" "${LOWDENSITY[@]}"
    sub "xl${M}_nw8"     "$D/run_xlarge_$m.pbs" "SPEC=baseline_xlarge_${m}_nwage8.yaml,EXTRA_SETTINGS=N_wage=8,NRANKS=2080,PPR=7,ITERS_PER_RESTART=1,PURPOSE=settings-robustness-N_wage=8" "${LOWDENSITY[@]}"
done
echo "10 jobs submitted. Watch with: qstat -u \$USER"
