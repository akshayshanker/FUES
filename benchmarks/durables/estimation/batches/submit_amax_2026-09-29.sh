#!/bin/bash
# submit_amax_2026-09-29.sh — asset-ceiling robustness, 29 Sep 2026 (11 jobs, about
# 310k SU expected, 515k at the walltime caps; tp66 had 520k SU left for 2026.q3).
#
# Motivation: a_max = 7.5 (= $750k at normalisation 1e-5) lies below the data
# mean of financial assets at ages 55-59 ($803k for both sexes; sd $503k
# females, $591k males). At the 28 Sep optima the mean and sd of financial
# assets at ages 45-59 carry 46% (females) / 44% (males) of the objective, and
# the simulation clamps policies at the grid edge (extrap_policy = 0), so the
# top of the asset distribution is compressed against the ceiling. The adjuster
# wealth grid moves with a_max, keeping the base offset (w_max = a_max + 5;
# adjuster wealth is w = R a + h + y). Grid sizes stay at 600, so the asset
# spacing coarsens with a_max ($1.25k at 7.5, $3.3k at 20).
#
# Same xxl scripts, ranks and segments as the 29 Sep seed-42 repeats, which ran
# at 95-97% CPU efficiency: 4,160 ranks on 40 normalsr nodes, 104 ranks/node,
# 3-iteration segments, venv staged on jobfs. Three waves ten minutes apart so
# that at most four jobs copy the venv from /home at the same time.
#
#   bash submit_amax_2026-09-29.sh wave1   # EGM females a_max 10/15/20, EGM males 10
#   bash submit_amax_2026-09-29.sh wave2   # EGM males 15/20; joint ceiling a_max = h_max = 20, both sexes
#   bash submit_amax_2026-09-29.sh wave3   # NEGM females a_max 10/15/20
#
# Results are filed under each variant spec's name; every job writes a manifest.
set -euo pipefail
D=benchmarks/durables/estimation/data-estimation/separable
P=PURPOSE=amax-robustness-2026-09-29

sub() {   # sub <job name> <script> <VAR=value,...>
    local name=$1 script=$2 vars=$3
    local id; id=$(qsub -N "$name" -v "$vars,$P" "$script")
    printf '  %-15s %-80s %s\n' "$name" "$vars" "$id"
}

case "${1:-}" in
wave1)
    sub xxlEGM_a10     "$D/females/run_xxl_egm.pbs"     "SPEC=baseline_xlarge_egm_amax10.yaml,EXTRA_SETTINGS=a_max=10;w_max=15"
    sub xxlEGM_a15     "$D/females/run_xxl_egm.pbs"     "SPEC=baseline_xlarge_egm_amax15.yaml,EXTRA_SETTINGS=a_max=15;w_max=20"
    sub xxlEGM_a20     "$D/females/run_xxl_egm.pbs"     "SPEC=baseline_xlarge_egm_amax20.yaml,EXTRA_SETTINGS=a_max=20;w_max=25"
    sub xxlEGMm_a10    "$D/males/run_xxl_egm_males.pbs" "SPEC=baseline_xlarge_egm_males_amax10.yaml,EXTRA_SETTINGS=a_max=10;w_max=15" ;;
wave2)
    sub xxlEGMm_a15    "$D/males/run_xxl_egm_males.pbs" "SPEC=baseline_xlarge_egm_males_amax15.yaml,EXTRA_SETTINGS=a_max=15;w_max=20"
    sub xxlEGMm_a20    "$D/males/run_xxl_egm_males.pbs" "SPEC=baseline_xlarge_egm_males_amax20.yaml,EXTRA_SETTINGS=a_max=20;w_max=25"
    sub xxlEGM_a20h20  "$D/females/run_xxl_egm.pbs"     "SPEC=baseline_xlarge_egm_amax20_hmax20.yaml,EXTRA_SETTINGS=a_max=20;h_max=20;w_max=30"
    sub xxlEGMm_a20h20 "$D/males/run_xxl_egm_males.pbs" "SPEC=baseline_xlarge_egm_males_amax20_hmax20.yaml,EXTRA_SETTINGS=a_max=20;h_max=20;w_max=30" ;;
wave3)
    sub xxlNEGM_a10    "$D/females/run_xxl_negm.pbs"    "SPEC=baseline_xlarge_negm_amax10.yaml,EXTRA_SETTINGS=a_max=10;w_max=15"
    sub xxlNEGM_a15    "$D/females/run_xxl_negm.pbs"    "SPEC=baseline_xlarge_negm_amax15.yaml,EXTRA_SETTINGS=a_max=15;w_max=20"
    sub xxlNEGM_a20    "$D/females/run_xxl_negm.pbs"    "SPEC=baseline_xlarge_negm_amax20.yaml,EXTRA_SETTINGS=a_max=20;w_max=25" ;;
*)
    echo "usage: $0 wave1|wave2|wave3" >&2; exit 1 ;;
esac
