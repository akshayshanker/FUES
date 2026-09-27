#!/bin/bash
# resubmit_2026-09-28_nfs_storm.sh — repair of the 28 Sep 2026 campaign after
# the NFS import storm (see CHANGELOG): the jobs below were submitted with the
# pre-staging scripts. Each is stopped and resubmitted with the current script,
# RESUMING FROM ITS OWN CHECKPOINT (RUN_ID + RESUME=1; the checkpoint holds the
# CE state and RNG, so the trajectory continues exactly). Queued jobs are simply
# resubmitted. Run from the repo root on Gadi.
#
#   bash resubmit_2026-09-28_nfs_storm.sh validate   # step A: sigma_w NEGM sweep only
#   bash resubmit_2026-09-28_nfs_storm.sh wave1      # step B: 7 running xxl + 2 sweeps
#   bash resubmit_2026-09-28_nfs_storm.sh wave2      # step C: remaining 7 running xxl
#   bash resubmit_2026-09-28_nfs_storm.sh queued     # step D: the never-started jobs
#
# Record of what actually ran (28 Sep 2026): validate 05:05 -> 179974956; the three
# tier-1 EGM seed runs stopped at their budgets and were resumed at 06:15 as
# 179975348/349/350 (wave1/wave2 entries for them are therefore no-ops); queued
# ran at 06:48 -> 179975695..713, with 686/688 re-issued as resumes 179975717/718.
set -euo pipefail
D=benchmarks/durables/estimation/data-estimation/separable
S=benchmarks/durables/estimation/selfgen
P="PURPOSE=resumed-after-nfs-storm-2026-09-28"

resume() {   # resume <old job id> <job name> <script> <RUN_ID> [SPEC=...]
    local old=$1 name=$2 script=$3 run=$4 spec=${5:-}
    qdel "$old" 2>/dev/null || true
    local vars="RUN_ID=$run,RESUME=1,$P"; [ -n "$spec" ] && vars="$spec,$vars"
    printf '  %-15s <- %s  %s\n' "$name" "$old" "$(qsub -N "$name" -v "$vars" "$script")"
}
fresh() {    # fresh <old job id> <job name> <script> [VAR=...,] [extra qsub args]
    local old=$1 name=$2 script=$3 vars=${4:-}; shift 4 || shift $#
    qdel "$old" 2>/dev/null || true
    local id
    if [ -n "$vars" ]; then id=$(qsub -N "$name" -v "$vars,$P" "$@" "$script"); else id=$(qsub -N "$name" -v "$P" "$@" "$script"); fi
    printf '  %-15s <- %s  %s\n' "$name" "$old" "$id"
}

case "${1:-}" in
validate)
    fresh 179972352 sw_sigW_NEGM "$S/run_selfgen_sweep_sigma_w_negm.pbs" ;;
wave1)
    resume 179972684 sw_gamClo_EGM  "$S/run_selfgen_sweep_gamma_c_egm.pbs"  20260928_025057 "SPEC=selfgen_sweep_gamma_c_low_egm.yaml"
    resume 179972685 sw_gamClo_NEGM "$S/run_selfgen_sweep_gamma_c_negm.pbs" 20260928_024941 "SPEC=selfgen_sweep_gamma_c_low_negm.yaml"
    resume 179972271 est_xxlNEGMm   "$D/males/run_xxl_negm_males.pbs" 20260928_012816
    resume 179972272 est_xxlEGMm    "$D/males/run_xxl_egm_males.pbs"  20260928_012809
    resume 179972338 xxlNEGM        "$D/females/run_xxl_negm.pbs"     20260928_013530
    resume 179972340 xxlEGM_s7      "$D/females/run_xxl_egm.pbs"      20260928_013521 "SPEC=baseline_xlarge_egm_seed7.yaml"
    resume 179972341 xxlNEGM_s7     "$D/females/run_xxl_negm.pbs"     20260928_013704 "SPEC=baseline_xlarge_negm_seed7.yaml"
    resume 179972342 xxlEGMm_s7     "$D/males/run_xxl_egm_males.pbs"  20260928_013551 "SPEC=baseline_xlarge_egm_males_seed7.yaml"
    resume 179972344 xxlEGM_s123    "$D/females/run_xxl_egm.pbs"      20260928_013511 "SPEC=baseline_xlarge_egm_seed123.yaml" ;;
wave2)
    resume 179972345 xxlNEGM_s123   "$D/females/run_xxl_negm.pbs"     20260928_013546 "SPEC=baseline_xlarge_negm_seed123.yaml"
    resume 179972346 xxlEGMm_s123   "$D/males/run_xxl_egm_males.pbs"  20260928_013635 "SPEC=baseline_xlarge_egm_males_seed123.yaml"
    resume 179972348 xxlEGM_s2026   "$D/females/run_xxl_egm.pbs"      20260928_013520 "SPEC=baseline_xlarge_egm_seed2026.yaml"
    resume 179972349 xxlNEGM_s2026  "$D/females/run_xxl_negm.pbs"     20260928_014822 "SPEC=baseline_xlarge_negm_seed2026.yaml"
    resume 179972350 xxlEGMm_s2026  "$D/males/run_xxl_egm_males.pbs"  20260928_033342 "SPEC=baseline_xlarge_egm_males_seed2026.yaml" ;;
queued)
    # tier 1 remainder (179972339 xxlNEGMm is dropped: it duplicates est_xxlNEGMm)
    qdel 179972339 2>/dev/null || true
    fresh 179972343 xxlNEGMm_s7    "$D/males/run_xxl_negm_males.pbs" "SPEC=baseline_xlarge_negm_males_seed7.yaml"
    fresh 179972347 xxlNEGMm_s123  "$D/males/run_xxl_negm_males.pbs" "SPEC=baseline_xlarge_negm_males_seed123.yaml"
    fresh 179972351 xxlNEGMm_s2026 "$D/males/run_xxl_negm_males.pbs" "SPEC=baseline_xlarge_negm_males_seed2026.yaml"
    # tier 2
    # 179972686 and 179972688 had started (old script) by the time this step ran on 28 Sep 06:48;
    # they were resumed from their checkpoints (iteration 14) instead of restarted:
    resume 179972686 xxlEGM_N20k "$D/females/run_xxl_egm.pbs" 20260928_044323 "SPEC=baseline_xlarge_egm_nsim20k.yaml,N_SIM=20000"
    resume 179972688 xxlEGM_s11  "$D/females/run_xxl_egm.pbs" 20260928_045352 "SPEC=baseline_xlarge_egm_seed11.yaml"
    fresh 179972687 xxlEGMm_N20k "$D/males/run_xxl_egm_males.pbs" "SPEC=baseline_xlarge_egm_males_nsim20k.yaml,N_SIM=20000"
    fresh 179972689 xxlEGMm_s11  "$D/males/run_xxl_egm_males.pbs" "SPEC=baseline_xlarge_egm_males_seed11.yaml"
    fresh 179972690 xxlEGM_s99   "$D/females/run_xxl_egm.pbs"     "SPEC=baseline_xlarge_egm_seed99.yaml"
    fresh 179972691 xxlEGMm_s99  "$D/males/run_xxl_egm_males.pbs" "SPEC=baseline_xlarge_egm_males_seed99.yaml"
    # tier 2b (settings robustness); N_wage runs at 56 ranks/node on 38 nodes
    LOW=(-l ncpus=3952,mem=18240GB,jobfs=1520GB)
    fresh 179972692 xlEGM_hmax8    "$D/females/run_xlarge_egm.pbs"  "SPEC=baseline_xlarge_egm_hmax8.yaml,EXTRA_SETTINGS=h_max=8;w_max=10.5"
    fresh 179972693 xlEGM_hmax125  "$D/females/run_xlarge_egm.pbs"  "SPEC=baseline_xlarge_egm_hmax12p5.yaml,EXTRA_SETTINGS=h_max=12.5;w_max=15"
    fresh 179972694 xlEGM_hmax15   "$D/females/run_xlarge_egm.pbs"  "SPEC=baseline_xlarge_egm_hmax15.yaml,EXTRA_SETTINGS=h_max=15;w_max=17.5"
    fresh 179972695 xlEGM_nw6      "$D/females/run_xlarge_egm.pbs"  "SPEC=baseline_xlarge_egm_nwage6.yaml,EXTRA_SETTINGS=N_wage=6,NRANKS=2080,PPR=7,ITERS_PER_RESTART=2" "${LOW[@]}"
    fresh 179972696 xlEGM_nw8      "$D/females/run_xlarge_egm.pbs"  "SPEC=baseline_xlarge_egm_nwage8.yaml,EXTRA_SETTINGS=N_wage=8,NRANKS=2080,PPR=7,ITERS_PER_RESTART=1" "${LOW[@]}"
    fresh 179972697 xlNEGM_hmax8   "$D/females/run_xlarge_negm.pbs" "SPEC=baseline_xlarge_negm_hmax8.yaml,EXTRA_SETTINGS=h_max=8;w_max=10.5"
    fresh 179972698 xlNEGM_hmax125 "$D/females/run_xlarge_negm.pbs" "SPEC=baseline_xlarge_negm_hmax12p5.yaml,EXTRA_SETTINGS=h_max=12.5;w_max=15"
    fresh 179972699 xlNEGM_hmax15  "$D/females/run_xlarge_negm.pbs" "SPEC=baseline_xlarge_negm_hmax15.yaml,EXTRA_SETTINGS=h_max=15;w_max=17.5"
    fresh 179972700 xlNEGM_nw6     "$D/females/run_xlarge_negm.pbs" "SPEC=baseline_xlarge_negm_nwage6.yaml,EXTRA_SETTINGS=N_wage=6,NRANKS=2080,PPR=7,ITERS_PER_RESTART=2" "${LOW[@]}"
    fresh 179972701 xlNEGM_nw8     "$D/females/run_xlarge_negm.pbs" "SPEC=baseline_xlarge_negm_nwage8.yaml,EXTRA_SETTINGS=N_wage=8,NRANKS=2080,PPR=7,ITERS_PER_RESTART=1" "${LOW[@]}" ;;
*)
    echo "usage: $0 validate|wave1|wave2|queued" >&2; exit 1 ;;
esac
