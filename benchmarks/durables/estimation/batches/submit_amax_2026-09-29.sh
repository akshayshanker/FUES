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
#   bash submit_amax_2026-09-29.sh wave4a  # 23:45: a_max 20 at seeds 11/123/7, EGM females (3 h 30 walltime)
#   bash submit_amax_2026-09-29.sh wave4b  # a_max 20 at seeds 7/2026, EGM males (5 h); NEGM males a_max 20 (7 h)
#   bash submit_amax_2026-09-29.sh wave5 negm_s11 egm_m_s11 negm_m_a10   # as the balance frees up
#
# Results are filed under each variant spec's name; every job writes a manifest.
set -euo pipefail
D=benchmarks/durables/estimation/data-estimation/separable
P=PURPOSE=amax-robustness-2026-09-29

sub() {   # sub <job name> <script> <VAR=value,...> [extra qsub args, e.g. -l walltime=3:30:00]
    local name=$1 script=$2 vars=$3; shift 3
    local id; id=$(qsub -N "$name" -v "$vars,$P" "$@" "$script")
    printf '  %-15s %-80s %s\n' "$name" "$vars" "$id"
}
# Shorter walltimes than the script headers lower the SU reservation PBS holds
# against the project (a job cannot start unless its full walltime is covered),
# so more jobs fit into the remaining balance; WALL_SECONDS tells the restart
# loop the same budget. A run that reaches the shorter limit stops on its
# checkpoint and is resumed with RUN_ID=<run id>,RESUME=1.
W330=(-l walltime=3:30:00); V330="WALL_SECONDS=12600"     # female EGM a_max runs converged in 1 h 48 min - 2 h 31 min
W400=(-l walltime=4:00:00); V400="WALL_SECONDS=14400"     # female NEGM a_max runs: 1 h 44 min - 2 h 21 min
W500=(-l walltime=5:00:00); V500="WALL_SECONDS=18000"     # male EGM a_max runs: 2 h 56 min - 3 h 54 min, two at the 6 h cap

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
wave4a)
    # Seed robustness of the ceiling-free (a_max = 20) optimum: the three best
    # baseline seeds for females (11, 123, 7) ...
    sub xxlEGM_a20s11   "$D/females/run_xxl_egm.pbs"     "SPEC=baseline_xlarge_egm_amax20_seed11.yaml,EXTRA_SETTINGS=a_max=20;w_max=25,$V330"  "${W330[@]}"
    sub xxlEGM_a20s123  "$D/females/run_xxl_egm.pbs"     "SPEC=baseline_xlarge_egm_amax20_seed123.yaml,EXTRA_SETTINGS=a_max=20;w_max=25,$V330" "${W330[@]}"
    sub xxlEGM_a20s7    "$D/females/run_xxl_egm.pbs"     "SPEC=baseline_xlarge_egm_amax20_seed7.yaml,EXTRA_SETTINGS=a_max=20;w_max=25,$V330"   "${W330[@]}" ;;
wave4b)
    # ... the two best for males (7, 2026), and NEGM males at the new ceiling
    # (completes the method x sex table; the seed-42 male NEGM repeat took 6 h 48 min).
    sub xxlEGMm_a20s7   "$D/males/run_xxl_egm_males.pbs"  "SPEC=baseline_xlarge_egm_males_amax20_seed7.yaml,EXTRA_SETTINGS=a_max=20;w_max=25,$V500"    "${W500[@]}"
    sub xxlEGMm_a20s26  "$D/males/run_xxl_egm_males.pbs"  "SPEC=baseline_xlarge_egm_males_amax20_seed2026.yaml,EXTRA_SETTINGS=a_max=20;w_max=25,$V500" "${W500[@]}"
    sub xxlNEGMm_a20    "$D/males/run_xxl_negm_males.pbs" "SPEC=baseline_xlarge_negm_males_amax20.yaml,EXTRA_SETTINGS=a_max=20;w_max=25" ;;
wave5)
    # Submitted as reservations free up (each job needs its full walltime covered
    # by the balance): NEGM females seed 11 at a_max 20, a third male EGM seed,
    # NEGM males at a_max 10. Pass the names of the jobs to submit, e.g. `wave5 negm_s11 egm_m_s11`.
    shift
    for j in "$@"; do case "$j" in
        negm_s11)  sub xxlNEGM_a20s11 "$D/females/run_xxl_negm.pbs"     "SPEC=baseline_xlarge_negm_amax20_seed11.yaml,EXTRA_SETTINGS=a_max=20;w_max=25,$V400"      "${W400[@]}" ;;
        egm_m_s11) sub xxlEGMm_a20s11 "$D/males/run_xxl_egm_males.pbs"  "SPEC=baseline_xlarge_egm_males_amax20_seed11.yaml,EXTRA_SETTINGS=a_max=20;w_max=25,$V500" "${W500[@]}" ;;
        negm_m_a10) sub xxlNEGMm_a10  "$D/males/run_xxl_negm_males.pbs" "SPEC=baseline_xlarge_negm_males_amax10.yaml,EXTRA_SETTINGS=a_max=10;w_max=15" ;;
        *) echo "unknown wave5 job: $j" >&2 ;;
    esac; done ;;
*)
    echo "usage: $0 wave1|wave2|wave3|wave4a|wave4b|wave5 <negm_s11|egm_m_s11|negm_m_a10>..." >&2; exit 1 ;;
esac
