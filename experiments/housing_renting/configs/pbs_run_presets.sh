#!/bin/bash

# ======================================================================
#  Experiment Configuration Library
# ======================================================================
#
# This file contains the parameter settings for different experimental
# runs. It is intended to be sourced by PBS submission scripts.
#
# Each configuration is an associative array.
#

# --- Configuration Set 1 (LOW POINTS FOR TESTING): Standard Resolution ---
# A standard, medium-sized run for quick validation.
declare -A STD_RES_SETTINGS
STD_RES_SETTINGS[periods]=10
STD_RES_SETTINGS[vfi_ngrid]=2000
STD_RES_SETTINGS[hd_points]=2000
STD_RES_SETTINGS[grid_points]=2000
STD_RES_SETTINGS[version_suffix]="test_0.1"
STD_RES_SETTINGS[delta_pb]=1

declare -A STD_RES_SETTINGS_3
STD_RES_SETTINGS_3[periods]=5
STD_RES_SETTINGS_3[vfi_ngrid]=2000
STD_RES_SETTINGS_3[hd_points]=2000
STD_RES_SETTINGS_3[grid_points]=2000
STD_RES_SETTINGS_3[version_suffix]="test_0.3"
STD_RES_SETTINGS_3[delta_pb]=1


declare -A STD_RES_SETTINGS_PB
STD_RES_SETTINGS_PB[periods]=5
STD_RES_SETTINGS_PB[vfi_ngrid]=2000
STD_RES_SETTINGS_PB[hd_points]=2000
STD_RES_SETTINGS_PB[grid_points]=2000
STD_RES_SETTINGS_PB[version_suffix]="test_0.1"
STD_RES_SETTINGS_PB[delta_pb]=0.6

# --- Configuration Set 2: High Resolution Benchmark ---
# A high-resolution run for producing final, accurate results.
declare -A HIGH_RES_SETTINGS_A
HIGH_RES_SETTINGS_A[periods]=5
HIGH_RES_SETTINGS_A[vfi_ngrid]=1E4
HIGH_RES_SETTINGS_A[hd_points]=6000
HIGH_RES_SETTINGS_A[grid_points]=6000
HIGH_RES_SETTINGS_A[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_A[delta_pb]=1

declare -A HIGH_RES_SETTINGS_A_PB
HIGH_RES_SETTINGS_A_PB[periods]=20
HIGH_RES_SETTINGS_A_PB[vfi_ngrid]=1E4
HIGH_RES_SETTINGS_A_PB[hd_points]=2000
HIGH_RES_SETTINGS_A_PB[grid_points]=10000
HIGH_RES_SETTINGS_A_PB[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_A_PB[delta_pb]=0.6

declare -A HIGH_RES_SETTINGS_A1_PB
HIGH_RES_SETTINGS_A1_PB[periods]=5
HIGH_RES_SETTINGS_A1_PB[vfi_ngrid]=1E4
HIGH_RES_SETTINGS_A1_PB[hd_points]=6000
HIGH_RES_SETTINGS_A1_PB[grid_points]=30000
HIGH_RES_SETTINGS_A1_PB[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_A1_PB[delta_pb]=0.6

declare -A HIGH_RES_SETTINGS_B
HIGH_RES_SETTINGS_B[periods]=5
HIGH_RES_SETTINGS_B[vfi_ngrid]=2E6
HIGH_RES_SETTINGS_B[hd_points]=2E4
HIGH_RES_SETTINGS_B[grid_points]=2000
HIGH_RES_SETTINGS_B[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_B[delta_pb]=1

declare -A HIGH_RES_SETTINGS_B1
HIGH_RES_SETTINGS_B1[periods]=5
HIGH_RES_SETTINGS_B1[vfi_ngrid]=1E6
HIGH_RES_SETTINGS_B1[hd_points]=2E4
HIGH_RES_SETTINGS_B1[grid_points]=2000
HIGH_RES_SETTINGS_B1[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_B1[delta_pb]=1

declare -A HIGH_RES_SETTINGS_C
HIGH_RES_SETTINGS_C[periods]=20
HIGH_RES_SETTINGS_C[vfi_ngrid]=1E6
HIGH_RES_SETTINGS_C[hd_points]=3E5
HIGH_RES_SETTINGS_C[grid_points]=2000
HIGH_RES_SETTINGS_C[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_C[delta_pb]=1

declare -A HIGH_RES_SETTINGS_C_PB
HIGH_RES_SETTINGS_C_PB[periods]=5
HIGH_RES_SETTINGS_C_PB[vfi_ngrid]=1E6
HIGH_RES_SETTINGS_C_PB[hd_points]=3E5
HIGH_RES_SETTINGS_C_PB[grid_points]=2000
HIGH_RES_SETTINGS_C_PB[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_C_PB[delta_pb]=0.6

declare -A HIGH_RES_SETTINGS_D
HIGH_RES_SETTINGS_D[periods]=5
HIGH_RES_SETTINGS_D[vfi_ngrid]=1E6
HIGH_RES_SETTINGS_D[hd_points]=6E5
HIGH_RES_SETTINGS_D[grid_points]=2000
HIGH_RES_SETTINGS_D[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_D[delta_pb]=1

declare -A HIGH_RES_SETTINGS_E
HIGH_RES_SETTINGS_E[periods]=5
HIGH_RES_SETTINGS_E[vfi_ngrid]=1E6
HIGH_RES_SETTINGS_E[hd_points]=12E5
HIGH_RES_SETTINGS_E[grid_points]=2000
HIGH_RES_SETTINGS_E[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_E[delta_pb]=1

declare -A HIGH_RES_SETTINGS_E_PB
HIGH_RES_SETTINGS_E_PB[periods]=5
HIGH_RES_SETTINGS_E_PB[vfi_ngrid]=1E6
HIGH_RES_SETTINGS_E_PB[hd_points]=12E5
HIGH_RES_SETTINGS_E_PB[grid_points]=2000
HIGH_RES_SETTINGS_E_PB[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_E_PB[delta_pb]=0.95

declare -A HIGH_RES_SETTINGS_F
HIGH_RES_SETTINGS_F[periods]=5
HIGH_RES_SETTINGS_F[vfi_ngrid]=1E6
HIGH_RES_SETTINGS_F[hd_points]=15E5
HIGH_RES_SETTINGS_F[grid_points]=2000
HIGH_RES_SETTINGS_F[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_F[delta_pb]=1

# Aliases for commonly used configurations
declare -A HIGH_RES_SETTINGS_K
HIGH_RES_SETTINGS_K[periods]=5
HIGH_RES_SETTINGS_K[vfi_ngrid]=1E6
HIGH_RES_SETTINGS_K[hd_points]=6E5
HIGH_RES_SETTINGS_K[grid_points]=2000
HIGH_RES_SETTINGS_K[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_K[delta_pb]=1
