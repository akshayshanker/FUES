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
STD_RES_SETTINGS[periods]=5
STD_RES_SETTINGS[vfi_ngrid]=1000
STD_RES_SETTINGS[hd_points]=2000
STD_RES_SETTINGS[grid_points]=2000
STD_RES_SETTINGS[version_suffix]="test_0.1"
STD_RES_SETTINGS[delta_pb]=1

declare -A STD_RES_SETTINGS_PB
STD_RES_SETTINGS_PB[periods]=5
STD_RES_SETTINGS_PB[vfi_ngrid]=2000
STD_RES_SETTINGS_PB[hd_points]=2000
STD_RES_SETTINGS_PB[grid_points]=6000
STD_RES_SETTINGS_PB[version_suffix]="test_0.1"
STD_RES_SETTINGS_PB[delta_pb]=0.6

# --- Configuration Set 1.1 (LOW POINTS FOR TESTING: High Resolution Benchmark ---
# A high-resolution run for producing final, accurate results.
declare -A HIGH_RES_SETTINGS_TEST
HIGH_RES_SETTINGS_TEST[periods]=5
HIGH_RES_SETTINGS_TEST[vfi_ngrid]=10000
HIGH_RES_SETTINGS_TEST[hd_points]=5000
HIGH_RES_SETTINGS_TEST[grid_points]=3000
HIGH_RES_SETTINGS_TEST[version_suffix]="test_0.1"

# --- Configuration Set 2: High Resolution Benchmark ---
# A high-resolution run for producing final, accurate results.
declare -A HIGH_RES_SETTINGS_A
HIGH_RES_SETTINGS_A[periods]=5
HIGH_RES_SETTINGS_A[vfi_ngrid]=1E4
HIGH_RES_SETTINGS_A[hd_points]=4E4
HIGH_RES_SETTINGS_A[grid_points]=6000
HIGH_RES_SETTINGS_A[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_A[delta_pb]=1

declare -A HIGH_RES_SETTINGS_A_PB
HIGH_RES_SETTINGS_A_PB[periods]=5
HIGH_RES_SETTINGS_A_PB[vfi_ngrid]=1000
HIGH_RES_SETTINGS_A_PB[hd_points]=1000
HIGH_RES_SETTINGS_A_PB[grid_points]=6000
HIGH_RES_SETTINGS_A_PB[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_A_PB[delta_pb]=0.95

declare -A HIGH_RES_SETTINGS_B
HIGH_RES_SETTINGS_B[periods]=5
HIGH_RES_SETTINGS_B[vfi_ngrid]=2E4
HIGH_RES_SETTINGS_B[hd_points]=6E4
HIGH_RES_SETTINGS_B[grid_points]=6000
HIGH_RES_SETTINGS_B[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_B[delta_pb]=1

declare -A HIGH_RES_SETTINGS_C
HIGH_RES_SETTINGS_C[periods]=5
HIGH_RES_SETTINGS_C[vfi_ngrid]=3E4
HIGH_RES_SETTINGS_C[hd_points]=8E4
HIGH_RES_SETTINGS_C[grid_points]=6000
HIGH_RES_SETTINGS_C[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_C[delta_pb]=1

declare -A HIGH_RES_SETTINGS_D
HIGH_RES_SETTINGS_D[periods]=5
HIGH_RES_SETTINGS_D[vfi_ngrid]=1E5
HIGH_RES_SETTINGS_D[hd_points]=1E5
HIGH_RES_SETTINGS_D[grid_points]=6000
HIGH_RES_SETTINGS_D[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_D[delta_pb]=1


declare -A HIGH_RES_SETTINGS_E
HIGH_RES_SETTINGS_E[periods]=5
HIGH_RES_SETTINGS_E[vfi_ngrid]=1E6
HIGH_RES_SETTINGS_E[hd_points]=1E5
HIGH_RES_SETTINGS_E[grid_points]=6000
HIGH_RES_SETTINGS_E[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_E[delta_pb]=1

declare -A HIGH_RES_SETTINGS_F
HIGH_RES_SETTINGS_F[periods]=5
HIGH_RES_SETTINGS_F[vfi_ngrid]=2E6
HIGH_RES_SETTINGS_F[hd_points]=2E5
HIGH_RES_SETTINGS_F[grid_points]=6000
HIGH_RES_SETTINGS_F[version_suffix]="test_0.1"
HIGH_RES_SETTINGS_F[delta_pb]=1


# --- Configuration Set 3: Quick Debug Run ---
# A very small, fast run for debugging purposes.
declare -A DEBUG_SETTINGS
DEBUG_SETTINGS[periods]=2
DEBUG_SETTINGS[vfi_ngrid]=50
DEBUG_SETTINGS[hd_points]=100
DEBUG_SETTINGS[grid_points]=100
DEBUG_SETTINGS[version_suffix]="test_0.1"
DEBUG_SETTINGS[delta_pb]=1

# --- Add more configurations as needed ---
# Example:
# declare -A SENSITIVITY_ANALYSIS_1
# SENSITIVITY_ANALYSIS_1[periods]=3
# ... 