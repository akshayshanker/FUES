"""FUES numerical constants.

These are compile-time constants consumed by @njit functions.
Kept as Python literals (not YAML) so Numba can inline them.
"""

# Numerical stability
EPS_D = 1e-14           # minimum grid separation (float64 safe)
EPS_SEP = 1e-08         # minimum separation for intersection creation
EPS_FWD_BACK = 0.5      # proximity threshold for forward/backward scans
PARALLEL_GUARD = 1e-10  # near-parallel line detection tolerance

# Scan direction flags (used in _scan internals)
TURN_LEFT = 1
TURN_RIGHT = 0
JUMP_YES = 1
JUMP_NO = 0
