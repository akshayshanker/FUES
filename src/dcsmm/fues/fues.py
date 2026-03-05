"""Fast Upper-Envelope Scan (FUES) — default entry point.

Re-exports ``FUES`` from the latest development version (v0.2dev).

Version history
---------------
**fues_v0dev** (``FUES_V0DEV``)
    Reference implementation. Straightforward single-pass scan with
    NumPy pre/post-processing. Easiest to read and understand; use
    this to study the algorithm.

**fues_v0_1dev** (``FUES_V0_1DEV``)
    Production-ready rewrite. Adds intersection-bounds checking,
    forward/backward scan helpers, single-intersection mode, and
    the ``disable_jump_checks`` flag. Same algorithmic complexity
    as v0dev but with improved robustness at discrete-choice switches.

**fues_v0_2dev** (``FUES_V0_2DEV``) — *current default*
    Optimised for constant-factor scaling. Key changes over v0.1dev:

    - ``np.empty`` for intersection buffer (avoids NaN fill).
    - Conditional ``m_bar``: skips ``max(abs,abs)`` when
      ``endog_mbar=False``.
    - ``assume_sorted`` parameter + auto-detection via ``np.diff``
      to skip ``argsort`` + 5 fancy-index copies when input is
      already sorted.
    - Linear O(K+J) merge for intersection points via
      ``_merge_sorted_with_few``, replacing O(N log N) ``argsort``.
    - ``cache=True`` on all scan helpers for faster Numba cold starts.
    - Pre-computed ``abs(del_a)`` array to avoid per-iteration
      ``np.abs`` calls.

Usage::

    # Default (v0.2dev):
    from dcsmm.fues.fues import FUES

    # Explicit version selection:
    from dcsmm.fues.fues_v0_1dev import FUES as FUES_v01
    from dcsmm.fues.fues_v0_2dev import FUES as FUES_v02

    # Via UE engine registry (e.g. in solve_nest):
    method = "FUES"          # latest (v0.2dev)
    method = "FUES_V0_1DEV"  # baseline
    method = "FUES_V0_2DEV"  # optimised (same as "FUES")

Author: Akshay Shanker, 2025, a.shanker@unsw.edu.au
"""

from .fues_v0_2dev import FUES  # noqa: F401
