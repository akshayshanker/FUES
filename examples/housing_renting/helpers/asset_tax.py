"""Asset-return tax utilities for housing_renting model.

This module provides functions to compute piecewise-linear capital income taxes
on asset holdings. The tax schedule is defined by brackets with:
- a0, a1: bracket bounds (assets)
- B: base tax at a0 (can have discontinuous jumps)
- tau_a: marginal tax rate within bracket

Tax formula: T(a) = B + tau_a * (a - a0) for a in (a0, a1]

Author: Akshay Shanker, University of New South Wales
"""

import numpy as np
from numba import njit


def parse_tax_table(tax_table_dict, debug=False):
    """Parse YAML tax_table into arrays for JIT functions.

    Parameters
    ----------
    tax_table_dict : dict
        Tax table from YAML config with 'brackets' list containing
        dicts with keys: 'a0', 'a1', 'B', 'tau_a', and optional 'left_closed', 'open'
    debug : bool
        If True, print debug information about brackets

    Returns
    -------
    a0_arr : ndarray
        Lower bounds of each bracket
    a1_arr : ndarray
        Upper bounds of each bracket (inf represented as 1e30)
    B_arr : ndarray
        Base tax at lower bound of each bracket
    tau_arr : ndarray
        Marginal tax rate within each bracket
    left_closed_arr : ndarray (bool as int8)
        1 if bracket is closed [a0, a1], 0 otherwise
    open_arr : ndarray (bool as int8)
        1 if bracket is open (a0, a1), 0 otherwise

    Interval types:
    - left_closed=True:  [a0, a1]  - includes both endpoints (closed interval)
    - open=True:         (a0, a1)  - excludes both endpoints
    - default:           (a0, a1]  - excludes a0, includes a1
    """
    brackets = tax_table_dict.get("brackets", [])
    n = len(brackets)

    if n == 0:
        # Return empty arrays if no brackets
        return (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
                np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.int8))

    a0_arr = np.zeros(n)
    a1_arr = np.zeros(n)
    B_arr = np.zeros(n)
    tau_arr = np.zeros(n)
    left_closed_arr = np.zeros(n, dtype=np.int8)
    open_arr = np.zeros(n, dtype=np.int8)

    if debug:
        print(f"[TAX DEBUG] parse_tax_table: Parsing {n} brackets")

    for i, br in enumerate(brackets):
        a0_arr[i] = br["a0"]
        # Handle infinity - YAML .inf parses to float('inf')
        a1_val = br["a1"]
        if a1_val == float('inf') or a1_val is None:
            a1_arr[i] = 1e30
        else:
            a1_arr[i] = a1_val
        B_arr[i] = br["B"]
        tau_arr[i] = br["tau_a"]
        # left_closed: if True, bracket is [a0, a1] (closed on both ends)
        left_closed_arr[i] = 1 if br.get("left_closed", False) else 0
        # open: if True, bracket is (a0, a1) - open on both sides
        open_arr[i] = 1 if br.get("open", False) else 0

        if debug:
            if open_arr[i]:
                interval_type = "(a0, a1)"
            elif left_closed_arr[i]:
                interval_type = "[a0, a1]"
            else:
                interval_type = "(a0, a1]"
            print(f"  Bracket {i}: a0={a0_arr[i]:.4f}, a1={a1_arr[i]:.4f}, B={B_arr[i]:.6f}, tau={tau_arr[i]:.4f}, interval={interval_type}")
            if left_closed_arr[i]:
                print(f"    -> LEFT_CLOSED: a={a0_arr[i]:.4f} uses THIS bracket (B={B_arr[i]:.6f})")

    return a0_arr, a1_arr, B_arr, tau_arr, left_closed_arr, open_arr


@njit
def total_tax_scalar(a, a0_arr, a1_arr, B_arr, tau_arr, left_closed_arr, open_arr):
    """Compute total tax T(a) for scalar asset value.

    Parameters
    ----------
    a : float
        Asset level
    a0_arr, a1_arr, B_arr, tau_arr, left_closed_arr, open_arr : ndarray
        Bracket parameters from parse_tax_table

    Returns
    -------
    float
        Total tax T(a)
    """
    n = len(a0_arr)
    for i in range(n):
        # Check interval type:
        # - open: (a0, a1) - excludes both endpoints
        # - left_closed: [a0, a1] - includes both endpoints (closed)
        # - default: (a0, a1] - excludes a0, includes a1
        if open_arr[i]:
            # Open: (a0, a1) - excludes both endpoints
            if a0_arr[i] < a < a1_arr[i]:
                return B_arr[i] + tau_arr[i] * (a - a0_arr[i])
        elif left_closed_arr[i]:
            # Closed: [a0, a1] - includes both endpoints
            if a0_arr[i] <= a <= a1_arr[i]:
                return B_arr[i] + tau_arr[i] * (a - a0_arr[i])
        else:
            # Left-open (default): (a0, a1] - excludes a0, includes a1
            if a0_arr[i] < a <= a1_arr[i]:
                return B_arr[i] + tau_arr[i] * (a - a0_arr[i])
    # If a < 0 or outside all brackets, no tax
    return 0.0


@njit
def marginal_tax_scalar(a, a0_arr, a1_arr, tau_arr, left_closed_arr, open_arr):
    """Compute marginal tax rate tau_a(a) for scalar asset value.

    Parameters
    ----------
    a : float
        Asset level
    a0_arr, a1_arr, tau_arr, left_closed_arr, open_arr : ndarray
        Bracket parameters from parse_tax_table

    Returns
    -------
    float
        Marginal tax rate tau_a(a)
    """
    n = len(a0_arr)
    for i in range(n):
        # Check interval type
        if open_arr[i]:
            # Open: (a0, a1) - excludes both endpoints
            if a0_arr[i] < a < a1_arr[i]:
                return tau_arr[i]
        elif left_closed_arr[i]:
            # Closed: [a0, a1] - includes both endpoints
            if a0_arr[i] <= a <= a1_arr[i]:
                return tau_arr[i]
        else:
            # Left-open (default): (a0, a1] - excludes a0, includes a1
            if a0_arr[i] < a <= a1_arr[i]:
                return tau_arr[i]
    return 0.0


@njit
def total_tax_array(a_grid, a0_arr, a1_arr, B_arr, tau_arr, left_closed_arr, open_arr):
    """Compute total tax T(a) for asset grid (1D array).

    Parameters
    ----------
    a_grid : ndarray
        1D array of asset levels
    a0_arr, a1_arr, B_arr, tau_arr, left_closed_arr, open_arr : ndarray
        Bracket parameters from parse_tax_table

    Returns
    -------
    ndarray
        Total tax T(a) for each asset level
    """
    n = len(a_grid)
    T = np.zeros(n)
    for j in range(n):
        T[j] = total_tax_scalar(a_grid[j], a0_arr, a1_arr, B_arr, tau_arr, left_closed_arr, open_arr)
    return T


@njit
def marginal_tax_array(a_grid, a0_arr, a1_arr, tau_arr, left_closed_arr, open_arr):
    """Compute marginal tax rates tau_a(a) for asset grid (1D array).

    Parameters
    ----------
    a_grid : ndarray
        1D array of asset levels
    a0_arr, a1_arr, tau_arr, left_closed_arr, open_arr : ndarray
        Bracket parameters from parse_tax_table

    Returns
    -------
    ndarray
        Marginal tax rate tau_a(a) for each asset level
    """
    n = len(a_grid)
    tau = np.zeros(n)
    for j in range(n):
        tau[j] = marginal_tax_scalar(a_grid[j], a0_arr, a1_arr, tau_arr, left_closed_arr, open_arr)
    return tau


def debug_bracket_lookup(a, a0_arr, a1_arr, B_arr, tau_arr, left_closed_arr, open_arr):
    """Debug function to show which bracket a given asset value falls into.

    This is a NON-JIT function for debugging purposes only.

    Parameters
    ----------
    a : float
        Asset level to look up
    a0_arr, a1_arr, B_arr, tau_arr, left_closed_arr, open_arr : ndarray
        Bracket parameters from parse_tax_table
    """
    print(f"\n[TAX DEBUG] Looking up bracket for a = {a:.6f}")
    n = len(a0_arr)
    found = False
    for i in range(n):
        # Determine interval type
        if open_arr[i]:
            interval_type = "(a0, a1)"
        elif left_closed_arr[i]:
            interval_type = "[a0, a1]"
        else:
            interval_type = "(a0, a1]"

        # Check if a falls in this bracket
        if open_arr[i]:
            # Open: (a0, a1) - excludes both endpoints
            in_bracket = (a0_arr[i] < a < a1_arr[i])
        elif left_closed_arr[i]:
            # Closed: [a0, a1] - includes both endpoints
            in_bracket = (a0_arr[i] <= a <= a1_arr[i])
        else:
            # Left-open (default): (a0, a1] - excludes a0, includes a1
            in_bracket = (a0_arr[i] < a <= a1_arr[i])

        marker = " <-- MATCH" if in_bracket else ""
        print(f"  Bracket {i}: {interval_type} a0={a0_arr[i]:.4f}, a1={a1_arr[i]:.4f}, B={B_arr[i]:.6f}, tau={tau_arr[i]:.4f}{marker}")

        if in_bracket and not found:
            T_a = B_arr[i] + tau_arr[i] * (a - a0_arr[i])
            print(f"    -> T(a) = {B_arr[i]:.6f} + {tau_arr[i]:.4f} * ({a:.4f} - {a0_arr[i]:.4f}) = {T_a:.6f}")
            found = True

    if not found:
        print(f"  WARNING: a={a:.6f} not found in any bracket!")


def snap_brackets_to_grid(tax_table_dict, a_grid):
    """Snap bracket boundaries to nearest grid points (optional utility).

    This avoids boundary issues when mapping bracket boundaries to discrete
    grid points. Recomputes B values to maintain continuity except at
    intended jumps.

    Parameters
    ----------
    tax_table_dict : dict
        Tax table from YAML config
    a_grid : ndarray
        Asset grid to snap to

    Returns
    -------
    dict
        Modified tax_table_dict with snapped bracket boundaries
    """
    brackets = tax_table_dict.get("brackets", [])
    if len(brackets) == 0:
        return tax_table_dict

    # Create copy to avoid modifying original
    new_brackets = []
    a_grid_sorted = np.sort(a_grid)

    for i, br in enumerate(brackets):
        new_br = br.copy()

        # Snap a0 to nearest grid point
        a0 = br["a0"]
        if a0 > 0:
            idx = np.argmin(np.abs(a_grid_sorted - a0))
            new_br["a0"] = a_grid_sorted[idx]

        # Snap a1 to nearest grid point (unless inf)
        a1 = br["a1"]
        if a1 != float('inf') and a1 is not None:
            idx = np.argmin(np.abs(a_grid_sorted - a1))
            new_br["a1"] = a_grid_sorted[idx]

        new_brackets.append(new_br)

    # Recompute B values to maintain continuity (except at intended jumps)
    for i in range(1, len(new_brackets)):
        prev = new_brackets[i - 1]
        curr = new_brackets[i]

        # Check if there's an intended jump (original B values differ
        # from what continuity would imply)
        orig_prev = brackets[i - 1]
        orig_curr = brackets[i]
        expected_B = orig_prev["B"] + orig_prev["tau_a"] * (orig_prev["a1"] - orig_prev["a0"])

        if abs(orig_curr["B"] - expected_B) > 1e-10:
            # Intended jump - keep the jump magnitude
            jump = orig_curr["B"] - expected_B
            new_expected = prev["B"] + prev["tau_a"] * (curr["a0"] - prev["a0"])
            curr["B"] = new_expected + jump
        else:
            # No jump - maintain continuity
            curr["B"] = prev["B"] + prev["tau_a"] * (curr["a0"] - prev["a0"])

    new_table = tax_table_dict.copy()
    new_table["brackets"] = new_brackets
    return new_table


def parse_tax_constraint_nodes(tax_table_dict, a_nxt_grid, debug=False):
    """Extract constraint node specifications from tax_table.

    Each bracket can have constraints at both LHS (a0) and RHS (a1):
    - LHS (a0): Where you enter this bracket (transition from lower bracket)
    - RHS (a1): Where you exit this bracket (transition to higher bracket)

    Both transitions can cause kinks in the policy function due to marginal
    tax rate changes.

    Parameters
    ----------
    tax_table_dict : dict
        Tax table from YAML with 'constraints' and 'brackets' keys.
        constraints block should contain:
        - enabled: bool
        - default_c_lb_pct: float (default 0.10)
        - default_c_ub_pct: float (default 0.10)
        Each bracket can have:
        - constraint_lhs: bool, dict, or omitted
        - constraint_rhs: bool, dict, or omitted

    a_nxt_grid : ndarray
        The a_nxt grid used in EGM (to find nearest grid indices)

    debug : bool
        If True, print debug information about constraint nodes

    Returns
    -------
    tuple : (constraint_nodes, n_points_per_node)
        constraint_nodes : list of dict
            Each dict contains:
            - 'a_idx': index in a_nxt_grid closest to bracket boundary
            - 'a_value': actual a_nxt value at that index
            - 'a_target': original target value from config
            - 'c_lb_pct': consumption lower bound percentage
            - 'c_ub_pct': consumption upper bound percentage
            - 'side': 'lhs' or 'rhs' (for debugging)
        n_points_per_node : int
            Number of constraint points to generate per tax bracket boundary

        Returns ([], 10) if constraints not enabled.

    Notes
    -----
    FOC filtering is handled separately in Python code (egm_preprocess)
    and applies on top of these constraint segments.
    """
    constraints_cfg = tax_table_dict.get('constraints', {})
    if not constraints_cfg.get('enabled', False):
        if debug:
            print("[TAX DEBUG] parse_tax_constraint_nodes: constraints NOT enabled")
        return [], 10  # Return default n_points when disabled

    # Get defaults
    default_lb = constraints_cfg.get('default_c_lb_pct', 0.10)
    default_ub = constraints_cfg.get('default_c_ub_pct', 0.10)
    n_points_per_node = constraints_cfg.get('n_points_per_node', 10)

    if debug:
        print(f"[TAX DEBUG] parse_tax_constraint_nodes: constraints ENABLED")
        print(f"  default_c_lb_pct={default_lb}, default_c_ub_pct={default_ub}, n_points_per_node={n_points_per_node}")

    brackets = tax_table_dict.get('brackets', [])
    nodes = []
    # Allow both sides at same gridpoint; we dedupe only exact (a_idx, side)
    seen_keys = set()

    def add_node(a_target, c_lb_pct, c_ub_pct, side, bracket_idx):
        """Helper to add a constraint node, avoiding duplicates."""
        # Skip infinity
        if a_target == float('inf') or a_target is None or a_target > 1e29:
            if debug:
                print(f"  [SKIP] Bracket {bracket_idx} {side.upper()}: a_target={a_target} is infinity")
            return

        # Find nearest grid index
        a_idx = int(np.argmin(np.abs(a_nxt_grid - a_target)))
        a_value = float(a_nxt_grid[a_idx])

        # Skip if we already have this exact node (same grid point and side)
        key = (a_idx, side)
        if key in seen_keys:
            if debug:
                print(f"  [SKIP] Bracket {bracket_idx} {side.upper()}: duplicate (a_idx={a_idx}, a_value={a_value:.4f})")
            return
        seen_keys.add(key)

        nodes.append({
            'a_idx': a_idx,
            'a_value': a_value,
            'a_target': float(a_target),
            'c_lb_pct': float(c_lb_pct),
            'c_ub_pct': float(c_ub_pct),
            'side': side,
            'bracket_idx': bracket_idx,  # For debugging
        })

        if debug:
            print(f"  [ADD] Bracket {bracket_idx} {side.upper()}: a_target={a_target:.4f} -> a_idx={a_idx}, a_value={a_value:.4f}")

    def parse_constraint_spec(spec, default_lb, default_ub):
        """Parse constraint specification (True, False, or dict)."""
        if spec is False:
            return None  # Disabled
        elif spec is True or spec is None:
            return (default_lb, default_ub)  # Use defaults
        elif isinstance(spec, dict):
            return (
                spec.get('c_lb_pct', default_lb),
                spec.get('c_ub_pct', default_ub)
            )
        else:
            return (default_lb, default_ub)

    for i, br in enumerate(brackets):
        a0 = br['a0']
        a1 = br['a1']
        left_closed = br.get('left_closed', False)

        if debug:
            interval_type = "[a0, a1]" if left_closed else "(a0, a1]"
            print(f"  Processing bracket {i}: a0={a0}, a1={a1}, interval={interval_type}")

        # Parse LHS constraint (at a0)
        # Default: True (add with defaults) unless first bracket with a0=0
        lhs_default = False if (i == 0 and a0 == 0) else True
        lhs_spec = br.get('constraint_lhs', lhs_default)
        lhs_params = parse_constraint_spec(lhs_spec, default_lb, default_ub)
        if lhs_params is not None:
            add_node(a0, lhs_params[0], lhs_params[1], 'lhs', i)

        # Parse RHS constraint (at a1)
        # Default: True (add with defaults) unless a1=inf
        rhs_spec = br.get('constraint_rhs', True)
        rhs_params = parse_constraint_spec(rhs_spec, default_lb, default_ub)
        if rhs_params is not None:
            add_node(a1, rhs_params[0], rhs_params[1], 'rhs', i)

    # Sort by grid index for consistent ordering
    nodes.sort(key=lambda n: n['a_idx'])

    if debug:
        print(f"[TAX DEBUG] parse_tax_constraint_nodes: Created {len(nodes)} constraint nodes:")
        for node in nodes:
            print(f"    {node['side'].upper()} at a_idx={node['a_idx']}, a_value={node['a_value']:.4f} (target={node['a_target']:.4f}, bracket={node.get('bracket_idx', '?')})")

    return nodes, n_points_per_node
