"""Construction helpers for durables.

Grid discretization from calibration + YAML settings.
Equation callables live in callables.py; grids use the same YAML as stage instantiation.
"""

import numpy as np
import quantecon as qe
import interpolation.splines as _splines


def nonlinspace(x_min, x_max, n, phi):
    """Grid with denser points near x_min (Druedahl convention).

    phi = 1.0: uniform (equivalent to linspace).
    phi > 1.0: packs points near x_min.
    """
    y = np.empty(n)
    y[0] = x_min
    for i in range(1, n):
        y[i] = y[i - 1] + (x_max - y[i - 1]) / (n - i) ** phi
    return y


# ============================================================
# UCGrid patch (numba dtype assertion workaround)
# ============================================================

_orig_UCGrid = _splines.UCGrid


def _patched_UCGrid(*args):
    fixed = []
    for tup in args:
        lo, hi, n = tup
        fixed.append((float(lo), float(hi), int(n)))
    return _orig_UCGrid(*fixed)


_splines.UCGrid = _patched_UCGrid
UCGrid = _splines.UCGrid


# ============================================================
# Construction helpers
# ============================================================

def make_grids(calibration, settings):
    """Construct all grids from calibration + YAML settings dicts."""
    b = float(settings.get("b", calibration.get("b", 1e-8)))
    a_min = float(settings.get("a_min", b))
    h_min = float(settings.get("h_min", b))
    grid_max_A = float(settings.get("a_max", 50.0))
    grid_max_H = float(settings.get("h_max", 50.0))
    grid_max_WE = float(settings.get("w_max", 100.0))
    _tau_adj = float(calibration.get("tau", 0.0))
    _R = float(calibration.get("R", 1.0))
    _R_H = float(calibration.get("R_H", 1.0))
    _delta = float(calibration.get("delta", 0.0))
    # w_min must cover the minimum next-period adjuster wealth:
    # R*a_min + R_H*(1-delta)*h_min + income_min
    # Use b for a_min/h_min, income_min ≈ 0 (conservative).
    _w_min_feasible = _R * a_min + _R_H * (1.0 - _delta) * h_min
    _w_min_setting = float(settings.get("w_min", b))
    w_min_WE = max(_w_min_setting, _w_min_feasible)
    n_a = int(settings.get("n_a", 50))
    n_h = int(settings.get("n_h", 50))
    n_w = int(settings.get("n_w", 50))
    grid_phi = float(settings.get("grid_phi", 1.0))

    phi_w = float(calibration.get("phi_w", 0.917))
    sigma_w = float(calibration.get("sigma_w", 0.082))
    N_wage = int(settings.get("N_wage", calibration.get("N_wage", 3)))

    labour_mc = qe.markov.tauchen(N_wage, phi_w, sigma_w, mu=0.0, n_std=3)
    z_vals = np.asarray(labour_mc.state_values)
    Pi = np.asarray(labour_mc.P)
    min_prob = 1e-3
    Pi = np.maximum(Pi, min_prob)
    Pi = Pi / Pi.sum(axis=1, keepdims=True)

    if grid_phi == 1.0:
        asset_grid_A = np.linspace(a_min, np.float64(grid_max_A), n_a)
        asset_grid_H = np.linspace(h_min, np.float64(grid_max_H), n_h)
        asset_grid_HE = np.linspace(h_min, np.float64(grid_max_H), n_h)
        asset_grid_WE = np.linspace(w_min_WE, np.float64(grid_max_WE), n_w)
    else:
        asset_grid_A = nonlinspace(a_min, np.float64(grid_max_A), n_a, grid_phi)
        asset_grid_H = nonlinspace(h_min, np.float64(grid_max_H), n_h, grid_phi)
        asset_grid_HE = nonlinspace(h_min, np.float64(grid_max_H), n_h, grid_phi)
        asset_grid_WE = nonlinspace(w_min_WE, np.float64(grid_max_WE), n_w, grid_phi)

    # UCGrid kept for backward compat; interp2d_nonuniform uses raw grids
    UGgrid_all = UCGrid((a_min, grid_max_A, n_a), (h_min, grid_max_H, n_h))

    return {
        "a": asset_grid_A,
        "h": asset_grid_H,
        "h_choice": asset_grid_HE,
        "we": asset_grid_WE,
        "z": z_vals,
        "Pi": Pi,
        "UGgrid_all": UGgrid_all,
    }
