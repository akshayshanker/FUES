"""Construction helpers for durables2_0.

Grid discretization from calibration + YAML settings.
Equation callables live in callables.py; grids use the same YAML as stage instantiation.
"""

import numpy as np
import quantecon as qe
import interpolation.splines as _splines


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
    b = float(calibration.get("b", 1e-8))
    grid_max_A = float(settings.get("a_max", 50.0))
    grid_max_H = float(settings.get("h_max", 50.0))
    grid_max_WE = float(settings.get("w_max", 100.0))
    n_a = int(settings.get("n_a", 50))
    n_h = int(settings.get("n_h", 50))
    n_w = int(settings.get("n_w", 50))

    phi_w = float(calibration.get("phi_w", 0.917))
    sigma_w = float(calibration.get("sigma_w", 0.082))
    N_wage = int(settings.get("N_wage", calibration.get("N_wage", 3)))

    labour_mc = qe.markov.tauchen(N_wage, phi_w, sigma_w, mu=0.0, n_std=3)
    z_vals = np.asarray(labour_mc.state_values)
    Pi = np.asarray(labour_mc.P)
    min_prob = 1e-3
    Pi = np.maximum(Pi, min_prob)
    Pi = Pi / Pi.sum(axis=1, keepdims=True)

    asset_grid_A = np.linspace(b, np.float64(grid_max_A), n_a)
    asset_grid_H = np.linspace(b, np.float64(grid_max_H), n_h)
    asset_grid_HE = np.linspace(b, np.float64(grid_max_H), n_h)
    asset_grid_WE = np.linspace(b, np.float64(grid_max_WE), n_w)

    X_all = qe.cartesian(
        [
            np.arange(len(z_vals)),
            np.arange(len(asset_grid_A)),
            np.arange(len(asset_grid_H)),
        ]
    )

    UGgrid_all = UCGrid((b, grid_max_A, n_a), (b, grid_max_H, n_h))

    return {
        "a": asset_grid_A,
        "h": asset_grid_H,
        "h_choice": asset_grid_HE,
        "we": asset_grid_WE,
        "z": z_vals,
        "Pi": Pi,
        "X_all": X_all,
        "UGgrid_all": UGgrid_all,
    }
