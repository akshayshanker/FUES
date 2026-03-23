"""Construction helpers for durables2_0.

Provides make_cp (ConsumerProblem from YAML) and make_grids
(space discretizations).  Equation callables, EGM recipe, and
transition morphisms live in callables.py.
"""

import numpy as np
import interpolation.splines as _splines
from examples.durables.durables import ConsumerProblem


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


# ============================================================
# Construction helpers
# ============================================================

def make_cp(calibration, settings):
    """Create ConsumerProblem from YAML calibration + settings."""
    config = dict(calibration)
    config.update(settings)
    theta = float(calibration.get("theta", 2.0))
    K = float(calibration.get("K", 200.0))
    cp = ConsumerProblem(
        config,
        r=float(calibration["r"]),
        beta=float(calibration["beta"]),
        alpha=float(calibration.get("alpha", 0.7)),
        delta=float(calibration.get("delta", 0.0)),
        kappa=float(calibration.get("kappa", 1.0)),
        sigma=float(calibration.get("sigma", 0.001)),
        r_H=float(calibration.get("r_H", 0.0)),
        b=float(calibration.get("b", 1e-8)),
        gamma_c=float(calibration.get("gamma_c", 3.0)),
        gamma_h=float(calibration.get("gamma_h", 1.0)),
        chi=float(calibration.get("chi", 0.0)),
        tau=float(calibration.get("tau", 0.2)),
        K=K,
        theta=theta,
        grid_max_A=float(settings.get("a_max", 50.0)),
        grid_max_H=float(settings.get("h_max", 50.0)),
        grid_max_WE=float(settings.get("w_max", 100.0)),
        grid_size_A=int(settings.get("n_a", 50)),
        grid_size_H=int(settings.get("n_h", 50)),
        grid_size_W=int(settings.get("n_w", 50)),
        m_bar=float(settings.get("m_bar", 1.4)),
        T=int(settings.get("T", 60)),
        t0=int(calibration.get("t0", settings.get("T", 60))),
        N_wage=int(calibration.get("N_wage", 3)),
        phi_w=float(calibration.get("phi_w", 0.917)),
        sigma_w=float(calibration.get("sigma_w", 0.082)),
    )
    # ConsumerProblem.__init__ does not store theta/K as attributes;
    # make_callables (in callables.py) needs them for term_u / term_du.
    cp.theta = theta
    cp.K = K
    return cp


def make_grids(cp):
    """Extract space discretizations from cp. Created once."""
    return {
        "a": cp.asset_grid_A,
        "h": cp.asset_grid_H,
        "h_choice": cp.asset_grid_HE,
        "we": cp.asset_grid_WE,
        "z": cp.z_vals,
        "Pi": cp.Pi,
        "X_all": cp.X_all,
        "UGgrid_all": cp.UGgrid_all,
    }


def make_settings(cp):
    """Package numerical settings from cp. Created once."""
    return {
        "b": cp.b,
        "m_bar": cp.m_bar,
        "root_eps": cp.root_eps,
        "egm_n": cp.EGM_N,
        "return_grids": getattr(cp, 'return_grids', False),
        "grid_max_A": cp.grid_max_A,
        "grid_max_H": cp.grid_max_H,
        "T": cp.T,
        "t0": cp.t0,
    }
