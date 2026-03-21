"""DurablesModel: grids, parameters, and equation callables.

Wraps ``ConsumerProblem`` from the original durables example
and exposes the numerical resources needed by stage operators.
"""

import interpolation.splines as _splines

# Patch UCGrid to skip the numba type assertion that fails
# with newer numba versions (int64 vs float64 mismatch).
_orig_UCGrid = _splines.UCGrid
def _patched_UCGrid(*args):
    """UCGrid wrapper that ensures bounds are float."""
    fixed = []
    for tup in args:
        lo, hi, n = tup
        fixed.append((float(lo), float(hi), int(n)))
    return _orig_UCGrid(*fixed)
_splines.UCGrid = _patched_UCGrid

from examples.durables.durables import ConsumerProblem


class DurablesModel:
    """Numerical resources for the durables model.

    Thin wrapper around ``ConsumerProblem`` that exposes
    grids, transition matrices, and utility callables in
    a uniform interface.

    Parameters
    ----------
    config : dict
        Configuration dict (grid sizes, HD settings, etc.).
    **kwargs
        Passed to ``ConsumerProblem.__init__``.
    """

    def __init__(self, config, **kwargs):
        # Ensure grid sizes are int and bounds are float
        # to avoid UCGrid numba type assertion failures
        for k in ('grid_size_A', 'grid_size_H', 'grid_size_W'):
            if k in kwargs:
                kwargs[k] = int(kwargs[k])
        for k in ('b', 'grid_max_A', 'grid_max_H', 'grid_max_WE'):
            if k in kwargs:
                kwargs[k] = float(kwargs[k])
        self.cp = ConsumerProblem(config, **kwargs)
        # Expose key attributes
        self.T = self.cp.T
        self.t0 = self.cp.t0
        self.m_bar = self.cp.m_bar
        self.z_vals = self.cp.z_vals
        self.Pi = self.cp.Pi
        self.asset_grid_A = self.cp.asset_grid_A
        self.asset_grid_H = self.cp.asset_grid_H
        self.asset_grid_WE = self.cp.asset_grid_WE
        self.R = self.cp.R
        self.R_H = self.cp.R_H
        self.beta = self.cp.beta
        self.delta = self.cp.delta
        self.alpha = self.cp.alpha
        self.tau = self.cp.tau
        self.b = self.cp.b
        self.X_all = self.cp.X_all
        self.term_u = self.cp.term_u
        self.term_du = self.cp.term_du
        self.N_HD_LAMBDA = self.cp.N_HD_LAMBDA

    @property
    def params(self):
        """Scalar model parameters as named dict."""
        cp = self.cp
        return {
            'beta': self.beta, 'R': self.R,
            'R_H': self.R_H, 'delta': self.delta,
            'tau': self.tau, 'b': self.b,
            'chi': cp.chi, 'alpha': self.alpha,
            'gamma_c': cp.gamma_c, 'gamma_h': cp.gamma_h,
            'kappa': cp.kappa, 'K': cp.K,
            'theta': cp.theta,
            'grid_max_A': cp.grid_max_A,
            'grid_max_H': cp.grid_max_H,
        }

    @property
    def callables(self):
        """Equation callables (njit functions from cp)."""
        cp = self.cp
        return {
            'u': cp.u, 'du_c': cp.du_c,
            'du_c_inv': cp.du_c_inv,
            'du_h': cp.du_h,
            'y_func': cp.y_func,
            'term_u': cp.term_u,
            'term_du': cp.term_du,
        }

    @property
    def grids(self):
        """Grid arrays and state-space infrastructure."""
        cp = self.cp
        return {
            'a': self.asset_grid_A,
            'h': self.asset_grid_H,
            'we': self.asset_grid_WE,
            'he': cp.asset_grid_HE,
            'ac': cp.asset_grid_AC,
            'z': self.z_vals,
            'Pi': self.Pi,
            'X_all': self.X_all,
            'UGgrid_all': cp.UGgrid_all,
        }

    @classmethod
    def from_period(cls, period, calibration, settings):
        """Construct from a DDSL calibrated period.

        Reads parameters from the calibration dict and
        grid sizes from the settings dict.

        Parameters
        ----------
        period : dict
            ``{'stages': {name: SymbolicModel, ...}}``.
        calibration : dict
            From ``load_syntax``.
        settings : dict
            From ``load_syntax``.
        """
        config = dict(calibration)
        config.update(settings)
        return cls(
            config,
            r=float(calibration['r']),
            beta=float(calibration['beta']),
            alpha=float(calibration.get('alpha', 0.7)),
            delta=float(calibration.get('delta', 0)),
            kappa=float(calibration.get('kappa', 1.0)),
            sigma=float(calibration.get('sigma', 0.001)),
            r_H=float(calibration.get('r_H', 0)),
            b=float(calibration.get('b', 1e-8)),
            gamma_c=float(calibration.get('gamma_c', 3)),
            gamma_h=float(calibration.get('gamma_h', 1)),
            chi=float(calibration.get('chi', 0)),
            tau=float(calibration.get('tau', 0.2)),
            K=float(calibration.get('K', 200)),
            theta=float(calibration.get('theta', 2)),
            grid_max_A=float(settings.get('a_max', 50)),
            grid_max_H=float(settings.get('h_max', 50)),
            grid_max_WE=float(settings.get('w_max', 100)),
            grid_size_A=int(settings.get('n_a', 50)),
            grid_size_H=int(settings.get('n_h', 50)),
            grid_size_W=int(settings.get('n_w', 50)),
            m_bar=float(settings.get('m_bar', 1.4)),
            T=int(settings.get('T', 60)),
            t0=int(calibration.get('t0', settings.get('T', 60))),
            N_wage=int(calibration.get('N_wage', 3)),
            phi_w=float(calibration.get('phi_w', 0.917)),
            sigma_w=float(calibration.get('sigma_w', 0.082)),
        )
