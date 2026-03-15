"""Numerical resources (rho output) for the retirement model.

The calibrated stage ``.mod`` objects are the single source of
truth for parameter values.  ``RetirementModel`` is a thin
container that:

1. Holds a reference to the period dict (delegates param access)
2. Constructs the asset grid (derived from settings)
3. Stores the ``@njit`` equation callables

Stage operators are built separately in ``operators.py``.
"""

import numpy as np
from numba import njit


# ============================================================================
# Default equation callables (log utility)
# ============================================================================

@njit(cache=True)
def _default_u(c):
    return np.log(c)


@njit(cache=True)
def _default_du(c):
    return 1.0 / c


@njit(cache=True)
def _default_uc_inv(x):
    return 1.0 / x


@njit(cache=True)
def _default_ddu(c):
    return -1.0 / (c ** 2)


class RetirementModel:
    """Numerical resources for the retirement choice model.

    Thin rho output: holds a reference to the calibrated period
    dict, constructs the asset grid, and stores compiled equation
    callables.  Parameters are **not** copied — property access
    delegates to the stage's ``.calibration`` and ``.settings``,
    which are the single source of truth.

    Parameters
    ----------
    period : dict
        Canonical period dict with ``"stages"`` key.
    callables : dict, optional
        Override equation callables (keys: ``u``, ``du``,
        ``uc_inv``, ``ddu``).  Defaults to log utility.
    """

    def __init__(self, period, callables=None):
        stages = period["stages"] if "stages" in period else period
        self._work = stages["work_cons"]

        cal = self._work.calibration
        settings = self._work.settings or {}

        b = float(cal.get('b', settings.get('b', 1e-10)))
        grid_max = float(settings.get('grid_max_A', 500))
        n = int(settings.get('grid_size', 3000))

        self.asset_grid_A = np.linspace(b, grid_max, n)
        self.grid_size = n
        self.eulerK = n

        if callables is None:
            callables = {}
        self.u = callables.get('u', _default_u)
        self.du = callables.get('du', _default_du)
        self.uc_inv = callables.get('uc_inv', _default_uc_inv)
        self.ddu = callables.get('ddu', _default_ddu)

    def _get(self, key, default=None):
        """Look up a param in calibration then settings."""
        cal = self._work.calibration
        settings = self._work.settings or {}
        if key in cal:
            return cal[key]
        if key in settings:
            return settings[key]
        if default is not None:
            return default
        raise KeyError(f"'{key}' not in calibration or settings")

    @property
    def r(self):
        return float(self._get('r'))

    @property
    def R(self):
        return 1.0 + self.r

    @property
    def beta(self):
        return float(self._get('beta'))

    @property
    def delta(self):
        return float(self._get('delta'))

    @property
    def y(self):
        return float(self._get('y'))

    @property
    def smooth_sigma(self):
        return float(self._get('smooth_sigma', 0))

    @property
    def T(self):
        return int(self._get('T', 20))

    @property
    def m_bar(self):
        return float(self._get('m_bar', 1.2))

    @property
    def b(self):
        return float(self._get('b', 1e-10))

    @property
    def padding_mbar(self):
        return float(self._get('padding_mbar', 0))

    @property
    def grid_max_A(self):
        return float(self._get('grid_max_A', 500))

    @classmethod
    def from_period(cls, period, equations=None):
        """Construct from a calibrated period dict.

        Convenience alias — the constructor already accepts
        the period dict directly.
        """
        return cls(period, callables=equations)

    @classmethod
    def with_test_defaults(cls, **overrides):
        """Construct with test defaults (for unit tests).

        Builds a mock period dict from scalar defaults
        so tests don't need dolo-plus.
        """
        defaults = dict(
            r=0.02, beta=0.98, delta=1.0, smooth_sigma=0,
            y=20, b=1e-10, grid_max_A=500, grid_size=3000,
            T=20, m_bar=1.2, padding_mbar=0,
        )
        defaults.update(overrides)

        class _MockStage:
            def __init__(self, params):
                self.calibration = {
                    k: params[k] for k in
                    ('r', 'beta', 'delta', 'smooth_sigma',
                     'y', 'b')
                }
                self.settings = {
                    k: params[k] for k in
                    ('grid_max_A', 'grid_size', 'T',
                     'm_bar', 'padding_mbar')
                }

        mock_period = {
            "stages": {"work_cons": _MockStage(defaults)},
        }
        return cls(mock_period)
