"""Regression test: durables2_0 forward simulation via kikku."""

import os
import sys
import unittest

import numpy as np

# Ensure src/ and repo root are on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SRC_ROOT)

from pathlib import Path
from examples.durables2_0.solve import solve
from examples.durables2_0.simulate import euler_errors

SYNTAX_DIR = Path(REPO_ROOT) / "examples" / "durables2_0" / "syntax"

# Small N for speed; enough to check statistics
N_SIM = 500
SEED = 42


class TestDurables2Simulate(unittest.TestCase):
    """Forward simulation via kikku + inline Euler."""

    @classmethod
    def setUpClass(cls):
        """Solve at t0=50 (10 periods) and simulate."""
        cls.nest, cls.cp, cls.grids, cls.callables = solve(
            SYNTAX_DIR,
            calib_overrides={'t0': 50},
            config_overrides={'n_a': 30, 'n_h': 30, 'n_w': 30},
        )
        cls.euler, cls.sim_data = euler_errors(
            cls.nest, cls.cp, cls.grids, cls.callables,
            N=N_SIM, seed=SEED,
        )

    def test_nest_has_topology(self):
        """solve() stores graph and inter_conn in nest."""
        self.assertIn('graph', self.nest)
        self.assertIn('inter_conn', self.nest)

    def test_output_shapes(self):
        """Euler and sim_data arrays have shape (T, N)."""
        T = self.cp.T
        self.assertEqual(self.euler.shape, (T, N_SIM))
        for key in ('a', 'h', 'c', 'y', 'z_idx', 'discrete',
                     'a_nxt', 'h_nxt'):
            self.assertIn(key, self.sim_data)
            self.assertEqual(self.sim_data[key].shape[1], N_SIM)

    def test_nonzero_adjustment_rate(self):
        """Adjustment rate should be between 5% and 95%."""
        d = self.sim_data['discrete']
        valid = d[d >= 0]
        adj_rate = np.mean(valid)
        self.assertGreater(adj_rate, 0.05,
                           f"Adj rate too low: {adj_rate:.3f}")
        self.assertLess(adj_rate, 0.95,
                        f"Adj rate too high: {adj_rate:.3f}")

    def test_euler_finite(self):
        """At least 10% of (agent, period) cells should have finite Euler."""
        valid = self.euler[~np.isnan(self.euler)]
        T_sim = self.cp.T - self.cp.t0
        min_expected = int(0.10 * T_sim * N_SIM)
        self.assertGreater(
            len(valid), min_expected,
            f"Only {len(valid)} finite Euler (need >{min_expected})")

    def test_keeper_euler_reasonable(self):
        """Keeper Euler errors should be < -1 on average (log10)."""
        d = self.sim_data['discrete']
        keep_mask = (d == 0) & ~np.isnan(self.euler)
        if not np.any(keep_mask):
            self.skipTest("No keeper Euler errors")
        mean_keeper = np.mean(self.euler[keep_mask])
        self.assertLess(mean_keeper, -1.0,
                        f"Keeper Euler mean too large: {mean_keeper:.3f}")

    def test_assets_bounded(self):
        """Simulated assets stay in [b, grid_max_A]."""
        a = self.sim_data['a']
        valid = a[~np.isnan(a)]
        self.assertTrue(np.all(valid >= 0),
                        "Negative assets found")
        self.assertTrue(np.all(valid <= self.cp.grid_max_A + 1),
                        "Assets exceed grid_max_A")

    def test_utility_stats_present(self):
        """sim_data includes NPV utility and branch counts."""
        for key in ('npv_utility', 'npv_utility_adj',
                     'npv_utility_keep', 'n_adj_periods',
                     'n_keep_periods'):
            self.assertIn(key, self.sim_data)
            self.assertEqual(len(self.sim_data[key]), N_SIM)

    def test_consumption_positive(self):
        """Simulated consumption should be positive where computed."""
        c = self.sim_data['c']
        valid = c[~np.isnan(c)]
        self.assertTrue(np.all(valid > 0),
                        "Non-positive consumption found")


if __name__ == '__main__':
    unittest.main()
