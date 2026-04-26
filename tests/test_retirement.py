"""Smoke test: solve the retirement model with each UE method and check Euler errors."""

import os
import sys
import unittest

# Ensure src/ and repo root are on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SRC_ROOT)

from pathlib import Path
from examples.retirement.solve import solve_nest
from examples.retirement.outputs import euler, get_policy, get_timing

SYNTAX_DIR = Path(REPO_ROOT) / "examples" / "retirement" / "syntax"
UE_METHODS = ("FUES", "RFC", "DCEGM", "CONSAV")

# Euler error threshold (log10 scale; -1.0 is generous, typical is ~ -1.5)
EULER_THRESHOLD = -1.0
GRID_SIZE = 500  # small grid for speed


class TestRetirementModel(unittest.TestCase):
    """Solve the retirement model and verify Euler errors for each UE method."""

    @classmethod
    def setUpClass(cls):
        """Solve once per method (warmup + timed) and cache results."""
        cls.solutions = {}
        cls.models = {}
        for method in UE_METHODS:
            # Warmup (JIT compile)
            _, m_, ops_, w_ = solve_nest(
                SYNTAX_DIR, method_switch=method,
                draw={"settings": {"grid_size": GRID_SIZE}},
            )
            # Timed run (reuse model + ops)
            nest, model, _, _ = solve_nest(
                SYNTAX_DIR, method_switch=method,
                draw={"settings": {"grid_size": GRID_SIZE}},
                model=m_, stage_ops=ops_, waves=w_,
            )
            cls.solutions[method] = nest
            cls.models[method] = model

    def test_euler_errors(self):
        """Each method should produce Euler errors below the threshold."""
        for method in UE_METHODS:
            with self.subTest(method=method):
                model = self.models[method]
                c_refined = get_policy(
                    self.solutions[method], "c", stage="labour_mkt_decision"
                )
                err = euler(model, c_refined)
                self.assertLess(
                    err, EULER_THRESHOLD,
                    f"{method}: Euler error {err:.4f} exceeds threshold {EULER_THRESHOLD}",
                )

    def test_solution_structure(self):
        """Verify that each solve returns the expected solution keys."""
        expected_stages = {"retire_cons", "work_cons", "labour_mkt_decision"}
        for method in UE_METHODS:
            with self.subTest(method=method):
                nest = self.solutions[method]
                self.assertIn("solutions", nest)
                self.assertGreater(len(nest["solutions"]), 0)
                sol_keys = set(nest["solutions"][-1].keys())
                for stage in expected_stages:
                    self.assertIn(stage, sol_keys)

    def test_timing_positive(self):
        """UE and total solve times should be positive."""
        for method in UE_METHODS:
            with self.subTest(method=method):
                ue_time, total_time = get_timing(self.solutions[method])
                self.assertGreater(ue_time, 0.0)
                self.assertGreater(total_time, 0.0)

    def test_policies_finite(self):
        """Consumption and value policies should contain no NaN/Inf."""
        import numpy as np
        for method in UE_METHODS:
            with self.subTest(method=method):
                nest = self.solutions[method]
                c = get_policy(nest, "c", stage="labour_mkt_decision")
                self.assertTrue(
                    np.all(np.isfinite(c)),
                    f"{method}: consumption policy contains NaN or Inf",
                )


if __name__ == "__main__":
    unittest.main()
