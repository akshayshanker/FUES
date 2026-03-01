"""
Simple test to verify that core imports work correctly.
"""

import unittest

from dcsmm.uenvelope import register, get_engine, available, EGM_UE

# Try importing from fues
from dcsmm.fues import FUES, FUES_v0dev, dcegm, rfc, interp_as

# Specific sub-module imports for fues
from dcsmm.fues.helpers import interp_as as fues_interp_as
from dcsmm.fues.fues_v0dev import FUES as FuesFUES
from dcsmm.fues.dcegm import dcegm as FuesDCEGM
from dcsmm.fues.rfc_simple import rfc as FuesRFC

class TestPackageImports(unittest.TestCase):
    def test_imports(self):
        """Test that all core modules can be imported."""
        # Verify items exist
        self.assertTrue(callable(FUES))
        self.assertTrue(callable(fues_interp_as))
        self.assertTrue(callable(EGM_UE))
        
        print("All imports successful!")

if __name__ == "__main__":
    unittest.main() 