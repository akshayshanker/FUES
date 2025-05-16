"""
Simple test to verify that core imports work correctly.
"""

import unittest

from dc_smm.uenvelope import register, get_engine, available, EGM_UE

# Try importing from fues
from dc_smm.fues import FUES, FUES_opt, dcegm, rfc, interp_as, upper_envelope

# Specific sub-module imports for fues
from dc_smm.fues.helpers import interp_as as fues_interp_as
from dc_smm.fues.fues import FUES as FuesFUES
from dc_smm.fues.dcegm import dcegm as FuesDCEGM
from dc_smm.fues.rfc_simple import rfc as FuesRFC

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