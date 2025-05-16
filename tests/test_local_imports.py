"""
Test file that checks imports using relative paths.
"""

import sys
import os
import unittest

# Add src to path
# This is to allow running tests directly from the tests/ directory
# without installing the package. In a real scenario, the package
# would be installed, and these path manipulations wouldn't be necessary.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Try importing from uenvelope
from dc_smm.uenvelope import register, get_engine, available, EGM_UE

# Try importing from fues
from dc_smm.fues import FUES, FUES_opt, dcegm, rfc, interp_as, upper_envelope

# Specific sub-module imports for fues
from dc_smm.fues.helpers import interp_as as fues_interp_as
from dc_smm.fues.fues import FUES as FuesFUES
from dc_smm.fues.dcegm import dcegm as FuesDCEGM
from dc_smm.fues.rfc_simple import rfc as FuesRFC


class TestLocalImports(unittest.TestCase):
    def test_core_imports(self):
        """Test that core modules can be imported using relative paths."""
        # Verify items exist
        self.assertTrue(callable(FuesFUES))
        self.assertTrue(callable(fues_interp_as))
        self.assertTrue(callable(EGM_UE))
        self.assertTrue(callable(register))
        self.assertTrue(callable(FuesDCEGM))
        self.assertTrue(callable(FuesRFC))
        self.assertTrue(callable(FUES_opt))
        self.assertTrue(callable(upper_envelope))

        print("All imports in TestLocalImports successful!")

def test_imports(): # Keep this function for now if other parts of the codebase call it directly
    """Test that core modules can be imported using relative paths."""
    # Verify items exist
    assert callable(FuesFUES)
    assert callable(fues_interp_as)
    assert callable(EGM_UE)
    assert callable(register)
    assert callable(FuesDCEGM)
    assert callable(FuesRFC)
    assert callable(FUES_opt)
    assert callable(upper_envelope)
    print("test_imports() successful!")

if __name__ == "__main__":
    unittest.main() 