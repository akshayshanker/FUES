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
from dcsmm.uenvelope import register, get_engine, available, EGM_UE

# Try importing from fues
from dcsmm.fues import FUES, FUES_v0dev, dcegm, rfc, interp_as

# Specific sub-module imports for fues
from dcsmm.fues.helpers import interp_as as fues_interp_as
from dcsmm.fues.fues_v0dev import FUES as FuesFUES
from dcsmm.fues.dcegm import dcegm as FuesDCEGM
from dcsmm.fues.rfc_simple import rfc as FuesRFC


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
        self.assertTrue(callable(FUES_v0dev))
        self.assertTrue(callable(interp_as))

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
    assert callable(FUES_v0dev)
    assert callable(interp_as)
    print("test_imports() successful!")

if __name__ == "__main__":
    unittest.main() 