"""Fast upper-envelope algorithms."""
from .fues import FUES  # Current production
from .fues_v0_2dev import FUES_jit  # JIT-compiled entry point
from .fues_v0dev import FUES as FUES_v0dev  # Original paper version
from .helpers import interp_as  # re-export

try:
    from .dcegm import dcegm
except ImportError:
    dcegm = None

try:
    from .rfc_simple import rfc
except ImportError:
    rfc = None

__all__ = ["FUES", "FUES_v0dev", "dcegm", "rfc", "interp_as"]
