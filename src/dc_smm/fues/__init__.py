"""Fast upper-envelope algorithms."""
from .fues import FUES  # Current production (was fues_current)
from .fues_v0dev import FUES as FUES_v0dev  # Original paper version
from .dcegm import dcegm
from .rfc_simple import rfc
from .helpers import interp_as, upper_envelope   # re-export

__all__ = ["FUES", "FUES_v0dev", "dcegm", "rfc",
           "interp_as", "upper_envelope"]
