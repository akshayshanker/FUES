"""FUES package
Fast Upper Envelope Scan implementation and related utilities.

After installing, you can do for example:

>>> from FUES import FUES, uniqueEG
"""

from .FUES import FUES, uniqueEG  # noqa: F401
from .FUES2 import FUES as FUES2  # noqa: F401
from .DCEGM import dcegm  # noqa: F401
from .RFC_simple import rfc  # noqa: F401
from . import math_funcs  # re‑export entire module

__all__ = [
    "FUES",
    "FUES2",
    "uniqueEG",
    "dcegm",
    "rfc",
    "math_funcs",
] 