"""
Low-level numerical helpers (interpolation, rootsearch, …).
"""
from .math_funcs import *           # re-export everything
__all__ = [n for n in globals() if not n.startswith("_")] 