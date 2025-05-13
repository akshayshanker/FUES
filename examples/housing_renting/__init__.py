"""
Housing model package for ModCraft.

This package provides implementations of the Housing model with discrete choice
using the StageCraft and Heptapod-B architecture.
"""

from .horses_h import (
    F_shocks_dcsn_to_arvl,
    F_h_cntn_to_dcsn
)
from .horses_c import (
    F_ownc_cntn_to_dcsn,
    F_ownc_dcsn_to_cntn
)
from .horses_common import F_id

from .horses_t import (
    F_t_cntn_to_dcsn
)

from .whisperer import (
    build_operators,
    solve_stage,
    run_time_iteration
)

__all__ = [
    # Horse functions (operator factories)
    'F_shocks_dcsn_to_arvl',
    'F_h_cntn_to_dcsn',
    'F_ownc_cntn_to_dcsn',
    'F_ownc_dcsn_to_cntn',
    'F_id',
    'F_t_cntn_to_dcsn',
    
    # Whisperer functions (external solver)
    'build_operators',
    'solve_stage',
    'run_time_iteration'
] 