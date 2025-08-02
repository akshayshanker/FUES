"""Income shock decomposition module."""

from .income_decomposition import (
    create_income_shock_decomposition,
    compute_income_statistics,
    get_state_indices,
    get_combined_index
)

__all__ = [
    'create_income_shock_decomposition',
    'compute_income_statistics',
    'get_state_indices', 
    'get_combined_index'
]