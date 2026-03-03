"""Retirement model example - FUES vs DC-EGM comparison."""

from .outputs import (
    plot_egrids, plot_cons_pol, plot_dcegm_cf,
    generate_timing_table_combined, generate_accuracy_table,
    euler, consumption_deviation,
)
from .solve import solve_canonical
from .benchmark import test_Timings

__all__ = [
    'plot_egrids',
    'plot_cons_pol',
    'plot_dcegm_cf',
    'generate_timing_table_combined',
    'generate_accuracy_table',
    'euler',
    'consumption_deviation',
    'solve_canonical',
    'test_Timings',
]
