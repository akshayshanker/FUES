"""Retirement model code -- FUES vs DC-EGM comparison."""

from .retirement import Operator_Factory, RetirementModel
from .solve_block import backward_induction, solve_canonical
from .helpers import (
    plot_egrids, plot_cons_pol, plot_dcegm_cf,
    generate_timing_table_combined, generate_accuracy_table,
    get_policy, get_timing, get_solution_at_age,
    euler, consumption_deviation,
)
from .benchmarks import test_Timings

__all__ = [
    "Operator_Factory",
    "RetirementModel",
    "euler",
    "consumption_deviation",
    "backward_induction",
    "solve_canonical",
    "test_Timings",
    "plot_egrids",
    "plot_cons_pol",
    "plot_dcegm_cf",
    "generate_timing_table_combined",
    "generate_accuracy_table",
    "get_policy",
    "get_timing",
    "get_solution_at_age",
]
