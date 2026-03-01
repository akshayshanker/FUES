"""Retirement model code — FUES vs DC-EGM comparison."""

from .retirement import Operator_Factory, RetirementModel, euler, consumption_deviation
from .benchmarks import test_Timings
from .plots import (
    plot_egrids, plot_cons_pol, plot_dcegm_cf,
    generate_timing_table_combined, generate_accuracy_table,
)

__all__ = [
    "Operator_Factory",
    "RetirementModel",
    "euler",
    "consumption_deviation",
    "test_Timings",
    "plot_egrids",
    "plot_cons_pol",
    "plot_dcegm_cf",
    "generate_timing_table_combined",
    "generate_accuracy_table",
]
