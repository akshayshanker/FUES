"""Plots, tables, and diagnostics for the retirement example."""

from .diagnostics import (
    get_policy,
    get_timing,
    get_solution_at_age,
    euler,
    consumption_deviation,
)
from .plots import plot_egrids, plot_cons_pol, plot_dcegm_cf
from .tables import generate_timing_table_combined, generate_accuracy_table

__all__ = [
    "plot_egrids",
    "plot_cons_pol",
    "plot_dcegm_cf",
    "generate_timing_table_combined",
    "generate_accuracy_table",
    "get_policy",
    "get_timing",
    "get_solution_at_age",
    "euler",
    "consumption_deviation",
]

