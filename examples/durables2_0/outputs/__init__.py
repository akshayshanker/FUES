"""Plots, tables, and diagnostics for the durables2_0 example."""

from .diagnostics import (
    derive_savings,
    get_policy,
    get_timing,
    get_solution_at_age,
    compute_euler_stats,
    print_euler_stats,
    consumption_deviation,
)
from .plots import (
    plot_policies,
    plot_grids,
    plot_lifecycle,
)
from .tables import (
    generate_timing_table,
    generate_accuracy_table,
)

__all__ = [
    "derive_savings",
    "get_policy",
    "get_timing",
    "get_solution_at_age",
    "compute_euler_stats",
    "print_euler_stats",
    "consumption_deviation",
    "plot_policies",
    "plot_grids",
    "plot_lifecycle",
    "generate_timing_table",
    "generate_accuracy_table",
]
