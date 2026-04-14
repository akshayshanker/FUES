"""Plots, tables, and diagnostics for the retirement example."""

from .diagnostics import (
    get_policy,
    get_timing,
    get_solution_at_age,
    euler,
    consumption_deviation,
)
from .tables import generate_timing_table_combined, generate_accuracy_table

# Plot functions (and therefore seaborn + HARK) loaded lazily via PEP 562
# __getattr__ — `from .outputs import plot_egrids` still works at access
# time but sweep-only installs (no seaborn/HARK) can import `outputs`
# without pulling the plotting stack.
_LAZY_PLOTS = {
    "plot_egrids", "plot_cons_pol", "plot_dcegm_cf",
    "nb_plot_egm_interactive", "nb_plot_cons_ages", "nb_plot_scaling",
    "nb_plot_egrids", "setup_nb_style",
}


def __getattr__(name):
    if name in _LAZY_PLOTS:
        from . import plots
        return getattr(plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Paper
    "plot_egrids",
    "plot_cons_pol",
    "plot_dcegm_cf",
    # Notebook
    "nb_plot_egm_interactive",
    "nb_plot_cons_ages",
    "nb_plot_scaling",
    "nb_plot_egrids",
    "setup_nb_style",
    # Tables
    "generate_timing_table_combined",
    "generate_accuracy_table",
    # Diagnostics
    "get_policy",
    "get_timing",
    "get_solution_at_age",
    "euler",
    "consumption_deviation",
]
