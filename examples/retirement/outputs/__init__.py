"""Plots, tables, and diagnostics for the retirement example."""

from .diagnostics import (
    get_policy,
    get_timing,
    get_solution_at_age,
    euler,
    consumption_deviation,
)
from .tables import generate_timing_table_combined, generate_accuracy_table

# Plot functions (and therefore seaborn + HARK) load lazily — sweep-only
# installs can import `outputs` without pulling the plotting stack.
from examples._lazy import make_lazy_plot_getter

_LAZY_PLOTS = {
    "plot_egrids", "plot_cons_pol", "plot_dcegm_cf",
    "nb_plot_egm_interactive", "nb_plot_cons_ages", "nb_plot_scaling",
    "nb_plot_egrids", "setup_nb_style",
}
__getattr__ = make_lazy_plot_getter(_LAZY_PLOTS, __name__ + ".plots")

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
