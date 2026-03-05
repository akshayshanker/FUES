"""Plots, tables, and diagnostics for the retirement example."""

from .diagnostics import (
    get_policy,
    get_timing,
    get_solution_at_age,
    euler,
    consumption_deviation,
)
from .plots import (
    # Paper figures
    plot_egrids, plot_cons_pol, plot_dcegm_cf,
    # Notebook (interactive / exploratory)
    nb_plot_egm_interactive, nb_plot_cons_ages, nb_plot_scaling,
    nb_plot_egrids,
    # Style
    setup_nb_style,
)
from .tables import generate_timing_table_combined, generate_accuracy_table

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
