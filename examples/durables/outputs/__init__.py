"""Plots, tables, and diagnostics for the durables example."""

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
    setup_nb_style,
    nb_plot_policies_comparison,
    nb_plot_keeper_ages,
    nb_plot_adjuster_comparison,
    nb_plot_adjuster_egm,
    nb_plot_keeper_egm,
    nb_plot_keeper_policy,
    nb_plot_value_functions,
    nb_plot_adjuster_egm_interactive,
    plot_policies,
    plot_grids,
    plot_lifecycle,
    plot_euler_histogram,
)
from .notebook import (
    FilteredStdout, ce_utility,
    print_solve_summary, build_comparison_row,
)
from .tables import (
    format_euler_detail,
    generate_comparison_table,
    generate_vertical_comparison,
    generate_cohort_table,
    generate_sweep_table,
    write_euler_detail,
)

__all__ = [
    "derive_savings",
    "get_policy",
    "get_timing",
    "get_solution_at_age",
    "compute_euler_stats",
    "print_euler_stats",
    "consumption_deviation",
    "setup_nb_style",
    "plot_policies",
    "plot_grids",
    "plot_lifecycle",
    "plot_euler_histogram",
    "format_euler_detail",
    "generate_comparison_table",
    "generate_vertical_comparison",
    "generate_sweep_table",
    "write_euler_detail",
]
