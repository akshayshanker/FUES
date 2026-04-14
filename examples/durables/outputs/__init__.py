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

# Plot functions are loaded lazily so importing `outputs` does not pull
# matplotlib/seaborn at module load — keeps Gadi (sweep-only) installs
# free of plotting deps. `from examples.durables.outputs import plot_*`
# still works (PEP 562 __getattr__), but only triggers the seaborn import
# at the moment the attribute is accessed.
_LAZY_PLOTS = {
    "setup_nb_style",
    "nb_plot_policies_comparison",
    "nb_plot_keeper_ages",
    "nb_plot_adjuster_comparison",
    "nb_plot_adjuster_egm",
    "nb_plot_keeper_egm",
    "nb_plot_keeper_policy",
    "nb_plot_value_functions",
    "nb_plot_adjuster_egm_interactive",
    "plot_policies",
    "plot_grids",
    "plot_lifecycle",
    "plot_euler_histogram",
}


def __getattr__(name):
    if name in _LAZY_PLOTS:
        from . import plots
        return getattr(plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
