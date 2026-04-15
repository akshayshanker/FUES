"""Shared lazy-import helpers for example packages."""

import importlib


def make_lazy_plot_getter(names, plots_module):
    """Build a PEP 562 ``__getattr__`` that lazy-loads plot functions.

    Parameters
    ----------
    names : set[str]
        Attribute names that live in the plots submodule. Anything else
        raises ``AttributeError`` unchanged.
    plots_module : str
        Absolute import path of the plots submodule, e.g.
        ``"examples.durables.outputs.plots"``. Typically
        ``__name__ + ".plots"`` from the calling ``__init__.py``.
    """
    def __getattr__(name):
        if name in names:
            return getattr(importlib.import_module(plots_module), name)
        raise AttributeError(f"module has no attribute {name!r}")
    return __getattr__
