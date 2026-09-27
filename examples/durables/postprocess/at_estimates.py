"""Lifecycle profiles and moment fit at the estimated parameters.

A finished estimation run stores the parameter estimates and the loss, but
not a usable solved model (the ``.nst`` files are empty). The functions
below re-solve at ``theta_best``, simulate, and write the 2-by-2 lifecycle
figure, the fit table evaluated at those estimates, and the cohort means.

The estimation's own ``fit_table.csv`` is the last cross-entropy evaluation
on rank 0, not the evaluation at ``theta_best``. The table written here is
the one that can be compared to ``summary.json['objective']``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from kikku.run.estimate import NAN_PENALTY, load_estimation_spec
from kikku.run.moments import _age_group_masks, make_moment_fn

from examples.durables.estimate import (
    check_parameter_names, resolved_calibration_keys,
)
from examples.durables.solve import solve
from examples.durables.solvers.beta_types import expand_types
from examples.durables.solvers.simulate import simulate_lifecycle

# Same prefixes as examples.durables.estimate._LEVEL_PREFIXES: moment keys
# that are levels in model units and must be multiplied by 1/normalisation
# so the loss compares Australian dollars with Australian dollars.
_LEVEL_PREFIXES = (
    "mean_", "sd_", "av_", "cond_discrete_0_mean_",
    "cond_discrete_1_mean_", "cond_discrete_0_sd_",
    "cond_discrete_1_sd_",
)

_MEAN_VARS = ("c", "a", "h")
_GADI_SCRATCH = "/scratch/tp66"


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def _spec_stem(spec_value):
    name = Path(str(spec_value)).name
    return name[:-5] if name.endswith(".yaml") else Path(name).stem


def _as_settings(grid):
    """Accept a cubic integer or a settings dict; return a settings dict or None."""
    if grid is None:
        return None
    if isinstance(grid, dict):
        return dict(grid)
    n = int(grid)
    return {"n_a": n, "n_h": n, "n_w": n}


def _stage(nest):
    return nest["periods"][0]["stages"]["keeper_cons"]


def _is_level_key(key):
    base = key.rsplit("__age", 1)[0] if "__age" in key else key
    return any(base.startswith(p) for p in _LEVEL_PREFIXES)


def _weight(data):
    """kikku's default SMM weight: 1/data^2 when |data| >= 1, else 1."""
    data = float(data)
    if abs(data) >= 1.0:
        return 1.0 / (data * data)
    return 1.0


def load_run(run_dir):
    """Read the estimates, loss and run settings from a results folder.

    Parameters
    ----------
    run_dir : path-like
        Folder that contains ``summary.json`` and, when the run wrote them,
        ``theta_*.json`` and ``manifest.json``.

    Returns
    -------
    dict
        ``theta_best``, ``theta_mean``, ``theta_se``, ``objective``,
        ``converged``, ``n_iter``, ``calib_overrides``, ``registry``,
        ``spec_name``, ``spec_factory``, ``grid``, ``N_sim``,
        ``simulation_seed``, ``solver_method``, ``beta_types``, and the raw
        ``manifest`` (or ``None`` when the file is absent).

        ``registry`` is the basename of ``manifest['mod']``, or the
        ``<mod>`` folder of the results layout ``<mod>/<spec>/est_<id>``.
        ``spec_factory`` is the manifest value, or
        ``spec_factory_males.yaml`` when ``spec_name`` ends with
        ``_males``, otherwise ``spec_factory.yaml``.
    """
    run_dir = Path(run_dir)
    summary = _read_json(run_dir / "summary.json")
    if summary is None:
        raise FileNotFoundError(f"summary.json is required under {run_dir}")

    theta_best = summary.get("theta_best") or _read_json(run_dir / "theta_best.json")
    theta_mean = summary.get("theta_mean") or _read_json(run_dir / "theta_mean.json")
    theta_se = summary.get("theta_se") or _read_json(run_dir / "theta_se.json")

    manifest = _read_json(run_dir / "manifest.json")

    if manifest is not None:
        registry = Path(str(manifest["mod"])).name
        spec_name = _spec_stem(manifest["spec"])
        spec_factory = manifest.get("spec_factory")
        grid = manifest.get("grid")
        n_sim = manifest.get("N_sim")
        simulation_seed = manifest.get("simulation_seed")
        solver_method = manifest.get("solver_method")
    else:
        spec_name = run_dir.parent.name
        registry = run_dir.parent.parent.name
        spec_factory = None
        grid = None
        n_sim = None
        simulation_seed = None
        solver_method = None

    if not spec_factory:
        if str(spec_name).endswith("_males"):
            spec_factory = "spec_factory_males.yaml"
        else:
            spec_factory = "spec_factory.yaml"

    beta_types = summary.get("beta_types")
    if beta_types is None and manifest is not None:
        beta_types = manifest.get("beta_types")
    if beta_types is None:
        beta_types = None

    return {
        "theta_best": theta_best,
        "theta_mean": theta_mean,
        "theta_se": theta_se,
        "objective": summary.get("objective"),
        "converged": summary.get("converged"),
        "n_iter": summary.get("n_iter"),
        "calib_overrides": summary.get("calib_overrides") or (
            manifest.get("calib_overrides") if manifest else None
        ) or {},
        "registry": registry,
        "spec_name": spec_name,
        "spec_factory": spec_factory,
        "grid": grid,
        "N_sim": n_sim,
        "simulation_seed": simulation_seed,
        "solver_method": solver_method,
        "beta_types": beta_types,
        "manifest": manifest,
    }


def types_spec_for_run(run, registry_root):
    """Return the ``estimation.types`` block of the run's spec, or ``None``.

    Parameters
    ----------
    run : dict
        Output of :func:`load_run`.
    registry_root : path-like
        Directory that contains the model registries (for example
        ``examples/durables/syntax``).

    Returns
    -------
    dict or None
        The ``types`` mapping ``{n, parameters: {name: {location, spread,
        transform}}}``, or ``None`` when the spec has no such block.
    """
    spec_name = run["spec_name"]
    filename = spec_name if str(spec_name).endswith(".yaml") else f"{spec_name}.yaml"
    path = Path(registry_root) / run["registry"] / "estimation" / filename
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text()) or {}
    types = (raw.get("estimation") or {}).get("types")
    return types or None


def types_spec_from_record(beta_types):
    """Rebuild the ``types`` block from the ``beta_types`` record of a run.

    The record the driver writes into ``summary.json`` carries, for every
    heterogeneous parameter, the names of its location and spread
    parameters and its transform, which is all ``expand_types`` needs.
    Returns ``None`` when the run had no types.
    """
    if not beta_types or not isinstance(beta_types, dict):
        return None
    params = beta_types.get("parameters") or {}
    if not params or "n" not in beta_types:
        return None
    parameters = {}
    for name, info in params.items():
        if not isinstance(info, dict) or "location" not in info or "spread" not in info:
            return None
        parameters[name] = {
            "location": info["location"],
            "spread": info["spread"],
            "transform": info.get("transform", "logit"),
        }
    return {"n": int(beta_types["n"]), "parameters": parameters}


def _types_spec_for_solve(run, registry_root):
    """The types block to expand ``theta_best`` with, and where it came from.

    The run's own record (``summary.json``) is the authority, because the
    spec file in the registry may have changed since the run. When both
    exist they must agree on ``n`` and on the parameter names; a
    disagreement is an error rather than a silent switch of model.
    """
    from_record = types_spec_from_record(run.get("beta_types"))
    from_file = types_spec_for_run(run, registry_root)
    if from_record is None and from_file is None:
        return None
    if from_record is not None and from_file is not None:
        same_n = int(from_record["n"]) == int(from_file.get("n", -1))
        same_names = set(from_record["parameters"]) == set(from_file.get("parameters") or {})
        if not (same_n and same_names):
            raise ValueError(
                "the run's beta_types record and the registry's spec file disagree "
                f"on the types block (record: {from_record}; file: {from_file}); "
                "re-solving would use a different model from the estimation")
    if from_record is not None:
        return from_record
    # An older run without the record: the file is the only source.
    return from_file


def solve_at_estimates(run, registry_root, grid=None, method_switch=None):
    """Re-solve the model at ``theta_best``.

    Parameters
    ----------
    run : dict
        Output of :func:`load_run`.
    registry_root : path-like
        Directory that contains the model registries.
    grid : dict or int, optional
        Settings overrides (``n_a``, ``n_h``, ``n_w``). An integer is
        read as a cubic grid. When omitted, the run's manifest grid is
        used.
    method_switch : str or dict, optional
        Upper-envelope tag passed to :func:`solve`. When omitted, the
        run's ``solver_method`` is used.

    Returns
    -------
    list of (share, nest, grids)
        One entry with share 1 when the spec has no ``types`` block;
        otherwise one entry per type from :func:`expand_types`.

    Raises
    ------
    ValueError
        If no grid is known (none in the call and none in the run).
    """
    settings = _as_settings(grid) or _as_settings(run.get("grid"))
    if not settings:
        raise ValueError(
            "no grid is known: pass grid= or record n_a, n_h, n_w in "
            "the run's manifest.json")
    if method_switch is None:
        method_switch = run.get("solver_method")

    registry_dir = str(Path(registry_root) / run["registry"])
    factory = run["spec_factory"]
    calib_overrides = dict(run.get("calib_overrides") or {})
    types_spec = _types_spec_for_solve(run, registry_root)

    if types_spec is None:
        members = [(1.0, dict(run["theta_best"]))]
    else:
        members = expand_types(run["theta_best"], types_spec)

    # Every calibration handed to solve() must consist of names the solver
    # will read; solve() ignores unknown names silently (the gamma_c defect).
    keys = resolved_calibration_keys(registry_dir, factory)
    for _share, calib_k in members:
        check_parameter_names(keys, list(calib_k), calib_overrides, None)

    solved = []
    for share, calib_k in members:
        nest, grids = solve(
            registry_dir,
            spec_factory_name=factory,
            method_switch=method_switch,
            draw={
                "calibration": {**calib_overrides, **calib_k},
                "settings": settings,
            },
            verbose=False,
            strip_solved=True,
        )
        solved.append((share, nest, grids))
    return solved


def simulate_at_estimates(run, solved, N=None, seed=None):
    """Simulate the lifecycle at the re-solved estimates.

    Parameters
    ----------
    run : dict
        Output of :func:`load_run`. ``N_sim`` and ``simulation_seed`` are
        used when ``N`` or ``seed`` is omitted.
    solved : list of (share, nest, grids)
        Output of :func:`solve_at_estimates`.
    N, seed : int, optional
        Number of agents and the simulation seed.

    Returns
    -------
    dict
        Panels from :func:`simulate_lifecycle`. One solved model is
        simulated as a single type; several are passed as ``types``.
    """
    if N is None:
        N = run.get("N_sim")
    if seed is None:
        seed = run.get("simulation_seed")
    if N is None:
        N = 10000
    if seed is None:
        seed = 99
    N = int(N)
    seed = int(seed)

    if len(solved) == 1:
        _share, nest, grids = solved[0]
        return simulate_lifecycle(nest, grids, N=N, seed=seed)

    _share0, nest0, grids0 = solved[0]
    return simulate_lifecycle(
        nest0, grids0, N=N, seed=seed, types=list(solved))


def data_points_by_age(registry_dir, spec_name):
    """Data means by age group for consumption, housing and financial assets.

    Parameters
    ----------
    registry_dir : path-like
        Registry folder (for example
        ``examples/durables/syntax/separable``).
    spec_name : str
        Estimation spec stem or filename.

    Returns
    -------
    dict
        ``{var: [(age_position, value_in_AUD), ...]}`` for each target
        with ``stat == 'mean'`` and ``var`` in ``{c, a, h}``. The age
        position is the mean of the rows
        :func:`kikku.run.moments._age_group_masks` assigns to that group,
        so the markers sit on the same years the moments use.
    """
    filename = spec_name if str(spec_name).endswith(".yaml") else f"{spec_name}.yaml"
    spec_path = Path(registry_dir) / "estimation" / filename
    spec = load_estimation_spec(str(spec_path))
    moment_spec = spec["moment_spec"]
    data_moments = spec["data_moments"]
    setup = moment_spec.get("setup") or {}
    t0 = int(setup.get("t0", 0))
    T = int(setup.get("T", 70))
    age_groups = setup.get("age_groups") or {}
    masks = _age_group_masks(T, age_groups, t0)

    points = {var: [] for var in _MEAN_VARS}
    for target in moment_spec.get("targets") or []:
        if str(target.get("stat")) != "mean":
            continue
        var = target.get("var")
        if var not in points:
            continue
        key = str(target["key"])
        for group, rows in masks.items():
            if not rows:
                continue
            full_key = f"{key}__age{group}"
            if full_key not in data_moments:
                continue
            age_pos = float(np.mean(rows))
            points[var].append((age_pos, float(data_moments[full_key])))
    for var in points:
        points[var].sort(key=lambda pair: pair[0])
    return {var: pairs for var, pairs in points.items() if pairs}


def simulated_moments_at_estimates(sim_data, moment_spec, sett):
    """Moments of the simulated panels, levels in Australian dollars.

    Applies :func:`kikku.run.moments.make_moment_fn` and then the same
    level-prefix denormalisation as ``examples.durables.estimate``:
    keys whose base name starts with a level prefix are multiplied by
    ``1 / normalisation``.
    """
    raw = make_moment_fn(moment_spec)(sim_data)
    denorm = 1.0 / float(sett.get("normalisation", 1.0))
    out = {}
    for key, value in raw.items():
        if _is_level_key(key):
            out[key] = float(value) * denorm
        else:
            out[key] = float(value)
    return out


def fit_at_estimates(run, sim_moments, data_moments):
    """Moment-by-moment fit at the estimates.

    Parameters
    ----------
    run : dict
        Output of :func:`load_run` (kept so the row can name the run;
        not used in the arithmetic).
    sim_moments, data_moments : dict
        Simulated and data moments on the same keys. Only keys that
        appear in ``data_moments`` with a finite data value and that
        are also present in ``sim_moments`` are scored.

    Returns
    -------
    list of dict
        Rows ``moment``, ``data``, ``simulated``, ``residual``,
        ``contribution``, ``contribution_pct``, ``penalised``. The
        contribution is ``w * (sim - data)^2`` with ``w = 1/data^2`` when
        ``|data| >= 1`` and ``w = 1`` otherwise, which is kikku's default
        weight. Only keys the moment function produces are scored (a data
        column the model has no counterpart for is not a model moment). A
        produced moment that is NaN (an age group outside the simulated
        horizon, for instance) is scored exactly as kikku's loss scores it:
        contribution ``NAN_PENALTY`` (1e6), ``penalised`` true, residual
        NaN. ``contribution_pct`` is 100 times the share of the sum over the
        finite rows, so the percentages add to 100 over the fit that the
        parameters can move; the penalised rows carry ``contribution_pct``
        NaN. Over the driver's filtered target set, the sum of all
        contributions equals kikku's loss.
    """
    del run  # the run identifies the estimates; the loss uses the two dicts
    keys = sorted(
        k for k, data in data_moments.items()
        if k in sim_moments and np.isfinite(float(data))
    )
    rows = []
    for key in keys:
        data = float(data_moments[key])
        sim = float(sim_moments[key])
        if np.isfinite(sim):
            residual = sim - data
            contribution = _weight(data) * residual * residual
            penalised = False
        else:
            residual = float("nan")
            contribution = float(NAN_PENALTY)
            penalised = True
        rows.append({
            "moment": key,
            "data": data,
            "simulated": sim,
            "residual": residual,
            "contribution": contribution,
            "penalised": penalised,
        })
    finite_total = sum(r["contribution"] for r in rows if not r["penalised"])
    for row in rows:
        if row["penalised"]:
            row["contribution_pct"] = float("nan")
        else:
            row["contribution_pct"] = (
                100.0 * row["contribution"] / finite_total
                if finite_total > 0.0 else 0.0)
    return rows


def plot_lifecycle_at_estimates(sim_data, nest, data_points, out_dir):
    """Draw mean lifecycle profiles in Australian dollars, with data markers.

    The 2-by-2 figure is that of :func:`plot_lifecycle`: consumption,
    housing, financial assets, and the adjustment rate, each the
    cross-sectional mean by age with the 25th-to-75th percentile band.
    Levels are model units divided by the settings' ``normalisation``.
    Data means are drawn as markers on the consumption, housing and
    asset panels. When ``sim_data`` carries ``type_idx``, each type's
    mean is drawn as a thin line.

    Returns
    -------
    pathlib.Path
        Path of ``lifecycle_at_estimates.png``.
    """
    import matplotlib.pyplot as plt
    from examples.durables.postprocess.plots import _nb_theme, _style_nb_ax

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage0 = _stage(nest)
    t0 = int(stage0.calibration["t0"])
    T = int(stage0.settings["T"])
    normalisation = float(stage0.settings.get("normalisation", 1.0))
    to_aud = 1.0 / normalisation
    age = np.arange(t0, T)
    theme = _nb_theme()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    panels = [
        ("c", "Consumption", "AUD"),
        ("h", "Housing", "AUD"),
        ("a", "Financial assets", "AUD"),
    ]
    type_idx = sim_data.get("type_idx")

    for i, (key, title, ylabel) in enumerate(panels):
        ax = axes[i]
        panel = np.asarray(sim_data[key][t0:T, :], dtype=float) * to_aud
        mean = np.nanmean(panel, axis=1)
        p25 = np.nanpercentile(panel, 25, axis=1)
        p75 = np.nanpercentile(panel, 75, axis=1)
        ax.plot(age, mean, lw=1.8, color=theme["accent"], label="simulated mean")
        ax.fill_between(age, p25, p75, alpha=0.18, color=theme["accent"],
                        label="25–75 percentile")
        if type_idx is not None:
            labeled = False
            for k in np.unique(type_idx):
                mask = type_idx == k
                mean_k = np.nanmean(panel[:, mask], axis=1)
                ax.plot(
                    age, mean_k, lw=0.9, alpha=0.75, color=theme["muted"],
                    label="by type" if not labeled else None,
                )
                labeled = True
        for age_pos, value in data_points.get(key, []):
            ax.plot(
                age_pos, value, "o", color=theme["raw"], ms=5.5,
                zorder=5, label="data" if age_pos == data_points[key][0][0] else None,
            )
        ax.set_title(title, fontweight="600")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Age")
        _style_nb_ax(ax)
        if i == 0:
            ax.legend(frameon=True, loc="best")

    ax_adj = axes[3]
    discrete = np.asarray(sim_data["discrete"][t0:T, :], dtype=float)
    adj_rate = np.nanmean(np.where(discrete >= 0, discrete, np.nan), axis=1)
    ax_adj.plot(age, adj_rate * 100.0, lw=1.8, color=theme["accent2"],
                label="simulated mean")
    if type_idx is not None:
        labeled = False
        for k in np.unique(type_idx):
            mask = type_idx == k
            rate_k = np.nanmean(
                np.where(discrete[:, mask] >= 0, discrete[:, mask], np.nan),
                axis=1)
            ax_adj.plot(
                age, rate_k * 100.0, lw=0.9, alpha=0.75, color=theme["muted"],
                label="by type" if not labeled else None,
            )
            labeled = True
    ax_adj.set_title("Adjustment rate", fontweight="600")
    ax_adj.set_ylabel("% adjusting")
    ax_adj.set_xlabel("Age")
    _style_nb_ax(ax_adj)

    fig.tight_layout()
    path = out_dir / "lifecycle_at_estimates.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _cohort_rows(run, registry_root, sim_moments, data_moments):
    """Rows of the cohort-means table: age group, variable, data, simulated."""
    registry_dir = Path(registry_root) / run["registry"]
    filename = (
        run["spec_name"] if str(run["spec_name"]).endswith(".yaml")
        else f"{run['spec_name']}.yaml"
    )
    spec = load_estimation_spec(str(registry_dir / "estimation" / filename))
    setup = spec["moment_spec"].get("setup") or {}
    t0 = int(setup.get("t0", 0))
    T = int(setup.get("T", 70))
    masks = _age_group_masks(T, setup.get("age_groups") or {}, t0)
    rows = []
    for target in spec["moment_spec"].get("targets") or []:
        if str(target.get("stat")) != "mean":
            continue
        var = target.get("var")
        if var not in _MEAN_VARS:
            continue
        key = str(target["key"])
        for group, row_idx in masks.items():
            if not row_idx:
                continue
            full_key = f"{key}__age{group}"
            if full_key not in data_moments or full_key not in sim_moments:
                continue
            rows.append({
                "age group": group,
                "variable": var,
                "data": float(data_moments[full_key]),
                "simulated": float(sim_moments[full_key]),
            })
    return rows


def write_outputs(run, registry_root, out_dir, grid=None, N=None, seed=None):
    """Solve, simulate and write the figure and the two CSV files.

    Parameters
    ----------
    run : dict
        Output of :func:`load_run`.
    registry_root : path-like
        Directory that contains the model registries.
    out_dir : path-like
        Folder for ``lifecycle_at_estimates.png``,
        ``fit_table_at_best.csv`` and ``cohort_means.csv``.
    grid, N, seed : optional
        Forwarded to the solver and the simulator.

    Returns
    -------
    dict
        Paths of the three files, keyed by filename.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    solved = solve_at_estimates(run, registry_root, grid=grid)
    sim_data = simulate_at_estimates(run, solved, N=N, seed=seed)
    nest = solved[0][1]

    registry_dir = Path(registry_root) / run["registry"]
    spec_name = run["spec_name"]
    filename = spec_name if str(spec_name).endswith(".yaml") else f"{spec_name}.yaml"
    spec = load_estimation_spec(str(registry_dir / "estimation" / filename))
    sett = _stage(nest).settings
    sim_moments = simulated_moments_at_estimates(
        sim_data, spec["moment_spec"], sett)
    data_moments = spec["data_moments"]
    data_points = data_points_by_age(registry_dir, spec_name)

    fig_path = plot_lifecycle_at_estimates(sim_data, nest, data_points, out_dir)

    fit_rows = fit_at_estimates(run, sim_moments, data_moments)
    fit_path = out_dir / "fit_table_at_best.csv"
    with fit_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "moment", "data", "simulated", "residual",
                "contribution", "contribution_pct", "penalised",
            ],
        )
        writer.writeheader()
        writer.writerows(fit_rows)
    n_penalised = sum(1 for r in fit_rows if r["penalised"])
    finite_total = sum(r["contribution"] for r in fit_rows if not r["penalised"])
    total = finite_total + n_penalised * float(NAN_PENALTY)
    print(
        f"sum of contributions: {total:.8f}  "
        f"(finite fit {finite_total:.8f} + {n_penalised} penalised moments x "
        f"{NAN_PENALTY:g})  objective: {run['objective']}")

    cohort_rows = _cohort_rows(run, registry_root, sim_moments, data_moments)
    cohort_path = out_dir / "cohort_means.csv"
    with cohort_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["age group", "variable", "data", "simulated"])
        writer.writeheader()
        writer.writerows(cohort_rows)

    return {
        "lifecycle_at_estimates.png": fig_path,
        "fit_table_at_best.csv": fit_path,
        "cohort_means.csv": cohort_path,
    }


def _default_out_dir(run_dir):
    est_name = Path(run_dir).name
    return (
        Path("paper-results") / "durables" / "estimation"
        / date.today().isoformat() / "at_estimates" / est_name
    )


def main(argv=None):
    """Command-line entry: re-solve a finished run and write the outputs."""
    parser = argparse.ArgumentParser(
        description=(
            "Re-solve a finished durables estimation at its reported "
            "estimates and write lifecycle profiles and the fit table."
        ),
    )
    parser.add_argument(
        "run_dir",
        help="Results folder that contains summary.json (an est_<id> directory).",
    )
    parser.add_argument(
        "--registry-root",
        default="examples/durables/syntax",
        help="Directory that contains the model registries (default: examples/durables/syntax).",
    )
    parser.add_argument(
        "--grid", type=int, default=None,
        help="Cubic grid size n_a = n_h = n_w. Default: the run's manifest grid.",
    )
    parser.add_argument(
        "--n-sim", type=int, default=None, dest="n_sim",
        help="Number of simulated agents. Default: the run's N_sim.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Simulation seed. Default: the run's simulation_seed.",
    )
    parser.add_argument(
        "--out", default=None,
        help=(
            "Output folder. Default: "
            "paper-results/durables/estimation/<YYYY-MM-DD>/at_estimates/<est_name>/."
        ),
    )
    args = parser.parse_args(argv)

    if os.path.isdir(_GADI_SCRATCH) and args.out is None:
        raise SystemExit(
            "This machine has /scratch/tp66 and is treated as Gadi. "
            "Pass --out pointing under scratch; the default output path "
            "is inside the repository checkout and must not be used there."
        )

    run = load_run(args.run_dir)
    out_dir = Path(args.out) if args.out is not None else _default_out_dir(args.run_dir)
    write_outputs(
        run, args.registry_root, out_dir,
        grid=args.grid, N=args.n_sim, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
