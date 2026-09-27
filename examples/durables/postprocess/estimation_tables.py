"""Parameter and moment-fit tables from a finished estimation run.

Reads the JSON and CSV files an estimation writes. Does not import
``at_estimates`` (that module is written separately). Markdown tables
reuse ``_md_table`` from ``tables.py`` (string cells pass through
``_cell``). LaTeX tables use a small writer in the same booktabs style,
because ``_tex_table`` hard-codes a comparison-table label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import date
from pathlib import Path

from .tables import _md_table

PARAMETER_ORDER = [
    "beta",
    "beta_bar",
    "sigma_beta",
    "alpha",
    "gamma_c",
    "gamma_h",
    "rho",
    "tau",
    "theta",
]

LATEX_SYMBOLS = {
    "beta": r"\beta",
    "beta_bar": r"\bar\beta",
    "sigma_beta": r"\sigma_\beta",
    "alpha": r"\alpha",
    "gamma_c": r"\gamma_c",
    "gamma_h": r"\gamma_h",
    "rho": r"\rho",
    "tau": r"\tau",
    "theta": r"\theta",
}

_FOUR_DEC = {"beta", "beta_bar"}
_FOOTER_FIXED = ("loss", "converged", "iterations", "n_samples", "grid")

_FIT_CAPTION_LAST_EVAL = (
    "The simulated column is the last cross-entropy evaluation, "
    "not the evaluation at theta_best."
)
_FIT_CAPTION_AT_BEST = (
    "Moment fit evaluated at the estimated parameters (theta_best)."
)


def _decimals(name):
    return 4 if name in _FOUR_DEC else 3


def _is_finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _smm_weight(data):
    """Default SMM weight of kikku.make_criterion: 1/data^2 if |data| >= 1 else 1."""
    return (1.0 / (data ** 2)) if abs(data) >= 1.0 else 1.0


def read_run(run_dir):
    """Load summary.json and, when present, manifest.json and theta_se.json.

    Parameters
    ----------
    run_dir : path
        Folder of one estimation run (the ``est_*`` directory).

    Returns
    -------
    dict
        ``theta_best``, ``theta_se``, ``objective``, ``converged``,
        ``n_iter``, ``summary``, ``manifest`` (or None), and
        ``beta_types`` (or None). ``summary.json`` is required.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found in {run_dir}")
    summary = json.loads(summary_path.read_text())

    manifest = None
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())

    theta_se = dict(summary.get("theta_se") or {})
    se_path = run_dir / "theta_se.json"
    if se_path.is_file():
        theta_se = json.loads(se_path.read_text())

    beta_types = summary.get("beta_types")
    if not beta_types and manifest is not None:
        beta_types = manifest.get("beta_types")
    if not beta_types:
        beta_types = None

    return {
        "summary": summary,
        "manifest": manifest,
        "theta_best": dict(summary.get("theta_best") or {}),
        "theta_se": theta_se,
        "objective": summary.get("objective"),
        "converged": summary.get("converged"),
        "n_iter": summary.get("n_iter"),
        "beta_types": beta_types,
    }


def _theta_best(run):
    if run.get("theta_best"):
        return run["theta_best"]
    summary = run.get("summary") or {}
    return dict(summary.get("theta_best") or {})


def _theta_se(run):
    if run.get("theta_se") is not None:
        return run["theta_se"]
    summary = run.get("summary") or {}
    return dict(summary.get("theta_se") or {})


def _beta_types(run):
    block = run.get("beta_types")
    if block:
        return block
    summary = run.get("summary") or {}
    if summary.get("beta_types"):
        return summary["beta_types"]
    manifest = run.get("manifest") or {}
    return manifest.get("beta_types") or None


def _manifest(run):
    return run.get("manifest")


def _param_names(runs):
    present = set()
    for _label, run in runs:
        present.update(_theta_best(run).keys())
    ordered = [name for name in PARAMETER_ORDER if name in present]
    extras = sorted(name for name in present if name not in PARAMETER_ORDER)
    return ordered + extras


def _format_best_se(name, best, se):
    """Return ``best (se)`` at the row's decimal count, or ``n/a``."""
    if not _is_finite(best):
        return "n/a"
    nd = _decimals(name)
    best_txt = f"{float(best):.{nd}f}"
    if not _is_finite(se):
        return f"{best_txt} (n/a)"
    se_txt = f"{float(se):.{nd}f}"
    return f"{best_txt} ({se_txt})"


def _param_cell(name, run):
    best = _theta_best(run).get(name)
    if best is None:
        return "n/a"
    return _format_best_se(name, best, _theta_se(run).get(name))


def _fmt_list(values):
    parts = []
    for v in values:
        if _is_finite(v):
            parts.append(f"{float(v):.4f}")
        else:
            parts.append(str(v))
    return ", ".join(parts)


def _types_footer_cell(block):
    """One cell describing the types block the driver writes.

    Shape written by ``examples.durables.estimate.types_summary``:
    ``{n, shares, parameters: {name: {location, spread, transform, nodes,
    implied_mean}}, nodes, implied_mean}`` (the last two repeat the ``beta``
    entry for convenience and are not printed twice). Any other key is
    printed as ``key=value``.
    """
    if not block or not isinstance(block, dict):
        return "n/a"
    params = block.get("parameters")
    shares = block.get("shares")
    n = block.get("n")
    if isinstance(params, dict) and params:
        lines = []
        if n is not None:
            lines.append(f"n={n}")
        for name, info in params.items():
            if isinstance(info, dict):
                values = info.get("nodes", info.get("values"))
                mean = info.get("implied_mean", info.get("mean"))
            else:
                values = info
                mean = None
            bits = [f"{name} nodes"]
            if values is not None:
                try:
                    bits.append(_fmt_list(values))
                except TypeError:
                    bits.append(str(values))
            if shares is not None:
                try:
                    bits.append("shares " + _fmt_list(shares))
                except TypeError:
                    bits.append("shares " + str(shares))
            if mean is not None:
                if _is_finite(mean):
                    bits.append(f"mean {float(mean):.4f}")
                else:
                    bits.append(f"mean {mean}")
            lines.append("; ".join(bits))
        leftover = [
            f"{k}={block[k]}"
            for k in block
            if k not in ("n", "shares", "parameters", "nodes", "implied_mean")
        ]
        if leftover:
            lines.extend(leftover)
        return " | ".join(lines)
    # Fallback: print whatever keys the block has.
    parts = []
    for key, val in block.items():
        parts.append(f"{key}={val}")
    return "; ".join(parts) if parts else "n/a"


def _types_footer_rows(runs, labels):
    """Footer rows for types: one line per listed parameter when possible."""
    # Collect parameter names from every run that has a parameters map.
    param_names = []
    seen = set()
    fallback_needed = False
    for _label, run in runs:
        block = _beta_types(run)
        if not block or not isinstance(block, dict):
            continue
        params = block.get("parameters")
        if isinstance(params, dict) and params:
            for name in params:
                if name not in seen:
                    seen.add(name)
                    param_names.append(name)
        else:
            fallback_needed = True

    rows = []
    for name in param_names:
        row = {"Parameter": f"{name} nodes"}
        for label, run in zip(labels, (r for _, r in runs)):
            block = _beta_types(run)
            cell = "n/a"
            if block and isinstance(block, dict):
                info = (block.get("parameters") or {}).get(name)
                if info is not None:
                    cell = _types_footer_cell(
                        {
                            "n": block.get("n"),
                            "shares": block.get("shares"),
                            "parameters": {name: info},
                        }
                    )
            row[label] = cell
        rows.append(row)

    if fallback_needed or (
        not param_names and any(_beta_types(run) for _, run in runs)
    ):
        row = {"Parameter": "beta_types"}
        for label, run in zip(labels, (r for _, r in runs)):
            block = _beta_types(run)
            row[label] = _types_footer_cell(block) if block else "n/a"
        rows.append(row)
    return rows


def _fixed_footer_rows(runs, labels):
    rows = []
    for key in _FOOTER_FIXED:
        row = {"Parameter": key}
        for label, run in zip(labels, (r for _, r in runs)):
            row[label] = _footer_value(key, run)
        rows.append(row)
    return rows


def _footer_value(key, run):
    if key == "loss":
        obj = run.get("objective")
        if obj is None:
            summary = run.get("summary") or {}
            obj = summary.get("objective")
        if not _is_finite(obj):
            return "n/a"
        return f"{float(obj):.6f}"
    if key == "converged":
        val = run.get("converged")
        if val is None:
            summary = run.get("summary") or {}
            val = summary.get("converged")
        if val is None:
            return "n/a"
        return "True" if val else "False"
    if key == "iterations":
        n_iter = run.get("n_iter")
        if n_iter is None:
            summary = run.get("summary") or {}
            n_iter = summary.get("n_iter")
        if n_iter is None:
            return "n/a"
        return str(n_iter)
    manifest = _manifest(run) or {}
    if key == "n_samples":
        n_samples = manifest.get("n_samples")
        return "n/a" if n_samples is None else str(n_samples)
    if key == "grid":
        grid = manifest.get("grid") or {}
        n_a, n_h, n_w = grid.get("n_a"), grid.get("n_h"), grid.get("n_w")
        if n_a is None or n_h is None or n_w is None:
            return "n/a"
        return f"{n_a} x {n_h} x {n_w}"
    return "n/a"


def _tex_name(name):
    if name in LATEX_SYMBOLS:
        return f"${LATEX_SYMBOLS[name]}$"
    return name


def _string_tex_table(rows, cols, caption, label="tab:estimation"):
    """LaTeX table in the booktabs style of ``tables._tex_table``.

    Cells are already formatted strings (for example ``0.945 (0.003)``).
    The first column is left-aligned; the rest are right-aligned.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
    ]
    if caption:
        lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    col_spec = "l" + "r" * (len(cols) - 1)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(cols) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        cells = [str(row.get(c, "")) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def parameter_table(runs, fmt="md"):
    """Parameter estimates across runs, one column per label.

    Rows are the names in ``PARAMETER_ORDER`` that appear in at least one
    run, then any other free parameter alphabetically. Each cell is
    ``best (se)`` at four decimals for ``beta`` and ``beta_bar`` and three
    otherwise; ``n/a`` when a run does not have that parameter. Footer
    rows report the loss, convergence, iterations, sample count, grid,
    and (when present) the discount-factor type nodes.
    """
    if fmt not in ("md", "tex"):
        raise ValueError(f"fmt must be 'md' or 'tex', got {fmt!r}")
    labels = [label for label, _run in runs]
    names = _param_names(runs)
    body = []
    for name in names:
        display = _tex_name(name) if fmt == "tex" else name
        row = {"Parameter": display}
        for label, run in runs:
            row[label] = _param_cell(name, run)
        body.append(row)
    body.extend(_fixed_footer_rows(runs, labels))
    body.extend(_types_footer_rows(runs, labels))
    cols = ["Parameter"] + labels
    caption = "Estimated parameters"
    if fmt == "md":
        return _md_table(body, cols, caption, params=None)
    tex_cols = ["Parameter"] + labels
    return _string_tex_table(body, tex_cols, caption, label="tab:estimation_parameters")


def _fit_csv_path(run_dir):
    run_dir = Path(run_dir)
    at_best = run_dir / "fit_table_at_best.csv"
    if at_best.is_file():
        return at_best, True
    ordinary = run_dir / "fit_table.csv"
    if not ordinary.is_file():
        raise FileNotFoundError(
            f"neither fit_table_at_best.csv nor fit_table.csv in {run_dir}"
        )
    return ordinary, False


def _read_fit_rows(run_dir):
    """Read the fit CSV and attach SMM-weighted contributions.

    ``kikku.diagnostics`` writes ``contribution`` as the unweighted
    squared residual when it is called without weights (the path
    ``estimate.py`` uses). The table recomputes
    ``w (simulated - data)^2`` with the criterion weights so the
    displayed contribution is the term that enters the SMM loss.
    """
    path, at_best = _fit_csv_path(run_dir)
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            data = float(raw["data"])
            sim = float(raw["simulated"])
            if raw.get("residual") not in (None, ""):
                resid = float(raw["residual"])
            else:
                resid = sim - data
            w = _smm_weight(data)
            contrib = w * (sim - data) ** 2
            rows.append(
                {
                    "moment": raw["moment"],
                    "data": data,
                    "simulated": sim,
                    "residual": resid,
                    "contribution": contrib,
                }
            )
    total = sum(r["contribution"] for r in rows)
    if total > 0 and math.isfinite(total):
        for r in rows:
            r["contribution_pct"] = 100.0 * r["contribution"] / total
    else:
        for r in rows:
            r["contribution_pct"] = 0.0
    rows.sort(key=lambda r: r["contribution"], reverse=True)
    return rows, at_best


def _fmt_fit_display(value):
    """Four significant figures; comma-grouped integers when |value| > 1e4.

    ``'{:.4g}'`` is the rule for ordinary cells (0.2645, 1.182). Above
    1e4 that format switches to scientific notation (1.478e+05); those
    values are printed as rounded integers with thousands separators
    (147,759) so a level moment in Australian dollars stays readable.
    """
    if not _is_finite(value):
        return "n/a"
    v = float(value)
    if abs(v) > 1e4:
        return f"{int(round(v)):,}"
    return f"{v:.4g}"


def _fmt_fit_pct(value):
    """Contribution share of the loss, two decimal places."""
    if not _is_finite(value):
        return "n/a"
    return f"{float(value):.2f}"


def _format_fit_display_rows(rows):
    """Display rows: moment, data, simulated, residual, contribution_pct.

    The raw ``contribution`` column is omitted from the displayed table;
    it remains in the full-precision CSV written by
    :func:`write_estimation_tables`.
    """
    formatted = []
    for r in rows:
        formatted.append(
            {
                "moment": r["moment"],
                "data": _fmt_fit_display(r["data"]),
                "simulated": _fmt_fit_display(r["simulated"]),
                "residual": _fmt_fit_display(r["residual"]),
                "contribution_pct": _fmt_fit_pct(r["contribution_pct"]),
            }
        )
    return formatted


def _write_fit_csv(path, rows):
    """Write the fit rows at full precision, including contribution."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "moment",
                "data",
                "simulated",
                "residual",
                "contribution",
                "contribution_pct",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "moment": r["moment"],
                    "data": r["data"],
                    "simulated": r["simulated"],
                    "residual": r["residual"],
                    "contribution": r["contribution"],
                    "contribution_pct": r["contribution_pct"],
                }
            )
    return path


def fit_table(run_dir, fmt="md", top_n=None):
    """Moment-fit table for one run, sorted by contribution (descending).

    Reads ``fit_table_at_best.csv`` when that file is present (written by
    the lifecycle tool), otherwise ``fit_table.csv``. In the latter case
    the caption states that the simulated column is the last
    cross-entropy evaluation, not the evaluation at ``theta_best``.
    Displayed columns are moment, data, simulated, residual and
    contribution percent (two decimals). Data, simulated and residual
    use four significant figures.
    """
    if fmt not in ("md", "tex"):
        raise ValueError(f"fmt must be 'md' or 'tex', got {fmt!r}")
    rows, at_best = _read_fit_rows(run_dir)
    if top_n is not None:
        rows = rows[: int(top_n)]
    caption = _FIT_CAPTION_AT_BEST if at_best else _FIT_CAPTION_LAST_EVAL
    cols = ["moment", "data", "simulated", "residual", "contribution_pct"]
    formatted = _format_fit_display_rows(rows)
    if fmt == "md":
        return _md_table(formatted, cols, caption, params=None)
    tex_cols = [
        "moment",
        "data",
        "simulated",
        "residual",
        r"contribution \%",
    ]
    tex_rows = []
    for r in formatted:
        tex_rows.append(
            {
                "moment": r["moment"].replace("_", r"\_"),
                "data": r["data"],
                "simulated": r["simulated"],
                "residual": r["residual"],
                r"contribution \%": r["contribution_pct"],
            }
        )
    return _string_tex_table(
        tex_rows, tex_cols, caption, label="tab:estimation_fit"
    )


def write_estimation_tables(runs, out_dir, fits=()):
    """Write ``parameters.md/.tex`` and one fit triple per ``fits`` entry.

    Parameters
    ----------
    runs : list of (label, run_dict)
        Columns of the parameter table; each ``run_dict`` is the return
        of ``read_run``.
    out_dir : path
        Destination folder, created if needed.
    fits : iterable of (label, run_dir)
        Each pair writes ``fit_<label>.md``, ``fit_<label>.tex`` and
        ``fit_<label>.csv``. The CSV keeps full precision, including
        the raw SMM contribution; the Markdown and LaTeX tables are
        the rounded display form.

    Returns
    -------
    list of Path
        Files written, in the order parameters then fits.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    md = parameter_table(runs, fmt="md")
    tex = parameter_table(runs, fmt="tex")
    for name, text in (("parameters.md", md), ("parameters.tex", tex)):
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    for label, run_dir in fits:
        rows, _at_best = _read_fit_rows(run_dir)
        written.append(_write_fit_csv(out_dir / f"fit_{label}.csv", rows))
        for fmt in ("md", "tex"):
            path = out_dir / f"fit_{label}.{fmt}"
            path.write_text(fit_table(run_dir, fmt=fmt), encoding="utf-8")
            written.append(path)
    return written


def _parse_label_path(item):
    if "=" not in item:
        raise SystemExit(f"expected LABEL=RUN_DIR, got {item!r}")
    label, path = item.split("=", 1)
    if not label or not path:
        raise SystemExit(f"expected LABEL=RUN_DIR, got {item!r}")
    return label, path


def main(argv=None):
    """Command line: ``--run LABEL=RUN_DIR ... [--fit LABEL=RUN_DIR ...] --out DIR``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Write estimation parameter and moment-fit tables."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RUN_DIR",
        help="Column of the parameter table. Repeat for several runs.",
    )
    parser.add_argument(
        "--fit",
        action="append",
        default=[],
        metavar="LABEL=RUN_DIR",
        help="Write fit_<label>.md/.tex for this run. Repeat as needed.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output folder. Default: "
            "paper-results/durables/estimation/<YYYY-MM-DD>/tables/"
        ),
    )
    args = parser.parse_args(argv)
    if args.out is None:
        if os.path.isdir("/scratch/tp66"):
            raise SystemExit(
                "This machine looks like Gadi (/scratch/tp66 exists). "
                "Pass --out with a directory under scratch; the default "
                "writes inside the repository."
            )
        today = date.today().isoformat()
        args.out = f"paper-results/durables/estimation/{today}/tables/"
    runs = []
    for item in args.run:
        label, path = _parse_label_path(item)
        runs.append((label, read_run(path)))
    fits = [_parse_label_path(item) for item in args.fit]
    written = write_estimation_tables(runs, args.out, fits=fits)
    for path in written:
        print(path)
    return written


if __name__ == "__main__":
    main()
