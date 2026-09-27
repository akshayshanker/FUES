#!/usr/bin/env python3
"""Collect every durables estimation result into one table.

Walks  <results>/<mod>/<spec>/est_<run>/summary.json           (single runs)
       <results>/<mod>/<spec>/<point>/est_<run>/summary.json   (sweep points)
       <results>/<mod>/<spec>/manifest_est_<run>.json          (written by the PBS
                                                                scripts since 28 Sep 2026)
and, when a log directory is given, <logs>/<job id>.gadi-pbs.OU for service
units, walltime and exit status. Variant fields (method, sex, CE seed, N_wage,
h_max, N_SIM, ranks) come from the manifest when present and from the spec
name otherwise, so runs made before manifests existed are still classified.

Usage:
    scripts/collect_estimation_results.py                      # Gadi defaults
    scripts/collect_estimation_results.py --results ~/gadi/g/data/tp66/results/durables/estimation \
        --logs ~/gadi/g/data/tp66/logs/durables --out results_table.csv --md results_table.md
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

DEFAULT_RESULTS = "/g/data/tp66/results/durables/estimation"
DEFAULT_LOGS = "/g/data/tp66/logs/durables"
PARAMS = ["alpha", "beta", "gamma_c", "gamma_h", "tau"]
COLUMNS = (["mod", "spec", "run_id", "sweep_point", "job_id", "job_name", "started", "purpose",
            "method", "sex", "seed", "n_wage", "h_max", "w_max", "n_sim", "nranks", "grid",
            "converged", "n_iter", "objective"] + PARAMS + [f"se_{p}" for p in PARAMS]
           + ["su", "walltime_h", "exit_status", "git_commit", "extra_settings", "path"])


def hms(s):
    h, m, sec = s.split(":")
    return int(h) + int(m) / 60 + int(sec) / 3600


def classify(spec, manifest):
    """Variant fields from the manifest, falling back to the spec name."""
    extra = dict(kv.split("=", 1) for kv in manifest.get("extra_settings", "").split(";") if "=" in kv)
    seed = re.search(r"_seed(\d+)", spec)
    nwage = extra.get("N_wage") or (re.search(r"_nwage(\d+)", spec) or [None, None])[1]
    hmax = extra.get("h_max")
    if hmax is None:
        m = re.search(r"_hmax(\d+)(p(\d+))?", spec)
        hmax = (m.group(1) + ("." + m.group(3) if m.group(3) else "")) if m else None
    methods = manifest.get("methods", "")
    method = "NEGM" if ("NEGM" in methods or "negm" in spec) else "EGM"
    return {
        "method": method,
        "sex": "males" if "males" in spec else "females",
        "seed": seed.group(1) if seed else "42",
        "n_wage": nwage or "4",
        "h_max": hmax or "10",
        "w_max": extra.get("w_max", "12.5"),
        "n_sim": manifest.get("n_sim") or ("20000" if "nsim20k" in spec else "10000"),
        "nranks": manifest.get("nranks", ""),
        "grid": manifest.get("grid", ""),
        "extra_settings": manifest.get("extra_settings", ""),
    }


def read_log(logs, job_id):
    """Service units, walltime and exit status from the PBS output log, if present."""
    if not job_id or not logs:
        return {}
    p = Path(logs) / f"{job_id}.gadi-pbs.OU"
    if not p.exists():
        return {}
    t = p.read_text(errors="replace")
    g = lambda pat: (re.search(pat, t) or [None, ""])[1]
    wall = g(r"Walltime Used:\s+([\d:]+)")
    return {"su": g(r"Service Units:\s+([\d.]+)"), "walltime_h": f"{hms(wall):.2f}" if wall else "",
            "exit_status": g(r"Exit Status:\s+(\S.*?)\s*\n").strip()}


def collect(results, logs):
    rows = []
    results = Path(results)
    for summary in sorted(results.glob("*/*/**/est_*/summary.json")):
        run_dir = summary.parent
        run_id = run_dir.name
        rel = run_dir.relative_to(results).parts          # mod, spec, [point], est_...
        mod, spec = rel[0], rel[1]
        point = rel[2] if len(rel) == 4 else ""
        # one manifest per job that worked on this run (a resumed run has several); take the latest
        manifests = sorted((results / mod / spec).glob(f"manifest_{run_id}*.json"), key=lambda q: q.stat().st_mtime)
        manifest = json.loads(manifests[-1].read_text()) if manifests else {}
        if len(manifests) > 1:
            manifest["job_id"] = "+".join(json.loads(m.read_text()).get("job_id", "") for m in manifests)
        s = json.loads(summary.read_text())
        row = {"mod": mod, "spec": spec, "run_id": run_id, "sweep_point": point or s.get("sweep_point") or "",
               "job_id": manifest.get("job_id", ""), "job_name": manifest.get("job_name", ""),
               "started": manifest.get("started", ""), "purpose": manifest.get("purpose", ""),
               "converged": s.get("converged"), "n_iter": s.get("n_iter"), "objective": s.get("objective"),
               "git_commit": manifest.get("git_commit", ""), "path": str(run_dir)}
        row.update(classify(spec, manifest))
        for p_ in PARAMS:
            row[p_] = (s.get("theta_best") or {}).get(p_)
            row[f"se_{p_}"] = (s.get("theta_se") or {}).get(p_)
        row.update(read_log(logs, row["job_id"]))
        rows.append({c: row.get(c, "") for c in COLUMNS})
    return rows


def write_md(rows, path):
    cols = ["spec", "sweep_point", "run_id", "method", "sex", "seed", "n_wage", "h_max", "n_sim",
            "converged", "n_iter", "objective"] + PARAMS + ["su"]
    fmt = lambda v: (f"{v:.4f}" if isinstance(v, float) else str(v)) if v not in (None, "") else ""
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--logs", default=DEFAULT_LOGS)
    ap.add_argument("--out", default="results_table.csv")
    ap.add_argument("--md", default=None, help="also write a Markdown table here")
    a = ap.parse_args()
    rows = collect(a.results, a.logs if Path(a.logs).exists() else None)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    if a.md:
        write_md(rows, a.md)
    print(f"{len(rows)} runs -> {a.out}" + (f", {a.md}" if a.md else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
