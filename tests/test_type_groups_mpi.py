"""Layered MPI communicator for discount-factor types: runs of the real driver.

Check J of the Cobb-Douglas estimation and post-processing spec
(``AI/devspecs/28092026/cd_estimation_and_postprocess.md``, Sections 5.6 and
7-J). Every MPI test launches ``examples.durables.estimate`` under the venv's
own ``mpiexec`` (MPICH, arm64) with ``python -m mpi4py`` so that an uncaught
exception on one rank aborts the whole job instead of hanging it. The model
is the separable registry at a 30^3 grid, ``t0 = 55`` (15 periods), 500
simulated agents, two cross-entropy iterations, two elite candidates.

The committed types spec has ``n: 4``; with four ranks that would be a single
group and a single candidate per iteration, so the tests run a copy of the
spec under the pytest temporary directory with ``n: 2`` (two groups of two
ranks, two candidates per iteration), the data file made absolute.

Oracles
-------
* J1: the serial driver with ``--n-samples 2`` evaluates the same candidate
  draws (``sampling_seed`` 42) to the same losses, so ``theta_best.json`` and
  ``convergence.csv`` must be byte-identical and ``summary.json`` equal.
* J2: with two candidates and ``n_elite`` 2, one failed candidate at
  iteration 0 leaves ``elite_mean_loss_0 == (best_loss_0 + 1e10) / 2``
  exactly (kikku's ``np.mean`` of two float64 values, ``BIG_LOSS = 1e10``).
* J3: when every evaluation fails, ``result.objective`` is the penalty and the
  driver exits with code 3 (spec 5.2 item 5).
* J4: kikku restores the sampling RNG state from the checkpoint, so a
  restarted run draws the same iteration-1 candidates as a single segment.
* J5: a two-point sweep on eight ranks gives two sub-communicators of two
  groups of two; one summary row per point.
* J6: the divisibility check raises on every rank before any ``Split`` and
  before any output folder exists.
* The no-types spec never splits: ``type_groups.SPLIT_CALLS`` stays 0 on
  every rank of a two-rank run of the driver (spec 3.1, check L).

Test hook: ``FUES_TYPES_FORCE_FAIL`` (documented in
``examples/durables/type_groups.py``) forces the named failure.
"""

import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PYTHON = sys.executable
MPIEXEC = Path(sys.executable).parent / "mpiexec"
HAS_MPI4PY = importlib.util.find_spec("mpi4py") is not None

mpi = pytest.mark.skipif(
    not (HAS_MPI4PY and MPIEXEC.exists()),
    reason=(
        "the MPI tests need mpi4py and the venv's own MPI launcher "
        f"({MPIEXEC}; MPICH installed into the venv). The x86_64 Open MPI "
        "under /usr/local/bin cannot launch this interpreter."
    ),
)

SEPARABLE = REPO / "examples" / "durables" / "syntax" / "separable"
TYPES_SPEC = SEPARABLE / "estimation" / "baseline_large_egm_types.yaml"
BASE_SPEC = SEPARABLE / "estimation" / "baseline_large_egm.yaml"
DATA_FILE = SEPARABLE / "estimation" / "moments_data.csv"
FIXTURES_TYPES = REPO / "tests" / "fixtures" / "estimation_run_types"

GRID = ["n_a=30", "n_h=30", "n_w=30"]
T0 = 55
N_SIM = 500
BIG_LOSS = 1e10  # kikku.run.estimate.BIG_LOSS
TIMEOUT = 1200

# The two-quantile normal nodes for K = 2: Phi^{-1}(0.25), Phi^{-1}(0.75).
Z2 = np.array([-0.6744897501960817, 0.6744897501960817])


def _env(extra=None):
    env = dict(os.environ)
    env.update({
        "NUMBA_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
        "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR", "/tmp/numba-cache-E2"),
    })
    env.pop("FUES_TYPES_FORCE_FAIL", None)
    env.update(extra or {})
    return env


def _spec_copy(path, source, *, n_types=None, sweep=None, drop_free=()):
    """A copy of ``source`` under the test directory with an absolute data file."""
    raw = yaml.safe_load(source.read_text())
    raw["moments"]["data_file"] = str(DATA_FILE)
    if n_types is not None:
        raw["estimation"]["types"]["n"] = int(n_types)
    for name in drop_free:
        del raw["estimation"]["free"][name]
    if sweep is not None:
        raw["estimation"]["sweep"] = sweep
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def _estimate(base, spec, *, ranks, run_id, extra=(), env=None, code=None):
    """Run the driver; return (CompletedProcess, results root ``base/results``).

    ``ranks`` None runs the plain interpreter (serial). ``code`` replaces
    ``-m examples.durables.estimate`` by ``-c code`` (the code receives the
    same command line in ``sys.argv[1:]``).
    """
    scratch = base / "scratch"
    results = base / "results"
    if code is None:
        entry = ["-m", "examples.durables.estimate"]
    else:
        entry = ["-c", code]
    if ranks is None:
        cmd = [PYTHON, *entry]
    else:
        cmd = [str(MPIEXEC), "-n", str(ranks), PYTHON, "-m", "mpi4py", *entry]
    cmd += [
        "--mod", "syntax/separable", "--spec", str(spec),
        "--settings-override", *GRID,
        "--params-override", f"t0={T0}",
        "--N-sim", str(N_SIM), "--max-iter", "2", "--n-elite", "2",
        "--scratch", str(scratch), "--results", str(results),
        "--local-results", "none", "--run-id", run_id, *extra,
    ]
    proc = subprocess.run(
        cmd, cwd=str(REPO), env=_env(env), capture_output=True, text=True,
        timeout=TIMEOUT,
    )
    # Preserve the driver's output beside its results for inspection.
    base.mkdir(parents=True, exist_ok=True)
    (base / "driver_command.txt").write_text(" ".join(cmd) + "\n")
    (base / "driver_stdout.txt").write_text(proc.stdout)
    (base / "driver_stderr.txt").write_text(proc.stderr)
    return proc, results


def _run_dir(results, spec, run_id, sub_label=None):
    parts = [results, "separable", Path(spec).stem]
    if sub_label:
        parts.append(sub_label)
    parts.append(f"est_{run_id}")
    return Path(os.path.join(*map(str, parts)))


def _report(proc):
    return (f"exit {proc.returncode}\n--- stdout (tail) ---\n{proc.stdout[-6000:]}"
            f"\n--- stderr (tail) ---\n{proc.stderr[-6000:]}")


def _convergence(run_dir):
    with (run_dir / "convergence.csv").open(newline="") as f:
        return [(int(r["iter"]), float(r["best_loss"]), float(r["elite_mean_loss"]))
                for r in csv.DictReader(f)]


def _stop_lines(stdout):
    return re.findall(r"\[types\] worker rank=\d+ received STOP after (\d+) evaluations", stdout)


def _member_time_lines(stdout):
    return [line for line in stdout.splitlines() if "member solve times" in line]


def _check_beta_types_block(block, n, theta_best):
    """The block records n, equal shares, increasing nodes and the implied mean."""
    assert block["n"] == n
    assert block["shares"] == [pytest.approx(1.0 / n)] * n
    nodes = block["nodes"]
    assert len(nodes) == n and all(0.0 < b < 1.0 for b in nodes)
    # Nodes increase in beta; they coincide only at zero spread.
    assert np.all(np.diff(nodes) >= 0)
    if theta_best["sigma_beta"] > 0.0:
        assert np.all(np.diff(nodes) > 0)
    assert block["implied_mean"] == pytest.approx(float(np.dot(block["shares"], nodes)))
    beta = block["parameters"]["beta"]
    assert beta["location"] == "beta_bar" and beta["spread"] == "sigma_beta"
    assert beta["transform"] == "logit"
    assert beta["nodes"] == nodes and beta["implied_mean"] == block["implied_mean"]
    # Closed form for K = 2 on the logit scale (spec 5.6), written here rather
    # than taken from beta_types.py.
    if n == 2:
        beta_bar, sigma_beta = theta_best["beta_bar"], theta_best["sigma_beta"]
        x = np.log(1.0 / beta_bar - 1.0) + sigma_beta * Z2
        expected = np.sort(1.0 / (1.0 + np.exp(x)))
        np.testing.assert_allclose(nodes, expected, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Shared runs (module scope: each MPI run costs about a minute)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def work(tmp_path_factory):
    root = tmp_path_factory.mktemp("types_mpi")
    return {
        "root": root,
        "spec2": _spec_copy(root / "types_k2.yaml", TYPES_SPEC, n_types=2),
        "no_types": _spec_copy(root / "no_types.yaml", BASE_SPEC),
    }


@pytest.fixture(scope="module")
def mpi_run(work):
    """J1 reference: four ranks, K = 2, two groups, two candidates per iteration."""
    proc, results = _estimate(work["root"] / "j1_mpi", work["spec2"], ranks=4, run_id="j1mpi")
    assert proc.returncode == 0, _report(proc)
    return proc, _run_dir(results, work["spec2"], "j1mpi")


@pytest.fixture(scope="module")
def serial_run(work):
    """The same estimation in serial: K members solved in sequence inside the trial."""
    proc, results = _estimate(work["root"] / "j1_serial", work["spec2"], ranks=None,
                              run_id="j1serial", extra=["--n-samples", "2"])
    assert proc.returncode == 0, _report(proc)
    return proc, _run_dir(results, work["spec2"], "j1serial")


# ---------------------------------------------------------------------------
# J1: the layered run reproduces the serial run
# ---------------------------------------------------------------------------

@mpi
def test_j1_layered_run_reproduces_the_serial_run(mpi_run, serial_run):
    proc_mpi, mpi_dir = mpi_run
    proc_ser, ser_dir = serial_run
    for fname in ("theta_best.json", "convergence.csv"):
        assert (mpi_dir / fname).read_bytes() == (ser_dir / fname).read_bytes(), fname
    summary_mpi = json.loads((mpi_dir / "summary.json").read_text())
    summary_ser = json.loads((ser_dir / "summary.json").read_text())
    # No field of summary.json depends on the run id; compare whole.
    assert summary_mpi == summary_ser
    assert summary_mpi["objective"] < BIG_LOSS
    _check_beta_types_block(summary_mpi["beta_types"], 2, summary_mpi["theta_best"])

    # theta carries the location and spread, never beta itself.
    theta = json.loads((mpi_dir / "theta_best.json").read_text())
    assert {"beta_bar", "sigma_beta"} <= set(theta) and "beta" not in theta

    # The manifest records the candidate count kikku saw (two group roots),
    # the merged --max-iter, and the types block with the nodes at theta_best.
    manifest = json.loads((mpi_dir / "manifest.json").read_text())
    assert manifest["n_samples"] == 2 and manifest["n_elite"] == 2
    assert manifest["max_iter"] == 2
    _check_beta_types_block(manifest["beta_types"], 2, theta)
    manifest_ser = json.loads((ser_dir / "manifest.json").read_text())
    assert manifest_ser["n_samples"] == 2 and manifest_ser["max_iter"] == 2
    assert manifest_ser["beta_types"] == manifest["beta_types"]

    # The layering is visible in the log: the split line, one member-time
    # line per candidate evaluation on each group root, and every worker's
    # STOP receipt (two workers, two evaluations each).
    assert "[types] K=2" in proc_mpi.stdout
    assert len(_member_time_lines(proc_mpi.stdout)) >= 4
    assert sorted(_stop_lines(proc_mpi.stdout)) == ["2", "2"]
    assert "Trial failures: 0 evaluation" in proc_mpi.stdout
    # Serial: member times printed too, no worker lines.
    assert len(_member_time_lines(proc_ser.stdout)) >= 4
    assert _stop_lines(proc_ser.stdout) == []


# ---------------------------------------------------------------------------
# J2: forced failures are logged, scored at the penalty, and never deadlock
# ---------------------------------------------------------------------------

@mpi
@pytest.mark.parametrize("mode", ["rank1", "root", "estimate"])
def test_j2_forced_failure_is_logged_and_scored_at_the_penalty(work, mode):
    proc, results = _estimate(work["root"] / f"j2_{mode}", work["spec2"], ranks=4,
                              run_id=f"j2{mode}", env={"FUES_TYPES_FORCE_FAIL": mode})
    assert proc.returncode == 0, _report(proc)
    run_dir = _run_dir(results, work["spec2"], f"j2{mode}")

    # The root that received the failure logged it before kikku saw it.
    assert "[trial failure] rank=0" in proc.stdout
    assert "FUES_TYPES_FORCE_FAIL" in proc.stdout
    if mode in ("rank1", "root"):
        k = 1 if mode == "rank1" else 0
        assert f"[types failure] group root rank=0 member k={k}" in proc.stdout
    else:
        assert "[types failure]" not in proc.stdout
    assert "Trial failures: 1 evaluation" in proc.stdout

    # One of the two iteration-0 candidates was scored at the penalty.
    rows = _convergence(run_dir)
    assert [r[0] for r in rows] == [0, 1]
    _, best0, elite0 = rows[0]
    assert best0 < BIG_LOSS
    assert elite0 == (best0 + BIG_LOSS) / 2
    _, best1, elite1 = rows[1]
    assert best1 <= best0 and elite1 < BIG_LOSS

    # No deadlock: both workers received STOP after two evaluations.
    assert sorted(_stop_lines(proc.stdout)) == ["2", "2"]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["objective"] == best1


# ---------------------------------------------------------------------------
# J3: every evaluation failing exits with code 3
# ---------------------------------------------------------------------------

@mpi
def test_j3_every_evaluation_failing_exits_with_code_3(work):
    proc, results = _estimate(work["root"] / "j3_all", work["spec2"], ranks=4,
                              run_id="j3all", env={"FUES_TYPES_FORCE_FAIL": "all"})
    assert proc.returncode == 3, _report(proc)
    match = re.search(r"Trial failures: (\d+) evaluation", proc.stdout)
    assert match, proc.stdout[-3000:]
    n_failures = int(match.group(1))
    # Two candidates per iteration over two iterations, plus kikku's
    # re-evaluation at best_theta on its rank 0 when no evaluation succeeded.
    assert n_failures >= 4
    assert n_failures == proc.stdout.count("[trial failure]")
    assert "every evaluation failed" in proc.stdout
    assert len(_stop_lines(proc.stdout)) == 2


# ---------------------------------------------------------------------------
# J4: restart segments reproduce the single-segment run
# ---------------------------------------------------------------------------

@mpi
def test_j4_restart_segments_reproduce_the_single_segment_run(work, mpi_run):
    base = work["root"] / "j4_restart"
    first, results = _estimate(base, work["spec2"], ranks=4, run_id="fixedid",
                               extra=["--max-iter-this-run", "1"])
    assert first.returncode == 42, _report(first)
    run_dir = _run_dir(results, work["spec2"], "fixedid")
    assert not (run_dir / "convergence.csv").exists()
    assert sorted(_stop_lines(first.stdout)) == ["1", "1"]

    second, _ = _estimate(base, work["spec2"], ranks=4, run_id="fixedid",
                          extra=["--max-iter-this-run", "1", "--resume"])
    assert second.returncode == 0, _report(second)
    assert "Resuming from checkpoint" in second.stdout
    _, single_dir = mpi_run
    assert (run_dir / "convergence.csv").read_bytes() == (single_dir / "convergence.csv").read_bytes()
    assert (run_dir / "theta_best.json").read_bytes() == (single_dir / "theta_best.json").read_bytes()


# ---------------------------------------------------------------------------
# J5: a sweep with types on eight ranks
# ---------------------------------------------------------------------------

@mpi
def test_j5_sweep_with_types_writes_one_row_per_point(work):
    spec = _spec_copy(work["root"] / "types_k2_sweep_tau.yaml", TYPES_SPEC, n_types=2,
                      sweep={"tau": [0.10, 0.15]}, drop_free=["tau"])
    proc, results = _estimate(work["root"] / "j5_sweep", spec, ranks=8, run_id="j5sweep")
    assert proc.returncode == 0, _report(proc)
    summary_path = results / "separable" / spec.stem / "sweep_summary_j5sweep.csv"
    assert summary_path.is_file(), _report(proc)
    with summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert sorted(float(r["true_tau"]) for r in rows) == [0.10, 0.15]
    assert all(float(r["objective"]) < BIG_LOSS for r in rows)
    assert {"beta_bar", "sigma_beta"} <= set(rows[0])
    for label in ("tau=0.1", "tau=0.15"):
        run_dir = _run_dir(results, spec, "j5sweep", sub_label=label)
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["calib_overrides"]["tau"] in (0.1, 0.15)
        _check_beta_types_block(summary["beta_types"], 2, summary["theta_best"])
    # Two sub-communicators of two groups of two: four workers.
    assert sorted(_stop_lines(proc.stdout)) == ["2"] * 4


# ---------------------------------------------------------------------------
# J6: a rank count that is not a multiple of n_points x K is refused early
# ---------------------------------------------------------------------------

@mpi
def test_j6_indivisible_rank_count_is_refused_before_any_output(work):
    base = work["root"] / "j6_divisibility"
    proc, results = _estimate(base, TYPES_SPEC.name, ranks=6, run_id="j6")
    assert proc.returncode != 0
    assert "not a multiple of" in proc.stderr, _report(proc)
    assert "ValueError" in proc.stderr
    assert not results.exists() and not (base / "scratch").exists()
    assert "[types] K=" not in proc.stdout


# ---------------------------------------------------------------------------
# A spec without types never splits (spec 3.1)
# ---------------------------------------------------------------------------

NO_SPLIT_CODE = (
    "import sys\n"
    "import examples.durables.estimate as est\n"
    "import examples.durables.type_groups as tg\n"
    "sys.argv = ['estimate'] + sys.argv[1:]\n"
    "est.main()\n"
    "assert tg.SPLIT_CALLS == 0, tg.SPLIT_CALLS\n"
    "print('SPLIT_CALLS', tg.SPLIT_CALLS, flush=True)\n"
)


@mpi
def test_spec_without_types_never_splits_the_communicator(work):
    proc, results = _estimate(work["root"] / "no_types", work["no_types"], ranks=2,
                              run_id="notypes", code=NO_SPLIT_CODE)
    assert proc.returncode == 0, _report(proc)
    assert proc.stdout.count("SPLIT_CALLS 0") == 2
    assert "[types]" not in proc.stdout
    run_dir = _run_dir(results, work["no_types"], "notypes")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "beta_types" not in summary
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["beta_types"] is None and manifest["n_samples"] == 2
    assert manifest["max_iter"] == 2


# ---------------------------------------------------------------------------
# The saved outputs of the serial types run K
# ---------------------------------------------------------------------------

def test_saved_types_run_carries_four_nodes_and_shares():
    runs = sorted(p for p in FIXTURES_TYPES.glob("est_*") if p.is_dir())
    assert runs, f"no saved types estimation run under {FIXTURES_TYPES}"
    for run in runs:
        for fname in ("summary.json", "theta_best.json", "theta_mean.json",
                      "theta_se.json", "fit_table.csv", "convergence.csv",
                      "manifest.json"):
            assert (run / fname).is_file(), f"{run.name} lacks {fname}"
        summary = json.loads((run / "summary.json").read_text())
        theta = summary["theta_best"]
        assert {"beta_bar", "sigma_beta"} <= set(theta) and "beta" not in theta
        _check_beta_types_block(summary["beta_types"], 4, theta)
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["beta_types"] == summary["beta_types"]
        assert manifest["n_samples"] == 6 and manifest["n_elite"] == 2
        assert manifest["max_iter"] == 2
        assert manifest["grid"] == {"n_a": 30, "n_h": 30, "n_w": 30}
        assert manifest["N_sim"] == N_SIM
