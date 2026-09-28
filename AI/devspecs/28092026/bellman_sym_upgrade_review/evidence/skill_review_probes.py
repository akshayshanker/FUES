"""Small observations for the pre-upgrade review; no model is solved or changed.

Run with FUES/.venv/bin/python. All trial functions below are synthetic.
The production criterion is imported normally and is never patched.
Use --no-mpi-init for this serial arithmetic probe when MPI cannot initialize;
this uses mpi4py's public runtime configuration, and no MPI calls are made.
"""

from contextlib import redirect_stderr
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import sys

import numpy as np


if "--no-mpi-init" in sys.argv:
    from mpi4py import rc
    rc(initialize=False, finalize=False)

estimation = import_module("kikku.run.estimate")
source = Path(estimation.__file__).resolve()
print(json.dumps({
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "scipy": version("scipy"),
    "numba": version("numba"),
    "kikku": version("kikku"),
    "criterion_source": str(source),
    "criterion_sha256": sha256(source.read_bytes()).hexdigest(),
    "precision": str(np.dtype(float)),
    "mpi_auto_initialize": "--no-mpi-init" not in sys.argv,
}, sort_keys=True))


def fail_trial(theta):
    raise KeyError("synthetic missing calibration parameter")


errors = StringIO()
failed = estimation.make_criterion(fail_trial, lambda x: x, {"level": 10.0})
with redirect_stderr(errors):
    failed_loss = failed({})
print(json.dumps({
    "case": "trial exception",
    "loss": failed_loss,
    "finite": bool(np.isfinite(failed_loss)),
    "equals_BIG_LOSS": failed_loss == estimation.BIG_LOSS,
    "last_sim_moments": failed.last_sim_moments,
    "recorded_error": errors.getvalue().splitlines()[0],
}, sort_keys=True))

missing = estimation.make_criterion(lambda theta: {}, lambda x: x, {"level": 10.0})
missing_loss = missing({})
print(json.dumps({
    "case": "missing simulated target",
    "loss": missing_loss,
    "finite": bool(np.isfinite(missing_loss)),
    "equals_NAN_PENALTY": missing_loss == estimation.NAN_PENALTY,
    "last_sim_moments": missing.last_sim_moments,
}, sort_keys=True))

for label, target, simulated in (("level in original units", 10.0, 11.0),
                                 ("same proportional error after scaling", 0.1, 0.11)):
    criterion = estimation.make_criterion(
        lambda theta: {"level": theta["level"]}, lambda x: x, {"level": target}
    )
    print(json.dumps({
        "case": label,
        "target": target,
        "simulated": simulated,
        "relative_error": (simulated - target) / target,
        "loss": criterion({"level": simulated}),
    }, sort_keys=True))

# Retirement's constrained-policy/value formulas in operators.py:149-154.
# Common continuation and work-cost terms cancel in this comparison.
resources, saving_floor = 1.0, 0.2
consumption = resources - saving_floor
print(json.dumps({
    "case": "retirement constrained current utility",
    "resources": resources,
    "saving_floor": saving_floor,
    "reported_consumption": consumption,
    "utility_used_by_existing_value_line": float(np.log(resources)),
    "utility_at_reported_consumption": float(np.log(consumption)),
    "difference": float(np.log(resources) - np.log(consumption)),
}, sort_keys=True))
