"""Compare current estimation and fit reconstruction on synthetic moments.

This imports the supplied worktree normally. No solver is called. MPI's
public runtime configuration disables automatic initialization for this
serial arithmetic check; distributed execution is not tested.
"""

from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys

from mpi4py import rc

rc(initialize=False, finalize=False)
worktree = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(worktree))
estimate = import_module("kikku.run.estimate")
fit = import_module("examples.durables.postprocess.at_estimates")
moments = import_module("kikku.run.moments")

for module in (estimate, fit, moments):
    path = Path(module.__file__).resolve()
    print(json.dumps({"module": module.__name__, "path": str(path),
                      "sha256": sha256(path.read_bytes()).hexdigest()}))

data = {"present": 100.0, "missing": 50.0}
simulated = {"present": 110.0}
criterion = estimate.make_criterion(lambda theta: simulated,
                                   lambda panel: panel, data)
rows = fit.fit_at_estimates({}, simulated, data)
print(json.dumps({"criterion_loss": criterion({}),
                  "fit_contributions": sum(r["contribution"] for r in rows),
                  "fit_keys": [r["moment"] for r in rows]}))
print(json.dumps({"age40_44_rows":
                  moments._age_group_masks(70, {"5": [40, 44]}, 0)["5"]}))
