# example_runner_usage.py
import numpy as np
import yaml
from dynx_runner import CircuitRunner
from my_stagecraft_api import (
    initialize_model_Circuit,   # ⇦ your existing builder
    compile_all_stages          # ⇦ grid/envelope enumeration
)
from my_solvers import backward_solver, forward_simulator   # whisperer equivalents

# ------------------------------------------------------------------
# 1.  model_factory that the runner will call for EVERY draw
# ------------------------------------------------------------------
def model_factory(epochs_cfgs, stage_cfgs, conn_cfg, *, n_periods=2):
    """
    epochs_cfgs: dict[str, dict]   (e.g. {"main": master_cfg})
    stage_cfgs : dict[str, dict]
    conn_cfg   : dict
    n_periods  : forwarded kwarg from CircuitRunner.factory_kwargs
    """
    # ---- create Circuit ------------------------------------------------
    model_circuit = initialize_model_Circuit(
        master_config      = epochs_cfgs["main"],      # simple use-case, one epoch
        stage_configs      = stage_cfgs,
        connections_config = conn_cfg,
        n_periods          = n_periods,
    )

    # ---- compile (grids, operators, ...) -------------------------------
    compile_all_stages(model_circuit)
    return model_circuit

# ------------------------------------------------------------------
# 2.  Load YAML configs (epoch, stages, connections)
# ------------------------------------------------------------------
epochs_cfgs = {"main": yaml.safe_load(open("cfgs/main_master.yml"))}
stage_cfgs  = {
    "OWNH": yaml.safe_load(open("cfgs/OWNH_stage.yml")),
    "RNTH": yaml.safe_load(open("cfgs/RNTH_stage.yml")),
}
conn_cfg    = yaml.safe_load(open("cfgs/connections.yml"))

# ------------------------------------------------------------------
# 3.  Declare which knobs we want the runner to vary
# ------------------------------------------------------------------
param_specs = {
    "main.parameters.beta": (0.8, 0.99, lambda n: np.random.uniform(0.8, 0.99, n))
}
enum_specs  = {
    "OWNH.methods.solver": ("FUES", "DCEGM")
}

# ------------------------------------------------------------------
# 4.  Build CircuitRunner
# ------------------------------------------------------------------
runner = CircuitRunner(
    epochs_cfgs   = epochs_cfgs,
    stage_cfgs    = stage_cfgs,
    conn_cfg      = conn_cfg,
    param_specs   = param_specs,
    enum_specs    = enum_specs,
    model_factory = model_factory,
    factory_kwargs= {"n_periods": 2},
    solver        = backward_solver,
    simulator     = forward_simulator,
    forward_T     = 400,              # run 400-period simulation
    metric_fns    = {
        "LL"        : lambda m: -m.log_likelihood(),      # scalar objective
        "EulerMax"  : lambda m: m.euler_error_max(),       # diagnostic only
    },
    cache         = True,
)

# ------------------------------------------------------------------
# 5.  Evaluate one point
# ------------------------------------------------------------------
x0      = runner.pack({"main.parameters.beta": 0.95})
metrics, model = runner.run(x0, enums=[0], keep=True)  # enums[0] = "FUES"
print(metrics)     # -> {'LL': -1234.5, 'EulerMax': 8.2e-4, 'time_seconds': 3.81}

# ------------------------------------------------------------------
# 6.  Monte-Carlo sweep (serial)
# ------------------------------------------------------------------
draws   = runner.sample_prior(100)
results = [runner.run(d, enums=[1])[0]["LL"]   # enums[1] = "DCEGM"
           for d in draws]
print("LL mean:", np.mean(results))
