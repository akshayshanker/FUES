"""Backward induction for the durables model (DDSL)."""

import time
import numpy as np
from pathlib import Path
from kikku.dynx import (
    load_syntax, instantiate_period,
    period_to_graph, backward_paths,
)
from kikku.dynx.methods import override_methods
from dolo.compiler.calibration import (
    calibrate as calibrate_stage,
)

from .model import make_grids
from .callables import make_callables
from .horses.keeper_egm import make_keeper_ops
from .horses.branching import make_tenure_ops
from .horses.adjuster_egm import make_adjuster_ops
from .horses.conditioning import make_conditioners

METHOD_SHORTCUT = [
    ('adjuster_cons', 'cntn_to_dcsn_mover', 'upper_envelope'),
]


def read_scheme_method(stage, scheme_name, mover='cntn_to_dcsn_mover',
                       default='FUES'):
    """Read method tag for a scheme from stage.methods."""
    if not hasattr(stage, 'methods'):
        return default
    mover_dict = stage.methods.get(mover, {})
    for scheme in mover_dict.get('schemes', []):
        if scheme.get('scheme') == scheme_name:
            tag = scheme.get('method', {})
            if isinstance(tag, dict):
                return tag.get('__yaml_tag__', default)
            return str(tag)
    return default


def _terminal_vlu_cntn(grids, tenure_callables):
    """Terminal condition: consume everything."""
    z_vals = grids["z"]
    a_grid = grids["a"]
    h_grid = grids["h"]
    x_all = grids["X_all"]
    term_u = tenure_callables["term_u"]
    d_a_term = tenure_callables["marginalBellman_d_a_terminal"]
    d_h_term = tenure_callables["marginalBellman_d_h_terminal"]
    g_w = tenure_callables["transitions"]["terminal_wealth"]

    shape = (len(z_vals), len(a_grid), len(h_grid))
    V = np.empty(shape)
    d_aV = np.empty(shape)
    d_hV = np.empty(shape)

    for state in range(len(x_all)):
        i_z, i_a, i_h = x_all[state]
        a = a_grid[i_a]
        h = h_grid[i_h]
        w = g_w(a, h)
        V[i_z, i_a, i_h] = term_u(w)
        d_aV[i_z, i_a, i_h] = d_a_term(w)
        d_hV[i_z, i_a, i_h] = d_h_term(w)

    return {"V": V, "d_aV": d_aV, "d_hV": d_hV}


def recalibrate_period(period, calib_h):
    """Pure fn: return new period with updated calibration."""
    new_stages = {}
    for name, stage in period["stages"].items():
        new_stages[name] = calibrate_stage(stage, calib_h)
    return {
        "stages": new_stages,
        "connectors": period.get("connectors", []),
    }


def solve_period(stage_ops, vlu_cntn, grids, age,
                 store_cntn=False, verbose=False):
    """Solve one period in wave order.

    Parameters
    ----------
    age : int
        Calendar age for this period (passed to tenure income transitions).
    """
    t0 = time.perf_counter()

    A_keep, C_keep, V_keep, dVw_keep, phi_keep, keeper_egm = (
        stage_ops["keeper_cons"]["dcsn_mover"](vlu_cntn, grids)
    )
    t_keeper = time.perf_counter() - t0

    # If NEGM, inject keeper output into adjuster before it runs
    if "inject_keeper" in stage_ops["adjuster_cons"]:
        stage_ops["adjuster_cons"]["inject_keeper"](C_keep, A_keep)

    t1 = time.perf_counter()
    A_adj, C_adj, H_adj, V_adj, dVw_adj, adj_egm = (
        stage_ops["adjuster_cons"]["dcsn_mover"](vlu_cntn, grids)
    )
    t_adj = time.perf_counter() - t1

    t2 = time.perf_counter()
    vlu_dcsn, pol_dcsn = stage_ops["tenure"]["dcsn_mover"](
        vlu_cntn,
        grids,
        age,
        A_keep,
        C_keep,
        V_keep,
        dVw_keep,
        phi_keep,
        A_adj,
        C_adj,
        H_adj,
        V_adj,
        dVw_adj,
    )
    t_discrete = time.perf_counter() - t2

    vlu_arvl = stage_ops["tenure"]["arvl_mover"](vlu_dcsn)

    solve_time = time.perf_counter() - t0
    if verbose:
        print(
            "  keeper: {:.1f}ms, adj: {:.1f}ms, tenure_choice: {:.1f}ms".format(
                t_keeper * 1000, t_adj * 1000, t_discrete * 1000
            )
        )

    # --- Assemble solution per solution_scheme.md ---

    keeper_sol = {
        "dcsn": {
            "V": V_keep,
            "d_wV": dVw_keep,
            "d_hV": phi_keep,
            "c": C_keep,
        },
    }

    adjuster_sol = {
        "dcsn": {
            "V": V_adj,
            "d_wV": dVw_adj,
            "c": C_adj,
            "h_choice": H_adj,
        },
    }

    tenure_sol = {
        "dcsn": {
            "V": vlu_dcsn["V"],
            "d_aV": vlu_dcsn["d_aV"],
            "d_hV": vlu_dcsn["d_hV"],
            "adj": pol_dcsn["adj"],
        },
        "arvl": vlu_arvl,
    }

    if store_cntn:
        keeper_cntn = {
            "V": vlu_cntn["V"],
            "d_a_nxtV": vlu_cntn["d_aV"],
            "d_h_nxtV": vlu_cntn["d_hV"],
        }
        if keeper_egm is not None:
            keeper_cntn["c"] = keeper_egm["c"]
            keeper_cntn["m_endog"] = keeper_egm["m_endog"]
        keeper_sol["cntn"] = keeper_cntn

        if adj_egm is not None:
            adjuster_sol["cntn"] = {
                "c": adj_egm["c"],
                "m_endog": adj_egm["m_endog"],
                "a_nxt_eval": adj_egm["a_nxt_eval"],
                "h_nxt_eval": adj_egm["h_nxt_eval"],
                "_refined": adj_egm["_refined"],
            }

    return {
        "keeper_cons": keeper_sol,
        "adjuster_cons": adjuster_sol,
        "tenure": tenure_sol,
        "solve_time": solve_time,
        "keeper_ms": t_keeper * 1000,
        "adj_ms": t_adj * 1000,
        "discrete_ms": t_discrete * 1000,
    }


def accrete_and_solve(
    H, period_inst, calibration, grids,
    store_cntn=False, verbose=False, progress=None,
):
    """Accrete backward, solving each period.

    Parameters
    ----------
    verbose : bool or str
        ``True``: per-period header and sub-stage timings (default, unchanged).
        ``False``: quiet.
        ``'summary'``: no per-period output; one summary line at the end.
    progress : str, optional
        ``'bar'``: tqdm progress bar (requires ``tqdm``); suppresses per-period
        text unless ``verbose`` is ``True`` (sub-stage timings still off during bar).
    """
    nest = {"periods": [], "solutions": []}

    stage_ref = period_inst["stages"]["keeper_cons"]
    T = int(stage_ref.settings["T"])
    warmup = int(stage_ref.settings.get("warmup_periods", 3))

    condition_V, condition_V_HD = make_conditioners(grids["Pi"])
    vlu_cntn = None

    # Callables are age-invariant (income transitions accept age as arg).
    # Build once; reuse across all periods.
    callables = make_callables(period_inst)

    # Operators are also age-invariant — build once.
    # Tenure dcsn_mover receives age as a runtime argument.
    stages = period_inst["stages"]
    for _sk in ("keeper_cons", "adjuster_cons", "tenure"):
        stages[_sk].calibration["return_grids"] = store_cntn

    keeper_ops = {"dcsn_mover": make_keeper_ops(
        callables, grids, stages["keeper_cons"])}
    adjuster_ops = make_adjuster_ops(
        callables, grids, stages["adjuster_cons"])
    tenure_dcsn, tenure_arvl, tenure_arvl_hd = make_tenure_ops(
        callables, grids, stages["tenure"],
        condition_V, condition_V_HD,
    )
    stage_ops = {
        "keeper_cons": keeper_ops,
        "adjuster_cons": adjuster_ops,
        "tenure": {
            "dcsn_mover": tenure_dcsn,
            "arvl_mover": tenure_arvl,
            "arvl_mover_hd": tenure_arvl_hd,
        },
    }

    if verbose:
        cal = stage_ref.calibration
        sett = stage_ref.settings
        ue_method = read_scheme_method(stages["adjuster_cons"], 'upper_envelope')
        print(f"  Method: {ue_method}")
        print(f"  Horizon: ages {int(cal['t0'])}–{T} ({H+1} periods)")
        print(f"  Grids: n_a={len(grids['a'])}, n_h={len(grids['h'])}, "
              f"n_w={len(grids['we'])}, N_wage={len(grids['z'])}")
        print(f"  Params: beta={cal['beta']}, R={cal['R']}, "
              f"tau={cal['tau']}, delta={cal['delta']}")
        print(f"  Shocks: phi_w={cal['phi_w']}, sigma_w={cal['sigma_w']}, "
              f"z_grid={grids['z']}")
        print(f"  Solver: m_bar={sett['m_bar']}, root_eps={sett['root_eps']}, "
              f"egm_n={sett['egm_n']}")

    use_bar = progress == "bar"
    if use_bar:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise ImportError(
                "progress='bar' requires tqdm; install with pip install tqdm"
            ) from exc
        pbar = tqdm(range(H + 1), desc="Solving")
        loop_iter = pbar
    else:
        pbar = None
        loop_iter = range(H + 1)

    show_period_header = verbose is True and not use_bar
    period_timing_verbose = verbose is True and not use_bar

    for h in loop_iter:
        age = T - h

        # Recalibrate period with age — this is the DDSL contract:
        # each period carries its own calibration as source of truth.
        calib_h = {**calibration, "age": age}
        period_h = recalibrate_period(period_inst, calib_h)
        nest["periods"].append(period_h)

        if h == 0:
            vlu_cntn = _terminal_vlu_cntn(grids, callables["tenure"])
        else:
            vlu_cntn = nest["solutions"][h - 1]["tenure"]["arvl"]

        if show_period_header:
            print(f"  Solving age {age} (h={h})...")
        sol = solve_period(stage_ops, vlu_cntn, grids, age,
                           store_cntn=store_cntn, verbose=period_timing_verbose)
        sol["h"] = h
        sol["t"] = age
        sol["callables"] = callables
        nest["solutions"].append(sol)

        if pbar is not None:
            steady = [s for s in nest["solutions"] if s["h"] > warmup]
            if steady:
                n = len(steady)
                avg_k = sum(s["keeper_ms"] for s in steady) / n
                avg_a = sum(s["adj_ms"] for s in steady) / n
                pbar.set_postfix(
                    age=age,
                    avg_keeper_ms=f"{avg_k:.0f}",
                    avg_adj_ms=f"{avg_a:.0f}",
                )
            else:
                pbar.set_postfix(age=age, warmup=True)

    if verbose == "summary":
        sols = nest["solutions"]
        nper = len(sols)
        steady = [s for s in sols if s["h"] > warmup]
        if not steady:
            steady = sols
        n_steady = len(steady)
        if nper > 0:
            total_s = sum(s["solve_time"] for s in sols)
            mk = sum(s["keeper_ms"] for s in steady) / n_steady
            maj = sum(s["adj_ms"] for s in steady) / n_steady
            md = sum(s["discrete_ms"] for s in steady) / n_steady
            print(
                f"Backward induction: {nper} periods in {total_s:.2f}s "
                f"(mean excl. {warmup} warmup: "
                f"keeper {mk:.0f} ms, adj {maj:.0f} ms, "
                f"tenure {md:.0f} ms)"
            )

    return nest


def precompile(syntax_dir='examples/durables2_0/syntax', method=None):
    """Warm Numba JIT caches with a tiny solve (small grids, 2 periods).

    Call once before timed solves to avoid JIT compilation overhead
    in the measurement. All @njit dispatchers are compiled on the first
    call; subsequent calls with any grid size reuse the cached machine code.
    """
    tiny_config = {'n_a': 10, 'n_h': 10, 'n_w': 10, 't0': 40, 'N_wage': 2}
    solve(syntax_dir, method=method,
          setting_overrides=tiny_config, verbose=False)


def solve(
    syntax_dir,
    method=None,
    method_overrides=None,
    calib_overrides=None,
    setting_overrides=None,
    grids=None,
    verbose=False,
    progress=None,
):
    """Full DDSL pipeline: load -> accrete+solve.

    Parameters
    ----------
    method : str, optional
        If given (e.g. ``\"NEGM\"``), applies to each target in
        :data:`METHOD_SHORTCUT` before instantiation. If ``None``, YAML
        defaults apply unless ``method_overrides`` is set.
    method_overrides : dict, optional
        ``{(stage, target, scheme): tag, ...}`` merged after the ``method``
        shortcut expansion; passed to :func:`kikku.dynx.methods.override_methods`.
    verbose : bool or str
        ``True``: print waves (unless ``progress='bar'``) and per-period solve output.
        ``False``: minimal.
        ``'summary'``: final timing summary only from backward induction.
    progress : str, optional
        ``'bar'``: tqdm progress bar during accretion (see :func:`accrete_and_solve`).

    Returns
    -------
    nest, grids
    """
    syntax_dir = Path(syntax_dir)

    # All stages share the same calibration and settings in this model.
    # CLI overrides (--calib-override, --setting-override) apply to ALL stages
    # because load_syntax merges calibration into every stage source.
    calibration, yaml_settings, stage_sources, period_template, inter_conn = load_syntax(
        syntax_dir, calib_overrides, setting_overrides
    )
    all_method_overrides = {}
    if method is not None:
        for target in METHOD_SHORTCUT:
            all_method_overrides[target] = method
    if method_overrides:
        all_method_overrides.update(method_overrides)
    if all_method_overrides:
        stage_sources = override_methods(stage_sources, all_method_overrides)
    period_inst = instantiate_period(
        calibration, yaml_settings, stage_sources, period_template
    )

    if grids is None:
        grids = make_grids(calibration, yaml_settings)

    # T, t0 are period-level calibration; any stage carries them after merge.
    stage_ref = period_inst["stages"]["keeper_cons"]
    T = int(stage_ref.settings["T"])
    t0 = int(stage_ref.calibration["t0"])

    # Keep wave derivation for diagnostics.
    graph = period_to_graph(period_inst)
    waves = backward_paths(graph, inter_conn)
    if verbose is True and progress != "bar":
        print(f"Waves: {waves}")

    H = T - t0
    store_cntn = bool(int(yaml_settings.get("store_cntn", 0)))
    nest = accrete_and_solve(
        H, period_inst, calibration, grids,
        store_cntn=store_cntn, verbose=verbose, progress=progress,
    )

    # Expose topology so the forward simulator uses the same graph
    # that was solved, not a fresh parse of the syntax directory.
    nest["graph"] = graph
    nest["inter_conn"] = inter_conn

    return nest, grids
