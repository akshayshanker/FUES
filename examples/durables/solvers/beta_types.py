"""Permanent heterogeneous types for the durables estimation.

Pure functions, no MPI (spec ``AI/devspecs/28092026/cd_estimation_and_postprocess.md``,
Section 5.6):

- :func:`discretise_types` — the K equiprobable nodes of one parameter on a
  logit, log or identity scale, and their equal shares;
- :func:`discretise_beta` — the logit case, kept as a thin wrapper;
- :func:`expand_types` — one calibration per type from a candidate parameter
  vector; member k takes the k-th quantile of every listed parameter at once;
- :func:`draw_types` — each agent's type, drawn once at birth from a random
  stream that no other draw uses;
- :func:`pool_by_type` — the K simulated sub-populations scattered back into
  full panels in the agents' original order, with ``type_idx`` and one
  ``(N,)`` array per listed parameter.

Composition in ``simulate_lifecycle``:
``draw_types -> K x simulate_type_subset -> pool_by_type``.
"""

import numpy as np
from scipy.stats import norm

_TRANSFORMS = ("logit", "log", "identity")


def discretise_types(location, spread, n, transform="logit"):
    """Nodes and equal shares of one heterogeneous parameter.

    Member k sits at the equiprobable normal quantile
    ``z_k = Phi^{-1}((k + 0.5) / n)``. The returned values are increasing.

    * ``logit`` — the parameter lies in (0, 1). On the scale
      ``x = ln(1/p - 1)`` the nodes are
      ``p / (p + (1 - p) exp(spread z))``. Larger ``x`` is a smaller ``p``,
      so the values are returned in reverse quantile order. This algebraic
      form returns ``location`` exactly at zero spread.
    * ``log`` — a positive parameter: ``location * exp(spread z)``.
    * ``identity`` — ``location + spread z``.

    Parameters
    ----------
    location : float
        Location of the distribution (the median type's value).
    spread : float
        Spread on the transform scale; ``0.0`` gives ``n`` copies of
        ``location``.
    n : int
        Number of types K.
    transform : {'logit', 'log', 'identity'}, optional
        Scale of the spread. Default ``'logit'``.

    Returns
    -------
    values : ndarray, float64, shape (n,)
        Parameter values, increasing.
    shares : ndarray, float64, shape (n,)
        All equal to ``1/n``.

    Raises
    ------
    ValueError
        If ``n < 1``, if ``transform`` is unknown, or if a logit location
        is not in (0, 1).
    """
    location = float(location)
    spread = float(spread)
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if transform not in _TRANSFORMS:
        raise ValueError(
            f"unknown transform {transform!r}; expected one of {_TRANSFORMS}")
    k = np.arange(n, dtype=np.float64)
    z = norm.ppf((k + 0.5) / n)
    if transform == "logit":
        if not (0.0 < location < 1.0):
            raise ValueError(
                f"logit location must be in (0, 1), got {location}")
        # z is increasing and the logit map is decreasing in x, so walk z
        # downwards to return the values increasing.
        scale = np.exp(spread * z[::-1])
        values = location / (location + (1.0 - location) * scale)
    elif transform == "log":
        values = location * np.exp(spread * z)
    else:
        values = location + spread * z
    shares = np.full(n, 1.0 / n, dtype=np.float64)
    return values.astype(np.float64), shares


def discretise_beta(beta_bar, sigma_beta, n):
    """Nodes and shares of the discount-factor distribution (logit wrapper).

    On the logit scale ``x = ln(1/beta - 1)`` the type distribution is normal
    with location ``x_bar = ln(1/beta_bar - 1)`` and standard deviation
    ``sigma_beta``; the K nodes sit at its equiprobable quantiles
    ``x_k = x_bar + sigma_beta * z_k`` with ``z_k = Phi^{-1}((k + 0.5) / K)``,
    mapped back by ``beta_k = 1 / (1 + exp(x_k))``. Each node carries the
    share ``1/K``.

    The map back is evaluated in the algebraically identical form
    ``beta_k = beta_bar / (beta_bar + (1 - beta_bar) exp(sigma_beta z_k))``
    because that form returns ``beta_bar`` exactly at ``sigma_beta = 0`` for
    every ``beta_bar`` in ``[0.5, 1)`` (``1 - beta_bar`` is exact there and
    ``beta_bar + (1 - beta_bar)`` rounds to one), whereas
    ``1 / (1 + exp(ln(1/beta_bar - 1)))`` differs from ``beta_bar`` by one
    unit in the last place for about forty percent of values.

    Parameters
    ----------
    beta_bar : float
        Location: the discount factor of the median type, in (0, 1).
    sigma_beta : float
        Spread on the logit scale, non-negative; ``0.0`` gives ``n`` copies
        of ``beta_bar``.
    n : int
        Number of types K.

    Returns
    -------
    nodes : ndarray, float64, shape (n,)
        Discount factors, increasing.
    shares : ndarray, float64, shape (n,)
        All equal to ``1/n``.
    """
    return discretise_types(beta_bar, sigma_beta, n, transform="logit")


def expand_types(theta, types_spec):
    """One calibration per type from a candidate parameter vector.

    Member k takes the k-th equiprobable quantile of every listed parameter
    at once (comonotonic types). Location and spread names are dropped from
    each calibration; each listed parameter is set to that member's value.

    Parameters
    ----------
    theta : dict
        Candidate parameters; must contain every location and spread name
        that ``types_spec`` maps and must not contain a listed parameter.
    types_spec : dict
        The ``types`` block of the estimation spec:
        ``{'n': K, 'parameters': {name: {'location', 'spread', 'transform'}}}``.
        ``transform`` defaults to ``'logit'`` when omitted.

    Returns
    -------
    list of (share_k, calibration_k)
        ``calibration_k`` is ``theta`` without the location and spread names,
        plus every listed parameter set to node k. Increasing in each
        listed parameter.

    Raises
    ------
    ValueError
        If ``n`` is missing or less than 1, if ``theta`` lacks a location
        or spread name, if a listed parameter is already in ``theta``, if
        a transform is unknown, or if a logit location is not in (0, 1).
    """
    if "n" not in types_spec:
        raise ValueError("types block is missing 'n'")
    n = int(types_spec["n"])
    if n < 1:
        raise ValueError(f"types n must be at least 1, got {n}")
    parameters = types_spec["parameters"]

    drop = set()
    nodes_by_name = {}
    for name, block in parameters.items():
        loc_name = block["location"]
        spr_name = block["spread"]
        transform = block.get("transform", "logit")
        missing = [key for key in (loc_name, spr_name) if key not in theta]
        if missing:
            raise ValueError(
                f"theta lacks the type parameters {missing} named by "
                f"types.parameters.{name}")
        if name in theta:
            raise ValueError(
                f"theta contains {name!r}, which the types block replaces by "
                f"{loc_name!r} and {spr_name!r}")
        values, _shares = discretise_types(
            theta[loc_name], theta[spr_name], n, transform)
        nodes_by_name[name] = values
        drop.add(loc_name)
        drop.add(spr_name)

    share = 1.0 / n
    base = {key: value for key, value in theta.items() if key not in drop}
    return [
        (share, {**base, **{name: float(nodes[k])
                            for name, nodes in nodes_by_name.items()}})
        for k in range(n)
    ]


def draw_types(N, shares, seed):
    """Each agent's type, drawn once at birth.

    The uniform draws come from ``SeedSequence(seed, spawn_key=(1,))``, a
    stream separate from the shock stream ``default_rng(seed)`` of
    ``draw_shocks`` and the initial-condition stream ``default_rng(seed + 1)``
    of ``make_initial_particles``. Agent i's type is
    ``min(searchsorted(cumsum(shares), u_i, side='right'), K - 1)``, which
    with equal shares is ``floor(K u_i)``. The shares do not depend on the
    candidate parameters, so the assignment is the same for every candidate.

    Returns
    -------
    ndarray, int64, shape (N,)
    """
    shares = np.asarray(shares, dtype=np.float64)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    u = rng.random(N)
    idx = np.searchsorted(np.cumsum(shares), u, side="right")
    return np.minimum(idx, len(shares) - 1).astype(np.int64)


def pool_by_type(parts, type_idx, N, member_values):
    """Scatter K simulated sub-populations back into full panels.

    Parameters
    ----------
    parts : list of (agent_idx, sim_data_k)
        One entry per type, in type order. ``agent_idx`` holds the original
        indices of the agents of type k; every array in ``sim_data_k`` has
        those agents on its last axis: ``(T, n_k)`` panels and ``(n_k,)``
        statistics.
    type_idx : ndarray, int64, shape (N,)
        Each agent's type.
    N : int
        Number of agents.
    member_values : dict
        ``{parameter_name: sequence of n values}``. Each name becomes an
        ``(N,)`` float64 array whose agent-i entry is that parameter's
        value for ``type_idx[i]``.

    Returns
    -------
    dict
        Every key of the parts as a ``(T, N)`` or ``(N,)`` array in the
        agents' original order, with the parts' dtypes, plus ``type_idx``
        (int64, the type index) and one float64 array per name in
        ``member_values``.

    Raises
    ------
    ValueError
        If the parts do not cover every agent exactly once, if a part holds
        agents of another type, or if an array has no agent axis of length
        ``n_k``.
    """
    type_idx = np.asarray(type_idx, dtype=np.int64)
    counts = np.zeros(N, dtype=np.int64)
    for k, (agent_idx, _) in enumerate(parts):
        agent_idx = np.asarray(agent_idx, dtype=np.int64)
        np.add.at(counts, agent_idx, 1)
        if not np.all(type_idx[agent_idx] == k):
            raise ValueError(f"part {k} holds agents whose type is not {k}")
    if not np.all(counts == 1):
        raise ValueError(
            f"parts cover {int(np.sum(counts > 0))} of {N} agents; "
            f"{int(np.sum(counts > 1))} agents appear more than once")

    keys = list(parts[0][1]) if parts else []
    pooled = {}
    for key in keys:
        full = None
        for agent_idx, sim_k in parts:
            arr = np.asarray(sim_k[key])
            n_k = len(agent_idx)
            if arr.ndim == 0 or arr.shape[-1] != n_k:
                raise ValueError(
                    f"{key!r} has shape {arr.shape}; expected the {n_k} agents "
                    f"of this type on the last axis")
            if full is None:
                fill = np.nan if np.issubdtype(arr.dtype, np.floating) else -1
                full = np.full(arr.shape[:-1] + (N,), fill, dtype=arr.dtype)
            full[..., np.asarray(agent_idx, dtype=np.int64)] = arr
        pooled[key] = full

    pooled["type_idx"] = type_idx.copy()
    for name, values in member_values.items():
        values = np.asarray(values, dtype=np.float64)
        pooled[name] = values[type_idx].astype(np.float64)
    return pooled
