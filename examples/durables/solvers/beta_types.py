"""Permanent discount-factor types for the durables estimation.

Pure functions, no MPI (spec ``AI/devspecs/28092026/cd_estimation_and_postprocess.md``,
Section 5.6):

- :func:`discretise_beta` — the K equiprobable nodes of the type distribution
  on the logit scale ``x = ln(1/beta - 1)`` and their equal shares;
- :func:`expand_types` — one calibration per type from a candidate parameter
  vector that carries ``beta_bar`` and ``sigma_beta`` in place of ``beta``;
- :func:`draw_types` — each agent's type, drawn once at birth from a random
  stream that no other draw uses;
- :func:`pool_by_type` — the K simulated sub-populations scattered back into
  full panels in the agents' original order.

Composition in ``simulate_lifecycle``:
``draw_types -> K x simulate_type_subset -> pool_by_type``.
"""

import numpy as np
from scipy.stats import norm


def discretise_beta(beta_bar, sigma_beta, n):
    """Nodes and shares of the discount-factor distribution.

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
    beta_bar = float(beta_bar)
    sigma_beta = float(sigma_beta)
    k = np.arange(n, dtype=np.float64)
    z = norm.ppf((k + 0.5) / n)
    # z is increasing and beta is decreasing in x, so walk z downwards to
    # return the nodes increasing in beta.
    scale = np.exp(sigma_beta * z[::-1])
    nodes = beta_bar / (beta_bar + (1.0 - beta_bar) * scale)
    shares = np.full(n, 1.0 / n, dtype=np.float64)
    return nodes.astype(np.float64), shares


def expand_types(theta, types_spec):
    """One calibration per type from a candidate parameter vector.

    Parameters
    ----------
    theta : dict
        Candidate parameters; must contain the location and spread names
        that ``types_spec`` maps and must not contain the target parameter.
    types_spec : dict
        The ``types`` block of the estimation spec, one entry:
        ``{'beta': {'n': K, 'location': 'beta_bar', 'spread': 'sigma_beta'}}``.

    Returns
    -------
    list of (share_k, calibration_k)
        ``calibration_k`` is ``theta`` without the location and spread names,
        plus the target parameter set to node k. Increasing in the target.
    """
    if len(types_spec) != 1:
        raise ValueError(
            f"types block must map exactly one parameter, got {sorted(types_spec)}")
    (target, block), = types_spec.items()
    location = block["location"]
    spread = block["spread"]
    n = int(block["n"])
    missing = [name for name in (location, spread) if name not in theta]
    if missing:
        raise ValueError(
            f"theta lacks the type parameters {missing} named by types.{target}")
    if target in theta:
        raise ValueError(
            f"theta contains {target!r}, which the types block replaces by "
            f"{location!r} and {spread!r}")
    nodes, shares = discretise_beta(theta[location], theta[spread], n)
    base = {name: value for name, value in theta.items()
            if name not in (location, spread)}
    return [(float(share), {**base, target: float(node)})
            for share, node in zip(shares, nodes)]


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


def pool_by_type(parts, type_idx, betas, N):
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
    betas : sequence of float, length K
        Each type's discount factor, read from its solved model.
    N : int
        Number of agents.

    Returns
    -------
    dict
        Every key of the parts as a ``(T, N)`` or ``(N,)`` array in the
        agents' original order, with the parts' dtypes, plus ``beta_type``
        (int64, the type index) and ``beta`` (float64, the type's value).

    Raises
    ------
    ValueError
        If the parts do not cover every agent exactly once, if a part holds
        agents of another type, or if an array has no agent axis of length
        ``n_k``.
    """
    type_idx = np.asarray(type_idx, dtype=np.int64)
    betas = np.asarray(betas, dtype=np.float64)
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

    pooled["beta_type"] = type_idx.copy()
    pooled["beta"] = betas[type_idx].astype(np.float64)
    return pooled
