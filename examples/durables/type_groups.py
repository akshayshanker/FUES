"""Layered MPI communicator for the permanent types of the estimation.

Within the communicator that reaches ``_run_single_estimation`` in
``estimate.py`` (the world, or a sweep point's sub-communicator), of size W
with W a multiple of K, the ranks are split twice (the two layers of
Eggsandbaskets ``gen_communicators``):

* ``group_comm``: W/K groups of K ranks, ``comm.Split(rank // K, rank)``;
  group rank k solves type member k of one candidate parameter vector;
* ``roots_comm``: the group roots, one rank per candidate,
  ``comm.Split(0 if group_rank == 0 else MPI.UNDEFINED, rank)``; kikku's
  ``estimate()`` runs on it and sees W/K candidates per iteration. It is
  ``None`` on the other ranks (the workers).

Message protocol on ``group_comm``; every message is a tuple whose first
element is the tag:

    (EVAL, theta)             root -> members: solve your member of theta
    (STOP,)                   root -> members: leave the worker loop
    (PART, payload, seconds)  member -> root (gathered): the member's part
    (FAIL, message, seconds)  member -> root (gathered): the member raised

Every rank, the root included, computes its member inside ``try/except`` and
always reaches the gather, so a failure on any rank cannot leave the others
blocked in a collective (an MPI call every member of the communicator must
make). The root raises only after the gather, having logged every failed
member with the candidate; kikku's criterion closure then scores the
candidate at the penalty loss. The roots broadcast STOP from a ``finally``
around ``estimate()``, so an exception inside ``estimate()`` itself still
releases the workers.

Composition: ``roots -> estimate(criterion)`` with
``criterion.trial = make_group_trial(...)``; ``workers -> worker_loop(...)``.
The member solve and the pooling step are supplied by ``estimate.py``.

Test hook, read nowhere else. The environment variable
``FUES_TYPES_FORCE_FAIL`` forces a failure at a named place, in the first
group only (the group that holds world rank 0) and on the first evaluation of
that place in the process, except for ``all``:

    rank<r>    the member with group rank r (a worker) raises inside its
               member solve;
    root       the group root raises inside its own member solve;
    estimate   the group root raises inside its trial after the gather and
               before pooling: outside any solve, inside kikku's
               ``estimate()`` call path;
    all        every rank raises inside every member solve.

When the variable is set the module prints a warning on import, so a
production run cannot carry it unnoticed.

Termination after an exception that escapes ``estimate()`` itself (a
checkpoint that cannot be written, for instance): the ``finally`` releases
the workers from the worker loop, but they then wait in the collectives
that follow ``estimate()`` while the root propagates the exception. The
process group ends because the driver is launched with ``python -m mpi4py``,
which aborts every rank when one rank dies with an uncaught exception; the
PBS scripts all launch that way, and a launch without ``-m mpi4py`` would
leave the workers waiting until the walltime.
"""

import os
import sys
import time
import traceback

EVAL = "EVAL"
STOP = "STOP"
FAIL = "FAIL"
PART = "PART"

# Number of calls of split_type_groups in this process. A spec without a
# types block must leave it at zero (spec 3.1); tests assert it.
SPLIT_CALLS = 0

_FORCE_FAIL_ENV = "FUES_TYPES_FORCE_FAIL"
_hook_calls = {"member": 0, "root_trial": 0}

if os.environ.get(_FORCE_FAIL_ENV):
    print(
        f"WARNING: {_FORCE_FAIL_ENV}={os.environ[_FORCE_FAIL_ENV]!r} is set: "
        "a type member will be made to fail on purpose. This is a test hook; "
        "unset it for any real estimation.",
        file=sys.stderr, flush=True)


def check_divisibility(comm, K, n_points=1):
    """Raise the same ``ValueError`` on every rank unless the rank count fits.

    ``comm.Get_size()`` must be a multiple of ``n_points * K``: K ranks per
    candidate (one per type member) and the same number of ranks for every
    sweep point, because the sweep colouring ``rank * n_points // size``
    gives sub-communicators of unequal size otherwise. Call it on every rank
    before any ``Split``: the check is deterministic in its arguments, so all
    ranks raise together and none is left waiting in a collective.
    """
    size = int(comm.Get_size())
    K = int(K)
    n_points = int(n_points)
    if K < 1 or n_points < 1:
        raise ValueError(
            f"K={K} type members and n_points={n_points} sweep points must both be positive")
    block = n_points * K
    if size % block != 0:
        raise ValueError(
            f"communicator size {size} is not a multiple of n_points x K = "
            f"{n_points} x {K} = {block}: a types estimation needs K ranks per "
            f"candidate (one per type member) and the same number of ranks for "
            f"every sweep point. Launch with a multiple of {block} ranks.")


def split_type_groups(comm, K):
    """Split ``comm`` into groups of K ranks; return ``(group_comm, roots_comm)``.

    ``roots_comm`` is the communicator of the group roots (group rank 0) and
    is ``None`` on every other rank, where MPI returns ``COMM_NULL``.
    """
    global SPLIT_CALLS
    from mpi4py import MPI

    SPLIT_CALLS += 1
    K = int(K)
    rank = comm.Get_rank()
    group_comm = comm.Split(rank // K, rank)
    group_rank = group_comm.Get_rank()
    roots_comm = comm.Split(0 if group_rank == 0 else MPI.UNDEFINED, rank)
    if group_rank != 0:
        return group_comm, None
    return group_comm, roots_comm


def _in_first_group(group_comm):
    """True in the group that holds world rank 0 (comm ranks 0 .. K-1)."""
    from mpi4py import MPI

    return MPI.COMM_WORLD.Get_rank() < group_comm.Get_size()


def _forced_failure(group_comm, k, place):
    """Raise when the ``FUES_TYPES_FORCE_FAIL`` test hook names this rank and place."""
    mode = os.environ.get(_FORCE_FAIL_ENV)
    if not mode:
        return
    _hook_calls[place] += 1
    if mode == "all":
        if place == "member":
            raise RuntimeError(
                f"forced failure ({_FORCE_FAIL_ENV}={mode}) in member solve k={k}")
        return
    if _hook_calls[place] != 1 or not _in_first_group(group_comm):
        return
    if place == "member" and (
            (mode == f"rank{k}" and k != 0) or (mode == "root" and k == 0)):
        raise RuntimeError(
            f"forced failure ({_FORCE_FAIL_ENV}={mode}) in member solve k={k}")
    if place == "root_trial" and mode == "estimate":
        raise RuntimeError(
            f"forced failure ({_FORCE_FAIL_ENV}={mode}) on the group root after "
            f"the gather, outside any solve")


def _run_member(group_comm, solve_member, theta, k):
    """This rank's part for member ``k``: ``(PART, payload, seconds)`` or ``(FAIL, message, seconds)``.

    An exception is not hidden: its traceback is printed on this rank (flushed,
    to stderr) and its type and message travel to the root in the FAIL part,
    where they are logged again with the candidate before the root raises.
    """
    from mpi4py import MPI

    start = time.perf_counter()
    try:
        _forced_failure(group_comm, k, "member")
        payload = solve_member(theta, k)
    except Exception as exc:
        seconds = time.perf_counter() - start
        print(f"[types member failure] rank={MPI.COMM_WORLD.Get_rank()} k={k} "
              f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return (FAIL, f"{type(exc).__name__}: {exc}", seconds)
    return (PART, payload, time.perf_counter() - start)


def make_group_trial(group_comm, K, solve_member, pool, log):
    """The root side of one candidate evaluation: ``trial(theta) -> panels``.

    Parameters
    ----------
    group_comm :
        This group's communicator; the caller is its rank 0.
    K : int
        Members per candidate; equals ``group_comm.Get_size()``.
    solve_member : callable
        ``solve_member(theta, k) -> payload``; solves member k of the candidate
        and returns the part the pool consumes.
    pool : callable
        ``pool(theta, payloads) -> panels`` with the K payloads in member order.
    log : callable
        ``log(message)``; prints with ``flush=True``.

    The trial broadcasts ``(EVAL, theta)``, computes member 0 itself, gathers
    the K parts, logs every member's solve time, and raises ``RuntimeError``
    listing every failed member only after the gather; otherwise it pools.
    """
    from mpi4py import MPI

    K = int(K)
    if group_comm.Get_rank() != 0:
        raise ValueError("make_group_trial is for the group root (group rank 0)")
    if group_comm.Get_size() != K:
        raise ValueError(
            f"group communicator has {group_comm.Get_size()} ranks, expected K={K}")
    world_rank = MPI.COMM_WORLD.Get_rank()

    def trial(theta):
        group_comm.bcast((EVAL, theta), root=0)
        part = _run_member(group_comm, solve_member, theta, 0)
        parts = group_comm.gather(part, root=0)
        # Every member has reported; only now may the root raise.
        times = " ".join(f"k={k} {p[2]:.2f}" for k, p in enumerate(parts))
        log(f"[types] group root rank={world_rank} member solve times (s): {times}")
        failures = [(k, p[1]) for k, p in enumerate(parts) if p[0] == FAIL]
        if failures:
            for k, message in failures:
                log(f"[types failure] group root rank={world_rank} member k={k}: "
                    f"{message}  theta={theta}")
            raise RuntimeError(
                f"{len(failures)} of {K} type members failed: "
                + "; ".join(f"k={k}: {message}" for k, message in failures))
        _forced_failure(group_comm, 0, "root_trial")
        return pool(theta, [p[1] for p in parts])

    return trial


def worker_loop(group_comm, solve_member):
    """Serve the group root until STOP: for every EVAL, compute this rank's member and gather it."""
    from mpi4py import MPI

    k = group_comm.Get_rank()
    if k == 0:
        raise ValueError("worker_loop is for the group members other than the root")
    n_eval = 0
    while True:
        message = group_comm.bcast(None, root=0)
        tag = message[0]
        if tag == STOP:
            print(f"[types] worker rank={MPI.COMM_WORLD.Get_rank()} received STOP "
                  f"after {n_eval} evaluations", flush=True)
            return
        if tag != EVAL:
            raise RuntimeError(
                f"unexpected message {tag!r} on the type group communicator")
        part = _run_member(group_comm, solve_member, message[1], k)
        n_eval += 1
        group_comm.gather(part, root=0)
