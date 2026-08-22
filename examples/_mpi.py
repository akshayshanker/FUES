"""Shared MPI helpers for example runners."""


def get_mpi_comm():
    """Return ``MPI.COMM_WORLD`` when running under ``mpiexec -n >1``; else None.

    Catches ``RuntimeError`` too — Macs with ``mpi4py`` installed but no MPI
    runtime library raise ``RuntimeError: cannot load MPI library`` on
    ``MPI.COMM_WORLD`` access. Serial fallback instead of crashing.
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            return comm
    except (ImportError, RuntimeError):
        pass
    return None
