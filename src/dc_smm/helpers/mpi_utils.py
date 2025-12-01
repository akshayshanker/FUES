"""
MPI utilities for DC-SMM housing model parallelization.

Provides a clean interface over mpi4py with automatic fallback to serial execution
when MPI is not available or not requested.
"""
import sys
from typing import Any, Optional, Union, List, Dict
import numpy as np


class DummyComm:
    """Mock MPI communicator for serial execution."""
    
    def __init__(self):
        self.rank = 0
        self.size = 1
    
    def Get_rank(self):
        """Return rank (0 for serial execution)."""
        return self.rank
    
    def Get_size(self):
        """Return size (1 for serial execution)."""
        return self.size
    
    def Barrier(self):
        """No-op barrier for serial execution."""
        pass
    
    def scatter(self, data, root=0):
        """Return first element for serial execution."""
        if self.rank == root and data:
            return data[0] if isinstance(data, list) else data
        return None
    
    def gather(self, data, root=0):
        """Return list with single element for serial execution."""
        if self.rank == root:
            return [data] if data is not None else []
        return None
    
    def bcast(self, data, root=0):
        """Return data unchanged for serial execution."""
        return data
    
    def Finalize(self):
        """No-op finalize for serial execution."""
        pass


def get_comm(enabled: bool = True):
    """
    Get MPI communicator or dummy for serial execution.
    
    Parameters
    ----------
    enabled : bool
        Whether to use MPI if available
        
    Returns
    -------
    comm : MPI communicator or DummyComm
        Active communicator object
    """
    if not enabled:
        return DummyComm()
    
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:
        return DummyComm()


def chunk_indices(n_items: int, n_chunks: int, k: int) -> range:
    """
    Get index range for chunk k of n_chunks from n_items total.
    
    Parameters
    ----------
    n_items : int
        Total number of items to distribute
    n_chunks : int  
        Number of chunks to create
    k : int
        Chunk index (0-based)
        
    Returns
    -------
    range
        Index range for chunk k
    """
    if n_chunks <= 0 or k < 0 or k >= n_chunks:
        return range(0, 0)
    
    chunk_size = n_items // n_chunks
    remainder = n_items % n_chunks
    
    # Distribute remainder across first chunks
    if k < remainder:
        start = k * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = k * chunk_size + remainder
        end = start + chunk_size
    
    return range(start, min(end, n_items))


def contiguous_chunk_array(arr, n_chunks: int, k: int):
    """
    Extract a contiguous chunk from array for better memory layout.
    
    Parameters
    ----------
    arr : array_like
        Array to chunk
    n_chunks : int
        Number of chunks
    k : int
        Chunk index
        
    Returns
    -------
    ndarray
        Contiguous chunk array
    """
    indices = chunk_indices(len(arr), n_chunks, k)
    chunk = arr[indices]
    # Ensure contiguity for MPI operations
    return np.ascontiguousarray(chunk) if hasattr(chunk, 'flags') else chunk


def scatter_dict_list(comm, payload: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Scatter list of dictionaries across MPI ranks.
    
    Parameters
    ----------
    comm : MPI communicator
        Active MPI communicator
    payload : list of dict, optional
        Data to scatter (only used on root rank)
        
    Returns
    -------
    list of dict
        Local chunk for this rank
    """
    if comm.size == 1:
        return payload if payload else []
    
    # Scatter payload
    local_data = comm.scatter(payload, root=0)
    return local_data if local_data is not None else []


def gather_nested(comm, local_blocks: List[Dict], root: int = 0) -> Optional[List[Dict]]:
    """
    Gather nested dictionaries from all ranks.
    
    Parameters
    ----------
    comm : MPI communicator
        Active MPI communicator
    local_blocks : list of dict
        Local data blocks to gather
    root : int
        Rank to gather data to
        
    Returns
    -------
    list of dict or None
        Gathered data (only on root rank)
    """
    if comm.size == 1:
        return local_blocks
    
    # Gather all local blocks
    all_blocks = comm.gather(local_blocks, root=root)
    
    if comm.rank == root and all_blocks:
        # Flatten list of lists
        result = []
        for rank_blocks in all_blocks:
            if rank_blocks:
                result.extend(rank_blocks)
        return result
    
    return None


def barrier_print(comm, msg: str, rank: Optional[int] = None) -> None:
    """
    Print message from specified rank after barrier.
    
    Parameters
    ----------
    comm : MPI communicator
        Active MPI communicator
    msg : str
        Message to print
    rank : int, optional
        Rank to print from (default: 0)
    """
    if rank is None:
        rank = 0
    
    comm.Barrier()
    if comm.rank == rank:
        print(msg, flush=True)
    comm.Barrier()


def broadcast_arrays(comm, arrays_dict: Dict[str, Any], root: int = 0) -> Dict[str, Any]:
    """
    Broadcast dictionary of arrays from root to all ranks.
    
    Parameters
    ----------
    comm : MPI communicator
        Active MPI communicator
    arrays_dict : dict
        Dictionary of arrays to broadcast
    root : int
        Root rank for broadcast
        
    Returns
    -------
    dict
        Broadcast arrays dictionary
    """
    return comm.bcast(arrays_dict, root=root)


def check_mpi_available() -> bool:
    """
    Check if mpi4py is available for import.
    
    Returns
    -------
    bool
        True if mpi4py can be imported
    """
    try:
        import mpi4py
        return True
    except ImportError:
        return False 