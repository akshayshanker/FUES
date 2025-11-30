# -----------------------------------------------------------------------------
#  mpi_vfi.py   – dense-grid VFI with pure-Python + NumPy + MPI
# -----------------------------------------------------------------------------
import numpy as np
from mpi4py import MPI
from quantecon.optimize.scalar_maximization import brent_max   # already used
from dc_smm.models.housing_renting.horses_common import piecewise_gradient
from dc_smm.models.housing_renting.horses_common import bellman_obj

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nproc = comm.Get_size()

def _solve_vfi_mpi_grid(V_next, w_grid, a_grid, H_grid,
                        beta, delta, m_bar,
                        u_func, h_nxt_ind_array, thorn,
                        n_grid=2000):
    """
    Dense-grid VFI solved in parallel with MPI (no Numba needed).

    * Work is split across (h,y) pairs: each rank gets a contiguous block.
    * All ranks receive read-only data (broadcast); they compute their block
      and send the results back to rank-0, which assembles full arrays.
    """
    n_W = w_grid.size
    n_A, n_H, n_Y = V_next.shape

    # ------------------------------------------------------------------
    # distribute the (h,y) combinations ----------------------------------------------------
    hy_pairs = [(h, y) for h in range(n_H) for y in range(n_Y)]
    chunk    = np.array_split(hy_pairs, nproc)[rank]   # each rank’s slice

    # local result buffers (only for the slice we own)
    local_c      = np.empty((n_W, len(chunk)))
    local_a      = np.empty_like(local_c)
    local_Q      = np.empty_like(local_c)
    local_V      = np.empty_like(local_c)
    local_lambda = np.empty_like(local_c)

    step_inv = 1.0 / (n_grid - 1)

    # ------------------------------------------------------------------
    # main loop over the slice assigned to *this* rank ------------------
    for idx, (h, y) in enumerate(chunk):
        H_val      = H_grid[h] * thorn
        h_nxt_ind  = h_nxt_ind_array[h]
        V_slice    = V_next[:, h_nxt_ind, y]           # contiguous 1-D

        for iw, w_val in enumerate(w_grid):
            a_low  = a_grid[0]
            a_high = min(w_val - 1e-12, a_grid[-1])
            if a_high <= a_low + 1e-14:
                a_high = a_low

            best_Q = -1e110
            best_a = a_low
            # exhaustive search
            for g in range(n_grid):
                a_try = a_low + (a_high - a_low) * g * step_inv
                Q_try = bellman_obj(a_try, w_val, H_val,
                                    beta, delta, a_grid, V_slice, u_func)
                if Q_try > best_Q:
                    best_Q = Q_try
                    best_a = a_try

            c_star = max(w_val - best_a, 1e-10)    # avoid non-positive c
            if np.isinf(best_Q):
                best_Q = -1e100
                best_a = 1e-100

            local_c[iw, idx]      = c_star
            local_a[iw, idx]      = best_a
            local_Q[iw, idx]      = best_Q

        # continuation value + λ  (vectorised over w)
        c_prime = piecewise_gradient(local_c[:, idx], w_grid, m_bar)
        uc_now  = 1.0 / local_c[:, idx]
        local_V[:, idx]      = (local_Q[:, idx]
                               - (1 - delta)*u_func(local_c[:, idx], H_val)) / delta
        local_lambda[:, idx] = (uc_now
                               - (1 - delta)*c_prime*uc_now) / delta

    # ------------------------------------------------------------------
    # gather results to rank-0 ------------------------------------------
    #  -> each array is 2-D (w × |chunk|); gather into flat vectors then reshape
    def gather_array(local_arr, dtype):
        recvbuf = None
        counts  = np.array([w_grid.size * len(slc) for slc in np.array_split(hy_pairs, nproc)], dtype='i')
        displs  = np.insert(np.cumsum(counts), 0, 0)[0:-1]
        if rank == 0:
            recvbuf = np.empty(sum(counts), dtype=dtype)
        comm.Gatherv(local_arr.ravel(), (recvbuf, counts, displs, MPI._typedict[dtype.char]), root=0)
        if rank == 0:
            # reshape to (w, h, y)
            full = np.empty((n_W, n_H, n_Y), dtype=dtype)
            idx0 = 0
            for (h, y), length in zip(hy_pairs, counts.repeat(w_grid.size)[:len(hy_pairs)]):
                full[:, h, y] = recvbuf[idx0: idx0 + n_W]
                idx0 += n_W
            return full
        else:
            return None

    policy_c   = gather_array(local_c,      np.float64)
    policy_a   = gather_array(local_a,      np.float64)
    Q_dcsn     = gather_array(local_Q,      np.float64)
    V_cntn     = gather_array(local_V,      np.float64)
    lambda_cntn= gather_array(local_lambda, np.float64)

    # ------------------------------------------------------------------
    # broadcast assembled arrays back to all ranks (optional) -----------
    policy_c    = comm.bcast(policy_c,    root=0)
    policy_a    = comm.bcast(policy_a,    root=0)
    Q_dcsn      = comm.bcast(Q_dcsn,      root=0)
    V_cntn      = comm.bcast(V_cntn,      root=0)
    lambda_cntn = comm.bcast(lambda_cntn, root=0)

    return policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn
