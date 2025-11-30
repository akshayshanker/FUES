# --------------------------------------------------------------------------
#  pool_vfi.py  – dense-grid VFI with pure-Python + NumPy + ProcessPool
# --------------------------------------------------------------------------
# Keep each worker single-threaded - set BEFORE any other imports
import os
os.environ["NUMBA_NUM_THREADS"]      = "1"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from numba import njit
from .horses_common import (          # already used elsewhere
    bellman_obj, piecewise_gradient
)

# --------------------------------------------------------------------------
# JIT-compiled exhaustive search
# --------------------------------------------------------------------------
@njit(cache=True)
def _search_best_a(w_val, H_val, beta, delta,
                   a_low, a_high, a_grid, V_slice,
                   u_func, n_grid):
    """
    Exhaustive search over `n_grid` candidate a′ on [a_low, a_high].
    Returns (best_a, best_Q).
    """
    step_inv = 1.0 / (n_grid - 1)
    best_Q = -1e110
    best_a = a_low

    #print(f"[{os.getpid()}] got task (iw={iw}, h={h}, y={y})", flush=True)


    for g in range(n_grid):
        a_try = a_low + (a_high - a_low) * g * step_inv
        # `bellman_obj` is already njit'ed in horses_common
        Q_try = bellman_obj(a_try, w_val, H_val,
                            beta, delta, a_grid, V_slice, u_func)
        if Q_try > best_Q:
            best_Q = Q_try
            best_a = a_try
    return best_a, best_Q

# --------------------------------------------------------------------------
# Global broadcast of V_next and u_func
# --------------------------------------------------------------------------
_GLOBAL_V = None
_GLOBAL_U_FUNC = None

def _init_pool(V_next, u_func):
    global _GLOBAL_V, _GLOBAL_U_FUNC
    _GLOBAL_V = V_next  # read-only view for every worker
    _GLOBAL_U_FUNC = u_func  # utility function for every worker

# --------------------------------------------------------------------------
# 1. Helper run by every worker -- must be top-level to be picklable
# --------------------------------------------------------------------------
def _solve_one_w(args):
    """
    Solve the inner maximisation for one (w,h,y) triple.

    Returns scalars that slot straight into the big 3-D result arrays.
    """

    
    
    (iw, w_val, h, y, h_nxt_ind, H_val,
     a_grid, beta, delta, m_bar, n_grid) = args

    n_grid = int(n_grid)          # safety – make absolutely sure

    # Get the slice and u_func from global variables
    V_slice = _GLOBAL_V[:, h_nxt_ind, y].copy()
    u_func = _GLOBAL_U_FUNC

    a_low  = a_grid[0]
    a_high = min(w_val - 1e-12, a_grid[-1])
    if a_high <= a_low + 1e-14:
        a_high = a_low

    # ----- fast search in compiled code ---------------------------------
    #if iw == 0 and h == 0 and y == 0:      # print only for the first (w,h,y)
    #    print(f"[PID {os.getpid():>6}] starting search (n_grid={n_grid})",
    #          flush=True)
    #print(f"[{os.getpid()}] starting task (iw={iw}, h={h}, y={y})", flush=True)

    best_a, best_Q = _search_best_a(
        w_val, H_val, beta, delta,
        a_low, a_high, a_grid, V_slice,
        u_func, n_grid
    )
    # --------------------------------------------------------------------

    c_star = max(w_val - best_a, 1e-10)                 # keep c > 0
    if np.isinf(best_Q):
        best_Q = -1e100
        best_a = 1e-100

    # continuation value + λ  — scalar version
    c_prime     = piecewise_gradient(np.array([c_star]), np.array([w_val]), m_bar)[0]
    uc_now      = 1.0 / c_star
    V_cntn      = (best_Q - (1 - delta)*u_func(c_star, H_val)) / delta
    lambda_cntn = (uc_now - (1 - delta)*c_prime*uc_now) / delta

    return (os.getpid(),            # ← NEW
            iw, h, y,
            c_star, best_a, best_Q, V_cntn, lambda_cntn)


# --------------------------------------------------------------------------
# 2. Public driver – API matches the old MPI/Numba kernel
# --------------------------------------------------------------------------
def _solve_vfi_pool_grid(V_next, w_grid, a_grid, H_grid,
                         beta,   delta, m_bar,
                         u_func, h_nxt_ind_array, thorn,
                         n_grid=2000,
                         max_workers=None,
                         chunksize=64):
    """
    Dense-grid VFI solved with `concurrent.futures.ProcessPoolExecutor`.

    All heavy work (search over `n_grid` candidate a′) is parallelised;
    everything else runs in the master process – so no duplicated
    logging, file-writes, RNGs, etc.

    Returns the same five arrays as the original MPI routine:
      policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn
    """
    # make sure n_grid is an int – it may arrive as float from settings
    n_grid = int(round(n_grid))
    if n_grid < 2:
        raise ValueError("n_grid must be ≥ 2")

    n_W = w_grid.size
    _, n_H, n_Y = V_next.shape
    max_workers = max_workers or os.cpu_count()

    # -------- allocate outputs on the master process -------------------
    policy_c     = np.empty((n_W, n_H, n_Y))
    policy_a     = np.empty_like(policy_c)
    Q_dcsn       = np.empty_like(policy_c)
    V_cntn       = np.empty_like(policy_c)
    lambda_cntn  = np.empty_like(policy_c)

    # -------- build flat work-list ------------------------------------
    tasks = []
    for h in range(n_H):
        H_val     = H_grid[h] * thorn
        h_nxt_ind = h_nxt_ind_array[h]
        for y in range(n_Y):
            for iw, w_val in enumerate(w_grid):
                tasks.append((iw, w_val, h, y, h_nxt_ind, H_val,
                              a_grid, beta, delta, m_bar, n_grid))

    # -------- fan-out / fan-in ----------------------------------------
    worker_hits = Counter()
    
    with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_pool,
            initargs=(V_next, u_func)) as pool:
        for (pid, iw, h, y, c_star, a_star,
             Q_star, V_c, lam) in pool.map(_solve_one_w, tasks,
                                           chunksize=chunksize):
            worker_hits[pid] += 1
            
            policy_c[iw, h, y]     = c_star
            policy_a[iw, h, y]     = a_star
            Q_dcsn[iw, h, y]       = Q_star
            V_cntn[iw, h, y]       = V_c
            lambda_cntn[iw, h, y]  = lam

    # after the pool closes, show the distribution
    print("─ Task distribution per worker ─")
    for pid, n in worker_hits.most_common():
        print(f"  PID {pid}: {n:6d} tasks")
    print("  total:", sum(worker_hits.values()))

    return policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn
