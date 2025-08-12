import os
import gc
import numpy as np
from numba import cuda, njit, prange
import math

from dc_smm.models.housing_renting.horses_common import (
    bellman_obj_gpu, piecewise_gradient, get_u_func
)

# ======================================================================
#  GPU VFI Kernel
# ======================================================================


@cuda.jit
def vfi_gpu_kernel(
    policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn,  # Output arrays
    V_next, w_grid, a_grid, H_grid, h_nxt_ind_array,  # Input data arrays
    beta, delta, m_bar, thorn, n_grid,               # Scalar parameters
    alpha, kappa, iota,                              # Utility parameters
    n_W, n_H, n_Y,                                   # Grid dimensions
):
    """
    Numba CUDA kernel to solve the VFI problem for a single (w, h, y) point.
    Uses 3D grid for maximum parallelism across all dimensions.
    """
    iw, h, y = cuda.grid(3)

    if iw < n_W and h < n_H and y < n_Y:
        # --- 1. Setup for this specific grid point ---
        w_val = w_grid[iw]
        H_val = H_grid[h] * thorn
        h_nxt_ind = h_nxt_ind_array[h]

        # V_slice is not created explicitly; we pass indices to bellman_obj_gpu
        
        a_low = a_grid[0]
        a_high = min(w_val - 1e-12, a_grid[-1]+30) #TODO: HARDWIRE THIS
        
        if a_high <= a_low + 1e-14:
            a_high = a_low
            
        # --- 2. Dense Grid Search ---
        best_Q = -1e110
        best_a = a_low
        step_inv = 1.0 / (n_grid - 1)

        for g in range(n_grid):
            a_try = a_low + (a_high - a_low) * g * step_inv
            Q_try = bellman_obj_gpu(
                a_try, w_val, H_val, beta, delta, a_grid, V_next,
                h_nxt_ind, y,
                alpha, kappa, iota
            )
            if Q_try > best_Q:
                best_Q = Q_try
                best_a = a_try
        
        c_star = w_val - best_a
        if math.isinf(best_Q) or c_star <= 0.0:
            c_star = 1e-10
            best_Q = -1e100
            best_a = 1e-100
            
        policy_c[iw, h, y] = c_star
        policy_a[iw, h, y] = best_a
        Q_dcsn[iw, h, y] = best_Q

@cuda.jit
def vfi_gpu_kernel_slice(
    V_prev,                       # Input: full V from previous iteration (device)
    a_grid, H_grid, w_grid_slice, # Grid arrays (all on device)
    h_nxt_ind_array,
    V_out_slice, P_out_slice, Q_out_slice,  # Outputs (slice only, device)
    beta, delta, m_bar, thorn, n_grid,     # Scalar parameters
    alpha, kappa, iota,                     # Utility parameters
    w_offset,                               # Offset for global indexing
    n_W_slice, n_H, n_Y                    # Slice dimensions
):
    """
    Modified kernel that processes a wealth slice while reading from full V_prev.
    
    Key difference from original:
    - Reads from full V_prev using global index (iw_global) 
    - Writes to slice arrays using local index (iw_local)
    - w_grid_slice is pre-sliced device array
    """
    iw_local, h, y = cuda.grid(3)
    
    if iw_local < n_W_slice and h < n_H and y < n_Y:
        iw_global = iw_local + w_offset
        
        w_val = w_grid_slice[iw_local]
        H_val = H_grid[h] * thorn
        h_nxt_ind = h_nxt_ind_array[h]
        
        a_low = a_grid[0]
        a_high = min(w_val - 1e-12, a_grid[-1] + 30)
        
        if a_high <= a_low + 1e-14:
            a_high = a_low
        
        best_Q = -1e110
        best_a = a_low
        step_inv = 1.0 / (n_grid - 1)
        
        for g in range(n_grid):
            a_try = a_low + (a_high - a_low) * g * step_inv
            
            Q_try = bellman_obj_gpu(
                a_try, w_val, H_val, beta, delta, a_grid, 
                V_prev,
                h_nxt_ind, y,
                alpha, kappa, iota
            )
            
            if Q_try > best_Q:
                best_Q = Q_try
                best_a = a_try
        
        c_star = w_val - best_a
        if math.isinf(best_Q) or c_star <= 0.0:
            c_star = 1e-10
            best_Q = -1e100
            best_a = 1e-100
        
        # Compute V from Q: V = (Q - (1-delta)*u(c,h)) / delta
        # where u(c,h) = alpha*log(c) + (1-alpha)*log(kappa*H + iota)
        if c_star > 0 and kappa * H_val + iota > 0:
            u_current = alpha * math.log(c_star) + (1 - alpha) * math.log(kappa * H_val + iota)
            V_out_slice[iw_local, h, y] = (best_Q - (1 - delta) * u_current) / delta
        else:
            V_out_slice[iw_local, h, y] = -1e100
        
        P_out_slice[iw_local, h, y] = best_a
        Q_out_slice[iw_local, h, y] = best_Q

# ======================================================================
#  GPU Continuation Value and Gradient Kernels
# ======================================================================

@cuda.jit
def calculate_gradient_gpu_kernel(
    policy_c, w_grid, gradient_c, m_bar, eps,
    n_W, n_H, n_Y
):
    """
    GPU kernel to calculate piecewise gradients of consumption policy.
    Each thread handles one (h, y) pair and computes gradients along w dimension.
    """
    h, y = cuda.grid(2)
    
    if h < n_H and y < n_Y:
        # Each thread computes gradient for all wealth points
        # This is necessary because gradient computation needs neighboring points
        
        # First pass: compute raw gradients
        for iw in range(n_W):
            left_ok = False
            right_ok = False
            g_raw = -1.0  # Invalid marker
            
            # Check left neighbor
            if iw > 0:
                df = policy_c[iw, h, y] - policy_c[iw-1, h, y]
                dx = w_grid[iw] - w_grid[iw-1]
                if dx > 0 and abs(df/dx) <= m_bar:
                    left_ok = True
            
            # Check right neighbor
            if iw < n_W - 1:
                df = policy_c[iw+1, h, y] - policy_c[iw, h, y]
                dx = w_grid[iw+1] - w_grid[iw]
                if dx > 0 and abs(df/dx) <= m_bar:
                    right_ok = True
            
            # Compute gradient based on available neighbors
            if left_ok and right_ok:
                g_raw = (policy_c[iw+1, h, y] - policy_c[iw-1, h, y]) / (w_grid[iw+1] - w_grid[iw-1])
            elif right_ok:
                g_raw = (policy_c[iw+1, h, y] - policy_c[iw, h, y]) / (w_grid[iw+1] - w_grid[iw])
            elif left_ok:
                g_raw = (policy_c[iw, h, y] - policy_c[iw-1, h, y]) / (w_grid[iw] - w_grid[iw-1])
            
            # Store only if positive
            if g_raw > 0:
                gradient_c[iw, h, y] = g_raw
            else:
                gradient_c[iw, h, y] = -1.0  # Mark as invalid
        
        # Second pass: fill invalid gradients with nearest valid neighbor
        for iw in range(n_W):
            if gradient_c[iw, h, y] <= 0:  # Invalid gradient
                # Search for nearest valid gradient
                found = False
                for offset in range(1, n_W):
                    # Check left
                    if iw - offset >= 0 and gradient_c[iw - offset, h, y] > 0:
                        gradient_c[iw, h, y] = gradient_c[iw - offset, h, y]
                        found = True
                        break
                    # Check right
                    if iw + offset < n_W and gradient_c[iw + offset, h, y] > 0:
                        gradient_c[iw, h, y] = gradient_c[iw + offset, h, y]
                        found = True
                        break
                
                if not found:
                    # Fallback: use eps
                    gradient_c[iw, h, y] = eps

@cuda.jit
def calculate_continuation_values_gpu_kernel(
    policy_c, Q_dcsn, gradient_c, H_grid, V_cntn, lambda_cntn,
    delta, thorn, alpha, kappa, iota, compute_lambda,
    n_W, n_H, n_Y
):
    """
    GPU kernel to calculate continuation values and optionally lambda.
    """
    iw, ih, iy = cuda.grid(3)
    
    if iw < n_W and ih < n_H and iy < n_Y:
        c_now = policy_c[iw, ih, iy]
        H_val = H_grid[ih] * thorn
        
        # Compute utility using log utility function
        if c_now <= 0:
            util = -1e12
        else:
            util = alpha * math.log(c_now) + (1 - alpha) * math.log(kappa * H_val + iota)
        
        # Compute continuation value
        V_cntn[iw, ih, iy] = (Q_dcsn[iw, ih, iy] - (1 - delta) * util) / delta
        
        # Compute lambda if requested
        if compute_lambda:
            uc_now = alpha / c_now  # Marginal utility for log utility
            c_prime = gradient_c[iw, ih, iy]
            lambda_cntn[iw, ih, iy] = (uc_now - (1 - delta) * c_prime * uc_now) / delta
        else:
            lambda_cntn[iw, ih, iy] = 0.0

@njit(parallel=True)
def _calculate_continuation_values_cpu(
    policy_c, Q_dcsn, H_grid, w_grid, V_cntn, lambda_cntn,
    delta, m_bar, thorn, u_func_cpu
):
    """
    JIT-compiled function to calculate continuation values on the CPU
    in parallel.
    """
    n_W, n_H, n_Y = policy_c.shape
    for h in prange(n_H):
        for y in range(n_Y):
            H_val = H_grid[h] * thorn
            #c_prime = piecewise_gradient(policy_c[:, h, y], w_grid, m_bar)
            for iw in range(n_W):
                c_now = policy_c[iw, h, y]
                uc_now = 1.0 / c_now
                V_cntn[iw, h, y] = (Q_dcsn[iw, h, y] - (1 - delta) * u_func_cpu(c_now, H_val)) / delta
                #lambda_cntn[iw, h, y] = (uc_now - (1 - delta) * c_prime[iw] * uc_now) / delta

# ======================================================================
#  Host-Side GPU Launcher
# ======================================================================

def solve_vfi_gpu(vlu_cntn, model):
    """
    Host function to manage data transfer and launch the VFI CUDA kernel.
    """
    # --- 1. Extract Parameters and Grids (CPU) ---
    w_grid = model.num.state_space.dcsn.grids.w.astype(np.float64)
    a_grid = model.num.state_space.cntn.grids.a_nxt.astype(np.float64)
    H_grid = model.num.state_space.cntn.grids.H_nxt.astype(np.float64)

    if "RNT" in model.stage_name:
        thorn = model.param.thorn
    else:
        thorn = 1
        
    beta = float(model.param.beta)
    delta = float(model.param.delta_pb)
    m_bar = float(model.settings_dict["m_bar"])
    n_grid = int(model.settings_dict["N_arg_grid_vfi"])

    alpha = float(model.param.alpha)
    kappa = float(model.param.kappa)
    iota = float(model.param.iota)
    
    h_nxt_ind_array = model.num.functions.g_ve_h_ind(H_ind=np.arange(H_grid.size))

    # --- 2. Allocate and Transfer Data to GPU ---
    d_V_next = cuda.to_device(vlu_cntn)
    d_w_grid = cuda.to_device(w_grid)
    d_a_grid = cuda.to_device(a_grid)
    d_H_grid = cuda.to_device(H_grid)
    d_h_nxt_ind_array = cuda.to_device(h_nxt_ind_array)
    
    n_W, n_H, n_Y = w_grid.size, H_grid.size, vlu_cntn.shape[2]
    
    # Debug output for grid dimensions
    print(f"[GPU DEBUG] Grid dimensions: n_W={n_W}, n_H={n_H}, n_Y={n_Y}")
    print(f"[GPU DEBUG] Total grid points: {n_W * n_H * n_Y:,}")
    
    # Allocate output arrays on the GPU
    d_policy_c = cuda.device_array((n_W, n_H, n_Y), dtype=np.float64)
    d_policy_a = cuda.device_array_like(d_policy_c)
    d_Q_dcsn = cuda.device_array_like(d_policy_c)
    d_V_cntn = cuda.device_array_like(d_policy_c)
    d_lambda_cntn = cuda.device_array_like(d_policy_c)

    # --- 3. Configure and Launch Kernel ---
    # Use 3D grid for maximum parallelism across all dimensions
    # Optimize thread configuration for better GPU utilization
    
    # Determine thread configuration based on problem size and GPU constraints
    if n_W * n_H * n_Y < 1000:  # Small problem
        threads_per_block = (min(n_W, 8), min(n_H, 8), min(n_Y, 8))
    else:  # Large problem - balance threads vs register usage
        # 8*8*8 = 512 threads (safer for complex VFI kernel)
        threads_per_block = (8, 8, 8)
    
    blocks_per_grid_x = max(1, (n_W + threads_per_block[0] - 1) // threads_per_block[0])
    blocks_per_grid_y = max(1, (n_H + threads_per_block[1] - 1) // threads_per_block[1])
    blocks_per_grid_z = max(1, (n_Y + threads_per_block[2] - 1) // threads_per_block[2])
    
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
    
    # Debug output for block configuration
    print(f"[GPU DEBUG] Grid dimensions: W={n_W}, H={n_H}, Y={n_Y}")
    print(f"[GPU DEBUG] Threads per block: {threads_per_block} = {threads_per_block[0]*threads_per_block[1]*threads_per_block[2]} threads")
    print(f"[GPU DEBUG] Blocks per grid: {blocks_per_grid}")
    print(f"[GPU DEBUG] Total blocks: {blocks_per_grid[0] * blocks_per_grid[1] * blocks_per_grid[2]:,}")
    
    # Check CUDA limits for 3D grid
    device = cuda.get_current_device()
    max_grid_dim = device.MAX_GRID_DIM_X, device.MAX_GRID_DIM_Y, device.MAX_GRID_DIM_Z
    print(f"[GPU DEBUG] Max 3D grid dimensions: {max_grid_dim}")
    
    if any(blocks_per_grid[i] > max_grid_dim[i] for i in range(3)):
        print("[GPU ERROR] Block grid dimensions exceed CUDA limits!")
        for i, (actual, limit) in enumerate(zip(blocks_per_grid, max_grid_dim)):
            if actual > limit:
                print(f"  Dimension {i}: {actual} > {limit}")
    expr_str = model.math["functions"]["u_func"]["expr"]
    param_vals = {
        "alpha": model.param.alpha,
        "kappa": model.param.kappa,
        "iota": model.param.iota,
    }
    u_func_cpu = get_u_func(expr_str, param_vals)

    vfi_gpu_kernel[blocks_per_grid, threads_per_block](
        d_policy_c, d_policy_a, d_Q_dcsn, d_V_cntn, d_lambda_cntn,
        d_V_next, d_w_grid, d_a_grid, d_H_grid, d_h_nxt_ind_array,
        beta, delta, m_bar, thorn, n_grid,
        alpha, kappa, iota,
        n_W, n_H, n_Y,
    )

    # --- 4. GPU Continuation Values Calculation ---
    # Get compute_lambda setting, handle both settings and settings_dict
    if hasattr(model, 'settings_dict'):
        compute_lambda = model.settings_dict.get("compute_lambda_gpu", False)
    elif hasattr(model, 'settings'):
        compute_lambda = model.settings.get("compute_lambda_gpu", False)
    else:
        compute_lambda = False
    
    if compute_lambda:
        # First compute gradients if lambda is needed
        d_gradient_c = cuda.device_array_like(d_policy_c)
        
        # Configure gradient kernel (2D grid for h, y)
        # Use balanced configuration: 16*16 = 256 threads
        gradient_threads = (16, 16)
        gradient_blocks_h = max(1, (n_H + gradient_threads[0] - 1) // gradient_threads[0])
        gradient_blocks_y = max(1, (n_Y + gradient_threads[1] - 1) // gradient_threads[1])
        
        # Ensure minimum GPU utilization
        if gradient_blocks_h * gradient_blocks_y == 1 or n_H * n_Y < 100:
            if n_H <= 8 and n_Y <= 8:
                gradient_threads = (min(n_H, 8), min(n_Y, 8))
                gradient_blocks_h = max(1, (n_H + gradient_threads[0] - 1) // gradient_threads[0])
                gradient_blocks_y = max(1, (n_Y + gradient_threads[1] - 1) // gradient_threads[1])
        
        gradient_blocks = (gradient_blocks_h, gradient_blocks_y)
        
        calculate_gradient_gpu_kernel[gradient_blocks, gradient_threads](
            d_policy_c, d_w_grid, d_gradient_c, m_bar, 0.9,
            n_W, n_H, n_Y
        )
    else:
        d_gradient_c = cuda.device_array_like(d_policy_c)  # Dummy array
    
    # Calculate continuation values on GPU
    # Use balanced thread configuration to avoid resource exhaustion
    # Note: continuation kernel expects (iw, ih, iy) order
    continuation_threads = (8, 8, 8)  # 512 threads (balanced)
    continuation_blocks_x = max(1, (n_W + continuation_threads[0] - 1) // continuation_threads[0])
    continuation_blocks_y = max(1, (n_H + continuation_threads[1] - 1) // continuation_threads[1])
    continuation_blocks_z = max(1, (n_Y + continuation_threads[2] - 1) // continuation_threads[2])
    
    # Ensure minimum GPU utilization for 3D kernel
    total_blocks = continuation_blocks_x * continuation_blocks_y * continuation_blocks_z
    if total_blocks <= 2 or n_W * n_H * n_Y < 1000:
        # Adjust thread configuration for small grids
        continuation_threads = (
            min(n_W, 4),
            min(n_H, 4),
            min(n_Y, 4)
        )
        continuation_blocks_x = max(1, (n_W + continuation_threads[0] - 1) // continuation_threads[0])
        continuation_blocks_y = max(1, (n_H + continuation_threads[1] - 1) // continuation_threads[1])
        continuation_blocks_z = max(1, (n_Y + continuation_threads[2] - 1) // continuation_threads[2])
    
    continuation_blocks = (continuation_blocks_x, continuation_blocks_y, continuation_blocks_z)
    
    calculate_continuation_values_gpu_kernel[continuation_blocks, continuation_threads](
        d_policy_c, d_Q_dcsn, d_gradient_c, d_H_grid, d_V_cntn, d_lambda_cntn,
        delta, thorn, alpha, kappa, iota, compute_lambda,
        n_W, n_H, n_Y
    )
    
    # --- 5. Copy Results Back to CPU ---
    policy_c = d_policy_c.copy_to_host()
    policy_a = d_policy_a.copy_to_host()  # Now we actually compute this on GPU
    Q_dcsn = d_Q_dcsn.copy_to_host()
    V_cntn = d_V_cntn.copy_to_host()
    lambda_cntn = d_lambda_cntn.copy_to_host()

    return policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn
def detect_num_gpus():
    """Detect number of available GPUs."""
    try:
        return cuda.gpus.count()
    except:
        return 0

def _split_0axis(N: int, world: int, rank: int):
    """Compute balanced slice for rank along axis 0."""
    q, r = divmod(N, world)
    a = rank * q + min(rank, r)
    b = a + q + (1 if rank < r else 0)
    return slice(a, b)

def _counts_displs(shape, world: int):
    """Compute counts and displacements for Allgatherv."""
    N = shape[0]
    rest = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    lens = [N // world + (1 if r < (N % world) else 0) for r in range(world)]
    counts = [l * rest for l in lens]
    displs = [0]
    for i in range(1, world):
        displs.append(displs[i-1] + counts[i-1])
    return lens, counts, displs, rest

def vfh_hd_grid_compute_slice(cfg, V_prev_h, P_prev_h, meta, sl, device_id):
    """
    Compute V_new[sl], P_new[sl] on assigned GPU.
    
    Parameters
    ----------
    V_prev_h : array
        Full V from previous iteration (host memory)
    P_prev_h : array or None
        Full policy from previous iteration (host memory)
    sl : slice
        Wealth dimension slice this rank computes
    device_id : int
        GPU device ID for this rank
    
    Returns
    -------
    V_new_slice, P_new_slice : arrays
        Computed slice of new value function and policy
    """
    cuda.select_device(device_id)
    
    d_V_prev = cuda.to_device(V_prev_h)
    
    n_W_slice = sl.stop - sl.start
    n_H = meta['n_H']
    n_Y = meta['n_Y']
    
    w_grid_slice = meta['w_grid'][sl]
    d_w_grid_slice = cuda.to_device(w_grid_slice)
    d_a_grid = cuda.to_device(meta['a_grid'])
    d_H_grid = cuda.to_device(meta['H_grid'])
    d_h_nxt_ind = cuda.to_device(meta['h_nxt_ind_array'])
    
    d_V_slice = cuda.device_array((n_W_slice, n_H, n_Y), dtype=np.float64)
    d_P_slice = cuda.device_array((n_W_slice, n_H, n_Y), dtype=np.float64)
    d_Q_slice = cuda.device_array((n_W_slice, n_H, n_Y), dtype=np.float64)
    
    threads = (8, 8, 8)
    blocks_x = (n_W_slice + threads[0] - 1) // threads[0]
    blocks_y = (n_H + threads[1] - 1) // threads[1]
    blocks_z = (n_Y + threads[2] - 1) // threads[2]
    blocks = (blocks_x, blocks_y, blocks_z)
    
    vfi_gpu_kernel_slice[blocks, threads](
        d_V_prev,
        d_a_grid, d_H_grid,
        d_w_grid_slice,
        d_h_nxt_ind,
        d_V_slice, d_P_slice, d_Q_slice,
        meta['beta'], meta['delta'], meta['m_bar'],
        meta['thorn'], meta['n_grid'],
        meta['alpha'], meta['kappa'], meta['iota'],
        sl.start,
        n_W_slice, n_H, n_Y
    )
    
    V_out = d_V_slice.copy_to_host()
    P_out = d_P_slice.copy_to_host()
    
    return V_out, P_out

def allgatherv_like(comm, part, full, counts, displs):
    """Safe Allgatherv with C-contiguous arrays."""
    part_c = np.ascontiguousarray(part)
    full_c = np.ascontiguousarray(full) 
    comm.Allgatherv(
        part_c.ravel(), 
        [full_c.ravel(), (counts, displs)]
    )
    return full_c

def initialize_vfh_from_config(cfg):
    """
    Initialize value function, policy, and metadata from config.
    
    Parameters
    ----------
    cfg : Config object
        Configuration with model parameters and grids
    
    Returns
    -------
    V_initial : ndarray
        Initial value function (n_W, n_H, n_Y)
    P_initial : ndarray or None
        Initial policy function (n_W, n_H, n_Y) if needed
    meta : dict
        Metadata including grids and parameters
    """
    # Extract grid dimensions from config
    n_W = getattr(cfg, 'n_W', 100)  # Wealth grid points
    n_H = getattr(cfg, 'n_H', 20)   # Housing grid points  
    n_Y = getattr(cfg, 'n_Y', 5)    # Income shock points
    
    # Initialize value function (start with zeros or simple heuristic)
    V_initial = np.zeros((n_W, n_H, n_Y), dtype=np.float64)
    
    # Optionally initialize with a simple heuristic (log utility of wealth)
    w_min = getattr(cfg, 'w_min', 0.1)
    w_max = getattr(cfg, 'w_max', 100.0)
    w_grid = np.linspace(w_min, w_max, n_W)
    
    h_min = getattr(cfg, 'h_min', 0.1)
    h_max = getattr(cfg, 'h_max', 10.0)
    H_grid = np.linspace(h_min, h_max, n_H)
    
    # Simple initialization: V = log(w) for each state
    for i_w in range(n_W):
        for i_h in range(n_H):
            for i_y in range(n_Y):
                V_initial[i_w, i_h, i_y] = np.log(max(w_grid[i_w], 0.01))
    
    # Initialize policy (optional, can be None)
    P_initial = None
    if getattr(cfg, 'compute_policy', True):
        P_initial = np.zeros((n_W, n_H, n_Y), dtype=np.float64)
        # Simple initial policy: save fraction of wealth
        for i_w in range(n_W):
            P_initial[i_w, :, :] = 0.3 * w_grid[i_w]
    
    # Create metadata dictionary
    meta = {
        'n_W': n_W,
        'n_H': n_H,
        'n_Y': n_Y,
        'w_grid': w_grid,
        'H_grid': H_grid,
        'a_grid': np.linspace(0, w_max * 0.9, n_W),  # Assets grid
        'h_nxt_ind_array': np.arange(n_H, dtype=np.int32),  # Housing transition indices
        'beta': getattr(cfg, 'beta', 0.96),
        'delta': getattr(cfg, 'delta', 0.9),
        'm_bar': getattr(cfg, 'm_bar', 1.0),
        'thorn': getattr(cfg, 'thorn', 1.0),
        'n_grid': getattr(cfg, 'n_grid', 50),  # Search grid points
        'alpha': getattr(cfg, 'alpha', 0.7),
        'kappa': getattr(cfg, 'kappa', 0.5),
        'iota': getattr(cfg, 'iota', 0.1),
    }
    
    return V_initial, P_initial, meta

def vfh_mpi_driver(cfg, comm, device_id):
    """MPI driver - all collectives live here."""
    from mpi4py import MPI
    rank, world = comm.Get_rank(), comm.Get_size()
    
    if rank == 0:
        V_prev, P_prev, meta = initialize_vfh_from_config(cfg)
        shapeV, dtypeV = V_prev.shape, V_prev.dtype
        shapeP = P_prev.shape if P_prev is not None else None
        dtypeP = P_prev.dtype if P_prev is not None else None
    else:
        V_prev = P_prev = meta = None
        shapeV = dtypeV = shapeP = dtypeP = None
    
    shapeV = comm.bcast(shapeV, root=0)
    dtypeV = comm.bcast(dtypeV, root=0)
    shapeP = comm.bcast(shapeP, root=0)
    dtypeP = comm.bcast(dtypeP, root=0) if shapeP else None
    meta = comm.bcast(meta, root=0)
    
    if rank != 0:
        V_prev = np.empty(shapeV, dtype=dtypeV, order='C')
        P_prev = np.empty(shapeP, dtype=dtypeP, order='C') if shapeP else None
    
    V_prev = np.ascontiguousarray(V_prev)
    comm.Bcast(V_prev, root=0)
    if P_prev is not None:
        P_prev = np.ascontiguousarray(P_prev)
        comm.Bcast(P_prev, root=0)
    
    lensV, countsV, displsV, restV = _counts_displs(shapeV, world)
    if P_prev is not None:
        lensP, countsP, displsP, restP = _counts_displs(shapeP, world)
    
    N = shapeV[0]
    sl = _split_0axis(N, world, rank)
    
    compute_policy_every = getattr(cfg, 'policy_every', 1)
    compute_policy_final = True
    
    for it in range(cfg.vfi_iters):
        is_last = (it == cfg.vfi_iters - 1)
        compute_policy_now = is_last or (it % compute_policy_every == 0)
        
        V_part, P_part = vfh_hd_grid_compute_slice(
            cfg, V_prev, P_prev, meta, sl, device_id
        )
        
        V_part = np.ascontiguousarray(V_part)
        
        V_new = np.empty_like(V_prev, order='C')
        comm.Allgatherv(
            V_part.ravel(),
            [V_new.ravel(), (countsV, displsV)]
        )
        
        if P_prev is not None and compute_policy_now:
            P_part = np.ascontiguousarray(P_part)
            P_new = np.empty_like(P_prev, order='C')
            comm.Allgatherv(
                P_part.ravel(),
                [P_new.ravel(), (countsP, displsP)]
            )
        else:
            P_new = P_prev
        
        if getattr(cfg, 'check_convergence', False):
            local_resid = np.max(np.abs(V_new - V_prev))
            global_resid = comm.allreduce(local_resid, op=MPI.MAX)
            converged = (global_resid < cfg.tol)
        else:
            converged = False
        
        V_prev, P_prev = V_new, P_new
        
        if converged:
            if rank == 0:
                print(f"Converged at iteration {it+1}, residual: {global_resid:.2e}")
            break
    
    if rank == 0:
        return (V_prev, P_prev, meta)
    else:
        return None 