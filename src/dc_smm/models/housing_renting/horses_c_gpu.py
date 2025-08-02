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
    Numba CUDA kernel to solve the VFI problem for a single (h, y, w) point.
    Modified to use 2D grid and iterate over w dimension internally.
    """
    h, y = cuda.grid(2)

    if h < n_H and y < n_Y:
        # Iterate over all wealth points for this (h,y) pair
        for iw in range(n_W):
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
    # Modified to use 2D grid to avoid CUDA limits with large grids
    # Kernel now iterates over wealth dimension internally
    threads_per_block = (16, 16)  # 256 threads per block
    blocks_per_grid_x = max(1, (n_H + threads_per_block[0] - 1) // threads_per_block[0])
    blocks_per_grid_y = max(1, (n_Y + threads_per_block[1] - 1) // threads_per_block[1])
    
    # Ensure minimum GPU utilization
    if blocks_per_grid_x * blocks_per_grid_y == 1:
        # If grid is too small, use smaller thread blocks to get more blocks
        if n_H <= 4 and n_Y <= 4:
            threads_per_block = (min(n_H, 4), min(n_Y, 4))
            blocks_per_grid_x = max(1, (n_H + threads_per_block[0] - 1) // threads_per_block[0])
            blocks_per_grid_y = max(1, (n_Y + threads_per_block[1] - 1) // threads_per_block[1])
    
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Debug output for block configuration
    print(f"[GPU DEBUG] Threads per block: {threads_per_block}")
    print(f"[GPU DEBUG] Blocks per grid: {blocks_per_grid}")
    print(f"[GPU DEBUG] Total blocks: {blocks_per_grid[0] * blocks_per_grid[1]:,}")
    
    # Check CUDA limits for 2D grid
    device = cuda.get_current_device()
    max_grid_dim = device.MAX_GRID_DIM_X, device.MAX_GRID_DIM_Y
    print(f"[GPU DEBUG] Max 2D grid dimensions: {max_grid_dim}")
    
    if any(blocks_per_grid[i] > max_grid_dim[i] for i in range(2)):
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
        gradient_threads = (16, 16)
        gradient_blocks_h = max(1, (n_H + gradient_threads[0] - 1) // gradient_threads[0])
        gradient_blocks_y = max(1, (n_Y + gradient_threads[1] - 1) // gradient_threads[1])
        
        # Ensure minimum GPU utilization
        if gradient_blocks_h * gradient_blocks_y == 1:
            if n_H <= 4 and n_Y <= 4:
                gradient_threads = (min(n_H, 4), min(n_Y, 4))
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
    # Use same thread configuration as main kernel for better occupancy
    # Note: continuation kernel expects (iw, ih, iy) order
    continuation_threads = (16, 16, 4)  # Increased from (8, 8, 8)
    continuation_blocks_x = max(1, (n_W + continuation_threads[0] - 1) // continuation_threads[0])
    continuation_blocks_y = max(1, (n_H + continuation_threads[1] - 1) // continuation_threads[1])
    continuation_blocks_z = max(1, (n_Y + continuation_threads[2] - 1) // continuation_threads[2])
    
    # Ensure minimum GPU utilization for 3D kernel
    total_blocks = continuation_blocks_x * continuation_blocks_y * continuation_blocks_z
    if total_blocks <= 2:
        # Adjust thread configuration for small grids
        continuation_threads = (
            min(n_W, 8),
            min(n_H, 8),
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