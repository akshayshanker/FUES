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
    policy_c, policy_a, Q_dcsn, vlu_dcsn, lambda_dcsn,  # Output arrays
    vlu_cntn, w_grid, a_grid, H_grid, h_nxt_ind_array,  # Input data arrays
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
                a_try, w_val, H_val, beta, delta, a_grid, vlu_cntn,
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
    policy_c, Q_dcsn, gradient_c, H_grid, vlu_dcsn, lambda_dcsn,
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
        vlu_dcsn[iw, ih, iy] = (Q_dcsn[iw, ih, iy] - (1 - delta) * util) / delta
        
        # Compute lambda if requested
        if compute_lambda:
            uc_now = alpha / c_now  # Marginal utility for log utility
            c_prime = gradient_c[iw, ih, iy]
            lambda_dcsn[iw, ih, iy] = (uc_now - (1 - delta) * c_prime * uc_now) / delta
        else:
            lambda_dcsn[iw, ih, iy] = 0.0

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
    d_vlu_cntn = cuda.to_device(vlu_cntn)
    d_w_grid = cuda.to_device(w_grid)
    d_a_grid = cuda.to_device(a_grid)
    d_H_grid = cuda.to_device(H_grid)
    d_h_nxt_ind_array = cuda.to_device(h_nxt_ind_array)
    
    n_W, n_H, n_Y = w_grid.size, H_grid.size, vlu_cntn.shape[2]
    
    # Debug output for grid dimensions (only if verbose)
    verbose = model.settings_dict.get("gpu_verbose", False)
    if verbose:
        print(f"[GPU DEBUG] Grid dimensions: n_W={n_W}, n_H={n_H}, n_Y={n_Y}")
        print(f"[GPU DEBUG] Total grid points: {n_W * n_H * n_Y:,}")
    
    # Allocate output arrays on the GPU
    d_policy_c = cuda.device_array((n_W, n_H, n_Y), dtype=np.float64)
    d_policy_a = cuda.device_array_like(d_policy_c)
    d_Q_dcsn = cuda.device_array_like(d_policy_c)
    d_vlu_dcsn = cuda.device_array_like(d_policy_c)
    d_lambda_dcsn = cuda.device_array_like(d_policy_c)

    # --- 3. Configure and Launch Kernel ---
    # Use 3D grid for maximum parallelism across all dimensions
    # Optimize thread configuration for better GPU utilization
    
    # Determine thread configuration based on problem size and GPU constraints
    # Key optimizations:
    # 1. Keep total threads per block at 256 or 512 for better occupancy
    # 2. Make X dimension (wealth) larger for coalesced memory access
    # 3. Ensure dimensions are powers of 2 when possible
    
    if n_W * n_H * n_Y < 1000:  # Small problem
        threads_per_block = (min(n_W, 8), min(n_H, 8), min(n_Y, 8))
    else:  # Large problem - optimize for memory coalescing and occupancy
        # Prefer 256 threads for better occupancy with complex kernels
        # Prioritize X dimension (wealth) for coalesced access
        if n_W >= 32:
            # 32*4*2 = 256 threads, optimized for wealth dimension
            threads_per_block = (32, 4, 2)
        elif n_W >= 16:
            # 16*8*2 = 256 threads
            threads_per_block = (16, 8, 2)
        else:
            # 8*8*4 = 256 threads (fallback for small wealth grids)
            threads_per_block = (8, 8, 4)
    
    blocks_per_grid_x = max(1, (n_W + threads_per_block[0] - 1) // threads_per_block[0])
    blocks_per_grid_y = max(1, (n_H + threads_per_block[1] - 1) // threads_per_block[1])
    blocks_per_grid_z = max(1, (n_Y + threads_per_block[2] - 1) // threads_per_block[2])
    
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
    
    # Debug output for block configuration (only if verbose)
    if verbose:
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
        d_policy_c, d_policy_a, d_Q_dcsn, d_vlu_dcsn, d_lambda_dcsn,
        d_vlu_cntn, d_w_grid, d_a_grid, d_H_grid, d_h_nxt_ind_array,
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
        # Optimize for small H dimension (typically 7-8)
        if n_H <= 8:
            # 8*32 = 256 threads, optimized for small H
            gradient_threads = (min(n_H, 8), min(n_Y, 32))
        else:
            # 16*16 = 256 threads for larger grids
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
    # Use optimized thread configuration for memory coalescing
    # Note: continuation kernel expects (iw, ih, iy) order
    # Prioritize X dimension (wealth) for coalesced memory access
    if n_W >= 32:
        continuation_threads = (32, 4, 2)  # 256 threads, wealth-optimized
    elif n_W >= 16:
        continuation_threads = (16, 8, 2)  # 256 threads
    else:
        continuation_threads = (8, 8, 4)  # 256 threads (fallback)
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
        d_policy_c, d_Q_dcsn, d_gradient_c, d_H_grid, d_vlu_dcsn, d_lambda_dcsn,
        delta, thorn, alpha, kappa, iota, compute_lambda,
        n_W, n_H, n_Y
    )
    
    # --- 5. Copy Results Back to CPU ---
    policy_c = d_policy_c.copy_to_host()
    policy_a = d_policy_a.copy_to_host()  # Now we actually compute this on GPU
    Q_dcsn = d_Q_dcsn.copy_to_host()
    vlu_dcsn = d_vlu_dcsn.copy_to_host()
    lambda_dcsn = d_lambda_dcsn.copy_to_host()

    return policy_c, policy_a, Q_dcsn, vlu_dcsn, lambda_dcsn


def warmup_gpu_kernels():
    """
    Warm up GPU kernels by running them on tiny grids.
    This forces CUDA JIT compilation before actual runs.
    """
    print("[GPU] Warming up kernels...")
    
    # Create tiny test arrays
    n_W, n_H, n_Y = 4, 2, 2
    
    # Allocate tiny arrays on GPU
    d_policy_c = cuda.device_array((n_W, n_H, n_Y), dtype=np.float64)
    d_policy_a = cuda.device_array_like(d_policy_c)
    d_Q_dcsn = cuda.device_array_like(d_policy_c)
    d_vlu_dcsn = cuda.device_array_like(d_policy_c)
    d_lambda_dcsn = cuda.device_array_like(d_policy_c)
    d_vlu_cntn = cuda.device_array_like(d_policy_c)
    d_gradient_c = cuda.device_array_like(d_policy_c)
    
    d_w_grid = cuda.to_device(np.linspace(0.1, 1.0, n_W))
    d_a_grid = cuda.to_device(np.linspace(0.0, 0.8, n_W))
    d_H_grid = cuda.to_device(np.linspace(0.5, 1.0, n_H))
    d_h_nxt_ind = cuda.to_device(np.arange(n_H))
    
    # Warmup VFI kernel
    threads = (2, 2, 2)
    blocks = (2, 1, 1)
    vfi_gpu_kernel[blocks, threads](
        d_policy_c, d_policy_a, d_Q_dcsn, d_vlu_dcsn, d_lambda_dcsn,
        d_vlu_cntn, d_w_grid, d_a_grid, d_H_grid, d_h_nxt_ind,
        0.95, 0.96, 1.0, 1.0, 10,
        0.7, 0.2, 0.1,
        n_W, n_H, n_Y
    )
    
    # Warmup gradient kernel
    gradient_threads = (2, 2)
    gradient_blocks = (1, 1)
    calculate_gradient_gpu_kernel[gradient_blocks, gradient_threads](
        d_policy_c, d_w_grid, d_gradient_c, 1.0, 0.9,
        n_W, n_H, n_Y
    )
    
    # Warmup continuation kernel
    continuation_threads = (2, 2, 2)
    continuation_blocks = (2, 1, 1)
    calculate_continuation_values_gpu_kernel[continuation_blocks, continuation_threads](
        d_policy_c, d_Q_dcsn, d_gradient_c, d_H_grid, d_vlu_dcsn, d_lambda_dcsn,
        0.96, 1.0, 0.7, 0.2, 0.1, False,
        n_W, n_H, n_Y
    )
    
    # Force synchronization to ensure compilation is complete
    cuda.synchronize()
    print("[GPU] Kernel warmup complete")

