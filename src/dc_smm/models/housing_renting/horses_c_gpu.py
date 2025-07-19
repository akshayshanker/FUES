import os
import gc
import numpy as np
from numba import cuda, njit, prange
import math

from dc_smm.models.housing_renting.horses_common import (
    bellman_obj_gpu, piecewise_gradient,get_u_func
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
):
    """
    Numba CUDA kernel to solve the VFI problem for a single (h, y, w) point.
    """
    h, y, iw = cuda.grid(3)

    n_H, n_Y, n_W = policy_c.shape[1], policy_c.shape[2], policy_c.shape[0]

    if h < n_H and y < n_Y and iw < n_W:
        # --- 1. Setup for this specific grid point ---
        w_val = w_grid[iw]
        H_val = H_grid[h] * thorn
        h_nxt_ind = h_nxt_ind_array[h]

        # V_slice is not created explicitly; we pass indices to bellman_obj_gpu
        
        a_low = a_grid[0]
        a_high = min(w_val - 1e-12, a_grid[-1])
        
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

# Note: Continuation value calculation remains on the CPU as it requires
# a neighborhood operation (piecewise_gradient) that is not straightforward
# to parallelize efficiently on the GPU without more complex strategies.

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
    
    # Allocate output arrays on the GPU
    d_policy_c = cuda.device_array((n_W, n_H, n_Y), dtype=np.float64)
    d_policy_a = cuda.device_array_like(d_policy_c)
    d_Q_dcsn = cuda.device_array_like(d_policy_c)
    d_V_cntn = cuda.device_array_like(d_policy_c)
    d_lambda_cntn = cuda.device_array_like(d_policy_c)

    # --- 3. Configure and Launch Kernel ---
    threads_per_block = (8, 8, 8)
    blocks_per_grid_x = (n_H + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n_Y + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_z = (n_W + threads_per_block[2] - 1) // threads_per_block[2]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
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
    )

    # --- 4. Copy Results Back to CPU ---
    # TODO: CLEANLY REMOVE lambda_cntn ETC.
    policy_c = d_policy_c.copy_to_host()
    #policy_a = d_policy_a.copy_to_host()
    policy_a = np.empty_like(d_policy_c)
    Q_dcsn = d_Q_dcsn.copy_to_host()

    # --- 5. CPU-side Calculation for Continuation Values ---
    V_cntn = np.empty_like(policy_c)
    lambda_cntn = np.empty_like(policy_c)
    
    # Use the new, fast, JIT-compiled function
    _calculate_continuation_values_cpu(
        policy_c, Q_dcsn, H_grid, w_grid, V_cntn, lambda_cntn,
        delta, m_bar, thorn, u_func_cpu
    )

    return policy_c, policy_a, Q_dcsn, V_cntn, lambda_cntn 