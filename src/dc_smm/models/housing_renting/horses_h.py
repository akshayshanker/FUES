"""GPU-accelerated and CPU-based solvers for the housing-renting model's
discrete housing choice.

This module provides the core operator factories and solver loops for the
discrete housing choice of both owners and renters. It is designed with a
dual-path architecture:

1.  **CPU Path:** A `numba.njit`-compiled version that is numerically
    identical to the original implementation.
2.  **GPU Path:** A `numba.cuda.jit`-compiled kernel for significantly
    faster execution on compatible hardware.

The dispatch between CPU and GPU is handled dynamically within the operator
factories based on the model's configuration (`methods['compute'] == 'GPU'`)
and the availability of a CUDA-enabled device. If a GPU is not available or
not requested, the code silently falls back to the CPU implementation.

Module Contents
---------------
F_shocks_dcsn_to_arvl(mover)
    Integrates over income shocks to map decision data to arrival data.
_interp_scalar_cpu(x_grid, y_grid, x)
    Numba-jitted scalar interpolation function for the CPU.
housing_choice_solver_owner_cpu(...)
    Numba-jitted kernel to solve the owner's housing choice problem on the CPU.
housing_choice_solver_renter_cpu(...)
    Numba-jitted kernel to solve the renter's housing choice problem on the CPU.
_interp_scalar_gpu(x_grid, y_grid, x)
    Numba-jitted, CUDA device function for scalar interpolation on the GPU.
housing_choice_solver_owner_gpu(...)
    CUDA kernel to solve the owner's housing choice problem on the GPU.
housing_choice_solver_renter_gpu(...)
    CUDA kernel to solve the renter's housing choice problem on the GPU.
F_h_cntn_to_dcsn_owner(mover, use_mpi, comm)
    Operator factory for the owner's housing choice, with CPU/GPU dispatch.
F_h_cntn_to_dcsn_renter(mover, use_mpi, comm)
    Operator factory for the renter's housing choice, with CPU/GPU dispatch.
"""
import numpy as np
from numba import njit, prange
from dc_smm.models.housing_renting.horses_common import interp_as
from dynx.stagecraft.solmaker import Solution
import os

# --- Conditional CUDA import and GPU Availability Check ---
try:
    from numba import cuda
    GPU_AVAILABLE = cuda.is_available()
except Exception:
    cuda = None
    GPU_AVAILABLE = False


# ==============================================================================
# --- Helper Functions ---
# TODO: Move these to a shared utility module.
# ==============================================================================


def _detect_ncpus():
    """Detect number of CPUs available."""
    try:
        import psutil

        return psutil.cpu_count(logical=True)
    except Exception:
        return int(os.environ.get("PBS_NCPUS", os.cpu_count() or 4))


@njit(inline="always")
def _interp_scalar_cpu(x_grid, y_grid, x):
    """C-style linear interpolation of one point for the CPU."""
    if x <= x_grid[0]:
        return y_grid[0]
    if x >= x_grid[-1]:
        return y_grid[-1]

    lo, hi = 0, len(x_grid) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_grid[mid] <= x:
            lo = mid
        else:
            hi = mid

    x_lo = x_grid[lo]
    inv_dx = 1.0 / (x_grid[hi] - x_lo)
    w_hi = (x - x_lo) * inv_dx
    return (1.0 - w_hi) * y_grid[lo] + w_hi * y_grid[hi]


# GPU device function - only defined when CUDA is available
if GPU_AVAILABLE and cuda is not None:
    @cuda.jit(device=True)
    def _interp_scalar_gpu(x_grid, y_grid, x):
        """C-style linear interpolation of one point for the GPU."""
        if x <= x_grid[0]:
            return y_grid[0]
        if x >= x_grid[-1]:
            return y_grid[-1]

        lo, hi = 0, len(x_grid) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if x_grid[mid] <= x:
                lo = mid
            else:
                hi = mid

        x_lo = x_grid[lo]
        inv_dx = 1.0 / (x_grid[hi] - x_lo)
        w_hi = (x - x_lo) * inv_dx
        return (1.0 - w_hi) * y_grid[lo] + w_hi * y_grid[hi]


def F_shocks_dcsn_to_arvl(mover):
    """Create operator for shock integration in backward step.
    
    This integrates over income shocks for a given state.
    
    Parameters
    ----------
    mover : Mover
        The dcsn_to_arvl mover object with self-contained model
        
    Returns
    -------
    callable
        The operator function that transforms dcsn perch data to arvl perch data
    """
    model = mover.model
    shock_info = model.num.shocks.income_shock
    Pi = shock_info.process.transition_matrix
    use_gpu = model.methods.get("compute") == "GPU" and GPU_AVAILABLE
    
    def operator(perch_data):
        """Transform decision data into arrival data by integrating over income shock."""
        if isinstance(perch_data, Solution):
            vlu_dcsn = perch_data.vlu
            lambda_dcsn = perch_data.lambda_
            
            if use_gpu and vlu_dcsn.size > 1000:  # Use GPU for larger problems
                # GPU implementation
                n_a, n_h, n_j = vlu_dcsn.shape
                n_i = Pi.shape[0]
                
                # Transfer to GPU
                d_vlu_dcsn = cuda.to_device(vlu_dcsn)
                d_Pi = cuda.to_device(Pi)
                d_vlu_arvl = cuda.device_array((n_a, n_h, n_i), dtype=np.float64)
                
                # Configure kernel - use balanced thread configuration
                if n_a * n_h * n_i < 1000:
                    threads = (min(n_a, 8), min(n_h, 8), min(n_i, 8))
                else:
                    threads = (8, 8, 8)  # 512 threads (safer)
                
                blocks_x = max(1, (n_a + threads[0] - 1) // threads[0])
                blocks_y = max(1, (n_h + threads[1] - 1) // threads[1])
                blocks_z = max(1, (n_i + threads[2] - 1) // threads[2])
                blocks = (blocks_x, blocks_y, blocks_z)
                
                # Launch kernel
                shock_integration_kernel[blocks, threads](
                    d_vlu_dcsn, d_Pi, d_vlu_arvl, n_a, n_h, n_j, n_i
                )
                
                # Get result
                vlu_arvl = d_vlu_arvl.copy_to_host()
                
                # Process lambda if available
                try:
                    if lambda_dcsn.size > 0:
                        d_lambda_dcsn = cuda.to_device(lambda_dcsn)
                        d_lambda_arvl = cuda.device_array((n_a, n_h, n_i), dtype=np.float64)
                        
                        shock_integration_kernel[blocks, threads](
                            d_lambda_dcsn, d_Pi, d_lambda_arvl, n_a, n_h, n_j, n_i
                        )
                        
                        lambda_arvl = d_lambda_arvl.copy_to_host()
                    else:
                        lambda_arvl = np.empty((0,0,0))
                except:
                    lambda_arvl = np.empty((0,0,0))
            else:
                # CPU implementation (original)
                vlu_arvl = np.einsum('ahj,ij->ahi', vlu_dcsn, Pi)
                try:
                    lambda_arvl = np.einsum('ahj,ij->ahi', lambda_dcsn, Pi)
                except:
                    lambda_arvl = np.empty((0,0,0))
            
            sol = Solution()
            sol.vlu = vlu_arvl
            sol.lambda_ = lambda_arvl
            return sol
        else: # Legacy dict support
            vlu_dcsn = perch_data["vlu"]
            lambda_dcsn = perch_data["lambda_"]
            
            if use_gpu and vlu_dcsn.size > 1000:
                # Similar GPU implementation for dict case
                n_a, n_h, n_j = vlu_dcsn.shape
                n_i = Pi.shape[0]
                
                d_vlu_dcsn = cuda.to_device(vlu_dcsn)
                d_Pi = cuda.to_device(Pi)
                d_vlu_arvl = cuda.device_array((n_a, n_h, n_i), dtype=np.float64)
                
                if n_a * n_h * n_i < 1000:
                    threads = (min(n_a, 8), min(n_h, 8), min(n_i, 8))
                else:
                    threads = (8, 8, 8)  # 512 threads (safer)
                
                blocks_x = max(1, (n_a + threads[0] - 1) // threads[0])
                blocks_y = max(1, (n_h + threads[1] - 1) // threads[1])
                blocks_z = max(1, (n_i + threads[2] - 1) // threads[2])
                blocks = (blocks_x, blocks_y, blocks_z)
                
                shock_integration_kernel[blocks, threads](
                    d_vlu_dcsn, d_Pi, d_vlu_arvl, n_a, n_h, n_j, n_i
                )
                
                vlu_arvl = d_vlu_arvl.copy_to_host()
                
                try:
                    if lambda_dcsn.size > 0:
                        d_lambda_dcsn = cuda.to_device(lambda_dcsn)
                        d_lambda_arvl = cuda.device_array((n_a, n_h, n_i), dtype=np.float64)
                        
                        shock_integration_kernel[blocks, threads](
                            d_lambda_dcsn, d_Pi, d_lambda_arvl, n_a, n_h, n_j, n_i
                        )
                        
                        lambda_arvl = d_lambda_arvl.copy_to_host()
                    else:
                        lambda_arvl = np.empty((0,0,0))
                except:
                    lambda_arvl = np.empty((0,0,0))
            else:
                vlu_arvl = np.einsum('ahj,ij->ahi', vlu_dcsn, Pi)
                try:
                    lambda_arvl = np.einsum('ahj,ij->ahi', lambda_dcsn, Pi)
                except:
                    lambda_arvl = np.empty((0,0,0))
                    
            return {"vlu": vlu_arvl, "lambda_": lambda_arvl}
    
    return operator

# ==============================================================================
# --- CPU Jitted Functions ---
# ==============================================================================


@njit(cache=True)
def housing_choice_solver_owner_cpu(resources_liquid_3d, H_grid, H_nxt_grid,
                                    w_grid, Q_cntn, v_cntn, lambda_cntn,
                                    tau, min_wealth):
    """Fully-compiled housing-choice kernel for the CPU (owner version).
    
    Optimized for single-core: deferred interpolation (only interpolate 
    lambda/v when Q improves) and pre-computed transition costs.
    """
    n_a, n_h, n_y = resources_liquid_3d.shape
    n_h_next = H_nxt_grid.size

    best_Q = np.full((n_a, n_h, n_y), -np.inf)
    best_lambda = np.zeros((n_a, n_h, n_y))
    best_v = np.zeros((n_a, n_h, n_y))
    best_idx = np.zeros((n_a, n_h, n_y), dtype=np.int32)

    # Pre-compute transition costs for all (ih, ih_next) pairs
    # This avoids redundant computation in the inner loop
    trans_costs = np.zeros((n_h, n_h_next))
    net_housing = np.zeros((n_h, n_h_next))
    for ih in range(n_h):
        h_now = H_grid[ih]
        for ih_next in range(n_h_next):
            h_next = H_nxt_grid[ih_next]
            moved = (h_now != h_next)
            trans_costs[ih, ih_next] = tau * h_next if moved else 0.0
            net_housing[ih, ih_next] = (h_now - h_next) if moved else 0.0

    for i_y in range(n_y):
        for ia in range(n_a):
            for ih in range(n_h):
                res_now = resources_liquid_3d[ia, ih, i_y]
                best_q_here = -np.inf
                best_i_next = 0

                for ih_next in range(n_h_next):
                    w_dcsn = res_now + net_housing[ih, ih_next] - trans_costs[ih, ih_next]

                    if w_dcsn >= min_wealth:
                        q_here = _interp_scalar_cpu(w_grid, Q_cntn[:, ih_next, i_y], w_dcsn)
                        if q_here > best_q_here:
                            best_q_here = q_here
                            best_i_next = ih_next

                # Only interpolate lambda and v once we know the best choice
                if best_q_here > -np.inf:
                    ih_next_best = best_i_next
                    w_dcsn_best = res_now + net_housing[ih, ih_next_best] - trans_costs[ih, ih_next_best]
                    best_Q[ia, ih, i_y] = best_q_here
                    best_lambda[ia, ih, i_y] = _interp_scalar_cpu(w_grid, lambda_cntn[:, ih_next_best, i_y], w_dcsn_best)
                    best_v[ia, ih, i_y] = _interp_scalar_cpu(w_grid, v_cntn[:, ih_next_best, i_y], w_dcsn_best)
                    best_idx[ia, ih, i_y] = best_i_next

    return best_Q, best_v, best_lambda, best_idx

@njit(cache=True)
def housing_choice_solver_renter_cpu(w_grid, S_grid, y_grid, w_rent_grid, 
                                     q_cntn, vlu_cntn, lambda_cntn, Pr, shock_grid):
    """Jitted function to solve the renter housing choice problem on the CPU.
    
    Optimized for single-core: deferred interpolation (only interpolate 
    lambda/v after finding the best service flow choice).
    """
    n_w, n_y = len(w_grid), len(y_grid)
    n_S = len(S_grid)
    
    vlu_dcsn = np.zeros((n_w, n_y))
    q_dcsn = np.zeros((n_w, n_y))
    lambda_dcsn = np.zeros((n_w, n_y))
    S_policy = np.zeros((n_w, n_y), dtype=np.int32)
    
    # Pre-compute rental costs
    rental_costs = Pr * S_grid
    
    for i_y in range(n_y):
        y_val = shock_grid[i_y]
        for i_w in range(n_w):
            w_dcsn_val = w_grid[i_w]
            
            best_q = -np.inf
            best_S_idx = 0
            
            # First pass: find optimal S by only interpolating q
            for i_S in range(n_S):
                w_cntn_val = w_dcsn_val - rental_costs[i_S] + y_val
                
                if w_cntn_val >= w_rent_grid[0]:
                    maximand = _interp_scalar_cpu(w_rent_grid, q_cntn[:, i_S, i_y], w_cntn_val)
                    if maximand > best_q:
                        best_q = maximand
                        best_S_idx = i_S
            
            # Second pass: interpolate lambda and v only for optimal choice
            if best_q > -np.inf:
                w_cntn_best = w_dcsn_val - rental_costs[best_S_idx] + y_val
                q_dcsn[i_w, i_y] = best_q
                lambda_dcsn[i_w, i_y] = _interp_scalar_cpu(w_rent_grid, lambda_cntn[:, best_S_idx, i_y], w_cntn_best)
                vlu_dcsn[i_w, i_y] = _interp_scalar_cpu(w_rent_grid, vlu_cntn[:, best_S_idx, i_y], w_cntn_best)
                S_policy[i_w, i_y] = best_S_idx
    
    return q_dcsn, vlu_dcsn, lambda_dcsn, S_policy

# ==============================================================================
# --- GPU Jitted Functions (only defined when CUDA is available) ---
# ==============================================================================

if GPU_AVAILABLE and cuda is not None:
    @cuda.jit
    def shock_integration_kernel(vlu_dcsn, Pi, vlu_arvl, n_a, n_h, n_j, n_i):
        """GPU kernel for shock integration: vlu_arvl[a,h,i] = sum_j vlu_dcsn[a,h,j] * Pi[i,j]"""
        a, h, i = cuda.grid(3)
        
        if a < n_a and h < n_h and i < n_i:
            sum_val = 0.0
            for j in range(n_j):
                sum_val += vlu_dcsn[a, h, j] * Pi[i, j]
            vlu_arvl[a, h, i] = sum_val

    @cuda.jit
    def housing_choice_solver_owner_gpu(
        resources_liquid_3d, H_grid, H_nxt_grid, w_grid,
        Q_cntn, v_cntn, lambda_cntn,
        tau, min_wealth,
        best_Q, best_v, best_lambda, best_idx
    ):
        """3D CUDA kernel for the owner housing choice. One thread per (a, h, y) state."""
        ia, ih, i_y = cuda.grid(3)
        n_a, n_h, n_y = resources_liquid_3d.shape

        if ia < n_a and ih < n_h and i_y < n_y:
            res_now, h_now = resources_liquid_3d[ia, ih, i_y], H_grid[ih]
            best_q_here, best_l_here, best_v_here, best_i_next = -np.inf, 0.0, -np.inf, 0

            for ih_next in range(H_nxt_grid.size):
                h_next = H_nxt_grid[ih_next]
                moved = (h_now != h_next)
                trans_fee = tau * h_next if moved else 0.0
                w_dcsn = res_now + moved * (h_now - h_next) - trans_fee

                if w_dcsn >= min_wealth:
                    q_here = _interp_scalar_gpu(w_grid, Q_cntn[:, ih_next, i_y], w_dcsn)
                    if q_here > best_q_here:
                        best_q_here = q_here
                        best_l_here = _interp_scalar_gpu(w_grid, lambda_cntn[:, ih_next, i_y], w_dcsn)
                        best_v_here = _interp_scalar_gpu(w_grid, v_cntn[:, ih_next, i_y], w_dcsn)
                        best_i_next = ih_next

            best_Q[ia, ih, i_y] = best_q_here
            best_lambda[ia, ih, i_y] = best_l_here
            best_v[ia, ih, i_y] = best_v_here
            best_idx[ia, ih, i_y] = best_i_next

    @cuda.jit
    def housing_choice_solver_renter_gpu(w_grid, S_grid, y_grid, w_rent_grid,
                                         q_cntn, vlu_cntn, lambda_cntn, Pr, shock_grid,
                                         q_dcsn, vlu_dcsn, lambda_dcsn, S_policy):
        """CUDA kernel for the renter housing choice solver."""
        i_w, i_y = cuda.grid(2)
        n_w, n_y = len(w_grid), len(y_grid)
        n_S = len(S_grid)

        if i_w < n_w and i_y < n_y:
            y_val, w_dcsn_val = shock_grid[i_y], w_grid[i_w]
            best_q, best_lambda, best_S_idx, best_v = -np.inf, 0.0, 0, -np.inf

            for i_S in range(n_S):
                S_val = S_grid[i_S]
                w_cntn_val = w_dcsn_val - Pr * S_val + y_val

                if w_cntn_val >= w_rent_grid[0]:
                    maximand = _interp_scalar_gpu(w_rent_grid, q_cntn[:, i_S, i_y], w_cntn_val)
                    if maximand > best_q:
                        best_q = maximand
                        best_lambda = _interp_scalar_gpu(w_rent_grid, lambda_cntn[:, i_S, i_y], w_cntn_val)
                        best_v = _interp_scalar_gpu(w_rent_grid, vlu_cntn[:, i_S, i_y], w_cntn_val)
                        best_S_idx = i_S
            
            q_dcsn[i_w, i_y] = best_q
            lambda_dcsn[i_w, i_y] = best_lambda
            S_policy[i_w, i_y] = best_S_idx
            vlu_dcsn[i_w, i_y] = best_v

# ==============================================================================
# --- Main Operator Functions with GPU Dispatch (Efficient Version) ---
# ==============================================================================

def F_h_cntn_to_dcsn_owner(mover, use_mpi=False, comm=None):
    """Operator factory for owner housing choice with efficient CPU/GPU dispatch."""
    model = mover.model
    shock_grid = model.num.shocks.income_shock.process.values
    a_grid, H_grid = model.num.state_space.dcsn.grids.a, model.num.state_space.dcsn.grids.H
    w_grid, H_nxt_grid = model.num.state_space.cntn.grids.w_own, model.num.state_space.cntn.grids.H_nxt
    params = model.param
    tau, r = params.phi, params.r
    
    use_gpu = model.methods.get("compute") == "GPU" and GPU_AVAILABLE
    
    # Pre-compute resources grid ONCE in factory (not every operator call)
    # This avoids repeated meshgrid allocations during solving
    n_a, n_H, n_y = len(a_grid), len(H_grid), len(shock_grid)
    a_mesh, _, y_mesh = np.meshgrid(a_grid, H_grid, shock_grid, indexing='ij')
    resources_liquid_3d = (1 + r) * a_mesh + y_mesh
    del a_mesh, y_mesh  # Free intermediate arrays immediately

    def operator(perch_data):
        vlu_cntn, lambda_cntn, Q_cntn = perch_data.vlu, perch_data.lambda_, perch_data.Q
        # NOTE: vlu_cntn is from OWNC which outputs on w_grid, shape (n_w, n_H_nxt, n_y)
        # Output should be on a_grid, shape (n_a, n_H, n_y)

        if use_gpu:
            # --- EFFICIENT GPU PATH ---
            d_resources = cuda.to_device(resources_liquid_3d)
            d_H_grid, d_H_nxt_grid, d_w_grid = cuda.to_device(H_grid), cuda.to_device(H_nxt_grid), cuda.to_device(w_grid)
            d_Q_cntn, d_v_cntn, d_lam_cntn = cuda.to_device(Q_cntn), cuda.to_device(vlu_cntn), cuda.to_device(lambda_cntn)
            
            # Output arrays must be on a_grid (decision grid), not w_grid (continuation grid)
            output_shape = (n_a, n_H, n_y)
            d_Q_out = cuda.device_array(output_shape, dtype=np.float64)
            d_v_out = cuda.device_array(output_shape, dtype=np.float64)
            d_lam_out = cuda.device_array(output_shape, dtype=np.float64)
            d_idx_out = cuda.device_array(output_shape, dtype=np.int32)
            
            # Optimize thread configuration for GPU resource constraints
            if n_a * n_H * n_y < 1000:  # Small problem
                threads = (min(n_a, 8), min(n_H, 8), min(n_y, 4))
            else:  # Large problem - balance threads vs register usage
                # 8*8*4 = 256 threads (safer for complex kernels)
                threads = (8, 8, 4)
            
            blocks_x = max(1, int(np.ceil(n_a / threads[0])))
            blocks_y = max(1, int(np.ceil(n_H / threads[1])))
            blocks_z = max(1, int(np.ceil(n_y / threads[2])))
            
            blocks = (blocks_x, blocks_y, blocks_z)

            housing_choice_solver_owner_gpu[blocks, threads](
                d_resources, d_H_grid, d_H_nxt_grid, d_w_grid,
                d_Q_cntn, d_v_cntn, d_lam_cntn, tau, w_grid[0],
                d_Q_out, d_v_out, d_lam_out, d_idx_out)
            
            Q_dcsn, vlu_dcsn, lambda_dcsn, H_policy = (d.copy_to_host() for d in [d_Q_out, d_v_out, d_lam_out, d_idx_out])
        else:
            # --- EFFICIENT CPU PATH ---
            Q_dcsn, vlu_dcsn, lambda_dcsn, H_policy = housing_choice_solver_owner_cpu(
                resources_liquid_3d, H_grid, H_nxt_grid, w_grid,
                Q_cntn, vlu_cntn, lambda_cntn, tau, w_grid[0])
            
        sol = Solution()
        sol.Q, sol.vlu, sol.lambda_ = Q_dcsn, vlu_dcsn, lambda_dcsn
        sol.policy["H"] = H_policy.astype(np.float64)
        return sol
    
    return operator

def F_h_cntn_to_dcsn_renter(mover, use_mpi=False, comm=None):
    """Operator factory for renter housing choice with CPU/GPU dispatch."""
    model = mover.model
    shock_grid = model.num.shocks.income_shock.process.values
    
    w_grid, y_grid = model.num.state_space.dcsn.grids.w, model.num.state_space.dcsn.grids.y
    w_rent_grid = model.num.state_space.cntn.grids.w_rent
    S_grid = model.num.state_space.cntn.grids.S
    
    Pr = model.param.Pr
    use_gpu = model.methods.get("compute") == "GPU" and GPU_AVAILABLE

    def operator(perch_data):
        vlu_cntn, lambda_cntn, Q_cntn = perch_data.vlu, perch_data.lambda_, perch_data.Q
        
        if use_gpu:
            # --- GPU Path ---
            d_w_grid, d_S_grid, d_y_grid = cuda.to_device(w_grid), cuda.to_device(S_grid), cuda.to_device(y_grid)
            d_w_rent_grid, d_shock_grid = cuda.to_device(w_rent_grid), cuda.to_device(shock_grid)
            d_q_cntn, d_v_cntn, d_lam_cntn = cuda.to_device(Q_cntn), cuda.to_device(vlu_cntn), cuda.to_device(lambda_cntn)

            n_w, n_y = len(w_grid), len(y_grid)
            d_q_out = cuda.device_array((n_w, n_y), dtype=np.float64)
            d_v_out = cuda.device_array((n_w, n_y), dtype=np.float64)
            d_lam_out = cuda.device_array((n_w, n_y), dtype=np.float64)
            d_S_pol = cuda.device_array((n_w, n_y), dtype=np.int32)

            # Optimize thread configuration for GPU resource constraints
            if n_w * n_y < 100:  # Small problem
                threads = (min(n_w, 16), min(n_y, 16))
            else:  # Large problem - balance threads vs register usage
                # 16*16 = 256 threads (safer for complex kernels)
                threads = (16, 16)
            
            blocks_x = max(1, int(np.ceil(n_w / threads[0])))
            blocks_y = max(1, int(np.ceil(n_y / threads[1])))
            blocks = (blocks_x, blocks_y)
            
            housing_choice_solver_renter_gpu[blocks, threads](
                d_w_grid, d_S_grid, d_y_grid, d_w_rent_grid, 
                d_q_cntn, d_v_cntn, d_lam_cntn, Pr, d_shock_grid,
                d_q_out, d_v_out, d_lam_out, d_S_pol)
            
            Q_dcsn, vlu_dcsn = d_q_out.copy_to_host(), d_v_out.copy_to_host()
            lambda_dcsn, S_policy = d_lam_out.copy_to_host(), d_S_pol.copy_to_host()
        else:
            # --- CPU Path ---
            Q_dcsn, vlu_dcsn, lambda_dcsn, S_policy = housing_choice_solver_renter_cpu(
                w_grid, S_grid, y_grid, w_rent_grid, 
                Q_cntn, vlu_cntn, lambda_cntn, Pr, shock_grid)
        
        sol = Solution()
        sol.Q, sol.vlu, sol.lambda_ = Q_dcsn, vlu_dcsn, lambda_dcsn
        sol.policy["S"] = S_policy.astype(np.float64)
        return sol
    
    return operator 