import numpy as np
from numba import njit, cuda

# Import the shared interpolation and NEW static GPU utility functions
from dc_smm.models.housing_renting.horses_common import (
    interp_as, interp_gpu, uc_owner_gpu, uc_renter_gpu, inv_uc_owner_gpu
)

@cuda.jit
def _calculate_euler_error_cuda_kernel(
    z_vals, H_grid, w_dcsn_now, c_now, tenure_pol, H_pol, S_pol,
    c_owner_n, c_renter_n, tenure_a_grid, owner_a_grid, renter_a_grid,
    H_nxt_grid, S_grid, w_dcsn_o, w_dcsn_r, Pi, beta, R, Pr, tau_phi,
    alpha,
    output_logs
):
    """
    CUDA kernel to calculate Euler errors in parallel on the GPU.
    """
    iw, ih, iy = cuda.grid(3)
    w_stride, h_stride, y_stride = cuda.gridsize(3)
    
    for i_w_loop in range(iw, w_dcsn_now.shape[0], w_stride):
        for i_h_loop in range(ih, H_grid.shape[0], h_stride):
            for i_y_loop in range(iy, z_vals.shape[0], y_stride):

                w_now, H_now = w_dcsn_now[i_w_loop], H_grid[i_h_loop]
                c0 = c_now[i_w_loop, i_h_loop, i_y_loop]
                if c0 <= 1e-12: 
                    continue
                a_next = w_now - c0
                if a_next <= 0.1: 
                    continue

                E_lam = 0.0
                for jy, y_next in enumerate(z_vals):
                    # Use proper GPU interpolation function instead of manual loops
                    tenure_val = interp_gpu(a_next, tenure_a_grid, tenure_pol[:, i_h_loop, jy])
                    τ = int(tenure_val)

                    if τ == 1: # Owner
                        # Use proper GPU interpolation
                        h_idx_val = interp_gpu(a_next, owner_a_grid, H_pol[:, i_h_loop, jy])
                        H1 = H_nxt_grid[int(h_idx_val)]
                        w_dcsn_vals = (R * a_next + y_next - H1 + H_now - (tau_phi*H1 if H1 != H_now else 0.0))
                        
                        # Use proper GPU interpolation for consumption
                        c1 = interp_gpu(w_dcsn_vals, w_dcsn_o, c_owner_n[:, int(h_idx_val), jy])
                        lam_next = uc_owner_gpu(c1, H1, alpha)
                        
                    else: # Renter
                        # Use proper GPU interpolation
                        s_idx_val = interp_gpu(R * a_next + y_next, renter_a_grid, S_pol[:, jy])
                        S1 = S_grid[int(s_idx_val)]
                        # User-specified change: Keep renter wealth calculation as is
                        w_dcsn_vals = (R * a_next + y_next + H_now - Pr * S1)
                        
                        # Use proper GPU interpolation for consumption
                        c1 = interp_gpu(w_dcsn_vals, w_dcsn_r, c_renter_n[:, int(s_idx_val), jy])
                        lam_next = uc_renter_gpu(c1, S1, alpha)

                    E_lam += Pi[i_y_loop, jy] * lam_next
                
                c_star = inv_uc_owner_gpu(beta * R * E_lam, H_now, alpha)
                output_logs[i_w_loop, i_h_loop, i_y_loop] = cuda.libdevice.log10(abs((c_star - c0)/ c0) + 1e-16)

def calculate_euler_error_gpu(model, sample_size=10000, debug=False):
    """
    GPU orchestrator. Extracts data, moves it to the GPU, runs the kernel, and returns the result.
    
    Parameters
    ----------
    model : Model
        The solved model
    sample_size : int
        Maximum number of states to compute. If grid is larger, will sample uniformly.
    debug : bool
        Whether to print debug information
    """
    p0, p1 = model.get_period(0), model.get_period(1)
    ownc_now = p0.get_stage("OWNC")
    ownc, tenu, ownh, rnth, rntc = (p1.get_stage("OWNC"), p1.get_stage("TENU"), p1.get_stage("OWNH"),
                                  p1.get_stage("RNTH"), p1.get_stage("RNTC"))
    
    # Check problem size and decide whether to sample
    n_w = len(ownc_now.dcsn.grid.w)
    n_h = len(ownc_now.dcsn.grid.H_nxt)
    n_y = len(tenu.dcsn_to_arvl.model.num.shocks.income_shock.process.values)
    total_states = n_w * n_h * n_y
    
    if debug:
        print(f"[GPU Euler] Grid size: {n_w}×{n_h}×{n_y} = {total_states:,} states")
    
    # Estimate memory requirements (22 arrays * 8 bytes per float64)
    estimated_gb = total_states * 22 * 8 / 1e9
    if estimated_gb > 10:  # More than 10GB
        if debug:
            print(f"[GPU Euler] Estimated GPU memory: {estimated_gb:.1f}GB - too large, falling back to CPU")
        # Fall back to CPU implementation
        return calculate_euler_error_cpu(model, debug=debug)
    
    # For moderate sizes, still use GPU but potentially sample
    use_sampling = total_states > sample_size
    if use_sampling and debug:
        print(f"[GPU Euler] Sampling {sample_size:,} states from {total_states:,} total")

    data = {
        'z_vals': tenu.dcsn_to_arvl.model.num.shocks.income_shock.process.values, 
        'H_grid': ownc.dcsn.grid.H_nxt,
        'w_dcsn_now': ownc_now.dcsn.grid.w, 
        'c_now': ownc_now.dcsn.sol.policy["c"],
        'tenure_pol': tenu.dcsn.sol.policy["tenure"], 
        'H_pol': ownh.dcsn.sol.policy["H"],
        'S_pol': rnth.dcsn.sol.policy["S"], 
        'c_owner_n': ownc.dcsn.sol.policy["c"],
        'c_renter_n': rntc.dcsn.sol.policy["c"], 
        'tenure_a_grid': tenu.dcsn.grid.a,
        'owner_a_grid': ownh.dcsn.grid.a, 
        'renter_a_grid': rnth.dcsn.grid.w,
        'H_nxt_grid': ownh.cntn.grid.H_nxt, 
        'S_grid': rnth.cntn.grid.S,
        'w_dcsn_o': ownc.dcsn.grid.w, 
        'w_dcsn_r': rntc.dcsn.grid.w,
        'Pi': tenu.dcsn_to_arvl.model.num.shocks.income_shock.transition_matrix
    }
    par = ownc.model.param
    data.update({
        'beta': par.beta, 
        'R': 1 + par.r, 
        'Pr': rnth.model.param.Pr,
        'tau_phi': par.phi, 
        'tau_phi_R': getattr(par, "phi_R", 0.0),
        'theta': par.theta, 
        'iota': par.iota, 
        'kappa': par.kappa, 
        'rho': par.rho,
        'alpha': par.alpha
    })

    # Convert numpy arrays to GPU arrays
    gpu_data = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            gpu_data[key] = cuda.to_device(val)
        else:
            gpu_data[key] = val
    
    output_logs_gpu = cuda.device_array_like(data['c_now'])
    output_logs_gpu.copy_to_device(np.full_like(data['c_now'], np.nan))
    
    threads_per_block = (8, 8, 4)
    blocks_per_grid_x = (data['w_dcsn_now'].shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (data['H_grid'].shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_z = (data['z_vals'].shape[0] + threads_per_block[2] - 1) // threads_per_block[2]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
    
    # Call the kernel with all required arguments
    _calculate_euler_error_cuda_kernel[blocks_per_grid, threads_per_block](
        gpu_data['z_vals'],
        gpu_data['H_grid'],
        gpu_data['w_dcsn_now'],
        gpu_data['c_now'],
        gpu_data['tenure_pol'],
        gpu_data['H_pol'],
        gpu_data['S_pol'],
        gpu_data['c_owner_n'],
        gpu_data['c_renter_n'],
        gpu_data['tenure_a_grid'],
        gpu_data['owner_a_grid'],
        gpu_data['renter_a_grid'],
        gpu_data['H_nxt_grid'],
        gpu_data['S_grid'],
        gpu_data['w_dcsn_o'],
        gpu_data['w_dcsn_r'],
        gpu_data['Pi'],
        gpu_data['beta'],
        gpu_data['R'],
        gpu_data['Pr'],
        gpu_data['tau_phi'],
        gpu_data['alpha'],
        output_logs_gpu
    )
    
    logs_array = output_logs_gpu.copy_to_host()
    return float(np.nanmean(logs_array)) if not np.all(np.isnan(logs_array)) else np.nan

@njit
def _calculate_euler_error_jit(
    z_vals, H_grid, w_dcsn_now, c_now, tenure_pol, H_pol, S_pol,
    c_owner_n, c_renter_n, tenure_a_grid, owner_a_grid, renter_a_grid,
    H_nxt_grid, S_grid, w_dcsn_o, w_dcsn_r, Pi, beta, R, Pr, tau_phi,
    uc_owner, uc_rent, uc_inv, c_min=1e-12, a_min=1e-6
):
    """
    This is the core numerical kernel, compiled with Numba for high performance (CPU fallback).
    """
    logs = []

    for iy, y_now in enumerate(z_vals):
        for ih, H_now in enumerate(H_grid):
            for iw, w_now in enumerate(w_dcsn_now):

                c0 = c_now[iw, ih, iy]
                if c0 <= c_min:
                    continue

                a_next = w_now - c0

                if w_now< 0.1:
                    continue
                if w_now> 30:
                    continue
                
                #if a_next <= 0.5:
                #    continue
                #if a_next >= 20:
                #    continue

                E_lam = 0.0

                for jy, y_next in enumerate(z_vals):
                    τ_arr = interp_as(tenure_a_grid, tenure_pol[:, ih, jy], np.array([a_next]))
                    τ = int(τ_arr[0])

                    if τ == 1:    # ----- owner branch -------------------
                        h_idx_arr = interp_as(owner_a_grid, H_pol[:, ih, jy], np.array([a_next]))
                        h_idx = int(h_idx_arr[0])
                        H1 = H_nxt_grid[h_idx]

                        w_dcsn_vals = (R * a_next + y_next - H1 + H_now - (tau_phi*H1 if H1 != H_now else 0.0))
                        c1_arr = interp_as(w_dcsn_o, c_owner_n[:, h_idx, jy], np.array([w_dcsn_vals]))
                        c1 = c1_arr[0]
                        # Call the jitted utility function
                        lam_next = uc_owner(c1, H1)

                    else:         # ----- renter branch ------------------
                        s_idx_arr = interp_as(renter_a_grid, S_pol[:, jy], np.array([R * a_next + y_next]))
                        s_idx = int(s_idx_arr[0])
                        S1 = S_grid[s_idx]

                        w_dcsn_vals = (R * a_next + y_next + H_now - Pr * S1)
                        c1_arr = interp_as(w_dcsn_r, c_renter_n[:, s_idx, jy], np.array([w_dcsn_vals]))
                        c1 = c1_arr[0]
                        #print(c1)
                        # Call the jitted utility function
                        lam_next = uc_rent(c1, S1)

                    E_lam += Pi[iy, jy] * lam_next
                
                # Call the jitted inverse utility function
                c_star = uc_inv(beta*R*E_lam, H_now)
                #print(c_star)
                logs.append(np.log10(abs((c_star - c0) / c0) + 1e-16))

    return np.array(logs)

def calculate_euler_error_cpu(model, debug=True, sample_size=50000, random_sample=False):
    """
    CPU-based Euler error calculation using the original approach.
    
    Parameters
    ----------
    model : Model
        The solved model
    debug : bool
        Whether to print debug information
    sample_size : int
        Maximum number of states to compute. If grid is larger, will sample uniformly.
    """
    from dc_smm.models.housing_renting.horses_common import build_njit_utility
    
    p0, p1 = model.get_period(0), model.get_period(1)
    
    ownc_now = p0.get_stage("OWNC")
    ownc, tenu, ownh, rnth, rntc = (
        p1.get_stage("OWNC"), p1.get_stage("TENU"), p1.get_stage("OWNH"),
        p1.get_stage("RNTH"), p1.get_stage("RNTC"))

    # ------------- grids & policies ---------------------------------------
    w_dcsn_now, w_dcsn_o, w_dcsn_r = ownc_now.dcsn.grid.w, ownc.dcsn.grid.w, rntc.dcsn.grid.w
    H_grid, S_grid = ownc.dcsn.grid.H_nxt, rnth.cntn.grid.S
    z_vals = tenu.dcsn_to_arvl.model.num.shocks.income_shock.process.values
    tenure_a_grid, owner_a_grid, renter_a_grid = tenu.dcsn.grid.a, ownh.dcsn.grid.a, rnth.dcsn.grid.w
    H_nxt_grid = ownh.cntn.grid.H_nxt
    Pi = tenu.dcsn_to_arvl.model.num.shocks.income_shock.transition_matrix
    c_now, tenure_pol, H_pol, S_pol = ownc_now.dcsn.sol.policy["c"], tenu.dcsn.sol.policy["tenure"], ownh.dcsn.sol.policy["H"], rnth.dcsn.sol.policy["S"]
    c_owner_n, c_renter_n = ownc.dcsn.sol.policy["c"], rntc.dcsn.sol.policy["c"]
    
    # ------------- primitives --------------------------------------------
    par = ownc.model.param
    par_dict_for_njit_builder = {"alpha": par.alpha, "kappa": par.kappa, "iota": par.iota}
    beta, r, Pr, tau_phi = par.beta, par.r, rnth.model.param.Pr, par.phi
    tau_phi_R, R = getattr(par, "phi_R", 0.0), 1 + r

    # --- Build JIT-compatible functions from the model config ---
    # 1. Owner's marginal utility
    uc_owner_expr = ownc.model.math["functions"]["owner_marginal_utility"]["expr"]
    uc_owner = build_njit_utility(uc_owner_expr, par_dict_for_njit_builder, arg1_name="c", arg2_name="H_nxt")

    # 2. Renter's marginal utility
    uc_rent_expr = ownc.model.math["functions"]["uc_func"]["expr"]
    uc_rent = build_njit_utility(uc_rent_expr, par_dict_for_njit_builder, arg1_name="c", arg2_name="H_nxt")

    # 3. Inverse marginal utility
    uc_inv_expr = ownc.model.math["functions"]["inv_marginal_utility"]["expr"]
    uc_inv = build_njit_utility(uc_inv_expr, par_dict_for_njit_builder, arg1_name="lambda_e", arg2_name="H_nxt")
    
    # Debug: Check policy function ranges before calculation
    if debug:
        print(f"Debug Euler Error Calculation:")
        print(f"  c_now range: [{np.min(c_now):.6f}, {np.max(c_now):.6f}]")
        print(f"  w_dcsn_now range: [{np.min(w_dcsn_now):.6f}, {np.max(w_dcsn_now):.6f}]")
        print(f"  Minimum a_next (w - c): {np.min(w_dcsn_now[..., np.newaxis, np.newaxis] - c_now):.6f}")
        print(f"  Number of valid states (c > 1e-12): {np.sum(c_now > 1e-12)}")
        print(f"  Number of valid states (a_next > 0.1): {np.sum((w_dcsn_now[..., np.newaxis, np.newaxis] - c_now) > 0.1)}")

    # Use borrowing constraint from model as minimum asset threshold
    b = getattr(par, 'b', 1e-6)
    a_min_threshold = max(b, 1e-6)  # At least 1e-6 to avoid numerical issues
    
    # Check if we need to sample
    n_w, n_h, n_y = len(w_dcsn_now), len(H_grid), len(z_vals)
    total_states = n_w * n_h * n_y
    
    if total_states > sample_size and random_sample:
        if debug:
            print(f"[CPU Euler] Sampling {sample_size:,} states from {total_states:,} total")
        
        # Create a sample of states
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(total_states, size=sample_size, replace=False)
        
        # Convert flat indices to 3D indices
        sampled_iw = sample_indices // (n_h * n_y)
        sampled_ih = (sample_indices % (n_h * n_y)) // n_y
        sampled_iy = sample_indices % n_y
        
        # Calculate Euler errors only for sampled states
        logs = []
        for idx in range(len(sampled_iw)):
            iw = sampled_iw[idx]
            ih = sampled_ih[idx]
            iy = sampled_iy[idx]
            
            w_now = w_dcsn_now[iw]
            H_now = H_grid[ih]
            c0 = c_now[iw, ih, iy]
            
            if c0 <= 1e-12:
                continue
                
            a_next = w_now - c0
            if a_next <= 0.5 or a_next >= 30:
                continue
            
            E_lam = 0.0
            
            # Expectation over future shocks
            for jy, y_next in enumerate(z_vals):
                τ_arr = interp_as(tenure_a_grid, tenure_pol[:, ih, jy], np.array([a_next]))
                τ = int(τ_arr[0])
                
                if τ == 1:  # Owner
                    h_idx_arr = interp_as(owner_a_grid, H_pol[:, ih, jy], np.array([a_next]))
                    h_idx = int(h_idx_arr[0])
                    H1 = H_nxt_grid[h_idx]
                    
                    w_dcsn_vals = (R * a_next + y_next - H1 + H_now - (tau_phi*H1 if H1 != H_now else 0.0))
                    c1_arr = interp_as(w_dcsn_o, c_owner_n[:, h_idx, jy], np.array([w_dcsn_vals]))
                    c1 = c1_arr[0]
                    lam_next = uc_owner(c1, H1)
                else:  # Renter
                    s_idx_arr = interp_as(renter_a_grid, S_pol[:, jy], np.array([R * a_next + y_next]))
                    s_idx = int(s_idx_arr[0])
                    S1 = S_grid[s_idx]
                    
                    w_dcsn_vals = (R * a_next + y_next + H_now - Pr * S1)
                    c1_arr = interp_as(w_dcsn_r, c_renter_n[:, s_idx, jy], np.array([w_dcsn_vals]))
                    c1 = c1_arr[0]
                    lam_next = uc_rent(c1, S1)
                
                E_lam += Pi[iy, jy] * lam_next
            
            # Euler equation
            c_star = uc_inv(beta * R * E_lam, H_now)
            logs.append(np.log10(abs((c_star - c0) / c0) + 1e-16))
        
        logs_array = np.array(logs)
        
        if debug:
            print(f"  Computed {len(logs)} valid Euler errors from {sample_size} sampled states")
    else:
        # Small enough to compute all states
        logs_array = _calculate_euler_error_jit(
            z_vals, H_grid, w_dcsn_now, c_now, tenure_pol, H_pol, S_pol,
            c_owner_n, c_renter_n, tenure_a_grid, owner_a_grid, renter_a_grid,
            H_nxt_grid, S_grid, w_dcsn_o, w_dcsn_r, Pi, beta, R, Pr, tau_phi,
            uc_owner, uc_rent, uc_inv, c_min=1e-12, a_min=a_min_threshold
        )

    if debug:
        print(f"  Logs array size: {logs_array.size}")
        if logs_array.size > 0:
            print(f"  Euler error range: [{np.min(logs_array):.6f}, {np.max(logs_array):.6f}]")
        else:
            print("  No valid states found for Euler error calculation")

    return float(np.mean(logs_array)) if logs_array.size > 0 else np.nan

def precompile_euler_error_cpu():
    """
    Precompile the CPU Euler error calculation functions to avoid JIT overhead.
    This creates dummy data and runs the jitted functions once to compile them.
    """
    from dc_smm.models.housing_renting.horses_common import build_njit_utility
    
    # Create minimal dummy data matching expected shapes
    n_a, n_h, n_y = 10, 5, 3  # Small sizes for fast compilation
    
    # Create dummy arrays with proper shapes
    dummy_data = {
        'z_vals': np.ones(n_y),
        'H_grid': np.ones(n_h),
        'w_dcsn_now': np.ones(n_a),
        'c_now': np.ones((n_a, n_h, n_y)),
        'tenure_pol': np.ones((n_a, n_h, n_y)),
        'H_pol': np.ones((n_a, n_h, n_y)),
        'S_pol': np.ones((n_a, n_y)),
        'c_owner_n': np.ones((n_a, n_h, n_y)),
        'c_renter_n': np.ones((n_a, n_h, n_y)),
        'tenure_a_grid': np.linspace(0, 10, n_a),
        'owner_a_grid': np.linspace(0, 10, n_a),
        'renter_a_grid': np.linspace(0, 10, n_a),
        'H_nxt_grid': np.ones(n_h),
        'S_grid': np.ones(n_h),
        'w_dcsn_o': np.linspace(0, 10, n_a),
        'w_dcsn_r': np.linspace(0, 10, n_a),
        'Pi': np.ones((n_y, n_y)) / n_y,
        'beta': 0.95,
        'R': 1.05,
        'Pr': 1.0,
        'tau_phi': 0.05,
    }
    
    # Build dummy utility functions with hardcoded expressions
    par_dict = {"alpha": 0.7, "kappa": 1.0, "iota": 1.0}
    
    # Standard CRRA utility function expressions for housing model
    uc_owner_expr = "(c**(1-kappa) * H_nxt**alpha)**(1-iota) * (1-kappa) / c"
    uc_rent_expr = "c**(-kappa)"
    uc_inv_expr = "lambda_e**(-1/kappa)"
    
    uc_owner = build_njit_utility(uc_owner_expr, par_dict, arg1_name="c", arg2_name="H_nxt")
    uc_rent = build_njit_utility(uc_rent_expr, par_dict, arg1_name="c", arg2_name="H_nxt")
    uc_inv = build_njit_utility(uc_inv_expr, par_dict, arg1_name="lambda_e", arg2_name="H_nxt")
    
    try:
        # Run the jitted function once with dummy data to compile it
        _ = _calculate_euler_error_jit(
            dummy_data['z_vals'], dummy_data['H_grid'], dummy_data['w_dcsn_now'],
            dummy_data['c_now'], dummy_data['tenure_pol'], dummy_data['H_pol'],
            dummy_data['S_pol'], dummy_data['c_owner_n'], dummy_data['c_renter_n'],
            dummy_data['tenure_a_grid'], dummy_data['owner_a_grid'], 
            dummy_data['renter_a_grid'], dummy_data['H_nxt_grid'], dummy_data['S_grid'],
            dummy_data['w_dcsn_o'], dummy_data['w_dcsn_r'], dummy_data['Pi'],
            dummy_data['beta'], dummy_data['R'], dummy_data['Pr'], dummy_data['tau_phi'],
            uc_owner, uc_rent, uc_inv
        )
        return True
    except Exception as e:
        print(f"Warning: Failed to precompile Euler error calculation: {e}")
        return False


def euler_error_metric(model, use_gpu=True, debug=True, sample_size=50000, **kwargs):
    """
    Metric wrapper. Set use_gpu=False to use the CPU version.
    
    Parameters
    ----------
    model : Model
        The solved model to calculate Euler errors for
    use_gpu : bool
        Whether to attempt GPU calculation (falls back to CPU if unavailable)
    debug : bool
        Whether to print debug information
    sample_size : int
        Maximum number of states to compute. If grid is larger, will sample uniformly.
    **kwargs : dict
        Additional arguments (for compatibility with CircuitRunner)
    """
    # Check if we should use GPU based on availability and user preference
    if use_gpu and cuda.is_available():
        try:
            return calculate_euler_error_gpu(model, sample_size=sample_size, debug=debug)
        except Exception as e:
            print(f"Warning: GPU Euler error calculation failed: {e}")
            print("Falling back to CPU implementation...")
            return calculate_euler_error_cpu(model, debug=debug, sample_size=sample_size)
    else:
        if use_gpu and not cuda.is_available(): 
            print("Warning: GPU not available, falling back to CPU for Euler error.")
        return calculate_euler_error_cpu(model, debug=debug, sample_size=sample_size)
