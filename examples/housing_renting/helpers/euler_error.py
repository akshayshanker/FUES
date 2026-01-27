import numpy as np
from numba import njit

# Conditional CUDA import
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    cuda = None
    CUDA_AVAILABLE = False


def _get_model_setting(model, key: str, default=None):
    """Robustly fetch a setting from a DynX model (settings_dict/settings/getattr)."""
    if model is None:
        return default

    settings_dict = getattr(model, "settings_dict", None)
    if isinstance(settings_dict, dict):
        return settings_dict.get(key, default)

    settings = getattr(model, "settings", None)
    if isinstance(settings, dict):
        return settings.get(key, default)
    if hasattr(settings, "get"):
        try:
            return settings.get(key, default)
        except Exception:
            pass

    return getattr(settings, key, default)


def _get_bool_setting(model, key: str, default: bool = False) -> bool:
    """Like `_get_model_setting`, but coerces to bool safely."""
    val = _get_model_setting(model, key, default)
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, (int, np.integer)):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off", ""):
            return False
    return bool(val)

# Import non-GPU functions unconditionally
from dc_smm.models.housing_renting.horses_common import interp_as

# Import tax functions for capital income tax adjustment
from examples.housing_renting.helpers.asset_tax import (
    parse_tax_table, total_tax_scalar, marginal_tax_scalar, snap_brackets_to_grid
)

# Import GPU functions only if CUDA is available
if CUDA_AVAILABLE:
    from dc_smm.models.housing_renting.horses_common import (
        interp_gpu, uc_owner_gpu, uc_renter_gpu, inv_uc_owner_gpu
    )

# GPU kernel only defined when CUDA is available
if CUDA_AVAILABLE and cuda is not None:
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
                    # TODO: Make W_MAX_FILTER configurable (hardwired to match CPU version)
                    if w_now >= 35.0:
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
    # Fall back to CPU if CUDA is not available
    if not CUDA_AVAILABLE:
        if debug:
            print("[GPU Euler] CUDA not available, falling back to CPU")
        return calculate_euler_error_cpu(model, debug=debug)

    # Handle infinite horizon (single period) vs finite horizon (multiple periods)
    n_periods = len(model.periods_list)
    if n_periods == 1:
        # Infinite horizon: stationary solution, use same period for both
        p0, p1 = model.get_period(0), model.get_period(0)
    else:
        # Finite horizon: use periods 0 and 1
        p0, p1 = model.get_period(0), model.get_period(1)

    ownc_now = p0.get_stage("OWNC")
    ownc = p1.get_stage("OWNC")
    tenu = p1.get_stage("TENU")
    ownh = p1.get_stage("OWNH")

    owner_only = _get_bool_setting(getattr(tenu, "model", None), "owner_only", default=False)

    if not owner_only:
        rnth = p1.get_stage("RNTH")
        rntc = p1.get_stage("RNTC")
    else:
        rnth = None
        rntc = None
    
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

    z_vals = tenu.dcsn_to_arvl.model.num.shocks.income_shock.process.values
    H_grid = ownc.dcsn.grid.H_nxt
    tenure_a_grid = tenu.dcsn.grid.a

    # Tenure policy may be missing in some owner-only experiments; default to "always own".
    try:
        tenure_pol = tenu.dcsn.sol.policy["tenure"]
    except Exception:
        tenure_pol = np.ones((len(tenure_a_grid), len(H_grid), len(z_vals)), dtype=np.float64)

    if owner_only:
        # Dummy renter-side arrays (never used when tenure == 1 everywhere).
        S_pol = np.zeros((2, len(z_vals)), dtype=np.float64)
        c_renter_n = np.zeros((2, 2, len(z_vals)), dtype=np.float64)
        renter_a_grid = np.array([0.0, 1.0], dtype=np.float64)
        S_grid = np.array([0.0, 1.0], dtype=np.float64)
        w_dcsn_r = np.array([0.0, 1.0], dtype=np.float64)
    else:
        S_pol = rnth.dcsn.sol.policy["S"]
        c_renter_n = rntc.dcsn.sol.policy["c"]
        renter_a_grid = rnth.dcsn.grid.w
        S_grid = rnth.cntn.grid.S
        w_dcsn_r = rntc.dcsn.grid.w

    data = {
        'z_vals': z_vals,
        'H_grid': H_grid,
        'w_dcsn_now': ownc_now.dcsn.grid.w,
        'c_now': ownc_now.dcsn.sol.policy["c"],
        'tenure_pol': tenure_pol,
        'H_pol': ownh.dcsn.sol.policy["H"],
        'S_pol': S_pol,
        'c_owner_n': ownc.dcsn.sol.policy["c"],
        'c_renter_n': c_renter_n,
        'tenure_a_grid': tenure_a_grid,
        'owner_a_grid': ownh.dcsn.grid.a,
        'renter_a_grid': renter_a_grid,
        'H_nxt_grid': ownh.cntn.grid.H_nxt,
        'S_grid': S_grid,
        'w_dcsn_o': ownc.dcsn.grid.w,
        'w_dcsn_r': w_dcsn_r,
        'Pi': tenu.dcsn_to_arvl.model.num.shocks.income_shock.transition_matrix
    }
    par = ownc.model.param
    Pr_val = getattr(par, "Pr", None)
    if Pr_val is None and (not owner_only) and rnth is not None:
        Pr_val = rnth.model.param.Pr
    if Pr_val is None:
        Pr_val = 0.0
    data.update({
        'beta': par.beta, 
        'R': 1 + par.r, 
        'Pr': Pr_val,
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
    H_nxt_grid, S_grid, w_dcsn_o, w_dcsn_r, Pi, beta, r, Pr, tau_phi,
    uc_owner, uc_rent, uc_inv,
    use_taxes, a0_arr, a1_arr, B_arr, tau_arr,
    a_grid_min, a_grid_max,
    c_min=1e-12, a_min=1e-6
):
    """
    This is the core numerical kernel, compiled with Numba for high performance (CPU fallback).

    Now supports capital income taxes when use_taxes=True.
    Tax adjustment:
    - Budget constraint: w_{t+1} = (1+r)*a - T(a) + y - housing_costs
    - Euler FOC: uses R_net = (1+r) - tau_a(a) instead of R = 1+r
    """
    logs = []

    for iy, y_now in enumerate(z_vals):
        for ih, H_now in enumerate(H_grid):
            for iw, w_now in enumerate(w_dcsn_now):

                c0 = c_now[iw, ih, iy]
                if c0 <= c_min:
                    continue

                a_next = w_now - c0

                # TODO: Make W_MAX_FILTER a proper config setting and/or fix VFI grid extrapolation
                # Hardwired filter: only include wealth values < 35 to avoid extrapolation issues
                W_MAX_FILTER = 35.0
                if w_now < 0.1:
                    continue
                if w_now >= W_MAX_FILTER:
                    continue

                # Clip a_next to asset grid bounds to avoid extrapolation
                a_next_clipped = min(max(a_next, a_grid_min), a_grid_max)

                # Compute tax on a_next if taxes are enabled
                if use_taxes:
                    T_a_next = total_tax_scalar(a_next_clipped, a0_arr, a1_arr, B_arr, tau_arr)
                    tau_a_next = marginal_tax_scalar(a_next_clipped, a0_arr, a1_arr, tau_arr)
                else:
                    T_a_next = 0.0
                    tau_a_next = 0.0

                # Net-of-tax return for Euler FOC
                R_net = (1.0 + r) - tau_a_next

                # Gross wealth from assets (used in budget constraint)
                # w_assets = (1+r)*a_next - T(a_next)
                w_assets = (1.0 + r) * a_next - T_a_next

                E_lam = 0.0

                for jy, y_next in enumerate(z_vals):
                    τ_arr_interp = interp_as(tenure_a_grid, tenure_pol[:, ih, jy], np.array([a_next]))
                    τ = int(τ_arr_interp[0])

                    if τ == 1:    # ----- owner branch -------------------
                        h_idx_arr = interp_as(owner_a_grid, H_pol[:, ih, jy], np.array([a_next]))
                        h_idx = int(h_idx_arr[0])
                        H1 = H_nxt_grid[h_idx]

                        w_dcsn_vals = (w_assets + y_next - H1 + H_now - (tau_phi*H1 if H1 != H_now else 0.0))
                        c1_arr = interp_as(w_dcsn_o, c_owner_n[:, h_idx, jy], np.array([w_dcsn_vals]))
                        c1 = c1_arr[0]
                        # Call the jitted utility function
                        lam_next = uc_owner(c1, H1)

                    else:         # ----- renter branch ------------------
                        s_idx_arr = interp_as(renter_a_grid, S_pol[:, jy], np.array([w_assets + y_next]))
                        s_idx = int(s_idx_arr[0])
                        S1 = S_grid[s_idx]

                        w_dcsn_vals = (w_assets + y_next + H_now - Pr * S1)
                        c1_arr = interp_as(w_dcsn_r, c_renter_n[:, s_idx, jy], np.array([w_dcsn_vals]))
                        c1 = c1_arr[0]
                        # Call the jitted utility function
                        lam_next = uc_rent(c1, S1)

                    E_lam += Pi[iy, jy] * lam_next

                # Call the jitted inverse utility function
                # Euler FOC: u'(c*) = beta * R_net * E[u'(c_{t+1})]
                c_star = uc_inv(beta * R_net * E_lam, H_now)
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

    # Handle infinite horizon (single period) vs finite horizon (multiple periods)
    n_periods = len(model.periods_list)
    if n_periods == 1:
        # Infinite horizon: stationary solution, use same period for both
        p0, p1 = model.get_period(0), model.get_period(0)
    else:
        # Finite horizon: use periods 0 and 1
        p0, p1 = model.get_period(0), model.get_period(1)

    ownc_now = p0.get_stage("OWNC")
    ownc = p1.get_stage("OWNC")
    tenu = p1.get_stage("TENU")
    ownh = p1.get_stage("OWNH")

    owner_only = _get_bool_setting(getattr(tenu, "model", None), "owner_only", default=False)

    # ------------- grids & policies ---------------------------------------
    w_dcsn_now = ownc_now.dcsn.grid.w
    w_dcsn_o = ownc.dcsn.grid.w
    H_grid = ownc.dcsn.grid.H_nxt
    z_vals = tenu.dcsn_to_arvl.model.num.shocks.income_shock.process.values
    tenure_a_grid = tenu.dcsn.grid.a
    owner_a_grid = ownh.dcsn.grid.a
    H_nxt_grid = ownh.cntn.grid.H_nxt
    Pi = tenu.dcsn_to_arvl.model.num.shocks.income_shock.transition_matrix

    c_now = ownc_now.dcsn.sol.policy["c"]
    c_owner_n = ownc.dcsn.sol.policy["c"]
    H_pol = ownh.dcsn.sol.policy["H"]

    # Tenure policy may be missing in some owner-only experiments; default to "always own".
    try:
        tenure_pol = tenu.dcsn.sol.policy["tenure"]
    except Exception:
        tenure_pol = np.ones((len(tenure_a_grid), len(H_grid), len(z_vals)), dtype=np.float64)

    if owner_only:
        # Dummy renter-side arrays (never used when tenure == 1 everywhere).
        w_dcsn_r = np.array([0.0, 1.0], dtype=np.float64)
        S_grid = np.array([0.0, 1.0], dtype=np.float64)
        renter_a_grid = np.array([0.0, 1.0], dtype=np.float64)
        S_pol = np.zeros((2, len(z_vals)), dtype=np.float64)
        c_renter_n = np.zeros((2, 2, len(z_vals)), dtype=np.float64)
    else:
        rnth = p1.get_stage("RNTH")
        rntc = p1.get_stage("RNTC")
        w_dcsn_r = rntc.dcsn.grid.w
        S_grid = rnth.cntn.grid.S
        renter_a_grid = rnth.dcsn.grid.w
        S_pol = rnth.dcsn.sol.policy["S"]
        c_renter_n = rntc.dcsn.sol.policy["c"]
    
    # ------------- primitives --------------------------------------------
    par = ownc.model.param
    par_dict_for_njit_builder = {"alpha": par.alpha, "kappa": par.kappa, "iota": par.iota}
    beta, r, tau_phi = par.beta, par.r, par.phi
    Pr = getattr(par, "Pr", None)
    if Pr is None and (not owner_only):
        try:
            Pr = rnth.model.param.Pr
        except Exception:
            Pr = None
    if Pr is None:
        Pr = 0.0
    tau_phi_R = getattr(par, "phi_R", 0.0)

    # ------------- capital income tax -------------------------------------
    # Extract tax settings from model (same pattern as horses_c.py lines 393-428)
    settings = ownc.model.settings_dict if hasattr(ownc.model, 'settings_dict') else {}
    use_taxes = settings.get('use_taxes', False)

    if use_taxes:
        tax_table = settings.get('tax_table', None)
        if tax_table is not None:
            # Snap bracket boundaries to grid points (same as solver)
            # This ensures Euler error uses same bracket boundaries as the solver
            if not tax_table.get('_snapped_to_grid', False):
                tax_table = snap_brackets_to_grid(tax_table, owner_a_grid)
                tax_table['_snapped_to_grid'] = True
                if debug:
                    print(f"  [TAX] Snapped {len(tax_table.get('brackets', []))} bracket boundaries to asset grid")

            a0_arr, a1_arr, B_arr, tau_arr = parse_tax_table(tax_table)
            if debug:
                print(f"  Taxes ENABLED: {len(a0_arr)} brackets")
        else:
            use_taxes = False
            a0_arr = np.zeros(0, dtype=np.float64)
            a1_arr = np.zeros(0, dtype=np.float64)
            B_arr = np.zeros(0, dtype=np.float64)
            tau_arr = np.zeros(0, dtype=np.float64)
    else:
        a0_arr = np.zeros(0, dtype=np.float64)
        a1_arr = np.zeros(0, dtype=np.float64)
        B_arr = np.zeros(0, dtype=np.float64)
        tau_arr = np.zeros(0, dtype=np.float64)
        if debug:
            print(f"  Taxes DISABLED")

    # --- Build JIT-compatible functions from the model config ---
    # 1. Owner's marginal utility
    uc_owner_expr = ownc.model.math["functions"]["owner_marginal_utility"]["expr"]
    uc_owner = build_njit_utility(uc_owner_expr, par_dict_for_njit_builder, arg1_name="c", arg2_name="H_nxt")

    # 2. Renter's marginal utility (unused in owner-only mode)
    if owner_only:
        uc_rent = uc_owner
    else:
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
            # TODO: Make W_MAX_FILTER a proper config setting (see also _calculate_euler_error_jit)
            W_MAX_FILTER = 35.0
            if w_now <= 0.01 or w_now >= W_MAX_FILTER:
                continue

            # Clip a_next to asset grid bounds to avoid extrapolation
            a_grid_min, a_grid_max = owner_a_grid[0], owner_a_grid[-1]
            a_next_clipped = np.clip(a_next, a_grid_min, a_grid_max)

            # Compute tax on a_next if taxes are enabled
            if use_taxes and len(a0_arr) > 0:
                T_a_next = total_tax_scalar(a_next_clipped, a0_arr, a1_arr, B_arr, tau_arr)
                tau_a_next = marginal_tax_scalar(a_next_clipped, a0_arr, a1_arr, tau_arr)
            else:
                T_a_next = 0.0
                tau_a_next = 0.0

            # Net-of-tax return for Euler FOC
            R_net = (1.0 + r) - tau_a_next
            # Gross wealth from assets (used in budget constraint)
            w_assets = (1.0 + r) * a_next - T_a_next

            E_lam = 0.0

            # Expectation over future shocks
            for jy, y_next in enumerate(z_vals):
                τ_arr_interp = interp_as(tenure_a_grid, tenure_pol[:, ih, jy], np.array([a_next]))
                τ = int(τ_arr_interp[0])

                if τ == 1:  # Owner
                    h_idx_arr = interp_as(owner_a_grid, H_pol[:, ih, jy], np.array([a_next]))
                    h_idx = int(h_idx_arr[0])
                    H1 = H_nxt_grid[h_idx]

                    w_dcsn_vals = (w_assets + y_next - H1 + H_now - (tau_phi*H1 if H1 != H_now else 0.0))
                    c1_arr = interp_as(w_dcsn_o, c_owner_n[:, h_idx, jy], np.array([w_dcsn_vals]))
                    c1 = c1_arr[0]
                    lam_next = uc_owner(c1, H1)
                else:  # Renter
                    s_idx_arr = interp_as(renter_a_grid, S_pol[:, jy], np.array([w_assets + y_next]))
                    s_idx = int(s_idx_arr[0])
                    S1 = S_grid[s_idx]

                    w_dcsn_vals = (w_assets + y_next + H_now - Pr * S1)
                    c1_arr = interp_as(w_dcsn_r, c_renter_n[:, s_idx, jy], np.array([w_dcsn_vals]))
                    c1 = c1_arr[0]
                    lam_next = uc_rent(c1, S1)

                E_lam += Pi[iy, jy] * lam_next

            # Euler equation with net-of-tax return
            c_star = uc_inv(beta * R_net * E_lam, H_now)
            logs.append(np.log10(abs((c_star - c0) / c0) + 1e-16))
        
        logs_array = np.array(logs)
        
        if debug:
            print(f"  Computed {len(logs)} valid Euler errors from {sample_size} sampled states")
    else:
        # Small enough to compute all states
        # Get asset grid bounds for clipping
        a_grid_min, a_grid_max = owner_a_grid[0], owner_a_grid[-1]
        logs_array = _calculate_euler_error_jit(
            z_vals, H_grid, w_dcsn_now, c_now, tenure_pol, H_pol, S_pol,
            c_owner_n, c_renter_n, tenure_a_grid, owner_a_grid, renter_a_grid,
            H_nxt_grid, S_grid, w_dcsn_o, w_dcsn_r, Pi, beta, r, Pr, tau_phi,
            uc_owner, uc_rent, uc_inv,
            use_taxes, a0_arr, a1_arr, B_arr, tau_arr,
            a_grid_min, a_grid_max,
            c_min=1e-12, a_min=a_min_threshold
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
        'r': 0.05,
        'Pr': 1.0,
        'tau_phi': 0.05,
        # Tax arrays (empty = no taxes)
        'use_taxes': False,
        'a0_arr': np.zeros(0, dtype=np.float64),
        'a1_arr': np.zeros(0, dtype=np.float64),
        'B_arr': np.zeros(0, dtype=np.float64),
        'tau_arr': np.zeros(0, dtype=np.float64),
        # Grid bounds for clipping
        'a_grid_min': 0.0,
        'a_grid_max': 10.0,
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
            dummy_data['beta'], dummy_data['r'], dummy_data['Pr'], dummy_data['tau_phi'],
            uc_owner, uc_rent, uc_inv,
            dummy_data['use_taxes'], dummy_data['a0_arr'], dummy_data['a1_arr'],
            dummy_data['B_arr'], dummy_data['tau_arr'],
            dummy_data['a_grid_min'], dummy_data['a_grid_max']
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
