
import numpy as np
from numba import njit

# Import the shared utility builder and interpolation functions
from dc_smm.models.housing_renting.horses_common import build_njit_utility, interp_as

@njit
def _calculate_euler_error_jit(
    z_vals, H_grid, w_dcsn_now, c_now, tenure_pol, H_pol, S_pol,
    c_owner_n, c_renter_n, tenure_a_grid, owner_a_grid, renter_a_grid,
    H_nxt_grid, S_grid, w_dcsn_o, w_dcsn_r, Pi, beta, R, Pr, tau_phi, tau_phi_R,
    uc_owner, uc_rent, uc_inv
):
    """
    This is the core numerical kernel, compiled with Numba for high performance.
    """
    logs = []

    for iy, y_now in enumerate(z_vals):
        for ih, H_now in enumerate(H_grid):
            for iw, w_now in enumerate(w_dcsn_now):

                c0 = c_now[iw, ih, iy]
                if c0 <= 1e-12:
                    continue

                a_next = w_now - c0
                if a_next <= 0.1:
                    continue

                E_lam = 0.0

                for jy, y_next in enumerate(z_vals):
                    τ_arr = interp_as(tenure_a_grid, tenure_pol[:, ih, jy], np.array([a_next]))
                    τ = int(τ_arr[0])

                    if τ == 1:    # ----- owner branch -------------------
                        h_idx_arr = interp_as(owner_a_grid, H_pol[:, ih, jy], np.array([a_next]))
                        h_idx = int(h_idx_arr[0])
                        H1 = H_nxt_grid[h_idx]

                        w_dcsn_vals = (R * a_next + y_next - H1 + H_now - (tau_phi if H1 != H_now else 0.0))
                        c1_arr = interp_as(w_dcsn_o, c_owner_n[:, h_idx, jy], np.array([w_dcsn_vals]))
                        c1 = c1_arr[0]
                        # Call the jitted utility function
                        lam_next = uc_owner(c1, H1)

                    else:         # ----- renter branch ------------------
                        s_idx_arr = interp_as(renter_a_grid, S_pol[:, jy], np.array([R * a_next + y_next]))
                        s_idx = int(s_idx_arr[0])
                        S1 = S_grid[s_idx]

                        w_dcsn_vals = (R * a_next + y_next + H_now + Pr * S1 - tau_phi_R)
                        c1_arr = interp_as(w_dcsn_r, c_renter_n[:, s_idx, jy], np.array([w_dcsn_vals]))
                        c1 = c1_arr[0]
                        # Call the jitted utility function
                        lam_next = uc_rent(c1, S1)

                    E_lam += Pi[iy, jy] * lam_next
                
                # Call the jitted inverse utility function
                c_star = uc_inv(E_lam, H_now)
                logs.append(np.log10(abs((c_star - c0) / c0) + 1e-16))

    return np.array(logs)

def calculate_euler_error(model):
    """
    Euler-error orchestrator.
    This function extracts data from the model and calls the fast JIT-compiled kernel.
    """
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
    beta, r, Pr, tau_phi = par.beta, par.r, rnth.model.param.Pr, par.phi
    tau_phi_R, R = getattr(par, "phi_R", 0.0), 1 + r

    # --- CORRECTED CODE: Build JIT-compatible functions from the model config ---
    # Instead of passing the methods directly, we find their string expressions
    # and parameters in the model config and compile them into new, JIT-safe functions.
    
    # 1. Owner's marginal utility
    uc_owner_expr = ownc.model.stage.math.functions.owner_marginal_utility.expr
    uc_owner = build_njit_utility(uc_owner_expr, par, h_placeholder="H_nxt")

    # 2. Renter's marginal utility (assuming it has a similar structure)
    uc_rent_expr = rntc.model.stage.math.functions.uc_func.expr
    uc_rent = build_njit_utility(uc_rent_expr, par, h_placeholder="H_nxt")

    # 3. Inverse marginal utility
    uc_inv_expr = ownc.model.stage.math.functions.inv_marginal_utility.expr
    uc_inv = build_njit_utility(uc_inv_expr, par, h_placeholder="H_nxt")
    
    # Call the JIT-compiled kernel with the newly compiled functions
    logs_array = _calculate_euler_error_jit(
        z_vals, H_grid, w_dcsn_now, c_now, tenure_pol, H_pol, S_pol,
        c_owner_n, c_renter_n, tenure_a_grid, owner_a_grid, renter_a_grid,
        H_nxt_grid, S_grid, w_dcsn_o, w_dcsn_r, Pi, beta, R, Pr, tau_phi, tau_phi_R,
        uc_owner, uc_rent, uc_inv
    )

    return float(np.mean(logs_array)) if logs_array.size > 0 else np.nan

def euler_error_metric(model):
    """Metric wrapper for the Euler error calculation."""
    return calculate_euler_error(model)
