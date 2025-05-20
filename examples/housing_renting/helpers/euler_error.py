import numpy as np

def calculate_euler_error(model):
    """
    Euler-error with branch-specific cash-on-hand formulas.
    """

    p0, p1 = model.get_period(0), model.get_period(1)
    
    ownc_now = p0.get_stage("OWNC")

    ownc, tenu, ownh, rnth, rntc = (
        p1.get_stage("OWNC"),
        p1.get_stage("TENU"),
        p1.get_stage("OWNH"),
        p1.get_stage("RNTH"),
        p1.get_stage("RNTC"))

    # ------------- grids --------------------------------------------------
    w_dcsn_now = ownc_now.dcsn.grid.w

    w_dcsn_o   = ownc.dcsn.grid.w
    w_dcsn_r   = rntc.dcsn.grid.w


    H_grid   = ownc.dcsn.grid.H_nxt
    S_grid   = rnth.cntn.grid.S

    y_grid   = ownc.dcsn.grid.y
    
    a_grid_O = ownh.dcsn.grid.a
    a_grid_R = rnth.dcsn.grid.w
    
    H_nxt_grid = ownh.cntn.grid.H_nxt
    

    Pi = tenu.dcsn_to_arvl.model.num.shocks.income_shock.transition_matrix
    z_vals = tenu.dcsn_to_arvl.model.num.shocks.income_shock.process.values

    # ------------- policies ----------------------------------------------
    c_now     = ownc_now.dcsn.sol["policy"]
    tenure_pol= tenu.dcsn.sol["tenure_policy"]
    H_pol     = ownh.dcsn.sol["H_policy"]
    S_pol     = rnth.dcsn.sol["S_policy"]
    c_owner_n = ownc.dcsn.sol["policy"]          # (a, H_nxt, y)
    c_renter_n= rntc.dcsn.sol["policy"]          # (a, y)

    # ------------- primitives --------------------------------------------
    par     = ownc.model.param
    beta, r = par.beta, par.r
    Pr      = rnth.model.param.Pr
    tau_phi = par.phi                # 🔄 owner adj. cost
    tau_phi_R = getattr(par, "phi_R", 0.0)  # 🔄 renter adj. cost (0 if absent)
    R = 1 + r

    uc_owner = ownc.model.num.functions.owner_marginal_utility
    uc_rent  = rntc.model.num.functions.uc_func
    uc_inv   = ownc.model.num.functions.inv_marginal_utility

    logs = []

    for iy, y_now in enumerate(z_vals):
        for ih, H_now in enumerate(H_grid):
            for iw, w_now in enumerate(w_dcsn_now):

                c0 = c_now[iw, ih, iy]
                if c0 <= 1e-12:
                    continue

                #lam_now = uc_owner(c=c0, H_nxt=H_now)
                a_next = w_now - c0
                if a_next <= 0.1:
                    continue

                E_lam = 0.0

                for jy, y_next in enumerate(z_vals):

                    # tenure choice
                    τ = int(np.interp(a_next, tenu.dcsn.grid.a, tenure_pol[:, ih, jy]))

                    if τ == 1:    # ----- owner branch -------------------
                        h_idx = int(np.interp(a_next, a_grid_O, H_pol[:, ih, jy]))
                        H1    = H_nxt_grid[h_idx]

                        w_dcsn_vals = (R * a_next + y_next
                                   - H1 + H_now
                                   - (tau_phi if H1 != H_now else 0.0))  # owner w'

                        c1 = np.interp(w_dcsn_vals, w_dcsn_o,  # 🔄 w-grid name
                                        c_owner_n[:, h_idx, jy])
                        lam_next = uc_owner(c=c1, H_nxt=H1)

                    else:         # ----- renter branch ------------------
                        s_idx = int(np.interp(a_next*R + y_next, a_grid_R, S_pol[:, jy])) # R and y drawn by the renter at the tenure cntn. 
                        S1    = S_grid[s_idx]

                        w_dcsn_vals = (R * a_next + y_next
                                   + H_now + Pr * S1
                                   - tau_phi_R)                          # renter w'

                        c1 = np.interp(w_dcsn_vals, w_dcsn_r,  # 🔄 w-grid name
                                        c_renter_n[:,s_idx, jy])
                        lam_next = uc_rent(c=c1, H_nxt=S1)

                    
                    E_lam += Pi[iy, jy] * lam_next

                c_star = uc_inv(lambda_e=beta * R * E_lam, H_nxt=H_now)
                #print(c_starr)
                logs.append(np.log10(abs((c_star - c0) / c0) + 1e-16))

    return float(np.mean(logs)) if logs else np.nan

def euler_error_metric(model):
    return calculate_euler_error(model)
