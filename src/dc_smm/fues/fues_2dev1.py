"""Fast Upper‑Envelope Scan (FUES)
Optimised baseline with pre‑allocated scratch buffers and a true
circular drop list (LB‑length) — no shifting, no repeated small
allocations.

Implements speed ideas **#1** and **#2** from the tuning menu while
keeping logic, comments and notation identical to Dobrescu & Shanker
(2023).

NB: Generators are not supported in Numba nopython mode.  The previous
commit used a `circ_iter_last()` generator which caused a typing error.
This revision inlines the reverse‑iteration logic with simple index
arithmetic to stay fully nopython‑compliant.
"""

from numba import njit
import numpy as np

# ---------------------------------------------------------------------
# Helpers that remain identical ---------------------------------------
# ---------------------------------------------------------------------

@njit
def uniqueEG(egrid, vf):
    egrid_rounded = np.round_(egrid, 10)
    unique_vals = np.unique(egrid_rounded)
    keep = np.full_like(egrid, False, dtype=np.bool_)
    for val in unique_vals:
        if np.isnan(val):
            continue
        idx = np.where(egrid_rounded == val)[0]
        keep[idx[np.argmax(vf[idx])]] = True
    return keep

# ---------------- Circular buffer utilities --------------------------

@njit
def circ_put(buf, head, value):
    """Write *value* at *head* position, return new head index."""
    buf[head] = value
    return (head + 1) % buf.size

# ---------------- Segment intersection (unchanged) -------------------

@njit
def linear_interp(x, x1, x2, y1, y2):
    """Linear interpolation helper."""
    if np.abs(x2 - x1) < 1e-200:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

@njit
def seg_intersect(a1, a2, b1, b2):
    da_x, da_y = a2[0] - a1[0], a2[1] - a1[1]
    db_x, db_y = b2[0] - b1[0], b2[1] - b1[1]
    dp_x, dp_y = a1[0] - b1[0], a1[1] - b1[1]
    dap_x, dap_y = -da_y, da_x
    denom = dap_x * db_x + dap_y * db_y
    if denom == 0.0:
        return np.array([np.nan, np.nan])
    t = (dap_x * dp_x + dap_y * dp_y) / denom
    return np.array([t * db_x + b1[0], t * db_y + b1[1]])

# ---------------- Intersection helpers -------------------

@njit
def add_intersection(inter_e, inter_v, inter_p1, inter_p2, inter_d, n_inter, max_inter,
                    intr_point, e_grid, a_prime, policy_2, del_a, idx1, idx2):
    """Add an intersection point to the arrays and interpolate policies.
    
    Returns updated n_inter and the interpolated values for the intersection.
    """
    if not np.isnan(intr_point[0]) and n_inter < max_inter:
        inter_e[n_inter] = intr_point[0]
        inter_v[n_inter] = intr_point[1]
        
        # Interpolate policies along the segment (idx1 to idx2)
        if e_grid[idx2] - e_grid[idx1] > 1e-200:
            t = (intr_point[0] - e_grid[idx1]) / (e_grid[idx2] - e_grid[idx1])
        else:
            t = 0.0
            
        inter_p1[n_inter] = a_prime[idx1] + t * (a_prime[idx2] - a_prime[idx1])
        inter_p2[n_inter] = policy_2[idx1] + t * (policy_2[idx2] - policy_2[idx1])
        inter_d[n_inter] = del_a[idx1] + t * (del_a[idx2] - del_a[idx1])
        
        return n_inter + 1, intr_point[0], intr_point[1], inter_p1[n_inter], inter_d[n_inter]
    
    return n_inter, 0.0, 0.0, 0.0, 0.0

@njit 
def find_backward_same_branch(m_buf, m_head, LB, e_grid, a_prime, i_plus_1, m_bar):
    """Find the first point in backward scan that's on same branch.
    
    Returns found flag and index.
    """
    for t in range(LB):
        idx_buf = (m_head - 1 - t) % LB
        m_idx = m_buf[idx_buf]
        if m_idx != -1 and m_idx < i_plus_1:
            de = max(1e-200, e_grid[i_plus_1] - e_grid[m_idx])
            grad_a = np.abs((a_prime[i_plus_1] - a_prime[m_idx]) / de)
            if grad_a < m_bar:
                return True, m_idx
    return False, -1

@njit
def find_forward_same_branch(e_grid, a_prime, start_idx, j_idx, N, LB, m_bar):
    """Find the first point in forward scan that's on same branch.
    
    Returns found flag and index.
    """
    for f in range(min(LB, N - start_idx - 1)):
        if start_idx + 1 + f >= N:
            break
        de = max(1e-200, e_grid[start_idx + 1 + f] - e_grid[j_idx])
        g_a = np.abs((a_prime[start_idx + 1 + f] - a_prime[j_idx]) / de)
        if g_a < m_bar:
            return True, start_idx + 1 + f
    return False, -1

# ---------------------------------------------------------------------
# Public wrapper -------------------------------------------------------
# ---------------------------------------------------------------------

@njit
def FUES(e_grid, vf, policy_1, policy_2, del_a,
         b=1e-10, m_bar=2.0, LB=4,
         endog_mbar=False, padding_mbar=0.0,
         include_intersections=True):
    """Sort input, call scanner, drop NaNs, return cleaned arrays.
    
    Parameters:
    -----------
    include_intersections : bool, default False
        If True, intersection points where discrete choices switch are included in output.
        If False, returns only the original upper envelope points.
    """

    idx = np.argsort(e_grid)
    e_grid = e_grid[idx]
    vf = vf[idx]
    policy_1 = policy_1[idx]
    policy_2 = policy_2[idx]
    del_a = del_a[idx]

    e_grid_out, vf_marked, intersections = _scan(
        e_grid, vf, policy_1, policy_2, del_a,
        m_bar, LB, True, endog_mbar, padding_mbar, include_intersections)

    # Extract kept points
    keep = ~np.isnan(vf_marked)
    e_kept = e_grid_out[keep]
    v_kept = vf[keep]
    p1_kept = policy_1[keep]
    p2_kept = policy_2[keep]
    d_kept = del_a[keep]
    
    if include_intersections:
        #print(intersections)
        # Extract intersection points
        inter_e, inter_v, inter_p1, inter_p2, inter_d = intersections
        
        # If we have intersections, merge them with kept points
        if len(inter_e) > 0:
            # Combine arrays
            all_e = np.concatenate((e_kept, inter_e))
            all_v = np.concatenate((v_kept, inter_v))
            all_p1 = np.concatenate((p1_kept, inter_p1))
            all_p2 = np.concatenate((p2_kept, inter_p2))
            all_d = np.concatenate((d_kept, inter_d))
            
            # Sort by e_grid to maintain order
            sort_idx = np.argsort(all_e)
            return (all_e[sort_idx], all_v[sort_idx], 
                    all_p1[sort_idx], all_p2[sort_idx], all_d[sort_idx])
    
    # Return only kept points (original behavior)
    return (e_kept, v_kept, p1_kept, p2_kept, d_kept)

# ---------------------------------------------------------------------
# Non-jitted wrapper for getting intersections separately ---------------
# ---------------------------------------------------------------------

def FUES_sep_intersect(e_grid, vf, policy_1, policy_2, del_a,
                       b=1e-10, m_bar=2.0, LB=4,
                       endog_mbar=False, padding_mbar=0.0):
    """
    Non-jitted wrapper that returns FUES results and intersection points separately.
    This is intended for plotting purposes only.
    
    Returns
    -------
    fues_result : tuple
        Standard FUES output (e_grid, vf, policy_1, policy_2, del_a)
    intersections : tuple
        Intersection points (inter_e, inter_v, inter_p1, inter_p2, inter_d)
    """
    # Sort inputs
    idx = np.argsort(e_grid)
    e_grid_sorted = e_grid[idx]
    vf_sorted = vf[idx]
    policy_1_sorted = policy_1[idx]
    policy_2_sorted = policy_2[idx]
    del_a_sorted = del_a[idx]
    
    # Call scan WITH intersection tracking to get both FUES result and intersections
    e_grid_out, vf_marked, intersections = _scan(
        e_grid_sorted, vf_sorted, policy_1_sorted, policy_2_sorted, del_a_sorted,
        m_bar, LB, True, endog_mbar, padding_mbar, True)
    
    # Extract kept points for FUES result
    print(intersections)
    keep = ~np.isnan(vf_marked)
    fues_result = (e_grid_out[keep], vf_sorted[keep],
                   policy_1_sorted[keep], policy_2_sorted[keep], del_a_sorted[keep])
    

    
    return fues_result, intersections

# ---------------------------------------------------------------------
# Core scan ------------------------------------------------------------
# ---------------------------------------------------------------------

@njit
def _scan(e_grid, vf, a_prime, policy_2, del_a,
          m_bar, LB, fwd_scan_do,
          endog_mbar, padding_mbar, ID_NM = True, include_intersections = True):
    """FUES single‑pass scan (no consecutive left turns).

    Implements:
    1. Pre‑allocated scratch arrays for gradient scans (idea #1).
    2. Circular buffer `m_buf` for last‑dropped indices (idea #2).
    3. Optional intersection tracking when include_intersections=True.
    """

    N = e_grid.size
    vf_full = vf.copy()
    
    # Arrays to track intersection points
    max_inter = N // 2
    inter_e = np.full(max_inter, np.nan)
    inter_v = np.full(max_inter, np.nan)
    inter_p1 = np.full(max_inter, np.nan)
    inter_p2 = np.full(max_inter, np.nan)
    inter_d = np.full(max_inter, np.nan)
    n_inter = 0
    
    # Track if last iteration created an intersection that should be used as k (tail)
    use_intersection_as_k = False
    intersection_e = 0.0
    intersection_v = 0.0
    intersection_a = 0.0
    intersection_d = 0.0

    # Scratch arrays (pre‑allocated once)
    g_m_vf = np.empty(LB)
    g_m_a = np.empty(LB)
    g_f_vf = np.empty(LB)
    g_f_a = np.empty(LB)

    # Circular buffer for recently dropped indices
    m_buf = np.full(LB, -1)        # -1 denotes empty slot
    m_head = 0                     # next write position

    # Index bookkeeping
    j, k = 0, -1
    idx_grad_base = k
    last_turn_left = False

    for i in range(N - 2):

        if i <= 1:                 # first two points always kept
            j, k = i, i - 1
            idx_grad_base = k
            last_turn_left = False
            continue

        # ------------- Gradients at current step --------------------
        # Use intersection values for k (tail) if we have one from last iteration
        if use_intersection_as_k and include_intersections:
            k_e = intersection_e
            k_v = intersection_v
            k_a = intersection_a
            k_d = intersection_d
        else:
            k_e = e_grid[k] if k >= 0 else e_grid[0]
            k_v = vf_full[k] if k >= 0 else vf_full[0]
            k_a = a_prime[k] if k >= 0 else a_prime[0]
            k_d = del_a[k] if k >= 0 else del_a[0]
        
        de_prev = max(1e-200, e_grid[j] - k_e)
        g_jm1 = (vf_full[j] - k_v) / de_prev

        de_lead = max(1e-200, e_grid[i+1] - e_grid[j])
        g_1 = (vf_full[i+1] - vf_full[j]) / de_lead

        M_max = max(np.abs(del_a[j]), np.abs(del_a[i+1])) + padding_mbar
        if not endog_mbar:
            M_max = m_bar

        del_pol = a_prime[i+1] - a_prime[j]
        g_tilde_a = np.abs((a_prime[i+1] - a_prime[j]) / de_lead)

        del_pol_a = (e_grid[i+1] - a_prime[i+1]) - (e_grid[j] - a_prime[j])

        #if ID_NM:
        #    right_turn_jump = ((g_1 < g_jm1) and (g_tilde_a > M_max)) or (del_pol_a<0 and g_1 < g_jm1)
            #left_turn = del_pol> 0 and g_1 > g_jm1
        #else:
        right_turn_jump = (g_1 < g_jm1) and (g_tilde_a > M_max)
            #left_turn = g_1 > g_jm1

        # ------------- Case A: right‑turn jump ----------------------
            
        #right_turn_jump = (g_1 < g_jm1) and (g_tilde_a > M_max)
        left_turn = g_1 > g_jm1

        # ------------- Case A: right‑turn jump ----------------------
        if right_turn_jump:
            keep_i1 = False
            if fwd_scan_do:
                # forward gradients into scratch arrays
                for f in range(LB):
                    if i+2+f >= N:  # CRITICAL: Add bounds check
                        break
                    de = max(1e-200, e_grid[i+1] - e_grid[i+2+f])
                    g_f_vf[f] = (vf_full[i+1] - vf_full[i+2+f]) / de
                    g_f_a[f] = np.abs((a_prime[j] - a_prime[i+2+f]) / de)
                # search first same‑branch point
                idx_f = -1
                for f in range(LB):
                    if i+2+f >= N:  # CRITICAL: Add bounds check
                        break
                    if g_f_a[f] < m_bar:
                        idx_f = f
                        break
                if idx_f != -1 and g_1 > g_f_vf[idx_f]:
                    keep_i1 = True
            if keep_i1:
                created_intersection = False
                
                # Case A intersection: Add intersection when jumping to new branch
                if include_intersections and idx_f != -1:
                    # Find backward point on same branch from i+1
                    found_b, idx_b = find_backward_same_branch(
                        m_buf, m_head, LB, e_grid, a_prime, i+1, m_bar)
                    
                    if found_b:
                        # Find intersection between (A=i+1, B=idx_b) and (k, i+2+idx_f)
                        p1 = np.array([e_grid[i+1], vf_full[i+1]])
                        p2 = np.array([e_grid[idx_b], vf_full[idx_b]])
                        p3 = np.array([e_grid[k], vf_full[k]]) if k >= 0 else np.array([e_grid[0], vf_full[0]])
                        p4 = np.array([e_grid[i+2+idx_f], vf_full[i+2+idx_f]])
                        
                        intr = seg_intersect(p1, p2, p3, p4)
                        
                        # Add intersection and get updated values
                        new_n_inter, inter_e_val, inter_v_val, inter_a_val, inter_d_val = add_intersection(
                            inter_e, inter_v, inter_p1, inter_p2, inter_d, n_inter, max_inter,
                            intr, e_grid, a_prime, policy_2, del_a, idx_b, i+1)
                        
                        if new_n_inter > n_inter:  # Intersection was added
                            n_inter = new_n_inter
                            
                            # Set up intersection to be used as k (tail) in next iteration
                            use_intersection_as_k = True
                            intersection_e = inter_e_val
                            intersection_v = inter_v_val
                            intersection_a = inter_a_val
                            intersection_d = inter_d_val
                            
                            # k will be set to intersection value via flag
                            idx_grad_base = idx_b
                            j = i+1
                            last_turn_left = False
                            created_intersection = True
                
                if not created_intersection:
                    # Normal case: i+1 is kept without intersection
                    k, idx_grad_base, j = j, j, i+1
                    last_turn_left = False
                    use_intersection_as_k = False  # Reset flag
            else:
                vf[i+1] = np.nan
                m_head = circ_put(m_buf, m_head, i+1)
            continue

        # ------------- Case B: drop due to value fall / monotone ----
        if (vf_full[i+1] - vf_full[j] < 0) or (ID_NM and (del_pol_a<0)):
            vf[i+1] = np.nan
            m_head = circ_put(m_buf, m_head, i+1)
            continue
    

        # ------------- Case C: left turn or right w/o jump ----------
        # Backward scan using circular buffer → fill g_m_* scratch
        for t in range(LB):
            idx_buf = (m_head - 1 - t) % LB
            m_idx = m_buf[idx_buf]
            if m_idx == -1:
                g_m_a[t] = np.inf
                g_m_vf[t] = -np.inf
            else:
                de = max(1e-100, e_grid[j] - e_grid[m_idx])
                g_m_vf[t] = (vf_full[j] - vf_full[m_idx]) / de
                g_m_a[t] = np.abs((a_prime[i+1] - a_prime[m_idx]) / de)

        keep_j = True
        m_ind = -1
        for t in range(LB-1, -1, -1):  # last same‑branch point if any
            if g_m_a[t] < m_bar:
                m_ind = int(m_buf[(m_head - 1 - t) % LB])
                if left_turn and g_1 >= g_m_vf[t] and g_tilde_a > m_bar:
                    keep_j = False
                break

        if not keep_j:
            pj = np.array([e_grid[j], vf_full[j]])
            pi1 = np.array([e_grid[i+1], vf_full[i+1]])
            pk = np.array([e_grid[k], vf_full[k]])
            pm = np.array([e_grid[m_ind], vf_full[m_ind]])
            intr = seg_intersect(pj, pk, pi1, pm)
            
            vf[j] = np.nan
            vf_full[j] = intr[1]
            e_grid[j] = intr[0] # should this be deleted?
            idx_grad_base = m_ind
            last_turn_left = True
            
            if include_intersections:
                # Add intersection and get updated values
                new_n_inter, inter_e_val, inter_v_val, inter_a_val, inter_d_val = add_intersection(
                    inter_e, inter_v, inter_p1, inter_p2, inter_d, n_inter, max_inter,
                    intr, e_grid, a_prime, policy_2, del_a, m_ind, i+1)
                
                if new_n_inter > n_inter:  # Intersection was added
                    n_inter = new_n_inter
                    
                    # This intersection becomes k (tail) for next iteration
                    # (Note: e_grid[j] is already set to intr[0] above)
                    use_intersection_as_k = True
                    intersection_e = inter_e_val
                    intersection_v = inter_v_val
                    intersection_a = inter_a_val
                    intersection_d = inter_d_val
        else:
            if left_turn:
                if last_turn_left:
                    vf[j] = np.nan
                
                
                # Add intersection for left turn case
                if include_intersections and not last_turn_left and k >= 0:
                    # Find forward point on same branch from j
                    found_fwd, idx_fwd = find_forward_same_branch(
                        e_grid, a_prime, j, j, N, LB, m_bar)
                    
                    # Find backward point on same branch from i+1
                    found_back, idx_back = find_backward_same_branch(
                        m_buf, m_head, LB, e_grid, a_prime, i+1, m_bar)
                    
                    if found_fwd and found_back:
                        # Find intersection between (j, idx_fwd) and (i+1, idx_back)
                        pj = np.array([e_grid[j], vf_full[j]])
                        pfwd = np.array([e_grid[idx_fwd], vf_full[idx_fwd]])
                        pi1 = np.array([e_grid[i+1], vf_full[i+1]])
                        pback = np.array([e_grid[idx_back], vf_full[idx_back]])
                        
                        intr = seg_intersect(pj, pfwd, pi1, pback)
                        
                        # Add intersection and get updated values
                        new_n_inter, inter_e_val, inter_v_val, inter_a_val, inter_d_val = add_intersection(
                            inter_e, inter_v, inter_p1, inter_p2, inter_d, n_inter, max_inter,
                            intr, e_grid, a_prime, policy_2, del_a, idx_back, i+1)
                        
                        if new_n_inter > n_inter:  # Intersection was added
                            n_inter = new_n_inter
                            
                            # This intersection becomes k (tail) for next iteration
                            use_intersection_as_k = True
                            intersection_e = inter_e_val
                            intersection_v = inter_v_val
                            intersection_a = inter_a_val
                            intersection_d = inter_d_val
                last_turn_left = True
            else:
                last_turn_left = False
            # update anchors and advance leader
            k = j
            idx_grad_base = k
            if not left_turn:
                use_intersection_as_k = False  # Reset flag only if not left turn
        # set new leading optimal index
        j = i + 1

    # Package intersection results
    if n_inter > 0:
        intersections = (
            inter_e[:n_inter].copy(),
            inter_v[:n_inter].copy(),
            inter_p1[:n_inter].copy(),
            inter_p2[:n_inter].copy(),
            inter_d[:n_inter].copy()
        )
    else:
        empty = np.zeros(0, dtype=np.float64)
        intersections = (empty.copy(), empty.copy(), empty.copy(), empty.copy(), empty.copy())
    
    return e_grid, vf, intersections