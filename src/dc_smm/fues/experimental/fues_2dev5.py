"""Fast Upper‑Envelope Scan (FUES) - Optimized Version
Performance optimized version with inlined critical scans and reduced overhead.
Based on fues_2dev4.py with efficiency improvements.

Key optimizations:
1. Inline critical backward/forward scans
2. Conditional intersection array allocation
3. Simplified flag logic
4. Early exit conditions
5. Reduced function call overhead
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

# ---------------- Intersection helper (kept for when needed) ---------

@njit
def add_intersection(inter_e, inter_v, inter_p1, inter_p2, inter_d, n_inter, max_inter,
                    intr_point, e_grid, a_prime, policy_2, del_a, idx1, idx2, idx3, idx4):
    """Add two intersection points to the arrays - one for each policy branch."""
    if not np.isnan(intr_point[0]) and n_inter + 1 < max_inter:
        # Add left branch point (slightly before intersection)
        inter_e[n_inter] = intr_point[0] - 1e-8
        inter_v[n_inter] = intr_point[1]
        
        # Interpolate policies along the left branch segment (idx1 to idx2)
        if e_grid[idx2] - e_grid[idx1] > 1e-200:
            t = (intr_point[0] - e_grid[idx1]) / (e_grid[idx2] - e_grid[idx1])
        else:
            t = 0.0
            
        inter_p1[n_inter] = a_prime[idx1] + t * (a_prime[idx2] - a_prime[idx1])
        inter_p2[n_inter] = policy_2[idx1] + t * (policy_2[idx2] - policy_2[idx1])
        inter_d[n_inter] = del_a[idx1] + t * (del_a[idx2] - del_a[idx1])
        
        # Add right branch point (slightly after intersection)
        inter_e[n_inter + 1] = intr_point[0] + 1e-8
        inter_v[n_inter + 1] = intr_point[1]
        
        # Interpolate policies along the right branch segment (idx3 to idx4)
        if e_grid[idx4] - e_grid[idx3] > 1e-200:
            t = (intr_point[0] - e_grid[idx3]) / (e_grid[idx4] - e_grid[idx3])
        else:
            t = 0.0
            
        inter_p1[n_inter + 1] = a_prime[idx3] + t * (a_prime[idx4] - a_prime[idx3])
        inter_p2[n_inter + 1] = policy_2[idx3] + t * (policy_2[idx4] - policy_2[idx3])
        inter_d[n_inter + 1] = del_a[idx3] + t * (del_a[idx4] - del_a[idx3])
        
        return n_inter + 2, intr_point[0], intr_point[1], inter_p1[n_inter], inter_d[n_inter]
    
    return n_inter, 0.0, 0.0, 0.0, 0.0

# ---------------------------------------------------------------------
# Public wrapper -------------------------------------------------------
# ---------------------------------------------------------------------

@njit
def FUES(e_grid, vf, policy_1, policy_2, del_a,
         b=1e-10, m_bar=2.0, LB=4,
         endog_mbar=False, padding_mbar=0.0,
         include_intersections=True):
    """Sort input, call scanner, drop NaNs, return cleaned arrays."""

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
    
    if include_intersections and intersections is not None:
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
# Non-jitted wrapper for getting intersections separately --------------
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
    
    # Ensure intersections is always a tuple even if None was returned
    if intersections is None:
        empty = np.zeros(0, dtype=np.float64)
        intersections = (empty, empty, empty, empty, empty)
    
    return fues_result, intersections

# ---------------------------------------------------------------------
# Core scan - Optimized version ---------------------------------------
# ---------------------------------------------------------------------

@njit
def _scan(e_grid, vf, a_prime, policy_2, del_a,
          m_bar, LB, fwd_scan_do,
          endog_mbar, padding_mbar, ID_NM = True, include_intersections = True):
    """FUES single‑pass scan - optimized with inlined scans."""

    N = e_grid.size
    vf_full = vf.copy()
    
    # Conditional intersection array allocation
    if include_intersections:
        max_inter = N // 2
        inter_e = np.full(max_inter, np.nan)
        inter_v = np.full(max_inter, np.nan)
        inter_p1 = np.full(max_inter, np.nan)
        inter_p2 = np.full(max_inter, np.nan)
        inter_d = np.full(max_inter, np.nan)
        n_inter = 0
        
        # Track intersection values for k
        use_intersection_as_k = False
        intersection_e = 0.0
        intersection_v = 0.0
        intersection_a = 0.0
        intersection_d = 0.0
        added_intersection_last_iter = False
    else:
        # Skip all intersection tracking
        n_inter = 0
        use_intersection_as_k = False
        added_intersection_last_iter = False

    # Circular buffer for recently dropped indices
    m_buf = np.full(LB, -1)        # -1 denotes empty slot
    m_head = 0                     # next write position

    # Index bookkeeping
    j, k = 0, -1
    last_turn_left = False
    prev_j = 0

    for i in range(N - 2):

        if i <= 1:                 # first two points always kept
            j, k = i, i - 1
            last_turn_left = False
            if include_intersections:
                added_intersection_last_iter = False
            continue

        # ------------- Gradients at current step --------------------
        # Get k values (tail) - optimized to avoid repeated conditionals
        if include_intersections and use_intersection_as_k:
            k_e = intersection_e
            k_v = intersection_v
            k_a = intersection_a
            k_d = intersection_d
        else:
            if k >= 0:
                k_e = e_grid[k]
                k_v = vf_full[k]
                k_a = a_prime[k]
                k_d = del_a[k]
            else:
                k_e = e_grid[0]
                k_v = vf_full[0]
                k_a = a_prime[0]
                k_d = del_a[0]
        
        # Pre-compute common values
        de_prev = max(1e-200, e_grid[j] - k_e)
        g_jm1 = (vf_full[j] - k_v) / de_prev

        de_lead = max(1e-200, e_grid[i+1] - e_grid[j])
        g_1 = (vf_full[i+1] - vf_full[j]) / de_lead

        # Compute M_max once
        if endog_mbar:
            M_max = max(np.abs(del_a[j]), np.abs(del_a[i+1])) + padding_mbar
        else:
            M_max = m_bar

        del_pol = a_prime[i+1] - a_prime[j]
        g_tilde_a = np.abs(del_pol / de_lead)
        del_pol_a = (e_grid[i+1] - a_prime[i+1]) - (e_grid[j] - a_prime[j])

        # Determine turn type
        right_turn_jump = (g_1 < g_jm1) and (g_tilde_a > M_max)
        left_turn = g_1 > g_jm1

        if include_intersections:
            added_intersection_last_iter = False
        
        # ------------- Case A: right‑turn jump (INLINED) ------------
        if right_turn_jump:
            keep_i1 = False
            
            if fwd_scan_do:
                # INLINE forward scan for Case A
                idx_f = -1
                for f in range(min(LB, N - i - 2)):
                    if i+2+f >= N:
                        break
                    de = max(1e-200, e_grid[i+1] - e_grid[i+2+f])
                    g_f_a = np.abs((a_prime[j] - a_prime[i+2+f]) / de)
                    if g_f_a < m_bar:
                        idx_f = i+2+f
                        # Check if i+1 dominates segment
                        g_f_vf_at_idx = (vf_full[i+1] - vf_full[i+2+f]) / de
                        if g_1 > g_f_vf_at_idx:
                            keep_i1 = True
                        break
                        
            if keep_i1:
                created_intersection = False
                
                # Case A intersection handling
                if include_intersections and fwd_scan_do and idx_f != -1:
                    # INLINE backward scan for intersection
                    idx_b = -1
                    for t in range(min(LB, m_head)):
                        idx_buf = (m_head - 1 - t) % LB
                        m_idx = m_buf[idx_buf]
                        if m_idx != -1 and m_idx < i+1:
                            de = max(1e-200, e_grid[i+1] - e_grid[m_idx])
                            grad_a = np.abs((a_prime[i+1] - a_prime[m_idx]) / de)
                            if grad_a < m_bar:
                                idx_b = m_idx
                                break
                    
                    if idx_b != -1:
                        # Compute intersection
                        p1 = np.array([e_grid[i+1], vf_full[i+1]])
                        p2 = np.array([e_grid[idx_b], vf_full[idx_b]])
                        p3 = np.array([e_grid[j], vf_full[j]])
                        p4 = np.array([e_grid[idx_f], vf_full[idx_f]])
                        
                        intr = seg_intersect(p1, p2, p3, p4)
                        
                        new_n_inter, inter_e_val, inter_v_val, inter_a_val, inter_d_val = add_intersection(
                            inter_e, inter_v, inter_p1, inter_p2, inter_d, n_inter, max_inter,
                            intr, e_grid, a_prime, policy_2, del_a, idx_b, i+1, j, idx_f)
                        
                        if new_n_inter > n_inter:
                            n_inter = new_n_inter
                            added_intersection_last_iter = True
                            use_intersection_as_k = True
                            intersection_e = inter_e_val
                            intersection_v = inter_v_val
                            intersection_a = inter_a_val
                            intersection_d = inter_d_val
                            created_intersection = True
                
                # Update indices
                k = j
                prev_j = j
                j = i+1
                last_turn_left = False
                if include_intersections and not created_intersection:
                    use_intersection_as_k = False
            else:
                vf[i+1] = np.nan
                m_head = circ_put(m_buf, m_head, i+1)
                if include_intersections:
                    use_intersection_as_k = False
            continue

        # ------------- Case B: drop due to value fall / monotone ----
        if (vf_full[i+1] - vf_full[j] < 0) or (ID_NM and (del_pol_a<0)):
            vf[i+1] = np.nan
            m_head = circ_put(m_buf, m_head, i+1)
            if include_intersections:
                use_intersection_as_k = False
            continue
    
        # ------------- Case C: left turn or right w/o jump (INLINED) -
        # INLINE backward scan for Case C
        keep_j = True
        m_ind = -1
        
        # Only search valid buffer entries
        for t in range(min(LB, m_head)):
            idx_buf = (m_head - 1 - t) % LB
            m_idx = m_buf[idx_buf]
            
            if m_idx != -1:
                de = max(1e-100, e_grid[j] - e_grid[m_idx])
                g_m_a = np.abs((a_prime[i+1] - a_prime[m_idx]) / de)
                if g_m_a < m_bar:
                    m_ind = m_idx
                    if left_turn and g_tilde_a and not last_turn_left:
                        g_m_vf = (vf_full[j] - vf_full[m_idx]) / de
                        if g_1 > g_m_vf:
                            keep_j = False
                    break

        ## CASE C.1.A Left Turn and drop j'th point 
        if not keep_j:
            vf[j] = np.nan
            m_head = circ_put(m_buf, m_head, j)
            
            created_intersection = False
            if include_intersections:
                use_intersection_as_k = False
                
                if not last_turn_left and m_ind != -1:
                    pj = np.array([e_grid[j], vf_full[j]])
                    pi1 = np.array([e_grid[i+1], vf_full[i+1]])
                    pk = np.array([e_grid[k], vf_full[k]])
                    pm = np.array([e_grid[m_ind], vf_full[m_ind]])
                    intr = seg_intersect(pj, pk, pi1, pm)
                    
                    new_n_inter, inter_e_val, inter_v_val, inter_a_val, inter_d_val = add_intersection(
                        inter_e, inter_v, inter_p1, inter_p2, inter_d, n_inter, max_inter,
                        intr, e_grid, a_prime, policy_2, del_a, m_ind, i+1, k, j)
                    
                    if new_n_inter > n_inter:
                        n_inter = new_n_inter
                        added_intersection_last_iter = True
                        use_intersection_as_k = True
                        intersection_e = inter_e_val
                        intersection_v = inter_v_val
                        intersection_a = inter_a_val
                        intersection_d = inter_d_val
                        created_intersection = True
            
            last_turn_left = True
            j = i+1
            if not created_intersection:
                k = prev_j
                if include_intersections:
                    use_intersection_as_k = False
       
        ## CASE C.1.B Left Turn and keep j'th point 
        else:
            if left_turn:
                if last_turn_left:
                    vf[j] = np.nan
                    m_head = circ_put(m_buf, m_head, j)
                    
                    # Remove last intersection if needed
                    if include_intersections and added_intersection_last_iter and n_inter > 0:
                        n_inter = n_inter - 1
                
                # Add intersection for left turn case (simplified)
                if include_intersections:
                    use_intersection_as_k = False
                    
                    if not last_turn_left and k >= 0:
                        # INLINE forward scan for intersection
                        idx_fwd = -1
                        for f in range(min(LB, N - j - 1)):
                            if j + 1 + f >= N:
                                break
                            de = max(1e-200, e_grid[j + 1 + f] - e_grid[j])
                            g_a = np.abs((a_prime[j + 1 + f] - a_prime[j]) / de)
                            if g_a < m_bar:
                                idx_fwd = j + 1 + f
                                break
                        
                        # INLINE backward scan for intersection
                        idx_back = -1
                        for t in range(min(LB, m_head)):
                            idx_buf = (m_head - 1 - t) % LB
                            m_idx = m_buf[idx_buf]
                            if m_idx != -1 and m_idx < i+1:
                                de = max(1e-200, e_grid[i+1] - e_grid[m_idx])
                                grad_a = np.abs((a_prime[i+1] - a_prime[m_idx]) / de)
                                if grad_a < m_bar:
                                    idx_back = m_idx
                                    break
                        
                        if idx_fwd != -1 and idx_back != -1:
                            pj = np.array([e_grid[j], vf_full[j]])
                            pfwd = np.array([e_grid[idx_fwd], vf_full[idx_fwd]])
                            pi1 = np.array([e_grid[i+1], vf_full[i+1]])
                            pback = np.array([e_grid[idx_back], vf_full[idx_back]])
                            
                            intr = seg_intersect(pj, pfwd, pi1, pback)
                            
                            new_n_inter, inter_e_val, inter_v_val, inter_a_val, inter_d_val = add_intersection(
                                inter_e, inter_v, inter_p1, inter_p2, inter_d, n_inter, max_inter,
                                intr, e_grid, a_prime, policy_2, del_a, idx_back, i+1, j, idx_fwd)
                            
                            if new_n_inter > n_inter:
                                n_inter = new_n_inter
                                added_intersection_last_iter = True
                                use_intersection_as_k = True
                                intersection_e = inter_e_val
                                intersection_v = inter_v_val
                                intersection_a = inter_a_val
                                intersection_d = inter_d_val
                
                last_turn_left = True
                k = j
                prev_j = j
                j = i + 1
            
            ## CASE C.1.C Right Turn and keep j'th point 
            else:
                last_turn_left = False
                k = j
                prev_j = j
                j = i + 1
                if include_intersections:
                    use_intersection_as_k = False

    # Package intersection results
    if include_intersections and n_inter > 0:
        intersections = (
            inter_e[:n_inter].copy(),
            inter_v[:n_inter].copy(),
            inter_p1[:n_inter].copy(),
            inter_p2[:n_inter].copy(),
            inter_d[:n_inter].copy()
        )
    elif include_intersections:
        empty = np.zeros(0, dtype=np.float64)
        intersections = (empty, empty, empty, empty, empty)
    else:
        intersections = None
    
    return e_grid, vf, intersections