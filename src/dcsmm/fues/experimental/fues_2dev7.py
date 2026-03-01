"""Fast Upper‑Envelope Scan (FUES) - Performance Critical Version
Aggressive optimizations focusing on the most common use case (no intersections).

Key optimizations:
1. Removed all intersection tracking from main path
2. Simplified flag logic
3. Reduced conditional branches
4. Direct indexing without bounds checks where safe
5. Optimized circular buffer usage
"""

from numba import njit
import numpy as np

# ---------------------------------------------------------------------
# Core helpers ---------------------------------------------------------
# ---------------------------------------------------------------------

@njit
def uniqueEG(egrid, vf):
    egrid_rounded = np.round(egrid, 10)
    unique_vals = np.unique(egrid_rounded)
    keep = np.full_like(egrid, False, dtype=np.bool_)
    for val in unique_vals:
        if np.isnan(val):
            continue
        idx = np.where(egrid_rounded == val)[0]
        keep[idx[np.argmax(vf[idx])]] = True
    return keep

# ---------------------------------------------------------------------
# Main FUES - Optimized for speed without intersections ---------------
# ---------------------------------------------------------------------

@njit
def FUES(e_grid, vf, policy_1, policy_2, del_a,
         b=1e-10, m_bar=2.0, LB=4,
         endog_mbar=False, padding_mbar=0.0,
         include_intersections=False):
    """Fast Upper Envelope Scan - optimized for common case."""
    
    # Sort inputs
    idx = np.argsort(e_grid)
    e_grid_s = e_grid[idx]
    vf_s = vf[idx]
    policy_1_s = policy_1[idx]
    policy_2_s = policy_2[idx]
    del_a_s = del_a[idx]

    # Call optimized scan
    e_out, vf_out = _scan_fast(
        e_grid_s, vf_s, policy_1_s, policy_2_s, del_a_s,
        m_bar, LB, True, endog_mbar, padding_mbar)

    # Extract kept points
    keep = ~np.isnan(vf_out)
    return (e_grid_s[keep], vf_s[keep], policy_1_s[keep], 
            policy_2_s[keep], del_a_s[keep])

@njit
def _scan_fast(e_grid, vf, a_prime, policy_2, del_a,
               m_bar, LB, fwd_scan_do, endog_mbar, padding_mbar):
    """Core scan optimized for speed."""
    
    N = e_grid.size
    vf_full = vf.copy()  # Keep original values
    
    # Circular buffer - use int32 for speed
    m_buf = np.full(LB, -1, dtype=np.int32)
    m_head = 0

    # Indices
    j = 0
    k = -1
    last_left = False
    prev_j = 0

    # Main loop - unroll initial iterations
    for i in range(2, N - 1):
        
        # Compute gradients efficiently
        e_j = e_grid[j]
        e_i1 = e_grid[i+1]
        v_j = vf_full[j]
        v_i1 = vf_full[i+1]
        
        # Get k values without conditionals where possible
        if k < 0:
            de_prev = e_j - e_grid[0]
            g_jm1 = (v_j - vf_full[0]) / max(1e-200, de_prev)
        else:
            de_prev = e_j - e_grid[k]
            g_jm1 = (v_j - vf_full[k]) / max(1e-200, de_prev)
        
        de_lead = e_i1 - e_j
        if de_lead < 1e-200:
            de_lead = 1e-200
        
        g_1 = (v_i1 - v_j) / de_lead
        
        # Policy gradient
        del_pol = a_prime[i+1] - a_prime[j]
        g_tilde_a = np.abs(del_pol / de_lead)
        
        # M threshold
        if endog_mbar:
            M_max = max(np.abs(del_a[j]), np.abs(del_a[i+1])) + padding_mbar
        else:
            M_max = m_bar
        
        # Case determination
        if g_1 < g_jm1 and g_tilde_a > M_max:
            # Case A: right-turn jump
            keep_i1 = False
            
            if fwd_scan_do:
                # Inline forward scan - optimized
                for f in range(min(LB, N - i - 2)):
                    idx_f = i + 2 + f
                    if idx_f >= N:
                        break
                    
                    de_f = e_i1 - e_grid[idx_f]
                    if de_f < 1e-200:
                        continue
                        
                    if np.abs((a_prime[j] - a_prime[idx_f]) / de_f) < m_bar:
                        if g_1 > (v_i1 - vf_full[idx_f]) / de_f:
                            keep_i1 = True
                        break
            
            if keep_i1:
                k = j
                prev_j = j
                j = i + 1
                last_left = False
            else:
                vf[i+1] = np.nan
                m_buf[m_head] = i + 1
                m_head = (m_head + 1) % LB
            continue
            
        # Case B: value fall
        if v_i1 < v_j:
            vf[i+1] = np.nan
            m_buf[m_head] = i + 1
            m_head = (m_head + 1) % LB
            continue
        
        # Case C: left turn or right without jump
        keep_j = True
        
        # Only do backward scan if left turn
        if g_1 > g_jm1 and g_tilde_a > M_max and not last_left:
            # Search buffer for same branch point
            for t in range(min(LB, m_head)):
                idx_buf = (m_head - 1 - t) % LB
                m_idx = m_buf[idx_buf]
                
                if m_idx >= 0:
                    de_m = e_j - e_grid[m_idx]
                    if de_m > 1e-100:
                        if np.abs((a_prime[i+1] - a_prime[m_idx]) / de_m) < m_bar:
                            if g_1 > (v_j - vf_full[m_idx]) / de_m:
                                keep_j = False
                            break
        
        if not keep_j:
            # Drop j
            vf[j] = np.nan
            m_buf[m_head] = j
            m_head = (m_head + 1) % LB
            j = i + 1
            k = prev_j
            last_left = True
        else:
            # Keep j
            if g_1 > g_jm1:
                # Left turn
                if last_left:
                    # Consecutive left turn - drop j
                    vf[j] = np.nan
                    m_buf[m_head] = j
                    m_head = (m_head + 1) % LB
                last_left = True
            else:
                last_left = False
            
            k = j
            prev_j = j
            j = i + 1

    return e_grid, vf

# ---------------------------------------------------------------------
# Compatibility wrapper for intersection tracking ---------------------
# ---------------------------------------------------------------------

def FUES_sep_intersect(e_grid, vf, policy_1, policy_2, del_a,
                       b=1e-10, m_bar=2.0, LB=4,
                       endog_mbar=False, padding_mbar=0.0):
    """
    For compatibility - just returns FUES result with empty intersections.
    Full intersection tracking would need the complete implementation.
    """
    # Get FUES result
    fues_result = FUES(e_grid, vf, policy_1, policy_2, del_a,
                      b, m_bar, LB, endog_mbar, padding_mbar, False)
    
    # Return empty intersections
    empty = np.zeros(0, dtype=np.float64)
    intersections = (empty, empty, empty, empty, empty)
    
    print(intersections)  # For compatibility
    
    return fues_result, intersections