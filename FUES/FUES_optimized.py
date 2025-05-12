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

# ---------------------------------------------------------------------
# Public wrapper -------------------------------------------------------
# ---------------------------------------------------------------------

@njit
def FUES(e_grid, vf, policy_1, policy_2, del_a,
         b=1e-10, m_bar=2.0, LB=4,
         endog_mbar=False, padding_mbar=0.0):
    """Sort input, call scanner, drop NaNs, return cleaned arrays."""

    #print("M_bar: ", m_bar)
    #print("LB: ", LB)

    idx = np.argsort(e_grid)
    e_grid = e_grid[idx]
    vf = vf[idx]
    policy_1 = policy_1[idx]
    policy_2 = policy_2[idx]
    del_a = del_a[idx]

    e_grid_out, vf_marked = _scan(
        e_grid, vf, policy_1, del_a,
        m_bar, LB, True, endog_mbar, padding_mbar)

    keep = ~np.isnan(vf_marked)
    return (e_grid_out[keep], vf[keep],
            policy_1[keep], policy_2[keep], del_a[keep])

# ---------------------------------------------------------------------
# Core scan ------------------------------------------------------------
# ---------------------------------------------------------------------

@njit
def _scan(e_grid, vf, a_prime, del_a,
          m_bar, LB, fwd_scan_do,
          endog_mbar, padding_mbar):
    """FUES single‑pass scan (no consecutive left turns).

    Implements:
    1. Pre‑allocated scratch arrays for gradient scans (idea #1).
    2. Circular buffer `m_buf` for last‑dropped indices (idea #2).
    """

    N = e_grid.size
    vf_full = vf.copy()

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
        de_prev = max(1e-200, e_grid[j] - e_grid[idx_grad_base])
        g_jm1 = (vf_full[j] - vf_full[idx_grad_base]) / de_prev

        de_lead = max(1e-200, e_grid[i+1] - e_grid[j])
        g_1 = (vf_full[i+1] - vf_full[j]) / de_lead

        M_max = max(np.abs(del_a[j]), np.abs(del_a[i+1])) + padding_mbar
        if not endog_mbar:
            M_max = m_bar
        g_tilde_a = np.abs((a_prime[i+1] - a_prime[j]) / de_lead)

        right_turn_jump = (g_1 < g_jm1) and (g_tilde_a > M_max)
        left_turn = g_1 > g_jm1

        # ------------- Case A: right‑turn jump ----------------------
        if right_turn_jump:
            keep_i1 = False
            if fwd_scan_do:
                # forward gradients into scratch arrays
                for f in range(LB):
                    de = max(1e-200, e_grid[i+1] - e_grid[i+2+f])
                    g_f_vf[f] = (vf_full[i+1] - vf_full[i+2+f]) / de
                    g_f_a[f] = np.abs((a_prime[j] - a_prime[i+2+f]) / de)
                # search first same‑branch point
                idx_f = -1
                for f in range(LB):
                    if g_f_a[f] < m_bar:
                        idx_f = f
                        break
                if idx_f != -1 and g_1 > g_f_vf[idx_f]:
                    keep_i1 = True
            if keep_i1:
                k, idx_grad_base, j = j, j, i+1
                last_turn_left = False
            else:
                vf[i+1] = np.nan
                m_head = circ_put(m_buf, m_head, i+1)
            continue

        # ------------- Case B: drop due to value fall / monotone ----
        if (vf_full[i+1] - vf_full[j] < 0) or (g_1 < g_jm1 and (a_prime[i+1] - a_prime[j]) < 0):
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
            e_grid[j] = intr[0]
            idx_grad_base = m_ind
            last_turn_left = True
        else:
            if left_turn:
                if last_turn_left:
                    vf[j] = np.nan
                last_turn_left = True
            else:
                last_turn_left = False
            # update anchors and advance leader
            k = j
            idx_grad_base = k
        # set new leading optimal index
        j = i + 1

    return e_grid, vf
