"""
FUES-REFINE  —  exact jump-kink augmentation
-------------------------------------------
Public API is identical to the classic FUES:

    m_ref, v_ref, pol1_ref, pol2_ref, da_ref = FUES(...)

After the fast upper-envelope scan it inserts two ε-separated nodes
around every kept jump:

1.  **m_int**  – the true intersection of value-line (kb → j) and
    (j → kf) found in the *unrefined* EGM grid.
2.  **m_int+ε** – an infinitesimal step on the *other* branch so that
    linear interpolation on either side is exact.

Both policy arrays and the value function use the correct branch anchors
(kb, kf) from the full grid.
"""

from numba import njit
import numpy as np

# --------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------
@njit
def circ_put(buf, head, val):
    buf[head] = val
    return (head + 1) % buf.size


@njit
def seg_intersect(a1, a2, b1, b2):
    """Intersection of two (x,y) line segments.  Returns (x*, y*)."""
    da_x, da_y = a2[0] - a1[0], a2[1] - a1[1]
    db_x, db_y = b2[0] - b1[0], b2[1] - b1[1]
    dp_x, dp_y = a1[0] - b1[0], a1[1] - b1[1]
    dap_x, dap_y = -da_y, da_x
    denom = dap_x * db_x + dap_y * db_y
    if denom == 0.0:
        return np.array([np.nan, np.nan])
    t = (dap_x * dp_x + dap_y * dp_y) / denom
    return np.array([t * db_x + b1[0], t * db_y + b1[1]])

# --------------------------------------------------------------------
# Jump detection and kink augmentation
# --------------------------------------------------------------------
@njit
def _collect_jump_idx(E, P1, m_bar):
    """Return indices j where segment (j→j+1) is a jump."""
    N = E.size
    out = np.empty(N - 1, np.int64)
    k = 0
    for i in range(N - 1):
        de = E[i + 1] - E[i]
        if de != 0.0 and abs((P1[i + 1] - P1[i]) / de) > m_bar:
            out[k] = i
            k += 1
    return out[:k]


@njit
def _insert_jump_kinks(Ec, Vc, P1c, P2c, DAc,
                       jump_idx, m_bar,
                       Ef, Vf, P1f, P2f):
    """
    Insert two ε-nodes around every kept jump.

    •  kb  = first *previous* full-grid index on same branch (slope ≤ m_bar).
    •  kf  = first *forward*  full-grid index on same branch.
    •  m_int = intersection of (kb→j) and (j→kf) in (m,v) plane.
    •  Add nodes at m_int   (forward branch)  and  m_int+ε (backward branch).
    """
    Nc = Ec.size
    K = jump_idx.size
    if K == 0:
        return Ec, Vc, P1c, P2c, DAc

    M = Nc + 2 * K
    E2 = np.empty(M)
    V2 = np.empty(M)
    P12 = np.empty(M)
    P22 = np.empty(M)
    DA2 = np.empty(M)

    # map cleaned indices → positions in full grid (unique E assumed)
    pos_full = np.empty(Nc, np.int64)
    f_ptr = 0
    for i in range(Nc):
        while Ef[f_ptr] != Ec[i]:
            f_ptr += 1
        pos_full[i] = f_ptr

    w = 0
    jp = 0
    tgt = jump_idx[jp] if K else -1

    for j in range(Nc - 1):
        # copy clean node
        E2[w] = Ec[j]
        V2[w] = Vc[j]
        P12[w] = P1c[j]
        P22[w] = P2c[j]
        DA2[w] = DAc[j]
        w += 1

        if j != tgt:
            continue

        pos_j = pos_full[j]
        Ej = Ec[j]
        P1j = P1c[j]

        # ---- find kb (backward) ----------------------------------
        kb = -1
        for b in range(pos_j - 1, -1, -1):
            de = Ej - Ef[b]
            if de != 0.0 and abs((P1j - P1f[b]) / de) <= m_bar:
                kb = b
                break

        # ---- find kf (forward) -----------------------------------
        kf = -1
        for f in range(pos_j + 2, Ef.size):          # skip j+1 (still jump)
            de = Ef[f] - Ej
            if de != 0.0 and abs((P1f[f] - P1j) / de) <= m_bar:
                kf = f
                break

        if kb == -1 or kf == -1:
            jp += 1
            tgt = jump_idx[jp] if jp < K else -1
            continue

        # ---- true intersection in (m,v) --------------------------
        mv_int = seg_intersect(np.array([Ef[kb], Vf[kb]]),
                               np.array([Ej,      Vf[pos_j]]),
                               np.array([Ej,      Vf[pos_j]]),
                               np.array([Ef[kf],  Vf[kf]]))
        m_int, v_int = mv_int[0], mv_int[1]
        if np.isnan(m_int):
            jp += 1
            tgt = jump_idx[jp] if jp < K else -1
            continue

        # policies at m_int on *forward* branch (j → kf)
        t_fwd = (m_int - Ej) / (Ef[kf] - Ej)
        p1_int = P1j + t_fwd * (P1f[kf] - P1j)
        p2_int = P2c[j] + t_fwd * (P2f[kf] - P2c[j])
        da_int = DAc[j]

        # ε-shift node on *backward* branch (j → kb)
        eps = 1e-12 * (Ef[kf] - Ej)
        m_eps = m_int + eps
        t_back = (m_eps - Ej) / (Ej - Ef[kb])
        v_eps = Vf[pos_j] + t_back * (Vf[pos_j] - Vf[kb])
        p1_eps = P1j + t_back * (P1j - P1f[kb])
        p2_eps = P2c[j] + t_back * (P2c[j] - P2f[kb])
        da_eps = DAc[j]

        # append m_int then m_int+ε (grid increasing)
        E2[w] = m_int;   V2[w] = v_int
        P12[w] = p1_int; P22[w] = p2_int; DA2[w] = da_int; w += 1
        E2[w] = m_eps;   V2[w] = v_eps
        P12[w] = p1_eps; P22[w] = p2_eps; DA2[w] = da_eps; w += 1

        jp += 1
        tgt = jump_idx[jp] if jp < K else -1

    # copy last clean node
    E2[w] = Ec[-1]; V2[w] = Vc[-1]
    P12[w] = P1c[-1]; P22[w] = P2c[-1]; DA2[w] = DAc[-1]

    return E2[:w + 1], V2[:w + 1], P12[:w + 1], P22[:w + 1], DA2[:w + 1]

# --------------------------------------------------------------------
# Public wrapper
# --------------------------------------------------------------------
@njit
def FUES(e_grid, vf, policy_1, policy_2, del_a,
         b=1e-10, m_bar=2.0, LB=4,
         endog_mbar=False, padding_mbar=0.0):
    """Fast upper-envelope scan + exact kink augmentation (API unchanged)."""
    order0 = np.argsort(e_grid)
    E0, V0 = e_grid[order0], vf[order0]
    P10, P20 = policy_1[order0], policy_2[order0]
    DA0 = del_a[order0]

    E_cl, V_cl, P1_cl, DA_cl, nan_m = _scan(
        E0, V0, P10, DA0, m_bar, LB)

    keep = ~nan_m
    Ec, Vc = E_cl[keep], V_cl[keep]
    P1c, P2c, DAc = P1_cl[keep], P20[keep], DA_cl[keep]

    jumps = _collect_jump_idx(Ec, P1c, m_bar)
    E_aug, V_aug, P1_aug, P2_aug, DA_aug = _insert_jump_kinks(
        Ec, Vc, P1c, P2c, DAc, jumps,
        m_bar, E0, V0, P10, P20)

    o = np.argsort(E_aug)
    return E_aug[o], V_aug[o], P1_aug[o], P2_aug[o], DA_aug[o]

# --------------------------------------------------------------------
# Core fast scan (unchanged logic, compact)
# --------------------------------------------------------------------
@njit
def _scan(E, V, P1, DA, m_bar, LB):
    """One-pass fast envelope scan; returns arrays + nan-mask."""
    N = E.size
    g_m_vf = np.empty(LB); g_m_da = np.empty(LB)
    g_f_vf = np.empty(LB); g_f_da = np.empty(LB)
    m_buf = np.full(LB, -1)
    m_head = 0
    nan_mask = np.zeros(N, np.bool_)

    j, k = 0, -1
    idx_base = k
    last_left = False

    for i in range(N - 2):
        if i <= 1:
            j, k = i, i - 1
            idx_base = k
            last_left = False
            continue

        de_prev = max(1e-200, E[j] - E[idx_base])
        g_prev = (V[j] - V[idx_base]) / de_prev
        de_lead = max(1e-200, E[i + 1] - E[j])
        g_lead = (V[i + 1] - V[j]) / de_lead
        g_tilde = abs((P1[i + 1] - P1[j]) / de_lead)
        right_jump = g_lead < g_prev and g_tilde > m_bar
        left_turn = g_lead > g_prev

        # ---- right-turn jump ------------------------------------
        if right_jump:
            keep = False
            for f in range(LB):
                if i + 2 + f >= N:
                    break
                de = max(1e-200, E[i + 1] - E[i + 2 + f])
                g_f_vf[f] = (V[i + 1] - V[i + 2 + f]) / de
                g_f_da[f] = abs((P1[j] - P1[i + 2 + f]) / de)
            for f in range(LB):
                if i + 2 + f >= N:
                    break
                if g_f_da[f] < m_bar and g_lead > g_f_vf[f]:
                    keep = True
                    break
            if keep:
                k, idx_base, j = j, j, i + 1
                last_left = False
            else:
                nan_mask[i + 1] = True
                m_head = circ_put(m_buf, m_head, i + 1)
            continue

        # ---- monotone / value drop ------------------------------
        if (V[i + 1] - V[j] < 0) or (g_lead < g_prev and (P1[i + 1] - P1[j]) < 0):
            nan_mask[i + 1] = True
            m_head = circ_put(m_buf, m_head, i + 1)
            continue

        # ---- left turn or gentle right --------------------------
        for t in range(LB):
            idx_buf = (m_head - 1 - t) % LB
            m_idx = m_buf[idx_buf]
            if m_idx == -1:
                g_m_da[t] = np.inf
                g_m_vf[t] = -np.inf
            else:
                de = max(1e-200, E[j] - E[m_idx])
                g_m_vf[t] = (V[j] - V[m_idx]) / de
                g_m_da[t] = abs((P1[i + 1] - P1[m_idx]) / de)

        keep_j = True
        m_ind = -1
        for t in range(LB - 1, -1, -1):
            if g_m_da[t] < m_bar:
                m_ind = int(m_buf[(m_head - 1 - t) % LB])
                if left_turn and g_lead >= g_m_vf[t] and g_tilde > m_bar:
                    keep_j = False
                break

        if not keep_j:                  # drop j – insert intersection
            nan_mask[j] = True
            idx_base = m_ind
            last_left = True
        else:                           # keep j
            if left_turn:
                if last_left:
                    nan_mask[j] = True
                last_left = True
                idx_base = m_ind if m_ind != -1 else k
            else:
                last_left = False
                idx_base = k
            k = j
            j = i + 1

    # build cleaned arrays
    keep_all = ~nan_mask
    return (
        E,
        V,
        P1,
        DA,
        nan_mask
    )
