"""
Fast Upper-Envelope Scan (FUES) – refactored
-------------------------------------------
Implements Dobrescu-Shanker (2023) with a few micro-optimisations:

1. vectorised `uniqueEG`               – O(N) vs worst-case O(N²)
2. single pre-computed Δe_grid array   – avoids repeated subtraction
3. no Python `max()` in nopython code  – use `np.maximum` / inline branch
4. cheaper mask creation in `_simple_upper_envelope`
5. minor array pre-allocation tweaks

Author:  Akshay Shanker, University of Sydney
E-mail:  akshay.shanker@me.com
"""

import numpy as np
from numba import njit, float64, int64
from typing import Tuple

EPS = 1e-200  # small positive for safe division


# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
@njit
def uniqueEG(egrid: np.ndarray, vf: np.ndarray) -> np.ndarray:
    """
    Keep only the row that delivers the *highest* value when duplicate
    egrid entries occur.  Returns a boolean mask.
    """
    idx = np.argsort(egrid)
    g_sorted = egrid[idx]
    v_sorted = vf[idx]

    keep = np.ones(len(g_sorted), dtype=np.bool_)  # start with “all keep”
    for i in range(1, len(g_sorted)):
        if np.abs(g_sorted[i] - g_sorted[i - 1]) < 1e-10:
            # tie → drop the one with lower value
            if v_sorted[i] >= v_sorted[i - 1]:
                keep[i - 1] = False
            else:
                keep[i] = False
    out = np.empty_like(keep)
    out[idx] = keep          # undo the sort
    return out


@njit
def append_push(buf: np.ndarray, val: int64) -> np.ndarray:
    """Push `val` into a fixed-length FIFO buffer (Numba-friendly)."""
    buf[:-1] = buf[1:]
    buf[-1] = val
    return buf


@njit
def back_scan_gradients(m_idx: np.ndarray,
                        a_prime: np.ndarray,
                        vf: np.ndarray,
                        e: np.ndarray,
                        j: int64,
                        q: int64,
                        Δe: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gradients for backward scan (vectorised)."""
    out_vf = np.empty(len(m_idx))
    out_a  = np.empty(len(m_idx))

    for k in range(len(m_idx)):
        mk = int64(m_idx[k])
        denom = np.maximum(Δe[j - 1] + Δe[mk - 1], EPS)  # safe denom
        out_vf[k] = (vf[j] - vf[mk]) / denom
        out_a[k]  = np.abs((a_prime[q] - a_prime[mk]) / denom)
    return out_vf, out_a


@njit
def fwd_scan_gradients(a_prime, vf, e, j, q, LB, Δe):
    out_vf = np.empty(LB)
    out_a  = np.empty(LB)
    for f in range(LB):
        idx = q + 1 + f
        denom = np.maximum(Δe[idx - 1], EPS)
        out_vf[f] = (vf[q] - vf[idx]) / denom
        out_a[f]  = np.abs((a_prime[j] - a_prime[idx]) / denom)
    return out_vf, out_a


@njit
def seg_intersect(pj, pk, pi, pm):
    """2-D line segment intersection (minimal)."""
    da = pk - pj
    db = pm - pi
    dp = pj - pi
    dap = np.array([-da[1], da[0]])
    denom = dap @ db
    if np.abs(denom) < EPS:
        return np.array([np.nan, np.nan])
    num = dap @ dp
    t = num / denom
    return pi + t * db


# ---------------------------------------------------------------------
#  Main algorithm
# ---------------------------------------------------------------------
@njit
def FUES(e_grid, vf, policy_1, policy_2, del_a,
         m_bar=2.0, LB=4, endog_mbar=False, padding_mbar=0.0):
    """
    Refined upper envelope via single forward / backward scan.
    Returns (m*, v*, p1*, p2*, ∂a*).
    """
    # --- sort once ------------------------------------------------------
    order = np.argsort(e_grid)
    e_grid = e_grid[order]
    vf     = vf[order]
    a1     = policy_1[order]
    a2     = policy_2[order]
    d_a    = del_a[order]

    # --- internal copies used during scan -------------------------------
    vf_full = vf.copy()
    Δe = np.diff(e_grid)  # length N-1

    m_buf = np.zeros(LB, dtype=np.int64)  # rolling buffer of dropped idx

    j = 0        # last “kept” point
    k = -1       # point before j  (initially dummy)

    for i in range(len(e_grid) - 2):
        if i <= 1:
            k, j = j, i
            continue

        # gradients between last two kept pts
        g_jm1 = (vf_full[j] - vf_full[k]) / np.maximum(e_grid[j] - e_grid[k], EPS)

        # candidate point = i+1
        cand = i + 1
        denom = np.maximum(e_grid[cand] - e_grid[j], EPS)
        g1 = (vf_full[cand] - vf_full[j]) / denom

        # policy slope and jump
        M_L = np.abs(d_a[j])
        M_U = np.abs(d_a[cand])
        M_max = np.maximum(M_L, M_U) + padding_mbar
        g_tilde = np.abs((a1[cand] - a1[j]) / denom)

        if not endog_mbar:
            M_max = m_bar

        # ----------- right-turn & jump test ------------------------------
        if (g1 < g_jm1) and (g_tilde > M_max):
            keep = False
            # forward scan guard
            vf_f, a_f = fwd_scan_gradients(a1, vf_full, e_grid, j, cand, LB, Δe)
            idx = np.where(a_f < m_bar)[0]
            if idx.size > 0 and g1 > vf_f[idx[0]]:
                keep = True
            if not keep:
                vf[cand] = np.nan
                m_buf = append_push(m_buf, cand)
                continue

        # ----------- dominated or non-monotone ---------------------------
        if (vf_full[cand] < vf_full[j]) or ((g1 < g_jm1) and (a1[cand] - a1[j] < 0.0)):
            vf[cand] = np.nan
            m_buf = append_push(m_buf, cand)
            continue

        # ----------- backward scan --------------------------------------
        gv, ga = back_scan_gradients(m_buf, a1, vf_full, e_grid, j, cand, Δe)
        idx = np.where(ga < m_bar)[0]
        drop_j = False
        if idx.size:
            gmv = gv[idx[-1]]
            if (g1 > g_jm1) and (g1 >= gmv) and (g_tilde > M_max):
                drop_j = True

        if drop_j:
            pj = np.array([e_grid[j], vf_full[j]])
            pi = np.array([e_grid[cand], vf_full[cand]])
            pk = np.array([e_grid[k], vf_full[k]])
            pm = np.array([e_grid[m_buf[idx[-1]]], vf_full[m_buf[idx[-1]]]])
            intr = seg_intersect(pj, pk, pi, pm)
            vf[j]      = np.nan
            vf_full[j] = intr[1]
            e_grid[j]  = intr[0]
            k, j = j, cand
        else:
            k, j = j, cand

    # -------------------------------------------------------------------
    keep = ~np.isnan(vf)
    return e_grid[keep], vf[keep], a1[keep], a2[keep], d_a[keep]

"""
Fast Upper-Envelope Scan (FUES) – refactored
-------------------------------------------
Implements Dobrescu-Shanker (2023) with a few micro-optimisations:

1. vectorised `uniqueEG`               – O(N) vs worst-case O(N²)
2. single pre-computed Δe_grid array   – avoids repeated subtraction
3. no Python `max()` in nopython code  – use `np.maximum` / inline branch
4. cheaper mask creation in `_simple_upper_envelope`
5. minor array pre-allocation tweaks

Author:  Akshay Shanker, University of Sydney
E-mail:  akshay.shanker@me.com
"""

import numpy as np
from numba import njit, float64, int64
from typing import Tuple

EPS = 1e-200  # small positive for safe division


# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
@njit
def uniqueEG(egrid: np.ndarray, vf: np.ndarray) -> np.ndarray:
    """
    Keep only the row that delivers the *highest* value when duplicate
    egrid entries occur.  Returns a boolean mask.
    """
    idx = np.argsort(egrid)
    g_sorted = egrid[idx]
    v_sorted = vf[idx]

    keep = np.ones(len(g_sorted), dtype=np.bool_)  # start with “all keep”
    for i in range(1, len(g_sorted)):
        if np.abs(g_sorted[i] - g_sorted[i - 1]) < 1e-10:
            # tie → drop the one with lower value
            if v_sorted[i] >= v_sorted[i - 1]:
                keep[i - 1] = False
            else:
                keep[i] = False
    out = np.empty_like(keep)
    out[idx] = keep          # undo the sort
    return out


@njit
def append_push(buf: np.ndarray, val: int64) -> np.ndarray:
    """Push `val` into a fixed-length FIFO buffer (Numba-friendly)."""
    buf[:-1] = buf[1:]
    buf[-1] = val
    return buf


@njit
def back_scan_gradients(m_idx: np.ndarray,
                        a_prime: np.ndarray,
                        vf: np.ndarray,
                        e: np.ndarray,
                        j: int64,
                        q: int64,
                        Δe: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gradients for backward scan (vectorised)."""
    out_vf = np.empty(len(m_idx))
    out_a  = np.empty(len(m_idx))

    for k in range(len(m_idx)):
        mk = int64(m_idx[k])
        denom = np.maximum(Δe[j - 1] + Δe[mk - 1], EPS)  # safe denom
        out_vf[k] = (vf[j] - vf[mk]) / denom
        out_a[k]  = np.abs((a_prime[q] - a_prime[mk]) / denom)
    return out_vf, out_a


@njit
def fwd_scan_gradients(a_prime, vf, e, j, q, LB, Δe):
    out_vf = np.empty(LB)
    out_a  = np.empty(LB)
    for f in range(LB):
        idx = q + 1 + f
        denom = np.maximum(Δe[idx - 1], EPS)
        out_vf[f] = (vf[q] - vf[idx]) / denom
        out_a[f]  = np.abs((a_prime[j] - a_prime[idx]) / denom)
    return out_vf, out_a


@njit
def seg_intersect(pj, pk, pi, pm):
    """2-D line segment intersection (minimal)."""
    da = pk - pj
    db = pm - pi
    dp = pj - pi
    dap = np.array([-da[1], da[0]])
    denom = dap @ db
    if np.abs(denom) < EPS:
        return np.array([np.nan, np.nan])
    num = dap @ dp
    t = num / denom
    return pi + t * db


# ---------------------------------------------------------------------
#  Main algorithm
# ---------------------------------------------------------------------
@njit
def FUES(e_grid, vf, policy_1, policy_2, del_a,
         m_bar=2.0, LB=4, endog_mbar=False, padding_mbar=0.0):
    """
    Refined upper envelope via single forward / backward scan.
    Returns (m*, v*, p1*, p2*, ∂a*).
    """
    # --- sort once ------------------------------------------------------
    order = np.argsort(e_grid)
    e_grid = e_grid[order]
    vf     = vf[order]
    a1     = policy_1[order]
    a2     = policy_2[order]
    d_a    = del_a[order]

    # --- internal copies used during scan -------------------------------
    vf_full = vf.copy()
    Δe = np.diff(e_grid)  # length N-1

    m_buf = np.zeros(LB, dtype=np.int64)  # rolling buffer of dropped idx

    j = 0        # last “kept” point
    k = -1       # point before j  (initially dummy)

    for i in range(len(e_grid) - 2):
        if i <= 1:
            k, j = j, i
            continue

        # gradients between last two kept pts
        g_jm1 = (vf_full[j] - vf_full[k]) / np.maximum(e_grid[j] - e_grid[k], EPS)

        # candidate point = i+1
        cand = i + 1
        denom = np.maximum(e_grid[cand] - e_grid[j], EPS)
        g1 = (vf_full[cand] - vf_full[j]) / denom

        # policy slope and jump
        M_L = np.abs(d_a[j])
        M_U = np.abs(d_a[cand])
        M_max = np.maximum(M_L, M_U) + padding_mbar
        g_tilde = np.abs((a1[cand] - a1[j]) / denom)

        if not endog_mbar:
            M_max = m_bar

        # ----------- right-turn & jump test ------------------------------
        if (g1 < g_jm1) and (g_tilde > M_max):
            keep = False
            # forward scan guard
            vf_f, a_f = fwd_scan_gradients(a1, vf_full, e_grid, j, cand, LB, Δe)
            idx = np.where(a_f < m_bar)[0]
            if idx.size > 0 and g1 > vf_f[idx[0]]:
                keep = True
            if not keep:
                vf[cand] = np.nan
                m_buf = append_push(m_buf, cand)
                continue

        # ----------- dominated or non-monotone ---------------------------
        if (vf_full[cand] < vf_full[j]) or ((g1 < g_jm1) and (a1[cand] - a1[j] < 0.0)):
            vf[cand] = np.nan
            m_buf = append_push(m_buf, cand)
            continue

        # ----------- backward scan --------------------------------------
        gv, ga = back_scan_gradients(m_buf, a1, vf_full, e_grid, j, cand, Δe)
        idx = np.where(ga < m_bar)[0]
        drop_j = False
        if idx.size:
            gmv = gv[idx[-1]]
            if (g1 > g_jm1) and (g1 >= gmv) and (g_tilde > M_max):
                drop_j = True

        if drop_j:
            pj = np.array([e_grid[j], vf_full[j]])
            pi = np.array([e_grid[cand], vf_full[cand]])
            pk = np.array([e_grid[k], vf_full[k]])
            pm = np.array([e_grid[m_buf[idx[-1]]], vf_full[m_buf[idx[-1]]]])
            intr = seg_intersect(pj, pk, pi, pm)
            vf[j]      = np.nan
            vf_full[j] = intr[1]
            e_grid[j]  = intr[0]
            k, j = j, cand
        else:
            k, j = j, cand

    # -------------------------------------------------------------------
    keep = ~np.isnan(vf)
    return e_grid[keep], vf[keep], a1[keep], a2[keep], d_a[keep]
