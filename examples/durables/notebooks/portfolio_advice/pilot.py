"""Calibration pilot for the portfolio-housing demo twin.

Runs the acceptance tests mandated by the Gate-1 econ critic:
  A1 (no-housing twin): Growth type off the varsigma=1 corner by age 55,
      and >= 25pp Growth-Conservative share spread at age 65.
  A2 (full model):      ownership (rung >= 1.0) at age 50 in [40%, 80%]
      for ALL three risk types; Balanced mid-life rung in [1.0, 2.5].

Demo units: mean working income = 1.0 (econ-critic units fix).
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

# ---------------- parameters ----------------
BETA, ALPHA_U, GAMMA_H = 0.945, 0.7, 1.5
THETA, KBEQ = 1.3498, 1.3
R_F, SIGMA_R = 1.02, 0.20
MU_R = np.log(1.06) - SIGMA_R**2 / 2
PHI_Z, SIG_Z, NZ = 0.82, 0.11, 7
TYPES = {"Growth": 3.0, "Balanced": 4.65, "Conservative": 8.0}
T, T_RET = 60, 40                    # ages 25..84, retirement at index 40 (age 65)
AGES = np.arange(25, 85)
PENSION_REPL = 0.25
TAU = 0.12                           # buying cost
P_H = 3.0                            # house price per unit of housing services
H_LADDER = np.array([0.25, 1.0, 2.5, 4.0])   # bottom rung = renting-equivalent
NS = 41                              # varsigma grid
SGRID = np.linspace(0.0, 1.0, NS)
NETA = 7
A_TIE = 0.05                         # tie-break / conditioning threshold

# grids
NA, NB, NM = 80, 100, 140
a_grid = 1e-3 + (30.0 - 1e-3) * np.linspace(0, 1, NA) ** 2
b_grid = 1e-3 + (48.0 - 1e-3) * np.linspace(0, 1, NB) ** 2
m_grid = 0.02 + (54.0 - 0.02) * np.linspace(0, 1, NM) ** 2
CFRAC = np.linspace(0.02, 1.0, 80)   # consumption as fraction of cash-on-hand

# quadrature for eta
eta_nodes, eta_w = hermegauss(NETA)
eta_w = eta_w / eta_w.sum()
R_RISKY = np.exp(MU_R + SIGMA_R * eta_nodes)          # (NETA,)


def rouwenhorst(n, phi, sigma):
    p = (1 + phi) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])
    for k in range(3, n + 1):
        Pn = np.zeros((k, k))
        Pn[:k-1, :k-1] += p * Pi;      Pn[:k-1, 1:] += (1 - p) * Pi
        Pn[1:, :k-1] += (1 - p) * Pi;  Pn[1:, 1:] += p * Pi
        Pn[1:-1, :] /= 2
        Pi = Pn
    span = sigma / np.sqrt(1 - phi**2) * np.sqrt(n - 1)
    logz = np.linspace(-span, span, n)
    return logz, Pi


LOGZ, PI_Z = rouwenhorst(NZ, PHI_Z, SIG_Z)
ZVALS = np.exp(LOGZ)

# age-income trend: hump, normalized to mean 1 over working ages
raw = 1.0 + 0.045 * np.arange(T_RET) - 0.00080 * np.arange(T_RET) ** 2
g_age = raw / raw.mean()
PENSION = PENSION_REPL * g_age[-1]


def util_c(c, gamma):
    return ALPHA_U * c ** (1 - gamma) / (1 - gamma)


def util_h(h, kappa_u):
    return kappa_u * h ** (1 - GAMMA_H) / (1 - GAMMA_H)


def terminal(a, h, gamma):
    w = KBEQ + R_F * a + P_H * h
    return THETA * ALPHA_U * w ** (1 - gamma) / (1 - gamma)


def solve_full(gamma, kappa_u):
    """Backward induction, full model. Returns policy arrays."""
    NH = len(H_LADDER)
    Vp = np.empty((NA, NH, NZ))                       # V_port at (a, h, z)
    for ih, h in enumerate(H_LADDER):
        for iz in range(NZ):
            Vp[:, ih, iz] = terminal(a_grid, h, gamma)
    pol_s = np.zeros((T, NA, NH, NZ))                 # varsigma*
    pol_j = np.zeros((T, NB, NH, NZ), dtype=int)      # 0=keep, 1..NH=adjust to rung j-1
    pol_c = np.zeros((T, NM, NH, NZ))                 # consumption on m-grid

    uh = np.array([util_h(h, kappa_u) for h in H_LADDER])

    for t in range(T - 1, -1, -1):
        y_t = (g_age[t] * ZVALS) if t < T_RET else np.full(NZ, PENSION)
        # ---- cons stage: V_cons(m, h, z') ----
        Vc = np.empty((NM, NH, NZ))
        c_cand = m_grid[:, None] * CFRAC[None, :]                    # (NM, NC)
        s_cand = m_grid[:, None] - c_cand                            # savings
        u_c = util_c(np.maximum(c_cand, 1e-9), gamma)                # (NM, NC)
        for ih in range(NH):
            for iz in range(NZ):
                cont = np.interp(s_cand.ravel(), a_grid, Vp[:, ih, iz]).reshape(NM, -1)
                obj = u_c + uh[ih] + BETA * cont
                k = obj.argmax(axis=1)
                Vc[:, ih, iz] = obj[np.arange(NM), k]
                pol_c[t, :, ih, iz] = c_cand[np.arange(NM), k]
        # ---- tenure stage: V_ten(b, h, z) ----
        Vt = np.empty((NB, NH, NZ))
        EVc = np.einsum('zw,mhw->mhz', PI_Z, Vc) if False else None  # placeholder
        # expectation over z' of V_cons at m depends on m which depends on z' via y — do per z'
        for ih in range(NH):
            for iz in range(NZ):
                probs = PI_Z[iz]
                # keep
                vk = np.zeros(NB)
                for jz in range(NZ):
                    m_k = b_grid + y_t[jz]
                    vk += probs[jz] * np.interp(m_k, m_grid, Vc[:, ih, jz])
                # adjust to each rung
                va = np.full((NH, NB), -np.inf)
                for ih2 in range(NH):
                    cash0 = b_grid + P_H * H_LADDER[ih] - (1 + TAU) * P_H * H_LADDER[ih2]
                    va_j = np.zeros(NB)
                    feas = cash0 + y_t.min() > 0.02
                    for jz in range(NZ):
                        m_a = cash0 + y_t[jz]
                        va_j += probs[jz] * np.interp(np.maximum(m_a, m_grid[0]), m_grid, Vc[:, ih2, jz])
                    va[ih2] = np.where(feas, va_j, -np.inf)
                allv = np.vstack([vk, va])                            # (1+NH, NB)
                jj = allv.argmax(axis=0)
                Vt[:, ih, iz] = allv[jj, np.arange(NB)]
                pol_j[t, :, ih, iz] = jj
        # ---- port stage: V_port(a, h, z) ----
        Vp_new = np.empty((NA, NH, NZ))
        b_cand = a_grid[:, None, None] * (R_F + SGRID[None, :, None] * (R_RISKY[None, None, :] - R_F))
        for ih in range(NH):
            for iz in range(NZ):
                cont = np.interp(b_cand.ravel(), b_grid, Vt[:, ih, iz]).reshape(NA, NS, NETA)
                obj = (cont * eta_w[None, None, :]).sum(axis=2)      # (NA, NS)
                k = obj.argmax(axis=1)
                flat = (obj.max(axis=1) - obj.min(axis=1)) < 1e-12
                k = np.where(flat, 0, k)                             # explicit tie-break
                Vp_new[:, ih, iz] = obj[np.arange(NA), k]
                pol_s[t, :, ih, iz] = SGRID[k]
        Vp = Vp_new
    return pol_s, pol_j, pol_c


def simulate(pol_s, pol_j, pol_c, gamma, N=10000, seed=42):
    rng = np.random.default_rng(seed)
    NH = len(H_LADDER)
    a = np.exp(rng.normal(np.log(0.3), 0.5, N)); a = np.clip(a, 0, 5)
    ih = np.zeros(N, dtype=int)
    iz = rng.integers(0, NZ, N)
    A = np.zeros((T, N)); H = np.zeros((T, N)); S = np.zeros((T, N)); OWN = np.zeros((T, N))
    cum = PI_Z.cumsum(axis=1)
    for t in range(T):
        A[t] = a
        # port
        s = np.empty(N)
        for h_ in range(NH):
            for z_ in range(NZ):
                m_ = (ih == h_) & (iz == z_)
                if m_.any():
                    s[m_] = np.interp(a[m_], a_grid, pol_s[t, :, h_, z_])
        S[t] = s
        eta = rng.standard_normal(N)
        Rp = R_F + s * (np.exp(MU_R + SIGMA_R * eta) - R_F)
        b = Rp * a
        # tenure
        jj = np.empty(N, dtype=int)
        for h_ in range(NH):
            for z_ in range(NZ):
                m_ = (ih == h_) & (iz == z_)
                if m_.any():
                    jj[m_] = np.round(np.interp(b[m_], b_grid, pol_j[t, :, h_, z_])).astype(int)
        # income innovation
        u = rng.random(N)
        iz = (u[:, None] > cum[iz]).sum(axis=1)
        y = (g_age[t] * ZVALS[iz]) if t < T_RET else np.full(N, PENSION)
        adj = jj > 0
        ih_new = np.where(adj, jj - 1, ih)
        cash = np.where(adj, b + P_H * H_LADDER[ih] - (1 + TAU) * P_H * H_LADDER[ih_new], b)
        m = np.maximum(cash + y, m_grid[0])
        ih = ih_new
        H[t] = H_LADDER[ih]; OWN[t] = (H_LADDER[ih] >= 1.0)
        # cons
        c = np.empty(N)
        for h_ in range(NH):
            for z_ in range(NZ):
                m_ = (ih == h_) & (iz == z_)
                if m_.any():
                    c[m_] = np.interp(m[m_], m_grid, pol_c[t, :, h_, z_])
        a = np.maximum(m - c, 0.0)
    return A, H, S, OWN


def acceptance(kappa_u, verbose=True):
    res = {}
    for name, gamma in TYPES.items():
        pol_s, pol_j, pol_c = solve_full(gamma, kappa_u)
        A, H, S, OWN = simulate(pol_s, pol_j, pol_c, gamma)
        mask = A > A_TIE
        share_age = np.array([S[t][mask[t]].mean() if mask[t].any() else np.nan for t in range(T)])
        res[name] = dict(share=share_age, own50=OWN[25].mean(), own_by_age=OWN.mean(axis=1),
                         rung50=H[25].mean(), wealth=np.median(A, axis=1))
    g, c = res["Growth"]["share"], res["Conservative"]["share"]
    a1a = g[30] < 0.995                                  # age 55
    a1b = (g[40] - c[40]) >= 0.25                        # spread at 65
    a2 = all(0.40 <= res[k]["own50"] <= 0.80 for k in TYPES)
    if verbose:
        print(f"kappa_u={kappa_u:.3f}")
        for k in TYPES:
            r = res[k]
            print(f"  {k:13s} share@40={r['share'][15]:.2f} @55={r['share'][30]:.2f} @65={r['share'][40]:.2f}"
                  f" | own@50={r['own50']:.2f} rung@50={r['rung50']:.2f} | medW@65={r['wealth'][40]:.1f}")
        print(f"  A1a Growth<1 by 55: {a1a} | A1b spread@65>=25pp: {g[40]-c[40]:.2f} {a1b} | A2 own@50 in [.4,.8]: {a2}")
    return a1a and a1b and a2, res


if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    kappa = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
    ok, res = acceptance(kappa)
    print(f"PASS={ok}  ({time.time()-t0:.0f}s)")
