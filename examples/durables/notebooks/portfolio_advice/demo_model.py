"""Demo-scale twin of the durables model extended with a portfolio choice.

Each year is a sequence of three decision steps, mirroring the production
blueprint (see PLAN.md and port.yaml in this folder):

    portfolio step -> housing step -> spending step

and the solver works backwards from retirement in the reverse order
(spending, housing, portfolio), exactly as the production stack orders its
backward waves. One function per step; the arrays each step passes to the
next carry the same objects the production stages export (values on the
incoming state grid).

Units: mean working income = 1. All wealth and house values are multiples
of average annual earnings.
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

# ---------------------------------------------------------------------------
# Calibration (frozen at Gate 1; see PLAN.md section 3 for provenance)
# ---------------------------------------------------------------------------
BETA = 0.945          # discount factor            (durables baseline)
ALPHA_U = 0.7         # utility weight on consumption (durables baseline)
GAMMA_H = 1.5         # curvature over housing services (durables baseline)
THETA, KBEQ = 1.3498, 1.3   # terminal-value weight and shifter (durables baseline)
KAPPA_U = 0.06        # housing utility weight     (pilot-tuned, ownership 45-58% at 50)
R_F = 1.02            # gross risk-free return
SIGMA_R = 0.20        # risky-return volatility
MU_R = np.log(1.06) - SIGMA_R ** 2 / 2   # E[gross risky return] = 1.06
PHI_Z, SIG_Z, NZ = 0.82, 0.11, 7         # log-income AR(1) (durables baseline), Rouwenhorst-7
PENSION_REPL = 0.25   # pension replacement rate on final working trend income
TAU = 0.12            # proportional transaction cost on purchase (durables baseline)
P_H = 3.0             # house price per unit of annual housing services
H_LADDER = np.array([0.25, 1.0, 2.5, 4.0])   # rent-equivalent, starter, family, premium
OWN_RUNG = 1.0        # ownership = rung >= 1.0

T, T_RET = 60, 40     # ages 25..84; retirement at age 65
AGES = np.arange(25, 85)

TYPES = {"Growth": 3.0, "Balanced": 4.65, "Conservative": 8.0}   # CRRA by profile

# numerical settings
NS, NETA = 41, 7
SGRID = np.linspace(0.0, 1.0, NS)
NA, NB, NM, NC = 80, 100, 140, 80
A_TIE = 0.05          # tie-break / conditioning threshold for share statistics

a_grid = 1e-3 + (30.0 - 1e-3) * np.linspace(0, 1, NA) ** 2
b_grid = 1e-3 + (67.0 - 1e-3) * np.linspace(0, 1, NB) ** 2   # covers a_max * max risky return
m_grid = 0.02 + (54.0 - 0.02) * np.linspace(0, 1, NM) ** 2
CFRAC = np.linspace(0.02, 1.0, NC)

_eta_nodes, _eta_w = hermegauss(NETA)
ETA_W = _eta_w / _eta_w.sum()
R_RISKY = np.exp(MU_R + SIGMA_R * _eta_nodes)

# S&P Target Date glide path (equity share by current age), transcribed from
# "S&P Target Date glidepath.xlsx", Life-Cycle-Prime-Time repository
# (Econ-ARK / T. Rowe Price project). Source: S&P Target Date indices.
SP_AGES = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
SP_EQUITY = np.array([0.912, 0.903, 0.900, 0.885, 0.837, 0.756, 0.649,
                      0.535, 0.453, 0.406, 0.356, 0.330, 0.321])

# US homeownership rate by age bracket (owner-occupied share), approximate
# published pattern (US Census Bureau, Housing Vacancies and Homeownership,
# recent years, rounded).
CENSUS_AGE_MID = np.array([30, 40, 50, 60, 72])
CENSUS_OWN = np.array([0.39, 0.62, 0.70, 0.75, 0.79])


def rouwenhorst(n, phi, sigma):
    p = (1 + phi) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])
    for k in range(3, n + 1):
        Pn = np.zeros((k, k))
        Pn[:k-1, :k-1] += p * Pi;      Pn[:k-1, 1:] += (1 - p) * Pi
        Pn[1:, :k-1] += (1 - p) * Pi;  Pn[1:, 1:] += p * Pi
        Pn[1:-1, :] /= 2
        Pi = Pn
    span = sigma / np.sqrt(1 - phi ** 2) * np.sqrt(n - 1)
    return np.linspace(-span, span, n), Pi


LOGZ, PI_Z = rouwenhorst(NZ, PHI_Z, SIG_Z)
ZVALS = np.exp(LOGZ)
_raw = 1.0 + 0.045 * np.arange(T_RET) - 0.00080 * np.arange(T_RET) ** 2
G_AGE = _raw / _raw.mean()            # age-earnings trend, mean 1 over working life
PENSION = PENSION_REPL * G_AGE[-1]

NH = len(H_LADDER)


def util_c(c, gamma):
    return ALPHA_U * c ** (1 - gamma) / (1 - gamma)


def util_h():
    return KAPPA_U * H_LADDER ** (1 - GAMMA_H) / (1 - GAMMA_H)


def terminal_value(gamma):
    """Terminal (bequest-like) value on the (a, h, z) grid.

    Terminal assets earn the deterministic risk-free return: the portfolio
    step does not run at the terminal instant (Gate-1 review, defect D1)."""
    V = np.empty((NA, NH, NZ))
    for ih, h in enumerate(H_LADDER):
        w = KBEQ + R_F * a_grid + P_H * h
        V[:, ih, :] = (THETA * ALPHA_U * w ** (1 - gamma) / (1 - gamma))[:, None]
    return V


def income(t):
    """Income by state for year t: age trend x persistent state, or pension."""
    return (G_AGE[t] * ZVALS) if t < T_RET else np.full(NZ, PENSION)


# ---------------------------------------------------------------------------
# The three decision steps (backward operators)
# ---------------------------------------------------------------------------

def spending_step(Vp_next, t, gamma):
    """Consumption choice on cash-on-hand m, given next year's portfolio-step
    value on the savings grid. Returns V_spend(m, h, z') and the consumption
    policy."""
    Vc = np.empty((NM, NH, NZ))
    pol_c = np.empty((NM, NH, NZ))
    c_cand = m_grid[:, None] * CFRAC[None, :]
    s_cand = m_grid[:, None] - c_cand
    u_c = util_c(np.maximum(c_cand, 1e-9), gamma)
    uh = util_h()
    for ih in range(NH):
        for iz in range(NZ):
            cont = np.interp(s_cand.ravel(), a_grid, Vp_next[:, ih, iz]).reshape(NM, NC)
            obj = u_c + uh[ih] + BETA * cont
            k = obj.argmax(axis=1)
            Vc[:, ih, iz] = obj[np.arange(NM), k]
            pol_c[:, ih, iz] = c_cand[np.arange(NM), k]
    return Vc, pol_c


def housing_step(Vc, t):
    """Keep-or-move choice on post-return balance b. Timing mirrors the
    production design: the keep/move decision is made before the year's
    income realises, but a mover picks the destination rung AFTER income is
    known (the adjuster chooses housing with cash in hand). Returns
    V_house(b, h, z), the keep/move policy (0/1) on (b, h, z), and the
    destination-rung policy on (b, h, realised z')."""
    y_t = income(t)
    Vt = np.empty((NB, NH, NZ))
    pol_j = np.empty((NB, NH, NZ), dtype=int)
    pol_rung = np.empty((NB, NH, NZ), dtype=int)
    for ih in range(NH):
        # mover's ex-post problem: best rung given realised income state z'
        va_post = np.empty((NB, NZ))
        for jz in range(NZ):
            cand = np.full((NH, NB), -np.inf)
            for ih2 in range(NH):
                cash0 = b_grid + P_H * H_LADDER[ih] - (1 + TAU) * P_H * H_LADDER[ih2]
                m_a = np.maximum(cash0 + y_t[jz], m_grid[0])
                cand[ih2] = np.where(cash0 + y_t[jz] > 0.02,
                                     np.interp(m_a, m_grid, Vc[:, ih2, jz]), -np.inf)
            pol_rung[:, ih, jz] = cand.argmax(axis=0)
            va_post[:, jz] = cand.max(axis=0)
        for iz in range(NZ):
            probs = PI_Z[iz]
            vk = np.zeros(NB)
            for jz in range(NZ):
                vk += probs[jz] * np.interp(b_grid + y_t[jz], m_grid, Vc[:, ih, jz])
            va = va_post @ probs
            move = va > vk
            Vt[:, ih, iz] = np.where(move, va, vk)
            pol_j[:, ih, iz] = move.astype(int)
    return Vt, pol_j, pol_rung


def portfolio_step(Vt, t, forced_share=None):
    """Risky-share choice on assets a; the return shock realises after the
    choice, so the expectation sits inside the maximisation. Returns
    V_port(a, h, z) and the share policy.

    forced_share pins the share (used for the benchmark glide-path solve)."""
    Vp = np.empty((NA, NH, NZ))
    pol_s = np.empty((NA, NH, NZ))
    if forced_share is None:
        b_cand = a_grid[:, None, None] * (R_F + SGRID[None, :, None] * (R_RISKY[None, None, :] - R_F))
        for ih in range(NH):
            for iz in range(NZ):
                cont = np.interp(b_cand.ravel(), b_grid, Vt[:, ih, iz]).reshape(NA, NS, NETA)
                obj = cont @ ETA_W
                k = obj.argmax(axis=1)
                flat = (obj.max(axis=1) - obj.min(axis=1)) < 1e-12
                k = np.where(flat, 0, k)          # explicit tie-break at a ~ 0
                Vp[:, ih, iz] = obj[np.arange(NA), k]
                pol_s[:, ih, iz] = SGRID[k]
    else:
        b_cand = a_grid[:, None] * (R_F + forced_share * (R_RISKY[None, :] - R_F))
        for ih in range(NH):
            for iz in range(NZ):
                cont = np.interp(b_cand.ravel(), b_grid, Vt[:, ih, iz]).reshape(NA, NETA)
                Vp[:, ih, iz] = cont @ ETA_W
                pol_s[:, ih, iz] = forced_share
    return Vp, pol_s


def solve_lifecycle(gamma, forced_share_path=None):
    """Backward induction over the three steps. Returns policies and the
    age-25 value function."""
    Vp = terminal_value(gamma)
    pol = {"share": np.empty((T, NA, NH, NZ)),
           "branch": np.empty((T, NB, NH, NZ), dtype=int),
           "rung": np.empty((T, NB, NH, NZ), dtype=int),
           "cons": np.empty((T, NM, NH, NZ))}
    for t in range(T - 1, -1, -1):
        Vc, pc = spending_step(Vp, t, gamma)
        Vt, pj, pr = housing_step(Vc, t)
        fs = None if forced_share_path is None else forced_share_path[t]
        Vp, ps = portfolio_step(Vt, t, forced_share=fs)
        pol["share"][t], pol["branch"][t] = ps, pj
        pol["rung"][t], pol["cons"][t] = pr, pc
    return pol, Vp


# ---------------------------------------------------------------------------
# Forward simulation (mirrors the timing of the backward solve)
# ---------------------------------------------------------------------------

def simulate(pol, N=10000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.clip(np.exp(rng.normal(np.log(0.3), 0.5, N)), 0, 5)
    ih = np.zeros(N, dtype=int)
    iz = rng.integers(0, NZ, N)
    out = {k: np.zeros((T, N)) for k in ("assets", "house", "share", "own", "cons")}
    cum = PI_Z.cumsum(axis=1)
    cum[:, -1] = 1.0
    for t in range(T):
        out["assets"][t] = a
        s = np.empty(N)
        for h_ in range(NH):
            for z_ in range(NZ):
                m_ = (ih == h_) & (iz == z_)
                if m_.any():
                    s[m_] = np.interp(a[m_], a_grid, pol["share"][t, :, h_, z_])
        out["share"][t] = s
        eta = rng.standard_normal(N)
        b = (R_F + s * (np.exp(MU_R + SIGMA_R * eta) - R_F)) * a
        jj = np.empty(N, dtype=int)
        ib = np.searchsorted(b_grid, b).clip(0, NB - 1)
        for h_ in range(NH):
            for z_ in range(NZ):
                m_ = (ih == h_) & (iz == z_)
                if m_.any():
                    jj[m_] = pol["branch"][t, ib[m_], h_, z_]
        u = rng.random(N)
        iz = (u[:, None] > cum[iz]).sum(axis=1)
        y = (G_AGE[t] * ZVALS[iz]) if t < T_RET else np.full(N, PENSION)
        adj = jj > 0
        ih_new = ih.copy()
        for h_ in range(NH):
            for z_ in range(NZ):
                m_ = adj & (ih == h_) & (iz == z_)
                if m_.any():
                    ih_new[m_] = pol["rung"][t, ib[m_], h_, z_]
        cash = np.where(adj, b + P_H * H_LADDER[ih] - (1 + TAU) * P_H * H_LADDER[ih_new], b)
        m = np.maximum(cash + y, m_grid[0])
        ih = ih_new
        out["house"][t] = H_LADDER[ih]
        out["own"][t] = H_LADDER[ih] >= OWN_RUNG
        c = np.empty(N)
        for h_ in range(NH):
            for z_ in range(NZ):
                m_ = (ih == h_) & (iz == z_)
                if m_.any():
                    c[m_] = np.interp(m[m_], m_grid, pol["cons"][t, :, h_, z_])
        out["cons"][t] = c
        a = np.maximum(m - c, 0.0)
    return out


def share_profile(panel):
    """Mean risky share by age, conditional on assets above the tie-break
    threshold (the share is undefined for an empty portfolio)."""
    A, S = panel["assets"], panel["share"]
    return np.array([S[t][A[t] > A_TIE].mean() if (A[t] > A_TIE).any() else np.nan
                     for t in range(T)])


def sp_path_by_year():
    """S&P glide path interpolated onto the model's age grid, as a forced
    share path for the benchmark solve."""
    return np.interp(AGES, SP_AGES, SP_EQUITY)


def value_of_guidance(gamma, pol_opt_V0, N=10000, seed=42):
    """Certainty-equivalent consumption gain of the model-optimal share rule
    over the S&P glide path, holding spending and housing rules optimal in
    both cases. Evaluated at the age-25 initial distribution."""
    _, V0_bench = solve_lifecycle(gamma, forced_share_path=sp_path_by_year())
    rng = np.random.default_rng(seed)
    a0 = np.clip(np.exp(rng.normal(np.log(0.3), 0.5, N)), 0, 5)
    iz0 = rng.integers(0, NZ, N)
    v_opt = np.array([np.interp(a0[i], a_grid, pol_opt_V0[:, 0, iz0[i]]) for i in range(N)]).mean()
    v_ben = np.array([np.interp(a0[i], a_grid, V0_bench[:, 0, iz0[i]]) for i in range(N)]).mean()
    lam = (v_opt / v_ben) ** (1.0 / (1.0 - gamma)) - 1.0
    return lam


# ---------------------------------------------------------------------------
# Decision-blueprint figure (drawn from the reviewed period-level graph:
# nodes port / tenure / keeper / adjuster, fan-out, fan-in, two shocks)
# ---------------------------------------------------------------------------

def draw_blueprint(ax):
    """One year of decisions as an auditable flow diagram."""
    ink, sub = "#0b0b0b", "#52514e"
    box = dict(boxstyle="round,pad=0.42", fc="#f4f4f2", ec=sub, lw=1.2)
    hot = dict(boxstyle="round,pad=0.42", fc="#e8f0fb", ec="#2a78d6", lw=1.6)

    def node(x, y, text, style, fs=10):
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color=ink, bbox=style, zorder=3)

    def arrow(x0, y0, x1, y1, text=None, dy=0.055, color=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color or sub, lw=1.4))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center",
                    fontsize=8.2, color=sub, style="italic")

    node(0.09, 0.50, "start of year\nsavings $a$,\nhome $h$", box, 9)
    node(0.32, 0.50, "portfolio step\nchoose risky share $\\varsigma$", hot)
    node(0.56, 0.50, "housing step\nkeep or move", box)
    node(0.80, 0.72, "spending step\n(kept home)", box, 9)
    node(0.80, 0.28, "spending step\n(new home)", box, 9)
    node(0.97, 0.50, "next\nyear", box, 9)

    arrow(0.15, 0.50, 0.24, 0.50)
    arrow(0.40, 0.50, 0.48, 0.50, "market return\nrealizes", 0.10)
    arrow(0.62, 0.56, 0.72, 0.70, "keep")
    arrow(0.62, 0.44, 0.72, 0.30, "move")
    arrow(0.87, 0.70, 0.95, 0.55)
    arrow(0.87, 0.30, 0.95, 0.45)
    ax.text(0.56, 0.30, "income for the year\nrealizes here", ha="center",
            fontsize=8.2, color=sub, style="italic")
    ax.text(0.32, 0.20, "new step: one specification file\n+ a solver entry + 3 parameters", ha="center",
            fontsize=8.6, color="#2a78d6")
    ax.set_xlim(0, 1.03); ax.set_ylim(0.05, 0.95); ax.axis("off")
