"""stageb_scalars_iv.py -- the stage-B scalars that were still floating point, in interval (Arb ball)
arithmetic, and the FIRST assembly of the acceptance that INCLUDES the tail term.

WHY THIS SCRIPT EXISTS
----------------------
QUEUE.md D1 (line 212) lists "four scalars presently in floating point that need interval forms".
Those four (m_p, gamma_p, d_w, pi_0 of Proposition T3) were in fact certified on 2026-09-09 by
t3_balls.py / T3_BALLS.md (delta <= 8.017047e-03 in balls), so D1 is stale on that item.  What is
NOT yet assembled rigorously -- and is flagged three times in ropnorm.py's own output and once in
QUEUE (FINDING 2026-09-10 ~22:00) -- is the TAIL:

    ropnorm_final.log, ASSEMBLY line: "the tail term tau_N tau'_N / d_N and the columns beyond
    N_cur are EXCLUDED"; accept_iv.py evaluates the same tail-free formula in balls.

So the acceptance quantity beta = 0.198178 quoted everywhere is a bound with one summand missing.
This script supplies it:

  (1) tau_N and tau'_N at N = 16000 (the truncation the completed rectangle reaches) from
      Proposition T2 (STAGEB_T2.md Sec. 3, formula (T2)), evaluated with the rigorous constants of
      stageb_consts.log and the certified eigenvalues, in Arb balls.  The formula and its five
      terms X_1..X_5 are transcribed from stageb_consts.py Sec. 5 (function tau), which is the
      audited implementation; nothing mathematical is changed.
  (2) the full STAGEB_STEP5.md Sec. 4 assembly WITH the tail:

        t_12 <= s_12 + tau_N,  t_21 <= s_21 + tau'_N            (STEP5 Sec. 3)
        ||Z|| <= s_12 s_21 / d_M            (= ||Z_0^(N)||, the computed band)
              +  tau_N tau'_N / d_N         (= ||R_N||, the tail BEYOND N -- the missing term)
              +  t_12 t_21 ||T|| / (c_perp d_M)                 (= ||Z - Z_0||)
        theta = K~_h ||Z||,  s = K~_h/(1-theta),  e = 1/c_perp,
        K <= || [[s, s t_12 e], [e t_21 s, e + e^2 t_21 s t_12]] ||_2       (STEP5 (*))
        beta = 4 K^2 delta,  rho = 2 K delta/(1 + sqrt(1-beta)),  Im omega >= Im mu - rho.

  (3) the largest tail tau the acceptance admits, and the smallest truncation N at which the whole
      thing closes.  ||P T (P_N' - P)|| <= ||P T (P_16000 - P)|| for every N' <= 16000 (restriction
      of the same operator to a subspace), so the certified strip bounds are valid for every
      N' <= 16000 and the scan below is rigorous.

EVERY INPUT IS QUOTED IN ITS CONSERVATIVE DIRECTION and its provenance is printed.  Nothing is
recomputed here that another certified run already produced; the arithmetic is all ball arithmetic.

  python stageb_scalars_iv.py            (seconds, < 0.2 GB, single core, reads one .npz read-only)
"""
import time
from math import floor
import numpy as np
from flint import arb, ctx

ctx.prec = 300
T0 = time.time()

# --------------------------------------------------------------------------- certified inputs
# Theta_n, gamma, mu_w, alpha, ||h||/||grad g||, ||h^*||/||grad g||, mu_2: stageb_consts.log
# (Arb enclosures; the UPPER end is quoted for everything that enters the bound multiplicatively,
#  the LOWER end for alpha, which divides).
TH0_S   = "506.548892"        # Theta_0 = ||T||           upper
TH1_S   = "2482.955855"       # Theta_1                   upper
TH2_S   = "21447.36325"       # Theta_2                   upper
TH4_S   = "906972.8975"       # Theta_4                   upper
GAMMA_S = "0.03664396213289260"     # gamma               upper
SGAM_S  = "0.19142612709056358"     # sqrt(gamma)         upper
MUW_S   = "0.19055585381226847"     # mu_w                upper
MU2_S   = "1.40807080928479"        # mu_2                upper
ALPHA_S = "0.019008746283155910"    # alpha = pi/(3X)     lower
HN_S    = "44.49994883600"    # ||h||/||grad g||          upper
HSN_S   = "31.23241601939"    # ||h^*||/||grad g||        upper
CNORM_S = "232.249"           # ||C||   (tnorm_rigorous.py, r_w geometry)   upper
# mu: the certified Galerkin eigenvalue at M = 7682 (kh_block_7682.log), an exact input
MU_RE_S, MU_IM_S = "-17.525873540058", "25.425171859977"
KH_S    = "1.2246711"         # K~_h, bordered inverse at M = 7682   upper.  NOT the log's printed
                              # "1.224671": kh_block.py computes outward(1/dd) and prints it with
                              # %.6f, which rounds an UPPER bound DOWN.  From the log's own certified
                              # sigma_min >= sqrt(0.666747054) - 5.04e-10 = 0.8165458059269...,
                              # K~_h <= 1.2246710384, and 1.2246711 is the smallest 8-digit decimal
                              # above it.  (STAGEB_AUDIT.md finding 6.)
DELTA_S = "8.017047e-03"      # delta, Proposition T3 at N = M in balls (T3_BALLS.md)    upper
S12_S   = "140.320484"        # ||P T (P_N - P)||,  ropnorm_final.log, 17/17 columns     upper
S21_S   = "140.228256"        # ||(P_N - P) T P||,  ropnorm_final.log, 17/17 columns     upper
MBLOCK  = 7682
NRECT   = 16000               # the truncation the completed rectangle reaches
ZEROS_BIG = "stokes_zeros_rigorous_L697000.npz"   # 29278 certified eigenvalues

TH0, TH1, TH2, TH4 = arb(TH0_S), arb(TH1_S), arb(TH2_S), arb(TH4_S)
gamma, sgamma, mu_w, mu2 = arb(GAMMA_S), arb(SGAM_S), arb(MUW_S), arb(MU2_S)
al, hn, hsn = arb(ALPHA_S), arb(HN_S), arb(HSN_S)
cnorm, kh, delta = arb(CNORM_S), arb(KH_S), arb(DELTA_S)
s12, s21 = arb(S12_S), arb(S21_S)
mu_re, mu_im = arb(MU_RE_S), arb(MU_IM_S)

z = np.load(ZEROS_BIG)
lam_lo, lam_hi = z["lam_lower"], z["lam_upper"]
lamM = arb(float(lam_hi[MBLOCK - 1]))          # lambda_M, certified UPPER bound
lamMp1 = arb(float(lam_lo[MBLOCK]))            # lambda_{M+1}, certified LOWER bound
r2, Msq = arb(2).sqrt(), arb(MBLOCK).sqrt()

# c_*, c-hat(lambda_M), s_w  (stageb_consts.py Sec. 4; recomputed here from the same inputs)
chatM = sgamma + gamma / lamM.sqrt()
c_star = (gamma * lamM).sqrt() + gamma
s_w = (2 * lamM * (1 + chatM / lamM.sqrt())).sqrt()
X = arb(2000) * arb("0.027546") * (arb.pi() / 400).cos()       # X = k r_w, as in stageb_consts.py

# Proposition T2's three composite constants (stageb_consts.py Sec. 5)
S_T2 = c_star * (lamM * hsn + 4 * TH2 + 4 * r2 * TH1 * mu2) + 4 * TH4
W_T2 = c_star * TH0 * mu_w + 2 * r2 * TH1 * s_w / X
G_T2 = Msq * TH0 * s_w * mu_w + lamM * hn
S_T2s = c_star * (lamM * hn + 4 * TH2 + 4 * r2 * TH1 * mu2) + 4 * TH4     # swapped (tau'_N)
G_T2s = Msq * TH0 * s_w * mu_w + lamM * hsn


def tau(N, certified=True, swap=False):
    """(T2) upper bound on tau_N; swap=True gives tau'_N (||h||, ||h^*|| interchanged).
    certified=True takes lambda_{N+1} from the certified zero file, False from Lemma W."""
    if certified:
        if N >= len(lam_lo):
            return None, None
        L = arb(float(lam_lo[N]))                      # lambda_{N+1}, certified lower bound
    else:
        L = 1 + al ** 2 * arb(N - 5) ** 2              # Lemma W
    if not (L > lamM):
        return None, None
    kL = L / (L - lamM)
    chat = sgamma + gamma / L.sqrt()
    kap = 1 + chat / L.sqrt()
    Sig3 = 1 / (5 * al ** 6 * arb(N - 5) ** 5)         # Lemma W tail sum (always)
    Sv, Gv, hv = (S_T2, G_T2, hsn) if not swap else (S_T2s, G_T2s, hn)
    # NB the h that multiplies X_5 is ||h|| for tau_N and ||h^*|| for tau'_N (T2 Sec. 3, X_5 uses
    # sum_i |beta_i|^2 <= ||h||^2); the S/G swap is as in stageb_consts.py.
    hv5 = hn if not swap else hsn
    X1 = kL ** 2 * Msq * Sv / L ** 2
    X2 = kL ** 2 * 8 * TH2 * (arb(MBLOCK) * lamM).sqrt() / (L * L.sqrt())
    X3 = kL ** 2 * (2 * kap * MBLOCK * Sig3).sqrt() * W_T2
    X4 = kL ** 2 * chat * Gv * Sig3.sqrt()
    X5 = kL * chat * hv5 / (al * arb(N - 5).sqrt())
    return X1 + X2 + X3 + X4 + X5, (X1, X2, X3, X4, X5, L, kL, Sig3)


def dist(lam):
    """d = |-i lam - mu| = sqrt(Re mu^2 + (lam + Im mu)^2), a rigorous lower bound for lam a lower bound."""
    return (mu_re ** 2 + (lam + mu_im) ** 2).sqrt()


def two_norm(m11, m12, m21, m22):
    a = m11 * m11 + m21 * m21
    b = m11 * m12 + m21 * m22
    c = m12 * m12 + m22 * m22
    half = arb(1) / 2
    d = (a - c) * half
    return ((a + c) * half + (d * d + b * b).sqrt()).sqrt()


def assemble(tau_N, taup_N, dN, sa=s12, sb=s21, verbose=False):
    """STEP5 Sec. 4 with the tail.  tau_N = taup_N = 0 reproduces accept_iv.py / ropnorm.py."""
    d_M = dist(lamMp1)
    c_perp = lamMp1 + mu_im - cnorm
    t12, t21 = sa + tau_N, sb + taup_N
    Z0 = sa * sb / d_M                       # computed band  M < j <= N
    RN = tau_N * taup_N / dN                 # THE TAIL,  j > N
    Zres = t12 * t21 * TH0 / (c_perp * d_M)  # ||Z - Z_0||
    Z = Z0 + RN + Zres
    theta = kh * Z
    if not (theta < arb(1)):
        return dict(Z=Z, theta=theta, ok=False)
    e = arb(1) / c_perp
    s = kh / (arb(1) - theta)
    K = two_norm(s, s * t12 * e, e * t21 * s, e + e * e * t21 * s * t12)
    beta = 4 * K * K * delta
    out = dict(Z=Z, theta=theta, K=K, beta=beta, Z0=Z0, RN=RN, Zres=Zres,
               t12=t12, t21=t21, ok=bool(beta < arb(1)))
    if out["ok"]:
        rho = 2 * K * delta / (1 + (1 - beta).sqrt())
        out["rho"] = rho
        out["imw"] = mu_im - rho
    return out


def show(label, r):
    if not r["ok"]:
        if "beta" not in r:
            print(f"  {label:34s} ||Z||={r['Z'].str(8):>13s} theta={r['theta'].str(8):>12s}"
                  f"   THETA NOT PROVABLY < 1")
        else:
            print(f"  {label:34s} ||Z||={r['Z'].str(8):>13s} theta={r['theta'].str(8):>12s} "
                  f"K={r['K'].str(8):>10s} beta={r['beta'].str(8):>12s}  DOES NOT CLOSE")
        return
    print(f"  {label:34s} ||Z||={r['Z'].str(8):>13s} theta={r['theta'].str(8):>12s} "
          f"K={r['K'].str(8):>10s} beta={r['beta'].str(8):>12s}  CLOSES   "
          f"rho={r['rho'].str(6)}  Im omega >= {r['imw'].str(10)}")


def main():
    print(__doc__.split("  python")[0].rstrip())
    print("=" * 122)
    print(f"prec = {ctx.prec} bits;  M = {MBLOCK}, N = {NRECT} = {NRECT/MBLOCK:.3f} M;  "
          f"certified zeros {ZEROS_BIG} ({len(lam_lo)} eigenvalues)")
    print(f"  lambda_M   <= {float(lamM.mid()):.6f} (certified upper)   "
          f"lambda_(M+1) >= {float(lamMp1.mid()):.6f} (certified lower)")
    print(f"  lambda_(N+1) >= {float(lam_lo[NRECT]):.6f} (certified lower)   "
          f"d_M >= {float(dist(lamMp1).mid()):.4f}   d_N >= {float(dist(arb(float(lam_lo[NRECT]))).mid()):.4f}")
    print(f"  c_* = {c_star.str(10)}   s_w = {s_w.str(10)}   "
          f"S = {float(S_T2.mid()):.6e}   W = {float(W_T2.mid()):.6e}   G = {float(G_T2.mid()):.6e}")
    print(f"  c_perp >= {float((lamMp1 + mu_im - cnorm).mid()):.4f}  (= lambda_(M+1) + Im mu - ||C||)")

    # ------------------------------------------------------------------ 1. the tail at N = 16000
    print("\n1. PROPOSITION T2 AT THE TRUNCATION THE RECTANGLE REACHES (rigorous, Arb balls)\n")
    print("   | N | N/M | lambda_(N+1) | route | tau_N <= | X_1 | X_2 | X_3 | X_4 | X_5 |")
    rows = {}
    for N in (NRECT, 2 * MBLOCK, 3 * MBLOCK, 28113):
        for cert in (True, False):
            for swap in (False, True):
                t_, pr = tau(N, cert, swap)
                if t_ is None:
                    continue
                rows[(N, cert, swap)] = t_
                if swap:
                    continue
                print(f"   | {N} | {N/MBLOCK:.3f} M | {float(pr[5].mid()):.5g} | "
                      f"{'certified' if cert else 'Lemma W  '} | {t_.str(8):>10s} | " +
                      " | ".join(pr[i].str(4) for i in range(5)) + " |")
    tauN = rows[(NRECT, True, False)]
    taupN = rows[(NRECT, True, True)]
    tauN_W = rows[(NRECT, False, False)]
    taupN_W = rows[(NRECT, False, True)]
    print(f"\n   tau_N  <= {tauN.str(10)} (certified lambda)   {tauN_W.str(10)} (Lemma W)")
    print(f"   tau'_N <= {taupN.str(10)} (certified lambda)   {taupN_W.str(10)} (Lemma W)")

    # ------------------------------------------------------------------ 2. assembly with the tail
    dN = dist(arb(float(lam_lo[NRECT])))
    print("\n2. THE STEP5 Sec. 4 ASSEMBLY, WITH AND WITHOUT THE TAIL TERM\n")
    show("tail EXCLUDED (= accept_iv.py)", assemble(arb(0), arb(0), dN))
    show("tail INCLUDED, Lemma W tau", assemble(tauN_W, taupN_W, dN))
    r = assemble(tauN, taupN, dN)
    show("tail INCLUDED, certified tau", r)
    if r["ok"]:
        print(f"\n   ||Z|| = {r['Z0'].str(8)} (computed band)  +  {r['RN'].str(8)} (TAIL)  +  "
              f"{r['Zres'].str(8)} (||Z-Z_0||)  =  {r['Z'].str(8)}")
        print(f"   t_12 <= {r['t12'].str(9)},  t_21 <= {r['t21'].str(9)}   "
              f"(certified strips {S12_S} / {S21_S} plus the tail)")
        print(f"   beta = {r['beta'].str(12)} < 1 by a factor {(arb(1)/r['beta']).str(6)};  "
              f"rho <= {r['rho'].str(10)} (units nu k^2)")
        # Collapse units are a CONVENTION: the rate in s^-1 times (T - t), and (T - t) depends on
        # the fitted singular time.  The paper's convention (tex/paper1.tex, Section 3.6 and
        # Theorem 2) is T = 0.0022865 on the x144 state, i.e. T - t = 2.4478e-5 s, giving 1.2435.
        # The older stage-B notes used T_est = 0.002278, i.e. T - t = 1.598e-5 s, giving 0.8118;
        # that figure is NOT the paper's and is printed here only so the two cannot be confused.
        # Every printed LOWER bound is rounded DOWN, not to nearest: %.4f of 1.243556 would print
        # 1.2436, which is larger than the quantity it claims to bound.  (That failure mode is how
        # a display-rounded K~_h once propagated into the published constants.)
        def flo(a, nd):
            v = float(a.mid()) - float(a.rad()) * 4
            return floor(v * 10 ** nd) / 10 ** nd
        imw_s = r["imw"] * 2000
        print(f"   Im omega >= {r['imw'].str(12)} > 0   (= {flo(imw_s, 2):.2f} s^-1)")
        print(f"   in collapse units: {flo(imw_s * arb('2.4478e-5'), 4):.4f} (T-t)^-1 "
              f"at the paper's T = 0.0022865, T - t = 2.4478e-5 s"
              f"   [the notes' older T_est = 0.002278, T - t = 1.598e-5 s, would give "
              f"{flo(imw_s * arb('1.5980e-5'), 4):.4f}; not the paper's convention]")

    # ------------------------------------------------------------------ 3. how much tail is admissible
    print("\n3. HOW MUCH TAIL THE ACCEPTANCE ADMITS, AND HOW FEW COLUMNS ARE NEEDED\n")
    lo, hi = 0.0, 1000.0
    for _ in range(80):
        mid = (lo + hi) / 2
        rr = assemble(arb(repr(mid)), arb(repr(mid)), dN)
        if rr["ok"]:
            lo = mid
        else:
            hi = mid
    print(f"   with the certified strips, the acceptance closes for every tau_N = tau'_N <= {lo:.3f}")
    print(f"   (against tau_16000 <= {float(tauN.mid()):.3f} certified: a factor "
          f"{lo/float(tauN.mid()):.2f} of headroom on the tail)")
    lo2, hi2 = 0.0, 1000.0
    for _ in range(80):
        mid = (lo2 + hi2) / 2
        rr = assemble(arb(repr(mid)), arb(repr(mid)), dist(lamMp1))   # worst case d_N = d_M
        if rr["ok"]:
            lo2 = mid
        else:
            hi2 = mid
    print(f"   (same with the tail weighted by d_M instead of d_N, i.e. N -> M: tau <= {lo2:.3f})")

    print("\n   smallest truncation N at which the WHOLE criterion closes (strip bounds are valid")
    print("   for every N <= 16000, being restrictions of the certified 16000-column strips):\n")
    Nmin = None
    for N in range(MBLOCK + 1, NRECT + 1, 1):
        pass
    lo3, hi3 = MBLOCK + 1, NRECT

    def closes(N):
        a, _ = tau(N, True, False)
        b, _ = tau(N, True, True)
        if a is None:
            return False
        return assemble(a, b, dist(arb(float(lam_lo[N]))))["ok"]

    if closes(hi3):
        while hi3 - lo3 > 1:
            mid = (lo3 + hi3) // 2
            if closes(mid):
                hi3 = mid
            else:
                lo3 = mid
        Nmin = hi3
        a, _ = tau(Nmin, True, False)
        b, _ = tau(Nmin, True, True)
        rm = assemble(a, b, dist(arb(float(lam_lo[Nmin]))))
        print(f"   N_min = {Nmin} = {Nmin/MBLOCK:.3f} M  (tau = {a.str(6)}, beta = {rm['beta'].str(8)}, "
              f"Im omega >= {rm['imw'].str(10)});  N = 16000 = {NRECT/MBLOCK:.3f} M was computed.")
        print(f"   columns actually needed: {Nmin - MBLOCK} of the {NRECT - MBLOCK} built "
              f"({(Nmin - MBLOCK)/500:.1f} of 17 blocks of 500).")
    else:
        print("   the criterion does NOT close at N = 16000 with the tail included.")

    print(f"\nwall {time.time() - T0:.1f} s")


if __name__ == "__main__":
    main()
