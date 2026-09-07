"""COLENC stage A, part 1: build RIGOROUSLY ENCLOSED collocation matrices (A, B) of the columnar normal-mode problem
(colsolve.py's discretization) on an EXPLICIT field: Omega(s), W(s) polynomials in s = (r / rmax)^2 with exactly rational
coefficients (the double-precision Chebyshev fit of the chosen column, converted exactly), on the Chebyshev-Gauss points of
[-rmax, rmax] with the parity mirror trick.  Every matrix entry is evaluated in mpmath interval arithmetic (points, barycentric
differentiation matrix, field values and derivatives, 1/r factors), then stored as midpoint + radius for ivmat.py's Krawczyk
enclosure (part 2, colenc_run.py).  Also stores the floating-point approximate eigenpair (from colsolve's formulation) as the
starting point.
   python colenc.py <snapshot> <z_column> <N> <rmax/R> <m> <k> <nu> [degree]   -> colenc_<tag>.npz"""
import sys, os, numpy as np
from fractions import Fraction
import mpmath as mp
from mpmath import iv
from scipy.interpolate import CubicSpline
from numpy.polynomial import chebyshev as C
from scipy.linalg import eig
f, zc, N, kmax, m, k, nu = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
deg = int(sys.argv[8]) if len(sys.argv) > 8 else 30
iv.prec = 80                                                     # interval evaluation precision (bits)
d = np.load(f); U, Ps, r, z, t = d["U"], d["Ps"], d["r"], d["z"], float(d["t"])
rr = np.concatenate([-r[:0:-1], r]); Psr = CubicSpline(rr, np.concatenate([Ps[:0:-1], Ps]), axis=0)(r, 1); uz = 2 * Ps + r[:, None] * Psr
i0, j0 = np.unravel_index(int(np.argmax(np.abs(U))), U.shape); R = float(r[i0]); rmax = float(kmax * R)
jz = int(np.argmin(np.abs(z - zc)))
sel = r <= rmax; rs = r[sel]; Om_c = U[sel, jz]; W_c = uz[sel, jz]
x = 2 * (rs / rmax) ** 2 - 1
c = C.chebfit(x, Om_c, deg); dd = C.chebfit(x, W_c, deg)
# exact rational coefficients of the field in the MONOMIAL basis of x = 2 s - 1 (converted from the Chebyshev coefficients exactly)
def cheb_to_monomial(coefs):
    # T_k(x) as exact rational polynomials in x
    T = [[Fraction(1)], [Fraction(0), Fraction(1)]]
    for kk in range(2, len(coefs)):
        prev, prev2 = T[-1], T[-2]
        new = [Fraction(0)] * (len(prev) + 1)
        for i_, a in enumerate(prev): new[i_ + 1] += 2 * a
        for i_, a in enumerate(prev2): new[i_] -= a
        T.append(new)
    out = [Fraction(0)] * len(coefs)
    for kk, ck in enumerate(coefs):
        cq = Fraction(float(ck))
        for i_, a in enumerate(T[kk]): out[i_] += cq * a
    return out
Om_mono = cheb_to_monomial(c); W_mono = cheb_to_monomial(dd)
rmaxQ = Fraction(rmax); nuQ = Fraction(nu); kQ = Fraction(k)
print(f"{os.path.basename(f)}: column z = {z[jz]:.5f} (asked {zc}); R = {R:.6f}; rmax = {kmax} R; field degree {deg} in s (exact rationals); N = {N}; m = {m}; k = {k}; nu = {nu}")
# --- interval Chebyshev-Gauss points on [-rmax, rmax], mirror trick ---
M2 = 2 * N
pi = iv.pi
xg = [iv.cos(pi * (j + iv.mpf(1) / 2) / M2) for j in range(M2)]          # intervals
rmax_iv = iv.mpf(rmaxQ.numerator) / rmaxQ.denominator
rg = [xi * rmax_iv for xi in xg]
# barycentric weights for Chebyshev-Gauss (first kind) points: w_j = (-1)^j sin(pi (j + 1/2) / M2)
wj = [((-1) ** j) * iv.sin(pi * (j + iv.mpf(1) / 2) / M2) for j in range(M2)]
pos = [j for j in range(M2) if j < M2 // 2]          # x_j > 0 for j < N (cos decreasing)
neg = [M2 - 1 - j for j in pos]                       # mirror: x_{M2-1-j} = -x_j exactly
n = len(pos)
def D_entry(i_, j_):
    if i_ == j_: return None
    return (wj[j_] / wj[i_]) / (rg[i_] - rg[j_])
# differentiation matrix on the 2N points, then parity reduction: D_sig[i, j] = D[pos_i, pos_j] + sig * D[pos_i, neg_j]
# diagonal: D[i, i] = -sum_{k != i} D[i, k]  (exact for polynomial interpolation of constants)
Dfull = [[None] * M2 for _ in range(M2)]
for i_ in range(M2):
    row_sum = iv.mpf(0)
    for j_ in range(M2):
        if i_ != j_:
            e = D_entry(i_, j_); Dfull[i_][j_] = e; row_sum += e
    Dfull[i_][i_] = -row_sum
De = [[Dfull[pos[i_]][pos[j_]] + Dfull[pos[i_]][neg[j_]] for j_ in range(n)] for i_ in range(n)]
Do = [[Dfull[pos[i_]][pos[j_]] - Dfull[pos[i_]][neg[j_]] for j_ in range(n)] for i_ in range(n)]
rp = [rg[j_] for j_ in pos]
# --- field at the points (interval) ---
def poly_eval(mono, xv):
    acc = iv.mpf(0)
    for a in reversed(mono):
        acc = acc * xv + iv.mpf(a.numerator) / a.denominator
    return acc
def poly_deriv(mono):
    return [a * i_ for i_, a in enumerate(mono)][1:]
Om_d = poly_deriv(Om_mono); W_d = poly_deriv(W_mono)
Om_p, Omp_p, W_p, Wp_p = [], [], [], []
for q in rp:
    xv = 2 * (q / rmax_iv) ** 2 - 1
    dxdr = 4 * q / rmax_iv ** 2
    Om_p.append(poly_eval(Om_mono, xv)); Omp_p.append(poly_eval(Om_d, xv) * dxdr)
    W_p.append(poly_eval(W_mono, xv)); Wp_p.append(poly_eval(W_d, xv) * dxdr)
# --- assemble A, B (complex intervals) following colsolve.py exactly ---
se = +1 if m % 2 == 1 else -1
Dv = De if se > 0 else Do; Dp = Do if se > 0 else De
def matmul_iv(X, Y):
    return [[sum((X[i_][l] * Y[l][j_] for l in range(n)), iv.mpf(0)) for j_ in range(n)] for i_ in range(n)]
print("  interval differentiation matrices built; forming second derivatives ...", flush=True)
D2v = matmul_iv(Dp, Dv); D2p = matmul_iv(Dv, Dp)
nu_iv = iv.mpf(nuQ.numerator) / nuQ.denominator; k_iv = iv.mpf(kQ.numerator) / kQ.denominator
inv_r = [1 / q for q in rp]
dim = 4 * n
Amid = np.zeros((dim, dim), dtype=complex); Arad = np.zeros((dim, dim)); Bmid = np.zeros((dim, dim), dtype=complex); Brad = np.zeros((dim, dim))
def put(Mm, Mr, i_, j_, val_re, val_im=None):
    # val_re, val_im: mpmath intervals (real); store midpoint and radius rigorously (outward)
    re_mid = mp.mpf(val_re.mid); re_rad = mp.mpf(val_re.delta) / 2
    if val_im is None: im_mid, im_rad = mp.mpf(0), mp.mpf(0)
    else: im_mid = mp.mpf(val_im.mid); im_rad = mp.mpf(val_im.delta) / 2
    Mm[i_, j_] = complex(float(re_mid), float(im_mid))
    # conversion to double: |float(x) - x| <= 2^-53 |x| ; disc radius >= sqrt(re_rad^2 + im_rad^2) + conversion errors
    conv = (abs(float(re_mid)) + abs(float(im_mid))) * 2.0 ** -52
    Mr[i_, j_] = float(mp.sqrt(re_rad ** 2 + im_rad ** 2)) * (1 + 2.0 ** -40) + conv + 1e-300
zero = iv.mpf(0)
for i_ in range(n):
    a_i = m * Om_p[i_] + k_iv * W_p[i_]                                   # advective frequency at point i
    for j_ in range(n):
        dij = iv.mpf(1) if i_ == j_ else zero
        lap_v = D2v[i_][j_] + inv_r[i_] * Dv[i_][j_] - dij * (m * m * inv_r[i_] ** 2 + k_iv ** 2)
        lap_p = D2p[i_][j_] + inv_r[i_] * Dp[i_][j_] - dij * (m * m * inv_r[i_] ** 2 + k_iv ** 2)
        # row r (block 0): [a + i nu (lap - 1/r^2)] u_r + [2 i Om + 2 nu m / r^2] u_th - i Dp p
        put(Amid, Arad, i_, j_, dij * a_i, nu_iv * (lap_v - dij * inv_r[i_] ** 2))
        put(Amid, Arad, i_, n + j_, dij * 2 * nu_iv * m * inv_r[i_] ** 2, dij * 2 * Om_p[i_])
        put(Amid, Arad, i_, 3 * n + j_, zero, -Dp[i_][j_])
        # row theta (block 1): [-i (V' + Om) - 2 nu m / r^2] u_r + [a + i nu (lap - 1/r^2)] u_th + (m / r) p ;  V' + Om = 2 Om + r Om'
        put(Amid, Arad, n + i_, j_, -dij * 2 * nu_iv * m * inv_r[i_] ** 2, -dij * (2 * Om_p[i_] + rp[i_] * Omp_p[i_]))
        put(Amid, Arad, n + i_, n + j_, dij * a_i, nu_iv * (lap_v - dij * inv_r[i_] ** 2))
        put(Amid, Arad, n + i_, 3 * n + j_, dij * m * inv_r[i_], None)
        # row z (block 2): [-i W'] u_r + [a + i nu lap_p] u_z + k p
        put(Amid, Arad, 2 * n + i_, j_, zero, -dij * Wp_p[i_])
        put(Amid, Arad, 2 * n + i_, 2 * n + j_, dij * a_i, nu_iv * lap_p)
        put(Amid, Arad, 2 * n + i_, 3 * n + j_, dij * k_iv, None)
        # continuity (block 3): (Dv + 1/r) u_r + (i m / r) u_th + i k u_z = 0
        put(Amid, Arad, 3 * n + i_, j_, Dv[i_][j_] + dij * inv_r[i_], None)
        put(Amid, Arad, 3 * n + i_, n + j_, zero, dij * m * inv_r[i_])
        put(Amid, Arad, 3 * n + i_, 2 * n + j_, zero, dij * k_iv)
for i_ in range(3 * n): Bmid[i_, i_] = 1.0
# wall rows: u_r = u_theta = u_z = 0 at the largest r (index 0 of pos, since x_0 = cos(pi/(2 M2)) is the largest) -- exact rows
iw = 0
for blk in ([0] if nu == 0 else [0, 1, 2]):
    Amid[blk * n + iw, :] = 0; Arad[blk * n + iw, :] = 0; Amid[blk * n + iw, blk * n + iw] = 1.0; Bmid[blk * n + iw, :] = 0
print(f"  matrices built: dim {dim}; max radius A {Arad.max():.2e} (max |A| {np.abs(Amid).max():.2e}); B exact", flush=True)
# floating-point approximate eigenpair
w, vec = eig(Amid, Bmid)
fin = np.isfinite(w) & (np.abs(w) < 1e9); w = w[fin]; vec = vec[:, fin]
order = np.argsort(-w.imag); w0 = w[order[0]]; x0 = vec[:, order[0]]
knorm = int(np.argmax(np.abs(x0))); x0 = x0 / x0[knorm]
print(f"  approximate leading eigenvalue: omega = {w0.real:+.6e} {w0.imag:+.6e}i  (Im omega (T-t) = {w0.imag * (0.002278 - t):.4f}); next Im {w[order[1]].imag:+.4e}; normalization index {knorm}")
tag = f"{os.path.basename(f).replace('.npz','')}_z{zc}_N{N}_m{m}_k{int(k)}_nu{nu}_d{deg}"
np.savez(f"colenc_{tag}.npz", Amid=Amid, Arad=Arad, Bmid=Bmid, Brad=Brad, w0=w0, x0=x0, knorm=knorm, t=t, R=R, rmax=rmax, N=N, m=m, k=k, nu=nu, deg=deg,
         Om_num=np.array([str(a.numerator) for a in Om_mono]), Om_den=np.array([str(a.denominator) for a in Om_mono]),
         W_num=np.array([str(a.numerator) for a in W_mono]), W_den=np.array([str(a.denominator) for a in W_mono]))
print(f"  saved colenc_{tag}.npz")
