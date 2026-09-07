"""COLSOLVE: normal modes of the columnar vortex (V(r) = r Omega(r), W(r)) taken through the swirl maximum of a snapshot,
the same field that lscert2.py certifies (Leibovich-Stewartson).  Perturbation ~ exp(i m theta + i k z - i omega t); growth
rate = Im omega.  Linearized incompressible Euler (or Navier-Stokes with nu > 0) about the columnar base, Chebyshev
collocation on r in [0, rmax] with the parity (axis) conditions imposed by the mirror trick (Trefethen Program 11 style):
for m odd, u_r and u_theta are even in r and u_z, p are odd; for m even the reverse.  Wall condition u_r = 0 at rmax
(no-slip for nu > 0).  Generalized eigenproblem A x = omega B x with the continuity row singular in B; infinite
eigenvalues filtered.  Reports the leading growth rates versus k and compares with the Leibovich-Stewartson local estimate.
   python colsolve.py <snapshot> [N] [rmax/R] [m] [k list, e.g. 200,400,800] [nu: 'snap' or a float, default 0]"""
import sys, os, numpy as np
sys.path.insert(0, ".")
from scipy.interpolate import CubicSpline
from numpy.polynomial import chebyshev as C
from scipy.linalg import eig
f = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 200; kmax = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
m = int(sys.argv[4]) if len(sys.argv) > 4 else 1
klist = [float(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else [100, 200, 400, 800, 1600]
nuarg = sys.argv[6] if len(sys.argv) > 6 else "0"
deg = int(os.environ.get("COL_DEG", "20"))          # polynomial degree of the fitted field (0 = use the spline of the grid column)
d = np.load(f); U, Ps, r, z, t = d["U"], d["Ps"], d["r"], d["z"], float(d["t"])
nu = float(d["nu"]) if nuarg == "snap" else float(nuarg)
rr = np.concatenate([-r[:0:-1], r]); Psr = CubicSpline(rr, np.concatenate([Ps[:0:-1], Ps]), axis=0)(r, 1)
uz = 2 * Ps + r[:, None] * Psr
i0, j0 = np.unravel_index(int(np.argmax(np.abs(U))), U.shape); R = float(r[i0]); rmax = float(kmax * R)
if os.environ.get("COL_Z"): j0 = int(np.argmin(np.abs(z - float(os.environ["COL_Z"]))))   # column at a chosen height instead of the swirl maximum's
sel = r <= rmax; rs = r[sel]; Om_c = U[sel, j0]; W_c = uz[sel, j0]
if deg > 0:
    x = 2 * (rs / rmax) ** 2 - 1
    c = C.chebfit(x, Om_c, deg); dd = C.chebfit(x, W_c, deg)
    Om_f = lambda q: C.chebval(2 * (q / rmax) ** 2 - 1, c); W_f = lambda q: C.chebval(2 * (q / rmax) ** 2 - 1, dd)
    Omp_f = lambda q: C.chebval(2 * (q / rmax) ** 2 - 1, C.chebder(c)) * 4 * q / rmax ** 2
    Wp_f = lambda q: C.chebval(2 * (q / rmax) ** 2 - 1, C.chebder(dd)) * 4 * q / rmax ** 2
    tag = f"polynomial degree {deg}"
else:
    rre = np.concatenate([-rs[:0:-1], rs]); sO = CubicSpline(rre, np.concatenate([Om_c[:0:-1], Om_c])); sW = CubicSpline(rre, np.concatenate([W_c[:0:-1], W_c]))
    Om_f, W_f, Omp_f, Wp_f = sO, sW, (lambda q: sO(q, 1)), (lambda q: sW(q, 1)); tag = "spline of the grid column"
# Chebyshev on [-rmax, rmax] with 2N points (no point at r = 0), mirror trick
M2 = 2 * N; j = np.arange(M2); xg = np.cos(np.pi * (j + 0.5) / M2)      # Chebyshev-Gauss points, symmetric, none at 0
# differentiation matrix on Gauss points via barycentric weights
def diffmat(xn):
    n = len(xn); w = (-1.0) ** np.arange(n) * np.sin(np.pi * (np.arange(n) + 0.5) / n)
    D = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if i != k: D[i, k] = w[k] / w[i] / (xn[i] - xn[k])
        D[i, i] = -D[i].sum()
    return D
Dfull = diffmat(xg) / rmax; rg = xg * rmax
pos = np.where(rg > 0)[0]; neg = np.where(rg < 0)[0][::-1]    # neg mirrored so that rg[neg[i]] = -rg[pos[i]]
assert np.allclose(rg[neg], -rg[pos])
def Dpar(sig): return Dfull[np.ix_(pos, pos)] + sig * Dfull[np.ix_(pos, neg)]
De, Do = Dpar(+1.0), Dpar(-1.0)                                    # act on even / odd functions, return values on r > 0
rp = rg[pos]; n = len(rp); I = np.eye(n); Rinv = np.diag(1.0 / rp)
Om = Om_f(rp); W = W_f(rp); Omp = Omp_f(rp); Wp = Wp_f(rp)
V = rp * Om; Vp = Om + rp * Omp
se = +1.0 if m % 2 == 1 else -1.0                                  # parity of u_r, u_theta; u_z, p have -se
Dv = De if se > 0 else Do; Dp = Do if se > 0 else De               # derivative acting on (u_r, u_theta) / on (u_z, p)
D2v = Dp @ Dv; D2p = Dv @ Dp                                    # derivative flips parity: second derivative = D_other @ D_own
print(f"{os.path.basename(f)}: t {t:.7f}; column z = {z[j0]:.5f}; R = {R:.5f}; rmax = {kmax} R; field: {tag}; N = {N} radial points; m = {m}; nu = {nu:.1e}")
print(f"  max Omega {np.abs(Om).max():.4e} (at r/R {rp[np.argmax(np.abs(Om))]/R:.3f}), max |W| {np.abs(W).max():.4e}")
for k in klist:
    Ad = np.diag(m * Om + k * W)                                     # advective frequency
    # rows: r-momentum, theta-momentum, z-momentum, continuity; unknowns u_r, u_theta, u_z, p; A x = omega B x
    Z = np.zeros((n, n), dtype=complex)
    if nu > 0:
        Lap_v = D2v + Rinv @ Dv - np.diag(m ** 2 / rp ** 2) - k ** 2 * I; Lap_p = D2p + Rinv @ Dp - np.diag(m ** 2 / rp ** 2) - k ** 2 * I
        Vr = 1j * nu * (Lap_v - Rinv @ Rinv); Vth = 1j * nu * (Lap_v - Rinv @ Rinv); Vz = 1j * nu * Lap_p
        Xr = 2 * nu * m * (Rinv @ Rinv); Xth = -2 * nu * m * (Rinv @ Rinv)          # i nu (-2 i m / r^2) u_theta in the r eq; i nu (2 i m / r^2) u_r in the theta eq
    else:
        Vr = Vth = Vz = Xr = Xth = Z
    # omega u_r = a u_r + 2i Om u_th - i p';  omega u_th = a u_th - i (V' + Om) u_r + (m/r) p;  omega u_z = a u_z - i W' u_r + k p;  continuity = 0
    A = np.block([[Ad + Vr, 2j * np.diag(Om) + Xr, Z, -1j * Dp],
                  [-1j * np.diag(Vp + Om) + Xth, Ad + Vth, Z, np.diag(m / rp)],
                  [-1j * np.diag(Wp), Z, Ad + Vz, k * I],
                  [Dv + Rinv, np.diag(1j * m / rp), 1j * k * I, Z]])
    B = np.block([[I, Z, Z, Z], [Z, I, Z, Z], [Z, Z, I, Z], [Z, Z, Z, Z]]).astype(complex)
    # wall: u_r = 0 at the last collocation point (largest r); for nu > 0 also u_theta = u_z = 0
    iw = int(np.argmax(rp))
    for blk in ([0] if nu == 0 else [0, 1, 2]):
        A[blk * n + iw, :] = 0; A[blk * n + iw, blk * n + iw] = 1.0; B[blk * n + iw, :] = 0
    w, vec = eig(A, B)
    fin = np.isfinite(w) & (np.abs(w) < 1e3 * np.abs(Ad).max() + 1e3 * nu * (N / rmax) ** 2 + 1e6)
    w = w[fin]; vec = vec[:, fin]
    order = np.argsort(-w.imag); w = w[order]; vec = vec[:, order]
    top = w[:4]
    nearfront = [q for q in w if (lambda e: (np.sum(rp * e) / np.sum(e)) / R < 1.5)(np.abs(vec[:, list(w).index(q)][:n]) ** 2 + np.abs(vec[:, list(w).index(q)][n:2*n]) ** 2 + np.abs(vec[:, list(w).index(q)][2*n:3*n]) ** 2)][:2]
    # check the leading eigenvector's location
    v0 = vec[:, 0]; e = np.abs(v0[:n]) ** 2 + np.abs(v0[n:2 * n]) ** 2 + np.abs(v0[2 * n:3 * n]) ** 2
    rc = float(np.sum(rp * e) / np.sum(e)); half = rp[e > 0.5 * e.max()]
    print(f"  k = {k:7.1f} (k R = {k*R:6.2f}): leading Im omega = {top[0].imag:+.4e} (Re {top[0].real:+.4e}); next {top[1].imag:+.3e}, {top[2].imag:+.3e}, {top[3].imag:+.3e}; "
          f"leading mode at r/R = {rc/R:.3f}, support [{half.min()/R:.3f}, {half.max()/R:.3f}] R; fastest modes centred inside 1.5 R: " + ", ".join(f"{q.imag:+.3e}" for q in nearfront))
