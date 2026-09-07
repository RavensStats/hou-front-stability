"""Stage B, step 2 (floating-point feasibility): the DEFECT of the certified discrete eigenpair against the CONTINUOUS
columnar operator.  The collocation eigenvector on the 2N mirrored Chebyshev-Gauss points defines polynomial interpolants
u_r, u_theta (parity se) and u_z, p (parity -se) of degree 2N - 1 in r; the residual of each continuous equation
(momentum rows and continuity) is evaluated on a fine grid from the exact Chebyshev representation of the interpolants,
and its L^2(r dr) norm on (0, r_max) is compared with ||omega B u||.  A relative defect of order 1e-9 is what Plum's
fixed point needs (with K the other factor).  python coldefect.py colenc_<tag>.npz"""
import sys, numpy as np
from numpy.polynomial import chebyshev as C
from fractions import Fraction
f = sys.argv[1]; d = np.load(f)
Amid, Bmid, w0, x0, N, m, k, nu, rmax = d["Amid"], d["Bmid"], complex(d["w0"]), d["x0"], int(d["N"]), int(d["m"]), float(d["k"]), float(d["nu"]), float(d["rmax"])
Om_mono = [Fraction(int(a), int(b)) for a, b in zip(d["Om_num"], d["Om_den"])]; W_mono = [Fraction(int(a), int(b)) for a, b in zip(d["W_num"], d["W_den"])]
Om_c = np.array([float(a) for a in Om_mono]); W_c = np.array([float(a) for a in W_mono])   # monomial coefficients in x = 2 (r/rmax)^2 - 1
# refine the eigenpair in floating point (one inverse-iteration step) for a clean residual
from scipy.linalg import solve
n = Amid.shape[0] // 4
M = Amid - w0 * Bmid
try:
    y = solve(M + 1e-30 * np.eye(len(M)), Bmid @ x0); x0 = y / y[np.argmax(np.abs(y))]
except Exception: pass
ur, ut, uzv, pv = x0[:n], x0[n:2 * n], x0[2 * n:3 * n], x0[3 * n:]
M2 = 2 * N; j = np.arange(M2); xg = np.cos(np.pi * (j + 0.5) / M2)      # Gauss points on [-1, 1]; pos = first N (x > 0)
se = 1.0 if m % 2 == 1 else -1.0
def mirror(vals, sig):
    full = np.zeros(M2, dtype=complex); full[:N] = vals; full[N:] = sig * vals[::-1]; return full
def cheb_coeffs(full):
    # exact interpolation on the Gauss points: coefficients via the discrete cosine transform (Gauss-Chebyshev quadrature is exact here)
    kk = np.arange(M2); th = np.pi * (j + 0.5) / M2
    c = np.array([2.0 / M2 * np.sum(full * np.cos(kk_ * th)) for kk_ in kk]); c[0] /= 2; return c
cr, ct, cz, cp = (cheb_coeffs(mirror(ur, se)), cheb_coeffs(mirror(ut, se)), cheb_coeffs(mirror(uzv, -se)), cheb_coeffs(mirror(pv, -se)))
# fine grid on (0, rmax): Chebyshev-Gauss with 8N points restricted to x > 0, plus quadrature weights for int_0^rmax |.|^2 r dr
Mf = 8 * N; xf_all = np.cos(np.pi * (np.arange(Mf) + 0.5) / Mf); xf = xf_all[xf_all > 0]; rf = rmax * xf
wq = (np.pi / Mf) * np.sqrt(1 - xf ** 2) * rmax          # Gauss-Chebyshev weights for int dx with weight 1/sqrt(1-x^2) -> plain dx
def ev(c, der=0):
    cc = c.copy()
    for _ in range(der): cc = C.chebder(cc)
    return C.chebval(xf, cc) / rmax ** der
UR, URp, URpp = ev(cr), ev(cr, 1), ev(cr, 2); UT, UTp, UTpp = ev(ct), ev(ct, 1), ev(ct, 2); UZ, UZp, UZpp = ev(cz), ev(cz, 1), ev(cz, 2); P, Pp = ev(cp), ev(cp, 1)
xs = 2 * (rf / rmax) ** 2 - 1; dxdr = 4 * rf / rmax ** 2
Om = np.polyval(Om_c[::-1], xs); Omp = np.polyval(np.polyder(Om_c[::-1]), xs) * dxdr; W = np.polyval(W_c[::-1], xs); Wp = np.polyval(np.polyder(W_c[::-1]), xs) * dxdr
a = m * Om + k * W
lap = lambda F, Fp, Fpp: Fpp + Fp / rf - (m ** 2 / rf ** 2 + k ** 2) * F
Rr = a * UR + 1j * nu * (lap(UR, URp, URpp) - UR / rf ** 2) + (2j * Om + 2 * nu * m / rf ** 2) * UT - 1j * Pp - w0 * UR
Rt = (-1j * (2 * Om + rf * Omp) - 2 * nu * m / rf ** 2) * UR + a * UT + 1j * nu * (lap(UT, UTp, UTpp) - UT / rf ** 2) + (m / rf) * P - w0 * UT
Rz = -1j * Wp * UR + a * UZ + 1j * nu * lap(UZ, UZp, UZpp) + k * P - w0 * UZ
Rc = URp + UR / rf + 1j * m * UT / rf + 1j * k * UZ
nrm = lambda F: np.sqrt(np.sum(np.abs(F) ** 2 * rf * wq))
ref = np.sqrt(nrm(w0 * UR) ** 2 + nrm(w0 * UT) ** 2 + nrm(w0 * UZ) ** 2)
print(f"{f}: N {N}, m {m}, k {k}, nu {nu}; omega = {w0.real:+.6e} {w0.imag:+.6e}i")
print(f"  ||omega u||_(L2, r dr) = {ref:.4e}")
print(f"  defect norms (L2, r dr): r-momentum {nrm(Rr):.3e}, theta-momentum {nrm(Rt):.3e}, z-momentum {nrm(Rz):.3e}, continuity {nrm(Rc):.3e}")
tot = np.sqrt(nrm(Rr) ** 2 + nrm(Rt) ** 2 + nrm(Rz) ** 2)
print(f"  relative momentum defect {tot/ref:.3e}; continuity defect relative to ||k u_z|| {nrm(Rc)/nrm(k*UZ):.3e}")
# where is the defect concentrated?
e = np.abs(Rr) ** 2 + np.abs(Rt) ** 2 + np.abs(Rz) ** 2
print(f"  momentum-defect energy centroid r/rmax = {np.sum(rf * e * rf * wq)/np.sum(e * rf * wq)/rmax:.3f}; fraction inside r < 0.1 rmax: {np.sum((e*rf*wq)[rf < 0.1*rmax])/np.sum(e*rf*wq):.2e}; near the wall r > 0.95 rmax: {np.sum((e*rf*wq)[rf > 0.95*rmax])/np.sum(e*rf*wq):.2e}")
print(f"  wall values: |u_r| {abs(C.chebval(1.0, cr)):.2e}, |u_t| {abs(C.chebval(1.0, ct)):.2e}, |u_z| {abs(C.chebval(1.0, cz)):.2e} (relative to max |u| {np.abs(x0[:3*n]).max():.2e}); axis: |u_z(0)| {abs(C.chebval(0.0, cz)):.2e}, |p(0)| {abs(C.chebval(0.0, cp)):.2e}")
