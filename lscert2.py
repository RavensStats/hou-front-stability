"""Computer-assisted certificate, exact version.  The columnar vortex (V(r), W(r)) built from the column through the swirl maximum
of a snapshot is DEFINED by polynomials Omega(s), W(s) in s = (r/rmax)^2 with exactly rational coefficients (the double-precision
least-squares Chebyshev fit, converted exactly).  Leibovich-Stewartson's discriminant is Phi = 2 r Omega^2 B with
B(r) = (2 r Omega + r^2 Omega') Omega' + 2 W W', a polynomial in r with rational coefficients.  B < 0 on [r1, r2] is certified by
(i) Sturm's theorem: B has no real root in [r1, r2] (exact rational arithmetic, sympy), and (ii) B(r_mid) < 0 exactly.
Usage: python lscert2.py <snapshot> [degree] [rmax/R] [z-column: 'max' or a float]"""
import sys, numpy as np
sys.path.insert(0, ".")
from axiphys import AxiPhys
from scipy.interpolate import CubicSpline
from numpy.polynomial import chebyshev as C
from fractions import Fraction
import sympy as sp
f = sys.argv[1]; deg = int(sys.argv[2]) if len(sys.argv) > 2 else 40; kmax = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
d = np.load(f); U, Om, Ps, r, z, t = d["U"], d["Om"], d["Ps"], d["r"], d["z"], float(d["t"])
zm = d["zmap"] if "zmap" in d.files else None; zm = tuple(float(v) for v in zm) if zm is not None else (0.0, 0.0)
P = AxiPhys(len(r), len(z) + 1, a=5.0)
if zm[0] > 0: P.set_zmap(zm)
P.r = r.copy()
rr = np.concatenate([-r[:0:-1], r]); Psr = CubicSpline(rr, np.concatenate([Ps[:0:-1], Ps]), axis=0)(r, 1)
uz = 2 * Ps + r[:, None] * Psr
i0, j0 = np.unravel_index(int(np.argmax(np.abs(U))), U.shape); R = float(r[i0]); rmax = float(kmax * R)
sel = r <= rmax; rs = r[sel]; Om_c = U[sel, j0]; W_c = uz[sel, j0]
x = 2 * (rs / rmax) ** 2 - 1
c = C.chebfit(x, Om_c, deg); dd = C.chebfit(x, W_c, deg)
res_O = np.abs(C.chebval(x, c) - Om_c).max() / np.abs(Om_c).max(); res_W = np.abs(C.chebval(x, dd) - W_c).max() / max(np.abs(W_c).max(), 1e-300)
print(f"{f}: t {t:.7f}; column z = {z[j0]:.5f} through the swirl maximum at r = {R:.5f}; rmax = {kmax} R = {rmax:.5f}; {sel.sum()} grid points; Chebyshev degree {deg} in s = (r/rmax)^2")
print(f"  explicit field vs grid column: max relative residual Omega {res_O:.2e}, W {res_W:.2e}")
# float scan for the negative annulus
rf = np.linspace(1e-6, rmax, 4000); xf = 2 * (rf / rmax) ** 2 - 1
Of = C.chebval(xf, c); Opf = C.chebval(xf, C.chebder(c)) * 4 * rf / rmax ** 2
Wf = C.chebval(xf, dd); Wpf = C.chebval(xf, C.chebder(dd)) * 4 * rf / rmax ** 2
Bf = (2 * rf * Of + rf ** 2 * Opf) * Opf + 2 * Wf * Wpf
k0 = int(np.argmin(Bf)); a = k0
while a > 0 and Bf[a - 1] < 0: a -= 1
b = k0
while b < len(rf) - 1 and Bf[b + 1] < 0: b += 1
r1f, r2f = rf[a], rf[b]
print(f"  float scan: B < 0 on [{r1f:.5f}, {r2f:.5f}] (r/R {r1f/R:.3f} to {r2f/R:.3f}); min B / max|Omega|^2 = {Bf[k0]/np.abs(Om_c).max()**2:.3e} at r = {rf[k0]:.5f}")
# exact polynomial: s = (r/rmax)^2, x = 2 s - 1; Omega(x) = sum c_k T_k(x) -> monomial in x exactly, then in r
rsym = sp.symbols("r")
cQ = [sp.Rational(Fraction(float(v))) for v in c]; dQ = [sp.Rational(Fraction(float(v))) for v in dd]
rmQ = sp.Rational(Fraction(rmax))
xs = 2 * (rsym / rmQ) ** 2 - 1
def cheb_poly(coefs):
    T0, T1 = sp.Integer(1), xs; total = coefs[0] * T0 + (coefs[1] * T1 if len(coefs) > 1 else 0)
    for k in range(2, len(coefs)):
        T0, T1 = T1, sp.expand(2 * xs * T1 - T0); total += coefs[k] * T1
    return sp.Poly(sp.expand(total), rsym)
OmP = cheb_poly(cQ); WP = cheb_poly(dQ)
OmD = OmP.diff(rsym); WD = WP.diff(rsym)
rP = sp.Poly(rsym, rsym)
B = (2 * rP * OmP + rP ** 2 * OmD) * OmD + 2 * WP * WD
print(f"  exact B(r): polynomial of degree {B.degree()} with rational coefficients")
r1 = sp.Rational(Fraction(float(r1f + 0.02 * (r2f - r1f)))); r2 = sp.Rational(Fraction(float(r2f - 0.02 * (r2f - r1f))))
nroots = B.count_roots(r1, r2)
mid = (r1 + r2) / 2; Bmid = B.eval(mid)
print(f"  Sturm count of real roots of B in [{float(r1):.6f}, {float(r2):.6f}]: {nroots}; exact sign of B at the midpoint: {'negative' if Bmid < 0 else 'non-negative'} (B(mid) = {float(Bmid):.4e})")
if nroots == 0 and Bmid < 0:
    print(f"  CERTIFIED (exact rational arithmetic): B < 0, hence Phi < 0, on [{float(r1):.6f}, {float(r2):.6f}] = [{float(r1)/R:.3f} R, {float(r2)/R:.3f} R] for the explicit polynomial columnar field")
else:
    print("  NOT certified on this interval")
