"""RETRACTED AND CORRECTED 2026-09-12.

This script previously used the Leibovich-Stewartson criterion in the form
    Phi = 2 V Omega [D(rV) D(Omega) + D(W^2)],
which is wrong twice.  Leibovich & Stewartson, JFM 126 (1983) 335-356, state: with Omega = V/r the
angular velocity and Gamma = rV the circulation, the flow is unstable where
    Phi = V (dOmega/dr) [ (dOmega/dr)(dGamma/dr) + (dW/dr)^2 ] < 0.
  * the prefactor is V DOmega, not 2 V Omega.  2 V Omega = 2V^2/r is non-negative, so it can never
    invert the test where Omega decreases outward -- which is exactly what makes LS weaker than
    Rayleigh.  The sign of the whole criterion was therefore inverted over the region of interest.
  * the axial term is (DW)^2, not D(W^2) = 2 W W'.  D(W^2) ~ V^2/L against DOmega DGamma ~ V^2/L^2,
    so the old bracket was not even dimensionally homogeneous.

CONSEQUENCE FOR THE PAPER.  The certificate this script produced is RETRACTED.  With the criterion
corrected, Phi on the degree-20 polynomial fit is negative only on r/R in [1.992, 2.000] -- at the
outer edge of the fit -- and interval arithmetic does not certify even that.  That is the failure the
paper's own caveat predicted: the correct criterion carries Omega' in the prefactor as well as inside
the bracket, and the fit's first derivative is wrong by up to a factor of two near the swirl maximum.
The old prefactor 2 r Omega^2 contained no derivative at all, which is why it certified.

WHAT SURVIVES is the uncertified grid evaluation (lscheck.py, ls_recheck.py): with the correct
criterion Phi is still negative in a band just outside the swirl maximum on all three states.

Phi is now enclosed DIRECTLY rather than factored, so no reasoning about the signs of separate
factors is needed.  Expect this script to report that it cannot certify; that is the correct result.
"""
import sys, numpy as np
sys.path.insert(0, ".")
from axiphys import AxiPhys
from scipy.interpolate import CubicSpline
from numpy.polynomial import chebyshev as C
import mpmath as mpm
mpm.mp.dps = 40
f = sys.argv[1]; deg = int(sys.argv[2]) if len(sys.argv) > 2 else 40; nsub = int(sys.argv[3]) if len(sys.argv) > 3 else 400
d = np.load(f); U, Om, Ps, r, z, t = d["U"], d["Om"], d["Ps"], d["r"], d["z"], float(d["t"])
zm = d["zmap"] if "zmap" in d.files else None; zm = tuple(float(v) for v in zm) if zm is not None else (0.0, 0.0)
P = AxiPhys(len(r), len(z) + 1, a=5.0)
if zm[0] > 0: P.set_zmap(zm)
P.r = r.copy()
rr = np.concatenate([-r[:0:-1], r]); Psr = CubicSpline(rr, np.concatenate([Ps[:0:-1], Ps]), axis=0)(r, 1); Psz = P.d_z(Ps)
uz = 2 * Ps + r[:, None] * Psr
i0, j0 = np.unravel_index(int(np.argmax(np.abs(U))), U.shape); R = float(r[i0]); rmax = float(2.0 * R)
sel = r <= rmax; rs = r[sel]; Om_c = U[sel, j0]; W_c = uz[sel, j0]
x = 2 * (rs / rmax) ** 2 - 1
c = C.chebfit(x, Om_c, deg); dd = C.chebfit(x, W_c, deg)
res_O = np.abs(C.chebval(x, c) - Om_c).max() / np.abs(Om_c).max(); res_W = np.abs(C.chebval(x, dd) - W_c).max() / max(np.abs(W_c).max(), 1e-300)
print(f"{f}: t {t:.7f}; column z = {z[j0]:.5f} through the swirl maximum at r = {R:.5f}; rmax = 2R = {rmax:.5f}; {sel.sum()} grid points; Chebyshev degree {deg} in s = (r/rmax)^2")
print(f"  explicit field vs grid column: max relative residual Omega {res_O:.2e}, W {res_W:.2e}")
# interval arithmetic: Clenshaw for the series and its x-derivative with interval coefficients
iv = mpm.iv
cI = [iv.mpf(float(v)) for v in c]; dI = [iv.mpf(float(v)) for v in dd]
def chebder_exact(coef):
    """derivative coefficients by the exact recurrence c'_{k-1} = c'_{k+1} + 2 k c_k (c'_0 halved)"""
    n = len(coef); der = [iv.mpf(0)] * (n + 1)
    for k in range(n - 1, 0, -1):
        der[k - 1] = der[k + 1] + 2 * k * coef[k]
    der[0] = der[0] / 2
    return der[:n]
cD = chebder_exact(cI); dD = chebder_exact(dI)
def clenshaw(coef, xv):
    b1 = iv.mpf(0); b2 = iv.mpf(0)
    for k in range(len(coef) - 1, 0, -1):
        b1, b2 = 2 * xv * b1 - b2 + coef[k], b1
    return xv * b1 - b2 + coef[0]
# float scan to locate the negative annulus
rf = np.linspace(1e-6, rmax, 4000); xf = 2 * (rf / rmax) ** 2 - 1
Of = C.chebval(xf, c); Opf = C.chebval(xf, C.chebder(c)) * 4 * rf / rmax ** 2
Wf = C.chebval(xf, dd); Wpf = C.chebval(xf, C.chebder(dd)) * 4 * rf / rmax ** 2
Bf = rf * Of * Opf * ((2 * rf * Of + rf ** 2 * Opf) * Opf + Wpf ** 2)
neg = np.where(Bf < 0)[0]
if len(neg) == 0: print("  float scan: Phi >= 0 everywhere; nothing to certify"); sys.exit()
# largest contiguous negative run containing the most negative point
k0 = int(np.argmin(Bf)); a = k0
while a > 0 and Bf[a - 1] < 0: a -= 1
b = k0
while b < len(rf) - 1 and Bf[b + 1] < 0: b += 1
r1f, r2f = rf[a], rf[b]
print(f"  float scan: B < 0 on [{r1f:.5f}, {r2f:.5f}] (r/R from {r1f/R:.3f} to {r2f/R:.3f}); min B/A^2-scale {Bf[k0]/np.abs(Om_c).max()**2:.3e} at r = {rf[k0]:.5f}")
# shrink by 2% each side and certify with intervals
r1 = r1f + 0.02 * (r2f - r1f); r2 = r2f - 0.02 * (r2f - r1f)
edges = np.linspace(r1, r2, nsub + 1); worst = -mpm.inf; ok = True; supB = None
for i in range(nsub):
    rI = iv.mpf([float(edges[i]), float(edges[i + 1])])
    rm = iv.mpf(rmax)
    xI = 2 * (rI / rm) ** 2 - 1
    O = clenshaw(cI, xI); Op = clenshaw(cD, xI) * 4 * rI / rm ** 2
    W = clenshaw(dI, xI); Wp = clenshaw(dD, xI) * 4 * rI / rm ** 2
    B = rI * O * Op * ((2 * rI * O + rI ** 2 * Op) * Op + Wp ** 2)
    hi = B.b
    if supB is None or hi > supB: supB = hi
    if not (hi < 0): ok = False; print(f"  NOT certified on [{edges[i]:.6f}, {edges[i+1]:.6f}]: sup B = {mpm.nstr(hi, 6)}")
    if not ok and i > 5: break
if ok:
    print(f"  CERTIFIED (interval arithmetic, {nsub} subintervals, {mpm.mp.dps} digits): Phi < 0 on [{r1:.6f}, {r2:.6f}] = [{r1/R:.3f} R, {r2/R:.3f} R]; sup of the interval upper bounds of B = {mpm.nstr(supB, 6)}")
