"""The Leibovich-Stewartson criterion as the paper and lscert.py/lscheck.py have it, against the
criterion as Leibovich and Stewartson actually state it.

JFM 126 (1983) 335-356, abstract, verbatim: with Omega = V/r the angular velocity and Gamma = rV
the circulation, the flow is unstable if

    V (dOmega/dr) [ (dOmega/dr)(dGamma/dr) + (dW/dr)^2 ] < 0.

The repository has  Phi = 2 V Omega [ D(rV) D(Omega) + D(W^2) ],  which differs twice:
  * prefactor 2 V Omega = 2V^2/r >= 0 instead of V DOmega, which is NEGATIVE wherever Omega
    decreases outward -- so the sign of the whole test inverts over the region of interest;
  * D(W^2) = 2 W W' instead of (DW)^2 = (W')^2.  D(W^2) ~ V^2/L while DOmega*DGamma ~ V^2/L^2,
    so the repository's bracket is not dimensionally homogeneous.  That alone settles it.

This script prints both on the same fields so the damage can be seen.
"""
import sys
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, ".")
from axiphys import AxiPhys

for f in sys.argv[1:]:
    d = np.load(f)
    U, Om, Ps, r, z, t = d["U"], d["Om"], d["Ps"], d["r"], d["z"], float(d["t"])
    zm = d["zmap"] if "zmap" in d.files else None
    zm = tuple(float(v) for v in zm) if zm is not None and len(zm) >= 2 else (0.0, 0.0)
    P = AxiPhys(len(r), len(z) + 1, a=5.0)
    if zm[0] > 0:
        P.set_zmap(zm)
    P.r = r.copy()
    rr = np.concatenate([-r[:0:-1], r])

    def Dr(F, par):
        FF = np.concatenate([F[:0:-1] * (1 if par == "even" else -1), F], axis=0)
        return CubicSpline(rr, FF, axis=0)(r, 1)

    Psr = Dr(Ps, "even")
    uz = 2 * Ps + r[:, None] * Psr
    V = r[:, None] * U          # swirl velocity
    Omg = U                     # angular velocity V/r = u_1
    W = uz
    DOm = Dr(Omg, "even")
    DGam = Dr(r[:, None] * V, "even")   # Gamma = r V
    DW = Dr(W, "even")

    phi_repo = 2 * V * Omg * (DGam * DOm + Dr(W ** 2, "even"))
    phi_true = V * DOm * (DOm * DGam + DW ** 2)

    A = float(np.abs(U).max())
    i0, j0 = np.unravel_index(int(np.argmax(np.abs(U))), U.shape)
    R2 = r < 2 * r[i0]
    col = slice(None)

    print(f"\n{f}  t {t:.7f}  swirl max at (r,z)=({r[i0]:.4f},{z[j0]:.4f})")
    for name, phi in (("repo  2 V Om [DGam DOm + D(W^2)]", phi_repo),
                      ("TRUE  V DOm [DOm DGam + (DW)^2]", phi_true)):
        im, jm = np.unravel_index(int(np.argmin(phi)), phi.shape)
        colmin = phi[col, j0]
        fneg = float(np.mean(phi[R2][:, :] < 0))
        # where on the swirl-max column is it negative, and is that outside the maximum?
        negr = r[np.where(colmin < 0)[0]]
        band = f"{negr.min():.4f}-{negr.max():.4f}" if negr.size else "none"
        print(f"  {name}")
        print(f"     min over field {phi.min():+.3e} at (r,z)=({r[im]:.4f},{z[jm]:.4f})")
        print(f"     fraction of r<2R with phi<0: {fneg:.3f}"
              f"   negative band on swirl-max column: r in {band}"
              f"   (swirl max at r={r[i0]:.4f})")
