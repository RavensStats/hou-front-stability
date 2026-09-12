"""Leibovich-Stewartson (JFM 126 1983) and Howard-Gupta (JFM 14 1962) criteria on an axisymmetric snapshot.
CORRECTED 2026-09-12 -- see the retraction note at the top of lscert.py.  The LS criterion is
  LS: unstable where Phi = V DOm [DOm D(rV) + (DW)^2] < 0   (V = u_theta = r u1, Om = u1, W = u_z, D = d/dr)
The earlier form, Phi = 2 V Om [D(rV) D(Om) + D(W^2)], had a non-negative prefactor and a
dimensionally inhomogeneous axial term.  The Howard-Gupta line below was always correct.
  HG: axisymmetric stability where (1/r^3) D(Gamma^2) - (1/4)(DW)^2 > 0  (Gamma = r V = r^2 u1)"""
import sys, glob, numpy as np
sys.path.insert(0, ".")
from axiphys import AxiPhys
from scipy.interpolate import CubicSpline
for f in sys.argv[1:]:
    d = np.load(f); U, Om, Ps, r, z, t = d["U"], d["Om"], d["Ps"], d["r"], d["z"], float(d["t"])
    zm = d["zmap"] if "zmap" in d.files else None; zm = tuple(float(v) for v in zm) if zm is not None and len(zm) >= 2 else (0.0, 0.0)
    P = AxiPhys(len(r), len(z) + 1, a=5.0)
    if zm[0] > 0: P.set_zmap(zm)
    P.r = r.copy()
    rr = np.concatenate([-r[:0:-1], r])
    def Dr(F, par):  # radial derivative with parity fold
        FF = np.concatenate([F[:0:-1] * (1 if par == "even" else -1), F], axis=0); return CubicSpline(rr, FF, axis=0)(r, 1)
    Psr = Dr(Ps, "even"); Psz = P.d_z(Ps); uz = 2 * Ps + r[:, None] * Psr
    V = r[:, None] * U; Omg = U; W = uz
    Phi = V * Dr(Omg, "even") * (Dr(Omg, "even") * Dr(r[:, None] * V, "even") + Dr(W, "even") ** 2)
    Gam2 = (r[:, None] ** 2 * U) ** 2; HG = Dr(Gam2, "even") / np.maximum(r[:, None], 1e-12) ** 3 - 0.25 * Dr(W, "even") ** 2
    A = float(np.abs(U).max()); i0, j0 = np.unravel_index(int(np.argmax(np.abs(U))), U.shape)
    im, jm = np.unravel_index(int(np.argmin(Phi)), Phi.shape)
    scale = A ** 4                                   # Phi ~ u1^2 * (r u1)^2 / r^2 ... normalize by A^4 for a dimensionless magnitude
    neg = Phi < -1e-3 * np.abs(Phi).max()
    print(f"{f}: t {t:.7f} A {A:.3e} at (r,z)=({r[i0]:.4f},{z[j0]:.4f})")
    print(f"   LS: min Phi/A^4 {Phi.min()/scale:+.3e} at (r,z)=({r[im]:.4f},{z[jm]:.4f}); fraction of (r<2R, |z|<2Z) with Phi<0: {np.mean(neg[(r<2*r[i0])][:, :] ):.3f}; Phi/A^4 on the u1-max column: min {Phi[:, j0].min()/scale:+.3e} at r {r[int(np.argmin(Phi[:, j0]))]:.4f}")
    print(f"   HG on the u1-max column: min {HG[:, j0].min() / A**2:+.3e} (x A^2) at r {r[int(np.argmin(HG[:, j0]))]:.4f}; Rayleigh D(Gamma^2)<0 fraction on that column (r<2R): {np.mean(Dr(Gam2,'even')[:, j0][r < 2*r[i0]] < 0):.3f}", flush=True)
