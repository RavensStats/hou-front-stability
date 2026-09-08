"""Diagnostics of the late-stage transition on the wide-map run nsz9 (lit check 203): for each snapshot from t = 0.0022857 on,
the two structures (front: |u1| maximum over z > 0.0006; inner: over z < 0.00045), their positions (R, Z) and R/Z; the global
maximum and |u^r| / |u^z| there; the vacuum diagnostic min_{r < R} |u1|(r, Z) / |u1|(R, Z) along the maximum's row (Hou Case 4:
a vacuum region for u1 opens between the front and r = 0); A(T-t) with T = 0.0022866.   python nsz9diag.py [glob]"""
import sys, glob, numpy as np
sys.path.insert(0, ".")
from axiphys import AxiPhys
from scipy.interpolate import CubicSpline
T = 0.0022866
pat = sys.argv[1] if len(sys.argv) > 1 else "nsz9_snap_t0.00228[56]*.npz"
print("file                        t          x     A(T-t) | front |u1| (R,Z)            R/Z | inner |u1| (R,Z)            R/Z | at max: |ur|/|uz|  ur      uz    | vacuum min/max (r<R at Z)  r_min")
for f in sorted(glob.glob(pat)):
    d = np.load(f); U, Ps, r, z, t = d["U"], d["Ps"], d["r"], d["z"], float(d["t"])
    if t < 0.0022857: continue
    zm = tuple(float(v) for v in d["zmap"])
    P = AxiPhys(len(r), len(z) + 1, a=5.0); P.set_zmap(zm); P.r = r.copy()
    rr = np.concatenate([-r[:0:-1], r]); Psr = CubicSpline(rr, np.concatenate([Ps[:0:-1], Ps]), axis=0)(r, 1); Psz = P.d_z(Ps)
    ur = -r[:, None] * Psz; uz = 2 * Ps + r[:, None] * Psr
    A = np.abs(U); i, j = np.unravel_index(int(np.argmax(A)), A.shape); um = A[i, j]
    mf = z > 0.0006; jf = np.where(mf)[0]; i1, j1 = np.unravel_index(int(np.argmax(A[:, mf])), A[:, mf].shape); j1 = jf[j1]
    mi = z < 0.00045; ji = np.where(mi)[0]; i2, j2 = np.unravel_index(int(np.argmax(A[:, mi])), A[:, mi].shape); j2 = ji[j2]
    row = A[:i + 1, j]; kmin = int(np.argmin(row)); vac = row[kmin] / um
    print(f"{f[:26]:26s} {t:.7f} {um/12000:5.0f} {um*(T-t):5.2f} | {A[i1,j1]:.3e} ({r[i1]:.5f},{z[j1]:.5f}) {r[i1]/z[j1]:5.2f} | {A[i2,j2]:.3e} ({r[i2]:.5f},{z[j2]:.5f}) {r[i2]/z[j2]:5.2f} | {abs(ur[i,j])/abs(uz[i,j]):6.3f} {ur[i,j]:8.1f} {uz[i,j]:8.1f} | {vac:.3f}  r={r[kmin]:.5f}")
