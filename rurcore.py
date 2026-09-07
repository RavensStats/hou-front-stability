"""Hou's escape quantity at the core: |r u^r| and |r u^z| restricted to the collapse region, versus amplification.
Regions: (a) the collapse core |x - x0| < c R (R = swirl-maximum radius, x0 = the swirl maximum, c = 3 and 10); (b) r < 0.1 (Hou's
window for the log-weighted criteria); (c) the global maximum (far field).  Also the swirl-maximum values.  A logarithmic
growth over the range would read +30 percent (log(1/(T - t)) rises by 3.3 units from x144 to x331).   python rurcore.py"""
import sys, glob, numpy as np
sys.path.insert(0, ".")
from axiphys import AxiPhys
from scipy.interpolate import CubicSpline
T = 0.0022865
print("file                                 t          amp   |r u_r| core3R  core10R  r<0.1   global  | |r u_z| core3R core10R  r<0.1  global | |u| r at max")
for f in sorted(glob.glob("axiphys_513_512_nsz_t0.002*.npz")) + sorted(glob.glob("nsz2_snap_t*.npz")) + sorted(glob.glob("nsz5_snap_t*.npz")) + sorted(glob.glob("nsz6_snap_t0.0022855*.npz")):
    d = np.load(f); U, Om, Ps, r, z, t = d["U"], d["Om"], d["Ps"], d["r"], d["z"], float(d["t"])
    if "nsz2" in f and t > 0.0022851: continue
    if "nsz5" in f and t < 0.0022851: continue
    zm = d["zmap"] if "zmap" in d.files else None; zm = tuple(float(v) for v in zm) if zm is not None else (0.9, 1.0)
    P = AxiPhys(len(r), len(z) + 1, a=5.0); P.set_zmap(zm); P.r = r.copy()
    rr = np.concatenate([-r[:0:-1], r]); Psr = CubicSpline(rr, np.concatenate([Ps[:0:-1], Ps]), axis=0)(r, 1); Psz = P.d_z(Ps)
    ur = -r[:, None] * Psz; uz = 2 * Ps + r[:, None] * Psr; uth = r[:, None] * U
    i0, j0 = np.unravel_index(int(np.argmax(np.abs(U))), U.shape); R = r[i0]; Z0 = z[j0]
    dist = np.sqrt((r[:, None] - R) ** 2 + (z[None, :] - Z0) ** 2)
    rur = np.abs(r[:, None] * ur); ruz = np.abs(r[:, None] * uz)
    m3 = dist < 3 * R; m10 = dist < 10 * R; mr = r[:, None] < 0.1 + 0 * z[None, :]
    amp = float(np.abs(U).max()) / 12000.0
    sp0 = np.sqrt(ur[i0, j0] ** 2 + uz[i0, j0] ** 2 + uth[i0, j0] ** 2) * R
    print(f"{f[:36]:36s} {t:.7f} {amp:6.0f}  {rur[m3].max():7.3f} {rur[m10].max():7.3f} {rur[mr].max():7.3f} {rur.max():7.3f} | {ruz[m3].max():7.3f} {ruz[m10].max():7.3f} {ruz[mr].max():7.3f} {ruz.max():7.3f} | {sp0:6.2f}   (R {R:.5f}, log(1/(T-t)) {np.log(1/(T-t)):.2f})")
