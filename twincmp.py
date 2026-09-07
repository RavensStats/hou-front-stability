import sys, numpy as np
sys.path.insert(0, ".")
from axiphys import AxiPhys
from scipy.interpolate import CubicSpline
def stats(f):
    d = np.load(f); U, Om, Ps, r, z, t = d["U"], d["Om"], d["Ps"], d["r"], d["z"], float(d["t"])
    zm = tuple(float(v) for v in d["zmap"]) if "zmap" in d.files else (0.9, 1.0)
    P = AxiPhys(len(r), len(z) + 1, a=5.0); P.set_zmap(zm); P.r = d["r"].copy()
    rr = np.concatenate([-r[:0:-1], r]); Psr = CubicSpline(rr, np.concatenate([Ps[:0:-1], Ps]), axis=0)(r, 1); Psz = P.d_z(Ps)
    ur = -r[:, None] * Psz; uz = 2 * Ps + r[:, None] * Psr; uth = r[:, None] * U
    speed = np.sqrt(ur ** 2 + uz ** 2 + uth ** 2); wmax = float(np.abs(r[:, None] * Om).max()); A = float(np.abs(U).max())
    i, j = np.unravel_index(int(np.argmax(np.abs(U))), U.shape)
    return t, A, r[i], z[j], float(speed.max()), wmax
for pair in [("nsz2_snap_t0.002280034.npz", "nsz3_snap_t0.002280036.npz"), ("nsz2_snap_t0.002281036.npz", "nsz3_snap_t0.002281043.npz"), ("nsz2_snap_t0.002282040.npz", "nsz3_snap_t0.002282044.npz")]:
    a = stats(pair[0]); b = stats(pair[1])
    print(f"t {a[0]:.9f}/{b[0]:.9f}  A {a[1]:.4e}/{b[1]:.4e} ({b[1]/a[1]-1:+.2e})  R {a[2]:.5f}/{b[2]:.5f}  Z {a[3]:.5f}/{b[3]:.5f}  sup|u| {a[4]:.1f}/{b[4]:.1f} ({b[4]/a[4]-1:+.2e})  wmax {a[5]:.3e}/{b[5]:.3e} ({b[5]/a[5]-1:+.2e})  w/A {a[5]/a[1]:.2f}/{b[5]/b[1]:.2f}", flush=True)
