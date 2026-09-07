"""The KNSS-hypothesis quantities on the snapshots: K1 = sup |u| |x'| (the |v| <= C/|x'| bound, |x'| = r), K1core = |u| r at the swirl maximum,
K2 = sup |u| (|x| + sqrt(T-t)) (the Type I constant), Gamma = r^2 u1: max, its radius, sign, and monotonicity beyond the maximum."""
import sys, glob, numpy as np
sys.path.insert(0, ".")
from axiphys import AxiPhys
from scipy.interpolate import CubicSpline
T = 0.0022865
for f in sorted(glob.glob("axiphys_513_512_nsz_t0.002*.npz")) + sorted(glob.glob("nsz2_snap_t*.npz")) + sorted(glob.glob("nsz5_snap_t*.npz")) + sorted(glob.glob("nsz6_snap_t0.0022855*.npz")):
    d = np.load(f); U, Om, Ps, r, z, t = d["U"], d["Om"], d["Ps"], d["r"], d["z"], float(d["t"])
    if "nsz2" in f and t > 0.0022851: continue
    if "nsz5" in f and t < 0.0022851: continue
    zm = d["zmap"] if "zmap" in d.files else None; zm = tuple(float(v) for v in zm) if zm is not None else (0.9, 1.0)
    P = AxiPhys(len(r), len(z) + 1, a=5.0); P.set_zmap(zm); P.r = r.copy()
    rr = np.concatenate([-r[:0:-1], r]); Psr = CubicSpline(rr, np.concatenate([Ps[:0:-1], Ps]), axis=0)(r, 1); Psz = P.d_z(Ps)
    ur = -r[:, None] * Psz; uz = 2 * Ps + r[:, None] * Psr; uth = r[:, None] * U
    sp = np.sqrt(ur ** 2 + uz ** 2 + uth ** 2); Tt = T - t
    K1 = float((sp * r[:, None]).max()); i0, j0 = np.unravel_index(int(np.argmax(np.abs(U))), U.shape)
    K1core = float(sp[i0, j0] * r[i0]); zz = np.minimum(z, 1 - z)[None, :]
    K2 = float((sp * (np.sqrt(r[:, None] ** 2 + zz ** 2) + np.sqrt(max(Tt, 0)))).max())
    G = r[:, None] ** 2 * U; ig, jg = np.unravel_index(int(np.argmax(G)), G.shape)
    col = G[:, jg]; mono = bool(np.all(np.diff(col[ig:]) <= 1e-9 * G.max()))   # non-increasing beyond the maximum on its column
    print(f"t {t:.7f} T-t {Tt:.2e}  K1 = sup|u| r {K1:.1f}  |u| r at the swirl max {K1core:.1f}  K2 = C_U0 {K2:.1f}  Gamma max {G.max():.2f} at (r {r[ig]:.4f}, z {z[jg]:.4f}) [r/R {r[ig]/r[i0]:.2f}]  Gamma min {G.min():.2f}  non-increasing beyond max: {mono}", flush=True)
