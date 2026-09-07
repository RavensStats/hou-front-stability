"""U4: scale-invariant local L^3 concentration (Barker-Prange 2003.06717: a Type I singularity needs ||u||_{L^3(B(0,R))} >= C(M) log(1/(T-t)) with R ~ (T-t)^{1/2-});
E3(k) = int_{|x| < k sqrt(T-t)} |u|^3 dx (dimensionless under Leray scaling), about the axis point the ring collapses to."""
import sys, glob, numpy as np
sys.path.insert(0, ".")
from axiphys import AxiPhys
from scipy.interpolate import CubicSpline
T = float(sys.argv[1]); ks = (4.0, 8.0, 16.0)
for f in sorted(glob.glob("axiphys_513_512_nsz_t0.002*.npz")) + sorted(glob.glob("nsz2_snap_t*.npz")) + sorted(glob.glob("nsz5_snap_t*.npz")):
    d = np.load(f); U, Om, Ps, r, z, t = d["U"], d["Om"], d["Ps"], d["r"], d["z"], float(d["t"])
    if "nsz2" in f and t > 0.0022851: continue
    if "nsz5" in f and t < 0.0022851: continue
    zm = d["zmap"] if "zmap" in d.files else None; zm = tuple(float(v) for v in zm) if zm is not None else (0.9, 1.0)
    P = AxiPhys(len(r), len(z) + 1, a=5.0); P.set_zmap(zm); P.r = r.copy()
    rr = np.concatenate([-r[:0:-1], r]); Psr = CubicSpline(rr, np.concatenate([Ps[:0:-1], Ps]), axis=0)(r, 1); Psz = P.d_z(Ps)
    ur = -r[:, None] * Psz; uz = 2 * Ps + r[:, None] * Psr; uth = r[:, None] * U
    u3 = (ur ** 2 + uz ** 2 + uth ** 2) ** 1.5; Tt = T - t; sq = np.sqrt(Tt)
    zz = np.minimum(z, 1.0 - z)[None, :]; rho2 = r[:, None] ** 2 + zz ** 2
    out = []
    for k in ks:
        mask = rho2 < (k * sq) ** 2
        E3 = 2 * np.pi * np.trapezoid(np.trapezoid(u3 * mask * r[:, None], z, axis=1), r)
        out.append(E3)
    print(f"t {t:.9f}  T-t {Tt:.2e}  log(1/(T-t)) {np.log(1/Tt):.2f}  " + "  ".join(f"E3(k={k:.0f}) {E:.3e}" for k, E in zip(ks, out)) + f"  ||u||_L3(B_4) {out[0]**(1/3):.3f}", flush=True)
