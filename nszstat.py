import sys, glob, numpy as np
sys.path.insert(0, ".")
from axiphys import AxiPhys
from scipy.interpolate import CubicSpline
T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0022865
for f in sorted(glob.glob(sys.argv[2] if len(sys.argv) > 2 else "nsz4_snap_t*.npz")):
    d = np.load(f); U, Om, Ps, r, z, t = d["U"], d["Om"], d["Ps"], d["r"], d["z"], float(d["t"])
    zm = tuple(float(v) for v in d["zmap"]) if "zmap" in d.files else (0.98, 1.0)
    P = AxiPhys(len(r), len(z) + 1, a=5.0); P.set_zmap(zm); P.r = r.copy()
    rr = np.concatenate([-r[:0:-1], r]); Psr = CubicSpline(rr, np.concatenate([Ps[:0:-1], Ps]), axis=0)(r, 1); Psz = P.d_z(Ps)
    ur = -r[:, None] * Psz; uz = 2 * Ps + r[:, None] * Psr; uth = r[:, None] * U
    speed = np.sqrt(ur ** 2 + uz ** 2 + uth ** 2); wmax = float(np.abs(r[:, None] * Om).max()); A = float(np.abs(U).max())
    i, j = np.unravel_index(int(np.argmax(np.abs(U))), U.shape); Tt = T - t
    W = np.abs(r[:, None] * Om); iw, jw = np.unravel_index(int(np.argmax(W)), W.shape)
    col = W[iw, :]; half = col > 0.5 * col[jw]; jl = jw
    while jl > 0 and half[jl - 1]: jl -= 1
    jr = jw
    while jr < len(z) - 1 and half[jr + 1]: jr += 1
    dz_w = z[jr] - z[jl] + 0.5 * ((z[jr + 1] - z[jr]) if jr < len(z) - 1 else 0) + 0.5 * ((z[jl] - z[jl - 1]) if jl > 0 else 0)
    row = W[:, jw]; halfr = row > 0.5 * row[iw]; il = iw
    while il > 0 and halfr[il - 1]: il -= 1
    ir = iw
    while ir < len(r) - 1 and halfr[ir + 1]: ir += 1
    dr_w = r[ir] - r[il] + 0.5 * ((r[ir + 1] - r[ir]) if ir < len(r) - 1 else 0) + 0.5 * ((r[il] - r[il - 1]) if il > 0 else 0)
    nz_cells = jr - jl + 1; sq = np.sqrt(max(Tt, 1e-300))
    print(f"   front: omega max at (r,z)=({r[iw]:.5f},{z[jw]:.5f})  FWHM_z {dz_w:.2e} ({nz_cells} cells; /sqrt(T-t) {dz_w/sq:.4f}; /sqrt(nu(T-t)) {dz_w/np.sqrt(5e-4*max(Tt,1e-300)):.2f})  FWHM_r {dr_w:.2e} ({ir-il+1} cells; /sqrt(T-t) {dr_w/sq:.3f})", flush=True)
    umer = np.sqrt(ur ** 2 + uz ** 2); Mmer = float(umer.max()) * sq                       # meridional Type I constant (Seregin-Sverak Thm 1.1 hypothesis)
    w_r = -r[:, None] * P.d_z(U); w_z = 2.0 * U + r[:, None] * CubicSpline(rr, np.concatenate([U[:0:-1], U]), axis=0)(r, 1)
    wr_max, wz_max = float(np.abs(w_r).max()), float(np.abs(w_z).max())                  # vorticity-direction split (Deng-Hou-Yu: xi ~ e_theta?)
    colz = uz[iw, :]; a0 = float(colz[1] / z[1])                                         # compressive strain at the symmetry plane on the ring column (u_z odd in z)
    lin = a0 * z; dev = np.abs(colz - lin) > 0.2 * np.abs(lin) + 1e-12; jdep = 1
    while jdep < len(z) - 1 and not dev[jdep + 1]: jdep += 1
    a_axis = float(uz[0, 1] / z[1])                                                      # the same on the axis
    print(f"   Mmer = sup|u_mer| sqrt(T-t) {Mmer:.3f}  |omega_r|max/|omega_th|max {wr_max/max(wmax,1e-300):.3f}  |omega_z|max/|omega_th|max {wz_max/max(wmax,1e-300):.3f}  strain a(T-t): ring column {a0*Tt:+.3f}, axis {a_axis*Tt:+.3f}; linear (20%) to z/sqrt(T-t) {z[jdep]/sq:.3f} (front at {z[jw]/sq:.3f})", flush=True)
    print(f"t {t:.9f}  T-t {Tt:.3e}  A {A:.4e} at ({r[i]:.5f},{z[j]:.5f}) [i {i} j {j}]  A(T-t) {A*Tt:.3f}  R/sqrt {r[i]/np.sqrt(Tt):.3f}  sup|u| {speed.max():.1f}  M = sup|u| sqrt(T-t) {speed.max()*np.sqrt(Tt):.4f}  omega(T-t) {wmax*Tt:.1f}  omega/A {wmax/A:.2f}  Gmax {np.abs(r[:,None]**2*U).max():.2f}", flush=True)
