"""roll-up test end state (lit 170 signatures without 3D derivatives): mode-energy density e(r,z), its peak and half-widths vs the front and sqrt(nu(T-t));
the 3D swirl velocity over theta at the peak (compactness in theta); the axisymmetric depletion."""
import numpy as np, sys
sys.path.insert(0, ".")
from axiphys import AxiPhys
d = np.load(sys.argv[1]); T = float(sys.argv[2]); nu = 5e-4
U, Om, r, z, t = d["U"], d["Om"], d["r"], d["z"], float(d["t"]); Tt = T - t; sq = np.sqrt(Tt); dB = np.sqrt(nu * Tt)
P = AxiPhys(len(r), U.shape[1] + 1, a=5.0); P.set_zmap((0.9, 1.0)); zh = P.z          # the base lives on the half-period mapped grid
M = 1 + sum(1 for k in d.files if k.startswith("ur")); modes = {m: {f: d[f"{f}{m}"] for f in ("ur", "ut", "uz")} for m in range(1, M)}
e = sum(np.abs(modes[m]["ur"]) ** 2 + np.abs(modes[m]["ut"]) ** 2 + np.abs(modes[m]["uz"]) ** 2 for m in modes)
ie, je = np.unravel_index(int(np.argmax(e)), e.shape); zf = z[je]
def fwhm(vec, coord, j):
    half = vec > 0.5 * vec[j]; a = j
    while a > 0 and half[a - 1]: a -= 1
    b = j
    while b < len(vec) - 1 and half[b + 1]: b += 1
    return abs(coord[b] - coord[a]), b - a + 1
wr, nr = fwhm(e[:, je], r, ie); wz, nz = fwhm(e[ie, :], z, je)
A = float(np.abs(U).max()); i0, j0 = np.unravel_index(int(np.argmax(np.abs(U))), U.shape); jh = int(np.argmin(np.abs(zh - zf)))
W = np.abs(r[:, None] * Om); iw, jw = np.unravel_index(int(np.argmax(W)), W.shape); wfz, nfz = fwhm(W[iw, :], zh, jw)
print(f"t {t:.7f}  T-t {Tt:.2e}  sqrt(T-t) {sq:.2e}  sqrt(nu(T-t)) {dB:.2e}  modes M-1 = {M-1}")
print(f"axisymmetric: |u1| max {A:.4e} at (r {r[i0]:.4f}, z {zh[j0]:.4f}); omega_theta max at (r {r[iw]:.4f}, z {zh[jw]:.4f}), front FWHM_z {wfz:.2e} ({nfz} cells) = {wfz/dB:.2f} sqrt(nu(T-t))")
print(f"mode energy density peak at (r {r[ie]:.4f}, z {zf:.4f}) -- relative to the ring: r/R {r[ie]/r[i0]:.2f}, z/Z {zf/max(zh[j0],1e-9):.2f}; FWHM_r {wr:.2e} ({nr} cells) = {wr/sq:.3f} sqrt(T-t) = {wr/dB:.1f} sqrt(nu(T-t)); FWHM_z {wz:.2e} ({nz} cells) = {wz/sq:.3f} sqrt(T-t) = {wz/dB:.1f} sqrt(nu(T-t))")
th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
ut3 = r[ie] * U[ie, jh] + sum(2 * np.real(modes[m]["ut"][ie, je] * np.exp(1j * m * th)) for m in modes)
print(f"3D swirl at the mode peak over theta: max {ut3.max():.1f} min {ut3.min():.1f} mean {ut3.mean():.1f}  (max/mean {ut3.max()/ut3.mean():.2f}); m = 1 amplitude {2*abs(modes[1]['ut'][ie, je]):.1f}, m = 2 {2*abs(modes[2]['ut'][ie, je]) if 2 in modes else 0:.1f}")
E = {m: float(np.sum((np.abs(modes[m]["ur"]) ** 2 + np.abs(modes[m]["ut"]) ** 2 + np.abs(modes[m]["uz"]) ** 2) * r[:, None])) for m in modes}
print("mode energies", {m: f"{E[m]:.3e}" for m in E}, "E2/E1", f"{E.get(2,0)/E[1]:.3f}")
