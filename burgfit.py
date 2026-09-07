"""U3: fit the vorticity profile across the resolved front to the Rott-Lundgren (Burgers) layer omega = (DU/(sqrt(2 pi) delta)) exp(-(z-z0)^2/(2 delta^2)),
delta^2 = nu/a; report delta in Burgers units, the implied strain a(T-t) = nu (T-t)/delta^2, the fitted vs measured velocity jump, and the residual."""
import sys, glob, numpy as np
sys.path.insert(0, ".")
from axiphys import AxiPhys
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
T = float(sys.argv[1]); nu = 5e-4
for f in sorted(glob.glob(sys.argv[2])):
    d = np.load(f); U, Om, Ps, r, z, t = d["U"], d["Om"], d["Ps"], d["r"], d["z"], float(d["t"])
    zm = d["zmap"] if "zmap" in d.files else None; zm = tuple(float(v) for v in zm)
    P = AxiPhys(len(r), len(z) + 1, a=5.0); P.set_zmap(zm); P.r = r.copy()
    rr = np.concatenate([-r[:0:-1], r]); Psz = P.d_z(Ps); ur = -r[:, None] * Psz
    W = r[:, None] * Om; iw, jw = np.unravel_index(int(np.argmax(np.abs(W))), W.shape)
    col = W[iw, :]; sgn = np.sign(col[jw]); col = col * sgn; zc = z; Tt = T - t; dB = np.sqrt(nu * Tt)
    half = col > 0.5 * col[jw]; jl = jw
    while jl > 0 and half[jl - 1]: jl -= 1
    jr = jw
    while jr < len(z) - 1 and half[jr + 1]: jr += 1
    fw = z[jr] - z[jl]; win = (z > z[jw] - 3 * fw) & (z < z[jw] + 3 * fw)
    zz, ww = z[win], col[win]
    def model(p): return p[0] * np.exp(-(zz - p[1]) ** 2 / (2 * p[2] ** 2)) + p[3]
    p0 = [col[jw], z[jw], fw / 2.355, 0.0]
    res = least_squares(lambda p: model(p) - ww, p0)
    A, z0, dl, b = res.x; dl = abs(dl)
    rms = np.sqrt(np.mean((model(res.x) - ww) ** 2)) / A
    a_impl = nu / dl ** 2; DU_fit = A * np.sqrt(2 * np.pi) * dl
    jm = int(np.argmin(np.abs(z - (z0 - 3 * dl)))); jp = int(np.argmin(np.abs(z - (z0 + 3 * dl))))
    DU_meas = abs(ur[iw, jp] - ur[iw, jm]); DU_int = abs(np.trapezoid(col[jm:jp + 1], z[jm:jp + 1]))
    print(f"{f}: t {t:.9f} T-t {Tt:.2e}  front at (r {r[iw]:.5f}, z0 {z0:.6f})  cells in +-3 delta: {jp - jm}")
    print(f"   Gaussian fit: delta {dl:.3e} = {dl / dB:.3f} sqrt(nu(T-t))  (FWHM/sqrt(nu(T-t)) {2.355 * dl / dB:.2f});  implied strain a(T-t) = nu(T-t)/delta^2 = {a_impl * Tt:.2f};  relative RMS residual {rms:.3f};  baseline/peak {b / A:+.3f}")
    print(f"   velocity jump: fit DU = sqrt(2 pi) delta omega_peak = {DU_fit:.1f}, measured u_r jump across +-3 delta {DU_meas:.1f}, integral of omega_theta dz {DU_int:.1f};  DU sqrt(T-t): fit {DU_fit * np.sqrt(Tt):.3f}, measured {DU_meas * np.sqrt(Tt):.3f};  sheet Re = DU/sqrt(nu a) with a = 1/(T-t): {DU_meas * np.sqrt(Tt) / np.sqrt(nu):.0f}", flush=True)
