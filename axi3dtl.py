"""AXI3DTL (T15, lit 148 protocol): TANGENT-LINEAR run -- the nonlinear axisymmetric base (AxiPhys, Hou-Luo variables) and the
linear mode-m perturbation (axi3dlin operator) are marched TOGETHER by one RK4, the perturbation linearized about the
instantaneous base at every stage.  Growth is measured relative to the collapsing base: the local exponent
lambda = d ln(a_pert / a_base) / d ln A (a = sqrt of the energy in the core box, A = max |u1|), so lambda > 0 means the
perturbation outruns the collapse (A ~ 1/(T-t): a_pert/a_base ~ (T-t)^{-lambda}).  Also printed: sigma (T-t) with T_est,
the translation-mode overlap (m = 1), the kz centroid and the production split.
   python axi3dtl.py <snapshot on a uniform-z AxiPhys grid> <T_est> <m> [steps]     env: AXP_A (map), AXP_FILTER=1 (both fields)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.fft import fft, ifft
from scipy.linalg import solve_banded
from axiphys import AxiPhys

snap = sys.argv[1]; T_est = float(sys.argv[2]); m = int(sys.argv[3]); nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 800
d = np.load(snap); U, Om, t = d["U"], d["Om"], float(d["t"])
n_r, n_zh = U.shape[0], U.shape[1] + 1
P = AxiPhys(n_r, n_zh, nu=float(os.environ["AXL_NU"]) if os.environ.get("AXL_NU") else (float(d["nu"]) if "nu" in d.files else 0.0), a=float(os.environ.get("AXP_A", "5.0")))   # AXL_NU: override the viscosity of base AND mode (Hou's second stage)
zm = tuple(float(v) for v in d["zmap"]) if "zmap" in d.files else (0.0, 0.0)
if os.environ.get("AXL_ZMAP"): zm = tuple(float(v) for v in os.environ["AXL_ZMAP"].split(","))
if zm[0] > 0: P.set_zmap(zm)
assert np.abs(d["z"] - P.z).max() < 1e-10, "snapshot z grid differs from AxiPhys(n_z) with zmap %s" % (zm,)
if np.abs(d["r"] - P.r).max() > 1e-12:                      # moving-map snapshot: install its radial grid as a FIXED map (metrics by spline in eta)
    rs = CubicSpline(P.eta, d["r"]); P.r = d["r"].copy(); P.drde = rs(P.eta, 1); P.d2rde = rs(P.eta, 2); P.params = ("fixed",)
    P._lu = [P._poisson_matrix(k) for k in np.sqrt(np.maximum(P.lam_z, 0.0))]
    print(f"  installed the snapshot's radial grid (min dr {np.diff(P.r).min():.2e}) as a fixed map", flush=True)
r = P.r; N = 2 * n_zh; z = np.arange(N) / N
nu = P.nu
if zm[0] > 0:                                              # zeta(z) for the sine-series transfer of the mapped base to the uniform full period
    beta, mm = zm; fine = np.linspace(0.0, 0.5, 200001); g = (1.0 - beta * np.cos(4 * np.pi * fine)) ** mm
    Zf = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(fine))]); Zf *= 0.5 / Zf[-1]
    zhalf = np.where(z <= 0.5, z, 1.0 - z); zeta_half = np.interp(zhalf, Zf, fine)
    S_T = np.sin(2.0 * np.pi * np.arange(1, n_zh)[:, None] * zeta_half[None, :])       # (n_zh-1, N)
filt = os.environ.get("AXP_FILTER", "0") == "1"
# ---------------- perturbation machinery (as axi3dlin) ----------------
from scipy.fft import dst
def full(Fh, parity):
    if zm[0] > 0:
        sgn = np.where(z <= 0.5, 1.0, -1.0 if parity == "odd" else 1.0)
        return ((dst(Fh, type=1, axis=1) / n_zh) @ S_T) * sgn[None, :]
    F = np.zeros((n_r, N)); F[:, 1:n_zh] = Fh
    F[:, n_zh+1:] = (-Fh[:, ::-1] if parity == "odd" else Fh[:, ::-1])
    if parity == "even":
        F[:, 0] = 2 * F[:, 1] - F[:, 2]; F[:, n_zh] = 2 * F[:, n_zh-1] - F[:, n_zh-2]
    return F
kz = 2 * np.pi * np.fft.fftfreq(N, d=1.0 / N)
def dz(F): return ifft(1j * kz[None, :] * fft(F, axis=1), axis=1)
eta = np.linspace(0.0, 1.0, n_r); h_eta = eta[1] - eta[0]
drde = CubicSpline(eta, r)(eta, 1)[:, None]
def dr(F, parity):
    s = 1.0 if parity == "even" else -1.0
    Fe = np.concatenate([s * F[2:0:-1], F], axis=0)
    D = np.empty_like(F)
    D[:-2] = (Fe[0:-4] - 8 * Fe[1:-3] + 8 * Fe[3:-1] - Fe[4:]) / (12 * h_eta)
    D[-2] = (F[-1] - F[-3]) / (2 * h_eta); D[-1] = (3 * F[-1] - 4 * F[-2] + F[-3]) / (2 * h_eta)
    return D / drde
par_rt = "even" if m % 2 == 1 else "odd"; par_zp = "odd" if m % 2 == 1 else "even"
pe = m - 1 if m >= 1 else 1; pz = m if m >= 1 else 0          # axis exponents of the seed: u_r, u_th ~ r^{m-1} (m >= 1) or r (m = 0); u_z ~ r^m
r_ = r[:, None]
with np.errstate(divide="ignore", invalid="ignore"):
    inv_r_vec = np.where(r > 0, 1.0 / np.maximum(r, 1e-300), 0.0)
inv_r = inv_r_vec[:, None]
def over_r(F, parity_of_F):
    G = F * inv_r; G[0] = dr(F, parity_of_F)[0] if parity_of_F == "odd" else 0.0; return G
I_n = np.eye(n_r); Dzp = dr(I_n, par_zp); Drt = dr(I_n, par_rt); Rinv = np.diag(inv_r_vec)
OVR = Rinv.copy(); OVR[0, :] = (Drt[0, :] if par_rt == "odd" else 0.0)
Lm = Drt @ Dzp + OVR @ Dzp - (m * m) * (OVR @ Rinv)
BW = 4
def to_banded(A):
    n = A.shape[0]; ab = np.zeros((2 * BW + 1, n))
    for k in range(-BW, BW + 1):
        dg = np.diagonal(A, k)
        if k >= 0: ab[BW - k, k:] = dg
        else: ab[BW - k, :n + k] = dg
    return ab
lus = []
for k in kz:
    A = -Lm + (k * k) * I_n
    if par_zp == "odd": A[0, :] = 0.0; A[0, 0] = 1.0
    A[-1, :] = Dzp[-1, :]
    if m == 0 and abs(k) < 1e-12: A[0, :] = 0.0; A[0, 0] = 1.0   # m = 0, k_z = 0: the Neumann problem is singular; pin p(0) = 0 (gauge)
    lus.append(to_banded(A))
pin = [(m == 0 and abs(k) < 1e-12) for k in kz]
def poisson(rhs, wall):
    rk = fft(rhs, axis=1); wk = fft(wall); out = np.empty_like(rk)
    for j in range(N):
        b = rk[:, j].copy(); b[-1] = wk[j]
        if par_zp == "odd" or pin[j]: b[0] = 0.0
        out[:, j] = solve_banded((BW, BW), lus[j], b, check_finite=False)
    return ifft(out, axis=1)
def grad(p): return dr(p, par_zp), 1j * m * over_r(p, par_zp), dz(p)
def div(vr, vt, vz): return dr(vr, par_rt) + over_r(vr + 1j * m * vt, par_rt) + dz(vz)
def base_fields(Uh, Psh):
    """instantaneous base on the full period: u1, Psz, U_r, U_th, U_z and their gradients"""
    B = {}
    u1 = full(Uh, "odd"); Ps = full(Psh, "odd")
    B["u1"] = u1; B["Psz"] = dz(Ps).real; Psr = dr(Ps, "even")
    B["Uth"] = r_ * u1; B["Ur"] = -r_ * B["Psz"]; B["Uz"] = 2.0 * Ps + r_ * Psr
    B["dUr_r"] = dr(B["Ur"], "odd"); B["dUr_z"] = dz(B["Ur"]).real; B["dUth_r"] = dr(B["Uth"], "odd"); B["dUth_z"] = dz(B["Uth"]).real
    B["dUz_r"] = dr(B["Uz"], "even"); B["dUz_z"] = dz(B["Uz"]).real
    return B
def rhs_pert(ur, ut, uz, B):
    def D(F, parity): return B["Ur"] * dr(F, parity) + B["Uz"] * dz(F) + 1j * m * B["u1"] * F
    Nr = D(ur, par_rt) + ur * B["dUr_r"] + uz * B["dUr_z"] - 2.0 * B["u1"] * ut
    Nt = D(ut, par_rt) + ur * B["dUth_r"] + uz * B["dUth_z"] + B["u1"] * ur - B["Psz"] * ut
    Nz = D(uz, par_zp) + ur * B["dUz_r"] + uz * B["dUz_z"]
    if nu > 0:                                              # viscous term: u_+- = u_r +- i u_th obey Lap_{m+-1}, u_z obeys Lap_m
        up = ur + 1j * ut; um = ur - 1j * ut
        Lp = lap_k(up, m + 1, par_rt); Lmn = lap_k(um, m - 1, par_rt); Lz = lap_k(uz, m, par_zp)
        Nr = Nr - nu * 0.5 * (Lp + Lmn); Nt = Nt - nu * 0.5 * (Lp - Lmn) / 1j; Nz = Nz - nu * Lz
    p = poisson(div(Nr, Nt, Nz), -Nr[-1]); gr, gt, gz = grad(p)
    return -Nr - gr, -Nt - gt, -Nz - gz
def lap_k(F, k, parity):
    """F_rr + F_r/r - k^2 F/r^2 + F_zz for a field of the given r-parity (behaving like r^|k| at the axis); axis row by the limit"""
    Fr = dr(F, parity); Frr = dr(Fr, "odd" if parity == "even" else "even")
    G = Frr + Fr * inv_r - (k * k) * F * inv_r * inv_r + dz(dz(F))
    G[0] = (2.0 * Frr[0] + dz(dz(F))[0]) if k == 0 else 0.0
    return G
wgt = r_ * drde * h_eta / N
def energy(ur, ut, uz): return float(np.sum((np.abs(ur) ** 2 + np.abs(ut) ** 2 + np.abs(uz) ** 2) * wgt))
def filt_pert(F, parity):
    """Hou-Li filter on the Fourier coefficients (as the base's filter_z) + the base's 8th-order radial filter with the field's parity fold"""
    Fk = fft(F, axis=1); kk = np.abs(kz) / np.abs(kz).max(); Fk *= np.exp(-36.0 * kk ** 36)[None, :]; F = ifft(Fk, axis=1)
    c = np.array([1, -8, 28, -56, 70, -56, 28, -8, 1]) / 256.0; s = 1.0 if parity == "even" else -1.0
    Fe = np.concatenate([s * F[4:0:-1], F], axis=0); G = F.copy()
    for i in range(0, n_r - 4):
        G[i] = F[i] - sum(c[k] * Fe[i + k] for k in range(9))
    return G
# ---------------- initial perturbation ----------------
dU, dO, Ps, ur0, uz0 = P.rhs(U, Om)
B = base_fields(U, Ps)
i0, j0 = np.unravel_index(int(np.argmax(np.abs(B["u1"]))), B["u1"].shape); R0, Z0 = r[i0], z[j0]
env = np.exp(-((r_ - R0) / (0.5 * R0)) ** 2) * (np.exp(-((z[None, :] - Z0) / (2 * Z0)) ** 2) + np.exp(-((z[None, :] - 1 + Z0) / (2 * Z0)) ** 2))
ur = env * (r_ / R0) ** pe * (1 + 0.1j); ut = 1j * ur * (1.0 + 0.5 * (r_ / R0) ** 2); uz = (env * (r_ / R0) ** pz * 0.3).astype(complex)
if os.environ.get("AXL_SEED", "") == "random":            # smooth random seed under the same envelope (fast short-wave family, RESULT 06:10)
    rng = np.random.default_rng(int(os.environ.get("AXL_RNG", "1")))
    def smooth_noise():
        Fk = rng.standard_normal((n_r, N)) + 1j * rng.standard_normal((n_r, N))
        Fk *= np.exp(-(np.abs(kz) / (2 * np.pi * 40)) ** 2)[None, :]; F = ifft(Fk, axis=1)
        for _ in range(6): F[1:-1] = 0.25 * F[:-2] + 0.5 * F[1:-1] + 0.25 * F[2:]
        return F
    ur = env * (r_ / R0) ** pe * smooth_noise(); ut = env * (r_ / R0) ** pe * smooth_noise(); uz = env * (r_ / R0) ** pz * smooth_noise()
    ur[0] = 0.5 * (ur[0] - 1j * ut[0]); ut[0] = 1j * ur[0]
p = poisson(-div(ur, ut, uz), ur[-1]); gr, gt, gz = grad(p); ur -= gr; ut -= gt; uz -= gz
def base_core(B):
    """sqrt of the base energy in the core box r < 3 R, |z| < 3 Z (mirror-periodic) and A = max |u1|"""
    A = float(np.abs(B["u1"]).max()); i, j = np.unravel_index(int(np.argmax(np.abs(B["u1"]))), B["u1"].shape)
    zz = np.minimum(z, 1 - z); box = (r_ < 3 * r[i]) & (zz[None, :] < 3 * zz[j])
    Eb = float(np.sum((B["Ur"] ** 2 + B["Uz"] ** 2 + B["Uth"] ** 2) * box * wgt))
    return A, np.sqrt(Eb), r[i], zz[j]
E0 = energy(ur, ut, uz); A0, ab0, _, _ = base_core(B)
dr_min = float(np.diff(r).min()); dzz = 1.0 / N
print(f"AXI3DTL m={m} tangent-linear from {os.path.basename(snap)} t0 {t:.6f} (T_est {T_est}, T-t {T_est-t:.2e}); grid {n_r} x {N}; zmap {zm}; nu {nu}; filter {filt}; A0 {A0:.4e} at ({R0:.4f}, {Z0:.4f}); E0 {E0:.3e} div0 {np.abs(div(ur, ut, uz)[:-1]).max():.2e}", flush=True)
tt0 = t; it = 0; t_start = time.time(); Elast, Alast, ablast, tlast = E0, A0, ab0, t; Eperp_last = E0; lnratio0 = 0.5 * np.log(E0) - np.log(ab0)
prev = (ur.copy(), ut.copy(), uz.copy())          # for the precession frequency: phase advance of <prev, cur> per unit time; alpha = omega (T-t) is Pineau-Vicol's rotation parameter
while it < nsteps:
    dU, dO, Ps, ur_b, uz_b = P.rhs(U, Om); B1 = base_fields(U, Ps)
    vmax = max(np.abs(ur_b).max(), np.abs(uz_b).max(), np.abs(B1["Uth"]).max(), 1e-12)
    dt = min(0.3 * min(dr_min, dzz) / vmax, 2e-6)
    k1b = (dU, dO); k1p = rhs_pert(ur, ut, uz, B1)
    U2, O2 = U + 0.5 * dt * k1b[0], Om + 0.5 * dt * k1b[1]; d2 = P.rhs(U2, O2); B2 = base_fields(U2, d2[2]); k2b = d2[:2]
    k2p = rhs_pert(ur + 0.5 * dt * k1p[0], ut + 0.5 * dt * k1p[1], uz + 0.5 * dt * k1p[2], B2)
    U3, O3 = U + 0.5 * dt * k2b[0], Om + 0.5 * dt * k2b[1]; d3 = P.rhs(U3, O3); B3 = base_fields(U3, d3[2]); k3b = d3[:2]
    k3p = rhs_pert(ur + 0.5 * dt * k2p[0], ut + 0.5 * dt * k2p[1], uz + 0.5 * dt * k2p[2], B3)
    U4, O4 = U + dt * k3b[0], Om + dt * k3b[1]; d4 = P.rhs(U4, O4); B4 = base_fields(U4, d4[2]); k4b = d4[:2]
    k4p = rhs_pert(ur + dt * k3p[0], ut + dt * k3p[1], uz + dt * k3p[2], B4)
    U = U + dt / 6 * (k1b[0] + 2 * k2b[0] + 2 * k3b[0] + k4b[0]); Om = Om + dt / 6 * (k1b[1] + 2 * k2b[1] + 2 * k3b[1] + k4b[1])
    ur = ur + dt / 6 * (k1p[0] + 2 * k2p[0] + 2 * k3p[0] + k4p[0]); ut = ut + dt / 6 * (k1p[1] + 2 * k2p[1] + 2 * k3p[1] + k4p[1]); uz = uz + dt / 6 * (k1p[2] + 2 * k2p[2] + 2 * k3p[2] + k4p[2])
    if filt:
        U = P.filter_r(P.filter_z(U)); Om = P.filter_r(P.filter_z(Om))
        ur = filt_pert(ur, par_rt); ut = filt_pert(ut, par_rt); uz = filt_pert(uz, par_zp)
    t += dt; it += 1
    if it % max(1, nsteps // 40) == 0 or it == nsteps:
        E = energy(ur, ut, uz); Bn = base_fields(U, P.rhs(U, Om)[2]); A, ab, Rb, Zb = base_core(Bn)
        sig = np.log(E / Elast) / (2 * (t - tlast)); lam = (0.5 * np.log(E / Elast) - np.log(ab / ablast)) / np.log(A / Alast)
        lam_cum = (0.5 * np.log(E) - np.log(ab) - lnratio0) / np.log(A / A0)
        ip = np.sum((np.conj(prev[0]) * ur + np.conj(prev[1]) * ut + np.conj(prev[2]) * uz) * wgt); omega = float(np.angle(ip)) / (t - tlast); prev = (ur.copy(), ut.copy(), uz.copy())
        Ek = (np.abs(fft(ur, axis=1)) ** 2 + np.abs(fft(ut, axis=1)) ** 2 + np.abs(fft(uz, axis=1)) ** 2) * r_ * drde
        kcen = float(np.sum(np.abs(kz)[None, :] * Ek) / np.sum(Ek)) / (2 * np.pi)
        ia, ja = np.unravel_index(int(np.argmax(np.abs(ur) ** 2 + np.abs(ut) ** 2 + np.abs(uz) ** 2)), ur.shape)
        if m == 1:
            Tr = Bn["dUr_r"] - 1j * over_r(Bn["Uth"], "odd"); Tt = Bn["dUth_r"] + 1j * over_r(Bn["Ur"], "odd"); Tz = Bn["dUz_r"]
            num = np.sum((np.conj(Tr) * ur + np.conj(Tt) * ut + np.conj(Tz) * uz) * wgt); trans = abs(num) / np.sqrt(np.sum((np.abs(Tr) ** 2 + np.abs(Tt) ** 2 + np.abs(Tz) ** 2) * wgt) * E)
        elif m == 0:                                     # time-translation mode d_t U (exact tangent-linear solution; relative exponent 1 on a Type I base)
            Tr, Tt, Tz = (Bn["Ur"] - B1["Ur"]) / dt, (Bn["Uth"] - B1["Uth"]) / dt, (Bn["Uz"] - B1["Uz"]) / dt
            num = np.sum((np.conj(Tr) * ur + np.conj(Tt) * ut + np.conj(Tz) * uz) * wgt); trans = abs(num) / np.sqrt(np.sum((np.abs(Tr) ** 2 + np.abs(Tt) ** 2 + np.abs(Tz) ** 2) * wgt) * E)
        else: trans = 0.0
        Eperp = E * (1.0 - trans ** 2); sig_perp = 0.5 * np.log(Eperp / Eperp_last) / (t - tlast) if it > nsteps // 40 else sig; Eperp_last = Eperp   # rate of the translation-orthogonal part (energy projection, approximate for the non-normal operator; lit 163)
        Elast, Alast, ablast, tlast = E, A, ab, t
        P_sw = float(np.sum(np.real(np.conj(ur) * ut) * (Bn["u1"] - Bn["dUth_r"]) * wgt))
        P_tot = P_sw + float(np.sum(np.real(-np.conj(ur) * (ur * Bn["dUr_r"] + uz * Bn["dUr_z"]) - np.conj(uz) * (ur * Bn["dUz_r"] + uz * Bn["dUz_z"]) - np.conj(ut) * (uz * Bn["dUth_z"] - Bn["Psz"] * ut)) * wgt))
        print(f"  t {t:.10f} ({(t-tt0)/(T_est-tt0):.3f} of T-t0) it {it} dt {dt:.1e}  A {A:.4e} (x{A/A0:.3f}) at ({Rb:.4f},{Zb:.4f})  E/E0 {E/E0:.3e}  sigma(T-t) {sig*(T_est-t):.3f}  lambda {lam:+.3f} (cum {lam_cum:+.3f})  |div| {np.abs(div(ur, ut, uz)[:-1]).max():.1e}  peak ({r[ia]:.4f},{z[ja]:.4f})  kz {kcen:.1f}  prod swirl/total {P_sw/max(P_tot,1e-300):.2f}  trans {trans:.3f}  sigma_perp(T-t) {sig_perp*(T_est-t):.3f}  omega(T-t) {omega*(T_est-t):+.3f}  ({time.time()-t_start:.0f}s)", flush=True)
        if not (np.isfinite(E) and np.isfinite(A)): break
np.savez(f"axi3dtl_m{m}_{os.path.basename(snap).replace('.npz','')}.npz", U=U, Om=Om, ur=ur, ut=ut, uz=uz, r=r, z=z, t=t)
