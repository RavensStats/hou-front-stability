"""AXI3DNL (T15 nonlinear stage; lit 157 design 3): axisymmetric base in Hou-Luo variables (AxiPhys: u1, omega1, psi1)
PLUS azimuthal modes m = 1..M-1 of the velocity (u_r, u_th, u_z)_m e^{i m theta} (real field: u_{-m} = conj u_m), all
marched by one RK4.  Mode equations: d_t u_m = -[L_m(U0) u_m]  -  (u'.grad u')_m  -  grad_m p_m  +  nu Lap_m u_m, where
L_m(U0) is the validated tangent-linear operator (axi3dlin/axi3dtl) and (u'.grad u')_m is the mode-mode convolution
evaluated pseudo-spectrally in theta on N_th = 3M points (dealiased).  Base feedback: the m = 0 component of -(u'.grad u')
forces u1 (theta-momentum / r) and omega1 (curl_theta / r).  Diagnostics: base A = max u1 and location, mode energies,
the 3D maximum of u_theta / r over theta (base + modes), the m = 0 forcing size, sigma (T-t) of the m = 1 energy.
   python axi3dnl.py <snapshot> <T_est> <M> <eps> [steps]     (eps: seed amplitude of m = 1 relative to max |U_theta|)
   env: AXL_ZMAP, AXP_A, AXL_SEED=random"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.fft import fft, ifft, dst, rfft, irfft
from scipy.linalg import solve_banded
from axiphys import AxiPhys

snap = sys.argv[1]; T_est = float(sys.argv[2]); M = int(sys.argv[3]); eps = float(sys.argv[4]); nsteps = int(sys.argv[5]) if len(sys.argv) > 5 else 400
d = np.load(snap); U, Om, t = d["U"], d["Om"], float(d["t"])
n_r, n_zh = U.shape[0], U.shape[1] + 1
P = AxiPhys(n_r, n_zh, nu=float(d["nu"]) if "nu" in d.files else 0.0, a=float(os.environ.get("AXP_A", "5.0")))
zm = tuple(float(v) for v in d["zmap"]) if "zmap" in d.files else (0.0, 0.0)
if os.environ.get("AXL_ZMAP"): zm = tuple(float(v) for v in os.environ["AXL_ZMAP"].split(","))
if zm[0] > 0: P.set_zmap(zm)
assert np.abs(d["z"] - P.z).max() < 1e-10
if np.abs(d["r"] - P.r).max() > 1e-12:
    rs = CubicSpline(P.eta, d["r"]); P.r = d["r"].copy(); P.drde = rs(P.eta, 1); P.d2rde = rs(P.eta, 2); P.params = ("fixed",)
    P._lu = [P._poisson_matrix(k) for k in np.sqrt(np.maximum(P.lam_z, 0.0))]
r = P.r; N = 2 * n_zh; z = np.arange(N) / N; nu = P.nu
if zm[0] > 0:
    beta, mm = zm; fine = np.linspace(0.0, 0.5, 200001); g = (1.0 - beta * np.cos(4 * np.pi * fine)) ** mm
    Zf = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(fine))]); Zf *= 0.5 / Zf[-1]
    zhalf = np.where(z <= 0.5, z, 1.0 - z); zeta_half = np.interp(zhalf, Zf, fine)
    S_T = np.sin(2.0 * np.pi * np.arange(1, n_zh)[:, None] * zeta_half[None, :])
    # inverse: uniform full period -> the base's stored half-period mapped grid (for the m = 0 feedback); least-squares via the sine series
    S_back = np.sin(2.0 * np.pi * np.arange(1, n_zh)[None, :] * np.interp(P.z, Zf, fine)[:, None])   # (n_zh-1 nodes, n_zh-1 modes) at the stored zeta nodes
def full(Fh, parity):
    if zm[0] > 0:
        sgn = np.where(z <= 0.5, 1.0, -1.0 if parity == "odd" else 1.0)
        return ((dst(Fh, type=1, axis=1) / n_zh) @ S_T) * sgn[None, :]
    F = np.zeros((n_r, N)); F[:, 1:n_zh] = Fh
    F[:, n_zh+1:] = (-Fh[:, ::-1] if parity == "odd" else Fh[:, ::-1])
    if parity == "even":
        F[:, 0] = 2 * F[:, 1] - F[:, 2]; F[:, n_zh] = 2 * F[:, n_zh-1] - F[:, n_zh-2]
    return F
def half(F):
    """m = 0 real field on the uniform full period (odd about z = 0, 1/2) -> the base's stored half-period grid"""
    if zm[0] > 0:
        coef = -2.0 * np.imag(fft(F, axis=1))[:, 1:n_zh] / N            # sine coefficients on the full period: F = sum c_k sin(2 pi k z)
        return coef @ S_back.T
    return F[:, 1:n_zh].real
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
r_ = r[:, None]
with np.errstate(divide="ignore", invalid="ignore"):
    inv_r_vec = np.where(r > 0, 1.0 / np.maximum(r, 1e-300), 0.0)
inv_r = inv_r_vec[:, None]
def over_r(F, parity_of_F):
    G = F * inv_r; G[0] = dr(F, parity_of_F)[0] if parity_of_F == "odd" else 0.0; return G
def prt(m): return "even" if m % 2 == 1 else "odd"          # parity of u_r, u_th of mode m (m = 0: odd)
def pzp(m): return "odd" if m % 2 == 1 else "even"          # parity of u_z, p of mode m (m = 0: even)
I_n = np.eye(n_r); Rinv = np.diag(inv_r_vec); BW = 4
def to_banded(A):
    n = A.shape[0]; ab = np.zeros((2 * BW + 1, n))
    for k in range(-BW, BW + 1):
        dg = np.diagonal(A, k)
        if k >= 0: ab[BW - k, k:] = dg
        else: ab[BW - k, :n + k] = dg
    return ab
LUS = {}
for m in range(1, M):
    Dzp = dr(I_n, pzp(m)); Drt = dr(I_n, prt(m)); OVR = Rinv.copy(); OVR[0, :] = (Drt[0, :] if prt(m) == "odd" else 0.0)
    Lm = Drt @ Dzp + OVR @ Dzp - (m * m) * (OVR @ Rinv)
    lus = []
    for k in kz:
        A = -Lm + (k * k) * I_n
        if pzp(m) == "odd": A[0, :] = 0.0; A[0, 0] = 1.0
        A[-1, :] = Dzp[-1, :]
        lus.append(to_banded(A))
    LUS[m] = lus
def poisson(rhs, wall, m):
    rk = fft(rhs, axis=1); wk = fft(wall); out = np.empty_like(rk); lus = LUS[m]
    for j in range(N):
        b = rk[:, j].copy(); b[-1] = wk[j]
        if pzp(m) == "odd": b[0] = 0.0
        out[:, j] = solve_banded((BW, BW), lus[j], b, check_finite=False)
    return ifft(out, axis=1)
def grad(p, m): return dr(p, pzp(m)), 1j * m * over_r(p, pzp(m)), dz(p)
def div(vr, vt, vz, m): return dr(vr, prt(m)) + over_r(vr + 1j * m * vt, prt(m)) + dz(vz)
def lap_k(F, k, parity):
    Fr = dr(F, parity); Frr = dr(Fr, "odd" if parity == "even" else "even")
    G = Frr + Fr * inv_r - (k * k) * F * inv_r * inv_r + dz(dz(F))
    G[0] = (2.0 * Frr[0] + dz(dz(F))[0]) if k == 0 else 0.0
    return G
def base_fields(Uh, Psh):
    B = {}
    u1 = full(Uh, "odd"); Ps = full(Psh, "odd")
    B["u1"] = u1; B["Psz"] = dz(Ps).real; Psr = dr(Ps, "even")
    B["Uth"] = r_ * u1; B["Ur"] = -r_ * B["Psz"]; B["Uz"] = 2.0 * Ps + r_ * Psr
    B["dUr_r"] = dr(B["Ur"], "odd"); B["dUr_z"] = dz(B["Ur"]).real; B["dUth_r"] = dr(B["Uth"], "odd"); B["dUth_z"] = dz(B["Uth"]).real
    B["dUz_r"] = dr(B["Uz"], "even"); B["dUz_z"] = dz(B["Uz"]).real
    return B
def lin_N(ur, ut, uz, B, m):
    """advective coupling with the base, mode m (the tangent-linear N of axi3dtl)"""
    def D(F, parity): return B["Ur"] * dr(F, parity) + B["Uz"] * dz(F) + 1j * m * B["u1"] * F
    Nr = D(ur, prt(m)) + ur * B["dUr_r"] + uz * B["dUr_z"] - 2.0 * B["u1"] * ut
    Nt = D(ut, prt(m)) + ur * B["dUth_r"] + uz * B["dUth_z"] + B["u1"] * ur - B["Psz"] * ut
    Nz = D(uz, pzp(m)) + ur * B["dUz_r"] + uz * B["dUz_z"]
    return Nr, Nt, Nz
Nth = 3 * M
def to_phys(coefs):
    """modes m = 1..M-1 (complex arrays) -> real field on N_th theta points: F(theta_j) = sum_m 2 Re(F_m e^{i m theta_j})"""
    C = np.zeros((Nth // 2 + 1,) + coefs[1].shape, dtype=complex)
    for m in range(1, M): C[m] = coefs[m]
    return irfft(C * Nth, n=Nth, axis=0)
def to_modes(F):
    C = rfft(F, axis=0) / Nth
    return {m: C[m] for m in range(1, M)}, C[0].real
def conv(modes):
    """(u'.grad u') for the modes-only field u' = sum_{m != 0} u_m e^{i m theta}: returns per-mode components m = 1..M-1 and the m = 0 part"""
    fld = {}
    for name in ("ur", "ut", "uz"):
        fld[name] = to_phys({m: modes[m][name] for m in range(1, M)})
        fld[name + "_r"] = to_phys({m: dr(modes[m][name], prt(m) if name != "uz" else pzp(m)) for m in range(1, M)})
        fld[name + "_z"] = to_phys({m: dz(modes[m][name]) for m in range(1, M)})
        fld[name + "_t"] = to_phys({m: 1j * m * modes[m][name] for m in range(1, M)})
    ur, ut, uz = fld["ur"], fld["ut"], fld["uz"]; ut_r = ut * inv_r[None]          # u_theta / r
    Nr = ur * fld["ur_r"] + ut_r * fld["ur_t"] + uz * fld["ur_z"] - ut * ut_r
    Nt = ur * fld["ut_r"] + ut_r * fld["ut_t"] + uz * fld["ut_z"] + ur * ut_r
    Nz = ur * fld["uz_r"] + ut_r * fld["uz_t"] + uz * fld["uz_z"]
    (Nr_m, Nr0), (Nt_m, Nt0), (Nz_m, Nz0) = to_modes(Nr), to_modes(Nt), to_modes(Nz)
    # axis rows: u_theta / r products are evaluated with inv_r = 0 at r = 0; the true axis values follow by regularity (fields ~ r^{|m|-1}); leave 0 there
    return {m: (Nr_m[m], Nt_m[m], Nz_m[m]) for m in range(1, M)}, (Nr0, Nt0, Nz0)
def rhs_all(Uh, Omh, modes):
    dU, dO, Ps, ur_b, uz_b = P.rhs(Uh, Omh); B = base_fields(Uh, Ps)
    NL, N0 = conv(modes)
    out = {}
    for m in range(1, M):
        ur, ut, uz = modes[m]["ur"], modes[m]["ut"], modes[m]["uz"]
        Nr, Nt, Nz = lin_N(ur, ut, uz, B, m)
        Nr = Nr + NL[m][0]; Nt = Nt + NL[m][1]; Nz = Nz + NL[m][2]
        if nu > 0:
            up = ur + 1j * ut; um = ur - 1j * ut
            Lp = lap_k(up, m + 1, prt(m)); Lmn = lap_k(um, m - 1, prt(m)); Lz = lap_k(uz, m, pzp(m))
            Nr = Nr - nu * 0.5 * (Lp + Lmn); Nt = Nt - nu * 0.5 * (Lp - Lmn) / 1j; Nz = Nz - nu * Lz
        p = poisson(div(Nr, Nt, Nz, m), -Nr[-1], m); gr, gt, gz = grad(p, m)
        out[m] = {"ur": -Nr - gr, "ut": -Nt - gt, "uz": -Nz - gz}
    # m = 0 feedback on the base: f = -N0;  d_t u1 += f_theta / r ;  d_t omega1 += (d_z f_r - d_r f_z) / r
    fr, ft, fz = -N0[0], -N0[1], -N0[2]
    fu1 = over_r(ft, "odd"); fom = over_r(dz(fr).real - dr(fz, "even"), "odd")
    dU = dU + half(fu1); dO = dO + half(fom)
    return dU, dO, out, B, (float(np.abs(fu1).max()), float(np.abs(fom).max()))
def energy(mode): return float(np.sum((np.abs(mode["ur"]) ** 2 + np.abs(mode["ut"]) ** 2 + np.abs(mode["uz"]) ** 2) * wgt))
wgt = r_ * drde * h_eta / N
# ---------------- seed: mode 1 (others zero), amplitude eps relative to max |U_theta| ----------------
dU, dO, Ps, _, _ = P.rhs(U, Om); B = base_fields(U, Ps)
i0, j0 = np.unravel_index(int(np.argmax(np.abs(B["u1"]))), B["u1"].shape); R0, Z0 = r[i0], z[j0]
env = np.exp(-((r_ - R0) / (0.5 * R0)) ** 2) * (np.exp(-((z[None, :] - Z0) / (2 * Z0)) ** 2) + np.exp(-((z[None, :] - 1 + Z0) / (2 * Z0)) ** 2))
modes = {m: {"ur": np.zeros((n_r, N), complex), "ut": np.zeros((n_r, N), complex), "uz": np.zeros((n_r, N), complex)} for m in range(1, M)}
ur = env * (1 + 0.1j); ut = 1j * ur * (1.0 + 0.5 * (r_ / R0) ** 2); uz = (env * (r_ / R0) * 0.3).astype(complex)
p = poisson(-div(ur, ut, uz, 1), ur[-1], 1); gr, gt, gz = grad(p, 1); ur -= gr; ut -= gt; uz -= gz
Uth_max = float(np.abs(B["Uth"]).max()); scale = eps * Uth_max / float(np.sqrt(np.abs(ur) ** 2 + np.abs(ut) ** 2 + np.abs(uz) ** 2).max())
modes[1]["ur"], modes[1]["ut"], modes[1]["uz"] = ur * scale, ut * scale, uz * scale
if os.environ.get("AXL_RESTART"):                            # continue a finished run: load base and modes from its saved state
    dR = np.load(os.environ["AXL_RESTART"]); U, Om, t = dR["U"], dR["Om"], float(dR["t"])
    for mm in range(1, M):
        for f in ("ur", "ut", "uz"):
            if f"{f}{mm}" in dR.files: modes[mm][f] = dR[f"{f}{mm}"]      # modes absent from the saved state (higher M) start at zero
    dU, dO, Ps, _, _ = P.rhs(U, Om); B = base_fields(U, Ps)
    print(f"  restart from {os.environ['AXL_RESTART']} at t = {t:.7f}", flush=True)
E0 = energy(modes[1]); A0 = float(np.abs(B["u1"]).max())
dr_min = float(np.diff(r).min()); dzz = 1.0 / N
def add(modes, k, c):
    return {m: {f: modes[m][f] + c * k[m][f] for f in ("ur", "ut", "uz")} for m in range(1, M)}
print(f"AXI3DNL M={M} (N_th {Nth}) eps {eps} from {os.path.basename(snap)} t0 {t:.6f} (T_est {T_est}); grid {n_r} x {N}; zmap {zm}; nu {nu}; A0 {A0:.4e} at ({R0:.4f},{Z0:.4f}); max|U_th| {Uth_max:.3e}; E1 {E0:.3e}", flush=True)
tt0 = t; it = 0; t_start = time.time(); Elast, tlast = E0, t
while it < nsteps:
    d1 = rhs_all(U, Om, modes); B1 = d1[3]
    vmax = max(np.abs(B1["Ur"]).max(), np.abs(B1["Uz"]).max(), np.abs(B1["Uth"]).max(), 1e-12)
    dt = min(0.3 * min(dr_min, dzz) / vmax, 2e-6)
    U2, O2, M2 = U + 0.5 * dt * d1[0], Om + 0.5 * dt * d1[1], add(modes, d1[2], 0.5 * dt); d2 = rhs_all(U2, O2, M2)
    U3, O3, M3 = U + 0.5 * dt * d2[0], Om + 0.5 * dt * d2[1], add(modes, d2[2], 0.5 * dt); d3 = rhs_all(U3, O3, M3)
    U4, O4, M4 = U + dt * d3[0], Om + dt * d3[1], add(modes, d3[2], dt); d4 = rhs_all(U4, O4, M4)
    U = U + dt / 6 * (d1[0] + 2 * d2[0] + 2 * d3[0] + d4[0]); Om = Om + dt / 6 * (d1[1] + 2 * d2[1] + 2 * d3[1] + d4[1])
    modes = {m: {f: modes[m][f] + dt / 6 * (d1[2][m][f] + 2 * d2[2][m][f] + 2 * d3[2][m][f] + d4[2][m][f]) for f in ("ur", "ut", "uz")} for m in range(1, M)}
    t += dt; it += 1
    if it % max(1, nsteps // 40) == 0 or it == nsteps:
        Ps = P.rhs(U, Om)[2]; B = base_fields(U, Ps); A = float(np.abs(B["u1"]).max()); i, j = np.unravel_index(int(np.argmax(np.abs(B["u1"]))), B["u1"].shape)
        E = {m: energy(modes[m]) for m in range(1, M)}; sig = np.log(E[1] / Elast) / (2 * (t - tlast)); Elast, tlast = E[1], t
        uth3d = B["Uth"][None] + to_phys({m: modes[m]["ut"] for m in range(1, M)})      # (N_th, n_r, N): swirl velocity over theta
        # 3D vorticity (lit 170 diagnostic): base (omega_r, omega_th, omega_z) = (-r d_z u1, r Om, 2 u1 + r d_r u1) plus the modes' curl
        w_r0 = -r_ * dz(B["u1"]).real; w_t0 = r_ * full(Om, "odd"); w_z0 = 2.0 * B["u1"] + r_ * dr(B["u1"], "odd")
        cr = {m: 1j * m * over_r(modes[m]["uz"], pzp(m)) - dz(modes[m]["ut"]) for m in range(1, M)}
        ct = {m: dz(modes[m]["ur"]) - dr(modes[m]["uz"], pzp(m)) for m in range(1, M)}
        cz = {m: over_r(dr(r_ * modes[m]["ut"], "odd" if prt(m) == "even" else "even"), "odd" if prt(m) == "even" else "even") - 1j * m * over_r(modes[m]["ur"], prt(m)) for m in range(1, M)}
        w3 = np.sqrt((w_r0[None] + to_phys(cr)) ** 2 + (w_t0[None] + to_phys(ct)) ** 2 + (w_z0[None] + to_phys(cz)) ** 2)
        w3max = float(w3.max()); w0max = float(np.sqrt(w_r0 ** 2 + w_t0 ** 2 + w_z0 ** 2).max())
        kk = np.unravel_index(int(np.argmax(w3)), w3.shape); w3loc = (r[kk[1]], z[kk[2]], kk[0])
        uth_max3d = float(np.abs(uth3d).max()); uth_max0 = float(np.abs(B["Uth"]).max())
        dO_base = float(np.abs(P.rhs(U, Om)[1]).max())
        Es = " ".join(f"E{m} {E[m]:.2e}" for m in range(1, M))
        print(f"  t {t:.10f} ({(t-tt0)/(T_est-tt0):.3f} of T-t0) it {it} dt {dt:.1e}  A {A:.4e} (x{A/A0:.3f}) at ({r[i]:.4f},{z[j]:.4f})  max_theta|u_th| {uth_max3d:.4e} (x{uth_max3d/uth_max0:.3f} of the m=0 max {uth_max0:.3e})  |omega|3D max {w3max:.3e} (x{w3max/w0max:.3f} of the m=0 max) at (r,z,j_th) = ({w3loc[0]:.4f},{w3loc[1]:.4f},{w3loc[2]})  {Es}  sigma1(T-t) {sig*(T_est-t):.3f}  feedback |f_u1| {d1[4][0]:.2e} |f_om| {d1[4][1]:.2e} (base |dOm| {dO_base:.2e})  |div1| {np.abs(div(modes[1]['ur'], modes[1]['ut'], modes[1]['uz'], 1)[:-1]).max():.1e}  ({time.time()-t_start:.0f}s)", flush=True)
        if not np.isfinite(A) or not np.isfinite(E[1]): break
np.savez(f"axi3dnl_M{M}_eps{eps}_{os.path.basename(snap).replace('.npz','')}{os.environ.get('AXL_TAG','')}.npz", U=U, Om=Om, r=r, z=z, t=t, **{f"{f}{m}": modes[m][f] for m in range(1, M) for f in ("ur", "ut", "uz")})
