"""AXI3DLIN (T15 stage 2): linear stability of a FROZEN axisymmetric swirling Euler state to the azimuthal mode m >= 1
(lit checks 136-138), on the snapshot's (r, z) grid with the full z period [0, 1) (Fourier).  Raw perturbation
variables (u_r', u_th', u_z', p') e^{i m theta}; axis regularity enters through the r-parity of each field
(u_r', u_th' ~ r^{|m|-1}: parity (-1)^{m-1}; u_z', p' ~ r^{|m|}: parity (-1)^m), imposed by the fold in the
4th-order stencils, and p'(0) = 0 for m >= 1.  Base: U_theta = r u1, U_r = -r psi1_z, U_z = 2 psi1 + r psi1_r.
  D F = U_r d_r F + U_z d_z F + i m u1 F
  N_r = D u_r' + u_r' d_r U_r + u_z' d_z U_r - 2 u1 u_th'
  N_th = D u_th' + u_r' d_r U_th + u_z' d_z U_th + u1 u_r' - psi1_z u_th'
  N_z = D u_z' + u_r' d_r U_z + u_z' d_z U_z
  d_t u' = -N - grad p',  grad p' = (d_r p', i m p'/r, d_z p'),  div v = d_r v_r + (v_r + i m v_th)/r + d_z v_z,
  -Lap_m p' = div N with the pressure operator built as the EXACT discrete divergence of the discrete gradient.
RK4 with projection at every stage; sigma = d ln E/dt with E = int |u'|^2 r dr dz, compared with 1/(T - t).
Mapped-z snapshots are transferred to the uniform full-period grid by exact sine-series evaluation.
   python axi3dlin.py <snapshot> <T_est> <m> [steps] [nz_full]"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.fft import fft, ifft, dst
from scipy.linalg import solve_banded

snap = sys.argv[1]; T_est = float(sys.argv[2]); m = int(sys.argv[3]); nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 600
Nreq = int(sys.argv[5]) if len(sys.argv) > 5 else 0
d = np.load(snap); U1h, Psh, r, zh, t0 = d["U"], d["Ps"], d["r"], d["z"], float(d["t"])
n_r = len(r); n_zh = len(zh) + 1
zm = tuple(float(v) for v in d["zmap"]) if "zmap" in d.files else (0.0, 0.0)
if os.environ.get("AXL_ZMAP"): zm = tuple(float(v) for v in os.environ["AXL_ZMAP"].split(","))   # override for mapped snapshots saved without the key (z513a5 series: 0.9,1)
uniform = np.abs(np.diff(zh) - zh[0]).max() < 1e-9
N = Nreq if Nreq else 2 * n_zh
z = np.arange(N) / N
def full(Fh, parity):
    """Half-period data (n_zh-1 interior values on the stored z grid) -> full period on the uniform grid z_j = j/N.
    Uniform stored grid with N = 2 n_zh: direct mirror.  Otherwise: exact sine-series evaluation at the new nodes."""
    if uniform and N == 2 * n_zh:
        F = np.zeros((n_r, N)); F[:, 1:n_zh] = Fh
        F[:, n_zh+1:] = (-Fh[:, ::-1] if parity == "odd" else Fh[:, ::-1])
        if parity == "even":
            F[:, 0] = 2 * F[:, 1] - F[:, 2]; F[:, n_zh] = 2 * F[:, n_zh-1] - F[:, n_zh-2]
        return F
    if zm[0] > 0:                                              # (beta, m) cos-family map: z(zeta) with density (1 - beta cos 4 pi zeta)^m
        beta, mm = zm; fine = np.linspace(0.0, 0.5, 200001); g = (1.0 - beta * np.cos(4 * np.pi * fine)) ** mm
        Zf = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(fine))]); Zf *= 0.5 / Zf[-1]
        zeta_of = lambda zz: np.interp(zz, Zf, fine)
    else:
        zeta_of = lambda zz: zz
    coef = dst(Fh, type=1, axis=1) / n_zh                      # F(zeta) = sum_k coef_k sin(2 pi k zeta), zeta in [0, 1/2]
    zhalf = np.where(z <= 0.5, z, 1.0 - z); sgn = np.where(z <= 0.5, 1.0, -1.0 if parity == "odd" else 1.0)
    S = np.sin(2.0 * np.pi * np.arange(1, n_zh)[None, :] * zeta_of(zhalf)[:, None])
    return (coef @ S.T) * sgn[None, :]
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
par_rt = "even" if m % 2 == 1 else "odd"                       # parity of u_r', u_th'  (~ r^{m-1})
pe = m - 1 if m >= 1 else 1; pz = m if m >= 1 else 0          # seed axis exponents: u_r, u_th ~ r^{m-1} (m >= 1) or r (m = 0); u_z ~ r^m
par_zp = "odd" if m % 2 == 1 else "even"                       # parity of u_z', p'     (~ r^m)
u1par = os.environ.get("AXL_U1PAR", "odd")                   # "even" for columnar test bases (u1 independent of z)
u1 = full(U1h, u1par); Ps = full(Psh, "odd")
Psz = dz(Ps).real; Psr = dr(Ps, "even")
r_ = r[:, None]
Uth = r_ * u1; Ur = -r_ * Psz; Uz = 2.0 * Ps + r_ * Psr
dUr_r = dr(Ur, "odd"); dUr_z = dz(Ur).real; dUth_r = dr(Uth, "odd"); dUth_z = dz(Uth).real; dUz_r = dr(Uz, "even"); dUz_z = dz(Uz).real   # U_r, U_th odd in r; U_z even
with np.errstate(divide="ignore", invalid="ignore"):
    inv_r_vec = np.where(r > 0, 1.0 / np.maximum(r, 1e-300), 0.0)
inv_r = inv_r_vec[:, None]
def over_r(F, parity_of_F):
    """F/r; axis value by L'Hopital when F is odd in r, 0 when F is even (regularity then requires F(0) = 0)."""
    G = F * inv_r
    G[0] = dr(F, parity_of_F)[0] if parity_of_F == "odd" else 0.0
    return G
# pressure operator = exact discrete div_m(grad_m), banded, one factorization per Fourier mode
I_n = np.eye(n_r)
Dzp = dr(I_n, par_zp); Drt = dr(I_n, par_rt)
Rinv = np.diag(inv_r_vec)
OVR = Rinv.copy(); OVR[0, :] = (Drt[0, :] if par_rt == "odd" else 0.0)      # matrix form of over_r for a field of parity par_rt
Lm = Drt @ Dzp + OVR @ Dzp - (m * m) * (OVR @ Rinv)                          # over_r(p) has axis value 0 for both parities of p that occur
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
    if par_zp == "odd": A[0, :] = 0.0; A[0, 0] = 1.0          # m odd: the exact axis row is k^2 p0 = 0 -> impose p'(0) = 0; m even: keep the exact discrete row
    A[-1, :] = Dzp[-1, :]                                      # wall: Neumann row, d_r p' = (given) so that the projected u_r' vanishes at r = 1
    if m == 0 and abs(k) < 1e-12: A[0, :] = 0.0; A[0, 0] = 1.0   # m = 0, k_z = 0: singular Neumann problem; pin p(0) = 0
    off = A.copy()
    for kk in range(-BW, BW + 1): off -= np.diag(np.diagonal(A, kk), kk)
    assert np.abs(off).max() < 1e-8 * np.abs(A).max(), "pressure operator not banded"
    lus.append(to_banded(A))
def poisson(rhs, wall):
    """solve -Lap_m p = rhs on the interior rows with d_r p = wall at r = 1 (wall: array over z) and the axis row per parity"""
    rk = fft(rhs, axis=1); wk = fft(wall); out = np.empty_like(rk)
    for j in range(N):
        b = rk[:, j].copy(); b[-1] = wk[j]
        if par_zp == "odd" or (m == 0 and j == 0): b[0] = 0.0
        out[:, j] = solve_banded((BW, BW), lus[j], b, check_finite=False)
    return ifft(out, axis=1)
def grad(p):
    return dr(p, par_zp), 1j * m * over_r(p, par_zp), dz(p)
def div(vr, vt, vz):
    return dr(vr, par_rt) + over_r(vr + 1j * m * vt, par_rt) + dz(vz)
def D(F, parity):
    return Ur * dr(F, parity) + Uz * dz(F) + 1j * m * u1 * F
nu = float(d["nu"]) if ("nu" in d.files and os.environ.get("AXL_VISC", "1") == "1") else 0.0
def lap_k(F, k, parity):
    Fr = dr(F, parity); Frr = dr(Fr, "odd" if parity == "even" else "even")
    G = Frr + Fr * inv_r - (k * k) * F * inv_r * inv_r + dz(dz(F))
    G[0] = (2.0 * Frr[0] + dz(dz(F))[0]) if k == 0 else 0.0
    return G
def rhs(ur, ut, uz):
    Nr = D(ur, par_rt) + ur * dUr_r + uz * dUr_z - 2.0 * u1 * ut
    Nt = D(ut, par_rt) + ur * dUth_r + uz * dUth_z + u1 * ur - Psz * ut
    Nz = D(uz, par_zp) + ur * dUz_r + uz * dUz_z
    if nu > 0:
        up = ur + 1j * ut; um = ur - 1j * ut
        Lp = lap_k(up, m + 1, par_rt); Lmn = lap_k(um, m - 1, par_rt); Lz = lap_k(uz, m, par_zp)
        Nr = Nr - nu * 0.5 * (Lp + Lmn); Nt = Nt - nu * 0.5 * (Lp - Lmn) / 1j; Nz = Nz - nu * Lz
    p = poisson(div(Nr, Nt, Nz), -Nr[-1])                     # d_r p = -N_r at the wall -> d_t u_r' = 0 there
    gr, gt, gz = grad(p)
    return -Nr - gr, -Nt - gt, -Nz - gz
def energy(ur, ut, uz):
    return float(np.sum((np.abs(ur) ** 2 + np.abs(ut) ** 2 + np.abs(uz) ** 2) * r_ * drde) * h_eta / N)
i0, j0 = np.unravel_index(int(np.argmax(np.abs(u1))), u1.shape)
R0, Z0 = r[i0], z[j0]
if os.environ.get("AXL_SEED_R0"): R0 = float(os.environ["AXL_SEED_R0"])      # fix the localized seed's placement across resolutions (certificate runs)
if os.environ.get("AXL_SEED_Z0"): Z0 = float(os.environ["AXL_SEED_Z0"])
kz_seed = int(os.environ.get("AXL_KZ", "0"))                  # > 0: seed a single Fourier mode e^{2 pi i kz z} (columnar gates)
if kz_seed > 0: R0 = float(os.environ.get("AXL_R0", "0.5")); env = np.exp(-((r_ - R0) / 0.15) ** 2) * np.exp(2j * np.pi * kz_seed * z[None, :])
else: env = np.exp(-((r_ - R0) / (0.5 * R0)) ** 2) * (np.exp(-((z[None, :] - Z0) / (2 * Z0)) ** 2) + np.exp(-((z[None, :] - 1 + Z0) / (2 * Z0)) ** 2))
ur = env * (r_ / R0) ** pe * (1 + 0.1j); ut = 1j * ur * (1.0 + 0.5 * (r_ / R0) ** 2); uz = (env * (r_ / R0) ** pz * 0.3).astype(complex)   # u_r' + i u_th' = O(r^{m+1}) (axis regularity)
if os.environ.get("AXL_SEED", "") == "random":            # seed independence (lit 144): smooth random field under the same envelope
    rng = np.random.default_rng(int(os.environ.get("AXL_RNG", "1")))
    def smooth_noise():
        Fk = rng.standard_normal((n_r, N)) + 1j * rng.standard_normal((n_r, N))
        Fk *= np.exp(-(np.abs(kz) / (2 * np.pi * 40)) ** 2)[None, :]; F = ifft(Fk, axis=1)
        for _ in range(6): F[1:-1] = 0.25 * F[:-2] + 0.5 * F[1:-1] + 0.25 * F[2:]
        return F
    ur = env * (r_ / R0) ** pe * smooth_noise(); ut = env * (r_ / R0) ** pe * smooth_noise(); uz = env * (r_ / R0) ** pz * smooth_noise()
    ur[0] = 0.5 * (ur[0] - 1j * ut[0]); ut[0] = 1j * ur[0]        # axis regularity of the m = 1 pair at the axis row
p = poisson(-div(ur, ut, uz), ur[-1]); gr, gt, gz = grad(p); ur -= gr; ut -= gt; uz -= gz   # u_r' - d_r p = 0 at the wall
E0 = energy(ur, ut, uz); vmax = max(np.abs(Ur).max(), np.abs(Uz).max(), np.abs(Uth).max())
dt = 0.3 * min(np.diff(r).min(), 1.0 / N) / vmax
print(f"AXI3DLIN m={m} frozen base {os.path.basename(snap)} t0 {t0:.6f} (T-t ~ {T_est - t0:.2e}, 1/(T-t) = {1/(T_est-t0):.3e}); grid {n_r} x {N} (dz {1/N:.2e}); zmap {zm}; nu {nu}; dt {dt:.2e}; div0 {np.abs(div(ur, ut, uz)[:-1]).max():.2e} E0 {E0:.3e}; base max u1 {np.abs(u1).max():.3e} at ({R0:.4f}, {Z0:.4f})", flush=True)
tt, tlast, Elast = 0.0, 0.0, E0; t_start = time.time()
for it in range(1, nsteps + 1):
    k1 = rhs(ur, ut, uz)
    k2 = rhs(ur + 0.5 * dt * k1[0], ut + 0.5 * dt * k1[1], uz + 0.5 * dt * k1[2])
    k3 = rhs(ur + 0.5 * dt * k2[0], ut + 0.5 * dt * k2[1], uz + 0.5 * dt * k2[2])
    k4 = rhs(ur + dt * k3[0], ut + dt * k3[1], uz + dt * k3[2])
    ur = ur + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); ut = ut + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); uz = uz + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    tt += dt
    if it % max(1, nsteps // 20) == 0 or it == nsteps:
        E = energy(ur, ut, uz); sig = np.log(E / Elast) / (2 * (tt - tlast)) if tt > tlast else 0.0; Elast, tlast = E, tt
        ia, ja = np.unravel_index(int(np.argmax(np.abs(ur) ** 2 + np.abs(ut) ** 2 + np.abs(uz) ** 2)), ur.shape)
        # diagnostics (lit 142 protocol): z-wavenumber centroid, radial extent, Reynolds-Orr production split (swirl-shear vs meridional)
        Ek = (np.abs(fft(ur, axis=1)) ** 2 + np.abs(fft(ut, axis=1)) ** 2 + np.abs(fft(uz, axis=1)) ** 2) * r_ * drde
        kcen = float(np.sum(np.abs(kz)[None, :] * Ek) / np.sum(Ek)) / (2 * np.pi)
        e_r = np.sum(np.abs(ur) ** 2 + np.abs(ut) ** 2 + np.abs(uz) ** 2, axis=1) * r * drde[:, 0]; rcen = float(np.sum(r * e_r) / np.sum(e_r)); rspr = float(np.sqrt(np.sum((r - rcen) ** 2 * e_r) / np.sum(e_r)))
        wgt = r_ * drde * h_eta / N
        P_sw = float(np.sum(np.real(np.conj(ur) * ut) * (u1 - dUth_r) * wgt))                                   # -Re(ur* ut) r d(u1)/dr
        P_tot = P_sw + float(np.sum(np.real(-np.conj(ur) * (ur * dUr_r + uz * dUr_z) - np.conj(uz) * (ur * dUz_r + uz * dUz_z) - np.conj(ut) * (uz * dUth_z - Psz * ut)) * wgt))
        if m == 1:   # projection on the axis-translation field (-d_x of the base, m = 1): ur ~ d_r U_r - i U_th/r, ut ~ d_r U_th + i U_r/r, uz ~ d_r U_z
            Tr = dUr_r - 1j * over_r(Uth, "odd"); Tt = dUth_r + 1j * over_r(Ur, "odd"); Tz = dUz_r
            num = np.sum((np.conj(Tr) * ur + np.conj(Tt) * ut + np.conj(Tz) * uz) * wgt); den = np.sqrt(np.sum((np.abs(Tr) ** 2 + np.abs(Tt) ** 2 + np.abs(Tz) ** 2) * wgt) * E)
            trans = abs(num) / den
        else: trans = 0.0
        print(f"    diag: kz centroid {kcen:.1f} (of {N//2})  r centroid {rcen:.4f} +- {rspr:.4f}  production swirl-shear {P_sw/E:.3e} total {P_tot/E:.3e} (sigma {sig:.3e})  translation-mode overlap {trans:.3f}", flush=True)
        print(f"  t' {tt:.3e} ({tt/(T_est-t0):.3f} of T-t)  E/E0 {E/E0:.4e}  sigma {sig:.4e}  sigma (T-t) {sig*(T_est-t0):.3f}  |div| {np.abs(div(ur, ut, uz)[:-1]).max():.2e}  peak at (r, z) = ({r[ia]:.4f}, {z[ja]:.4f})  ({time.time()-t_start:.0f}s)", flush=True)
        if not np.isfinite(E): break
