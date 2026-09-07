"""Frozen-base m-mode EIGENPROBLEM (stage 1 of the eigenvalue enclosure): reuse axi3dlin's discrete linearized operator
(projection included) as a matrix-free linear map on the divergence-free perturbations and compute its leading eigenvalues
with ARPACK (largest real part), instead of reading the growth rate off a time march.  Reports sigma (T-t) for the leading
eigenvalues, the a posteriori residual ||L v - lambda v|| / ||v|| of each eigenpair (exact for the discrete operator), the
divergence of the eigenvector, its energy centroid and translation-mode overlap (m = 1).
   python axi3deig.py <snapshot> <T_est> <m> [nz_full] [k]"""
import os, sys, time, re
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
snap = sys.argv[1]; T_est = float(sys.argv[2]); m = int(sys.argv[3]); Nreq = sys.argv[4] if len(sys.argv) > 4 else "0"; k = int(sys.argv[5]) if len(sys.argv) > 5 else 4
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "axi3dlin.py"), encoding="utf-8").read()
setup = src.split("for it in range(1, nsteps + 1):")[0]                       # everything before the time loop (operator + seed + projection)
sys.argv = ["axi3dlin.py", snap, str(T_est), str(m), "1", Nreq]
g = {"__name__": "axi3deig_setup", "__file__": os.path.abspath("axi3dlin.py")}
exec(compile(setup, "axi3dlin_setup", "exec"), g)
rhs_pert, r, N, n_r = g["rhs"], g["r"], g["N"], g["n_r"]; wgt = g["r_"] * g["drde"] * g["h_eta"] / N; div = g["div"]; t0 = g["t0"]
ur0, ut0, uz0 = g["ur"], g["ut"], g["uz"]
shape = (n_r, N); nn = n_r * N
def unpack(x): return x[:nn].reshape(shape), x[nn:2*nn].reshape(shape), x[2*nn:].reshape(shape)
def pack(a, b, c): return np.concatenate([a.ravel(), b.ravel(), c.ravel()])
calls = [0]
def matvec(x):
    calls[0] += 1
    a, b, c = unpack(np.asarray(x, dtype=complex)); fa, fb, fc = rhs_pert(a, b, c)
    return pack(fa, fb, fc)
# Time-stepper (exponential) transform, lit 194: Arnoldi on P = exp(L Delta) instead of L.  P's eigenvalues of largest
# modulus are L's of largest real part, and the damped continuum that stalls unshifted Arnoldi on L is contracted to
# the origin.  No inner solve; one RK4 tangent-linear integration over Delta per Arnoldi vector (axi3dlin's stepping).
PROP = float(os.environ.get("EIG_PROP", "0"))                       # Delta as a fraction of T - t0 (0 = plain operator)
dt_lin = g["dt"]; Delta = PROP * (T_est - t0); nprop = max(1, int(np.ceil(Delta / dt_lin))) if PROP > 0 else 0
dtp = Delta / nprop if nprop else 0.0
def propagate(x):
    calls[0] += 1
    ur, ut, uz = unpack(np.asarray(x, dtype=complex))
    for _ in range(nprop):
        k1 = rhs_pert(ur, ut, uz)
        k2 = rhs_pert(ur + 0.5 * dtp * k1[0], ut + 0.5 * dtp * k1[1], uz + 0.5 * dtp * k1[2])
        k3 = rhs_pert(ur + 0.5 * dtp * k2[0], ut + 0.5 * dtp * k2[1], uz + 0.5 * dtp * k2[2])
        k4 = rhs_pert(ur + dtp * k3[0], ut + dtp * k3[1], uz + dtp * k3[2])
        ur = ur + dtp / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); ut = ut + dtp / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); uz = uz + dtp / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return pack(ur, ut, uz)
# Shift-invert (lit 194, Meerbergen-Roose; Lehoucq-Salinger): Arnoldi on (L - s)^{-1} with s near the wanted eigenvalue,
# each application an inner GMRES solve on the matrix-free operator.  EIG_SHIFT gives s as sigma (T-t) (real part) and
# EIG_SHIFT_IM its imaginary part in the same units; EIG_GMRES_TOL / EIG_GMRES_RESTART / EIG_GMRES_MAXIT control the inner solve.
SHIFT = os.environ.get("EIG_SHIFT")
inner = [0, 0]                                                       # inner matvecs, inner solves
if SHIFT is not None:
    from scipy.sparse.linalg import gmres
    Tt0 = T_est - t0
    s_shift = (float(SHIFT) + 1j * float(os.environ.get("EIG_SHIFT_IM", "0"))) / Tt0
    gtol = float(os.environ.get("EIG_GMRES_TOL", "1e-6")); grest = int(os.environ.get("EIG_GMRES_RESTART", "60")); gmax = int(os.environ.get("EIG_GMRES_MAXIT", "20"))
    def shifted(x):
        inner[0] += 1
        return matvec(x) - s_shift * np.asarray(x, dtype=complex)
    A_s = LinearOperator((3 * nn, 3 * nn), matvec=shifted, dtype=complex)
    # optional preconditioner (EIG_PREC=diag): the shifted axial diffusion, diagonal in the z-Fourier index:
    # M^{-1} x = ifft_z( fft_z(x) / (nu (kz^2 + m^2 / rc^2) - s) ), rc the swirl-maximum radius; exact for the damped
    # short waves that clutter the spectrum, identity-like (1/(-s)) for the long waves that carry the wanted eigenvalue.
    PREC = os.environ.get("EIG_PREC", "")
    if PREC == "diag":
        nu_lin = g["nu"]; kz_lin = g["kz"]; rc = float(os.environ.get("EIG_PREC_RC", "0.0137"))
        dcoef = nu_lin * (kz_lin[None, :] ** 2 + (m / rc) ** 2) - s_shift
        def precond(x):
            a, b_, c = unpack(np.asarray(x, dtype=complex))
            out = [np.fft.ifft(np.fft.fft(f, axis=1) / dcoef, axis=1) for f in (a, b_, c)]
            return pack(*out)
        M_op = LinearOperator((3 * nn, 3 * nn), matvec=precond, dtype=complex)
        print(f"  preconditioner: diag shifted axial diffusion, nu {nu_lin:.1e}, rc {rc}", flush=True)
    else:
        M_op = None
    def solve(b):
        calls[0] += 1; inner[1] += 1; t1 = time.time()
        x, info = gmres(A_s, np.asarray(b, dtype=complex), rtol=gtol, restart=grest, maxiter=gmax, M=M_op)
        rres = np.linalg.norm(shifted(x) - b) / np.linalg.norm(b)
        print(f"    inner solve {inner[1]}: info {info}, relative residual {rres:.2e}, {time.time()-t1:.0f}s (cumulative inner matvecs {inner[0]})", flush=True)
        return x
    L = LinearOperator((3 * nn, 3 * nn), matvec=solve, dtype=complex)
    print(f"  shift-invert: s = {s_shift.real:.4e} {s_shift.imag:+.4e}i (sigma (T-t) {float(SHIFT)}), GMRES tol {gtol} restart {grest} maxiter {gmax}", flush=True)
elif PROP > 0:
    L = LinearOperator((3 * nn, 3 * nn), matvec=propagate, dtype=complex)
    print(f"  time-stepper transform: Delta = {Delta:.3e} ({PROP} of T-t), {nprop} RK4 steps of {dtp:.3e} per application", flush=True)
else:
    L = LinearOperator((3 * nn, 3 * nn), matvec=matvec, dtype=complex)
v0 = pack(ur0, ut0, uz0)
t_start = time.time()
from scipy.sparse.linalg import ArpackNoConvergence
ncv = int(os.environ.get("EIG_NCV", str(max(3 * k + 2, 24)))); maxit = int(os.environ.get("EIG_MAXITER", "400")); tol = float(os.environ.get("EIG_TOL", "1e-7"))
try:
    vals, vecs = eigs(L, k=k, which=("LM" if (PROP > 0 or SHIFT is not None) else "LR"), v0=v0, ncv=ncv, tol=tol, maxiter=maxit)
except ArpackNoConvergence as e:
    print(f"  ARPACK did not converge ({calls[0]} applications); {len(e.eigenvalues)} eigenpairs converged -- reporting those", flush=True)
    vals, vecs = e.eigenvalues, e.eigenvectors; k = len(vals)
    if k == 0: sys.exit(1)
if SHIFT is not None:
    nus = vals.copy(); vals = s_shift + 1.0 / nus          # lambda = s + 1/nu
elif PROP > 0:
    mus = vals.copy(); vals = np.log(mus) / Delta        # lambda = log(mu)/Delta; imaginary part known modulo 2 pi / Delta
order = np.argsort(-vals.real); vals, vecs = vals[order], vecs[:, order]
Tt = T_est - t0
print(f"AXI3DEIG m={m} {os.path.basename(snap)} t0 {t0:.6f} T-t {Tt:.2e} grid {n_r} x {N}: {calls[0]} operator applications in {time.time()-t_start:.0f}s")
for j in range(k):
    lam = vals[j]; v = vecs[:, j]; Lv = matvec(v)
    res = np.linalg.norm(Lv - lam * v) / np.linalg.norm(v)
    if PROP > 0:
        Pv = propagate(v); mu = np.exp(lam * Delta); resP = np.linalg.norm(Pv - mu * v) / np.linalg.norm(v)
        print(f"  eigenvalue {j+1}: propagator residual ||Pv - mu v||/||v|| = {resP:.2e} (mu = {mu.real:+.5f} {mu.imag:+.5f}i; omega ambiguous mod 2pi/Delta = {2*np.pi/Delta*Tt:.1f} in collapse units)", flush=True)
    a, b, c = unpack(v); E = float(np.sum((np.abs(a)**2 + np.abs(b)**2 + np.abs(c)**2) * wgt))
    dv = np.abs(div(a, b, c)[:-1]).max() / (np.abs(v).max() + 1e-300)
    Ek = (np.abs(np.fft.fft(a, axis=1))**2 + np.abs(np.fft.fft(b, axis=1))**2 + np.abs(np.fft.fft(c, axis=1))**2)
    kz = np.abs(np.fft.fftfreq(N, d=1.0 / N)); kcen = float(np.sum(kz[None, :] * Ek) / np.sum(Ek))
    rc = float(np.sum(r[:, None] * (np.abs(a)**2 + np.abs(b)**2 + np.abs(c)**2) * wgt) / E)
    print(f"  eigenvalue {j+1}: lambda = {lam.real:+.5e} {lam.imag:+.5e}i  ->  sigma (T-t) = {lam.real * Tt:+.4f}, omega (T-t) = {lam.imag * Tt:+.4f};  residual ||Lv - lambda v||/||v|| = {res:.2e};  |div|/|v| {dv:.1e};  kz centroid {kcen:.1f};  r centroid {rc:.4f}", flush=True)
np.savez(f"axi3deig_m{m}_{os.path.basename(snap).replace('.npz','')}_N{N}{'_prop' if PROP > 0 else ''}{'_si' if SHIFT is not None else ''}.npz", vals=vals, t0=t0, T_est=T_est, N=N, n_r=n_r)
