"""AXIPHYS: physical-space axisymmetric Euler/Navier-Stokes with swirl in Hou-Luo
variables, from Hou's interior initial data (arXiv:2107.05870 / 2107.06509),
to generate a developed seed for the rescaled profile solver.

  u1 = u^theta/r, omega1 = omega^theta/r, psi1 = psi/r on (r, z) in [0,1] x [0, 1/2]
  u1_t + u^r u1_r + u^z u1_z = 2 u1 psi1_z + nu L u1
  omega1_t + u^r omega1_r + u^z omega1_z = (u1^2)_z + nu L omega1
  -L psi1 = omega1,  L = d_rr + (3/r) d_r + d_zz,  u^r = -r psi1_z,  u^z = 2 psi1 + r psi1_r
Symmetry: u1, omega1, psi1 odd in z about z = 0 and z = 1/2 (period 1) -> sine
modes sin(2 pi k z); even in r at the axis; psi1 = 0 at the wall r = 1.
Data: u1(0) = 12000 (1 - r^2)^18 sin(2 pi z)/(1 + 12.5 sin^2(pi z)), omega1(0) = 0.

Discretization: sine-spectral in z (DST-I on the interior nodes), 4th-order
finite differences in a mapped radial coordinate r = sinh(a eta)/sinh(a)
(dense near the axis), Poisson per sine mode by a banded solve, RK4 with CFL
control, optional small hyperviscosity in z for the inviscid run.
Hou's checkpoints: Euler ||omega||_inf x ~300 by t = 0.002264 and ~6000 by
t = 0.002277; NS (nu = 5e-4) x 1e7 by t = 0.0022868.

  python axiphys.py <n_r> <n_z> <t_end> [nu]     (AXP_A map strength, AXP_TAG)
"""
import os, sys, time
import numpy as np
from scipy.fft import dst, idst
from scipy.linalg import solve_banded


def _unband(ab):
    n = ab.shape[1]; A = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - 2), min(n, i + 3)):
            A[i, j] = ab[2 + i - j, j]
    return A


class AxiPhys:
    def __init__(self, n_r=256, n_z=256, nu=0.0, a=3.0, dim=3.0, Lr=1.0, Lz=0.5, q2=1.0):
        self.n_r, self.n_z, self.nu, self.dim = n_r, n_z, float(nu), float(dim)
        self.Lr, self.q2 = float(Lr), float(q2)          # box [0, Lr] x [0, Lz]; q2 multiplies the radial part of the Laplacian (rescaled-frame anisotropy)
        self._Lz = float(Lz)
        # radial map r = sinh(a eta)/sinh(a), eta in [0, 1], uniform in eta
        eta = np.linspace(0.0, 1.0, n_r)
        self.eta, self.h = eta, eta[1] - eta[0]
        self.a = a
        self.moving = os.environ.get("AXP_MOVING", "0") == "1"
        self.params = None                        # (R_c, w) of the Gaussian-density map when moving
        self.set_map(None)
        # z: interior nodes of [0, 1/2], z_j = j/(2 n_z), j = 1..n_z-1 (DST-I)
        self.Lz = self._Lz
        self.z = np.arange(1, n_z) * self.Lz / n_z
        self.kz = np.pi * np.arange(1, n_z) / self.Lz      # sin(pi k z / Lz): odd about z = 0 and z = Lz
        self.zeta = self.z.copy(); self.zmap = None          # z = z(zeta): parity-preserving map (set_zmap)
        # radial FD operators in eta (4th order interior, 2nd order near ends), then chain rule
        self.D1 = self._fd(n_r, self.h, 1); self.D2 = self._fd(n_r, self.h, 2)
        # Poisson per mode: -(psi_rr + 3/r psi_r) + k^2 psi = om; in eta: psi_r = psi_eta/r', psi_rr = psi_ee/r'^2 - psi_e r''/r'^3
        zb = float(os.environ.get("AXP_ZBETA", "0")); self.set_zmap((zb, float(os.environ.get("AXP_ZM", "1"))) if zb > 0 else None)

    def set_zmap(self, zp):
        """z = z(zeta) with z_zeta = (1 - beta cos 4 pi zeta)^m / norm (even about zeta = 0 and 1/2, so the
        sine parity of every field survives the map; lit checks 71/77).  zp = None restores the uniform grid.
        The mapped z-Laplacian L_z = z_zeta^{-2} D2 - z_zetazeta z_zeta^{-3} D1 (spectral D1, D2) is
        diagonalized once in the z_zeta-weighted symmetric form (lit check 80, Lynch-Rice-Thomas), so the
        Poisson solve stays per-mode: Om_hat = Om W^{1/2} Q, radial solves with k^2 -> lambda, Psi = Psi_hat Q^T W^{-1/2}."""
        from scipy.linalg import eigh
        zeta = self.zeta
        if zp is None:
            self.z = zeta.copy(); self.zmap = None
            self.zd = np.ones_like(zeta); self.zdd = np.zeros_like(zeta)
            self.lam_z = self.kz ** 2; self.Vz = None
        elif len(zp) == 2:
            beta, m = zp
            Lz = self.Lz; wz = 2.0 * np.pi / Lz                # cos(2 pi zeta / Lz): even about 0 and Lz
            fine = np.linspace(0.0, Lz, 200001)
            g = (1.0 - beta * np.cos(wz * fine)) ** m
            Z = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(fine))]); norm = Z[-1] / Lz
            self.z = np.interp(zeta, fine, Z / norm); self.zmap = zp
            self._zfine_zeta, self._zfine_z = fine, Z / norm          # inverse-map table zeta(z)
            self.zd = (1.0 - beta * np.cos(wz * zeta)) ** m / norm
            self.zdd = m * (1.0 - beta * np.cos(wz * zeta)) ** (m - 1) * wz * beta * np.sin(wz * zeta) / norm
        else:
            # front-following map (lit 101): mesh-point density in z, rho(z) = 1 + alpha sum_images exp(-((z - z_i)/2w)^2),
            # images z_c, -z_c, 2Lz - z_c, 2Lz + z_c (even about 0 and Lz), with a fraction frac of the points in |z - z_c| < 3w;
            # zeta(z) = Lz C(z)/C(Lz), z_zeta = C(Lz)/(Lz rho), z_zetazeta = -(C(Lz)/Lz)^2 rho'/rho^3 (analytic)
            z_c, w, frac = zp; Lz = self.Lz
            fine = np.linspace(0.0, Lz, 200001)
            imgs = [z_c, -z_c, 2 * Lz - z_c, 2 * Lz + z_c]
            sw = float(os.environ.get("AXP_ZSHOULDER", "4.0")) * w
            g = sum(np.exp(-((fine - zi) / sw) ** 2) for zi in imgs)
            dg = sum(-2.0 * (fine - zi) / sw ** 2 * np.exp(-((fine - zi) / sw) ** 2) for zi in imgs)
            win = np.abs(fine - z_c) <= 3 * w
            Ig, Itot = np.trapezoid(g[win], fine[win]), np.trapezoid(g, fine); Lw = float(fine[win][-1] - fine[win][0])
            alpha = max((frac * Lz - Lw) / (Ig - frac * Itot), 0.0)
            rho = 1.0 + alpha * g; drho = alpha * dg
            C = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(fine))]); C1 = C[-1]
            zeta_of_z = Lz * C / C1
            self.z = np.interp(zeta, zeta_of_z, fine); self.zmap = zp
            self._zfine_zeta, self._zfine_z = zeta_of_z, fine
            rho_z = np.interp(self.z, fine, rho); drho_z = np.interp(self.z, fine, drho)
            self.zd = C1 / (Lz * rho_z); self.zdd = -(C1 / Lz) ** 2 * drho_z / rho_z ** 3
        if zp is not None:
            I = np.eye(self.n_z - 1)
            D1 = self._dzeta(I.T).T; D2 = self._dzzeta(I.T).T          # columns: derivatives of unit vectors -> operator matrices
            Lz = (D2 / self.zd[:, None] ** 2) - D1 * (self.zdd / self.zd ** 3)[:, None]
            W = self.zd
            S = -(np.sqrt(W)[:, None] * Lz / np.sqrt(W)[None, :])       # W^{1/2} (-Lz) W^{-1/2}: symmetric up to truncation
            self._zsym = np.abs(S - S.T).max() / np.abs(S).max()
            S = 0.5 * (S + S.T)
            lam, Q = eigh(S)
            self.lam_z = lam                                          # -Lz eigenvalues (>= 0), replace k^2
            self.Vz = Q; self._Wh = np.sqrt(W)
        self._lu = [self._poisson_matrix(k) for k in np.sqrt(np.maximum(self.lam_z, 0.0))]

    def set_map(self, params):
        """Radial map r(eta).  params None: fixed sinh map.  params (R_c, w, frac): Luo-Hou-type
        mesh-point density in r, dens(r) = alpha0 + alpha1 exp(-((r-R_c)/w)^2), with a fraction
        `frac` of the points inside |r - R_c| < 3w (lit check 71); eta(r) = C(r)/C(1) inverted
        on a fine grid; dr/deta = C(1)/dens and d2r/deta2 = -C(1)^2 dens'/dens^3 analytic."""
        eta, a = self.eta, self.a
        if params is None:
            self.r = self.Lr * np.sinh(a * eta) / np.sinh(a)
            self.drde = self.Lr * a * np.cosh(a * eta) / np.sinh(a)
            self.d2rde = self.Lr * a * a * np.sinh(a * eta) / np.sinh(a)
        else:
            R_c, w, frac = params
            fine = np.linspace(0.0, self.Lr, 40001)
            g = np.exp(-((fine - R_c) / (2.0 * w)) ** 2)      # density shoulder 2w wide (4th-order stencils need >= ~8 cells per transition)
            win = np.abs(fine - R_c) <= 3 * w
            Ig, Itot = np.trapezoid(g[win], fine[win]), np.trapezoid(g, fine)
            Lw = float(fine[win][-1] - fine[win][0])
            # alpha0 Lw + alpha1 Ig = frac (alpha0 + alpha1 Itot)  with alpha0 = 1
            alpha1 = (frac - Lw) / (Ig - frac * Itot)
            dens = 1.0 + max(alpha1, 0.0) * g
            C = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(fine))])
            C1 = C[-1]
            self.r = np.interp(eta, C / C1, fine)
            dens_r = np.interp(self.r, fine, dens)
            ddens_r = np.interp(self.r, fine, -2.0 * (fine - R_c) / (2.0 * w) ** 2 * max(alpha1, 0.0) * g)
            self.drde = C1 / dens_r
            self.d2rde = -C1 ** 2 * ddens_r / dens_r ** 3
        self.params = params
        if hasattr(self, "lam_z"):
            self._lu = [self._poisson_matrix(k) for k in np.sqrt(np.maximum(self.lam_z, 0.0))]

    def rezone(self, fields, params):
        """Move the radial grid to `params` and cubic-interpolate the (n_r, n_z-1) fields onto it
        (static rezoning; the fields are even in r, so the spline is built on the reflected data)."""
        from scipy.interpolate import CubicSpline
        r_old = self.r.copy()
        out = []
        for F in fields:
            rr = np.concatenate([-r_old[:0:-1], r_old]); FF = np.concatenate([F[:0:-1], F], axis=0)
            out.append(F if False else None)
            out[-1] = CubicSpline(rr, FF, axis=0)
        self.set_map(params)
        return [cs(self.r) for cs in out]

    def target_params(self, U, Om, frac=0.4):
        """Monitor |omega^theta| = r|omega1| (lit check 71): window centre at its radial maximum,
        half-width from the half-maximum radial extent along that z row (floored at 4 sinh-map cells)."""
        W = np.abs(self.r[:, None] * Om)
        i, j = np.unravel_index(int(np.argmax(W)), W.shape)
        row = W[:, j]; half = row >= 0.5 * row[i]
        idx = np.where(half)[0]
        hi = self.r[idx.max()]                            # axis-directed scenario: window [0, winf hi] including the axis
        winf = float(os.environ.get("AXP_WINF", "1.5"))
        R_c = 0.5 * winf * hi; w = max(winf * hi / 6.0, 2e-3)
        return (float(R_c), float(w), frac)

    @staticmethod
    def _fd(n, h, order):
        """4th-order centered stencils with EVEN reflection across the axis (parity closure,
        lit check 53): rows 0..1 fold the negative-index columns onto positive ones; the wall
        end keeps one-sided stencils (fields vanish there)."""
        D = np.zeros((n, n))
        if order == 1:
            c = np.array([1, -8, 0, 8, -1]) / (12 * h)
        else:
            c = np.array([-1, 16, -30, 16, -1]) / (12 * h * h)
        for i in range(0, n - 2):
            for k, off in enumerate((-2, -1, 0, 1, 2)):
                j = i + off
                if j < 0:
                    j = -j                       # even extension (sinh map is odd in eta -> even in r): same value, no sign
                    D[i, j] += c[k]
                else:
                    D[i, j] += c[k]
        if order == 1:
            D[n - 2, :] = 0.0; D[n - 2, n - 3:n] = np.array([-1, 0, 1]) / (2 * h)
            D[n - 1, :] = 0.0; D[n - 1, n - 3:n] = np.array([1, -4, 3]) / (2 * h)
        else:
            D[n - 2, :] = 0.0; D[n - 2, n - 3:n] = np.array([1, -2, 1]) / (h * h)
            D[n - 1, :] = 0.0; D[n - 1, n - 4:n] = np.array([-1, 4, -5, 2]) / (h * h)
        return D

    def d_r(self, F):            # F: (n_r, n_z-1)
        return (self.D1 @ F) / self.drde[:, None]

    def d_rr(self, F):
        Fe = self.D1 @ F; Fee = self.D2 @ F
        return Fee / self.drde[:, None] ** 2 - Fe * (self.d2rde / self.drde ** 3)[:, None]

    def d_z(self, F):
        return self._dzeta(F) if self.zmap is None else self._dzeta(F) / self.zd[None, :]

    def d_zz(self, F):
        if self.zmap is None:
            return self._dzzeta(F)
        return self._dzzeta(F) / self.zd[None, :] ** 2 - self._dzeta(F) * (self.zdd / self.zd ** 3)[None, :]

    def _dzeta(self, F):            # sine-spectral: F = sum f_k sin(k z) -> F_z = sum k f_k cos(k z) (evaluate via DCT of shifted)
        fk = dst(F, type=1, axis=1) / (2 * self.n_z)          # DST-I coefficients (scipy normalization)
        # derivative: cos(2 pi k z_j); use the identity via a DCT-I on the extended grid
        from scipy.fft import dct
        gk = fk * self.kz[None, :]
        # cos series on interior nodes: sum_k g_k cos(k pi j/n_z), j = 1..n_z-1
        ext = np.concatenate([np.zeros((F.shape[0], 1)), gk, np.zeros((F.shape[0], 1))], axis=1)   # k = 0..n_z
        full = dct(ext, type=1, axis=1)                       # values at j = 0..n_z (DCT-I already carries the factor 2 on interior modes)
        return full[:, 1:-1]

    def _dzzeta(self, F):
        fk = dst(F, type=1, axis=1) / (2 * self.n_z)
        return idst(-fk * self.kz[None, :] ** 2, type=1, axis=1) * (2 * self.n_z) / (2 * self.n_z) * 1.0 if False else \
            -dst(fk * self.kz[None, :] ** 2 * (2 * self.n_z), type=1, axis=1) / (2 * self.n_z)

    def _poisson_matrix(self, k):
        n = self.n_r
        # operator on psi in eta: -(psi_ee/r'^2 - psi_e r''/r'^3 + (3/r) psi_e/r') + k^2 psi
        A = -(self.D2 / self.drde[:, None] ** 2) + self.D1 * (self.d2rde / self.drde ** 3)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            three_over_r = np.where(self.r > 0, self.dim / np.maximum(self.r, 1e-300), 0.0)
        A = A - self.D1 * (three_over_r / self.drde)[:, None]
        A = self.q2 * A + k * k * np.eye(n)
        # axis: regularity psi_r = 0 -> use the limit (psi_rr + 3/r psi_r) -> 4 psi_rr at r = 0
        A[0, :] = -self.q2 * (1.0 + self.dim) * (self.D2[0, :] / self.drde[0] ** 2) + k * k * np.eye(n)[0, :]
        A[0, :] += self.q2 * (1.0 + self.dim) * self.D1[0, :] * (self.d2rde[0] / self.drde[0] ** 3)
        # wall: psi = 0
        A[-1, :] = 0.0; A[-1, -1] = 1.0
        if os.environ.get("AXP_DENSE", "0") == "1":
            return np.linalg.inv(A)  # dense inverse per mode (legacy path)
        # the 4th-order stencils with the parity fold and the one-sided wall rows are pentadiagonal: banded storage (l = u = 2)
        n = A.shape[0]; ab = np.zeros((5, n))
        for k in range(-2, 3):                      # ab[2 + i - j, j] = A[i, j]; diagonal k = j - i
            dg = np.diagonal(A, k)
            if k >= 0: ab[2 - k, k:] = dg
            else: ab[2 - k, :n + k] = dg
        if not hasattr(self, "_band_checked"):
            assert np.abs(A - _unband(ab)).max() == 0.0, "Poisson matrix not pentadiagonal"; self._band_checked = True
        return ab

    def poisson(self, Om):
        if self.zmap is None:
            ok = dst(Om, type=1, axis=1)                      # unnormalized coefficients (linear -> fine)
        else:
            ok = (Om * self._Wh[None, :]) @ self.Vz
        ok[-1, :] = 0.0                                       # wall row rhs
        pk = np.empty_like(ok)
        if self._lu[0].shape[0] == 5:
            for j in range(ok.shape[1]):
                pk[:, j] = solve_banded((2, 2), self._lu[j], ok[:, j], check_finite=False)
        else:
            for j in range(ok.shape[1]):
                pk[:, j] = self._lu[j] @ ok[:, j]
        if self.zmap is None:
            return idst(pk, type=1, axis=1)
        return (pk @ self.Vz.T) / self._Wh[None, :]

    @staticmethod
    def _band_matvec(ab, w):
        """A w for the (2,2)-banded storage ab (ab[2 + i - j, j] = A[i, j])."""
        n = len(w); out = ab[2, :] * w
        out[:-1] += ab[1, 1:] * w[1:]; out[:-2] += ab[0, 2:] * w[2:]
        out[1:] += ab[3, :-1] * w[:-1]; out[2:] += ab[4, :-2] * w[:-2]
        return out

    def frac_lap(self, F, alpha):
        """-(-Lap)^alpha F for 0 < alpha < 1 on the uniform-z grid.  Per sine mode k, A = -(d_rr + dim/r d_r) + k^2 (the Poisson
        operator with the axis-limit row and the wall row) and A^alpha = A . A^{-beta}, beta = 1 - alpha, with
        A^{-beta} = (sin(pi beta)/pi) int_0^inf t^{-beta} (t I + A)^{-1} dt evaluated by the sinc quadrature in y = ln t
        (Bonito-Pasciak 2015, lit 135) plus the analytic tails: high (t >> lam_max): e^{-beta y_max}/beta . u,
        low (t << lam_min): e^{(1-beta) y_min}/(1-beta) . A^{-1} u."""
        assert self.zmap is None, "frac_lap needs the uniform z grid (mode-separable operator)"
        beta = 1.0 - alpha
        if not hasattr(self, "_frac_bands"):
            self._frac_bands = [self._poisson_matrix(k) for k in self.kz]
            lam_min = float(self.kz.min() ** 2); lam_max = 4.0 / float(np.diff(self.r).min()) ** 2 + float(self.kz.max() ** 2)
            h = float(os.environ.get("AXP_FRAC_H", "1.0")); self._frac_y = np.arange(np.log(lam_min) - 8.0, np.log(lam_max) + 8.0, h); self._frac_h = h
        y = self._frac_y; h = self._frac_h; c = np.sin(np.pi * beta) / np.pi
        fk = dst(F, type=1, axis=1); out = np.zeros_like(fk)
        for j, ab in enumerate(self._frac_bands):
            u = fk[:, j].copy(); u[-1] = 0.0
            Au = self._band_matvec(ab, u); Au[-1] = 0.0
            acc = np.zeros_like(u)
            for yy in y:
                t = np.exp(yy); abt = ab.copy(); abt[2, :-1] += t
                acc += h * np.exp((1.0 - beta) * yy) * solve_banded((2, 2), abt, Au, check_finite=False)
            acc += np.exp(-beta * y[-1]) / beta * Au                                              # high tail
            acc += np.exp((1.0 - beta) * y[0]) / (1.0 - beta) * solve_banded((2, 2), ab, Au, check_finite=False)   # low tail
            out[:, j] = c * acc
        G = -idst(out, type=1, axis=1)
        G[-1, :] = 0.0
        return G

    def filter_z(self, F, alpha=36.0, pexp=36.0):
        """Hou-Li 2007 exponential filter on the sine coefficients (in zeta): rho(k/N) = exp(-alpha (k/N)^p)."""
        fk = dst(F, type=1, axis=1)
        k = np.arange(1, self.n_z) / self.n_z
        return idst(fk * np.exp(-alpha * k ** pexp)[None, :], type=1, axis=1)

    def filter_r(self, F):
        """8th-order explicit filter in the radial index (uniform eta): F - delta^8 F / 256, transfer 1 - sin^8(k h/2)
        (damps only near-Nyquist modes: 6% at 4 cells, 100% at 2 cells); even fold at the axis, untouched last 4 wall rows."""
        n = self.n_r
        c = np.array([1, -8, 28, -56, 70, -56, 28, -8, 1]) / 256.0
        G = F.copy()
        Fe = np.concatenate([F[4:0:-1], F], axis=0)                 # even extension by 4 rows at the axis
        for i in range(0, n - 4):
            G[i] = F[i] - sum(c[k] * Fe[i + 4 + k - 4] for k in range(9))
        return G

    def rhs(self, U, Om, r_t=None):
        Ps = self.poisson(Om)
        Psz = self.d_z(Ps); Psr = self.d_r(Ps)
        ur = -self.r[:, None] * Psz
        uz = (self.dim - 1.0) * Ps + self.r[:, None] * Psr
        ura = ur if r_t is None else ur - r_t[:, None]        # mesh velocity: advection relative to the moving grid
        dU = -(ura * self.d_r(U) + uz * self.d_z(U)) + 2.0 * U * Psz
        dO = -(ura * self.d_r(Om) + uz * self.d_z(Om)) + self.d_z(U * U) - (self.dim - 3.0) * Psz * Om
        if self.nu > 0 and float(os.environ.get("AXP_ALPHA", "1")) < 1.0:                 # T9: hypodissipation nu (-Lap)^alpha, applied as a split step in main()
            pass
        elif self.nu > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_r = np.where(self.r > 0, 1.0 / np.maximum(self.r, 1e-300), 0.0)[:, None]
            for F, dF in ((U, dU), (Om, dO)):
                lap = self.d_rr(F) + self.dim * inv_r * self.d_r(F) + self.d_zz(F)
                lap[0, :] = (1.0 + self.dim) * self.d_rr(F)[0, :] + self.d_zz(F)[0, :]
                dF += self.nu * (float(os.environ.get("AXP_NU2", "1")) if F is Om else 1.0) * lap      # AXP_NU2: multiplier of the viscosity on omega1 (Hou 2405.10916 Sec. 5 two-viscosity model: 10)
        return dU, dO, Ps, ur, uz

    def initial(self):
        r = self.r[:, None]; z = self.z[None, :]
        eps4 = float(os.environ.get("AXP_ZADD", "0"))        # T8 family: sin(2 pi z) + eps4 sin(4 pi z) admixture (Hou's Case-4 direction)
        U = 12000.0 * (1 - r ** 2) ** 18 * (np.sin(2 * np.pi * z) + eps4 * np.sin(4 * np.pi * z)) / (1 + float(os.environ.get("AXP_ZSHARP", "12.5")) * np.sin(np.pi * z) ** 2)
        if os.environ.get("AXP_SWIRLFREE", "0") == "1":      # swirl-free start: the same shape as the vorticity dipole omega1 (odd in z)
            return np.zeros_like(U), U * float(os.environ.get("AXP_SFAMP", "1"))
        return U, np.zeros_like(U)


def zfront_params(P, U, Om, frac):
    """Front-following z-map target (AXP_ZFOLLOW): centre at the z of the |omega_theta| maximum, width = its FWHM in z
    (floored at 4 cells and at AXP_ZWMIN); the map puts a fraction `frac` of the z points within |z - z_c| < 3 w."""
    W = np.abs(P.r[:, None] * Om); i, j = np.unravel_index(int(np.argmax(W)), W.shape)
    col = W[i, :]; half = col > 0.5 * col[j]; jl = j
    while jl > 0 and half[jl - 1]: jl -= 1
    jr = j
    while jr < len(P.z) - 1 and half[jr + 1]: jr += 1
    wf = (float(P.z[jr] - P.z[jl]) + float(np.diff(P.z).min())) * float(os.environ.get("AXP_ZWMULT", "3"))   # window = ZWMULT x FWHM (map contrast ~ frac Lz / (6 w))
    wf = max(wf, 4 * float(np.diff(P.z).min()), float(os.environ.get("AXP_ZWMIN", "0")))
    return (float(P.z[j]), min(wf, 0.25 * P.Lz), frac)


def rezone_z(P, fields, zp_new):
    """Move the z-map to zp_new and evaluate the sine series of each field exactly at the new nodes (as axidyn_split)."""
    nz = P.n_z
    coeffs = [dst(F, type=1, axis=1) / nz for F in fields]
    zt_zeta, zt_z = P._zfine_zeta, P._zfine_z
    P.set_zmap(zp_new)
    zeta_old = np.interp(P.z, zt_z, zt_zeta)
    S = np.sin(np.pi * np.arange(1, nz)[None, :] * zeta_old[:, None] / P.Lz)
    return [c @ S.T for c in coeffs]


def main(n_r, n_z, t_end, nu=0.0):
    a = float(os.environ.get("AXP_A", "3.0")); tag = os.environ.get("AXP_TAG", ""); dim = float(os.environ.get("AXP_DIM", "3"))
    P = AxiPhys(n_r, n_z, nu=nu, a=a, dim=dim)
    U, Om = P.initial()
    t_start = 0.0
    if os.environ.get("AXP_RESTART"):                        # continue from a snapshot on the same grid
        d0 = np.load(os.environ["AXP_RESTART"]); U, Om, t_start = d0["U"], d0["Om"], float(d0["t"])
        print(f"  restart from {os.environ['AXP_RESTART']} at t = {t_start:.6f}")
        if d0["r"].shape != P.r.shape or np.abs(d0["r"] - P.r).max() > 1e-12:
            from scipy.interpolate import CubicSpline
            r_old = d0["r"]; rr = np.concatenate([-r_old[:0:-1], r_old])
            U = CubicSpline(rr, np.concatenate([U[:0:-1], U]), axis=0)(P.r); Om = CubicSpline(rr, np.concatenate([Om[:0:-1], Om]), axis=0)(P.r)
            print(f"  (interpolated from a different radial grid, n_r {len(r_old)})")
        if d0["z"].shape != P.z.shape or np.abs(d0["z"] - P.z).max() > 1e-12:
            # old grid uniform (DST nodes): exact sine-series evaluation at the new z nodes
            z_old = d0["z"]; nz_old = len(z_old) + 1
            zeta_new = P.z
            zm_old = tuple(float(v) for v in os.environ["AXP_RESTART_ZMAP"].split(",")) if os.environ.get("AXP_RESTART_ZMAP") else                      (tuple(float(v) for v in d0["zmap"]) if "zmap" in d0.files else (0.0, 0.0))
            if zm_old[0] > 0:                                              # old grid mapped: evaluate the old sine series at zeta_old(z_new)
                Pold = AxiPhys(len(d0["r"]), nz_old, a=a); Pold.set_zmap(tuple(zm_old))   # 2- or 3-parameter old map: use its inverse table
                zeta_new = np.interp(P.z, Pold._zfine_z, Pold._zfine_zeta)
            S = np.sin(2.0 * np.pi * np.arange(1, nz_old)[None, :] * zeta_new[:, None])       # (n_z_new-1, nz_old-1)
            U = (dst(U, type=1, axis=1) / nz_old) @ S.T; Om = (dst(Om, type=1, axis=1) / nz_old) @ S.T
            print(f"  (sine-interpolated onto the mapped z grid, min dz {np.diff(P.z).min():.2e})")
    rezone_every = int(os.environ.get("AXP_REZONE", "0")); filt = os.environ.get("AXP_FILTER", "0") == "1"
    if filt: print("  Hou-Li exponential filter (alpha = p = 36) on the sine coefficients after every step")
    if P.moving:
        prm = P.target_params(U, Om, float(os.environ.get("AXP_FRAC", "0.4")))
        U, Om = P.rezone([U, Om], prm)
        print(f"  moving map on: params (R_c, w, frac) = ({prm[0]:.4f}, {prm[1]:.4f}, {prm[2]:.2f}) min dr {np.diff(P.r).min():.2e}, rezone every {rezone_every} steps")
    zfollow = os.environ.get("AXP_ZFOLLOW", "0") == "1"
    if zfollow:                                                   # front-following z-map (lit 101), re-fitted with the radial rezone
        zpf = zfront_params(P, U, Om, float(os.environ.get("AXP_ZFRAC", "0.2")))
        U, Om = rezone_z(P, [U, Om], zpf)
        print(f"  z-follow map on: (z_c, w, frac) = ({zpf[0]:.5f}, {zpf[1]:.2e}, {zpf[2]:.2f}) min dz {np.diff(P.z).min():.2e} max dz {np.diff(P.z).max():.2e} zsym {P._zsym:.1e}", flush=True)
    print(f"AXIPHYS: n_r {n_r} n_z {n_z} nu {nu} dim {dim} map a {a} (min dr {np.diff(P.r).min():.2e}, dz {np.diff(P.z).min():.2e}, zmap {P.zmap})  t_end {t_end}", flush=True)
    t, it, t0 = t_start, 0, time.time()
    w0 = None; wtot0 = None; next_print = t_start; bkm = 0.0; t_last = t_start
    dr_min = float(np.diff(P.r).min()); dz = float(np.diff(P.z).min())
    while t < t_end:
        dU, dO, Ps, ur, uz = P.rhs(U, Om)
        vmax = max(np.abs(ur).max(), np.abs(uz).max(), 1e-12)
        dt = min(float(os.environ.get("AXP_CFL", "0.4")) * min(dr_min, dz) / vmax, float(os.environ.get("AXP_DTMAX", "2e-6")))   # lit 84: RK4 + spectral allows ~0.8-0.9 dz/v
        if nu > 0:                                   # (1+n)-aware diffusion limits (lit check 53)
            dt = min(dt, 0.52 / (1.0 + P.dim) * dr_min ** 2 / nu, 2.785 / (nu * np.pi ** 2 / dz ** 2))
        dt = min(dt, t_end - t)
        k1 = (dU, dO)
        U2, O2 = U + 0.5 * dt * k1[0], Om + 0.5 * dt * k1[1]; k2 = P.rhs(U2, O2)[:2]
        U3, O3 = U + 0.5 * dt * k2[0], Om + 0.5 * dt * k2[1]; k3 = P.rhs(U3, O3)[:2]
        U4, O4 = U + dt * k3[0], Om + dt * k3[1]; k4 = P.rhs(U4, O4)[:2]
        U = U + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Om = Om + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if nu > 0 and float(os.environ.get("AXP_ALPHA", "1")) < 1.0:                       # T9: fractional dissipation, explicit split step (dt nu lam_max^alpha << 1)
            al = float(os.environ.get("AXP_ALPHA", "1")); U = U + dt * nu * P.frac_lap(U, al); Om = Om + dt * nu * float(os.environ.get("AXP_NU2", "1")) * P.frac_lap(Om, al)
        t += dt; it += 1
        if filt:
            U = P.filter_r(P.filter_z(U)); Om = P.filter_r(P.filter_z(Om))
        if P.moving and rezone_every and it % rezone_every == 0:
            tgt = P.target_params(U, Om, P.params[2]); relax = float(os.environ.get("AXP_RELAX", "0.5"))
            prm = (P.params[0] + relax * (tgt[0] - P.params[0]), P.params[1] + relax * (tgt[1] - P.params[1]), P.params[2])
            U, Om = P.rezone([U, Om], prm); dr_min = np.diff(P.r).min()
            if zfollow:
                tz = zfront_params(P, U, Om, P.zmap[2]); relax = float(os.environ.get("AXP_RELAX", "0.5"))
                zpf = (P.zmap[0] + relax * (tz[0] - P.zmap[0]), P.zmap[1] + relax * (tz[1] - P.zmap[1]), P.zmap[2])
                U, Om = rezone_z(P, [U, Om], zpf); dz = float(np.diff(P.z).min())
        if t >= next_print or t >= t_end:
            wth = np.abs(P.r[:, None] * Om).max()
            if w0 is None and wth > 0:
                w0 = wth
            i, j = np.unravel_index(int(np.argmax(np.abs(U))), U.shape)
            # lit-check-49 diagnostics: L^d norm of u with the d-dimensional measure r^{n-2} dr dz, BKM integral of ||omega||_inf dt
            u2 = ur ** 2 + uz ** 2 + (P.r[:, None] * U) ** 2
            wtot = np.sqrt((P.r[:, None] * P.d_z(U)) ** 2 + (2.0 * U + P.r[:, None] * P.d_r(U)) ** 2 + (P.r[:, None] * Om) ** 2).max()
            dd = P.dim                                     # critical norm L^d of R^d (space dimension = n), measure r^{n-2} dr dz
            L5 = float(np.trapezoid(np.trapezoid(u2 ** (dd / 2.0) * P.r[:, None] ** (P.dim - 2.0), P.z, axis=1), P.r)) ** (1.0 / dd)
            Gmax = float(np.abs(P.r[:, None] ** 2 * U).max())   # swirl Gamma = r^2 u1: max principle -> non-increasing
            bkm = bkm + wtot * (t - t_last); t_last = t
            wgt = P.r[:, None] ** (P.dim - 2.0)
            Kz = float(np.trapezoid(np.trapezoid(uz ** 2 * wgt, P.z, axis=1), P.r)); K = float(np.trapezoid(np.trapezoid(u2 * wgt, P.z, axis=1), P.r))
            extra = f"  |om1| {np.abs(Om).max():.3e}  Kz/K {Kz / max(K, 1e-300):.4f}  Gmax {Gmax:.4e}"
            if P.moving: extra += f"  map ({P.params[0]:.4f},{P.params[1]:.4f}) dr_min {dr_min:.1e}"
            print(f"  t {t:.6f} it {it} dt {dt:.2e}  |u1| {np.abs(U).max():.4e} at (r={P.r[i]:.4f}, z={P.z[j]:.4f})  |omega| {wth:.4e} (x{wth/(w0 or 1):.3g})"
                  f"  |omega|tot {wtot:.3e}  L{dd:.0f}u {L5:.3e}  BKM {bkm:.3e}{extra}  |psi1| {np.abs(Ps).max():.3e}  vmax {vmax:.3e}  ({time.time()-t0:.0f}s)", flush=True)
            next_print = t + float(os.environ.get("AXP_EVERY", "1e-4"))
            if not np.isfinite(wth):
                print("  non-finite; stop"); break
            if wtot0 is None:
                wtot0 = wtot
            if wtot / wtot0 > float(os.environ.get("AXP_STOPAMP", "1e300")):
                print(f"  |omega|tot amplification {wtot/wtot0:.3g} reached AXP_STOPAMP; stop"); np.savez(f"axiphys_{n_r}_{n_z}{tag}.npz", U=U, Om=Om, Ps=Ps, r=P.r, z=P.z, t=t, nu=nu); break
            np.savez(f"axiphys_{n_r}_{n_z}{tag}_t{t:.5f}.npz", U=U, Om=Om, Ps=Ps, r=P.r, z=P.z, t=t, nu=nu, dim=P.dim, zmap=np.array(P.zmap if P.zmap else (0.0, 0.0)))
        if it % 2000 == 0:
            np.savez(f"axiphys_{n_r}_{n_z}{tag}.npz", U=U, Om=Om, Ps=Ps, r=P.r, z=P.z, t=t, nu=nu)
    np.savez(f"axiphys_{n_r}_{n_z}{tag}.npz", U=U, Om=Om, Ps=Ps, r=P.r, z=P.z, t=t, nu=nu)
    print(f"  saved axiphys_{n_r}_{n_z}{tag}.npz  ({time.time()-t0:.0f}s, {it} steps)")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
