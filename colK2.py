"""Stage B step 3 preview, corrected: the inverse-norm constant on the DISCRETE DIVERGENCE-FREE subspace.
Null space Z of the constraint rows (continuity + wall rows) gives x = Z c; the pencil restricted to it is (Z^H W A Z, Z^H W B Z)
with W the r dr quadrature weight on the velocity blocks; singular values of the restricted, weighted (A - mu B) above the
eigen-direction give 1/K.  python colK2.py colenc_<tag>.npz"""
import sys, numpy as np
from scipy.linalg import null_space, svd
f = sys.argv[1]; d = np.load(f)
A, B, w0, rmax, k, nu, N = d["Amid"], d["Bmid"], complex(d["w0"]), float(d["rmax"]), float(d["k"]), float(d["nu"]), int(d["N"])
n = A.shape[0] // 4; M2 = 2 * N; xg = np.cos(np.pi * (np.arange(M2) + 0.5) / M2); rp = rmax * xg[:N]
wq = (np.pi / M2) * np.sqrt(1 - xg[:N] ** 2) * rmax * rp; scale = nu * k ** 2
# constraint rows: continuity block (rows 3n..4n) and the wall rows (u = 0 at index 0 of each velocity block)
Ccont = A[3 * n:, :3 * n]                              # continuity acts on velocity only (pressure column block is zero)
wall = np.zeros((3, 3 * n), dtype=complex)
for b in range(3): wall[b, b * n + 0] = 1.0
Cons = np.vstack([Ccont, wall])
Z = null_space(Cons)                                    # 3n x (3n - N - 3) orthonormal basis of the discrete div-free, no-slip velocities
Av = A[:3 * n, :3 * n]; Gp = A[:3 * n, 3 * n:]          # velocity-velocity block and the pressure-gradient block
# on the div-free subspace the pressure enforces the constraint: project the momentum residual onto Z^H W (Leray form)
Wv = np.diag(np.concatenate([wq, wq, wq]))
# weighted orthonormal basis: Q = Z (Z^H W Z)^{-1/2}
Mzz = Z.conj().T @ Wv @ Z; ev, U = np.linalg.eigh(Mzz); Q = Z @ U @ np.diag(ev ** -0.5) @ U.conj().T
Ared = Q.conj().T @ Wv @ Av @ Q / scale                # Leray-projected, weighted, nondimensional
w, V = np.linalg.eig(Ared); order = np.argsort(-w.imag)
print(f"{f}: N {N}; Leray-reduced dim {Ared.shape[0]}; leading eigenvalues (units nu k^2): {w[order[0]]:.5f}, {w[order[1]]:.5f}, {w[order[2]]:.5f}; full-pencil omega/(nu k^2) = {w0/scale:.5f}")
mu = w[order[0]]
s = svd(Ared - mu * np.eye(len(Ared)), compute_uv=False)
print(f"  singular values of (A_Leray - mu): smallest {s[-1]:.2e}, then {s[-2]:.4f}, {s[-3]:.4f}, {s[-4]:.4f}  -> K_Leray = 1/s_2 = {1/s[-2]:.4f}; 1/gap = {1/abs(w[order[0]]-w[order[1]]):.4f}; non-normality ratio {(1/s[-2])/(1/abs(w[order[0]]-w[order[1]])):.3f}")
for sh in (0.5, 1.0, 2.0):
    for di in (1.0, 1j):
        s2 = svd(Ared - (mu + di * sh * abs(w[order[0]]-w[order[1]])) * np.eye(len(Ared)), compute_uv=False)
        print(f"  resolvent norm at mu + {di}*{sh} gap: {1/s2[-1]:.4f} vs 1/|dist| {1/(sh*abs(w[order[0]]-w[order[1]])):.4f}")
