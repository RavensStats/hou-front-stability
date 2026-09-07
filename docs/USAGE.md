# Script usage (arguments and environment variables)

Generated from the scripts' docstrings and their `os.environ` reads. Positional arguments are listed as the docstring gives them; environment variables are read at start-up. All scripts take a snapshot `.npz` produced by `axiphys.py` (keys: `U` = u_theta / r, `Om` = omega_theta / r, `Ps` = psi / r, `r`, `z`, `t`, `zmap`, `nu`).

## axiphys.py

AXIPHYS: physical-space axisymmetric Euler/Navier-Stokes with swirl in Hou-Luo variables, from Hou's interior initial data (arXiv:2107.05870 / 2107.06509), to generate a developed seed for the rescaled profile solver.

Usage: `python axiphys.py <n_r> <n_z> <t_end> [nu]     (AXP_A map strength, AXP_TAG)`

Environment: `AXP_A` (default `3.0`), `AXP_ALPHA` (default `1`), `AXP_CFL` (default `0.4`), `AXP_DENSE` (default `0`), `AXP_DIM` (default `3`), `AXP_DTMAX` (default `2e-6`), `AXP_EVERY` (default `1e-4`), `AXP_FILTER` (default `0`), `AXP_FRAC` (default `0.4`), `AXP_FRAC_H` (default `1.0`), `AXP_MOVING` (default `0`), `AXP_NU2` (default `1`), `AXP_RELAX` (default `0.5`), `AXP_RESTART`, `AXP_RESTART_ZMAP`, `AXP_REZONE` (default `0`), `AXP_SFAMP` (default `1`), `AXP_STOPAMP` (default `1e300`), `AXP_SWIRLFREE` (default `0`), `AXP_TAG`, `AXP_WINF` (default `1.5`), `AXP_ZADD` (default `0`), `AXP_ZBETA` (default `0`), `AXP_ZFOLLOW` (default `0`), `AXP_ZFRAC` (default `0.2`), `AXP_ZM` (default `1`), `AXP_ZSHARP` (default `12.5`), `AXP_ZSHOULDER` (default `4.0`), `AXP_ZWMIN` (default `0`), `AXP_ZWMULT` (default `3`)

## axi3dlin.py

AXI3DLIN (T15 stage 2): linear stability of a FROZEN axisymmetric swirling Euler state to the azimuthal mode m >= 1 (lit checks 136-138), on the snapshot's (r, z) grid with the full z period [0, 1) (Fourier).  Raw perturbation variables (u_r', u_th', u_z', p') e^{i m theta}; axis regularity enters through the r-parity of each field (u_r', u_th' ~ r^{|m|-1}: parity (-1)^{m-1}; u_z', p' ~ r^{|m|}: parity (-1)^m), imposed by the fold in the 4th-order stencils, and p'(0) = 0 for m >= 1.  Base: U_theta = r u1, U_r = -r psi1_z, U_z = 2 psi1 + r psi1_r.   D F = U_r d_r F + U_z d_z F + i m u1 F   N_r = D u_r' + u_r' d_r U_r + u_z' d_z U_r - 2 u1 u_th'   N_th = D u_th' + u_r' d_r U_th + u_z' d_z U_th + u1 u_r' - psi1_z u_th'   N_z = D u_z' + u_r' d_r U_z + u_z' d_z U_z   d_t u' = -N - grad p',  grad p' = (d_r p', i m p'/r, d_z p'),  div v = d_r v_r + (v_r + i m v_th)/r + d_z v_z,   -Lap_m p' = div N with the pressure operator built as the EXACT discrete divergence of the discrete gradient. RK4 with projection at every stage; sigma = d ln E/dt with E = int |u'|^2 r dr dz, compared with 1/(T - t). Mapped-z snapshots are transferred to the uniform full-period grid by exact sine-series evaluation.    python axi3dlin.py <snapshot> <T_est> <m> [steps] [nz_full]

Usage: `python axi3dlin.py <snapshot> <T_est> <m> [steps] [nz_full]`

Environment: `AXL_KZ` (default `0`), `AXL_R0` (default `0.5`), `AXL_RNG` (default `1`), `AXL_SEED`, `AXL_SEED_R0`, `AXL_SEED_Z0`, `AXL_U1PAR` (default `odd`), `AXL_VISC` (default `1`), `AXL_ZMAP`

## axi3dtl.py

AXI3DTL (T15, lit 148 protocol): TANGENT-LINEAR run -- the nonlinear axisymmetric base (AxiPhys, Hou-Luo variables) and the linear mode-m perturbation (axi3dlin operator) are marched TOGETHER by one RK4, the perturbation linearized about the instantaneous base at every stage.  Growth is measured relative to the collapsing base: the local exponent lambda = d ln(a_pert / a_base) / d ln A (a = sqrt of the energy in the core box, A = max |u1|), so lambda > 0 means the perturbation outruns the collapse (A ~ 1/(T-t): a_pert/a_base ~ (T-t)^{-lambda}).  Also printed: sigma (T-t) with T_est, the translation-mode overlap (m = 1), the kz centroid and the production split.    python axi3dtl.py <snapshot on a uniform-z AxiPhys grid> <T_est> <m> [steps]     env: AXP_A (map), AXP_FILTER=1 (both fields)

Usage: `python axi3dtl.py <snapshot on a uniform-z AxiPhys grid> <T_est> <m> [steps]     env: AXP_A (map), AXP_FILTER=1 (both fields)`

Environment: `AXL_NU`, `AXL_RNG` (default `1`), `AXL_SEED`, `AXL_ZMAP`, `AXP_A` (default `5.0`), `AXP_FILTER` (default `0`)

## axi3dnl.py

AXI3DNL (T15 nonlinear stage; lit 157 design 3): axisymmetric base in Hou-Luo variables (AxiPhys: u1, omega1, psi1) PLUS azimuthal modes m = 1..M-1 of the velocity (u_r, u_th, u_z)_m e^{i m theta} (real field: u_{-m} = conj u_m), all marched by one RK4.  Mode equations: d_t u_m = -[L_m(U0) u_m]  -  (u'.grad u')_m  -  grad_m p_m  +  nu Lap_m u_m, where L_m(U0) is the validated tangent-linear operator (axi3dlin/axi3dtl) and (u'.grad u')_m is the mode-mode convolution evaluated pseudo-spectrally in theta on N_th = 3M points (dealiased).  Base feedback: the m = 0 component of -(u'.grad u') forces u1 (theta-momentum / r) and omega1 (curl_theta / r).  Diagnostics: base A = max u1 and location, mode energies, the 3D maximum of u_theta / r over theta (base + modes), the m = 0 forcing size, sigma (T-t) of the m = 1 energy.    python axi3dnl.py <snapshot> <T_est> <M> <eps> [steps]     (eps: seed amplitude of m = 1 relative to max |U_theta|)    env: AXL_ZMAP, AXP_A, AXL_SEED=random

Usage: `python axi3dnl.py <snapshot> <T_est> <M> <eps> [steps]     (eps: seed amplitude of m = 1 relative to max |U_theta|)`

Environment: `AXL_RESTART`, `AXL_ZMAP`, `AXP_A` (default `5.0`)

## axi3deig.py

Frozen-base m-mode EIGENPROBLEM (stage 1 of the eigenvalue enclosure): reuse axi3dlin's discrete linearized operator (projection included) as a matrix-free linear map on the divergence-free perturbations and compute its leading eigenvalues with ARPACK (largest real part), instead of reading the growth rate off a time march.  Reports sigma (T-t) for the leading eigenvalues, the a posteriori residual ||L v - lambda v|| / ||v|| of each eigenpair (exact for the discrete operator), the divergence of the eigenvector, its energy centroid and translation-mode overlap (m = 1).    python axi3deig.py <snapshot> <T_est> <m> [nz_full] [k]

Usage: `python axi3deig.py <snapshot> <T_est> <m> [nz_full] [k]`

Environment: `EIG_GMRES_MAXIT` (default `20`), `EIG_GMRES_RESTART` (default `60`), `EIG_GMRES_TOL` (default `1e-6`), `EIG_MAXITER` (default `400`), `EIG_NCV`, `EIG_PREC`, `EIG_PREC_RC` (default `0.0137`), `EIG_PROP` (default `0`), `EIG_SHIFT`, `EIG_SHIFT_IM` (default `0`), `EIG_TOL` (default `1e-7`)

## colsolve.py

COLSOLVE: normal modes of the columnar vortex (V(r) = r Omega(r), W(r)) taken through the swirl maximum of a snapshot, the same field that lscert2.py certifies (Leibovich-Stewartson).  Perturbation ~ exp(i m theta + i k z - i omega t); growth rate = Im omega.  Linearized incompressible Euler (or Navier-Stokes with nu > 0) about the columnar base, Chebyshev collocation on r in [0, rmax] with the parity (axis) conditions imposed by the mirror trick (Trefethen Program 11 style): for m odd, u_r and u_theta are even in r and u_z, p are odd; for m even the reverse.  Wall condition u_r = 0 at rmax (no-slip for nu > 0).  Generalized eigenproblem A x = omega B x with the continuity row singular in B; infinite eigenvalues filtered.  Reports the leading growth rates versus k and compares with the Leibovich-Stewartson local estimate.    python colsolve.py <snapshot> [N] [rmax/R] [m] [k list, e.g. 200,400,800] [nu: 'snap' or a float, default 0]

Usage: `python colsolve.py <snapshot> [N] [rmax/R] [m] [k list, e.g. 200,400,800] [nu: 'snap' or a float, default 0]`

Environment: `COL_DEG` (default `20`), `COL_Z`

## lscert2.py

Computer-assisted certificate, exact version.  The columnar vortex (V(r), W(r)) built from the column through the swirl maximum of a snapshot is DEFINED by polynomials Omega(s), W(s) in s = (r/rmax)^2 with exactly rational coefficients (the double-precision least-squares Chebyshev fit, converted exactly).  Leibovich-Stewartson's discriminant is Phi = 2 r Omega^2 B with B(r) = (2 r Omega + r^2 Omega') Omega' + 2 W W', a polynomial in r with rational coefficients.  B < 0 on [r1, r2] is certified by (i) Sturm's theorem: B has no real root in [r1, r2] (exact rational arithmetic, sympy), and (ii) B(r_mid) < 0 exactly. Usage: python lscert2.py <snapshot> [degree] [rmax/R] [z-column: 'max' or a float]

## lscheck.py

Leibovich-Stewartson (JFM 126 1983) and Howard-Gupta (JFM 14 1962) criteria on an axisymmetric snapshot (lit 183): LS: non-axisymmetric short-wave instability where Phi = 2 V Om [D(rV) D(Om) + D(W^2)] < 0  (V = u_theta = r u1, Om = u1, W = u_z, D = d/dr) HG: axisymmetric stability where (1/r^3) D(Gamma^2) - (1/4)(DW)^2 > 0  (Gamma = r V = r^2 u1)

## knssq.py

The KNSS-hypothesis quantities on the snapshots: K1 = sup |u| |x'| (the |v| <= C/|x'| bound, |x'| = r), K1core = |u| r at the swirl maximum, K2 = sup |u| (|x| + sqrt(T-t)) (the Type I constant), Gamma = r^2 u1: max, its radius, sign, and monotonicity beyond the maximum.

## logfit2.py

log-derivative fit on snapshot amplitudes (9-decimal times): R = A/A' = (T-t)/p linear in t

## burgfit.py

U3: fit the vorticity profile across the resolved front to the Rott-Lundgren (Burgers) layer omega = (DU/(sqrt(2 pi) delta)) exp(-(z-z0)^2/(2 delta^2)), delta^2 = nu/a; report delta in Burgers units, the implied strain a(T-t) = nu (T-t)/delta^2, the fitted vs measured velocity jump, and the residual.

## nszstat.py

(no docstring)

## l3conc.py

U4: scale-invariant local L^3 concentration (Barker-Prange 2003.06717: a Type I singularity needs ||u||_{L^3(B(0,R))} >= C(M) log(1/(T-t)) with R ~ (T-t)^{1/2-}); E3(k) = int_{|x| < k sqrt(T-t)} |u|^3 dx (dimensionless under Leray scaling), about the axis point the ring collapses to.

## figdata_extract.py

Extract (fraction of remaining time, sigma (T-t)) series from the tangent-linear / frozen-base logs into CSV for the paper's figures.

## make_figures.py

Paper figures from the extracted CSV series and the recorded constants.  Output: figures/*.png and *.pdf.
