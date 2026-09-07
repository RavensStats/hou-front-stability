"""Standalone tests for ivmat.py (no pytest).  Prints PASS/FAIL lines, exit code 1 on failure."""
import os
import sys
import time
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ivmat  # noqa: E402
from ivmat import IvMat, mid_rad_matmul, krawczyk_eigpair, enclose_pencil_eigen, gamma, outward  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("PASS: " if cond else "FAIL: ") + msg)
    if not cond:
        FAILS.append(msg)


def fr(z):
    """exact (Fraction re, Fraction im) of a complex/real float."""
    z = complex(z)
    return Fraction(z.real), Fraction(z.imag)


def fmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def inside(exact, mid, rad):
    """exact (Fraction pair) lies in the disc |z - mid| <= rad, checked exactly."""
    m = fr(mid)
    d = (exact[0] - m[0], exact[1] - m[1])
    return d[0] * d[0] + d[1] * d[1] <= Fraction(float(rad)) ** 2


rng = np.random.default_rng(12345)
UF = Fraction(2) ** -53

# ---------------------------------------------------------------- rounding helpers
ok = True
for k in [0, 1, 2, 5, 60, 104, 2004, 4008, 10 ** 6]:
    ok &= Fraction(gamma(k)) >= Fraction(k) * UF / (1 - Fraction(k) * UF)
check(ok, "gamma(k) >= k u/(1 - k u) exactly (Fraction check)")
ok = True
for k in [0, 1, 3, 8, 60, 1008, 10 ** 5]:
    for x in rng.uniform(1e-300, 1e300, 5).tolist() + [1.0, 0.3, 7e-10]:
        ok &= Fraction(float(outward(x, k))) >= Fraction(x) * (1 + 2 * UF) ** k
check(ok, "outward(x,k) >= x (1+2u)^k exactly (Fraction check)")

# ---------------------------------------------------------------- (a) matmul enclosure
n = 50
A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
C = mid_rad_matmul(IvMat(A), IvMat(B))
ok = True
for _ in range(20):
    i, j = rng.integers(n), rng.integers(n)
    ex = (Fraction(0), Fraction(0))
    for kk in range(n):
        p = fmul(fr(A[i, kk]), fr(B[kk, j]))
        ex = (ex[0] + p[0], ex[1] + p[1])
    ok &= inside(ex, C.mid[i, j], C.rad[i, j])
check(ok, "(a) point matmul: exact Fraction product inside enclosure on 20 entries")
scale = max(1.0, float(np.max(np.abs(C.mid))))
check(float(C.rad.max()) <= 1e-10 * scale,
      "(a) point matmul radii sensible: max rad %.3e <= 1e-10 * scale %.2f" % (C.rad.max(), scale))
check(np.allclose(C.mid, A @ B, rtol=1e-12, atol=1e-12), "(a) midpoint agrees with numpy A@B")

# interval inputs: perturb inside the discs and check exact product of the perturbed matrices
rA = rng.uniform(1e-13, 2e-13, (n, n))
rB = rng.uniform(1e-13, 2e-13, (n, n))
rB[:, :5] = 0.0
Civ = mid_rad_matmul(IvMat(A, rA), IvMat(B, rB))
thA, thB = rng.uniform(0, 2 * np.pi, (n, n)), rng.uniform(0, 2 * np.pi, (n, n))
Ap = A + 0.9 * rA * np.exp(1j * thA)
Bp = B + 0.9 * rB * np.exp(1j * thB)
ok, ok_pert = True, True
for _ in range(20):
    i, j = rng.integers(n), rng.integers(n)
    ex = (Fraction(0), Fraction(0))
    for kk in range(n):
        for (P, Pm, Pr, r, c) in ((Ap, A, rA, i, kk), (Bp, B, rB, kk, j)):
            d0, d1 = fr(P[r, c] - Pm[r, c])  # float difference; verify the perturbed entry is in its disc
            e0, e1 = fr(P[r, c])
            m0, m1 = fr(Pm[r, c])
            ok_pert &= (e0 - m0) ** 2 + (e1 - m1) ** 2 <= Fraction(float(Pr[r, c])) ** 2
        p = fmul(fr(Ap[i, kk]), fr(Bp[kk, j]))
        ex = (ex[0] + p[0], ex[1] + p[1])
    ok &= inside(ex, Civ.mid[i, j], Civ.rad[i, j])
check(ok_pert, "(a) perturbed matrices really lie inside the interval inputs (exact check)")
check(ok, "(a) interval matmul: exact product of perturbed matrices inside enclosure (20 entries)")
check(np.all(Civ.rad >= C.rad), "(a) interval radii dominate point radii")

# scalar / add / sub exactness checks with Fractions on a 1-vector
v = IvMat(np.array([0.1 + 0.7j, -3.3 + 1e-3j, 2.0]), np.array([1e-12, 0.0, 3e-15]))
s = IvMat(np.array(0.3 - 0.9j), np.array(2e-14))
sv = v.scale(s)
ok = True
for i in range(3):
    for (ths, thv) in [(0.3, 1.1), (2.0, 4.0), (5.5, 0.0)]:
        sp = s.mid + 0.999 * s.rad * np.exp(1j * ths)
        vp = v.mid[i] + 0.999 * v.rad[i] * np.exp(1j * thv)
        ok &= inside(fmul(fr(sp), fr(vp)), sv.mid[i], sv.rad[i])
check(ok, "scalar interval multiply encloses exact products of interior points")
d = v - sv
ok = True
for i in range(3):
    a, b = fr(v.mid[i]), fr(sv.mid[i])
    ok &= inside((a[0] - b[0], a[1] - b[1]), d.mid[i], d.rad[i]) and d.rad[i] >= v.rad[i] + sv.rad[i]
check(ok, "interval subtraction encloses exact midpoint difference and sums radii")

# contains: rigor at the boundary
Y = IvMat(np.array([1.0 + 0j]), np.array([1e-10]))
check(Y.contains(IvMat(np.array([1.0 + 3e-11j]), np.array([5e-11]))), "contains: clearly interior -> True")
check(not Y.contains(IvMat(np.array([1.0 + 5e-11j]), np.array([5e-11]))), "contains: touching boundary -> False")
check(not Y.contains(IvMat(np.array([1.0 + 0j]), np.array([np.inf]))), "contains: infinite radius -> False")
check(not IvMat(np.array([1.0 + 0j]), np.array([np.inf])).contains(IvMat(np.array([1.0 + 0j]), np.array([1.0]))),
      "contains: infinite outer radius -> False")


# ---------------------------------------------------------------- helpers for eigen tests
def nearest_eig_dist(w, eigs):
    return float(np.min(np.abs(eigs - w)))


def report_cert(tag, ok_flag, W, X, eigs, A_, B_):
    """any certified interval must contain a numpy eigenvalue of the pencil (within rad + 1e-10)"""
    if not ok_flag:
        return
    dmin = nearest_eig_dist(W.mid, eigs)
    check(dmin <= float(W.rad) + 1e-10,
          "%s: certified W contains a pencil eigenvalue (dist %.2e <= rad %.2e + 1e-10)" % (tag, dmin, W.rad))
    res = np.abs(A_ @ X.mid - complex(W.mid) * (B_ @ X.mid)).max()
    check(res < 1e-8, "%s: residual of certified midpoint pair %.2e" % (tag, res))


# ---------------------------------------------------------------- (b) standard problem, B = I
n = 30
A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
B = np.eye(n)
eigs, vecs = np.linalg.eig(A)
gaps = np.array([np.min(np.abs(np.delete(eigs, i) - eigs[i])) for i in range(n)])
i0 = int(np.argmax(gaps))
w0, x0 = eigs[i0], vecs[:, i0]
k = int(np.argmax(np.abs(x0)))
ok_flag, W, X = krawczyk_eigpair(IvMat(A), IvMat(B), w0, x0, k)
check(ok_flag, "(b) B=I, n=30: certificate obtained")
if ok_flag:
    check(abs(W.mid - w0) <= float(W.rad) + 1e-10, "(b) W contains the numpy eigenvalue")
    check(float(W.rad) < 1e-9, "(b) rad(W) = %.3e < 1e-9" % W.rad)
    check(X.mid[k] == 1.0 and X.rad[k] == 0.0, "(b) normalization component is exactly 1")
    report_cert("(b)", ok_flag, W, X, eigs, A, B)

# same with a small interval radius on A (matrix entries themselves enclosures)
ok_flag2, W2, X2 = enclose_pencil_eigen(A, B, w0, x0, k, A_rad=np.full((n, n), 1e-13))
check(ok_flag2, "(b') A with radius 1e-13: certificate obtained")
if ok_flag2:
    check(float(W2.rad) > float(W.rad) and abs(W2.mid - w0) <= float(W2.rad) + 1e-10,
          "(b') interval-A enclosure wider (rad %.3e) and still contains eigenvalue" % W2.rad)

# ---------------------------------------------------------------- (c) singular B
n = 30
A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
Bd = np.ones(n)
Bd[-2:] = 0.0
B = np.diag(Bd)
try:
    import scipy.linalg as sla
    ge, gv = sla.eig(A, B)
    src = "scipy.linalg.eig"
except ImportError:  # Schur-complement fallback: finite eigenvalues of the pencil
    m = n - 2
    S = A[:m, :m] - A[:m, m:] @ np.linalg.solve(A[m:, m:], A[m:, :m])
    e1, v1 = np.linalg.eig(S)
    ge = e1
    gv = np.vstack([v1, -np.linalg.solve(A[m:, m:], A[m:, :m] @ v1)])
    src = "Schur complement"
fin = np.where(np.isfinite(ge))[0]
check(len(fin) == n - 2, "(c) pencil has %d finite eigenvalues (expected %d) [%s]" % (len(fin), n - 2, src))
fe = ge[fin]
gaps = np.array([np.min(np.abs(np.delete(fe, i) - fe[i])) for i in range(len(fe))])
i0 = fin[int(np.argmax(gaps))]
w0, x0 = ge[i0], gv[:, i0]
k = int(np.argmax(np.abs(x0)))
ok_flag, W, X = enclose_pencil_eigen(A, B, w0, x0, k)
check(ok_flag, "(c) singular B (two zero rows): certificate obtained")
if ok_flag:
    check(abs(W.mid - w0) <= float(W.rad) + 1e-10, "(c) W contains the %s eigenvalue (rad %.3e)" % (src, W.rad))
    report_cert("(c)", ok_flag, W, X, fe, A, B)

# ---------------------------------------------------------------- (d) negative controls
n = 30
A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
B = np.eye(n)
eigs, vecs = np.linalg.eig(A)
n_cert_wrong = 0
n_cert = 0
for trial in range(6):
    i0 = int(rng.integers(n))
    w0, x0 = eigs[i0], vecs[:, i0]
    k = int(np.argmax(np.abs(x0)))
    if trial < 3:
        wbad, xbad = w0 * 1.1, x0                                   # 10 % wrong eigenvalue
    else:
        wbad, xbad = w0 * 1.1, x0 + 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    ok_flag, W, X = krawczyk_eigpair(IvMat(A), IvMat(B), wbad, xbad, k)
    if ok_flag:
        n_cert += 1
        dmin = nearest_eig_dist(W.mid, eigs)
        if dmin > float(W.rad) + 1e-10:
            n_cert_wrong += 1
        print("      (d) trial %d: certified, rad(W)=%.2e, dist to nearest eigenvalue %.2e" % (trial, W.rad, dmin))
    else:
        print("      (d) trial %d: not certified (flag False)" % trial)
check(n_cert_wrong == 0, "(d) negative control: no certified interval misses every pencil eigenvalue "
      "(%d/6 certified, all containing a true eigenvalue)" % n_cert)

# a really wrong pair (w0 far away, random x0): should not certify
ok_flag, W, X = krawczyk_eigpair(IvMat(A), IvMat(B), 100.0 + 100.0j,
                                 rng.standard_normal(n) + 1j * rng.standard_normal(n), 0)
check((not ok_flag) or nearest_eig_dist(W.mid, eigs) <= float(W.rad) + 1e-10,
      "(d) far-away w0 and random x0: flag=%s (never a wrong certificate)" % ok_flag)

# ---------------------------------------------------------------- (e) performance sanity, n = 400
n = 400
A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
eigs, vecs = np.linalg.eig(A)
i0 = int(np.argmax(np.abs(eigs)))
x0 = vecs[:, i0]
k = int(np.argmax(np.abs(x0)))
t0 = time.time()
ok_flag, W, X = krawczyk_eigpair(IvMat(A), IvMat(np.eye(n)), eigs[i0], x0, k)
dt = time.time() - t0
check(ok_flag and abs(W.mid - eigs[i0]) <= float(W.rad) + 1e-10,
      "(e) n=400 certified in %.2f s, rad(W)=%.2e" % (dt, W.rad if ok_flag else np.nan))

print()
if FAILS:
    print("%d FAILURE(S):" % len(FAILS))
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print("ALL TESTS PASSED")
