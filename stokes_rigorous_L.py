"""STOKES_RIGOROUS (Stage B step 1, rigorous): certified enclosures of ALL zeros of the Stokes determinant Delta(x) of the
(m = 1, k = 2000) mode on the cylinder of radius r_w = rmax cos(pi/(4N)), N = 100, rmax = 0.027546 (X = k r_w = 55.09), Arb balls
(python-flint, prec 128).  Delta from stokes_mode.py, third column divided by I_m(X) > 0 (a positive constant, ~1e22: keeps the
entries O(1); zeros unchanged); expanded (a31 = 0):
    Delta(x) = m^2 (X^2 + x^2) J^2 - X^2 x^2 J'^2 - X Ir x^3 J J',   J = J_m(x), J' = J_{m-1}(x) - m J/x,   Ir = I_m'(X)/I_m(X).
Zeros x_j <-> lambda_j / k^2 = 1 + (x_j/X)^2.  Steps:
(1) (0, 2]: for m = 1, with q(y) = J_1(x)/x = sum_n (-1)^n y^n / (2 4^n n! (n+1)!), y = x^2, one has EXACTLY
    Delta = y^2 f(y),  f = (1 - X Ir) q^2 - 4 X^2 q q' - y (4 X^2 q'^2 + 2 X Ir q q'),
    (J_1 = x q, J_1' = q + 2 y q'); f > 0 on y in [0, 4] is certified by ball evaluation on 200 subintervals with a rigorous
    series tail => Delta > 0 on (0, 2]: the only zero of Delta with x <= 2 is the spurious x = 0 (order 4 for m = 1).
    (The min-max bound x > j_{0,1} of stokes_mode.py is therefore not needed.)
(2) Scan of [2, XR], XR = 6035 > x_max = 6034.8, step 0.02, tight ball values with certified signs (an undecidable sign, which
    never occurs at prec 128 unless a zero sits within ~1e-35 of a grid point, is handled by replacing the point by x -+ 0.005);
    each sign change is bisected (exact dyadic midpoints, certified signs) to width <= 1e-10: at least one zero per bracket.
(3) Completeness (argument principle): Delta is entire and real on the real axis; the winding number of Delta around the
    rectangle R = (2, XR) x (-h, h), h = 2, equals the number of zeros in R with multiplicity.  By Delta(conj z) = conj Delta(z)
    the winding number is (1/pi) x (arg change along the upper half: (XR,0) -> (XR,h) -> (2,h) -> (2,0)).  The path is cut
    into segments; on each, Delta is enclosed via a TAYLOR polynomial of J_m about the segment centre c: coefficients
    J_m^(n)(c)/n! = 2^-n sum_j (-1)^j C(n,j) J_{m-n+2j}(c)/n! (J_nu(c) from Arb for nu = 0, 1 and the upward recurrence,
    J_{-nu} = (-1)^nu J_nu), remainder |J_m^(NT)| <= e^{|Im z|} (integer order) => |R| <= e^{|Im|} rho^NT / NT!, and the
    polynomial expression of Delta evaluated at the interval ball; the enclosure is a rectangle (convex): if it excludes 0 the
    segment's image lies in an open half-plane and its arg change is the principal arg of the endpoint ratio; we further
    require the ratio ball to lie in the right half-plane (|increment| < pi/2) and subdivide otherwise.  (Arb's own wide-ball
    Bessel evaluation is far too loose for |z| < ~300, hence the Taylor enclosure.)  Count in R == number of brackets =>
    every bracket holds exactly one simple zero, no real zero is missed in (2, XR), and R has no non-real zero.
Output: stokes_zeros_rigorous.npz (lower, upper, lam_lower, lam_upper: outward-rounded float64), stokes_rigorous.log.
    python stokes_rigorous.py"""
import time, numpy as np
from math import comb, factorial as fac, nextafter, inf
from flint import arb, acb, ctx, acb_poly
ctx.prec = 128
t0 = time.time()
import sys
Lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1.2e4
m, k, rmax, Ncol, h = 1, 2000, '0.027546', 100, 2.0
tag = f'_L{int(Lam)}'
X = arb(k) * arb(rmax) * (arb.pi() / (4 * Ncol)).cos()          # ball of radius ~1e-38 around k rmax cos(pi/400)
Ir = X.bessel_i(m - 1) / X.bessel_i(m) - arb(m) / X                # I_m'(X)/I_m(X) via I_m' = I_{m-1} - m I_m / X
X2, XIr = X * X, X * Ir
x_max = float((X * arb(Lam - 1).sqrt()).upper()); XR = 2.0 + 0.25 * (int((x_max - 2.0) / 0.25) + 1)   # XR > x_max, XR - 2 a multiple of 2 rho
log = open(f'stokes_rigorous{tag}.log', 'w')
def out(s): print(s); log.write(s + '\n'); log.flush()
out(f"m = {m}, k = {k}, r_w = {rmax} cos(pi/{4*Ncol}); X = {X}; Ir = {Ir}; prec = {ctx.prec} bits")
def Delta(x):                                                       # x: arb or acb (tight balls)
    J = x.bessel_j(m); Jp = x.bessel_j(m - 1) - m * J / x
    return m * m * (X2 + x * x) * J * J - X2 * x * x * Jp * Jp - XIr * x ** 3 * J * Jp
def sgn(v): return 1 if v > 0 else (-1 if v < 0 else 0)            # certified sign (0 = undecidable)

# ---------------- (1) no zero in (0, 2] ----------------
assert m == 1
Nq = 30; cq = [arb((-1) ** n) / (2 * 4 ** n * fac(n) * fac(n + 1)) for n in range(Nq)]
tail_q = 2 * arb(1) / (fac(Nq) * fac(Nq + 1)); tail_qp = 2 * arb(1) / (4 * fac(Nq - 1) * fac(Nq + 1))   # y <= 4: term ratios < 1/2 (bounds hold a fortiori with the 1/2)
def qqp(y):
    q = arb(0); qp = arb(0)
    for n in range(Nq - 1, -1, -1): q = q * y + cq[n]
    for n in range(Nq - 1, 0, -1): qp = qp * y + n * cq[n]
    return q + arb(0, tail_q), qp + arb(0, tail_qp)
def f_small(y):
    q, qp = qqp(y); return (1 - XIr) * q * q - 4 * X2 * q * qp - y * (4 * X2 * qp * qp + 2 * XIr * q * qp)
for xt in (0.3, 1.1, 1.9):                                          # identity check Delta = x^4 f(x^2) (necessary condition)
    xa = arb(xt); d = Delta(xa) - xa ** 4 * f_small(xa * xa); assert d.contains(0) and d.rad() < 1e-30, d
fmin = min(float(f_small((arb(2 * i + 1) + arb(0, 1)) / 100).lower()) for i in range(200))   # y in [i/50, (i+1)/50] covers [0, 4]
assert fmin > 0, fmin
out(f"(1) Delta = x^4 f(x^2) with f > {fmin:.4g} certified on x in (0, 2] (200 subintervals): no zero in (0, 2]")

# ---------------- (2) scan and bisection ----------------
step = 0.02; ngrid = int(round((XR - 2.0) / step)); pts = []
for i in range(ngrid + 1):
    x = 2.0 + i * step; s = sgn(Delta(arb(x)))
    if s == 0:
        for xx in (x - step / 4, x + step / 4):
            s2 = sgn(Delta(arb(xx))); assert s2 != 0; pts.append((xx, s2))
    else: pts.append((x, s))
brackets = [(pts[i][0], pts[i + 1][0]) for i in range(len(pts) - 1) if pts[i][1] * pts[i + 1][1] < 0]
out(f"(2) scan of [2, {XR}] step {step}: {len(pts)} points, {len(brackets)} certified sign changes (t = {time.time()-t0:.0f} s)")
def bisect(a, b, tol=1e-10):
    a, b = arb(a), arb(b); sa = sgn(Delta(a)); assert sa != 0 and sgn(Delta(b)) == -sa
    while not (b - a < tol):
        for fr in (0.5, 0.375, 0.625, 0.4375, 0.5625):
            c = a + (b - a) * fr; sc = sgn(Delta(c))
            if sc != 0: break
        assert sc != 0
        if sc == sa: a = c
        else: b = c
    return a, b
enc = [bisect(a, b) for a, b in brackets]
out(f"    bisection to width <= 1e-10 done (t = {time.time()-t0:.0f} s)")

# ---------------- (3) argument principle on the rectangle (2, XR) x (-h, h) ----------------
NT = 18
def acc(r, vert): return acb(0, r) if vert else acb(r)
def seg_enclosure(c, rho, vert):
    # Taylor polynomial of J_m about c (NT terms) + rigorous remainder; returns Delta on {c + s t, |t| <= rho} and at t = -rho, +rho
    Jn = [c.bessel_j(0), c.bessel_j(1)]
    for nu in range(1, NT + m): Jn.append(2 * nu / c * Jn[nu] - Jn[nu - 1])
    Jat = lambda nu: Jn[nu] if nu >= 0 else (-1) ** (-nu) * Jn[-nu]
    a = [sum((-1) ** j * comb(n, j) * Jat(m - n + 2 * j) for j in range(n + 1)) / (2 ** n * fac(n)) for n in range(NT)]
    E = (abs(c.imag) + (rho if vert else 0)).exp()                  # sup e^{|Im z|} on the segment
    RJ, RJp = E * arb(rho) ** NT / fac(NT), E * arb(rho) ** (NT - 1) / fac(NT - 1)
    PJ = acb_poly(a) + acb(arb(0, RJ), arb(0, RJ)); PJp = acb_poly([(n + 1) * a[n + 1] for n in range(NT - 1)]) + acb(arb(0, RJp), arb(0, RJp))
    Z = acb_poly([c, 1]); D = m * m * (acb(X2) + Z * Z) * PJ * PJ - acb(X2) * Z * Z * PJp * PJp - acb(XIr) * Z ** 3 * PJ * PJp
    tb = acb(0, arb(0, rho)) if vert else acb(arb(0, rho)); t1 = acc(rho, vert)
    return D(tb), D(-t1), D(t1)
def track(c, rho, vert, direction, depth=0):
    # certified arg change of Delta along the segment from c - direction*s*rho to c + direction*s*rho (s = i if vert else 1)
    W, w_lo, w_hi = seg_enclosure(c, rho, vert)
    if not W.contains(0):
        r = w_hi / w_lo if direction > 0 else w_lo / w_hi
        if r.real > 0: return r.arg()
    assert depth < 14, (c, rho)
    s = acc(rho / 2, vert)
    return track(c - s, rho / 2, vert, direction, depth + 1) + track(c + s, rho / 2, vert, direction, depth + 1)
rho = 0.125; nseg = int(round((XR - 2.0) / (2 * rho))); assert 2.0 + 2 * rho * nseg == XR; nv = int(round(h / (2 * rho)))
total = arb(0)
for j in range(nv): total += track(acb(XR, (2 * j + 1) * rho), rho, True, +1)          # right side, upward
for j in range(nseg): total += track(acb(XR - (2 * j + 1) * rho, h), rho, False, -1)  # top side, leftward
for j in range(nv): total += track(acb(2.0, h - (2 * j + 1) * rho), rho, True, -1)     # left side, downward
wind = total / arb.pi(); nz = int(round(float(wind.mid()))); assert (wind - nz).abs_upper() < 0.5, wind
out(f"(3) argument principle: winding number of Delta around (2, {XR}) x (-{h}, {h}) = {wind} => {nz} zeros (with multiplicity)"
    f" in the rectangle; brackets = {len(enc)} -> {'COMPLETE: every bracket has exactly one simple zero, none missed, none non-real' if nz == len(enc) else 'MISMATCH'}"
    f" (t = {time.time()-t0:.0f} s)")
assert nz == len(enc)

# ---------------- (4) output ----------------
def f_lo(a):
    f = float(a.lower())
    while not (arb(f) <= a): f = nextafter(f, -inf)
    return f
def f_hi(a):
    f = float(a.upper())
    while not (arb(f) >= a): f = nextafter(f, inf)
    return f
lo = np.array([f_lo(a) for a, b in enc]); hi = np.array([f_hi(b) for a, b in enc])
lam = [(1 + (a / X) ** 2, 1 + (b / X) ** 2) for a, b in enc]
llo = np.array([f_lo(u) for u, v in lam]); lhi = np.array([f_hi(v) for u, v in lam])
np.savez(f'stokes_zeros_rigorous{tag}.npz', lower=lo, upper=hi, lam_lower=llo, lam_upper=lhi, X=float(X.mid()), x_max=x_max, XR=XR)
xL = X * arb(Lam - 1).sqrt(); n_below = int(np.sum(hi <= x_max)); n_amb = int(np.sum((lo <= x_max) & (hi > x_max)))
out(f"(4) certified zeros in (2, {XR}): {len(enc)}; with upper end <= x_max = {x_max}: {n_below} (brackets straddling x_max: {n_amb});"
    f" with upper end <= X sqrt(Lambda-1) = {xL}: {int(np.sum(hi <= f_lo(xL)))}  [stokes_mode.py float count: 3840]")
out(f"    max bracket width {np.max(hi - lo):.3e} (x), {np.max(lhi - llo):.3e} (lambda/k^2); first ten zeros:")
for i in range(10): out(f"    [{lo[i]:.12f}, {hi[i]:.12f}]  lambda/k^2 in [{llo[i]:.14f}, {lhi[i]:.14f}]")
out(f"    wall time {time.time()-t0:.1f} s; saved stokes_zeros_rigorous{tag}.npz")
