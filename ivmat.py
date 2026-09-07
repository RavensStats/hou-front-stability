"""
ivmat.py -- rigorous complex midpoint-radius interval matrices (numpy only) and a
Krawczyk enclosure of one eigenpair of the generalized problem  A x = w B x,
B possibly singular (zero rows allowed).

Floating-point model (IEEE-754 binary64, round to nearest, NO overflow assumed;
non-finite results make every certificate fail):
    fl(a op b) = (a op b)(1 + d) + e,   |d| <= u = 2^-53,   |e| <= 2^-1075,
op in {+,-,*,/};  e = 0 for +,- (gradual underflow) and for non-underflowing
products.  gamma_k := k u/(1 - k u)  (Higham, ASNA 2nd ed. 2002, Lemma 3.1).
Standard consequence (Higham eq. (3.4)): a product of at most k factors (1+d_i)
equals 1 + theta, |theta| <= gamma_k, for ANY order of evaluation.

Every radius computed here is an UPPER bound of the exact radius formula: the
floating-point evaluation of the formula is followed by `outward` (a rigorous
multiplicative correction for the accumulated relative rounding error) plus an
absolute floor _EPS_ABS = 2^-1000 per operation (>> the 2^-1075 per-product
underflow error), see `outward` for the proof.

BLAS assumption (the only external one): a real float64 matrix product
fl(P Q) (numpy -> dgemm/dgemv or numpy's own loops) computes each entry as a
classical inner product in some order, with or without FMA; then
    |fl(P Q) - P Q| <= gamma_n |P| |Q|    (n = inner dimension; Higham (3.13),
    valid for any summation order; FMA only reduces the rounding count).
Strassen-type / 3M complex BLAS are NOT covered.  Complex products are built
here from four REAL products (never the complex zgemm) so only this real bound
is relied upon.  Elementwise complex multiplication is also done in real ops.
"""
import numpy as np

U = 2.0 ** -53            # unit roundoff
_EPS_ABS = 2.0 ** -1000   # absolute floor per radius operation (underflow guard)


def outward(x, k):
    """Return fl(x*f) >= t for every t >= 0 with  t <= x (1+2u)^k,  x >= 0 float64.
    This covers t <= x(1+u)^k, t <= x/(1-u)^k and t <= x/(1-gamma_k) (k u <= 1/4).
    f = 1 + (k+1) 2^-52 is exactly representable.  Proof: fl(x f) >= x f (1-u)
    = x(1 + (2k+1)u - 2(k+1)u^2) >= x(1 + 2ku + 4k^2u^2) >= x(1+2u)^k, because
    (1+2u)^k <= exp(2ku) <= 1 + 2ku + (2ku)^2 for 2ku <= 1, and
    (4k^2+2k+2) u <= 1 for k <= 4e7.  Sub-normal x is covered by _EPS_ABS floors."""
    if not 0 <= k <= 40_000_000:
        raise ValueError("outward: k out of the proven range")
    return np.multiply(x, 1.0 + (k + 1) * 2.0 ** -52, dtype=np.float64)


def gamma(k):
    """Rigorous upper bound of gamma_k = k u/(1-k u), 0 <= k <= 2^40.  k*u is exact;
    fl(1-ku) <= (1-ku)(1+u), so q = fl(ku/fl(1-ku)) >= gamma_k (1-u)/(1+u), i.e.
    gamma_k <= q (1+u)/(1-u) <= q (1+2u)^2  -> outward(q, 2)."""
    k = int(k)
    if not 0 <= k <= 2 ** 40:
        raise ValueError("gamma: k out of range")
    return float(outward((k * U) / (1.0 - k * U), 2))


_G1, _G2 = gamma(1), gamma(2)


def abs_ub(z):
    """Upper bound of |z|:  |z| <= |Re z| + |Im z| <= fl(|Re z| + |Im z|) (1+u).
    Overestimates by <= sqrt(2).  Deliberately avoids np.abs/hypot (platform-dependent
    accuracy, and squares may underflow)."""
    z = np.asarray(z)
    if np.iscomplexobj(z):
        return outward(np.abs(z.real) + np.abs(z.imag), 1)
    return np.abs(z).astype(np.float64)


def _cmul(sm, am):
    """Complex product in real ops: Re = fl(fl(sr ar) - fl(si ai)), Im = fl(fl(sr ai) + fl(si ar)).
    Each monomial carries <= 2 roundings, so |Re err| <= gamma_2(|sr ar|+|si ai|),
    |Im err| <= gamma_2(|sr ai|+|si ar|), hence |fl(s a) - s a| <= sqrt(2) gamma_2 |s||a|
    (Higham Lemma 3.5, using (|sr ar|+|si ai|)^2+(|sr ai|+|si ar|)^2 <= 2|s|^2|a|^2)."""
    sr, si, ar, ai = sm.real, sm.imag, am.real, am.imag
    out = np.empty(np.broadcast(sm, am).shape, dtype=np.complex128)
    out.real = sr * ar - si * ai
    out.imag = sr * ai + si * ar
    return out


class IvMat:
    """Disc interval matrix { Z : |Z_ij - mid_ij| <= rad_ij }, mid complex128, rad float64 >= 0.
    Shapes: () scalar, (n,) vector, (n,m) matrix."""
    __slots__ = ("mid", "rad")

    def __init__(self, mid, rad=None):
        self.mid = np.array(mid, dtype=np.complex128)
        if rad is None:
            self.rad = np.zeros(self.mid.shape)
        else:
            self.rad = np.array(np.broadcast_to(np.asarray(rad, dtype=np.float64), self.mid.shape))
        if np.isnan(self.rad).any():                 # NaN (e.g. inf*0 after overflow) -> "unknown" -> inf
            self.rad = np.where(np.isnan(self.rad), np.inf, self.rad)
        if not bool(np.all(self.rad >= 0)):
            raise ValueError("IvMat: radii must be nonnegative")

    shape = property(lambda self: self.mid.shape)
    ndim = property(lambda self: self.mid.ndim)

    def __getitem__(self, idx):
        return IvMat(self.mid[idx], self.rad[idx])

    def __neg__(self):                       # exact
        return IvMat(-self.mid, self.rad)

    def upper(self):
        """Elementwise upper bound of |z| over the disc: |mid| + rad, one rounding."""
        return outward(abs_ub(self.mid) + self.rad, 1) + _EPS_ABS

    def __add__(self, other):
        return _addsub(self, other, +1)

    def __sub__(self, other):
        return _addsub(self, other, -1)

    def __matmul__(self, other):
        return mid_rad_matmul(self, other)

    def scale(self, s):
        """Interval scalar s (IvMat of shape () or a number) times self.
        Exact:   |s a - sm am| <= |sm| ra + sr |am| + sr ra.
        Rounding of mid = _cmul(sm, am): <= sqrt(2) gamma_2 |sm||am| <= 2 gamma_2 |sm|_ub |am|_ub
        (2*gamma_2 is an exact doubling).  Radius chain: <= 5 roundings per monomial
        -> outward(., 8) + floor."""
        s = s if isinstance(s, IvMat) else IvMat(np.asarray(s, dtype=np.complex128))
        a_s, a_a = abs_ub(s.mid), abs_ub(self.mid)
        rad = a_s * self.rad + s.rad * a_a + s.rad * self.rad + ((2.0 * _G2) * a_s) * a_a
        return IvMat(_cmul(s.mid, self.mid), outward(rad, 8) + _EPS_ABS)

    def contains(self, other, safety=8):
        """Strict interior containment  other inside int(self), elementwise:
            |mid_o - mid_s| + rad_o < rad_s .
        With d = fl(mid_o - mid_s):  |mid_o - mid_s| <= (|Re d| + |Im d|)/(1-u); the left side
        is evaluated with 3 roundings -> outward(., 3 + safety) + floor, compared strictly.
        Any non-finite quantity -> False."""
        if self.shape != other.shape:
            return False
        d = other.mid - self.mid
        lhs = outward(np.abs(d.real) + np.abs(d.imag) + other.rad, 3 + int(safety)) + _EPS_ABS
        return bool(np.all(np.isfinite(lhs)) and np.all(np.isfinite(self.rad))
                    and np.all(np.isfinite(other.mid)) and np.all(lhs < self.rad))


def _addsub(a, b, sign):
    """a +/- b.  mid = fl(ma +/- mb): each real part rounded once, |Re z| <= |fl(Re z)|/(1-u),
    so |fl(z) - z| <= gamma_1 |fl(z)| <= gamma_1 abs_ub(mid).  rad = ra + rb + gamma_1|mid|,
    3 roundings -> outward(., 3) + floor."""
    mid = a.mid + b.mid if sign > 0 else a.mid - b.mid
    return IvMat(mid, outward(a.rad + b.rad + _G1 * abs_ub(mid), 3) + _EPS_ABS)


def mid_rad_matmul(A, B):
    """Rigorous enclosure of {A' B' : A' in A, B' in B} for IvMat A (.., n) and B (n, ..).
    Exact part:  |A'B' - Am Bm| <= Ar|Bm| + |Am|Br + Ar Br   (elementwise, nonneg products).
    Midpoint:  mid = fl(Am Bm) via 4 real products P1=fl(ArBr) ... (Ar := Re Am etc.):
        Re mid = fl(P1 - P2), Im mid = fl(P3 + P4).  Under the BLAS assumption every
        monomial carries <= n+1 roundings, so |Re err| <= gamma_{n+1} sum_k(|a_r b_r|+|a_i b_i|)
        <= gamma_{n+1} (|Am||Bm|) (Cauchy-Schwarz on (|a_r|,|a_i|).(|b_r|,|b_i|)), same for Im,
        hence |mid - Am Bm| <= sqrt(2) gamma_{n+1} |Am||Bm| <= gamma_{2n+4} |Am||Bm|.
    Radius:  T = fl(|Am|_ub |Bm|_ub) >= |Am||Bm| (1-u)^n (nonnegative inner products), and the
        other three products likewise; each monomial of the radius formula carries <= n+4
        roundings -> outward(., n+8) + (n+8) floor."""
    if not isinstance(A, IvMat):
        A = IvMat(A)
    if not isinstance(B, IvMat):
        B = IvMat(B)
    n = A.mid.shape[-1]
    if B.mid.shape[0] != n:
        raise ValueError("mid_rad_matmul: shape mismatch")
    Ar, Ai = np.ascontiguousarray(A.mid.real), np.ascontiguousarray(A.mid.imag)
    Br, Bi = np.ascontiguousarray(B.mid.real), np.ascontiguousarray(B.mid.imag)
    P1, P2, P3, P4 = Ar @ Br, Ai @ Bi, Ar @ Bi, Ai @ Br
    mid = np.empty(P1.shape, dtype=np.complex128)
    mid.real = P1 - P2
    mid.imag = P3 + P4
    absA, absB = abs_ub(A.mid), abs_ub(B.mid)
    rad = gamma(2 * n + 4) * (absA @ absB)
    ra_nz, rb_nz = bool(A.rad.any()), bool(B.rad.any())
    if ra_nz:
        rad = rad + A.rad @ absB
    if rb_nz:
        rad = rad + absA @ B.rad
    if ra_nz and rb_nz:
        rad = rad + A.rad @ B.rad
    return IvMat(mid, outward(rad, n + 8) + (n + 8) * _EPS_ABS)


def krawczyk_eigpair(A, B, w0, x0, k_norm_index, expand=1e-3, iters=10, inflate0=10.0):
    """Rigorous enclosure of one eigenpair of  A x = w B x  near (w0, x0), normalization
    x[k] = 1 fixed.  A, B: IvMat (n x n) (or arrays -> point matrices).  Returns
    (ok, W, X): ok True iff certified; W IvMat shape () enclosing w, X IvMat shape (n,)
    enclosing x (X[k] = 1 exactly).  If ok, then for EVERY (A', B') in (A, B) there is
    exactly one (w, x) with A' x = w B' x, x_k = 1, w in W, x in X (all components).

    Unknown y = (x without component k, w) in C^n,  F(y) = A x - w B x,
    J(y) = [ (A - w B) without column k | -B x ].  R = fl-inverse of J(y0).
    Krawczyk operator on the disc box Y = y0 + rho*discs:
        K(Y) = y0 - R F(y0) + (I - R M)(Y - y0),   M = J(Y) = [A - W B | -B X] (interval),
    every term evaluated with the rigorous interval arithmetic above (F(y0) and R F(y0)
    are intervals).  Certificate: K(Y) strictly inside Y (IvMat.contains).
    Theorem (Krawczyk 1969 / Moore 1977 / Rump; see S.M. Rump, "Verification methods:
    rigorous results using floating-point arithmetic", Acta Numerica 19 (2010), Sec. 7
    (Krawczyk operator, existence/uniqueness theorem) and Sec. 13 (eigenpairs, this
    normalization); A. Neumaier, "Interval Methods for Systems of Equations", CUP 1990,
    Ch. 5; Rump, LAA 324 (2001)): if K(Y) is in the interior of Y for an interval matrix
    M containing {J(y): y in Y}, then R and every element of M are nonsingular and F has
    exactly one zero in Y, which lies in K(Y).  Proof for C^n with convex disc boxes:
    F(y)-F(y0) = (int_0^1 J(y0+t(y-y0)) dt)(y-y0) with the integral in the convex set M,
    so y -> y - R F(y) maps the compact convex Y into K(Y) inside int Y (Brouwer);
    nonsingularity from the symmetry Y - y0 = -(Y - y0).
    The returned enclosure is K(Y) (contains every zero of F in Y).  Failure -> (False, None, None).
    """
    A = A if isinstance(A, IvMat) else IvMat(A)
    B = B if isinstance(B, IvMat) else IvMat(B)
    n, k = A.shape[0], int(k_norm_index)
    x0 = np.asarray(x0, dtype=np.complex128)
    x0 = x0 / x0[k]
    x0[k] = 1.0
    w0 = complex(w0)
    idx = np.array([j for j in range(n) if j != k])
    Jm = np.empty((n, n), dtype=np.complex128)                  # floating-point Jacobian
    Jm[:, :n - 1] = (A.mid - w0 * B.mid)[:, idx]
    Jm[:, n - 1] = -(B.mid @ x0)
    try:
        R = np.linalg.inv(Jm)
    except np.linalg.LinAlgError:
        return False, None, None
    if not np.all(np.isfinite(R)):
        return False, None, None
    Riv, Iiv, X0, W0 = IvMat(R), IvMat(np.eye(n)), IvMat(x0), IvMat(w0)
    F0 = (A @ X0) - (B @ X0).scale(W0)                          # rigorous F(y0)
    RF = Riv @ F0                                               # rigorous R F(y0)
    y0 = np.append(x0[idx], w0)
    center = IvMat(y0) - RF                                     # y0 - R F(y0)
    floor = 1e-14 * (1.0 + float(np.max(np.abs(y0))))           # candidate-box floor (heuristic only)
    rho = np.full(n, inflate0 * float(np.max(RF.upper())) + floor)
    with np.errstate(over="ignore", invalid="ignore"):          # inf/NaN only ever make contains() False
        return _krawczyk_loop(A, B, Riv, Iiv, center, y0, x0, w0, idx, rho, n, expand, iters, floor)


def _krawczyk_loop(A, B, Riv, Iiv, center, y0, x0, w0, idx, rho, n, expand, iters, floor):
    for _ in range(int(iters)):
        if not np.all(np.isfinite(rho)):
            break
        xr = np.zeros(n)
        xr[idx] = rho[:n - 1]
        Xiv, Wiv = IvMat(x0, xr), IvMat(w0, rho[n - 1])
        AWB = A - B.scale(Wiv)                                  # A - W B
        BX = B @ Xiv                                            # B X
        M = IvMat(np.concatenate([AWB.mid[:, idx], -BX.mid[:, None]], axis=1),
                  np.concatenate([AWB.rad[:, idx], BX.rad[:, None]], axis=1))
        ImRM = Iiv - (Riv @ M)
        K = center + (ImRM @ IvMat(np.zeros(n), rho))
        Y = IvMat(y0, rho)
        if Y.contains(K):
            xm, xr = x0.copy(), np.zeros(n)
            xm[idx], xr[idx] = K.mid[:n - 1], K.rad[:n - 1]
            return True, K[n - 1], IvMat(xm, xr)
        ext = abs_ub(K.mid - y0) + K.rad                        # epsilon-inflation
        rho = np.maximum(ext, rho) * (1.0 + expand) + floor
    return False, None, None


def enclose_pencil_eigen(A_mid, B_mid, w0, x0, k_norm_index, A_rad=None, B_rad=None, **kw):
    """Convenience wrapper on numpy arrays: A = <A_mid, A_rad>, B = <B_mid, B_rad>
    (rad None -> exact point matrices).  Returns krawczyk_eigpair(...) = (ok, W, X)."""
    return krawczyk_eigpair(IvMat(A_mid, A_rad), IvMat(B_mid, B_rad), w0, x0, k_norm_index, **kw)
