"""Table of states for paper 1, regenerated from the snapshots themselves (referee issue 35: one label per state).
For every snapshot on disk: t, T - t, the swirl amplification A/A0 and its location, the vorticity amplification,
the KNSS quantity sup |u| |x'| and where it is attained, the same quantity at the swirl maximum, and the meridional
quantities |r u^r|, |r u^z| at the core (within three swirl radii of the maximum) and globally.
    python papertable.py <T> [glob ...]      default T = 0.0022865, all axiphys/nsz snapshots
"""
import sys, glob
import numpy as np
from scipy.interpolate import CubicSpline
sys.path.insert(0, ".")
from axiphys import AxiPhys

T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0022865
pats = sys.argv[2:] or ["axiphys_*_nsz_t0.002*.npz", "axiphys_*_e513a5_t0.002*.npz", "nsz?_snap_t0.002*.npz",
                        "nsz??_snap_t0.002*.npz"]
files = sorted({f for p in pats for f in glob.glob(p)})

# Hou's data u1(0) = 12000 (1-r^2)^18 sin(2 pi z) / (1 + 12.5 sin^2(pi z)), omega1(0) = 0.
# Three labels are in use and the table carries all three (they are NOT proportional):
#   A/A0   : max|u1| / max|u1|(0), the true amplification of the swirl maximum (max|u1|(0) = 3265.99)
#   A/1.2e4: max|u1| / 12000, the coefficient of the datum -- the label used in the paper's prose
#   |om|/|om|(0): max of the full vorticity vector / its initial max, which is Hou's label
def u1_0(r, z):
    return 12000.0 * (1 - r ** 2) ** 18 * np.sin(2 * np.pi * z) / (1 + 12.5 * np.sin(np.pi * z) ** 2)
zz = np.linspace(0, 1, 2000001)
A0 = float(np.max(u1_0(0.0, zz)))
def omega_max_initial(r, z):
    """max |omega| at t = 0 on the given grid: omega = (-r d_z u1, r om1, 2 u1 + r d_r u1), om1(0) = 0"""
    U = u1_0(r[:, None], z[None, :])
    dz_U = np.gradient(U, z, axis=1, edge_order=2)
    dr_U = np.gradient(U, r, axis=0, edge_order=2)
    return float(np.sqrt((r[:, None] * dz_U) ** 2 + (2 * U + r[:, None] * dr_U) ** 2).max())

_cache = {}
print(f"# T = {T:.7f}, initial swirl maximum A0 = {A0:.6f}")
print("| file | t | T - t | A/A0 | A/1.2e4 | (R, Z) | om/om0 | sup|u||x'| | at (r,z) | |u|r core | |r u^r| core | |r u^z| core | "
      "|r u^r| glob | |r u^z| glob | sup|u| | sup|u_mer| |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
om0 = None
for f in files:
    try:
        d = np.load(f)
        U, Om, r, z, t = d["U"], d["Om"], d["r"], d["z"], float(d["t"])
        Ps = d["Ps"] if "Ps" in d.files else None
    except Exception as e:
        print(f"| {f} | LOAD FAILED: {e} |"); continue
    if Ps is None:
        print(f"| {f} | {t:.9f} | no streamfunction stored |"); continue
    rr = np.concatenate([-r[:0:-1], r])
    Psr = CubicSpline(rr, np.concatenate([Ps[:0:-1], Ps]), axis=0)(r, 1)
    zm = tuple(float(v) for v in d["zmap"]) if "zmap" in d.files else None
    key = (len(r), len(z), zm)                          # AxiPhys construction is expensive: one per grid/map
    if key not in _cache:
        Pn = AxiPhys(len(r), len(z) + 1, a=5.0)
        if zm is not None and zm[0] > 0:
            Pn.set_zmap(zm)
        _cache[key] = Pn
    P = _cache[key]
    assert np.abs(P.z - z).max() < 1e-10, (f, "z grid differs from AxiPhys with zmap", zm)
    Psz = P.d_z(Ps)
    ur = -r[:, None] * Psz
    uz = 2 * Ps + r[:, None] * Psr
    ut = r[:, None] * U
    umag = np.sqrt(ur ** 2 + ut ** 2 + uz ** 2)
    umer = np.sqrt(ur ** 2 + uz ** 2)                   # meridional speed, for the Type I constant
    rgrid = np.repeat(r[:, None], len(z), axis=1)
    q = umag * rgrid                                   # |u| |x'|, the KNSS quantity
    i, j = np.unravel_index(np.abs(U).argmax(), U.shape)
    A = abs(float(U[i, j])); R, Z = float(r[i]), float(z[j])
    iq, jq = np.unravel_index(q.argmax(), q.shape)
    zz_m = np.minimum(z, 1.0 - z)[None, :]              # mirror-periodic distance to the symmetry plane
    core = (rgrid < 3 * R) & (zz_m < 3 * min(Z, 1.0 - Z))   # the core box of eq (lambda): r < 3R, |z| < 3Z
    Ur = -rgrid * Psz                                   # full vorticity vector, as the solver defines it
    dzU = P.d_z(U); rs2 = CubicSpline(rr, np.concatenate([U[:0:-1], U]), axis=0)(r, 1)
    om = float(np.sqrt((rgrid * dzU) ** 2 + (rgrid * Om) ** 2 + (2 * U + rgrid * rs2) ** 2).max())
    om0 = omega_max_initial(r, z)
    print(f"| {f} | {t:.9f} | {T - t:.3e} | {A / A0:.1f} | {A / 12000.0:.1f} | ({R:.5f}, {Z:.5f}) | {om / om0:.1f} | {float(q.max()):.2f} | "
          f"({float(r[iq]):.4f}, {float(z[jq]):.4f}) | {float(q[i, j]):.2f} | "
          f"{float((rgrid * np.abs(ur))[core].max()):.2f} | {float((rgrid * np.abs(uz))[core].max()):.2f} | "
          f"{float((rgrid * np.abs(ur)).max()):.2f} | {float((rgrid * np.abs(uz)).max()):.2f} | "
          f"{float(umag.max()):.6e} | {float(umer.max()):.6e} |")
