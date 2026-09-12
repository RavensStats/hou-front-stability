"""Re-grid an AxiPhys snapshot onto a different number of axial points, on the SAME axial map, by the
exact sine-series evaluation axiphys.py uses on restart. Produces the twin states needed for a
co-evolving resolution pair (round-3 referee: no co-evolving run has one).

    python regrid_z.py <snapshot.npz> <n_zh_new> <out.npz>

n_zh_new is the solver's n_z convention (the stored z array has n_zh - 1 points).
"""
import sys
import numpy as np
from scipy.fft import dst
sys.path.insert(0, ".")
from axiphys import AxiPhys

src, nzh_new, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = np.load(src)
U, Om, Ps, r, z, t = d["U"], d["Om"], d["Ps"], d["r"], d["z"], float(d["t"])
nu = float(d["nu"]); dim = float(d["dim"]) if "dim" in d.files else 3.0
zm = tuple(float(v) for v in d["zmap"]) if "zmap" in d.files else (0.0, 0.0)
nzh_old = len(z) + 1
a = 5.0

P = AxiPhys(len(r), nzh_new, nu=nu, a=a, dim=dim)
if zm[0] > 0:
    P.set_zmap(zm)
P.r = r.copy()                                    # keep the radial grid exactly

# the old field is a sine series in the old map's zeta; evaluate it at the new nodes' zeta
if zm[0] > 0:
    Pold = AxiPhys(len(r), nzh_old, a=a)
    Pold.set_zmap(zm)
    zeta_new = np.interp(P.z, Pold._zfine_z, Pold._zfine_zeta)
else:
    zeta_new = P.z
S = np.sin(2.0 * np.pi * np.arange(1, nzh_old)[None, :] * zeta_new[:, None])
def regrid(F):
    return (dst(F, type=1, axis=1) / nzh_old) @ S.T

Un, Omn = regrid(U), regrid(Om)
Psn = regrid(Ps) if Ps is not None else None

# round trip on the original grid, as a check that the transfer is the identity there
S_back = np.sin(2.0 * np.pi * np.arange(1, nzh_old)[None, :] * (
    np.interp(z, Pold._zfine_z, Pold._zfine_zeta) if zm[0] > 0 else z)[:, None])
U_rt = (dst(U, type=1, axis=1) / nzh_old) @ S_back.T
err = float(np.abs(U_rt - U).max() / max(np.abs(U).max(), 1e-300))
print(f"round-trip relative error on the source grid: {err:.2e}")

print(f"  max|u1|  source {np.abs(U).max():.6e}   regridded {np.abs(Un).max():.6e}")
print(f"  max|om1| source {np.abs(Om).max():.6e}   regridded {np.abs(Omn).max():.6e}")
np.savez(out, U=Un, Om=Omn, Ps=Psn, r=P.r, z=P.z, t=t, nu=nu, dim=dim, zmap=np.array(zm))
print(f"wrote {out}: {Un.shape[0]} x {Un.shape[1]} (n_zh = {nzh_new}), same map {zm}, t = {t:.9f}")
