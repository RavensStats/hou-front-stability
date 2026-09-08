"""Arbiter comparison of the late-stage transition at nu = 5e-4 across the three maps (nsz7: 2048 front-following, floor 1.5e-4;
nsz8: 1536 front-following; nsz9: 2048 wide fixed fine region, ZWMIN 5e-4, ZFRAC 0.3).  Per snapshot: map width and contrast,
max |u1| and its position, A(T-t) with T = 0.0022866, max |omega1| and position, the inner-region (z < 0.0005) maximum of |u1|,
max |psi1|.   python cmp789.py"""
import glob, numpy as np
T = 0.0022866
print("file                          t          zwid     contr   |u1|max     (r, z)            x      A(T-t)  |om1|max   (r, z)            inner|u1|  (r,z)          |psi1|max")
for tag in ("nsz7", "nsz8", "nsz9"):
    for f in sorted(glob.glob(f"{tag}_snap_t0.00228[56]*.npz")):
        d = np.load(f); U, Om, Ps, r, z, t = d["U"], d["Om"], d["Ps"], d["r"], d["z"], float(d["t"])
        if t < 0.0022857: continue
        dz = np.diff(z); zwid = float(np.sum(dz[dz < 2 * dz.min()])); contr = float(dz.max() / dz.min())
        i, j = np.unravel_index(int(np.argmax(np.abs(U))), U.shape); um = float(np.abs(U).max())
        io, jo = np.unravel_index(int(np.argmax(np.abs(Om))), Om.shape)
        inner = np.abs(U)[:, z < 0.0005]; ii, ji = np.unravel_index(int(np.argmax(inner)), inner.shape)
        print(f"{f:28s} {t:.7f} {zwid:.2e} {contr:7.0f} {um:.4e} ({r[i]:.5f},{z[j]:.5f}) {um/12000:6.0f} {um*(T-t):6.2f}  {np.abs(Om).max():.3e} ({r[io]:.5f},{z[jo]:.5f}) {inner.max():.3e} ({r[ii]:.5f},{z[z<0.0005][ji]:.5f}) {np.abs(Ps).max():.3e}")
