"""Hou's vacuum-region diagnostic (H23b Sec. 3.2.4, Case 4) across the late-stage transition on the wide-map run nsz9:
|u1|(f R, Z) / |u1|(R, Z) along the row of the maximum for fractions f of the maximum's radius R, inside (f < 1) and outside
(f > 1).  Before the transition the swirl ramps gradually inside the front (15-40 percent at 0.7-0.9 R); after it the inside
empties and the outside drops too, the structure becoming a thin ring in r.   python vacuum_profile.py [glob]"""
import sys, glob, numpy as np
fr = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.2]
pat = sys.argv[1] if len(sys.argv) > 1 else "nsz9_snap_t0.00228[56]*.npz"
print("file                          x    R       Z      |u1|(f R, Z)/max for f =", fr)
for f in sorted(glob.glob(pat)):
    d = np.load(f); U, r, z, t = d["U"], d["r"], d["z"], float(d["t"])
    if t < 0.0022852: continue
    A = np.abs(U); i, j = np.unravel_index(int(np.argmax(A)), A.shape); R = r[i]
    vals = [A[np.argmin(abs(r - g * R)), j] / A[i, j] for g in fr]
    print(f"{f[:28]:28s} {A[i,j]/12000:4.0f} {R:.5f} {z[j]:.5f} " + " ".join(f"{v:.3f}" for v in vals))
