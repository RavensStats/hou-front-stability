"""log-derivative fit on snapshot amplitudes (9-decimal times): R = A/A' = (T-t)/p linear in t"""
import glob, numpy as np
pts = []
for f in sorted(glob.glob("axiphys_513_512_nsz_t0.002*.npz")) + sorted(glob.glob("nsz2_snap_t*.npz")) + sorted(glob.glob("nsz4_snap_t*.npz")) + sorted(glob.glob("nsz5_snap_t*.npz")) + sorted(glob.glob("nsz6_snap_t*.npz")):
    d = np.load(f); t = float(d["t"])
    if "nsz5" not in f and "nsz6" not in f and t > 0.0022851: continue
    if "nsz6" in f and t < 0.0022854: continue   # nsz2/nsz4 lose the front past 0.002285
    if "nsz4" in f and (t < 0.0022842 or t > 0.0022851): continue
    if "nsz5" in f and t < 0.0022851: continue      # duplicate of nsz2's 0.002284 point
    pts.append((t, float(np.abs(d["U"]).max())))
pts = sorted(pts); t = np.array([p[0] for p in pts]); A = np.array([p[1] for p in pts])
R = 1.0 / np.gradient(np.log(A), t)
for k in range(len(t)): print(f"  t {t[k]:.9f}  A {A[k]:.4e}  R {R[k]:.3e}")
for w in (4, 5, 6):
    for s in range(max(0, len(t) - w - 4), len(t) - w + 1):
        tt, RR = t[s:s+w], R[s:s+w]; c = np.polyfit(tt, RR, 1); print(f"window {w} [{tt[0]:.7f}, {tt[-1]:.7f}]: p {-1.0/c[0]:.3f}  T {-c[1]/c[0]:.7f}")
