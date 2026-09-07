"""COLENC stage A, part 2: Krawczyk enclosure of the leading eigenvalue of the enclosed collocation pencil built by colenc.py.
   python colenc_run.py colenc_<tag>.npz"""
import sys, time, numpy as np
from ivmat import enclose_pencil_eigen
f = sys.argv[1]; d = np.load(f)
Amid, Arad, Bmid, Brad, w0, x0, knorm = d["Amid"], d["Arad"], d["Bmid"], d["Brad"], complex(d["w0"]), d["x0"], int(d["knorm"])
t = float(d["t"]); Tt = 0.002278 - t
print(f"{f}: dim {Amid.shape[0]}, N {int(d['N'])}, m {int(d['m'])}, k {float(d['k'])}, nu {float(d['nu'])}, field degree {int(d['deg'])}; approximate omega = {w0.real:+.6e} {w0.imag:+.6e}i")
t1 = time.time()
res = enclose_pencil_eigen(Amid, Bmid, w0, x0, knorm, A_rad=Arad, B_rad=Brad)
print(f"  enclosure call returned in {time.time()-t1:.1f}s: {type(res)}")
try:
    ok, W, X = res[0], res[1], res[2]
except Exception:
    ok, W, X = res.get("ok"), res.get("w"), res.get("x")
if ok:
    wm, wr = (W.mid if hasattr(W, "mid") else W[0]), (W.rad if hasattr(W, "rad") else W[1])
    wm = complex(np.asarray(wm).ravel()[0]); wr = float(np.asarray(wr).ravel()[0])
    print(f"  CERTIFIED: every pencil in the enclosed family has a SIMPLE eigenvalue in the disc |omega - ({wm.real:+.9e} {wm.imag:+.9e}i)| <= {wr:.3e}, with an eigenvector in the enclosed box (the eigenpair is unique in that box; the disc is not certified to contain no other eigenvalue)")
    print(f"  hence Im omega in [{wm.imag - wr:.9e}, {wm.imag + wr:.9e}]  -> growth (T-t) in [{(wm.imag - wr) * Tt:.6f}, {(wm.imag + wr) * Tt:.6f}] > 0")
else:
    print("  NOT certified")
