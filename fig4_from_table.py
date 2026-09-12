"""Figure 4 (the constants) regenerated from STATES_TABLE.md, which papertable.py computes from the archived
snapshots, instead of from values transcribed into the plotting script.  Writes figures/fig4_constants.{pdf,png}
and prints the series the paper quotes in Section 5.3.
    python fig4_from_table.py [T]        default T = 0.0022865
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\mtgra\Desktop\Milennium Prize" + "\\"
T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0022865
A0 = 3265.986324                       # max |u_1|(0); see papertable.py

# max |omega|(0) for Hou's datum, on the same formula papertable.py uses
r = np.linspace(0, 1, 4001)
z = np.linspace(0, 1, 4001)
U0 = 12000.0 * (1 - r[:, None] ** 2) ** 18 * np.sin(2 * np.pi * z)[None, :] / (1 + 12.5 * np.sin(np.pi * z) ** 2)[None, :]
dz = np.gradient(U0, z, axis=1, edge_order=2)
dr = np.gradient(U0, r, axis=0, edge_order=2)
OM0 = float(np.sqrt((r[:, None] * dz) ** 2 + (2 * U0 + r[:, None] * dr) ** 2).max())

rows = []
for line in open(ROOT + "STATES_TABLE.md", encoding="utf-8"):
    if not line.startswith("| ") or line.startswith("| file") or line.startswith("|---"):
        continue
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    f, t, aa0, lab, omr, q = c[0], float(c[1]), float(c[3]), float(c[4]), float(c[6]), float(c[7])
    if "e513a5" in f or T - t <= 0 or q > 100:      # Euler state; past T; the excluded degraded snapshot
        continue
    rows.append(dict(f=f, t=t, lab=lab, A=aa0 * A0, om=omr * OM0, q=q, tau=T - t))
rows.sort(key=lambda d: d["t"])
# one point per instant: where two maps share a time, keep the one whose map is not degraded (the later run)
seen, keep = {}, []
for d in rows:
    seen.setdefault(round(d["t"], 9), []).append(d)
for t in sorted(seen):
    keep.append(sorted(seen[t], key=lambda d: d["f"])[-1])

lab = [d["lab"] for d in keep]
print(f"max|omega|(0) = {OM0:.4f}; T = {T}")
print("label      A(T-t)   sup|u||x'|   omega(T-t)")
for d in keep:
    print(f"x{d['lab']:7.1f}   {d['A']*d['tau']:6.3f}   {d['q']:9.2f}   {d['om']*d['tau']:9.1f}   {d['f'][:26]}")

fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
axs[0].semilogx(lab, [d["q"] for d in keep], "o-"); axs[0].set_ylim(60, 80)
axs[0].set_title(r"$\sup |u|\,|x'|$ ($T$-free)")
axs[1].semilogx(lab, [d["A"] * d["tau"] for d in keep], "o-"); axs[1].set_ylim(3, 4.5)
axs[1].set_title(r"$A\,(T-t)$, $T$ = %.7f" % T)
axs[2].semilogx(lab, [d["om"] * d["tau"] for d in keep], "o-")
axs[2].set_title(r"$|\omega|_{\max}\,(T-t)$")
for a in axs:
    a.set_xlabel(r"label $\max|u_1|/12000$"); a.grid(alpha=0.3)
fig.savefig(ROOT + "figures/fig4_constants.pdf", bbox_inches="tight")
fig.savefig(ROOT + "figures/fig4_constants.png", dpi=180, bbox_inches="tight")
print("wrote figures/fig4_constants.{pdf,png}")
