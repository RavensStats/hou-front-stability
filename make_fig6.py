"""Figure 6: the columnar (parallel-flow) spectrum of the front column of the x144 viscous state versus axial wavenumber,
in collapse units, at nu = 5e-4 (leading and second eigenvalue), inviscid, and nu = 5e-3; the Rayleigh bound of the column;
the 2D frozen-base rate; the certified discrete eigenvalues (Krawczyk) at k = 1600, 2000 (nu 5e-4), 1200 (5e-3), 2000 (inviscid).
Data from colsolve_ns144.log and the colenc_run_*.log certificates."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
Tt = 0.002278 - 0.002262; R = 0.013773
k5e4 = np.array([800, 1200, 1600, 2000, 2400.]); s5e4 = np.array([4.5254e4, 4.8883e4, 5.0395e4, 5.0822e4, 5.0611e4]) * Tt; s2 = np.array([3.231e4, 3.747e4, 3.988e4, 4.091e4, 4.113e4]) * Tt
kinv = np.array([1600, 3200, 6400.]); sinv = np.array([5.1877e4, 5.4586e4, 5.5556e4]) * Tt
k5e3 = np.array([400, 800, 1200, 1600, 2400.]); s5e3 = np.array([3.5069e4, 4.1452e4, 4.1178e4, 3.7344e4, 2.2529e4]) * Tt
cert = [(1600, 5.0415123e4, "C0"), (2000, 5.0850344e4, "C0"), (1200, 4.1178849e4, "C3"), (2000, 5.3106752e4, "k")]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(k5e4 * R, s5e4, "o-", color="C0", label="nu = 5e-4, leading")
ax.plot(k5e4 * R, s2, "o--", color="C0", alpha=0.5, label="nu = 5e-4, second")
ax.plot(kinv * R, sinv, "s-", color="k", label="inviscid, leading")
ax.plot(k5e3 * R, s5e3, "^-", color="C3", label="nu = 5e-3, leading")
ax.axhline(5.99e4 * Tt, color="gray", ls=":", label="Rayleigh bound of the column (0.96)")
ax.axhline(0.80, color="C2", ls="-.", label="2D frozen-base rate (0.80)")
for k, s, c in cert: ax.plot([k * R], [s * Tt], marker="*", ms=13, color=c, mec="gold", mew=1.2, ls="none")
ax.plot([], [], marker="*", ms=11, color="w", mec="gold", mew=1.2, ls="none", label="certified (Krawczyk)")
ax.set_xscale("log"); ax.set_xlabel("k R (axial wavenumber times swirl radius)"); ax.set_ylabel("growth rate, collapse units  Im(omega) (T - t)")
ax.set_ylim(0.3, 1.02); ax.set_title("Columnar normal modes of the front column (m = 1), x144 state"); ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig("figures/fig6_columnar.png", dpi=180); fig.savefig("figures/fig6_columnar.pdf"); print("saved figures/fig6_columnar.{png,pdf}")
