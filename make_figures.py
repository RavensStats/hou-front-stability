"""Paper figures from the extracted CSV series and the recorded constants.  Output: figures/*.png and *.pdf."""
import os, glob, csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
def load(name):
    rows = list(csv.DictReader(open(f"figdata/{name}.csv")))
    return np.array([float(r["fraction"]) for r in rows]), np.array([float(r["sigma_Tt"]) for r in rows])
def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=180, bbox_inches="tight"); fig.savefig(f"figures/{name}.pdf", bbox_inches="tight"); plt.close(fig); print("wrote", name)
# Fig 1: Euler x28, tangent-linear rates
fig, ax = plt.subplots(figsize=(6, 4))
for name, lab, st in [("euler_x28_m1_localized_tl", "m = 1, localized seed (co-evolving)", "-"), ("euler_x28_m0_localized_tl", "m = 0, localized seed (co-evolving)", "-"), ("euler_x28_m0_random_tl", "m = 0, random seed (co-evolving)", "--"), ("euler_x28_m1_frozen_N1024", "m = 1, localized seed (frozen base)", ":")]:
    f, s = load(name); ax.plot(100 * f, s, st, label=lab)
ax.axhline(0.5, color="k", lw=0.8); ax.axhline(1.0, color="k", lw=0.5, ls="--"); ax.axhline(1.5, color="k", lw=0.5, ls="--")
ax.text(1, 0.52, "outruns the collapse", fontsize=8); ax.text(1, 1.02, "axis translation", fontsize=8); ax.text(1, 1.52, "time translation", fontsize=8)
ax.set_xlabel("fraction of the remaining time (%)"); ax.set_ylabel(r"amplitude rate $\sigma\,(T-t)$"); ax.set_title("Euler, x28 state"); ax.legend(fontsize=8); ax.set_ylim(0, 1.7)
save(fig, "fig1_euler_rates")
# Fig 2: viscous x144, rates at 5e-4 and 5e-3
fig, ax = plt.subplots(figsize=(6, 4))
for name, lab, st in [("ns5e-4_x144_m1_localized_tl", r"$\nu$ = 5e-4, m = 1 localized", "-"), ("ns5e-4_x144_m1_random_tl", r"$\nu$ = 5e-4, m = 1 random", "--"), ("ns5e-4_x144_m0_random_tl", r"$\nu$ = 5e-4, m = 0 random", "--"), ("ns5e-4_x144_m2_random_tl", r"$\nu$ = 5e-4, m = 2 random", "--"), ("ns5e-3_x144_m1_random_tl", r"$\nu$ = 5e-3, m = 1 random", "-."), ("ns5e-3_x144_m0_random_tl", r"$\nu$ = 5e-3, m = 0 random", "-."), ("ns5e-3_x144_m1_localized_tl", r"$\nu$ = 5e-3, m = 1 localized", ":")]:
    f, s = load(name); ax.plot(100 * f, s, st, label=lab)
ax.axhline(0.5, color="k", lw=0.8); ax.set_xlabel("fraction of the remaining time (%)"); ax.set_ylabel(r"amplitude rate $\sigma\,(T-t)$"); ax.set_title("Navier-Stokes, x144 state"); ax.legend(fontsize=7, ncol=2); ax.set_ylim(-1.5, 1.3)
save(fig, "fig2_viscous_rates")
# Fig 3: crossing fraction vs viscosity
nus = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]; cross = [11, 11.5, 17.4, 27, 30.2]
fig, ax = plt.subplots(figsize=(5, 3.6)); ax.semilogx(nus, cross, "o-"); ax.set_xlabel(r"viscosity $\nu$"); ax.set_ylabel("crossing fraction of the remaining time (%)"); ax.set_title(r"m = 1 short-wave family: $\sigma(T-t)$ crosses 1/2"); ax.grid(alpha=0.3)
save(fig, "fig3_crossing_vs_nu")
# Fig 4: the constants vs amplification (recorded values)
amp = [25, 49, 93, 131, 166, 222, 268, 331]
K1 = [72.3, 72.3, 72.2, 72.2, 72.2, 72.2, 72.2, 72.1]
ATt = [3.37, 3.70, 3.80, 3.85, 3.86, 4.00, 4.16, 4.14]
om = [42.2, 65.9, 82.3, 85.6, 96.2, 104.4, 110.0, 118.0]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
axs[0].semilogx(amp, K1, "o-"); axs[0].set_ylim(60, 80); axs[0].set_title(r"$\sup |u|\,|x'|$ (KNSS quantity, T-free)")
axs[1].semilogx(amp, ATt, "o-"); axs[1].set_ylim(3, 4.5); axs[1].set_title(r"$A\,(T-t)$, T = 0.0022865")
axs[2].semilogx(amp, om, "o-"); axs[2].set_title(r"$|\omega|_{\max}\,(T-t)$")
for a in axs: a.set_xlabel("amplification"); a.grid(alpha=0.3)
save(fig, "fig4_constants")
# Fig 5: resolution certificate for the long-wave entries
fig, ax = plt.subplots(figsize=(6, 4))
for name, lab, st in [("euler_x28_m0_frozen_N512", "m = 0, N_z 512", "-"), ("euler_x28_m0_frozen_N1024", "m = 0, N_z 1024", "--"), ("euler_x28_m0_frozen_R257", "m = 0, 257 radial", "o"), ("euler_x28_m1_frozen_N512", "m = 1, N_z 512", "-"), ("euler_x28_m1_frozen_N1024", "m = 1, N_z 1024", "--"), ("euler_x28_m1_frozen_R257", "m = 1, 257 radial", "s")]:
    f, s = load(name); ax.plot(100 * f, s, st, label=lab, ms=4)
ax.set_xlim(0, 30); ax.set_xlabel("fraction of the remaining time (%)"); ax.set_ylabel(r"$\sigma\,(T-t)$ (frozen base)"); ax.set_title("Resolution pairs, Euler x28"); ax.legend(fontsize=7, ncol=2)
save(fig, "fig5_resolution")
