r"""Extract the growth-rate series from the tangent-linear (co-evolving) and frozen-base logs into CSV for the paper's
figures.

The logs print sigma ALREADY NORMALIZED by a remaining time computed from the command-line T_est = 0.002278, which is
not a singular time (CORRECTION_T.md).  This script therefore never lifts a normalized value out of a log: it recovers
the RAW rate in inverse seconds and re-normalizes it with the paper's conventions, and it recomputes the fraction of the
remaining time on the same convention.

Conventions (one per base state, stated in the CSV header and in Section 3 of the paper):
  * viscous x144 runs: T = 0.0022865, the log-derivative fit of Section 5.1 (t0 = 0.002262022034, T - t0 = 2.4478e-5);
  * Euler x28 runs:    T_E = 0.002278 is KEPT, because no Euler singular-time fit converges (Section 3.1); the Euler
    numbers therefore carry a systematic band and rest on ratios, which are invariant.

The two solvers normalize differently, so there is no single correction factor:
  * axi3dlin.py (frozen base) prints sigma * (T_est - t0)  -- and also prints the raw sigma, which we read directly;
  * axi3dtl.py (co-evolving)  prints sigma * (T_est - t)   -- raw sigma = printed / (T_est - t) at the CURRENT time t.
The translation-orthogonal rate of the m = 1 co-evolving runs is re-measured here with the solver's own estimator
(axi3dtl.py: E_perp = E (1 - trans^2), sigma_perp = (1/2) d ln E_perp / dt, so sigma_perp = sigma + (1/2) d ln(1 -
trans^2) / dt), taking the derivative of ln(1 - trans^2) from a smooth polynomial fit of the logged overlap rather than
from a two-point difference: the older logs print t to only 1e-7, which is a tenth of an output interval, and the
two-point differences recorded at the time were correspondingly noisy (they are non-monotone).  The fit residual in
trans is below 1e-3 and the result is stable to three digits across polynomial degrees 5 to 7.
"""
import re, os, numpy as np

T_EST_DEFAULT = 0.002278              # what the runs in this repository were given on the command line
pat_hdr = re.compile(r"\(T_est ([\d.eE+-]+),")   # axi3dtl prints its own T_est in the header; read it rather than assume
SNAP_T = {"euler_x28": 0.0022081115154412964, "ns_x144": 0.002262022034429895}
T_TRUE = {"euler_x28": 0.002278, "ns_x144": 0.0022865}

def snap_time(npz, key):
    """Exact snapshot time from the .npz if it is present, else the recorded value."""
    if os.path.exists(npz):
        try:
            return float(np.load(npz)["t"])
        except Exception:
            pass
    return SNAP_T[key]

BASES = {
    "euler_x28": ("axiphys_513_512_e513a5_t0.00221.npz", "euler_x28"),
    "ns_x144": ("axiphys_513_512_nsz_t0.00226.npz", "ns_x144"),
}
T0 = {k: snap_time(*v) for k, v in BASES.items()}

# name -> (log file, solver kind, base state)
runs = {
 "euler_x28_m0_localized_tl": ("axi3dtl_x28_m0_loc.log", "tl", "euler_x28"),
 "euler_x28_m0_random_tl": ("axi3dtl_x28_m0_rand.log", "tl", "euler_x28"),
 "euler_x28_m1_localized_tl": ("axi3dtl_m1_x28.log", "tl", "euler_x28"),
 "ns5e-4_x144_m1_localized_tl": ("axi3dtl_m1_ns144.log", "tl", "ns_x144"),
 "ns5e-4_x144_m1_random_tl": ("axi3dtl_m1r_ns144.log", "tl", "ns_x144"),
 "ns5e-4_x144_m0_random_tl": ("axi3dtl_ns144_m0_rand.log", "tl", "ns_x144"),
 "ns5e-4_x144_m2_random_tl": ("axi3dtl_ns144_m2_rand.log", "tl", "ns_x144"),
 "ns5e-3_x144_m0_random_tl": ("axi3dtl_ns144_nu5e-3_m0_rand.log", "tl", "ns_x144"),
 "ns5e-3_x144_m1_random_tl": ("axi3dtl_ns144_nu5e-3_m1_rand.log", "tl", "ns_x144"),
 "ns5e-3_x144_m1_localized_tl": ("axi3dtl_ns144_nu5e-3_m1_loc.log", "tl", "ns_x144"),
 "ns1e-3_x144_m1_random_tl": ("axi3dtl_ns144_nu1e-3_m1_rand.log", "tl", "ns_x144"),
 "ns2e-3_x144_m1_random_tl": ("axi3dtl_ns144_nu2e-3_m1_rand.log", "tl", "ns_x144"),
 "ns1e-2_x144_m1_random_tl": ("axi3dtl_ns144_nu1e-2_m1_rand.log", "tl", "ns_x144"),
 "euler_x28_m1_frozen_N1024": ("axi3dlin_m1b_x28.log", "fr", "euler_x28"),
 "euler_x28_m1_frozen_N512": ("axi3dlin_m1h_x28.log", "fr", "euler_x28"),
 "euler_x28_m1_frozen_N2048": ("axi3dlin_m1b_x28_N2048.log", "fr", "euler_x28"),
 "euler_x28_m1_frozen_R257": ("axi3dlin_m1_x28_R257.log", "fr", "euler_x28"),
 "euler_x28_m0_frozen_N1024": ("axi3dlin_m0_x28_N1024.log", "fr", "euler_x28"),
 "euler_x28_m0_frozen_N512": ("axi3dlin_m0_x28_N512.log", "fr", "euler_x28"),
 "euler_x28_m0_frozen_N2048": ("axi3dlin_m0_x28_N2048.log", "fr", "euler_x28"),
 "euler_x28_m0_frozen_R257": ("axi3dlin_m0_x28_R257.log", "fr", "euler_x28"),
 "ns5e-4_x144_m1_random_frozen_N1024": ("axi3dlin_m1r_ns144_N1024.log", "fr", "ns_x144"),
 "ns5e-4_x144_m1_random_frozen_N2048": ("axi3dlin_m1r_ns144_N2048.log", "fr", "ns_x144"),
}

# co-evolving: t, fraction on T_est, sigma*(T_est - t), then optionally trans and sigma_perp*(T_est - t)
pat_tl = re.compile(r"^\s*t\s+([\d.eE+-]+)\s+\(([\d.]+) of T-t0\).*?sigma\(T-t\)\s+([-+\d.]+)")
pat_trans = re.compile(r"\btrans\s+([\d.]+)")
# frozen: t', fraction, RAW sigma, sigma*(T_est - t0)
pat_fr = re.compile(r"^\s*t'\s+([\d.eE+-]+)\s+\(([\d.]+) of T-t\)\s+E/E0\s+[\d.eE+-]+\s+sigma\s+([-+\d.eE]+)\s+sigma \(T-t\)\s+([-+\d.]+)")

os.makedirs("figdata", exist_ok=True)
summary = []
for name, (f, kind, base) in runs.items():
    if not os.path.exists(f):
        print("missing", f)
        continue
    t0 = T0[base]
    Tn = T_TRUE[base]
    T_EST = T_EST_DEFAULT
    if kind == "tl":
        for line in open(f, encoding="utf-8", errors="ignore"):
            mh = pat_hdr.search(line)
            if mh:
                T_EST = float(mh.group(1))
                break
    rows = []
    for line in open(f, encoding="utf-8", errors="ignore"):
        if kind == "tl":
            m = pat_tl.match(line)
            if not m:
                continue
            # the older logs print t with only 7 decimals, a tenth of an output interval; where that is so, recover
            # t from the printed fraction instead, which pins it to a few parts in 1e8 of the remaining time
            ts = m.group(1)
            t = float(ts)
            if len(ts.split(".")[-1]) < 9:
                t = t0 + float(m.group(2)) * (T_EST - t0)
            sig_norm_old = float(m.group(3))
            raw = sig_norm_old / (T_EST - t)          # undo the log's normalization at the CURRENT time
            frac = (t - t0) / (Tn - t0)
            sig = raw * (Tn - t)                      # re-normalize at the current time on the paper's T
            mt = pat_trans.search(line)
            trans = float(mt.group(1)) if mt else None
            perp = ""
        else:
            m = pat_fr.match(line)
            if not m:
                continue
            tp = float(m.group(1))
            raw = float(m.group(3))                   # the frozen solver prints the raw rate; use it
            t = t0 + tp
            frac = tp / (Tn - t0)
            sig = raw * (Tn - t0)                     # frozen base: normalize at the SNAPSHOT time throughout
            trans, perp = None, ""
        rows.append([frac, sig, raw, t, trans, perp])

    # translation-orthogonal rate, re-measured with the solver's own estimator (see the module docstring)
    if kind == "tl" and "_m1_" in name and rows and all(r[4] is not None for r in rows):
        fo = np.array([(r[3] - t0) / (T_EST - t0) for r in rows])       # fraction on the run's own T_est axis
        tr = np.array([r[4] for r in rows])
        pol = np.poly1d(np.polyfit(fo, tr, 6))
        dpol = pol.deriv()
        resid = float(np.abs(pol(fo) - tr).max())
        dln = -2.0 * pol(fo) * dpol(fo) / (1.0 - pol(fo) ** 2) / (T_EST - t0)   # d ln(1 - trans^2) / dt
        for r, d in zip(rows, dln):
            r[5] = f"{(r[2] + 0.5 * d) * (Tn - r[3]):.6f}"
        print(f"    (sigma_perp re-measured; max trans fit residual {resid:.1e})")
    with open(f"figdata/{name}.csv", "w") as out:
        out.write(f"# base {base}: t0 = {t0:.12f}, T = {Tn}, T - t0 = {Tn - t0:.6e}; raw rates from {f}\n")
        out.write("fraction,sigma_Tt,sigma_raw,t,trans,sigma_perp_Tt\n")
        for fr, sg, raw, t, tr, pp in rows:
            out.write(f"{fr:.6f},{sg:.6f},{raw:.6e},{t:.10f},{'' if tr is None else f'{tr:.3f}'},{pp}\n")
    summary.append((name, len(rows)))
    print(f"{name:38s} {len(rows):4d} rows  (T = {Tn}, T - t0 = {Tn - t0:.4e})")
