"""Extract (fraction of remaining time, sigma (T-t)) series from the tangent-linear / frozen-base logs into CSV for the paper's figures."""
import re, glob, os
runs = {
 "euler_x28_m0_localized_tl": "axi3dtl_x28_m0_loc.log", "euler_x28_m0_random_tl": "axi3dtl_x28_m0_rand.log",
 "euler_x28_m1_localized_tl": "axi3dtl_m1_x28.log", "ns5e-4_x144_m1_localized_tl": "axi3dtl_m1_ns144.log", "ns5e-4_x144_m1_random_tl": "axi3dtl_m1r_ns144.log",
 "ns5e-4_x144_m0_random_tl": "axi3dtl_ns144_m0_rand.log", "ns5e-4_x144_m2_random_tl": "axi3dtl_ns144_m2_rand.log",
 "ns5e-3_x144_m0_random_tl": "axi3dtl_ns144_nu5e-3_m0_rand.log", "ns5e-3_x144_m1_random_tl": "axi3dtl_ns144_nu5e-3_m1_rand.log", "ns5e-3_x144_m1_localized_tl": "axi3dtl_ns144_nu5e-3_m1_loc.log",
 "ns1e-3_x144_m1_random_tl": "axi3dtl_ns144_nu1e-3_m1_rand.log", "ns2e-3_x144_m1_random_tl": "axi3dtl_ns144_nu2e-3_m1_rand.log", "ns1e-2_x144_m1_random_tl": "axi3dtl_ns144_nu1e-2_m1_rand.log",
 "euler_x28_m1_frozen_N1024": "axi3dlin_m1b_x28.log", "euler_x28_m1_frozen_N512": "axi3dlin_m1h_x28.log", "euler_x28_m1_frozen_R257": "axi3dlin_m1_x28_R257.log",
 "euler_x28_m0_frozen_N1024": "axi3dlin_m0_x28_N1024.log", "euler_x28_m0_frozen_N512": "axi3dlin_m0_x28_N512.log", "euler_x28_m0_frozen_R257": "axi3dlin_m0_x28_R257.log",
 "ns5e-4_x144_m1_random_frozen_N1024": "axi3dlin_m1r_ns144_N1024.log", "ns5e-4_x144_m1_random_frozen_N2048": "axi3dlin_m1r_ns144_N2048.log",
}
pat_tl = re.compile(r"\((\d\.\d+) of T-t0\).*?sigma\(T-t\) ([-+\d.]+)(?:.*?trans ([\d.]+))?(?:.*?sigma_perp\(T-t\) ([-+\d.infa]+))?")
pat_fr = re.compile(r"\((\d\.\d+) of T-t\).*?sigma \(T-t\) ([-+\d.]+)")
for name, f in runs.items():
    if not os.path.exists(f): print("missing", f); continue
    rows = []
    for line in open(f, encoding="utf-8", errors="ignore"):
        if not line.startswith("  t"): continue
        m = pat_tl.search(line) or pat_fr.search(line)
        if m: rows.append([m.group(1), m.group(2)] + ([m.group(3) or ""] if m.re is pat_tl else []))
    with open(f"figdata/{name}.csv", "w") as out:
        out.write("fraction,sigma_Tt" + (",trans" if rows and len(rows[0]) > 2 else "") + "\n")
        for r in rows: out.write(",".join(r) + "\n")
    print(name, len(rows), "rows")
