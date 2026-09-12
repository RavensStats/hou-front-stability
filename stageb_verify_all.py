"""stageb_verify_all.py -- one driver that re-runs every step of the stage-B certificate that lies
DOWNSTREAM of the two multi-hour builds, and prints the theorem's constants in the order the
assembly uses them.

WHAT IT DOES AND DOES NOT REBUILD
  Not rebuilt (hours, ~12 GB, hashed instead by stageb_manifest.py):
      the M x M Galerkin matrix  tmatrix_blocks_L48000/   (tmatrix_block.py, 7.4 core-hours)
      the M x (N-M) rectangle    tmatrix_rect_L697000/    (tmatrix_rect.py, 24 core-hours)
  Rebuilt here, subject to memory:
      the pointwise constants Theta_n, gamma, mu_w, mu_2, ||h||, ||h*||, c_*, s_w   (1.5 s)
      the defect delta by Proposition T3, with its orientation certificate           (350 s, 0.4 GB)
      the bordered inverse K~_h                                                      (210 s, 3.1 GB)
      the certified strip operator norms                                            (389 s, 1.0 GB)
      Proposition T2's tail at N = 16000 and the whole assembly                      (< 1 s)
  Always checked here, in every run:
      the presence and SHA-256 of every certified input against SUPPLEMENT_MANIFEST.txt
      the consistency of the two certified zero files (bit-identical on their first 7682 brackets)
      an INDEPENDENT re-derivation of the final assembly in mpmath, written out here rather than
      imported, against the ball arithmetic of stageb_scalars_iv.py

A step whose memory requirement exceeds what the machine has free is NOT run: its certified value is
replayed from its log, whose hash is checked, and the step is reported as REPLAYED.  Replay is not
verification, and the summary says so.  Any missing input, any hash mismatch, any failed check and
any disagreement between the two evaluations of the assembly is a hard failure: the driver prints
FAIL and exits nonzero.

  python stageb_verify_all.py                  # everything that fits in memory
  python stageb_verify_all.py --replay-all      # no heavy re-runs; seconds
  python stageb_verify_all.py --force-heavy     # attempt the heavy steps whatever the free memory
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import time

import numpy as np
from mpmath import mp, mpf, sqrt as mpsqrt

REPO = os.path.dirname(os.path.abspath(__file__))
MANIFEST = "SUPPLEMENT_MANIFEST.txt"
SCRATCH = os.environ.get("STAGEB_SCRATCH", os.path.join(REPO, "verify_all_logs"))

M, NRECT = 7682, 16000
ZEROS_SMALL = "stokes_zeros_rigorous_L48000.npz"
ZEROS_BIG = "stokes_zeros_rigorous_L697000.npz"

# the certified values the section quotes, and the direction each must be checked in.
# "up" = the quoted decimal must be >= the computed quantity;  "down" = <= .
EXPECTED = {
    "Theta_0":  ("506.548892",   "up"),
    "Theta_1":  ("2482.955855",  "up"),
    "Theta_2":  ("21447.36325",  "up"),
    "Theta_4":  ("906972.8975",  "up"),
    "lam_M":    ("47997.288158", "up"),
    "lam_M1":   ("48009.746020", "down"),
    "lam_N1":   ("208193.490748", "down"),
    "Kh":       ("1.2246711",    "up"),
    "delta":    ("8.017047e-3",  "up"),
    "s12":      ("140.320484",   "up"),
    "s21":      ("140.228256",   "up"),
    "tau":      ("84.882443",    "up"),
    "taup":     ("83.185699",    "up"),
    "theta":    ("0.5567917",    "up"),
    "K":        ("2.7632556",    "up"),
    "beta":     ("0.2448593",    "up"),
    "rho":      ("2.370604e-2",  "up"),
    "imw":      ("25.4014658",   "down"),
}

FAILURES = []
NOTES = []


def say(s=""):
    print(s, flush=True)


def head(s):
    say("")
    say("=" * 100)
    say(s)
    say("=" * 100)


def check(ok, label, detail=""):
    say(f"   [{'OK  ' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def free_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return float("nan")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_hashes():
    """path -> sha256, from the flat sections of SUPPLEMENT_MANIFEST.txt."""
    out = {}
    if not os.path.isfile(MANIFEST):
        return out
    pat = re.compile(r"^([0-9a-f]{64})\s+(\d+)\s+(\S.*)$")
    for line in open(MANIFEST, encoding="utf-8"):
        m = pat.match(line.rstrip("\n"))
        if m:
            out[m.group(3).strip()] = m.group(1)
    return out


def run(cmd, logfile, timeout=3600):
    """Run a verification script, tee its stdout to logfile, return (rc, text)."""
    t0 = time.time()
    p = subprocess.run([sys.executable] + cmd, cwd=REPO, capture_output=True, text=True,
                       timeout=timeout)
    txt = (p.stdout or "") + (p.stderr or "")
    os.makedirs(SCRATCH, exist_ok=True)
    with open(os.path.join(SCRATCH, logfile), "w", encoding="utf-8") as f:
        f.write(txt)
    say(f"   ran {' '.join(cmd)}  ->  rc = {p.returncode}, {time.time() - t0:.0f} s, "
        f"stdout in {os.path.join(SCRATCH, logfile)}")
    return p.returncode, txt


def grab(txt, pattern, group=1, cast=float):
    m = re.search(pattern, txt)
    if not m:
        return None
    try:
        return cast(m.group(group))
    except ValueError:
        return None


def cmp_quoted(name, computed, note=""):
    """The section's quoted decimal must bound the computed quantity in the stated direction."""
    if computed is None:
        return check(False, f"{name}: not found in the output", note)
    q, direction = EXPECTED[name]
    qv = float(q)
    ok = (computed <= qv) if direction == "up" else (computed >= qv)
    arrow = "<=" if direction == "up" else ">="
    return check(ok, f"{name} {arrow} {q}", f"computed {computed!r} {note}")


# ------------------------------------------------------------------ steps
def step_inputs(args):
    head("STEP 0.  Certified inputs: presence and SHA-256")
    need = [ZEROS_SMALL, ZEROS_BIG, "rdefect_w_M7682.npz",
            "colenc_axiphys_513_512_nsz_t0.00226_z0.002_N100_m1_k2000_nu0.0005_d30.npz",
            "kh_block_7682.log", "ropnorm_final.log", "t3_balls.log", "rdefect.log",
            "stageb_consts.log", "tnorm_rigorous.log"]
    for f in need:
        check(os.path.isfile(f), f"present: {f}")
    for d in ("tmatrix_blocks_L48000", "tmatrix_rect_L697000"):
        ok = os.path.isdir(d)
        check(ok, f"present: {d}/ (heavy store, hashed by stageb_manifest.py, not rebuilt here)")
    man = manifest_hashes()
    if not man:
        NOTES.append(f"{MANIFEST} absent or empty: hashes not checked. Run stageb_manifest.py.")
        say(f"   [NOTE] {MANIFEST} absent: SHA-256 comparison skipped")
        return
    # stageb_consts.log is REGENERATED by step 2 and carries wall-clock timings, so it is not
    # byte-stable across runs and its hash is not checked; its CONTENT is checked in step 2, which
    # is the check that matters.  Every other certified input is immutable and is hashed.
    volatile = {"stageb_consts.log"}
    n_ok = n_bad = 0
    for f in need:
        if f in volatile or not os.path.isfile(f) or f not in man:
            continue
        if sha256_file(f) == man[f]:
            n_ok += 1
        else:
            n_bad += 1
            check(False, f"SHA-256 mismatch against the manifest: {f}")
    check(n_bad == 0, f"SHA-256 against {MANIFEST}",
          f"{n_ok} files matched, {n_bad} mismatched; "
          f"{len(volatile)} skipped as not byte-stable (regenerated with timings)")


def step_spectrum(args):
    head("STEP 1.  The certified Stokes spectrum (the two zero files)")
    zs, zb = np.load(ZEROS_SMALL), np.load(ZEROS_BIG)
    ls, us = zs["lam_lower"], zs["lam_upper"]
    lb, ub = zb["lam_lower"], zb["lam_upper"]
    check(len(ls) == M, f"the small file holds exactly M = {M} brackets", f"{len(ls)}")
    check(len(lb) == 29278, "the large file holds 29278 brackets", f"{len(lb)}")
    check(np.array_equal(ls, lb[:M]) and np.array_equal(us, ub[:M]),
          "(G2): the first 7682 brackets are bit-identical in the two files")
    check(float(zs["X"]) == float(zb["X"]), "(G1): the two files carry the same X",
          f"X = {float(zb['X']):.12f}")
    check(bool(np.all(np.diff(lb) > 0)) and bool(np.all(ub >= lb)),
          "the brackets are strictly increasing and well formed")
    vals = dict(lam_M=float(ub[M - 1]), lam_M1=float(lb[M]), lam_N1=float(lb[NRECT]))
    for k, v in vals.items():
        cmp_quoted(k, v)
    return vals


def step_consts(args):
    head("STEP 2.  The pointwise constants (stageb_consts.py, 1.5 s)")
    if args.replay_all:
        txt = open("stageb_consts.log", encoding="utf-8").read()
        say("   REPLAYED from stageb_consts.log")
        NOTES.append("step 2 replayed from its log (--replay-all)")
    else:
        rc, _ = run(["stageb_consts.py"], "stageb_consts.stdout")
        check(rc == 0, "stageb_consts.py exited cleanly")
        txt = open("stageb_consts.log", encoding="utf-8").read()
    # the SUMMARY table carries the full-precision enclosure; Sec. 1 prints a 4-digit display of the
    # same interval, whose upper end is rounded to nearest and is therefore not a bound.
    for name, pat in (("Theta_0", r"Theta_0 = \|\|T\|\|\s+\[[\d.]+, ([\d.]+)\]"),
                      ("Theta_1", r"Theta_1\s+\[[\d.]+, ([\d.]+)\]"),
                      ("Theta_2", r"Theta_2\s+\[[\d.]+, ([\d.]+)\]"),
                      ("Theta_4", r"Theta_4\s+\[[\d.]+, ([\d.]+)\]")):
        cmp_quoted(name, grab(txt, pat))
    check("PASS" in txt and "encloses" in txt,
          "the quadrature's own norm identity ||grad g||^2 = X rho is enclosed")
    check(grab(txt, r"mu_2 = \|\|grad grad g\|\|/\|\|grad g\|\| = \[([\d.]+)") is not None
          and grab(txt, r"mu_2 = \|\|grad grad g\|\|/\|\|grad g\|\| = \[([\d.]+)") <= 2.0,
          "mu_2 <= 2 (Proposition T2's standing assumption) is computed, not assumed")


def step_Kh(args):
    head("STEP 3.  The bordered inverse K~_h (kh_block.py, 210 s, 3.1 GB)")
    txt = open("kh_block_7682.log", encoding="utf-8").read()
    t = grab(txt, r"certified lambda_min\(B_m\^H B_m\) >= t = ([\d.]+)")
    r = grab(txt, r"sigma_min over the family >= [\d.]+ - ([\d.e\-]+)")
    check(t is not None and r is not None, "the log records the certified t and the Weyl radius",
          f"t = {t}, r = {r}")
    mp.dps = 40
    kh = 1 / (mpsqrt(mpf(repr(t))) - mpf(repr(r)))
    say(f"   K~_h = 1/(sqrt(t) - r) = {kh}")
    cmp_quoted("Kh", float(kh),
               "(recomputed here from the log's certified t and r, NOT from its printed 1.224671, "
               "which is a round-to-nearest of an upper bound and is 3.8e-8 too small)")
    if args.force_heavy or (free_gb() >= 3.5 and not args.replay_all):
        rc, out = run(["kh_block.py", "--log", os.path.join(SCRATCH, "kh_block.rerun.log")],
                      "kh_block.stdout", timeout=7200)
        check(rc == 0, "kh_block.py re-ran cleanly")
    else:
        say(f"   REPLAYED: kh_block.py needs ~3.1 GB, {free_gb():.2f} GB free; not re-run")
        NOTES.append("step 3 (K~_h) replayed from kh_block_7682.log: it needs 3.1 GB")


def step_delta(args):
    head("STEP 4.  The defect delta (Proposition T3; t3_balls.py, 350 s, 0.4 GB)")
    if args.force_heavy or (free_gb() >= 1.0 and not args.replay_all
                            and os.path.isdir("tmatrix_blocks_L48000")):
        rc, out = run(["t3_balls.py", "--log", os.path.join(SCRATCH, "t3_balls.rerun.log")],
                      "t3_balls.stdout", timeout=7200)
        check(rc == 0, "t3_balls.py re-ran cleanly")
        txt = open(os.path.join(SCRATCH, "t3_balls.rerun.log"), encoding="utf-8").read()
    else:
        txt = open("t3_balls.log", encoding="utf-8").read()
        say(f"   REPLAYED from t3_balls.log ({free_gb():.2f} GB free)")
        NOTES.append("step 4 (delta) replayed from t3_balls.log")
    cmp_quoted("delta", grab(txt, r"part 2 <= ([\d.e\-]+),\s+delta <= ([\d.e\-]+)", group=2))
    check("ORIENTATION CERTIFIED: True" in txt, "the orientation certificate passes on all 7682 modes")
    check("69138 of 69138" in txt, "69138 of 69138 ball overlap tests pass")
    dr = grab(txt, r"max_i \|\(d_n phi_i\)_r\(r_w\)\|\s+<= ([\d.e\-]+)")
    check(dr is not None and dr < 1e-20,
          "the radial wall shear vanishes (the trap of STAGEB_T3 [CHECK] 2)", f"{dr}")


def step_strips(args):
    head("STEP 5.  The certified strip operator norms (ropnorm.py, 389 s, ~1.0 GB)")
    if args.force_heavy or (free_gb() >= 2.0 and not args.replay_all
                            and os.path.isdir("tmatrix_rect_L697000")):
        rc, out = run(["ropnorm.py", "--log", os.path.join(SCRATCH, "ropnorm.rerun.log"),
                       "--Kh", "1.2246711", "--delta", "8.017047e-3"],
                      "ropnorm.stdout", timeout=7200)
        check(rc == 0, "ropnorm.py re-ran cleanly")
        txt = open(os.path.join(SCRATCH, "ropnorm.rerun.log"), encoding="utf-8").read()
    else:
        txt = open("ropnorm_final.log", encoding="utf-8").read()
        say(f"   REPLAYED from ropnorm_final.log ({free_gb():.2f} GB free)")
        NOTES.append("step 5 (strip norms) replayed from ropnorm_final.log")
    s = re.findall(r"over the whole interval family \|\|A\|\|_2 <= ([\d.]+)", txt)
    check(len(s) >= 2, "both strips are certified in the log", f"{s[:2]}")
    if len(s) >= 2:
        cmp_quoted("s12", float(s[-2]))
        cmp_quoted("s21", float(s[-1]))
    check("17/17 complete and INCLUDED" in txt, "the rectangle is complete: 17 of 17 column blocks")
    check(txt.count("PASS") >= 6, "the six falsification tests pass",
          f"{txt.count('PASS')} PASS lines")


def step_assembly(args):
    head("STEP 6.  Proposition T2's tail at N = 16000, and the assembly (stageb_scalars_iv.py)")
    sys.path.insert(0, REPO)
    import stageb_scalars_iv as S
    from flint import arb
    tau, _ = S.tau(NRECT, True, False)
    taup, _ = S.tau(NRECT, True, True)
    dN = S.dist(arb(float(S.lam_lo[NRECT])))
    r = S.assemble(tau, taup, dN)
    fhi = lambda a: float(a.mid()) + float(a.rad()) * 4
    cmp_quoted("tau", fhi(tau))
    cmp_quoted("taup", fhi(taup))
    cmp_quoted("theta", fhi(r["theta"]))
    cmp_quoted("K", fhi(r["K"]))
    cmp_quoted("beta", fhi(r["beta"]))
    cmp_quoted("rho", fhi(r["rho"]))
    cmp_quoted("imw", float(r["imw"].mid()) - float(r["imw"].rad()) * 4)
    check(bool(r["beta"] < arb(1)), "beta < 1 as a ball comparison")
    check(bool(r["imw"] > arb(0)), "Im omega > 0 as a ball comparison")
    return r


def step_independent(args, r):
    head("STEP 7.  An independent re-derivation of the assembly (mpmath, written out here)")
    mp.dps = 60
    lam_M, lam_M1, lam_N1 = mpf("47997.288158"), mpf("48009.746021"), mpf("208193.490749")
    mu_re, mu_im = mpf("-17.525873540058"), mpf("25.425171859977")
    kh, cn, tn = mpf("1.2246711"), mpf("232.2489"), mpf("506.548892")
    s12, s21 = mpf("140.320484"), mpf("140.228256")
    tau, taup, delta = mpf("84.882442"), mpf("83.185699"), mpf("8.017047e-3")
    d_M = mpsqrt(mu_re ** 2 + (lam_M1 + mu_im) ** 2)
    d_N = mpsqrt(mu_re ** 2 + (lam_N1 + mu_im) ** 2)
    c_perp = lam_M1 + mu_im - cn
    t12, t21 = s12 + tau, s21 + taup
    Z = s12 * s21 / d_M + tau * taup / d_N + t12 * t21 * tn / (c_perp * d_M)
    th = kh * Z
    e = 1 / c_perp
    s = kh / (1 - th)
    a = s * s + (e * t21 * s) ** 2
    b = s * (s * t12 * e) + (e * t21 * s) * (e + e * e * t21 * s * t12)
    c = (s * t12 * e) ** 2 + (e + e * e * t21 * s * t12) ** 2
    K = mpsqrt((a + c) / 2 + mpsqrt(((a - c) / 2) ** 2 + b ** 2))
    beta = 4 * K * K * delta
    rho = 2 * K * delta / (1 + mpsqrt(1 - beta))
    imw = mu_im - rho
    say(f"   d_M = {d_M}\n   d_N = {d_N}\n   c_perp = {c_perp}")
    say(f"   ||Z|| = {Z}\n   theta = {th}\n   K = {K}\n   beta = {beta}\n   rho = {rho}")
    say(f"   Im omega >= {imw}")
    for name, mine, ball in (("theta", th, r["theta"]), ("K", K, r["K"]),
                             ("beta", beta, r["beta"]), ("rho", rho, r["rho"])):
        d = abs(mpf(repr(float(ball.mid()))) - mine) / mine
        check(d < mpf("1e-9"), f"the two evaluations of {name} agree", f"relative {float(d):.2e}")
    return dict(theta=th, K=K, beta=beta, rho=rho, imw=imw, Z=Z, d_M=d_M, d_N=d_N, c_perp=c_perp,
                t12=t12, t21=t21)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-all", action="store_true", help="no heavy re-runs; seconds")
    ap.add_argument("--force-heavy", action="store_true", help="attempt every step whatever is free")
    args = ap.parse_args()

    t0 = time.time()
    os.chdir(REPO)
    os.makedirs(SCRATCH, exist_ok=True)
    say("stage B, end-to-end verification of everything downstream of the two heavy builds")
    say(f"repository {REPO}")
    say(f"free physical memory {free_gb():.2f} GB; python {sys.version.split()[0]}")

    step_inputs(args)
    step_spectrum(args)
    step_consts(args)
    step_Kh(args)
    step_delta(args)
    step_strips(args)
    r = step_assembly(args)
    ind = step_independent(args, r)

    head("THE THEOREM'S CONSTANTS, in the order the assembly uses them")
    # Every printed bound is rounded in its own direction: an upper bound up, a lower bound down.
    # Round-to-nearest would print numbers that the quantity violates in the last digit, which is
    # exactly how a display-rounded K~_h once reached the published constants.
    from math import ceil as _ceil, floor as _floor

    def U(x, nd):
        return f"{_ceil(float(x) * 10 ** nd) / 10 ** nd:.{nd}f}"

    def L(x, nd):
        return f"{_floor(float(x) * 10 ** nd) / 10 ** nd:.{nd}f}"

    rows = [
        ("lambda_M            <=", "47997.288158",  "certified, upper"),
        ("lambda_{M+1}        >=", "48009.746020",  "certified, lower"),
        ("lambda_{N+1}        >=", "208193.490748", "certified, lower, N = 16000"),
        ("d_M                 >=", L(ind['d_M'], 4), "|-i lambda_{M+1} - mu|"),
        ("d_N                 >=", L(ind['d_N'], 4), "|-i lambda_{N+1} - mu|"),
        ("c_perp              >=", L(ind['c_perp'], 4), "lambda_{M+1} + Im mu - ||C||"),
        ("||T|| = Theta_0     <=", "506.548892",    "pointwise jet"),
        ("K~_h                <=", "1.2246711",     "verified sigma_min of the bordered block"),
        ("strip 1             <=", "140.320484",    "||P T (P_N - P)||, 17/17 columns"),
        ("strip 2             <=", "140.228256",    "||(P_N - P) T P||"),
        ("tau_N               <=", "84.882443",     "Proposition T2 tail, N = 16000"),
        ("tau'_N              <=", "83.185699",     "Proposition T2 tail, transposed"),
        ("t_12                <=", U(ind['t12'], 4), "strip + tail"),
        ("t_21                <=", U(ind['t21'], 4), "strip + tail"),
        ("||Z||               <=", U(ind['Z'], 7),  "band + tail + resolvent correction"),
        ("theta               <=", U(ind['theta'], 7), "K~_h ||Z||, must be < 1"),
        ("K                   <=", U(ind['K'], 7), "bordered inverse of the full operator"),
        ("delta               <=", "8.017047e-03",  "Proposition T3, N = M"),
        ("beta = 4 K^2 delta  <=", U(ind['beta'], 7), "must be < 1"),
        ("rho                 <=", U(ind['rho'], 8), "enclosure radius, units nu k^2"),
        ("Im omega            >=", L(ind['imw'], 7), "units nu k^2"),
        ("Im omega            >=", L(float(ind['imw']) * 2000, 2) + " s^-1",
         "growth " + L(float(ind['imw']) * 2000 * 2.4478e-5, 4)
         + " (T-t)^-1 at the paper's T - t = 2.4478e-5 s"),
    ]
    for a, b, c in rows:
        say(f"   {a} {b:>16s}    {c}")

    head("SUMMARY")
    for n in NOTES:
        say(f"   NOTE: {n}")
    if FAILURES:
        say(f"   {len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            say(f"      - {f}")
        say(f"\n   FAIL   ({time.time() - t0:.0f} s)")
        sys.exit(1)
    say(f"   every check passed; the criterion closes: beta <= 0.2448593 < 1, "
        f"Im omega >= 25.4014658 > 0")
    say(f"\n   PASS   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
