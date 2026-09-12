"""stageb_manifest.py -- SHA-256 manifest of the certified inputs, the heavy intermediate stores,
the logs and the scripts of the stage-B certificate.

The paper's reproduction paragraph promises hashes rather than data: the two large intermediate
objects (the M x M Galerkin block store and the M x (N-M) rectangle) total about 12 GB and are not
distributed, so a referee who rebuilds them must be able to check that what he built is what was
used.  This script produces that manifest.

  * every file is STREAMED through hashlib in 1 MiB chunks; nothing is loaded into memory and
    nothing is copied, so peak memory is a few MB whatever the file size;
  * a directory is summarised by a DIRECTORY DIGEST: the SHA-256 of the sorted listing
    "<relative path> <size> <sha256>\n" of every file inside it.  Two rebuilds of the same store
    agree iff their digests agree, and the per-file lines are printed as well so that a single
    mismatching file can be identified;
  * entries that are absent are recorded as MISSING rather than skipped silently.

  python stageb_manifest.py                    # everything, ~12 GB, I/O bound
  python stageb_manifest.py --no-stores        # skip the two heavy directories (seconds)
  python stageb_manifest.py --out other.txt
"""
import argparse
import hashlib
import os
import sys
import time

CHUNK = 1 << 20

# ---------------------------------------------------------------- what goes in the manifest
CERTIFIED_DATA = [
    ("stokes_zeros_rigorous_L48000.npz",
     "certified Stokes zeros, lambda/k^2 <= 4.8e4 (7682 brackets; the Galerkin block)"),
    ("stokes_zeros_rigorous_L697000.npz",
     "certified Stokes zeros, lambda/k^2 <= 6.97e5 (29278 brackets; rectangle, tail, constants)"),
    ("colenc_axiphys_513_512_nsz_t0.00226_z0.002_N100_m1_k2000_nu0.0005_d30.npz",
     "the exact rational degree-30 field (Omega, W) of the front column, and the collocation pencil"),
    ("rdefect_w_M7682.npz",
     "the stored Galerkin eigenvector w at M = 7682 (exact binary64)"),
]

STORES = [
    ("tmatrix_blocks_L48000",
     "the rigorous M x M Galerkin matrix of T in the certified Stokes basis (tmatrix_block.py)"),
    ("tmatrix_rect_L697000",
     "the rigorous M x (N-M) coupling rectangle, N = 16000 (tmatrix_rect.py)"),
]

LOGS = [
    ("stokes_rigorous_L697000.log", "certified spectrum: 29278 zeros, argument-principle count"),
    ("tmatrix_rect.log", "rectangle build, 17 of 17 column blocks"),
    ("tmatrix_rect_w1.log", "rectangle build, worker on columns 15-16"),
    ("tmatrix_rect_w2.log", "rectangle build, worker setup"),
    ("kh_block_7682.log", "bordered inverse: K~_h <= 1.2246711 (printed there as 1.224671)"),
    ("ropnorm.log", "strip operator norms, running history"),
    ("ropnorm_final.log", "strip operator norms, 17/17 columns: 140.320484 / 140.228256"),
    ("stageb_consts.log", "pointwise constants: Theta_n, gamma, mu_w, mu_2, ||h||, ||h^*||, c_*, s_w"),
    ("tnorm_rigorous.log", "||T|| (withdrawn) and ||C|| <= 232.2489"),
    ("rdefect.log", "defect part 1 <= 9.124907e-11"),
    ("rdelta.log", "defect, cancellation-free route: 1.095725e-2"),
    ("t3_balls.log", "defect, signed route with the orientation certificate: 8.017047e-3"),
    ("accept_iv.log", "acceptance in ball arithmetic, tail excluded"),
    ("lemmaK_check.out", "Lemma 1 and determinant-identity checks (audit)"),
]

SCRIPTS = [
    ("stokes_rigorous_L.py", "certified Stokes spectrum by the argument principle"),
    ("stokes_mode.py", "the closed-form mode ansatz and the wall determinant"),
    ("tmatrix_block.py", "the M x M Galerkin matrix"),
    ("tmatrix_rect.py", "the M x (N-M) rectangle"),
    ("kh_block.py", "the verified bordered inverse K~_h"),
    ("ropnorm.py", "the certified strip operator norms"),
    ("stageb_consts.py", "the pointwise constants and Proposition T2's tau_N"),
    ("tnorm_rigorous.py", "||T||, ||C|| by subdivision in interval arithmetic"),
    ("rdefect.py", "the in-block part of the defect"),
    ("rdelta.py", "the defect, cancellation-free route"),
    ("t3_balls.py", "the four scalar families, the orientation certificate, the defect"),
    ("accept_iv.py", "the acceptance, tail excluded"),
    ("stageb_scalars_iv.py", "Proposition T2 at N = 16000 and the full assembly with the tail"),
    ("stageb_verify_all.py", "the end-to-end driver"),
    ("stageb_lemmaK_check.py", "Lemma 1 and determinant-identity checks (audit)"),
    ("ivmat.py", "midpoint-radius matrices with a-priori floating-point error bounds"),
]


def sha256_file(path):
    """Stream a file through SHA-256.  Returns (hexdigest, size).  Memory: one CHUNK."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def walk_dir(root):
    """Every file under root, as (relative posix path, absolute path), sorted."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            ap = os.path.join(dirpath, name)
            rp = os.path.relpath(ap, root).replace(os.sep, "/")
            out.append((rp, ap))
    out.sort()
    return out


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def emit_simple(out, entries, heading):
    print(heading, file=out)
    print("-" * len(heading), file=out)
    total = nfiles = 0
    for name, what in entries:
        if not os.path.isfile(name):
            print(f"MISSING  {'-'*64}  {name}    [{what}]", file=out)
            continue
        d, n = sha256_file(name)
        total += n
        nfiles += 1
        print(f"{d}  {n:>12d}  {name}", file=out)
        print(f"{'':64}  {'':>12}    [{what}]", file=out)
    print(f"({nfiles} files, {human(total)})\n", file=out)
    return nfiles, total


def emit_store(out, root, what, verbose=True):
    """Per-file hashes plus a directory digest over the sorted listing."""
    print(f"STORE  {root}", file=out)
    print(f"       [{what}]", file=out)
    if not os.path.isdir(root):
        print("       MISSING\n", file=out)
        return 0, 0
    files = walk_dir(root)
    lines = []
    total = 0
    t0 = time.time()
    for i, (rp, ap) in enumerate(files):
        d, n = sha256_file(ap)
        total += n
        lines.append(f"{rp} {n} {d}\n")
        if i % 25 == 0:
            el = time.time() - t0
            sys.stderr.write(f"\r  {root}: {i+1}/{len(files)} files, {human(total)}, {el:.0f} s")
            sys.stderr.flush()
    sys.stderr.write("\r" + " " * 78 + "\r")
    digest = hashlib.sha256("".join(lines).encode()).hexdigest()
    print(f"       {len(files)} files, {human(total)}", file=out)
    print(f"       DIRECTORY DIGEST (sha256 of the sorted '<path> <size> <sha256>' listing)", file=out)
    print(f"       {digest}", file=out)
    if verbose:
        print(f"       per-file listing follows, '<path> <size> <sha256>':", file=out)
        for ln in lines:
            print("       " + ln.rstrip(), file=out)
    print("", file=out)
    return len(files), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="SUPPLEMENT_MANIFEST.txt")
    ap.add_argument("--no-stores", action="store_true",
                    help="skip the two heavy directories (they are ~12 GB and I/O bound)")
    ap.add_argument("--brief", action="store_true",
                    help="directory digests only, without the per-file listings")
    A = ap.parse_args()

    t0 = time.time()
    nfiles = nbytes = 0
    with open(A.out, "w", encoding="utf-8") as out:
        print("SHA-256 manifest of the stage-B certificate (supplementary material to paper 1)", file=out)
        print("=" * 100, file=out)
        print("Generated by stageb_manifest.py.  Every file is streamed through hashlib in 1 MiB", file=out)
        print("chunks; a directory is summarised by the SHA-256 of its sorted per-file listing.", file=out)
        print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", file=out)
        print("", file=out)

        a, b = emit_simple(out, CERTIFIED_DATA, "CERTIFIED INPUT DATA")
        nfiles += a; nbytes += b
        a, b = emit_simple(out, LOGS, "LOGS OF THE CERTIFIED COMPUTATIONS")
        nfiles += a; nbytes += b
        a, b = emit_simple(out, SCRIPTS, "SCRIPTS")
        nfiles += a; nbytes += b

        if not A.no_stores:
            print("HEAVY INTERMEDIATE STORES (not distributed; rebuild and compare the digest)", file=out)
            print("-" * 74, file=out)
            for root, what in STORES:
                a, b = emit_store(out, root, what, verbose=not A.brief)
                nfiles += a; nbytes += b
        else:
            print("HEAVY INTERMEDIATE STORES: skipped (--no-stores)\n", file=out)

        print("=" * 100, file=out)
        print(f"TOTAL: {nfiles} files, {human(nbytes)}, hashed in {time.time() - t0:.0f} s", file=out)

    print(f"{A.out}: {nfiles} files, {human(nbytes)}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
