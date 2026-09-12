# Reproduce the headline results in an afternoon

Machine used: 16 cores, 16 GB, Python 3.14 with numpy, scipy, sympy, mpmath, matplotlib. Set `OMP_NUM_THREADS=1` per run; the codes are memory-bandwidth bound.

## 1. The m = 1 instability of Hou's viscous collapse (Section 3.2)

Snapshot: `axiphys_513_512_nsz_t0.00226.npz` (the nu = 5e-4 continuation of Hou's data at t = 0.002262, 144 times the initial amplitude; produced by `axiphys.py` from Hou's initial data, see Section 2). Tangent-linear run of the random-seed m = 1 family on the co-evolving base:

```
AXL_SEED=random python axi3dtl.py axiphys_513_512_nsz_t0.00226.npz 0.002278 1 400
```

Expected: the column `sigma(T-t)` rises through 0.475 (10 percent of the remaining time), 0.627 (14 percent), 0.768 (18 percent). About 2 hours. The localized (long-wave) seed is the same command without `AXL_SEED`; expected 0.44 at 51 percent (stable).

Certificate: `python axi3dlin.py axiphys_513_512_nsz_t0.00226.npz 0.002278 1 400 2048` (frozen base, random seed via `AXL_SEED=random`) reproduces the 1024-point rates to 3-7 percent at 4-8 percent of the remaining time.

## 2. The T-free KNSS quantity (Section 6.0)

```
python knssq.py
```

prints, for every resolved snapshot, `K1 = sup|u| r` (expected 72.1-72.4 from x144 to x331), the value at the swirl maximum (21-22), the Type I constant (74.6-75.4) and the circulation maximum (64.7-64.9). Snapshots `nsz2_snap_t*.npz`, `nsz5_snap_t*.npz`, `nsz6_snap_t*.npz` are the ones used; they were produced by

```
AXP_RESTART=axiphys_513_512_nsz_t0.00228.npz AXP_ZBETA=0.97 AXP_ZM=1 AXP_FILTER=1 AXP_MOVING=1 AXP_REZONE=100 AXP_A=5 AXP_TAG=_nsz2 python axiphys.py 513 512 0.0023 5e-4
```

(x25 to x217; about 12 hours) and, for the front-resolved points,

```
AXP_RESTART=nsz2_snap_t0.002284044.npz AXP_ZBETA=0.98 AXP_ZM=1 AXP_ZFOLLOW=1 AXP_ZFRAC=0.2 AXP_ZWMULT=3 AXP_ZWMIN=1.5e-4 AXP_FILTER=1 AXP_MOVING=1 AXP_REZONE=100 AXP_A=5 AXP_CFL=0.6 AXP_EVERY=2.5e-7 AXP_TAG=_nsz6 python axiphys.py 513 1024 0.0022865 5e-4
```

(x131 to x331; about 4 hours; keep the window floor `AXP_ZWMIN`, without it the map over-tightens and the Poisson solve degrades).

## 3. The singular time and exponent (Section 5.1)

```
python logfit2.py
```

Sliding-window log-derivative fits; expected p = 1.00-1.06 and T = 0.0022864-0.0022866 over the last three windows.

## 4. The front as a Rott-Lundgren layer (Section 5.4)

```
python burgfit.py 0.0022865 "nsz5_snap_t0.002285299.npz"
```

Expected: standard deviation 0.58 sqrt(nu (T - t)), implied strain 2.9 in Type I units, residual 5 percent.

## 5. The Leibovich-Stewartson condition -- RETRACTED certificate, and what replaces it

The exact certificate that earlier versions of this section reproduced is **retracted**. The criterion
had been misstated (a non-negative prefactor `2 V Om` in place of `V DOm`, and `D(W^2)` in place of
`(DW)^2`); see `RECORD.md` and the header of `lscert.py`. Corrected, the quantity is not certifiable
on the polynomial fit, and these scripts now say so.

```
python lscert2.py axiphys_513_512_nsz_t0.00226.npz 20 2.0
```

Expected: **NOT certified.** That is the correct result, not a failure of the script. The corrected
criterion carries `Om'` in the prefactor as well as the bracket, and the degree-20 fit's first
derivative is wrong by up to a factor of two near the swirl maximum, which is where the annulus of
interest lies.

What survives is the grid evaluation, which is what the paper now reports:

```
python lscheck.py axiphys_513_512_nsz_t0.00226.npz
python ls_recheck.py axiphys_513_512_nsz_t0.00226.npz
```

Expected: `Phi` negative in a band just outside the swirl maximum, its minimum at
`(r, z) = (0.0140, 0.0040)` against the swirl maximum at `(0.0138, 0.0037)`. `ls_recheck.py` prints
the misstated and the correct form side by side, which is how the error was established.

## 6. The certified columnar eigenvalue (Section 3.6)

```
python tests/test_ivmat.py
python colenc.py axiphys_513_512_nsz_t0.00226.npz 0.002 60 2.0 1 2000 5e-4 30
python colenc_run.py colenc_axiphys_513_512_nsz_t0.00226_z0.002_N60_m1_k2000_nu0.0005_d30.npz
```

Expected: `ALL TESTS PASSED`; then `CERTIFIED ... Im omega in [5.0850368e4 +- 2e-8]`, growth 0.8125 (T - t)^-1 (15 s to build the interval pencil at N = 60; N = 150 takes minutes). The floating-point spectrum versus wavenumber, viscosity and column height: `COL_Z=0.002 COL_DEG=0 python colsolve.py axiphys_513_512_nsz_t0.00226.npz 200 2.0 1 800,1600,2400 5e-4` (expected leading Im omega 4.53e4, 5.04e4, 5.06e4).

## 7. Everything else

`figdata_extract.py` and `make_figures.py` regenerate the figure data and the figures from the logs; `RECORD.md` lists the corrections and retractions that touched published numbers.
