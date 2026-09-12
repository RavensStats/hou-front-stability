# hou-front-stability

Codes, data and record for the paper *A certified unstable eigenvalue of a columnar model of the front of Hou's Navier-Stokes collapse, and measured non-axisymmetric growth on the collapsing state* (arXiv id to be added). The current paper is `paper/paper1.pdf`.

`paper/NOTE_KNSS.md` and `paper/COMMENT_FOCM.md` are **superseded**: they were withdrawn as standalone pieces and folded into Section 6 of the paper. They are kept for the record, not as companions, and should not be cited. The earlier combined draft, which carried a different title, has been removed from this repository.

What is here:

- `axiphys.py` -- axisymmetric Euler / Navier-Stokes with swirl in Hou-Luo variables, moving radial map and front-following axial map (produces the snapshots).
- `axi3dlin.py`, `axi3dtl.py`, `axi3dnl.py`, `axi3deig.py` -- azimuthal-mode stability about a frozen or co-evolving axisymmetric state: linear, tangent-linear, nonlinear coupled modes, and Arnoldi (direct, time-stepper, shift-invert) on the linearized operator.
- `colsolve.py`, `colenc.py`, `colenc_run.py`, `coldefect.py`, `ivmat.py` -- the one-dimensional columnar (parallel-flow) model of a column of the flow, its interval-enclosed collocation pencil on an explicit rational polynomial field, the Krawczyk eigenpair enclosure with rigorous floating-point error bounds, and the defect of the certified eigenvector against the continuous operator.
- `lscert.py`, `lscert2.py`, `lscheck.py`, `ls_recheck.py` -- the Leibovich-Stewartson short-wave instability condition: grid scan, interval enclosure and the exact Sturm sequence. **The certificate these produced is retracted**: the criterion was misstated. All three carry retraction headers and now compute it as published; `ls_recheck.py` evaluates the old and the correct form side by side. See `RECORD.md`.
- `knssq.py`, `logfit2.py`, `burgfit.py`, `nszstat.py`, `l3conc.py`, `rollup_end.py`, `twincmp.py` -- the measured quantities (the T-free KNSS quantity sup |u| |x'|, the log-derivative singular-time fit, the Rott-Lundgren front fit, front statistics, the local L^3 norm, the nonlinear end state, twin comparisons).
- `figdata/`, `figures/`, `figdata_extract.py`, `make_figures.py` -- figure data (CSV) and the figures. **See `figures/README.md`: the filenames do not match the printed figure numbers** (`fig6_columnar` is Figure 4, `fig4_constants` is Figure 5).
- `logs/` -- the run and certificate logs (the Krawczyk enclosures at N = 60, 100, 150 and at the other parameters; the module tests; the columnar spectra; the defect and the inverse-norm preview; the tangent-linear and frozen-base stability runs behind Table 2). The Sturm-certificate logs are retained as the record of a retracted result.
- `tests/` -- standalone tests (`python tests/test_ivmat.py`).
- `REPRODUCE.md` -- commands and expected numbers for the headline results (an afternoon).
- `DATA_MANIFEST.md` / `.sha256` -- the 41 snapshot files (637 MB) the paper uses, with sizes and hashes; the files themselves are deposited on Zenodo (DOI to be added) rather than in this repository. Every entry was re-verified against the files on 2026-09-12: all 41 present, all hashes matching.
- `docs/USAGE.md` -- every script's arguments and environment variables. `CONTRIBUTING.md` -- how to report issues and add a snapshot reader.
- `joss/` -- a draft Journal of Open Source Software paper. Not submitted, and it carries the earlier title.
- `RECORD.md` -- the corrections and retractions that touched published numbers, in date order, extracted from the working record.

Licenses: code MIT (`LICENSE`); data, figures and text CC BY 4.0 (`LICENSE-DATA`). Requirements: Python 3.11+, numpy, scipy, sympy, mpmath, matplotlib. Set `OMP_NUM_THREADS=1` per run.
