# hou-front-stability

Codes, data pointers and record for the paper *Hou's Navier-Stokes singularity scenario: a resolved front in the class excluded by the Liouville theorems, and its certified centrifugal instability* (arXiv id to be added), and for the two-page companion note (`paper/NOTE_KNSS.md`, `paper/COMMENT_FOCM.md`).

What is here:

- `axiphys.py` -- axisymmetric Euler / Navier-Stokes with swirl in Hou-Luo variables, moving radial map and front-following axial map (produces the snapshots).
- `axi3dlin.py`, `axi3dtl.py`, `axi3dnl.py`, `axi3deig.py` -- azimuthal-mode stability about a frozen or co-evolving axisymmetric state: linear, tangent-linear, nonlinear coupled modes, and Arnoldi (direct, time-stepper, shift-invert) on the linearized operator.
- `colsolve.py`, `colenc.py`, `colenc_run.py`, `coldefect.py`, `ivmat.py` -- the one-dimensional columnar (parallel-flow) model of a column of the flow, its interval-enclosed collocation pencil on an explicit rational polynomial field, the Krawczyk eigenpair enclosure with rigorous floating-point error bounds, and the defect of the certified eigenvector against the continuous operator.
- `lscert2.py`, `lscheck.py` -- the Leibovich-Stewartson short-wave instability condition: float scan and the exact Sturm-sequence certificate.
- `knssq.py`, `logfit2.py`, `burgfit.py`, `nszstat.py`, `l3conc.py`, `rollup_end.py`, `twincmp.py` -- the measured quantities (the T-free KNSS quantity sup |u| |x'|, the log-derivative singular-time fit, the Rott-Lundgren front fit, front statistics, the local L^3 norm, the nonlinear end state, twin comparisons).
- `figdata/`, `figures/`, `figdata_extract.py`, `make_figures.py` -- figure data (CSV) and the five figures.
- `logs/` -- the certificate logs (exact Sturm certificates; the Krawczyk enclosures at N = 60, 100, 150 and at the other parameters; the module tests; the columnar spectra; the defect and the inverse-norm preview).
- `tests/` -- standalone tests (`python tests/test_ivmat.py`).
- `REPRODUCE.md` -- commands and expected numbers for the headline results (an afternoon).
- `DATA_MANIFEST.md` / `.sha256` -- the 21 snapshot files (201 MB) the paper uses, with sizes and hashes; the files themselves are deposited on Zenodo (DOI to be added) rather than in this repository.
- `docs/USAGE.md` -- every script's arguments and environment variables. `CONTRIBUTING.md` -- how to report issues and add a snapshot reader.
- `joss/` -- the Journal of Open Source Software paper.
- `RECORD.md` -- the corrections and retractions that touched published numbers, in date order, extracted from the working record.

Licenses: code MIT (`LICENSE`); data, figures and text CC BY 4.0 (`LICENSE-DATA`). Requirements: Python 3.11+, numpy, scipy, sympy, mpmath, matplotlib. Set `OMP_NUM_THREADS=1` per run.
