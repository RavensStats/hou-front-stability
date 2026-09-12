---
title: 'axiphys and axi3d: axisymmetric collapse, azimuthal-mode stability and exact instability certificates for the Hou interior Navier-Stokes scenario'
tags:
  - Python
  - fluid dynamics
  - Navier-Stokes
  - singularity formation
  - linear stability
  - computer-assisted proof
authors:
  - name: "Andrew Mullen"
    orcid: "0009-0001-8581-4286"
    affiliation: 1
affiliations:
  - name: "[affiliation]"
    index: 1
date: 7 September 2026
bibliography: paper.bib
---

# Summary

`axiphys` integrates the axisymmetric incompressible Euler and Navier-Stokes equations with swirl in the Hou-Luo variables (swirl over radius, azimuthal vorticity over radius, stream function over radius) on the periodic cylinder, with a moving radial map and a front-following axial map that keep a collapsing vortex front resolved as its scale shrinks by three orders of magnitude. `axi3dlin`, `axi3dtl` and `axi3dnl` compute, respectively, the linearized evolution of a single azimuthal Fourier mode about a frozen axisymmetric state, the tangent-linear evolution of a mode about the co-evolving collapse (the growth rate measured in the collapsing frame against the threshold that decides whether a perturbation outruns the collapse), and the nonlinear coupled evolution of the base with a truncated set of azimuthal modes and their Reynolds-stress feedback. `axi3deig` wraps the linearized operator as a matrix-free map for Arnoldi iteration in three forms (direct, time-stepper transform, shift-invert). `lscert2` turns the Leibovich-Stewartson short-wave instability criterion on a columnar swirling field into an exact statement: the field is fitted by polynomials with rational coefficients and the sign of the criterion on an annulus is certified by Sturm root counting in exact arithmetic. `colsolve` solves the one-dimensional normal-mode problem of a column of the flow treated as a parallel swirling flow with axial velocity (Chebyshev collocation with axis parity), `colenc` builds the same collocation pencil on an explicit rational polynomial field with every entry enclosed in interval arithmetic, and `ivmat` provides midpoint-radius interval matrices with rigorous floating-point error bounds and a Krawczyk eigenpair enclosure, so that an unstable eigenvalue of the discretized columnar problem can be certified. `knssq`, `logfit2`, `burgfit` and `nszstat` measure the scale-invariant quantities that the exclusion theorems for axisymmetric singularities require, the singular time and exponent by a log-derivative fit, and the front's profile against the Rott-Lundgren strained layer. The package reproduces, in an afternoon, the headline numbers of the accompanying paper [@ours] on the scenario of @Hou2023b.

# Statement of need

The numerical candidates for finite-time singularity of the three-dimensional incompressible equations [@LuoHou2014; @Hou2023a; @Hou2023b] are computed under an imposed axisymmetry. Whether they survive non-axisymmetric perturbation, and whether they satisfy the hypotheses of the theorems that exclude axisymmetric singularities with swirl [@KNSS2009; @SereginSverak2009], are questions that the published codes, which are axisymmetric and private, do not address. No public code existed for either question; the present package is the first, and it is built so that both questions can be re-asked of any axisymmetric state, from any code, saved in the documented snapshot format. Its intended users are researchers in mathematical fluid dynamics who want to test a candidate singular solution against three-dimensional stability and against the exclusion theorems, and numerical analysts working on rigorous eigenvalue enclosures for such operators, for whom the matrix-free operator, its certified resolution pairs, the exact Sturm certificate and the certified discrete columnar eigenvalue are the starting point.

# Functionality and validation

The axisymmetric solver reproduces the published checkpoint of @Hou2023b at 144 times the initial amplitude to three percent and the inviscid checkpoints of @Hou2023a to one to two percent; radial and axial refinement pairs agree to 0.3 percent. The mode solvers recover the trivial directions of the collapse (time translation at rate 3/2 and axis translation at rate 1 in collapse units) as gates, and their rates are certified by 1024-against-2048 axial and 257-against-513 radial pairs. The exact certificate is a Python script over `sympy` rationals with no floating-point step after the fit; the interval-arithmetic alternative failed by the dependency problem, which is why the Sturm route is used. Snapshots are `numpy` archives with keys documented in `DATA_MANIFEST.md`; every snapshot the paper uses is listed there with its SHA-256. `REPRODUCE.md` gives the commands and expected numbers for the five headline results; `tests/` holds the manufactured-solution and conservation tests.

# Acknowledgements

[to be added]

# References
