# Contributing

- Issues: open a GitHub issue with the script name, the exact command, the snapshot file (name and SHA-256 from DATA_MANIFEST.md) and the printed output. Reproduction commands with expected numbers are in REPRODUCE.md.
- Adding a snapshot reader: snapshots are numpy archives with keys U (u_theta / r), Om (omega_theta / r), Ps (psi / r), r, z (half-period axial grid), t, zmap, nu; a converter from another code's output only needs to write those keys. Please add the new reader next to the loaders at the top of the script you extend and a line in docs/USAGE.md.
- Tests: `python tests/test_axisym_manufactured.py` and the other files in tests/ run standalone (no pytest); add one for any new solver.
- Record: RECORD.md lists the corrections and retractions that touched published numbers; the full dated working record (queue, literature checks) is kept by the author and available on request.
- Style: one script per question, environment variables for options, explicit file lists in commits.
