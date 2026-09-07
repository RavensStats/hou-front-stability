# JOSS submission checklist (what the reviewers check; status)

- [x] LICENSE (MIT) and LICENSE-DATA (CC BY 4.0) at the root; repository to be made public by Andrew.
- [x] Statement of need, functionality, validation in paper.md.
- [ ] Author line, ORCID, affiliation in paper.md; arXiv id in paper.bib (entry `ours`).
- [x] Installation: numpy, scipy, sympy, mpmath, matplotlib; Python 3.11+ (document in README).
- [x] Example usage with expected output: REPRODUCE.md.
- [x] Automated tests: tests/ (run `python tests/test_axisym_manufactured.py` etc.; no pytest by repo convention -- consider adding a pytest wrapper).
- [x] Community guidelines: CONTRIBUTING.md.
- [x] API documentation: docs/USAGE.md (generated from docstrings and environment reads; regenerate after changes).
- [ ] Scrub README findings 217 and 249 (unrelated material) before the repository goes public.
- Compile check: `pandoc joss/paper.md --bibliography joss/paper.bib -o /tmp/paper.pdf` or the JOSS GitHub action (openjournals/openjournals-draft-action).
