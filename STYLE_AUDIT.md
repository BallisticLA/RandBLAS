# RandBLAS style audit

This audit records the evidence behind `STYLE_GUIDE.md`.
The guide is normative; this file is descriptive.
Keeping the two separate prevents an old inconsistency from becoming a new rule.

## Review snapshot

| Field | Value |
|---|---|
| Review date | 2026-08-16 |
| Commit | `952251cf9dd386452ea9a88e553c9966513e85d1` |
| Branch | `docs/randblas-style-guide` |
| Tracked files | 148 |
| Baseline build | Release, C++20, OpenMP enabled, MKL sparse BLAS unavailable |
| Baseline tests | 438 passed, 0 failed after rebuilding the existing build directory |

The worktree contained two untracked paths before the audit started: `.claude/`
and `docs/`.
The former belongs to the user and is outside this review.
The latter contains the approved execution plan for this audit.

The first test invocation found a stale binary from 2026-08-07 that referenced
the removed files `test_philox.cc` and `philox_kat_vectors.txt`.
Rebuilding `build-randblas` against the recorded commit reduced the registered
test count from 474 to the current 438, all of which passed.
No RandBLAS source was changed to obtain the clean baseline.

## Method

We reviewed all tracked first-party text files at the recorded commit.
Automated searches covered the full corpus; close reading used the anchors
listed below.

Files are compared only to peers.
A test is not an outlier because it differs from a CMake module, and a small
Random123 compatibility layer is not judged as though it were tutorial prose.

We call a convention a **strong consensus** when it appears in at least five
eligible files and at least 90% of eligible occurrences.
A **working consensus** needs five eligible files and a share of at least 70%.
Everything else is **mixed**.
Mixed evidence is either omitted from the guide, split by file stratum, or
presented as an explicit forward-looking recommendation.

## Corpus

The snapshot contains the following extensions.

| Extension | Files | Extension | Files |
|---|---:|---|---:|
| `.cc` | 37 | `.hh` | 32 |
| `.rst` | 16 | `.md` | 13 |
| `.cmake` | 8 | `.yml` | 8 |
| `.txt` | 7 | `.ps1` | 4 |
| `.in` | 3 | `.py` | 3 |
| no extension | 3 | `.gitignore` | 2 |
| `.sh` | 2 | `.yaml` | 2 |
| `.cloc_exlude` | 1 | `.conf` | 1 |
| `.css` | 1 | `.git-keep` | 1 |
| `.js` | 1 | `.keep` | 1 |
| `.mm` | 1 | `.xml` | 1 |

The top-level distribution is:

| Area | Files | Area | Files |
|---|---:|---|---:|
| `test/` | 40 | `RandBLAS/` | 34 |
| `rtd/` | 25 | `examples/` | 12 |
| `.github/` | 11 | `CMake/` | 10 |
| `docker/` | 2 | `install/` | 2 |
| top-level files | 12 |  |  |

### Comparison strata

| Stratum | Files | Scope |
|---|---:|---|
| Public and core headers | 11 | `RandBLAS.hh` and `RandBLAS/*.hh` |
| Sparse and testing support headers | 19 | `RandBLAS/sparse_data/*.hh` and `RandBLAS/testing/*.hh` |
| Tests | 32 | `test/**/*.cc` and `test/**/*.hh` |
| Examples and benchmarks | 7 | `examples/**/*.cc` |
| Build and configuration | 16 | CMake lists/modules/templates and `RandBLAS/config.h.in` |
| Documentation | 29 | tracked Markdown and reStructuredText |
| Python and web-theme sources | 5 | tracked Python, JavaScript, and CSS |
| Automation and installers | 16 | tracked shell, PowerShell, YAML, and workflow files |
| **Classified total** | **135** | Every file used in at least one peer comparison |

The documentation stratum includes `AGENTS.md` for coverage, but that file is
an agent policy rather than contributor prose.
Its requirements can constrain the guide; its phrasing does not establish the
project's prose style.

### Exclusions

These 13 files are accounted for but excluded from dominant-style counts.

| Path or glob | Reason | Compared elsewhere? |
|---|---|---|
| `.cloc_exlude` | Tool-specific path list | No |
| `.gitignore`, `examples/.gitignore` | Git pattern data | No |
| `LICENSE` | Fixed legal text | No |
| `docker/tsan/Dockerfile` | Single Dockerfile; no peers | Its commands inform workflow review only |
| `examples/sparse-data-matrices/.keep`, `test/.git-keep` | Empty-directory sentinels | No |
| `rtd/requirements.txt` | Dependency manifest | No |
| `rtd/source/Doxyfile` | Doxygen configuration syntax | Configuration facts only |
| `rtd/source/DoxygenLayout.xml` | Machine-readable layout | No |
| `rtd/themes/randblas_rtd/theme.conf` | Theme metadata | Configuration facts only |
| `test/basic_rng/r123_kat_vectors.txt` | Known-answer test data | No |
| `test/basic_rng/r123_rngNxW.mm` | Numeric fixture | No |

### Close-reading anchors

The automated review is anchored by representative files from each major
stratum:

- Core API: `RandBLAS/base.hh`, `RandBLAS/dense_skops.hh`, and
  `RandBLAS/skge.hh`.
- Sparse internals: `RandBLAS/sparse_data/csr_spmm_impl.hh`,
  `RandBLAS/sparse_data/spmm_dispatch.hh`, and
  `RandBLAS/sparse_data/csr_matrix.hh`.
- Tests: `test/datastructures/test_denseskop.cc`,
  `test/linops/test_lskge3.cc`, and
  `test/linops/test_spmm/test_spmm_csr.cc`.
- Examples: `examples/total-least-squares/tls_dense_skop.cc`,
  `examples/sparse-low-rank-approx/svd_rank1_plus_noise.cc`, and
  `examples/simple-kernel-benchmarks/spmm_performance.cc`.
- Other languages: `CMakeLists.txt`, `CMake/rb_config.cmake`,
  `rtd/source/conf.py`, `rtd/source/tutorial/gemm.rst`, and `README.md`.

The anchors help interpret raw counts.
They do not override the full-corpus evidence.

