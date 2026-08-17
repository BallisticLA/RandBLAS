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

## Convention profile

The C++ corpus contains 69 files and 24,284 lines.
The tables below report eligible occurrences, not just files that happen to
contain a pattern.

### C++ structure and spacing

| Convention | Public/core | Support headers | Tests | Examples | Assessment |
|---|---:|---:|---:|---:|---|
| `template <...>` | 71/78 (91.0%) | 127/132 (96.2%) | 109/124 (87.9%) | 26/26 (100%) | Strong in library code; working in tests |
| Space after `#include` | 93/94 (98.9%) | 144/144 (100%) | 236/236 (100%) | 99/99 (100%) | Strong |
| Same-line namespace brace | 17/18 (94.4%) | 27/27 (100%) | 1/1 (100%) | No eligible occurrence | Strong in library code |
| Spaced inheritance colon | 0/1 | 1/1 | 60/62 (96.8%) | No eligible occurrence | Strong in tests; too little library evidence |
| No duplicate include target | 10/11 files | 19/19 files | 31/32 files | 4/7 files | Strong outside examples |
| Project include uses quotes | 25/37 (67.6%) | 96/96 (100%) | 82/96 (85.4%) | 6/6 (100%) | Strong in support/examples; mixed in public/core |

The public/core include result needs context.
`RandBLAS.hh` is an installed umbrella header and uses angle-bracket paths for
all nine `RandBLAS/` includes.
`RandBLAS/util.hh` accounts for the other three angle-bracket project
includes.
Eight of the remaining root headers use repository-qualified quoted includes;
`RandBLAS/random_gen.hh` has one quoted relative include.
The guide therefore recommends repository-qualified quotes within component
headers while preserving the umbrella header's installed-header convention.

Every one of the 32 headers has effective multiple-inclusion protection.
Thirty use exactly one `#pragma once`, `RandBLAS.hh` uses a conventional
include guard, and `RandBLAS/sparse_data/csr_trsm_impl.hh` contains two
`#pragma once` directives.
New headers should use one `#pragma once` after the license block.

Same-line opening braces dominate the broader C++ corpus: 1,876 of 2,033
observed opening braces (92.3%) share a line with preceding code.
Tests contain most of the exceptions.
Of 459 `TEST` or `TEST_F` definitions, 335 (73.0%) put the opening brace on
the macro line and 124 put it on the next line.
Fixture/type declarations are mixed, so the guide adopts the overall
same-line convention for new code without demanding a cleanup of old tests.

Braces around single-statement control bodies are mixed.
The scan found 742 same-line braced controls and 285 candidates whose body
starts on the following line without a same-line brace.
The latter pattern appears in core headers, tests, and examples.
The guide requires braces when a body has multiple statements or nested
control flow; a single short statement may remain unbraced when the local
code is clear.

### Whitespace and line length

| Stratum | Tab-free files | Tabbed lines | Files with trailing whitespace | Trailing-whitespace lines | Lines over 100 columns |
|---|---:|---:|---:|---:|---:|
| Public/core | 10/11 | 2 | 8/11 | 179 | 131 |
| Support headers | 18/19 | 9 | 9/19 | 85 | 125 |
| Tests | 30/32 | 28 | 22/32 | 146 | 166 |
| Examples | 7/7 | 0 | 5/7 | 43 | 92 |
| **Total** | **65/69 (94.2%)** | **39** | **44/69** | **453** | **514** |

Four-space block indentation is consistent across the close-reading anchors,
and tabs are confined to four files.
This is strong evidence for spaces and four-space block indentation.

The repository does not establish a clean trailing-whitespace convention:
453 lines in 44 files contain it.
Nor does it support a hard 100-column limit: 514 lines in 43 files exceed
that length, often for signatures, formulas, macro bodies, or URLs.
The guide treats both as forward-looking recommendations: remove trailing
whitespace and wrap prose or code when doing so helps a reader, but do not
distort a mathematical expression or tabular signature merely to hit a
number.

### Pointer and reference binding

Pointer binding is mixed.
A declaration-shaped scan found 274 occurrences resembling `T* ptr` and 330
resembling `T *ptr`.
The split changes by stratum: examples favor `T* ptr`, while recent sparse
kernels and much of the public API favor `T *ptr`.

References are much less ambiguous.
The same scan found 282 occurrences resembling `T &ref` and 47 resembling
`T& ref`.
The guide recommends declarator binding (`T *ptr`, `T &ref`) for new code and
labels the pointer half as a recommendation rather than observed consensus.

### Naming and API shape

The following patterns recur across the public and support headers.

| Subject | Evidence | Assessment |
|---|---|---|
| Namespace | `RandBLAS` with descriptive lower-case nested namespaces such as `dense` and `sparse_data` | Working consensus |
| Public types | `RNGState`, `DenseDist`, `DenseSkOp`, `SparseDist`, `COOMatrix`, `CSRMatrix`, `CSCMatrix`, `IndexBase` | Strong PascalCase pattern |
| Concepts | `SignedInteger`, `SketchingDistribution`, `SketchingOperator`, `SparseMatrix` | Strong PascalCase pattern |
| Functions | `sketch_general`, `sketch_sparse`, `fill_dense`, `left_spmm`, `safe_int_product` | Strong snake_case pattern |
| Variables and fields | `n_rows`, `row_stride`, `next_state`, `index_base` | Strong snake_case pattern |
| Template parameters | `T`, `RNG`, `SKOP` for principal types; `sint_t`, `state_t` for index/state aliases | Working semantic pattern |
| Macros | Compiler/configuration macros are uppercase; local matrix-index macros are normally uppercase | Working consensus with named legacy exceptions |

The lower-case helper structs `stride_64t`, `dims64_t`, and
`submat_spec_64t` are established public exceptions.
They do not overturn the broader type pattern.

The library uses BLAS++ vocabulary throughout: the scan found 127
`blas::Layout` references and 128 `blas::Op` references.
It found 953 `int64_t` references and 174 `randblas_require(...)` calls in
headers.
These counts support the existing API pattern: explicit layout/operation
flags, signed 64-bit dimensions and strides, constrained signed index types,
and validation near public entry points.

Concepts are used directly in template parameter lists when the compiler
supports them.
The `SignedInteger`, `SketchingOperator`, `SketchingDistribution`, and
`SparseMatrix` compatibility macros preserve builds where the relevant
concept feature test is unavailable.
New templates should reuse those concepts instead of introducing parallel
SFINAE machinery.

### Includes and preprocessor code

Project headers normally precede third-party and standard-library headers.
Conditional OpenMP includes use
`#if defined(RandBLAS_HAS_OpenMP)`, followed by `<omp.h>`.
The library contains ten OpenMP pragmas.

Indented preprocessor directives are common inside function bodies: the scan
found 106 in library headers, 14 in tests, and 16 in examples.
They are mostly short-lived matrix-index macros or compiler branches.
The pattern is deliberate enough to document, but these macros should be
undefined as soon as their local job is complete.

Macro naming has two domains.
Public/compiler configuration names are uppercase or carry a recognizable
project prefix (`RandBLAS_HAS_OpenMP`, `RandBLAS_OPTIMIZE_OFF`).
Function-like validation macros keep their existing lower-case API spelling
(`randblas_require`, `randblas_error_if`).
The lower-case `matA` macros in `RandBLAS/util.hh` are legacy local names, not
a convention for new macros.

### Source documentation

Nineteen of 32 headers contain `///` documentation; six contain at least one
`/** ... */` block.
Across C++ files, the scan found 3,225 `///` lines and 17 block-comment
openers.
Only two headers use `@file`, so that tag is optional rather than a file
template requirement.

All 77 detected parameter direction annotations use `@param[in]`.
The public docs also use `@tparam`, `@returns`, and project math commands such
as `\math{...}`.
This establishes `///` plus explicit Doxygen fields as the preferred form for
public contracts.
Implementation comments are most useful when they explain an invariant,
dispatch choice, numerical concern, or performance tradeoff.

Web documentation is written in reStructuredText.
The current source has 62 underlined headings, 272 inline `:math:` roles, and
six note/warning directives.
Repository notes are Markdown and contain 90 fenced-code markers and 121 ATX
headings.
The two formats have distinct jobs: public tutorials/API pages belong under
`rtd/source/`, while implementation rationale belongs in the nearest
`DevNotes.md`.

### Tests

The test corpus contains 62 fixture declarations, 455 `TEST_F` definitions,
and four plain `TEST` definitions.
Fixture classes normally begin with `Test`; test names are normally
snake_case descriptions of behavior or parameter combinations.

The scan found 141 `EXPECT_*` and 76 `ASSERT_*` uses.
The counts do not imply that one family is preferred everywhere.
Use `ASSERT_*` when later statements cannot run safely after failure; use
`EXPECT_*` when independent checks can still provide useful information.

Tests are organized by abstraction level, not by source filename alone:
basic RNG behavior, data structures, low-level linear operations, wrapper
APIs, and testing helpers live in separate subdirectories.
This organization agrees with `test/DevNotes.md` and is more important than
minor brace differences between old test files.

### CMake, Python, and automation

CMake is the only non-C++ stratum large enough for useful mechanical counts.
Of 400 detected command invocations, 398 use lower-case command names.
Four-space continuation/block indentation appears on 224 lines, compared
with 18 two-space lines.
Project-facing variables retain their established spelling
(`RandBLAS_HAS_OpenMP`, `BUILD_TESTS`); new private helpers use a leading
`_rb_` prefix, as in `CMake/rb_summary.cmake`.

Only three Python files are tracked.
They use four-space indentation but mix quote, import, and whitespace styles;
eight Python lines contain a tab or trailing whitespace.
The JavaScript and CSS strata each contain one file.
The guide therefore gives small-language recommendations rather than claims
of statistical consensus: use four spaces in Python, preserve the local
language's conventional formatter shape, and follow the surrounding file in
single-file strata.

Shell, PowerShell, YAML, and workflow files are similarly task-specific.
They should favor explicit commands and platform terminology already used by
the installation and CI files.
No cross-language indentation rule is inferred from them.

## Outlier analysis

An outlier is a file that departs from its peers in several independent
ways.
This is stricter than finding a typo.

For each eligible convention, the analyzer computed

```text
deviation_rate = nonconforming eligible occurrences / eligible occurrences
convention_weight = dominant convention share within the stratum
file_score = sum(deviation_rate * convention_weight)
             / sum(applicable convention_weight)
```

The scored features were tab use, license placement, include spacing,
duplicate includes, template spacing, namespace brace placement, inheritance
spacing, header protection, and project-include form.
A feature entered a stratum's score only when its dominant form reached the
70% working-consensus threshold.
Trailing whitespace and line length were not scored because the repository
is mixed on both.

A file became a candidate when it was in the top 10% of its stratum with a
score of at least 0.20, or when it violated two strong structural
conventions.
We then read the file, its log, and the blame for the relevant lines.

### Ranked register

| Rank | File | Stratum | Score | Strong deviations | Provenance | Classification | Recommended action |
|---:|---|---|---:|---|---|---|---|
| 1 | `test/linops/test_sparse_trsm.cc` | Tests | 0.330 | No project license, all sampled templates omit the space, mixed quote/angle project includes, one tabbed line | Most formatting dates to the sparse TRSM tests added in 2025; later changes added coverage without normalizing the file | Cleanup candidate | Normalize in a focused test cleanup; do not mix with kernel changes |
| 2 | `test/test_io.cc` | Tests | 0.314 | No project license, `TestIO: public` omits spaces; include groups are split by declarations | Introduced in 2024 and extended in 2026 | Cleanup candidate | Add the license and normalize fixture/include layout when the tests are made assertion-based |
| 3 | `RandBLAS/sparse_data/csr_trsm_impl.hh` | Support headers | 0.238 | `#pragma once` appears before and after the license | Both directives arrived with the original 2025 sparse TRSM implementation | Cleanup candidate | Remove the leading duplicate so the license is first and one guard remains |
| 4 | `test/basic_rng/test_r123.cc` | Tests | 0.211 | Upstream license, tabs, macro-heavy compatibility code, duplicate includes, mixed spacing and project-include form | Adapted from official Random123 tests; `test/DevNotes.md` explicitly calls it “extremely messy” because the upstream suite supports more compilers and languages | Intentional exception | Preserve upstream-shaped regions; require a narrow reason for local changes and avoid drive-by formatting |
| 5 | `test/downstream/main.cc` | Tests | 0.205 | No project license is its only scored deviation | Added in 2026 as a deliberately tiny installed-package smoke test | Not an outlier | Adding the license is reasonable, but the file does not exhibit a cluster of style deviations |
| 6 | `RandBLAS.hh` | Public/core | 0.193 | Uses an include guard rather than `#pragma once` | The umbrella header and installed-path angle includes date to the original 2022 layout | Intentional exception | Preserve the installed umbrella-header convention; new component headers use `#pragma once` |
| 7 | `RandBLAS/sparse_data/csc_trsm_impl.hh` | Support headers | 0.157 | `#pragma once` precedes the license; one of three templates omits the space | Added beside the CSR implementation in 2025 | Not an outlier | Move the guard after the license in a focused cleanup; the evidence does not classify the whole file as an outlier |
| 8 | `RandBLAS/random_gen.hh` | Public/core | 0.144 | The `r123ext` namespace and several short bodies use next-line braces | The block originated with the Random123 extension code in 2022 | Not an outlier | Follow the guide in new code; avoid reformatting this compatibility block without a functional reason |
| 9 | `test/datastructures/test_denseskop.cc` | Tests | 0.107 | Half of eligible templates omit the space; one fixture uses `public::testing::Test` | The deviations accumulated across several feature and test-infrastructure changes | Not an outlier | Fix the localized spacing defects when editing those tests |
| 10 | `RandBLAS/base.hh` | Public/core | 0.078 | Duplicate `<iostream>` include, one malformed `#include<iostream>`, three unspaced templates | The duplicate include is legacy code from 2022 and survived later API work | Cleanup candidate | Remove the duplicate and normalize only the affected declarations in a focused cleanup |
| 11 | `RandBLAS/util.hh` | Public/core | 0.001 | Two tabbed lines; otherwise its scored forms follow the stratum | Utility functions and local matrix-index macros have evolved since 2022 | Not an outlier | Treat local macros as implementation devices, not a competing project style |

`RandBLAS/base.hh` enters the register through the secondary structural rule,
not its aggregate score: the duplicate include and malformed include spacing
violate two strong include conventions.

### Rejected example seeds

The three sparse-low-rank example candidates each scored 0.000 on eligible
features:

- `examples/sparse-low-rank-approx/qrcp_matrixmarket.cc`;
- `examples/sparse-low-rank-approx/svd_rank1_plus_noise.cc`;
- `examples/sparse-low-rank-approx/svd_matrixmarket.cc`.

Their timing and matrix-index macros came from the same May 2024 example
work and serve local application code.
Long lines, trailing whitespace, and duplicate includes are common enough in
the seven-file example stratum that they do not distinguish these files as
outliers.
The guide still recommends avoiding duplicate includes and cleaning trailing
whitespace in new edits.

### Result

The audit identifies four cleanup candidates, one outlying intentional
exception, and one established umbrella-header exception.
It also identifies several localized defects in otherwise ordinary files.
That distinction is useful: a future cleanup can be small and reviewable,
while the Random123-derived test retains the compatibility shape that
explains its difference.

## Guide traceability

The following table traces the guide's normative rules to repository evidence.
“Project requirement” means the rule comes from `AGENTS.md` or a DevNotes file
rather than a mechanical style count.
Recommendations are directions for new work, not claims that old files already
agree.

| Guide section | Rule set | Evidence | Confidence | Exceptions or limits |
|---|---|---|---|---|
| 2 | Preserve correctness, ownership, thread safety, and public API compatibility during style changes | `AGENTS.md`; `RandBLAS/DevNotes.md`; [close-reading method](#close-reading-anchors) | Project requirement | None; this is the safety boundary for cleanup work |
| 2 | Keep sampled matrices independent of thread count | `AGENTS.md`; `RandBLAS/DevNotes.md`; `test/datastructures/test_denseskop.cc` | Project requirement | None |
| 2 | Follow neighboring BLAS-like parameter order | [naming and API profile](#naming-and-api-shape) | Working consensus | Match the nearest operation because the signatures serve different kernels |
| 2, 9 | Benchmark kernel optimizations and record the relevant configuration | `AGENTS.md`; `RandBLAS/sparse_data/DevNotes.md` | Project requirement | Applies to performance changes, not formatting-only work |
| 2, 8 | Use comments for contracts, invariants, dispatch, numerical concerns, and performance rationale | [source-documentation profile](#source-documentation); `RandBLAS/DevNotes.md` | Working consensus | Short local comments may still label data or a compact transformation |
| 3 | Put the license first, use one `#pragma once`, and group project, third-party, then standard includes | [C++ structure and spacing](#c-structure-and-spacing); [include profile](#includes-and-preprocessor-code) | Strong consensus for license/guard/include syntax; working consensus for group order | `RandBLAS.hh` keeps its include guard and installed-path angle includes |
| 4 | Use four spaces, avoid tabs, add spaces to `template <...>` and inheritance, and place opening braces on the same line | [C++ structure and spacing](#c-structure-and-spacing); [whitespace profile](#whitespace-and-line-length) | Strong consensus, except template spacing in tests is working consensus | Existing test/type brace placement is mixed; use the rule for new code |
| 4 | Brace multi-statement or nested bodies; allow a clear, short single statement to follow local form | [C++ structure and spacing](#c-structure-and-spacing) | Correctness rule plus mixed local evidence | The guide deliberately does not require all single statements to be braced |
| 4 | Bind pointers and references to the declarator and remove trailing whitespace on touched lines | [pointer/reference profile](#pointer-and-reference-binding); [whitespace profile](#whitespace-and-line-length) | Recommendation for pointers and trailing whitespace; working consensus for references | Do not reformat untouched code solely to apply these recommendations |
| 5 | Use the documented naming table; keep local macros uppercase, narrow, and explicitly undefined | [naming profile](#naming-and-api-shape); [preprocessor profile](#includes-and-preprocessor-code) | Strong or working consensus by entity | Named public lower-case types and validation macros remain established exceptions |
| 6 | Reuse C++20 concepts and BLAS++ enum types; use `int64_t` dimensions/strides and validate public inputs | [naming and API profile](#naming-and-api-shape); `AGENTS.md` | Project requirement backed by strong occurrence counts | Sparse index buffers may use a constrained signed index type |
| 6 | Format long signatures one logical parameter per line and preserve documented ownership flags | [close-reading anchors](#close-reading-anchors); `RandBLAS/sparse_data/DevNotes.md` | Working convention for signatures; project requirement for ownership | Compact signatures may stay on one line when they remain readable |
| 7 | Make component headers self-contained; include direct dependencies; keep the OpenMP include conditional | [include profile](#includes-and-preprocessor-code); build model in `AGENTS.md` | Build requirement for direct dependencies; strong local convention for OpenMP | The installed umbrella header follows its documented exception |
| 8 | Prefer `///` public contracts with Doxygen fields and keep implementation rationale in DevNotes | [source-documentation profile](#source-documentation) | Strong consensus for `///`; working convention for field coverage | `@file` is optional; `@param[out]` extends the repository's direction syntax to output buffers |
| 9 | Put OpenMP pragmas next to their loops and cover every affected sparse format/transpose/layout path | [preprocessor profile](#includes-and-preprocessor-code); `AGENTS.md`; `RandBLAS/sparse_data/DevNotes.md`; `test/DevNotes.md` | Working convention for pragma placement; project requirement for path coverage | Coverage is limited to paths affected by the change |
| 10 | Use fixtures for shared setup, choose `ASSERT_*` for unsafe continuation, and use `EXPECT_*` for independent checks | [test profile](#tests) | Strong fixture convention; semantic GoogleTest rule | A plain `TEST` remains suitable when no shared setup exists |
| 10 | Give RNG changes deterministic checks and test new sketch operations across their applicable sides, transposes, layouts, submatrices, and formats | `AGENTS.md`; `test/DevNotes.md`; [test profile](#tests) | Project requirement | Only applicable combinations need coverage |
| 11 | Keep examples focused on a use or benchmark and keep local helper macros narrow | [example anchors](#close-reading-anchors); [rejected example seeds](#rejected-example-seeds) | Working editorial rule | Example-local timing and index macros are permitted |
| 11 | Use lower-case CMake commands, four-space blocks, established public variable spellings, and `_rb_` for private helpers | [CMake profile](#cmake-python-and-automation) | Strong for command case; working for indentation and helper names | Existing public/cache names retain their spelling |
| 11 | Use four spaces in Python; otherwise follow local form in small language strata and prefer explicit automation | [CMake, Python, and automation](#cmake-python-and-automation) | Recommendation | No repository-wide quote rule is inferred from three Python files |
| 12 | Preserve the two documented exceptions and do not copy their special forms into ordinary component headers or tests | [outlier register](#ranked-register) | Explicit exception | Applies only to `RandBLAS.hh` and upstream-shaped regions of `test_r123.cc` |

## Rubric dry run

We applied the guide to five pairs without editing the source files.
The dry run asks whether the guide distinguishes a strong rule from a
recommendation, whether deviations cluster, and whether provenance supplies a
narrow exception.

| Stratum and pair | Guide result | Outlier result | Ambiguity exposed |
|---|---|---|---|
| Core: `RandBLAS/dense_skops.hh` / `RandBLAS/base.hh` | Both contain old local spacing and documentation forms. `base.hh` also has a duplicate `<iostream>` and malformed `#include<iostream>`, which violate two strong include rules. | `base.hh` is a focused cleanup candidate; `dense_skops.hh` is ordinary core code with localized debt. | A recommendation such as pointer binding cannot classify a file by itself. |
| Sparse: `csr_spmm_impl.hh` / `csr_trsm_impl.hh` | The SpMM header follows the structural rules apart from minor include grouping. The TRSM header puts `#pragma once` before the license and repeats it after the license. | Only `csr_trsm_impl.hh` is a cleanup candidate. | None: the duplicate protection is a strong, mechanical defect. |
| Tests: `test_lskge3.cc` / `test_denseskop.cc` | Both show older test-brace forms. `test_denseskop.cc` adds several unspaced templates and one malformed inheritance clause. | `test_denseskop.cc` has localized defects but is not an outlier; neither file has the independent cluster needed for that label. | Test brace placement is mixed, so the guide governs new code without retroactively condemning either file. |
| Examples: `tls_dense_skop.cc` / `svd_rank1_plus_noise.cc` | Both implement a recognizable numerical use and contain example-local rough edges. The SVD example's duplicate `<chrono>` is a cleanup item, while its index macro is allowed when kept local. | Neither example is an outlier; the seeded SVD example scored 0.000 on eligible stratum rules. | “Teach a real use” is an editorial review question, not a mechanical outlier feature. |
| Other languages: `CMake/rb_config.cmake` / `rtd/source/conf.py` | The CMake file follows the reliable command-case and block-indentation rules. The Python file has mixed import and quote forms, but the guide makes those local-style questions. | Neither file is an outlier. | The small Python corpus supports an indentation recommendation, not a repository-wide formatter or quote rule. |

The dry run produced no rule that reverses an evidence-based outlier
classification.
It did expose two useful boundaries: recommendations cannot create an outlier
on their own, and mixed test/Python evidence must remain local guidance.
