# RandBLAS style guide

RandBLAS is a header-only numerical library.
Its style exists to make templated, performance-sensitive code reviewable
without hiding the constraints that make the code correct.

This guide applies to new and substantially modified first-party files.
It describes the repository at commit `952251c` and makes a few explicit
recommendations where the repository is mixed.
See [`STYLE_AUDIT.md`](STYLE_AUDIT.md) for counts, counterexamples, and the
outlier register.

The words **must**, **should**, and **may** are deliberate.
“Must” marks a correctness or project requirement.
“Should” marks established house style.
“Recommendation” chooses a direction where the current files do not agree.

## 1. Purpose and scope

Use this guide for library headers, tests, examples, CMake, documentation,
and project automation.
Match the local file when a language has too little evidence for a
repository-wide rule.

Existing deviations do not create a second style.
Do not combine a functional change with a broad cleanup merely because this
guide makes the cleanup easy to see.
The [outlier audit](STYLE_AUDIT.md#outlier-analysis) records good candidates
for later, focused work.

## 2. Guiding principles

1. **Correctness comes first.** Numerical behavior, memory ownership,
   thread safety, and public API compatibility must survive a style change.
2. **Random generation is reproducible.** Sampling code must produce the
   same random matrix regardless of the number of OpenMP threads.
3. **BLAS-like APIs stay recognizable.** Layout, operation, side, dimension,
   stride, and scaling arguments should follow the order used by neighboring
   RandBLAS entry points.
4. **Performance claims need measurements.** A kernel optimization must be
   benchmarked before and after the change.
5. **Comments carry the reason.** Code states what happens; comments should
   explain invariants, dispatch choices, numerical concerns, or performance
   tradeoffs.
6. **Changes stay reviewable.** Prefer a focused patch over incidental
   reformatting of an entire file.

These principles come from `AGENTS.md`, `RandBLAS/DevNotes.md`, and the
sparse/testing DevNotes.

## 3. File layout

C++ source and component headers should use this order:

1. the project license block;
2. one `#pragma once` for a header;
3. the file-level Doxygen comment, when useful;
4. RandBLAS includes;
5. third-party includes;
6. standard-library includes;
7. declarations and definitions inside the narrowest useful namespace.

The library-header evidence is summarized under
[C++ structure and spacing](STYLE_AUDIT.md#c-structure-and-spacing).

```cpp
// Copyright, 2024. See LICENSE for copyright holder information.
// ...the complete project license block...

#pragma once

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"

#include <blas.hh>

#include <cstdint>
#include <vector>
```

Do not duplicate an include or write the directive without a separating
space:

```cpp
// Avoid.
#include <iostream>
#include<iostream>
```

`RandBLAS.hh` is an intentional exception.
It is the installed umbrella header, retains its include guard, and includes
installed `RandBLAS/` paths with angle brackets.
New component headers should use one `#pragma once`.

## 4. C++ formatting

Indent blocks with four spaces and do not introduce tabs.
Put an opening brace on the same line as a namespace, type, function, test
macro, or control statement.
The repository follows this form in 92.3% of measured brace occurrences.

```cpp
namespace RandBLAS::sparse_data {

template <typename T>
void scale(int64_t n, T alpha, T *x) {
    for (int64_t i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
}

}  // namespace RandBLAS::sparse_data
```

Avoid the following form in new code:

```cpp
template<typename T>
void scale(int64_t n, T alpha, T *x)
{
    // A tab was used for this indentation in the form being avoided.
    for (int64_t i = 0; i < n; ++i)
        x[i] *= alpha;
}
```

Use a space in `template <...>` and around an inheritance colon:

```cpp
template <typename T, SignedInteger sint_t>
class TestSparse : public ::testing::Test {
};
```

Braces must enclose a body with multiple statements or nested control flow.
A short single-statement body may remain unbraced when it is clear and agrees
with the surrounding code.

Recommendation: bind `*` and `&` to the declarator in new code.
Pointer binding is mixed, while `T &ref` has a clear majority.

```cpp
void apply(
    int64_t n,
    const T *input,
    T *output,
    const RNGState<> &state
) {
    // ...
}
```

Recommendation: remove trailing whitespace on touched lines.
There is no hard line-length limit.
Wrap prose, signatures, and expressions when the result is easier to read;
do not damage a formula, URL, or compact tabular signature to satisfy an
arbitrary column count.
The mixed evidence is recorded under
[Whitespace and line length](STYLE_AUDIT.md#whitespace-and-line-length).

## 5. Naming

Use the following names for new interfaces:

| Entity | Form | Examples |
|---|---|---|
| Top-level namespace | `RandBLAS` | `RandBLAS` |
| Nested namespace | lower-case or snake_case | `dense`, `sparse_data` |
| Public type or concept | PascalCase | `DenseSkOp`, `CSRMatrix`, `SparseMatrix` |
| Function | snake_case | `sketch_general`, `left_spmm` |
| Variable or field | snake_case | `n_rows`, `next_state` |
| Principal template type | short semantic uppercase | `T`, `RNG`, `SKOP` |
| Index/state alias | lower-case with `_t` | `sint_t`, `state_t` |
| Configuration/compiler macro | uppercase with a project prefix when public | `RandBLAS_HAS_OpenMP` |
| Test fixture | PascalCase beginning with `Test` | `TestDenseMoments` |
| Test name | snake_case behavior or case | `submatrix_a_double_colmajor` |

The evidence and established exceptions are listed under
[Naming and API shape](STYLE_AUDIT.md#naming-and-api-shape).

Keep the established BLAS-like abbreviations in APIs such as `lskge3`,
`rskges`, and `left_spmm`.
When a name's “left” or “right” is easy to misread, document which operand
occupies that position.

Local matrix-index macros should be uppercase.
Public validation macros retain their existing lower-case spellings.
A local macro must not escape the region that needs it; `#undef` it after its
last use.

## 6. Public APIs and templates

RandBLAS requires C++20.
Templates should express existing constraints with `SignedInteger`,
`SketchingOperator`, `SketchingDistribution`, or `SparseMatrix` rather than
new SFINAE machinery.
Keep the compatibility branches that define those names as `typename` when
the relevant concept feature test is absent.

Public dense and sparse operations should use BLAS++ types such as
`blas::Layout`, `blas::Op`, `blas::Side`, `blas::Uplo`, and `blas::Diag`.
Dimensions and strides should use `int64_t`; sparse index buffers may use a
`SignedInteger` template parameter.

Match the parameter order of the nearest public operation.
A long signature should put one logical parameter on each line and keep the
closing parenthesis aligned with the declaration:

```cpp
template <SparseMatrix SpMat, typename T = SpMat::scalar_t>
void left_spmm(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opB,
    int64_t m,
    int64_t n,
    int64_t k,
    T alpha,
    const SpMat &A,
    const T *B,
    int64_t ldb,
    T beta,
    T *C,
    int64_t ldc
) {
    // ...
}
```

Public entry points must validate dimensions, strides, offsets, enum values,
and structural assumptions before a kernel relies on them.
Use the existing `randblas_require` and error helpers.

Ownership must be visible in a type's contract.
Sparse matrix views and owning matrices share representations, so constructors
and destructors must preserve the documented ownership flags.

## 7. Includes and dependencies

Every component header should compile when included on its own with the
project's configured dependencies.
Include what the file uses rather than depending on incidental transitive
includes.

Within a component header, use repository-qualified quotes for project files
and angle brackets for BLAS++, Random123, OpenMP, and standard headers:

```cpp
#include "RandBLAS/base.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"

#include <blas.hh>

#include <algorithm>
#include <vector>
```

Keep OpenMP's header conditional:

```cpp
#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif
```

Do not add a new BLAS++ dependency casually.
The project currently relies on a small subset of BLAS operations while using
its enums broadly.
Discuss an expansion before it becomes part of the public dependency surface.

## 8. Comments and documentation

Public contracts should use `///` Doxygen comments.
Document template parameters, input/output direction, dimensional
requirements, ownership, state advancement, and return values.

```cpp
/// Fill a dense sketching operator without mutating `state`.
///
/// @tparam T   Matrix scalar type.
/// @tparam RNG Random123 counter-based generator type.
/// @param[in] dist  Distribution and dimensions to sample.
/// @param[out] buff Destination matrix buffer.
/// @param[in] state Initial random state.
/// @returns the state immediately after the sampled matrix.
```

The repository strongly favors `///` over `/** ... */` for header prose.
The `@file` tag is optional.
See [Source documentation](STYLE_AUDIT.md#source-documentation).

Do not narrate obvious syntax.
Comments should explain facts a reader cannot recover locally, such as:

- why random output cannot depend on thread count;
- how a CSR/CSC transpose view changes dispatch;
- why a loop order is faster for one layout;
- which matrix a “left” or “right” name describes;
- whether floating-point reassociation is intentional.

Put public tutorials and API pages under `rtd/source/` in reStructuredText.
Use `:math:` and the project's math macros consistently.
Put implementation rationale in the nearest `DevNotes.md`, and update it when
a dispatch or ownership design changes.

## 9. Parallel and performance-sensitive code

Sampling code must remain independent of `OMP_NUM_THREADS`.
Partition counter ranges by logical matrix location, not by a thread's
arrival order.
Add or run thread-count-independence tests whenever sampling changes.

OpenMP pragmas should sit immediately before the loop they control.
Use explicit scheduling or private/reduction clauses when correctness or
reproducibility depends on them:

```cpp
#pragma omp parallel for schedule(static)
for (int64_t row = 0; row < n_rows; ++row) {
    // Each row receives a deterministic counter range.
}
```

Sparse dispatch changes must account for matrix format, transpose flags, and
dense layout.
`left_spmm` has twelve principal paths, and `right_spmm` transforms into that
dispatcher.
Tests must cover every affected path rather than only the easiest storage
format.

Ask for agreement before undertaking a performance optimization.
Benchmark the old and new implementation under comparable settings, then
record the result and relevant BLAS/OpenMP configuration in the change
description.

## 10. Tests

Put a test at the same abstraction level as the behavior:

- basic RNG and statistical behavior in `test/basic_rng/`;
- matrix and sketch-operator types in `test/datastructures/`;
- low-level and wrapper operations in `test/linops/`;
- reusable test support in `RandBLAS/testing/` or a nearby helper header.

Use a `Test...` fixture and a snake_case case name when shared setup is
useful:

```cpp
class TestDenseSampling : public ::testing::Test {
};

TEST_F(TestDenseSampling, output_is_thread_count_independent) {
    // Arrange, act, and compare deterministic buffers.
}
```

Use `ASSERT_*` when failure makes later statements unsafe.
Use `EXPECT_*` for independent comparisons that can continue after one
failure.

RNG changes must have deterministic reference checks and, when distributional
behavior changes, appropriate statistical tests.
New sketching operations should exercise left and right application,
transposition, layouts, submatrices, and relevant sparse formats.

After a code change, run the focused test while iterating and the full suite
before completion:

```bash
cd /path/to/randnla/dev
source sourceme.sh
ctest --test-dir build-randblas --output-on-failure
```

## 11. Examples, CMake, Python, and automation

Examples should teach a real RandBLAS use or measure a named kernel.
Keep argument parsing, data preparation, correctness checks, and timing code
separate enough that a reader can find the RandBLAS call.
Local timing or matrix-index macros may be used when they make the numerical
code clearer, but they should be narrow and explicitly undefined.

CMake commands should be lower-case and blocks should use four-space
indentation.
Retain the spelling of public/cache variables such as `BUILD_TESTS` and
`RandBLAS_HAS_OpenMP`.
Prefix new private helper variables and functions with `_rb_`.
The [CMake profile](STYLE_AUDIT.md#cmake-python-and-automation) found 398 of
400 commands in lower case.

Recommendation: use four-space indentation and conventional import grouping
in Python.
The repository has only three Python files and does not establish a reliable
quote-style rule.

Shell, PowerShell, YAML, JavaScript, and CSS should follow their file's local
shape.
Automation should favor explicit platform and dependency names over clever
shell compression.

## 12. Intentional exceptions

Two files have narrow, documented reasons to differ:

- `RandBLAS.hh` is the installed umbrella header.
  It keeps its include guard and angle-bracket installed paths.
- `test/basic_rng/test_r123.cc` is adapted from the official Random123 test
  suite.
  Its upstream license, macro-heavy compatibility code, and several spacing
  choices should not be copied into ordinary RandBLAS tests.

An exception may preserve upstream provenance, generated syntax, a public
compatibility surface, or a measured kernel requirement.
Document the reason next to the code or in the nearest DevNotes.
Keep the exception as small as possible.

## 13. Contributor checklist

Before requesting review, check the following:

- [ ] The change uses four-space indentation, same-line braces, and
      `template <...>` spacing in new C++.
- [ ] New headers carry the project license, one `#pragma once`, and their
      direct dependencies.
- [ ] Public names, BLAS++ flags, dimensions, strides, and validation match
      neighboring APIs.
- [ ] Doxygen and DevNotes describe any changed contract, ownership rule, or
      dispatch design.
- [ ] Sampling output is unchanged across OpenMP thread counts.
- [ ] Sparse changes cover every affected format/transpose/layout path.
- [ ] Focused tests cover the behavior and failure modes.
- [ ] Performance changes include comparable before/after measurements.
- [ ] `source sourceme.sh; ctest --test-dir build-randblas` passes from the
      RandNLA workspace.
- [ ] The patch contains no unrelated formatting sweep.
