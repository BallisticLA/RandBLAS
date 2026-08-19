# RandBLAS style guide

RandBLAS is a header-only numerical library.
This guide records the conventions for writing its source and documentation.
It applies to new and substantially modified code; it is not a reason for
drive-by reformatting.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for project constraints, development
setup, tests, and review preparation.
See [`STYLE_AUDIT.md`](STYLE_AUDIT.md) for the repository evidence behind the
guide.
When this guide is silent, follow the surrounding file.

## C++ source

### Formatting

Use four spaces for indentation and do not introduce tabs.
Put opening braces on the same line as functions, types, namespaces, tests,
and control statements.
Write a space in `template <...>` and around an inheritance colon.

```cpp
template <typename T>
void scale(int64_t n, T alpha, T *x) {
    for (int64_t i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
}
```

The repository does not have a settled rule for spacing around `*` and `&`.
Follow the declaration you are modifying.

There is no fixed line-length limit.
Wrap prose, signatures, and expressions when doing so makes them easier to
read; do not damage a formula or compact tabular code to hit a column count.
Do not wrap a function signature merely because it has several parameters.
Keep the parameters together when the declaration fits comfortably and remains
easy to read. Put one logical parameter on each line only when that layout is
necessary to make a long or semantically dense signature readable. In that
case, put the closing parenthesis on its own line.

### Headers and includes

Put the project license first in a new header, followed by one `#pragma once`.
Include the headers a file uses rather than relying on transitive includes.
Group RandBLAS, third-party, and standard-library headers in that order.

Use repository-qualified quotes for RandBLAS component headers and angle
brackets for external headers:

```cpp
#include "RandBLAS/base.hh"

#include <blas.hh>

#include <cstdint>
```

`RandBLAS.hh` is the installed umbrella header.
It intentionally keeps an include guard and uses installed `RandBLAS/` paths
in angle brackets.

### Naming

| Entity | Form | Examples |
|---|---|---|
| Namespace | `RandBLAS`; lower-case nested names | `RandBLAS::sparse_data` |
| Public type or concept | PascalCase | `DenseSkOp`, `SparseMatrix` |
| Function | snake_case | `sketch_general`, `left_spmm` |
| Variable or field | snake_case | `n_rows`, `next_state` |
| Principal template type | short semantic uppercase | `T`, `RNG`, `SKOP` |
| Index or state alias | lower-case with `_t` | `sint_t`, `state_t` |

Keep established BLAS-like abbreviations such as `lskge3`, `rskges`, and
`left_spmm`.
When “left” or “right” is easy to misread, document which operand the name
describes.

### Public interface conventions

RandBLAS requires C++20.
Reuse its `SignedInteger`, `SketchingOperator`, `SketchingDistribution`, and
`SparseMatrix` concepts rather than adding parallel constraint machinery.

Use BLAS++ types such as `blas::Layout`, `blas::Op`, `blas::Side`,
`blas::Uplo`, and `blas::Diag` in BLAS-like APIs.
Dimensions and strides use `int64_t`; sparse index buffers may use a
`SignedInteger` template parameter.
Match the parameter order of the nearest public operation.

Make ownership explicit in the contract of an owning object or view.

## Documentation

RandBLAS's website is built with Sphinx and Breathe; it is not a Doxygen
website.
The build has three stages:

1. Doxygen parses the C++ declarations and writes XML.
2. Breathe reads that XML for directives such as `doxygenfunction` in the
   API-reference pages.
3. Sphinx renders the reStructuredText pages and the reStructuredText embedded
   in C++ comments.

Doxygen is therefore an extraction layer.
Write public documentation for the Sphinx output that readers actually see.

### Public C++ comments

Use `///` comments for declarations included in the web API reference.
For a function with a few arguments, explain the contract in plain prose.
Do not add a field for every parameter merely because Doxygen supports one.

Use a parameter dropdown only when an interface has many parameters whose
precise meanings have complicated relationships with one another. Put the
structured reStructuredText inside
`@verbatim embed:rst:leading-slashes`. List each parameter name, or a natural
group of related names, followed by indented bullets. State entry and exit
behavior in those bullets when the direction matters:

```cpp
// =============================================================================
/// Sample a matrix window into caller-owned storage.
///
/// @verbatim embed:rst:leading-slashes
/// .. dropdown:: Full parameter descriptions
///    :animate: fade-in-slide-down
///
///      n_rows, n_cols
///       - The dimensions of the window.
///
///      row_offset, col_offset
///       - The position of the window in the full matrix.
///
///      nnz
///       - On exit: the number of entries written to ``values``.
///
///      values
///       - A caller-owned buffer with enough capacity for the requested window.
/// @endverbatim
void sample_window(
    int64_t n_rows, int64_t n_cols,
    int64_t row_offset, int64_t col_offset,
    int64_t &nnz, double *values
) {
    // ...
}
```

Do not put a blank documentation line between `@endverbatim` and the
declaration.
Do not use `@tparam`, `@param`, `@return`, or `@returns` for new web-facing
comments; their Breathe rendering is not the project's chosen presentation.
Existing uses are legacy examples, not a second convention.

Outside an embedded reStructuredText block, write inline mathematics with the
project's `\math{...}` alias.
Put the surrounding sentence punctuation inside the alias: `\math{n > 0.}`
Inside an embedded block or an `.rst` file, use reStructuredText math roles and
directives.

A comment may define a local math macro at the start of an embedded block:

```cpp
/// @verbatim embed:rst:leading-slashes
///
///   .. |vals| mathmacro:: \mathtt{vals}
///
/// @endverbatim
```

Prefer a definition under `rtd/source/` when several API entries or pages need
the same macro.

Use a line of equals signs to separate major documented declarations and a
line of hyphens for members within a type.
Use `// MARK: description` to make a long source or test file easier to
navigate.

### Other documentation

Tutorials, policy prose, and API-reference composition belong under
`rtd/source/` and use reStructuredText.
Implementation rationale belongs in the nearest `DevNotes.md`.
Update the relevant API-reference directive when a public declaration changes.

## Other files

Use a `Test...` fixture when several tests share setup, and give test cases
snake_case names.
For CMake and the repository's smaller language strata, follow the local file.
Use lower-case CMake command names.
