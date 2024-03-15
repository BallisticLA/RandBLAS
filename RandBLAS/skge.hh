#ifndef randblas_skge_hh
#define randblas_skge_hh

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/dense.hh"
#include "RandBLAS/sparse_skops.hh"

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <string>

#include <math.h>
#include <typeinfo>

namespace RandBLAS {

using namespace RandBLAS::dense;
using namespace RandBLAS::sparse;


// =============================================================================
/// \fn sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d,
///     int64_t n, int64_t m, T alpha, SKOP &S, int64_t s_ro, int64_t s_co,
///     const T *A, int64_t lda, T beta, T *B, int64_t ldb
/// ) 
/// @verbatim embed:rst:leading-slashes
///
///   .. |op| mathmacro:: \operatorname{op}
///   .. |mat| mathmacro:: \operatorname{mat}
///   .. |submat| mathmacro:: \operatorname{submat}
///   .. |lda| mathmacro:: \mathrm{lda}
///   .. |ldb| mathmacro:: \mathrm{ldb}
///   .. |opA| mathmacro:: \mathrm{opA}
///   .. |opS| mathmacro:: \mathrm{opS}
///
/// @endverbatim
/// Perform a GEMM-like operation
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\submat(S))}_{d \times m} \cdot \underbrace{\op(\mat(A))}_{m \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars, \math{\op(X)} either returns a matrix \math{X}
/// or its transpose, and \math{S} is a sketching operator.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(A)` and :math:`\mat(B)`?
///     Their shapes are defined implicitly by :math:`(d, m, n, \opA)`.
///     Their precise contents are determined by :math:`(A, \lda)`, :math:`(B, \ldb)`,
///     and "layout", following the same convention as the Level 3 BLAS function "GEMM."
///
/// What is :math:`\submat(S)`?
///     Its shape is defined implicitly by :math:`(\opS, d, m)`.
///     If :math:`{\submat(S)}` is of shape :math:`r \times c`,
///     then it is the :math:`r \times c` submatrix of :math:`{S}` whose upper-left corner
///     appears at index :math:`(\texttt{s_ro}, \texttt{s_co})` of :math:`{S}`.
/// @endverbatim
/// @param[in] layout
///     Layout::ColMajor or Layout::RowMajor
///      - Matrix storage for \math{\mat(A)} and \math{\mat(B)}.
///
/// @param[in] opS
///      - If \math{\opS} = NoTrans, then \math{ \op(\submat(S)) = \submat(S)}.
///      - If \math{\opS} = Trans, then \math{\op(\submat(S)) = \submat(S)^T }.
///
/// @param[in] opA
///      - If \math{\opA} == NoTrans, then \math{\op(\mat(A)) = \mat(A)}.
///      - If \math{\opA} == Trans, then \math{\op(\mat(A)) = \mat(A)^T}.
///
/// @param[in] d
///     A nonnegative integer.
///     - The number of rows in \math{\mat(B)}
///     - The number of rows in \math{\op(\mat(S))}.
///
/// @param[in] n
///     A nonnegative integer.
///     - The number of columns in \math{\mat(B)}
///     - The number of columns in \math{\op(\mat(A))}.
///
/// @param[in] m
///     A nonnegative integer.
///     - The number of columns in \math{\op(\submat(S))}
///     - The number of rows in \math{\op(\mat(A))}.
///
/// @param[in] alpha
///     A real scalar.
///     - If zero, then \math{A} is not accessed.
///
/// @param[in] S
///    A DenseSkOp or SparseSkOp object.
///    - Defines \math{\submat(S)}.
///
/// @param[in] s_ro
///     A nonnegative integer.
///     - The rows of \math{\submat(S)} are a contiguous subset of rows of \math{S}.
///     - The rows of \math{\submat(S)} start at \math{S[\texttt{s_ro}, :]}.
///
/// @param[in] s_co
///     A nonnnegative integer.
///     - The columns of \math{\submat(S)} are a contiguous subset of columns of \math{S}.
///     - The columns \math{\submat(S)} start at \math{S[:,\texttt{s_co}]}. 
///
/// @param[in] A
///     Pointer to a 1D array of real scalars.
///     - Defines \math{\mat(A)}.
///
/// @param[in] lda
///     A nonnegative integer.
///     * Leading dimension of \math{\mat(A)} when reading from \math{A}.
///     * If layout == ColMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i + j \cdot \lda].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a column in \math{\mat(A)}.
///     * If layout == RowMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i \cdot \lda + j].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a row in \math{\mat(A)}.
///
/// @param[in] beta
///     A real scalar.
///     - If zero, then \math{B} need not be set on input.
///
/// @param[in, out] B
///    Pointer to 1D array of real scalars.
///    - On entry, defines \math{\mat(B)}
///      on the RIGHT-hand side of \math{(\star)}.
///    - On exit, defines \math{\mat(B)}
///      on the LEFT-hand side of \math{(\star)}.
///
/// @param[in] ldb
///    - Leading dimension of \math{\mat(B)} when reading from \math{B}.
///    - Refer to documentation for \math{\lda} for details. 
///
template <typename T, typename SKOP>
void sketch_general(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(submat(S)) is d-by-m
    T alpha,
    SKOP &S,
    int64_t s_ro,
    int64_t s_co,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
);

template <typename T, typename RNG>
void sketch_general(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(submat(S)) is d-by-m
    T alpha,
    SparseSkOp<T, RNG> &S,
    int64_t s_ro,
    int64_t s_co,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
) {
    return sparse::lskges(
        layout, opS, opA, d, n, m, alpha, S,
        s_ro, s_co, A, lda, beta, B, ldb
    );
}

template <typename T, typename RNG>
void sketch_general(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(submat(S)) is d-by-m
    T alpha,
    DenseSkOp<T, RNG> &S,
    int64_t s_ro,
    int64_t s_co,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
) {
    return dense::lskge3(
        layout, opS, opA, d, n, m, alpha, S,
        s_ro, s_co, A, lda, beta, B, ldb
    );
}

// =============================================================================
/// \fn sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n,
///    T alpha, const T *A, int64_t lda, SKOP &S,
///    int64_t s_ro, int64_t s_co, T beta, T *B, int64_t ldb
/// )
/// Perform a GEMM-like operation
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\mat(A))}_{m \times n} \cdot \underbrace{\op(\submat(S))}_{n \times d} + \beta \cdot \underbrace{\mat(B)}_{m \times d},    \tag{$\star$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars, \math{\op(X)} either returns a matrix \math{X}
/// or its transpose, and \math{S} is a sketching operator.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(A)` and :math:`\mat(B)`?
///     Their shapes are defined implicitly by :math:`(m, d, n, \opA)`.
///     Their precise contents are determined by :math:`(A, \lda)`, :math:`(B, \ldb)`,
///     and "layout", following the same convention as the Level 3 BLAS function "GEMM."
///
/// What is :math:`\submat(S)`?
///     Its shape is defined implicitly by :math:`(\opS, n, d)`.
///     If :math:`{\submat(S)}` is of shape :math:`r \times c`,
///     then it is the :math:`r \times c` submatrix of :math:`{S}` whose upper-left corner
///     appears at index :math:`(\texttt{s_ro}, \texttt{s_co})` of :math:`{S}`.
/// @endverbatim
/// @param[in] layout
///     Layout::ColMajor or Layout::RowMajor
///      - Matrix storage for \math{\mat(A)} and \math{\mat(B)}.
///
/// @param[in] opA
///      - If \math{\opA} == NoTrans, then \math{\op(\mat(A)) = \mat(A)}.
///      - If \math{\opA} == Trans, then \math{\op(\mat(A)) = \mat(A)^T}.
///
/// @param[in] opS
///      - If \math{\opS} = NoTrans, then \math{ \op(\submat(S)) = \submat(S)}.
///      - If \math{\opS} = Trans, then \math{\op(\submat(S)) = \submat(S)^T }.
///
/// @param[in] m
///     A nonnegative integer.
///     - The number of rows in \math{\mat(B)}.
///     - The number of rows in \math{\op(\mat(A))}.
///
/// @param[in] d
///     A nonnegative integer.
///     - The number of columns in \math{\mat(B)}
///     - The number of columns in \math{\op(\mat(S))}.
///
/// @param[in] n
///     A nonnegative integer.
///     - The number of columns in \math{\op(\mat(A))}
///     - The number of rows in \math{\op(\submat(S))}.
///
/// @param[in] alpha
///     A real scalar.
///     - If zero, then \math{A} is not accessed.
///
/// @param[in] A
///     Pointer to a 1D array of real scalars.
///     - Defines \math{\mat(A)}.
///
/// @param[in] lda
///     A nonnegative integer.
///     * Leading dimension of \math{\mat(A)} when reading from \math{A}.
///     * If layout == ColMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i + j \cdot \lda].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a column in \math{\mat(A)}.
///     * If layout == RowMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i \cdot \lda + j].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a row in \math{\mat(A)}.
///
/// @param[in] S
///    A DenseSkOp or SparseSkOp object.
///    - Defines \math{\submat(S)}.
///
/// @param[in] s_ro
///     A nonnegative integer.
///     - The rows of \math{\submat(S)} are a contiguous subset of rows of \math{S}.
///     - The rows of \math{\submat(S)} start at \math{S[\texttt{s_ro}, :]}.
///
/// @param[in] s_co
///     A nonnnegative integer.
///     - The columns of \math{\submat(S)} are a contiguous subset of columns of \math{S}.
///     - The columns \math{\submat(S)} start at \math{S[:,\texttt{s_co}]}. 
///
/// @param[in] beta
///     A real scalar.
///     - If zero, then \math{B} need not be set on input.
///
/// @param[in, out] B
///    Pointer to 1D array of real scalars.
///    - On entry, defines \math{\mat(B)}
///      on the RIGHT-hand side of \math{(\star)}.
///    - On exit, defines \math{\mat(B)}
///      on the LEFT-hand side of \math{(\star)}.
///
/// @param[in] ldb
///    - Leading dimension of \math{\mat(B)} when reading from \math{B}.
///    - Refer to documentation for \math{\lda} for details. 
///
template <typename T, typename SKOP>
void sketch_general(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opS,
    int64_t m, // B is m-by-d
    int64_t d, // op(submat(S)) is n-by-d
    int64_t n, // op(A) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    SKOP &S,
    int64_t s_ro,
    int64_t s_co,
    T beta,
    T *B,
    int64_t ldb
);

template <typename T, typename RNG>
void sketch_general(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opS,
    int64_t m, // B is m-by-d
    int64_t d, // op(submat(S)) is n-by-d
    int64_t n, // op(A) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    DenseSkOp<T, RNG> &S,
    int64_t s_ro,
    int64_t s_co,
    T beta,
    T *B,
    int64_t ldb
) {
    return dense::rskge3(layout, opA, opS, m, d, n, alpha, A, lda,
        S, s_ro, s_co, beta, B, ldb
    );
}


template <typename T, typename RNG>
void sketch_general(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opS,
    int64_t m, // B is m-by-d
    int64_t d, // op(submat(S)) is n-by-d
    int64_t n, // op(A) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    SparseSkOp<T, RNG> &S,
    int64_t s_ro,
    int64_t s_co,
    T beta,
    T *B,
    int64_t ldb
) {
    return sparse::rskges(layout, opA, opS, m, d, n, alpha, A, lda,
        S, s_ro, s_co, beta, B, ldb
    );
}

// =============================================================================
/// \fn sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d,
///     int64_t n, int64_t m, T alpha, SKOP &S, const T *A, int64_t lda, T beta, T *B, int64_t ldb
/// ) 
/// @verbatim embed:rst:leading-slashes
///
///   .. |op| mathmacro:: \operatorname{op}
///   .. |mat| mathmacro:: \operatorname{mat}
///   .. |lda| mathmacro:: \mathrm{lda}
///   .. |ldb| mathmacro:: \mathrm{ldb}
///   .. |opA| mathmacro:: \mathrm{opA}
///   .. |opS| mathmacro:: \mathrm{opS}
///
/// @endverbatim
/// Perform a GEMM-like operation
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(S)}_{d \times m} \cdot \underbrace{\op(\mat(A))}_{m \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars, \math{\op(X)} either returns a matrix \math{X}
/// or its transpose, and \math{S} is a sketching operator.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(A)` and :math:`\mat(B)`?
///     Their shapes are defined implicitly by :math:`(d, m, n, \opA)`.
///     Their precise contents are determined by :math:`(A, \lda)`, :math:`(B, \ldb)`,
///     and "layout", following the same convention as the Level 3 BLAS function "GEMM."
/// @endverbatim
/// @param[in] layout
///     Layout::ColMajor or Layout::RowMajor
///      - Matrix storage for \math{\mat(A)} and \math{\mat(B)}.
///
/// @param[in] opS
///      - If \math{\opS} = NoTrans, then \math{ \op(S) = S}.
///      - If \math{\opS} = Trans, then \math{\op(S) = S^T }.
///
/// @param[in] opA
///      - If \math{\opA} == NoTrans, then \math{\op(\mat(A)) = \mat(A)}.
///      - If \math{\opA} == Trans, then \math{\op(\mat(A)) = \mat(A)^T}.
///
/// @param[in] d
///     A nonnegative integer.
///     - The number of rows in \math{\mat(B)}
///     - The number of rows in \math{\op(\mat(S))}.
///
/// @param[in] n
///     A nonnegative integer.
///     - The number of columns in \math{\mat(B)}
///     - The number of columns in \math{\op(\mat(A))}.
///
/// @param[in] m
///     A nonnegative integer.
///     - The number of columns in \math{\op(S)}
///     - The number of rows in \math{\op(\mat(A))}.
///
/// @param[in] alpha
///     A real scalar.
///     - If zero, then \math{A} is not accessed.
///
/// @param[in] S
///    A DenseSkOp or SparseSkOp object.
///
/// @param[in] A
///     Pointer to a 1D array of real scalars.
///     - Defines \math{\mat(A)}.
///
/// @param[in] lda
///     A nonnegative integer.
///     * Leading dimension of \math{\mat(A)} when reading from \math{A}.
///     * If layout == ColMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i + j \cdot \lda].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a column in \math{\mat(A)}.
///     * If layout == RowMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i \cdot \lda + j].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a row in \math{\mat(A)}.
///
/// @param[in] beta
///     A real scalar.
///     - If zero, then \math{B} need not be set on input.
///
/// @param[in, out] B
///    Pointer to 1D array of real scalars.
///    - On entry, defines \math{\mat(B)}
///      on the RIGHT-hand side of \math{(\star)}.
///    - On exit, defines \math{\mat(B)}
///      on the LEFT-hand side of \math{(\star)}.
///
/// @param[in] ldb
///    - Leading dimension of \math{\mat(B)} when reading from \math{B}.
///    - Refer to documentation for \math{\lda} for details. 
///
template <typename T, typename SKOP>
void sketch_general(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(S) is d-by-m
    T alpha,
    SKOP &S,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
) {
    if (opS == blas::Op::NoTrans) {
        randblas_require(S.dist.n_rows == d);
        randblas_require(S.dist.n_cols == m);
    } else {
        randblas_require(S.dist.n_rows == m);
        randblas_require(S.dist.n_cols == d);
    }
    return sketch_general(layout, opS, opA, d, n, m, alpha, S, 0, 0, A, lda, beta, B, ldb);
};

// =============================================================================
/// \fn sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n,
///    T alpha, const T *A, int64_t lda, SKOP &S, T beta, T *B, int64_t ldb
/// )
/// Perform a GEMM-like operation
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\mat(A))}_{m \times n} \cdot \underbrace{\op(S)}_{n \times d} + \beta \cdot \underbrace{\mat(B)}_{m \times d},    \tag{$\star$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars, \math{\op(X)} either returns a matrix \math{X}
/// or its transpose, and \math{S} is a sketching operator.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(A)` and :math:`\mat(B)`?
///     Their shapes are defined implicitly by :math:`(m, d, n, \opA)`.
///     Their precise contents are determined by :math:`(A, \lda)`, :math:`(B, \ldb)`,
///     and "layout", following the same convention as the Level 3 BLAS function "GEMM."
/// @endverbatim
/// @param[in] layout
///     Layout::ColMajor or Layout::RowMajor
///      - Matrix storage for \math{\mat(A)} and \math{\mat(B)}.
///
/// @param[in] opA
///      - If \math{\opA} == NoTrans, then \math{\op(\mat(A)) = \mat(A)}.
///      - If \math{\opA} == Trans, then \math{\op(\mat(A)) = \mat(A)^T}.
///
/// @param[in] opS
///      - If \math{\opS} = NoTrans, then \math{ \op(S) = S}.
///      - If \math{\opS} = Trans, then \math{\op(S) = S^T }.
///
/// @param[in] m
///     A nonnegative integer.
///     - The number of rows in \math{\mat(B)}.
///     - The number of rows in \math{\op(\mat(A))}.
///
/// @param[in] d
///     A nonnegative integer.
///     - The number of columns in \math{\mat(B)}
///     - The number of columns in \math{\op(\mat(S))}.
///
/// @param[in] n
///     A nonnegative integer.
///     - The number of columns in \math{\op(\mat(A))}
///     - The number of rows in \math{\op(S)}.
///
/// @param[in] alpha
///     A real scalar.
///     - If zero, then \math{A} is not accessed.
///
/// @param[in] A
///     Pointer to a 1D array of real scalars.
///     - Defines \math{\mat(A)}.
///
/// @param[in] lda
///     A nonnegative integer.
///     * Leading dimension of \math{\mat(A)} when reading from \math{A}.
///     * If layout == ColMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i + j \cdot \lda].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a column in \math{\mat(A)}.
///     * If layout == RowMajor, then
///         @verbatim embed:rst:leading-slashes
///             .. math::
///                 \mat(A)[i, j] = A[i \cdot \lda + j].
///         @endverbatim
///       In this case, \math{\lda} must be \math{\geq} the length of a row in \math{\mat(A)}.
///
/// @param[in] S
///    A DenseSkOp or SparseSkOp object.
///    - Defines \math{S}.
///
/// @param[in] beta
///     A real scalar.
///     - If zero, then \math{B} need not be set on input.
///
/// @param[in, out] B
///    Pointer to 1D array of real scalars.
///    - On entry, defines \math{\mat(B)}
///      on the RIGHT-hand side of \math{(\star)}.
///    - On exit, defines \math{\mat(B)}
///      on the LEFT-hand side of \math{(\star)}.
///
/// @param[in] ldb
///    - Leading dimension of \math{\mat(B)} when reading from \math{B}.
///    - Refer to documentation for \math{\lda} for details. 
///
template <typename T, typename SKOP>
void sketch_general(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opS,
    int64_t m, // B is m-by-d
    int64_t d, // op(S) is n-by-d
    int64_t n, // op(A) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    SKOP &S,
    T beta,
    T *B,
    int64_t ldb
) {
    if (opS == blas::Op::NoTrans) {
        randblas_require(S.dist.n_rows == n);
        randblas_require(S.dist.n_cols == d);
    } else {
        randblas_require(S.dist.n_rows == d);
        randblas_require(S.dist.n_cols == n);
    }
    return sketch_general(layout, opA, opS, m, d, n, alpha, A, lda, S, 0, 0, beta, B, ldb);
};

// =============================================================================
/// \fn sketch_vector(blas::Op opS, int64_t d, int64_t m, T alpha, SKOP &S,
///    int64_t s_ro, int64_t s_co, const T *x, incx, T beta, T *y, incy
/// )
/// Perform a GEMV-like operation. If :math:`{\opS} = \texttt{NoTrans}`, then we perform
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(y) = \alpha \cdot \underbrace{\submat(S)}_{d \times m} \cdot \underbrace{\mat(x)}_{m \times 1} + \beta \cdot \underbrace{\mat(y)}_{d \times 1},    \tag{$\star$}
/// @endverbatim
/// otherwise, we perform
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(y) = \alpha \cdot \underbrace{\submat(S)^T}_{m \times d} \cdot \underbrace{\mat(x)}_{d \times 1} + \beta \cdot \underbrace{\mat(y)}_{m \times 1},    \tag{$\diamond$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars and \math{S} is a sketching operator.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(x)` and :math:`\mat(y)`?
///     Their shapes are defined as tall vectors of dimension :math:`(\mat(x), L_x \times 1)`, :math:`(\mat(y), L_y \times 1)`,
///     where :math:`(L_x, L_y)` are lengths so that :math:`\opS(\submat(S)) \mat(x)` is well-defined and the same shape as :math:`\mat(y)`. 
///     Their precise contents are determined in a way that is identical to the Level 2 BLAS function "GEMV."
///
/// Why no "layout" argument?
///     The GEMV in CBLAS accepts a parameter that specifies row-major or column-major layout of the matrix.
///     Since our matrix is a sketching operator, and since RandBLAS has no notion of the layout of a sketching operator, we do not have a layout parameter.
/// @endverbatim
///
/// @param[in] opS
///      - If \math{\opS} = NoTrans, then \math{ \op(\submat(S)) = \submat(S)}.
///      - If \math{\opS} = Trans, then \math{\op(\submat(S)) = \submat(S)^T }.
///
/// @param[in] d
///     A nonnegative integer.
///     - The number of rows in \math{\submat(S)}.
///
/// @param[in] m
///     A nonnegative integer.
///     - The number of columns in \math{\submat(S)}.
///
/// @param[in] alpha
///     A real scalar.
///     - If zero, then \math{x} is not accessed.
///     
/// @param[in] S
///    A DenseSkOp or SparseSkOp object.
///    - Defines \math{S}.
///
/// @param[in] s_ro
///     A nonnegative integer.
///     - \math{\submat(S)} is a contiguous submatrix of \math{S[\texttt{s_ro}:(\texttt{s_ro} + d), :]}.
///
/// @param[in] s_co
///     A nonnnegative integer.
///     - \math{\submat(S)} is a contiguous submatrix of \math{S[:,\texttt{s_co}:(\texttt{s_co} + m)]}. 
///
/// @param[in] x
///     Pointer to a 1D array of real scalars.
///     - Defines \math{\mat(x)}.
///
/// @param[in] incx
///     A nonnegative integer. 
///     * Stride between elements of x. incx must not be zero.
///     * RandBLAS currently does not support negative values for LDA, so incx cannot be negative unlike GEMV in the BLAS.
///
/// @param[in] beta
///     A real scalar.
///     - If zero, then \math{y} need not be set on input.
///
/// @param[in, out] y
///    Pointer to 1D array of real scalars.
///    - On entry, defines \math{\mat(y)} on the RIGHT-hand side of
///      \math{(\star)} (if \math{\opS = \texttt{NoTrans}}) or
///      \math{(\diamond)} (if \math{\opS = \texttt{Trans}})
///    - On exit, defines \math{\mat(y)} on the LEFT-hand side of the same.
///
/// @param[in] incy
///     A positive integer.
///     * Stride between elements of y. incy must not be zero.
///     * RandBLAS currently does not support negative values for LDA, so incy cannot be negative unlike GEMV in the BLAS.
///
template <typename T, typename SKOP>
void sketch_vector(
    blas::Op opS,
    int64_t d, // rows in submat(S)
    int64_t m, // cols in submat(S)
    T alpha,
    SKOP &S,
    int64_t s_ro,
    int64_t s_co,
    const T *x,
    int64_t incx,
    T beta,
    T *y,
    int64_t incy
) {
    int64_t _d, _m;
    if (opS == blas::Op::Trans) {
        _d = m;
        _m = d;
    } else {
        _d = d;
        _m = m;
    }
    return sketch_general(blas::Layout::RowMajor, opS, blas::Op::NoTrans, _d, 1, _m, alpha, S, s_ro, s_co, x, incx, beta, y, incy);
}

// =============================================================================
/// \fn sketch_vector(blas::Op opS, int64_t d, int64_t m, T alpha, SKOP &S,
///    int64_t s_ro, int64_t s_co, const T *x, incx, T beta, T *y, incy
/// )
/// Perform a GEMV-like operation. If :math:`{\opS} = \texttt{NoTrans}`, then we perform
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(y) = \alpha \cdot \underbrace{S}_{d \times m} \cdot \underbrace{\mat(x)}_{m \times 1} + \beta \cdot \underbrace{\mat(y)}_{d \times 1},    \tag{$\star$}
/// @endverbatim
/// otherwise, we perform
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(y) = \alpha \cdot \underbrace{S^T}_{m \times d} \cdot \underbrace{\mat(x)}_{d \times 1} + \beta \cdot \underbrace{\mat(y)}_{m \times 1},    \tag{$\diamond$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars and \math{S} is a sketching operator.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(x)` and :math:`\mat(y)`?
///     Their shapes are defined as tall vectors of dimension :math:`(\mat(x), L_x \times 1)`, :math:`(\mat(y), L_y \times 1)`,
///     where :math:`(L_x, L_y)` are lengths so that :math:`\opS(S) \mat(x)` is well-defined and the same shape as :math:`\mat(y)`. 
///     Their precise contents are determined in a way that is identical to the Level 2 BLAS function "GEMV."
///
/// Why no "layout" argument?
///     The GEMV in CBLAS accepts a parameter that specifies row-major or column-major layout of the matrix.
///     Since our matrix is a sketching operator, and since RandBLAS has no notion of the layout of a sketching operator, we do not have a layout parameter.
/// @endverbatim
///
/// @param[in] opS
///      - If \math{\opS} = NoTrans, then \math{ \op(S) = S}.
///      - If \math{\opS} = Trans, then \math{\op(S) = S^T }.
///
/// @param[in] alpha
///     A real scalar.
///     - If zero, then \math{x} is not accessed.
///     
/// @param[in] S
///    A DenseSkOp or SparseSkOp object.
///    - Defines \math{S}.
///
/// @param[in] x
///     Pointer to a 1D array of real scalars.
///     - Defines \math{\mat(x)}.
///
/// @param[in] incx
///     A nonnegative integer. 
///     * Stride between elements of x. incx must not be zero.
///     * RandBLAS currently does not support negative values for LDA, so incx cannot be negative unlike GEMV in the BLAS.
///
/// @param[in] beta
///     A real scalar.
///     - If zero, then \math{y} need not be set on input.
///
/// @param[in, out] y
///    Pointer to 1D array of real scalars.
///    - On entry, defines \math{\mat(y)} on the RIGHT-hand side of
///      \math{(\star)} (if \math{\opS = \texttt{NoTrans}}) or
///      \math{(\diamond)} (if \math{\opS = \texttt{Trans}})
///    - On exit, defines \math{\mat(y)} on the LEFT-hand side of the same.
///
/// @param[in] incy
///     A positive integer.
///     * Stride between elements of y. incy must not be zero.
///     * RandBLAS currently does not support negative values for LDA, so incy cannot be negative unlike GEMV in the BLAS.
///
template <typename T, typename SKOP>
void sketch_vector(
    blas::Op opS,
    T alpha,
    SKOP &S,
    const T *x,
    int64_t incx,
    T beta,
    T *y,
    int64_t incy
) {
    int64_t d = S.dist.n_rows;
    int64_t m = S.dist.n_cols;
    return sketch_vector(opS, d, m, alpha, S, 0, 0, x, incx, beta, y, incy);
}

}  // end namespace RandBLAS
#endif
