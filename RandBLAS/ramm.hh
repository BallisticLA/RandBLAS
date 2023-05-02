#ifndef randblas_ramm_hh
#define randblas_ramm_hh

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/dense.hh"
#include "RandBLAS/sparse.hh"

#include <blas.hh>

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <string>

#include <math.h>
#include <typeinfo>

namespace RandBLAS::ramm {

using namespace RandBLAS::base;
using namespace RandBLAS::dense;
using namespace RandBLAS::sparse;


// =============================================================================
/// \fn ramm_general_left(blas::Layout layout, blas::Op transS, blas::Op transA, int64_t d,
///     int64_t n, int64_t m, T alpha, SKOP &S, int64_t i_os, int64_t j_os,
///     const T *A, int64_t lda, T beta, T *B, int64_t ldb
/// ) 
/// @verbatim embed:rst:leading-slashes
///
///   .. |op| mathmacro:: \operatorname{op}
///   .. |mat| mathmacro:: \operatorname{mat}
///   .. |submat| mathmacro:: \operatorname{submat}
///   .. |lda| mathmacro:: \mathrm{lda}
///   .. |ldb| mathmacro:: \mathrm{ldb}
///   .. |transA| mathmacro:: \mathrm{transA}
///   .. |transS| mathmacro:: \mathrm{transS}
///
/// @endverbatim
/// ramm_general_left: Perform a GEMM-like operation
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\submat(S))}_{d \times m} \cdot \underbrace{\op(\mat(A))}_{m \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars, \math{\op(X)} either returns a matrix \math{X}
/// or its transpose, and \math{S} is a sketching operator.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(A)` and :math:`\mat(B)`?
///     Their shapes are defined implicitly by :math:`(d, m, n, \transA)`.
///     Their precise contents are determined by :math:`(A, \lda)`, :math:`(B, \ldb)`,
///     and "layout", following the same convention as BLAS.
///
/// What is :math:`\submat(S)`?
///     Its shape is defined implicitly by :math:`(\transS, d, m)`.
///     If :math:`{\submat(S)}` is of shape :math:`r \times c`,
///     then it is the :math:`r \times c` submatrix of :math:`{S}` whose upper-left corner
///     appears at index :math:`(\texttt{i_os}, \texttt{j_os})` of :math:`{S}`.
/// @endverbatim
/// @param[in] layout
///     Layout::ColMajor or Layout::RowMajor
///      - Matrix storage for \math{\mat(A)} and \math{\mat(B)}.
///
/// @param[in] transS
///      - If \math{\transS} = NoTrans, then \math{ \op(\submat(S)) = \submat(S)}.
///      - If \math{\transS} = Trans, then \math{\op(\submat(S)) = \submat(S)^T }.
/// @param[in] transA
///      - If \math{\transA} == NoTrans, then \math{\op(\mat(A)) = \mat(A)}.
///      - If \math{\transA} == Trans, then \math{\op(\mat(A)) = \mat(A)^T}.
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
/// @param[in] i_os
///     A nonnegative integer.
///     - The rows of \math{\submat(S)} are a contiguous subset of rows of \math{S}.
///     - The rows of \math{\submat(S)} start at \math{S[\texttt{i_os}, :]}.
///
/// @param[in] j_os
///     A nonnnegative integer.
///     - The columns of \math{\submat(S)} are a contiguous subset of columns of \math{S}.
///     - The columns \math{\submat(S)} start at \math{S[:,\texttt{j_os}]}. 
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
void ramm_general_left(
    blas::Layout layout,
    blas::Op transS,
    blas::Op transA,
    int64_t d, // B is d-by-n
    int64_t n, // \op(A) is m-by-n
    int64_t m, // \op(\submat(S)) is d-by-m
    T alpha,
    SKOP &S,
    int64_t i_os,
    int64_t j_os,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
);

template <typename T, typename RNG>
void ramm_general_left(
    blas::Layout layout,
    blas::Op transS,
    blas::Op transA,
    int64_t d, // B is d-by-n
    int64_t n, // \op(A) is m-by-n
    int64_t m, // \op(\submat(S)) is d-by-m
    T alpha,
    sparse::SparseSkOp<T, RNG> &S,
    int64_t row_offset,
    int64_t col_offset,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
) {
    return sparse::lskges(
        layout, transS, transA, d, n, m, alpha, S,
        row_offset, col_offset, A, lda, beta, B, ldb
    );
}

template <typename T, typename RNG>
void ramm_general_left(
    blas::Layout layout,
    blas::Op transS,
    blas::Op transA,
    int64_t d, // B is d-by-n
    int64_t n, // \op(A) is m-by-n
    int64_t m, // \op(\submat(S)) is d-by-m
    T alpha,
    dense::DenseSkOp<T, RNG> &S,
    int64_t row_offset,
    int64_t col_offset,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
) {
    return dense::lskge3(
        layout, transS, transA, d, n, m, alpha, S,
        row_offset, col_offset, A, lda, beta, B, ldb
    );
}

// =============================================================================
/// \fn ramm_general_right(blas::Layout layout, blas::Op transA, blas::Op transS, int64_t m, int64_t d, int64_t n,
///    T alpha, const T *A, int64_t lda, SKOP &S,
///    int64_t i_os, int64_t j_os, T beta, T *B, int64_t ldb
/// )
/// ramm_general_right: Perform a GEMM-like operation
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\mat(A))}_{m \times n} \cdot \underbrace{\op(\submat(S))}_{n \times d} + \beta \cdot \underbrace{\mat(B)}_{m \times d},    \tag{$\star$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars, \math{\op(X)} either returns a matrix \math{X}
/// or its transpose, and \math{S} is a sketching operator.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(A)` and :math:`\mat(B)`?
///     Their shapes are defined implicitly by :math:`(m, d, n, \transA)`.
///     Their precise contents are determined by :math:`(A, \lda)`, :math:`(B, \ldb)`,
///     and "layout", following the same convention as BLAS.
///
/// What is :math:`\submat(S)`?
///     Its shape is defined implicitly by :math:`(\transS, n, d)`.
///     If :math:`{\submat(S)}` is of shape :math:`r \times c`,
///     then it is the :math:`r \times c` submatrix of :math:`{S}` whose upper-left corner
///     appears at index :math:`(\texttt{i_os}, \texttt{j_os})` of :math:`{S}`.
/// @endverbatim
/// @param[in] layout
///     Layout::ColMajor or Layout::RowMajor
///      - Matrix storage for \math{\mat(A)} and \math{\mat(B)}.
///
/// @param[in] transA
///      - If \math{\transA} == NoTrans, then \math{\op(\mat(A)) = \mat(A)}.
///      - If \math{\transA} == Trans, then \math{\op(\mat(A)) = \mat(A)^T}.
///
/// @param[in] transS
///      - If \math{\transS} = NoTrans, then \math{ \op(\submat(S)) = \submat(S)}.
///      - If \math{\transS} = Trans, then \math{\op(\submat(S)) = \submat(S)^T }.
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
/// @param[in] i_os
///     A nonnegative integer.
///     - The rows of \math{\submat(S)} are a contiguous subset of rows of \math{S}.
///     - The rows of \math{\submat(S)} start at \math{S[\texttt{i_os}, :]}.
///
/// @param[in] j_os
///     A nonnnegative integer.
///     - The columns of \math{\submat(S)} are a contiguous subset of columns of \math{S}.
///     - The columns \math{\submat(S)} start at \math{S[:,\texttt{j_os}]}. 
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
void ramm_general_right(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transS,
    int64_t m, // B is m-by-d
    int64_t d, // op(\submat(S)) is n-by-d
    int64_t n, // op(A) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    SKOP &S,
    int64_t i_os,
    int64_t j_os,
    T beta,
    T *B,
    int64_t ldb
);

template <typename T, typename RNG>
void ramm_general_right(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transS,
    int64_t m, // B is m-by-d
    int64_t d, // op(\submat(S)) is n-by-d
    int64_t n, // op(A) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    dense::DenseSkOp<T, RNG> &S,
    int64_t i_os,
    int64_t j_os,
    T beta,
    T *B,
    int64_t ldb
) {
    return dense::rskge3(layout, transA, transS, m, d, n, alpha, A, lda,
        S, i_os, j_os, beta, B, ldb
    );
}


template <typename T, typename RNG>
void ramm_general_right(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transS,
    int64_t m, // B is m-by-d
    int64_t d, // op(\submat(S)) is n-by-d
    int64_t n, // op(A) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    sparse::SparseSkOp<T, RNG> &S,
    int64_t i_os,
    int64_t j_os,
    T beta,
    T *B,
    int64_t ldb
) {
    return sparse::rskges(layout, transA, transS, m, d, n, alpha, A, lda,
        S, i_os, j_os, beta, B, ldb
    );
}

}  // end namespace RandBLAS::ramm
#endif 