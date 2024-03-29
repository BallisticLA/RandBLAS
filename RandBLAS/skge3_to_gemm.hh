#ifndef randblas_skge3_to_gemm_hh
#define randblas_skge3_to_gemm_hh

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"

#include <blas.hh>

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <tuple>

#include <math.h>
#include <typeinfo>
#include "dense_skops.hh"


namespace RandBLAS::dense {

using RandBLAS::DenseSkOp;
using RandBLAS::fill_dense;

// =============================================================================
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
/// LSKGE3: Perform a GEMM-like operation
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\submat(S))}_{d \times m} \cdot \underbrace{\op(\mat(A))}_{m \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars, \math{\op(X)} either returns a matrix \math{X}
/// or its transpose, and \math{S} is a sketching operator that takes Level 3 BLAS effort to apply.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(A)` and :math:`\mat(B)`?
///     Their shapes are defined implicitly by :math:`(d, m, n, \opA)`.
///     Their precise contents are determined by :math:`(A, \lda)`, :math:`(B, \ldb)`,
///     and "layout", following the same convention as BLAS.
///
/// What is :math:`\submat(S)`?
///     Its shape is defined implicitly by :math:`(\opS, d, m)`.
///     If :math:`{\submat(S)}` is of shape :math:`r \times c`,
///     then it is the :math:`r \times c` submatrix of :math:`{S}` whose upper-left corner
///     appears at index :math:`(\texttt{ro_s}, \texttt{co_s})` of :math:`{S}`.
/// @endverbatim
/// @param[in] layout
///     Layout::ColMajor or Layout::RowMajor
///      - Matrix storage for \math{\mat(A)} and \math{\mat(B)}.
///
/// @param[in] opS
///      - If \math{\opS} = NoTrans, then \math{ \op(\submat(S)) = \submat(S)}.
///      - If \math{\opS} = Trans, then \math{\op(\submat(S)) = \submat(S)^T }.
/// @param[in] opA
///      - If \math{\opA} == NoTrans, then \math{\op(\mat(A)) = \mat(A)}.
///      - If \math{\opA} == Trans, then \math{\op(\mat(A)) = \mat(A)^T}.
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
///    A DenseSkOp object.
///    - Defines \math{\submat(S)}.
///
/// @param[in] ro_s
///     A nonnegative integer.
///     - The rows of \math{\submat(S)} are a contiguous subset of rows of \math{S}.
///     - The rows of \math{\submat(S)} start at \math{S[\texttt{ro_s}, :]}.
///
/// @param[in] co_s
///     A nonnnegative integer.
///     - The columns of \math{\submat(S)} are a contiguous subset of columns of \math{S}.
///     - The columns \math{\submat(S)} start at \math{S[:,\texttt{co_s}]}. 
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
template <typename T, typename RNG>
void lskge3(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(S) is d-by-m
    T alpha,
    DenseSkOp<T,RNG> &S,
    int64_t ro_s,
    int64_t co_s,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
){
    auto [rows_submat_S, cols_submat_S] = dims_before_op(d, m, opS);
    if (!S.buff) {
        // We'll make a shallow copy of the sketching operator, take responsibility for filling the memory
        // of that sketching operator, and then call LSKGE3 with that new object.
        T *buff = new T[rows_submat_S * cols_submat_S];
        fill_dense(S.dist, rows_submat_S, cols_submat_S, ro_s, co_s, buff, S.seed_state);
        DenseDist D{rows_submat_S, cols_submat_S, DenseDistName::BlackBox, S.dist.major_axis};
        DenseSkOp S_(D, S.seed_state, buff);
        lskge3(layout, opS, opA, d, n, m, alpha, S_, 0, 0, A, lda, beta, B, ldb);
        delete [] buff;
        return;
    }
    randblas_require( S.dist.n_rows >= rows_submat_S + ro_s );
    randblas_require( S.dist.n_cols >= cols_submat_S + co_s );
    auto [rows_A, cols_A] = dims_before_op(m, n, opA);
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= rows_A);
        randblas_require(ldb >= d);
    } else {
        randblas_require(lda >= cols_A);
        randblas_require(ldb >= n);
    }

    auto [pos, lds] = offset_and_ldim(S.layout, S.dist.n_rows, S.dist.n_cols, ro_s, co_s);
    T* S_ptr = &S.buff[pos];
    if (S.layout != layout)
        opS = (opS == blas::Op::NoTrans) ? blas::Op::Trans : blas::Op::NoTrans;

    blas::gemm(layout, opS, opA, d, n, m, alpha, S_ptr, lds, A, lda, beta, B, ldb);
    return;
}

// =============================================================================
/// RSKGE3: Perform a GEMM-like operation
/// @verbatim embed:rst:leading-slashes
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\mat(A))}_{m \times n} \cdot \underbrace{\op(\submat(S))}_{n \times d} + \beta \cdot \underbrace{\mat(B)}_{m \times d},    \tag{$\star$}
/// @endverbatim
/// where \math{\alpha} and \math{\beta} are real scalars, \math{\op(X)} either returns a matrix \math{X}
/// or its transpose, and \math{S} is a sketching operator that takes Level 3 BLAS effort to apply.
/// 
/// @verbatim embed:rst:leading-slashes
/// What are :math:`\mat(A)` and :math:`\mat(B)`?
///     Their shapes are defined implicitly by :math:`(m, d, n, \opA)`.
///     Their precise contents are determined by :math:`(A, \lda)`, :math:`(B, \ldb)`,
///     and "layout", following the same convention as BLAS.
///
/// What is :math:`\submat(S)`?
///     Its shape is defined implicitly by :math:`(\opS, n, d)`.
///     If :math:`{\submat(S)}` is of shape :math:`r \times c`,
///     then it is the :math:`r \times c` submatrix of :math:`{S}` whose upper-left corner
///     appears at index :math:`(\texttt{ro_s}, \texttt{co_s})` of :math:`{S}`.
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
///    A DenseSkOp object.
///    - Defines \math{\submat(S)}.
///
/// @param[in] ro_s
///     A nonnegative integer.
///     - The rows of \math{\submat(S)} are a contiguous subset of rows of \math{S}.
///     - The rows of \math{\submat(S)} start at \math{S[\texttt{ro_s}, :]}.
///
/// @param[in] co_s
///     A nonnnegative integer.
///     - The columns of \math{\submat(S)} are a contiguous subset of columns of \math{S}.
///     - The columns \math{\submat(S)} start at \math{S[:,\texttt{co_s}]}. 
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
template <typename T, typename RNG>
void rskge3(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opS,
    int64_t m, // B is m-by-d
    int64_t d, // op(S) is n-by-d
    int64_t n, // op(A) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    DenseSkOp<T,RNG> &S,
    int64_t ro_s,
    int64_t co_s,
    T beta,
    T *B,
    int64_t ldb
){
    auto [rows_submat_S, cols_submat_S] = dims_before_op(n, d, opS);
    if (!S.buff) {
        // We'll make a shallow copy of the sketching operator, take responsibility for filling the memory
        // of that sketching operator, and then call RSKGE3 with that new object.
        T *buff = new T[rows_submat_S * cols_submat_S];
        fill_dense(S.dist, rows_submat_S, cols_submat_S, ro_s, co_s, buff, S.seed_state);
        DenseDist D{rows_submat_S, cols_submat_S, DenseDistName::BlackBox, S.dist.major_axis};
        DenseSkOp S_(D, S.seed_state, buff);
        rskge3(layout, opA, opS, m, d, n, alpha, A, lda, S_, 0, 0, beta, B, ldb);
        delete [] buff;
        return;
    }
    randblas_require( S.dist.n_rows >= rows_submat_S + ro_s );
    randblas_require( S.dist.n_cols >= cols_submat_S + co_s );
    auto [rows_A, cols_A] = dims_before_op(m, n, opA);
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= rows_A);
        randblas_require(ldb >= m);
    } else {
        randblas_require(lda >= cols_A);
        randblas_require(ldb >= d);
    }

    auto [pos, lds] = offset_and_ldim(S.layout, S.dist.n_rows, S.dist.n_cols, ro_s, co_s);
    T* S_ptr = &S.buff[pos];
    if (S.layout != layout)
        opS = (opS == blas::Op::NoTrans) ? blas::Op::Trans : blas::Op::NoTrans;

    blas::gemm(layout, opA, opS, m, d, n, alpha, A, lda, S_ptr, lds, beta, B, ldb);
    return;
}

} // end namespace RandBLAS::dense

#endif
