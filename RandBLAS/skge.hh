// Copyright, 2024. See LICENSE for copyright holder information.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef randblas_skge_hh
#define randblas_skge_hh

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/random_gen.hh"
#include "RandBLAS/dense_skops.hh"
#include "RandBLAS/sparse_skops.hh"
#include "RandBLAS/skge3_to_gemm.hh"
#include "RandBLAS/skges_to_spmm.hh"

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <string>

#include <math.h>
#include <typeinfo>

namespace RandBLAS {

using namespace RandBLAS::dense;
using namespace RandBLAS::sparse;


/* Intended macro definitions.

   .. |op| mathmacro:: \operatorname{op}
   .. |mat| mathmacro:: \operatorname{mat}
   .. |submat| mathmacro:: \operatorname{submat}
   .. |lda| mathmacro:: \texttt{lda}
   .. |ldb| mathmacro:: \texttt{ldb}
   .. |opA| mathmacro:: \texttt{opA}
   .. |opS| mathmacro:: \texttt{opS}
*/


// MARK: SUBMAT(S), LEFT

// =============================================================================
/// \fn sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d,
///     int64_t n, int64_t m, T alpha, SKOP &S, int64_t ro_s, int64_t co_s,
///     const T *A, int64_t lda, T beta, T *B, int64_t ldb
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Sketch from the left in a GEMM-like operation
/// 
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\submat(S))}_{d \times m} \cdot \underbrace{\op(\mat(A))}_{m \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
/// 
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(X)` either returns a matrix :math:`X`
/// or its transpose, and :math:`S` is a sketching operator.
///
/// .. dropdown:: FAQ
///   :animate: fade-in-slide-down
///
///     **What are** :math:`\mat(A)` **and** :math:`\mat(B)` **?**
///
///       Their shapes are defined implicitly by :math:`(d, m, n, \opA).`
///       Their precise contents are determined by :math:`(A, \lda),` :math:`(B, \ldb),`
///       and "layout", following the same convention as GEMM from BLAS.
///
///       If layout == ColMajor, then
///
///             .. math::
///                 \mat(A)[i, j] = A[i + j \cdot \lda].
///
///       In this case, :math:`\lda` must be :math:`\geq` the length of a column in :math:`\mat(A).`
///
///       If layout == RowMajor, then
///
///             .. math::
///                 \mat(A)[i, j] = A[i \cdot \lda + j].
///
///       In this case, :math:`\lda` must be :math:`\geq` the length of a row in :math:`\mat(A).`
///
///     **What is** :math:`\submat(S)` **?**
///
///       Its shape is defined implicitly by :math:`(\opS, d, m).`
///
///       If :math:`{\submat(S)}` is of shape :math:`r \times c,`
///       then it is the :math:`r \times c` submatrix of :math:`{S}` whose upper-left corner
///       appears at index :math:`(\texttt{ro_s}, \texttt{co_s})` of :math:`{S}.`
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Either Layout::ColMajor or Layout::RowMajor
///       * Matrix storage for :math:`\mat(A)` and :math:`\mat(B).`
///
///      opS - [in]
///       * Either Op::Trans or Op::NoTrans.
///       * If :math:`\opS` = NoTrans, then :math:`\op(\submat(S)) = \submat(S).`
///       * If :math:`\opS` = Trans, then :math:`\op(\submat(S)) = \submat(S)^T.`
///
///      opA - [in]
///       * If :math:`\opA` == NoTrans, then :math:`\op(\mat(A)) = \mat(A).`
///       * If :math:`\opA` == Trans, then :math:`\op(\mat(A)) = \mat(A)^T.`
///
///      d - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(B)`
///       * The number of rows in :math:`\op(\submat(S)).`
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(B)`
///       * The number of columns in :math:`\op(\mat(A)).`
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\op(\submat(S))`
///       * The number of rows in :math:`\op(\mat(A)).`
///
///      alpha - [in]
///       * A real scalar.
///       * If zero, then :math:`A` is not accessed.
///
///      S - [in]  
///       * A DenseSkOp or SparseSkOp object.
///       * Defines :math:`\submat(S).`
///
///      ro_s - [in]
///       * A nonnegative integer.
///       * The rows of :math:`\submat(S)` are a contiguous subset of rows of :math:`S.`
///       * The rows of :math:`\submat(S)` start at :math:`S[\texttt{ro_s}, :].`
///
///      co_s - [in]
///       * A nonnnegative integer.
///       * The columns of :math:`\submat(S)` are a contiguous subset of columns of :math:`S.`
///       * The columns of :math:`\submat(S)` start at :math:`S[:,\texttt{co_s}].` 
///
///      A - [in]
///       * Pointer to a 1D array of real scalars.
///       * Defines :math:`\mat(A).`
///
///      lda - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(A)` when reading from :math:`A.`
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`B` need not be set on input.
///
///      B - [in,out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(B)`
///         on the RIGHT-hand side of :math:`(\star).`
///       * On exit, defines :math:`\mat(B)`
///         on the LEFT-hand side of :math:`(\star).`
///
///      ldb - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(B)` when reading from :math:`B.`
///
/// @endverbatim
template <typename T, typename SKOP>
inline void sketch_general(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(submat(S)) is d-by-m
    T alpha,
    SKOP &S,
    int64_t ro_s,
    int64_t co_s,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
);

template <typename T, typename RNG>
inline void sketch_general(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(submat(S)) is d-by-m
    T alpha,
    SparseSkOp<T, RNG> &S,
    int64_t ro_s,
    int64_t co_s,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
) {
    return sparse::lskges(
        layout, opS, opA, d, n, m, alpha, S,
        ro_s, co_s, A, lda, beta, B, ldb
    );
}

template <typename T, typename RNG>
inline void sketch_general(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(submat(S)) is d-by-m
    T alpha,
    DenseSkOp<T, RNG> &S,
    int64_t ro_s,
    int64_t co_s,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
) {
    return dense::lskge3(
        layout, opS, opA, d, n, m, alpha, S,
        ro_s, co_s, A, lda, beta, B, ldb
    );
}

// MARK: SUBMAT(S), RIGHT

// =============================================================================
/// \fn sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n,
///    T alpha, const T *A, int64_t lda, SKOP &S,
///    int64_t ro_s, int64_t co_s, T beta, T *B, int64_t ldb
/// )
/// @verbatim embed:rst:leading-slashes
/// Sketch from the right in a GEMM-like operation
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\mat(A))}_{m \times n} \cdot \underbrace{\op(\submat(S))}_{n \times d} + \beta \cdot \underbrace{\mat(B)}_{m \times d},    \tag{$\star$}
/// 
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(X)` either returns a matrix :math:`X`
/// or its transpose, and :math:`S` is a sketching operator.
/// 
/// .. dropdown:: FAQ
///   :animate: fade-in-slide-down
///
///     **What are** :math:`\mat(A)` **and** :math:`\mat(B)` **?**
///
///       Their shapes are defined implicitly by :math:`(m, d, n, \opA).`
///       Their precise contents are determined by :math:`(A, \lda),` :math:`(B, \ldb),`
///       and "layout", following the same convention as the Level 3 BLAS function "GEMM."
///
///     **What is** :math:`\submat(S)` **?**
///
///       Its shape is defined implicitly by :math:`(\opS, n, d).`
///       If :math:`{\submat(S)}` is of shape :math:`r \times c,`
///       then it is the :math:`r \times c` submatrix of :math:`{S}` whose upper-left corner
///       appears at index :math:`(\texttt{ro_s}, \texttt{co_s})` of :math:`{S}.`
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Either Layout::ColMajor or Layout::RowMajor
///       * Matrix storage for :math:`\mat(A)` and :math:`\mat(B).`
///
///      opA - [in]
///       * If :math:`\opA` == NoTrans, then :math:`\op(\mat(A)) = \mat(A).`
///       * If :math:`\opA` == Trans, then :math:`\op(\mat(A)) = \mat(A)^T.`
///
///      opS - [in]
///       * Either Op::Trans or Op::NoTrans.
///       * If :math:`\opS` = NoTrans, then :math:`\op(\submat(S)) = \submat(S).`
///       * If :math:`\opS` = Trans, then :math:`\op(\submat(S)) = \submat(S)^T.`
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(B).`
///       * The number of rows in :math:`\op(\mat(A)).`
///
///      d - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(B)`
///       * The number of columns in :math:`\op(\submat(S)).`
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\op(\mat(A)).`
///       * The number of rows in :math:`\op(\submat(S)).`
///
///      alpha - [in]
///       * A real scalar.
///       * If zero, then :math:`A` is not accessed.
///
///      A - [in]
///       * Pointer to a 1D array of real scalars.
///       * Defines :math:`\mat(A).`
///
///      lda - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(A)` when reading from :math:`A.`
///
///      S - [in]  
///       * A DenseSkOp or SparseSkOp object.
///       * Defines :math:`\submat(S).`
///       * Defines :math:`\submat(S).`
///
///      ro_s - [in]
///       * A nonnegative integer.
///       * The rows of :math:`\submat(S)` are a contiguous subset of rows of :math:`S.`
///       * The rows of :math:`\submat(S)` start at :math:`S[\texttt{ro_s}, :].`
///
///      co_s - [in]
///       * A nonnegative integer.
///       * The columns of :math:`\submat(S)` are a contiguous subset of columns of :math:`S.`
///       * The columns :math:`\submat(S)` start at :math:`S[:,\texttt{co_s}].` 
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`B` need not be set on input.
///
///      B - [in,out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(B)`
///         on the RIGHT-hand side of :math:`(\star).`
///       * On exit, defines :math:`\mat(B)`
///         on the LEFT-hand side of :math:`(\star).`
///
///      ldb - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(B)` when reading from :math:`B.`
///
/// @endverbatim
template <typename T, typename SKOP>
inline void sketch_general(
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
    int64_t ro_s,
    int64_t co_s,
    T beta,
    T *B,
    int64_t ldb
);

template <typename T, typename RNG>
inline void sketch_general(
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
    int64_t ro_s,
    int64_t co_s,
    T beta,
    T *B,
    int64_t ldb
) {
    return dense::rskge3(layout, opA, opS, m, d, n, alpha, A, lda,
        S, ro_s, co_s, beta, B, ldb
    );
}


template <typename T, typename RNG>
inline void sketch_general(
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
    int64_t ro_s,
    int64_t co_s,
    T beta,
    T *B,
    int64_t ldb
) {
    return sparse::rskges(layout, opA, opS, m, d, n, alpha, A, lda,
        S, ro_s, co_s, beta, B, ldb
    );
}


// MARK: FULL(S), LEFT

// =============================================================================
/// \fn sketch_general(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d,
///     int64_t n, int64_t m, T alpha, SKOP &S, const T *A, int64_t lda, T beta, T *B, int64_t ldb
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Sketch from the left in a GEMM-like operation
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(S)}_{d \times m} \cdot \underbrace{\op(\mat(A))}_{m \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(X)` either returns a matrix :math:`X`
/// or its transpose, and :math:`S` is a sketching operator.
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Either Layout::ColMajor or Layout::RowMajor
///       * Matrix storage for :math:`\mat(A)` and :math:`\mat(B).`
///
///      opS - [in]
///       * Either Op::Trans or Op::NoTrans.
///       * If :math:`\opS` = NoTrans, then :math:`\op(S) = S.`
///       * If :math:`\opS` = Trans, then :math:`\op(S) = S^T.`
///
///      opA - [in]
///       * If :math:`\opA` == NoTrans, then :math:`\op(\mat(A)) = \mat(A).`
///       * If :math:`\opA` == Trans, then :math:`\op(\mat(A)) = \mat(A)^T.`
///
///      d - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(B)`
///       * The number of rows in :math:`\op(\mat(S)).`
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(B)`
///       * The number of columns in :math:`\op(\mat(A)).`
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\op(S).`
///       * The number of rows in :math:`\op(\mat(A)).`
///
///      alpha - [in]
///       * A real scalar.
///       * If zero, then :math:`A` is not accessed.
///
///      S - [in]  
///       * A DenseSkOp or SparseSkOp object.
///       * Defines :math:`\submat(S).`
///
///      A - [in]
///       * Pointer to a 1D array of real scalars.
///       * Defines :math:`\mat(A).`
///
///      lda - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(A)` when reading from :math:`A.`
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`B` need not be set on input.
///
///      B - [in,out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(B)`
///         on the RIGHT-hand side of :math:`(\star).`
///       * On exit, defines :math:`\mat(B)`
///         on the LEFT-hand side of :math:`(\star).`
///
///      ldb - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(B)` when reading from :math:`B.`
///
/// @endverbatim
template <typename T, typename SKOP>
inline void sketch_general(
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

// MARK: FULL(S), RIGHT

// =============================================================================
/// \fn sketch_general(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n,
///    T alpha, const T *A, int64_t lda, SKOP &S, T beta, T *B, int64_t ldb
/// )
/// @verbatim embed:rst:leading-slashes
/// Sketch from the right in a GEMM-like operation
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\mat(A))}_{m \times n} \cdot \underbrace{\op(S)}_{n \times d} + \beta \cdot \underbrace{\mat(B)}_{m \times d},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(X)` either returns a matrix :math:`X`
/// or its transpose, and :math:`S` is a sketching operator.
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Either Layout::ColMajor or Layout::RowMajor
///       * Matrix storage for :math:`\mat(A)` and :math:`\mat(B).`
///
///      opA - [in]
///       * If :math:`\opA` == NoTrans, then :math:`\op(\mat(A)) = \mat(A).`
///       * If :math:`\opA` == Trans, then :math:`\op(\mat(A)) = \mat(A)^T.`
///
///      opS - [in]
///       * Either Op::Trans or Op::NoTrans.
///       * If :math:`\opS` = NoTrans, then :math:`\op(S) = S.`
///       * If :math:`\opS` = Trans, then :math:`\op(S) = S^T.`
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(B).`
///       * The number of rows in :math:`\op(\mat(A)).`
///
///      d - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(B).`
///       * The number of columns in :math:`\op(\mat(S)).`
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\op(\mat(A)).`
///       * The number of rows in :math:`\op(S).`
///
///      alpha - [in]
///       * A real scalar.
///       * If zero, then :math:`A` is not accessed.
///
///      A - [in]
///       * Pointer to a 1D array of real scalars.
///       * Defines :math:`\mat(A).`
///
///      lda - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(A)` when reading from :math:`A.`
///
///      S - [in]  
///       * A DenseSkOp or SparseSkOp object.
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`B` need not be set on input.
///
///      B - [in,out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(B)`
///         on the RIGHT-hand side of :math:`(\star).`
///       * On exit, defines :math:`\mat(B)`
///         on the LEFT-hand side of :math:`(\star).`
///
///      ldb - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(B)` when reading from :math:`B.`
///
/// @endverbatim
template <typename T, typename SKOP>
inline void sketch_general(
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

}  // end namespace RandBLAS
#endif
