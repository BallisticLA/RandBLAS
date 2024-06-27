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

#ifndef randblas_sksy_hh
#define randblas_sksy_hh

#include "RandBLAS/util.hh"
#include "RandBLAS/base.hh"
#include "RandBLAS/skge.hh"

namespace RandBLAS {

using namespace RandBLAS::dense;
using namespace RandBLAS::sparse;

/* Intended macro definitions.

   .. |mat| mathmacro:: \operatorname{mat}
   .. |submat| mathmacro:: \operatorname{submat}
   .. |lda| mathmacro:: \texttt{lda}
   .. |ldb| mathmacro:: \texttt{ldb}
*/

// MARK: SUBMAT(S)

// =============================================================================
/// \fn sketch_symmetric(blas::Layout layout, int64_t n,
///     int64_t d, T alpha,  const T *A, int64_t lda,
///     SKOP &S, int64_t ro_s, int64_t co_s,
///     T beta, T *B, int64_t ldb, T sym_check_tol = 0
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Check that :math:`\mat(A)` is symmetric up to tolerance :math:`\texttt{sym_check_tol}`, then sketch from the right in a SYMM-like operation
/// 
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\mat(A)}_{n \times n} \cdot \underbrace{\submat(S)}_{n \times d}  + \beta \cdot \underbrace{\mat(B)}_{n \times d},    \tag{$\star$}
/// 
/// where :math:`\alpha` and :math:`\beta` are real scalars and :math:`S` is a sketching operator.
///
/// .. dropdown:: FAQ
///   :animate: fade-in-slide-down
///
///     **What's** :math:`\mat(A)?`
///
///       It's a symmetric matrix of order :math:`n`. Its precise contents depend on :math:`(A, \lda)`,
///       according to 
///
///             .. math::
///                 \mat(A)[i, j] = A[i + j \cdot \lda] = A[i \cdot \lda + j].
///
///       Note that the the "layout" parameter passed to this function is not used here.
///       That's because this function requires :math:`\mat(A)` to be stored in the format
///       of a general matrix (with both upper and lower triangles).
///
///       This function's default behavior is to check that :math:`\mat(A)` is symmetric before
///       attempting sketching. That check  can be skipped (at your own peril!) by calling this
///       function with sym_check_tol < 0.
///
///     **What's** :math:`\mat(B)?`
///
///       It's an :math:`n \times d` matrix.  Its precise contents depend on :math:`(B,\ldb)` and "layout."
///
///       If layout == ColMajor, then
///
///             .. math::
///                 \mat(B)[i, j] = B[i + j \cdot \ldb].
///
///       In this case, :math:`\ldb` must be :math:`\geq n.`
///
///       If layout == RowMajor, then
///
///             .. math::
///                 \mat(B)[i, j] = B[i \cdot \ldb + j].
///
///       In this case, :math:`\ldb` must be :math:`\geq d.`
///
///     **What is** :math:`\submat(S)` **?**
///
///       It's the :math:`n \times d` submatrix of :math:`{S}` whose upper-left corner appears
///       at index :math:`(\texttt{ro_s}, \texttt{co_s})` of :math:`{S}.`
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Either Layout::ColMajor or Layout::RowMajor
///       * Matrix storage for :math:`\mat(B).`
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(B).` 
///       * The number of rows and columns in :math:`\mat(A).`
///
///      d - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(B)` and :math:`\submat(S).`
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
inline void sketch_symmetric(
    // B = alpha*A*S + beta*B, where A is a symmetric matrix stored in the format of a general matrix.
    blas::Layout layout,
    int64_t n, // number of rows in B
    int64_t d, // number of columns in B
    T alpha,
    const T* A,
    int64_t lda,
    SKOP &S,
    int64_t ro_s,
    int64_t co_s,
    T beta,
    T* B,
    int64_t ldb,
    T sym_check_tol = 0
) {
    RandBLAS::util::require_symmetric(layout, A, n, lda, sym_check_tol);
    sketch_general(layout, blas::Op::NoTrans, blas::Op::NoTrans, n, d, n, alpha, A, lda, S, ro_s, co_s, beta, B, ldb);
}


// =============================================================================
/// \fn sketch_symmetric(blas::Layout layout, int64_t d,
///     int64_t n, T alpha, SKOP &S, int64_t ro_s, int64_t co_s,
///     const T *A, int64_t lda, T beta, T *B, int64_t ldb, T sym_check_tol = 0
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Check that :math:`\mat(A)` is symmetric up to tolerance :math:`\texttt{sym_check_tol}`, then sketch from the left in a SYMM-like operation
/// 
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\submat(S)}_{d \times n} \cdot \underbrace{\mat(A)}_{n \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
/// 
/// where :math:`\alpha` and :math:`\beta` are real scalars and :math:`S` is a sketching operator.
///
/// .. dropdown:: FAQ
///   :animate: fade-in-slide-down
///
///     **What's** :math:`\mat(A)?`
///
///       It's a symmetric matrix of order :math:`n`. Its precise contents depend on :math:`(A, \lda)`,
///       according to 
///
///             .. math::
///                 \mat(A)[i, j] = A[i + j \cdot \lda] = A[i \cdot \lda + j].
///
///       Note that the the "layout" parameter passed to this function is not used here.
///       That's because this function requires :math:`\mat(A)` to be stored in the format
///       of a general matrix (with both upper and lower triangles).
///
///       This function's default behavior is to check that :math:`\mat(A)` is symmetric before
///       attempting sketching. That check  can be skipped (at your own peril!) by calling this
///       function with sym_check_tol < 0.
///
///     **What's** :math:`\mat(B)?`
///
///       It's a :math:`d \times n` matrix.  Its precise contents depend on :math:`(B,\ldb)` and "layout."
///
///       If layout == ColMajor, then
///
///             .. math::
///                 \mat(B)[i, j] = B[i + j \cdot \ldb].
///
///       In this case, :math:`\ldb` must be :math:`\geq d.`
///
///       If layout == RowMajor, then
///
///             .. math::
///                 \mat(B)[i, j] = B[i \cdot \ldb + j].
///
///       In this case, :math:`\ldb` must be :math:`\geq n.`
///
///     **What is** :math:`\submat(S)` **?**
///
///       It's the :math:`d \times n` submatrix of :math:`{S}` whose upper-left corner appears
///       at index :math:`(\texttt{ro_s}, \texttt{co_s})` of :math:`{S}.`
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Either Layout::ColMajor or Layout::RowMajor
///       * Matrix storage for :math:`\mat(B).`
///
///      d - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(B)` and :math:`\submat(S).`
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(B).` 
///       * The number of rows and columns in :math:`\mat(A).`
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
inline void sketch_symmetric(
    // B = alpha*S*A + beta*B
    blas::Layout layout,
    int64_t d, // number of rows in B
    int64_t n, // number of columns in B
    T alpha,
    SKOP &S,
    int64_t ro_s,
    int64_t co_s,
    const T* A,
    int64_t lda,
    T beta,
    T* B,
    int64_t ldb,
    T sym_check_tol = 0
) {
    RandBLAS::util::require_symmetric(layout, A, n, lda, sym_check_tol);
    sketch_general(layout, blas::Op::NoTrans, blas::Op::NoTrans, d, n, n, alpha, S, ro_s, co_s, A, lda, beta, B, ldb);
}

// MARK: FULL(S)

// =============================================================================
/// \fn sketch_symmetric(blas::Layout layout, T alpha, 
///     const T *A, int64_t lda, SKOP &S,
///     T beta, T *B, int64_t ldb, T sym_check_tol = 0
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Check that :math:`\mat(A)` is symmetric up to tolerance :math:`\texttt{sym_check_tol}`, then sketch from the right in a SYMM-like operation
/// 
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\mat(A)}_{n \times n} \cdot S  + \beta \cdot \underbrace{\mat(B)}_{n \times d},    \tag{$\star$}
/// 
/// where :math:`\alpha` and :math:`\beta` are real scalars and :math:`S` is an :math:`n \times d` sketching operator.
///
/// .. dropdown:: FAQ
///   :animate: fade-in-slide-down
///
///     **What's** :math:`\mat(A)?`
///
///       It's a symmetric matrix of order :math:`n`, where :math:`n = \texttt{S.dist.n_cols}`.
///       Its precise contents depend on :math:`(A, \lda)`, according to 
///
///             .. math::
///                 \mat(A)[i, j] = A[i + j \cdot \lda] = A[i \cdot \lda + j].
///
///       Note that the the "layout" parameter passed to this function is not used here.
///       That's because this function requires :math:`\mat(A)` to be stored in the format
///       of a general matrix (with both upper and lower triangles).
///
///       This function's default behavior is to check that :math:`\mat(A)` is symmetric before
///       attempting sketching. That check  can be skipped (at your own peril!) by calling this
///       function with sym_check_tol < 0.
///
///     **What's** :math:`\mat(B)?`
///
///      It's an :math:`n \times d` matrix, where  :math:`n = \texttt{S.dist.n_cols}`
///      and :math:`d = \texttt{S.dist.n_rows}`.
///      Its precise contents depend on :math:`(B,\ldb)` and "layout."
///
///       If layout == ColMajor, then
///
///             .. math::
///                 \mat(B)[i, j] = B[i + j \cdot \ldb].
///
///       In this case, :math:`\ldb` must be :math:`\geq n.`
///
///       If layout == RowMajor, then
///
///             .. math::
///                 \mat(B)[i, j] = B[i \cdot \ldb + j].
///
///       In this case, :math:`\ldb` must be :math:`\geq d.`
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Either Layout::ColMajor or Layout::RowMajor
///       * Matrix storage for :math:`\mat(B).`
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
inline void sketch_symmetric(
    // B = alpha*A*S + beta*B, where A is a symmetric matrix stored in the format of a general matrix.
    blas::Layout layout,
    T alpha,
    const T* A,
    int64_t lda,
    SKOP &S,
    T beta,
    T* B,
    int64_t ldb,
    T sym_check_tol = 0
) {
    int64_t n = S.dist.n_rows;
    int64_t d = S.dist.n_cols;
    RandBLAS::util::require_symmetric(layout, A, n, lda, sym_check_tol);
    sketch_general(layout, blas::Op::NoTrans, blas::Op::NoTrans, n, d, n, alpha, A, lda, S, 0, 0, beta, B, ldb);
}


// =============================================================================
/// \fn sketch_symmetric(blas::Layout layout, T alpha, SKOP &S,
///     const T *A, int64_t lda, T beta, T *B, int64_t ldb, T sym_check_tol = 0
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Check that :math:`\mat(A)` is symmetric up to tolerance :math:`\texttt{sym_check_tol}`, then sketch from the left in a SYMM-like operation
/// 
/// .. math::
///     \mat(B) = \alpha \cdot S \cdot \underbrace{\mat(A)}_{n \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
/// 
/// where :math:`\alpha` and :math:`\beta` are real scalars and :math:`S` is a :math:`d \times n` sketching operator.
///
/// .. dropdown:: FAQ
///   :animate: fade-in-slide-down
///
///     **What's** :math:`\mat(A)?`
///
///       It's a symmetric matrix of order :math:`n`. Its precise contents depend on :math:`(A, \lda)`,
///       according to 
///
///             .. math::
///                 \mat(A)[i, j] = A[i + j \cdot \lda] = A[i \cdot \lda + j].
///
///       Note that the the "layout" parameter passed to this function is not used here.
///       That's because this function requires :math:`\mat(A)` to be stored in the format
///       of a general matrix (with both upper and lower triangles).
///
///       This function's default behavior is to check that :math:`\mat(A)` is symmetric before
///       attempting sketching. That check  can be skipped (at your own peril!) by calling this
///       function with sym_check_tol < 0.
///
///     **What's** :math:`\mat(B)?`
///
///       It's a :math:`d \times n` matrix.  Its precise contents depend on :math:`(B,\ldb)` and "layout."
///
///       If layout == ColMajor, then
///
///             .. math::
///                 \mat(B)[i, j] = B[i + j \cdot \ldb].
///
///       In this case, :math:`\ldb` must be :math:`\geq d.`
///
///       If layout == RowMajor, then
///
///             .. math::
///                 \mat(B)[i, j] = B[i \cdot \ldb + j].
///
///       In this case, :math:`\ldb` must be :math:`\geq n.`
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Either Layout::ColMajor or Layout::RowMajor
///       * Matrix storage for :math:`\mat(B).`
///
///      alpha - [in]
///       * A real scalar.
///       * If zero, then :math:`A` is not accessed.
///
///      S - [in]  
///       * A DenseSkOp or SparseSkOp object.
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
inline void sketch_symmetric(
    // B = alpha*S*A + beta*B
    blas::Layout layout,
    T alpha,
    SKOP &S,
    const T* A,
    int64_t lda,
    T beta,
    T* B,
    int64_t ldb,
    T sym_check_tol = 0
) {
    int64_t d = S.dist.n_rows;
    int64_t n = S.dist.n_cols;
    RandBLAS::util::require_symmetric(layout, A, n, lda, sym_check_tol);
    sketch_general(layout, blas::Op::NoTrans, blas::Op::NoTrans, d, n, n, alpha, S, 0, 0, A, lda, beta, B, ldb);
}

} // end namespace RandBLAS
#endif
