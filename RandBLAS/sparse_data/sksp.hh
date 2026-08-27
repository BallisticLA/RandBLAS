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

#pragma once

#include "RandBLAS/base.hh"
#include "RandBLAS/dense_skops.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/util.hh"
#include "RandBLAS/sparse_skops.hh"


namespace RandBLAS::sparse_data {

// MARK: LSKSP3

// =============================================================================
/// \fn lsksp3(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d,
///     int64_t n, int64_t m, T alpha, const DenseSkOp<T,RNG> &S, int64_t ro_s, int64_t co_s,
///     SpMat &A, int64_t ro_a, int64_t co_a, T beta, T *B, int64_t ldb
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Sketch from the left in an SpMM-like operation
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\submat(\mtxS))}_{d \times m} \cdot \underbrace{\op(\submat(\mtxA))}_{m \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(\mtxX)` either returns a matrix :math:`\mtxX`
/// or its transpose, :math:`\mtxA` is a sparse matrix, and :math:`\mtxS` is a dense sketching operator.
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Layout::ColMajor or Layout::RowMajor.
///       * Matrix storage for :math:`\mat(B)`.
///
///      opS - [in]
///       * If :math:`\opS` = NoTrans, then :math:`\op(\submat(\mtxS)) = \submat(\mtxS)`.
///       * If :math:`\opS` = Trans, then :math:`\op(\submat(\mtxS)) = \submat(\mtxS)^T`.
///
///      opA - [in]
///       * If :math:`\opA` = NoTrans, then :math:`\op(\submat(\mtxA)) = \submat(\mtxA)`.
///       * If :math:`\opA` = Trans, then :math:`\op(\submat(\mtxA)) = \submat(\mtxA)^T`.
///
///      d - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(B)`.
///       * The number of rows in :math:`\op(\submat(\mtxS))`.
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(B)`.
///       * The number of columns in :math:`\op(\mat(\mtxA))`.
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\op(\submat(\mtxS))`
///       * The number of rows in :math:`\op(\mat(\mtxA))`.
///
///      alpha - [in]
///       * A real scalar.
///
///      S - [in]
///       * A DenseSkOp object.
///       * Defines :math:`\submat(\mtxS)`.
///
///      ro_s - [in]
///       * A nonnegative integer.
///       * The rows of :math:`\submat(\mtxS)` are a contiguous subset of rows of :math:`\mtxS`.
///       * The rows of :math:`\submat(\mtxS)` start at :math:`S[\texttt{ro_s}, :]`.
///
///      co_s - [in]
///       * A nonnegative integer.
///       * The columns of :math:`\submat(\mtxS)` are a contiguous subset of columns of :math:`\mtxS`.
///       * The columns :math:`\submat(\mtxS)` start at :math:`S[:,\texttt{co_s}]`. 
///
///      A - [in]
///       * A RandBLAS sparse matrix object.
///       * Defines :math:`\submat(\mtxA)`.
///
///      ro_a - [in]
///       * A nonnegative integer.
///       * The rows of :math:`\submat(\mtxA)` are a contiguous subset of rows of :math:`\mtxA`.
///       * The rows of :math:`\submat(\mtxA)` start at :math:`\mtxA[\texttt{ro_a}, :]`.
///
///      co_a - [in]
///       * A nonnegative integer.
///       * The columns of :math:`\submat(\mtxA)` are a contiguous subset of columns of :math:`\mtxA`.
///       * The columns :math:`\submat(\mtxA)` start at :math:`\mtxA[:,\texttt{co_a}]`. 
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`B` need not be set on input.
///
///      B - [in, out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(B)`
///         on the RIGHT-hand side of :math:`(\star)`.
///       * On exit, defines :math:`\mat(B)`
///         on the LEFT-hand side of :math:`(\star)`.
///
///      ldb - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(B)` when reading from :math:`B`.
///
/// @endverbatim
template <typename T, SparseMatrix SpMat, typename DenseSkOp>
void lsksp3(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(submat(\mtxA)) is m-by-n
    int64_t m, // op(submat(\mtxS)) is d-by-m
    T alpha,
    const DenseSkOp &S,
    int64_t ro_s,
    int64_t co_s,
    const SpMat &A,
    int64_t ro_a,
    int64_t co_a,
    T beta,
    T *B,
    int64_t ldb
) {
    // B = op(submat(\mtxS)) @ op(submat(\mtxA))
    auto [rows_submat_S, cols_submat_S] = dims_before_op(d, m, opS);
    constexpr bool maybe_denseskop = !std::is_same_v<std::remove_cv_t<DenseSkOp>, BLASFriendlyOperator<T>>;
    if constexpr (maybe_denseskop) {
        if (!S.buff) {
            // DenseSkOp doesn't permit defining a "black box" distribution, so we have to pack the submatrix
            // into an equivalent datastructure ourselves.
            auto submat_S = submatrix_as_blackbox<BLASFriendlyOperator<T>>(S, rows_submat_S, cols_submat_S, ro_s, co_s);
            lsksp3(layout, opS, opA, d, n, m, alpha, submat_S, 0, 0, A, ro_a, co_a, beta, B, ldb);
            return;
        } // else, proceed with the rest of the function call.
    } 
    randblas_require( S.buff != nullptr );
    auto [rows_submat_A, cols_submat_A] = dims_before_op(m, n, opA);
    validate_submat_dims(A.n_rows, A.n_cols, rows_submat_A, cols_submat_A, ro_a, co_a);
    validate_submat_dims(S.n_rows, S.n_cols, rows_submat_S, cols_submat_S, ro_s, co_s);
    if (layout == blas::Layout::ColMajor) {
        randblas_require(ldb >= d);
    } else {
        randblas_require(ldb >= n);
    }

    auto [pos, lds] = offset_and_ldim(S.layout, S.n_rows, S.n_cols, ro_s, co_s);
    T* S_ptr = &S.buff[pos];
    if (S.layout != layout)
        opS = (opS == blas::Op::NoTrans) ? blas::Op::Trans : blas::Op::NoTrans;

    right_spmm(layout, opS, opA, d, n, m, alpha, S_ptr, lds, A, ro_a, co_a, beta, B, ldb);
    return;
}

// MARK: RSKSP3

// =============================================================================
/// \fn rsksp3(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m,
///     int64_t d, int64_t n, T alpha, const SpMat &A, int64_t ro_a, int64_t co_a,
///     const DenseSkOp<T,RNG> &S, int64_t ro_s, int64_t co_s, T beta, T *B, int64_t ldb
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Sketch from the right in an SpMM-like operation
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\submat(\mtxA))}_{m \times n} \cdot \underbrace{\op(\submat(\mtxS))}_{n \times d} + \beta \cdot \underbrace{\mat(B)}_{m \times d},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(\mtxX)` either returns a matrix :math:`\mtxX`
/// or its transpose, :math:`\mtxA` is a sparse matrix, and :math:`\mtxS` is a dense sketching operator.
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Layout::ColMajor or Layout::RowMajor.
///       * Matrix storage for :math:`\mat(B)`.
///
///      opA - [in]
///       * If :math:`\opA` == NoTrans, then :math:`\op(\submat(\mtxA)) = \submat(\mtxA)`.
///       * If :math:`\opA` == Trans, then :math:`\op(\submat(\mtxA)) = \submat(\mtxA)^T`.
///
///      opS - [in]
///       * If :math:`\opS` = NoTrans, then :math:`\op(\submat(\mtxS)) = \submat(\mtxS)`.
///       * If :math:`\opS` = Trans, then :math:`\op(\submat(\mtxS)) = \submat(\mtxS)^T`.
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(B)`.
///       * The number of rows in :math:`\op(\submat(\mtxA))`.
///
///      d - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(B)`
///       * The number of columns in :math:`\op(\submat(\mtxS))`.
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\op(\submat(\mtxA))`
///       * The number of rows in :math:`\op(\submat(\mtxS))`.
///
///      alpha - [in]
///       * A real scalar.
///
///      A - [in]
///       * A RandBLAS sparse matrix object.
///       * Defines :math:`\submat(\mtxA)`.
///
///      ro_a - [in]
///       * A nonnegative integer.
///       * The rows of :math:`\submat(\mtxA)` are a contiguous subset of rows of :math:`\mtxA`.
///       * The rows of :math:`\submat(\mtxA)` start at :math:`\mtxA[\texttt{ro_a}, :]`.
///
///      co_a - [in]
///       * A nonnegative integer.
///       * The columns of :math:`\submat(\mtxA)` are a contiguous subset of columns of :math:`\mtxA`.
///       * The columns :math:`\submat(\mtxA)` start at :math:`\mtxA[:,\texttt{co_a}]`. 
///
///      S - [in]
///       * A DenseSkOp object.
///       * Defines :math:`\submat(\mtxS)`.
///
///      ro_s - [in]
///       * A nonnegative integer.
///       * The rows of :math:`\submat(\mtxS)` are a contiguous subset of rows of :math:`\mtxS`.
///       * The rows of :math:`\submat(\mtxS)` start at :math:`\mtxS[\texttt{ro_s}, :]`.
///
///      co_s - [in]
///       * A nonnegative integer.
///       * The columns of :math:`\submat(\mtxS)` are a contiguous subset of columns of :math:`\mtxS`.
///       * The columns :math:`\submat(\mtxS)` start at :math:`\mtxS[:,\texttt{co_s}]`. 
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`B` need not be set on input.
///
///      B - [in, out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(B)`
///         on the RIGHT-hand side of :math:`(\star)`.
///       * On exit, defines :math:`\mat(B)`
///         on the LEFT-hand side of :math:`(\star)`.
///
///      ldb - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(B)` when reading from :math:`B`.
///
/// @endverbatim
template <typename T, SparseMatrix SpMat, typename DenseSkOp>
void rsksp3(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opS,
    int64_t m, // B is m-by-d
    int64_t d, // op(submat(\mtxA)) is m-by-n
    int64_t n, // op(submat(\mtxS)) is n-by-d
    T alpha,
    const SpMat &A,
    int64_t ro_a,
    int64_t co_a,
    const DenseSkOp &S,
    int64_t ro_s,
    int64_t co_s,
    T beta,
    T *B,
    int64_t ldb
) {
    auto [rows_submat_S, cols_submat_S] = dims_before_op(n, d, opS);
    constexpr bool maybe_denseskop = !std::is_same_v<std::remove_cv_t<DenseSkOp>, BLASFriendlyOperator<T>>;
    if constexpr (maybe_denseskop) {
        if (!S.buff) {
            // DenseSkOp doesn't permit defining a "black box" distribution, so we have to pack the submatrix
            // into an equivalent datastructure ourselves.
            auto submat_S = submatrix_as_blackbox<BLASFriendlyOperator<T>>(S, rows_submat_S, cols_submat_S, ro_s, co_s);
            rsksp3(layout, opA, opS, m, d, n, alpha, A, ro_a, co_a, submat_S, 0, 0, beta, B, ldb);
            return;
        }
    }
    randblas_require( S.buff != nullptr );
    auto [rows_submat_A, cols_submat_A] = dims_before_op(m, n, opA);
    validate_submat_dims(A.n_rows, A.n_cols, rows_submat_A, cols_submat_A, ro_a, co_a);
    validate_submat_dims(S.n_rows, S.n_cols, rows_submat_S, cols_submat_S, ro_s, co_s);
    if (layout == blas::Layout::ColMajor) {
        randblas_require(ldb >= m);
    } else {
        randblas_require(ldb >= d);
    }

    auto [pos, lds] = offset_and_ldim(S.layout, S.n_rows, S.n_cols, ro_s, co_s);
    T* S_ptr = &S.buff[pos];
    if (S.layout != layout)
        opS = (opS == blas::Op::NoTrans) ? blas::Op::Trans : blas::Op::NoTrans;

    left_spmm(layout, opA, opS, m, d, n, alpha, A, ro_a, co_a, S_ptr, lds, beta, B, ldb);
    return;
}

// MARK: LSKSPS

// =============================================================================
/// \fn lsksps(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d,
///     int64_t n, int64_t m, T alpha, const SparseSkOp<T,RNG,sint_t> &S, int64_t ro_s, int64_t co_s,
///     const SpMat &A, int64_t ro_a, int64_t co_a, T beta, T *B, int64_t ldb
/// )
/// @verbatim embed:rst:leading-slashes
/// Sketch from the left in an SpMM-like operation
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\submat(\mtxS))}_{d \times m} \cdot \underbrace{\op(\mtxA)}_{m \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(\mtxX)` either returns a matrix :math:`\mtxX`
/// or its transpose, and both :math:`\mtxA` and :math:`\mtxS` are sparse.
/// @endverbatim
//
// The sparse-times-sparse backend (spgemm, MKL-backed) has no submatrix
// parameters for either operand and no op flag for its second operand, so
// this function differs from lsksp3 in two ways: A must be passed in full
// (ro_a == co_a == 0; spgemm has nothing to extract a submatrix into without
// first copying A, which this PR does not do), and op(A) is folded into a
// transpose *view* of A (A.transpose(), the same lightweight technique
// spsymm_dispatch.hh's Case D uses) rather than passed as a flag. S keeps
// full submatrix support: submatrix_as_coo / coo_view_of_skop already build
// exactly the owning or non-owning COOMatrix window spgemm's first operand
// needs, mirroring how LSKGES handles a SparseSkOp against a dense operand.
template <typename T, SparseMatrix SpMat, typename RNG, SignedInteger sint_t>
void lsksps(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(A) is m-by-n
    int64_t m, // op(submat(S)) is d-by-m
    T alpha,
    const SparseSkOp<T,RNG,sint_t> &S,
    int64_t ro_s,
    int64_t co_s,
    const SpMat &A,
    int64_t ro_a,
    int64_t co_a,
    T beta,
    T *B,
    int64_t ldb
) {
    randblas_require(ro_a == 0 && co_a == 0);
    randblas_require(A.index_base == IndexBase::Zero);
    auto [rows_submat_S, cols_submat_S] = dims_before_op(d, m, opS);
    validate_submat_dims(S.n_rows, S.n_cols, rows_submat_S, cols_submat_S, ro_s, co_s);
    randblas_require(A.n_rows == m && A.n_cols == n);
    if (layout == blas::Layout::ColMajor) {
        randblas_require(ldb >= d);
    } else {
        randblas_require(ldb >= n);
    }

    if (d == 0 || n == 0)
        return;
    bool full_operator = (S.n_rows == rows_submat_S && S.n_cols == cols_submat_S && ro_s == 0 && co_s == 0);
    if (alpha == T(0) || A.nnz == 0) {
        RandBLAS::util::lascl(layout, d, n, beta, B, ldb);
        return;
    }

    // A submatrix of S must be materialized to know its nnz (there is no
    // cheap structural check for a windowed operator), so this second empty
    // check runs after building Scoo rather than before.
    COOMatrix<T, sint_t> Scoo = full_operator
        ? RandBLAS::sparse::coo_view_of_skop(S)
        : RandBLAS::sparse::submatrix_as_coo(S, rows_submat_S, cols_submat_S, ro_s, co_s);
    if (Scoo.nnz == 0) {
        RandBLAS::util::lascl(layout, d, n, beta, B, ldb);
        return;
    }

    if (opA == blas::Op::NoTrans) {
        spgemm(layout, opS, alpha, Scoo, A, beta, B, ldb);
    } else {
        auto At = A.transpose();
        spgemm(layout, opS, alpha, Scoo, At, beta, B, ldb);
    }
    return;
}

// MARK: RSKSPS

// =============================================================================
/// \fn rsksps(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m,
///     int64_t d, int64_t n, T alpha, const SpMat &A, int64_t ro_a, int64_t co_a,
///     const SparseSkOp<T,RNG,sint_t> &S, int64_t ro_s, int64_t co_s, T beta, T *B, int64_t ldb
/// )
/// @verbatim embed:rst:leading-slashes
/// Sketch from the right in an SpMM-like operation
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\mtxA)}_{m \times n} \cdot \underbrace{\op(\submat(\mtxS))}_{n \times d} + \beta \cdot \underbrace{\mat(B)}_{m \times d},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(\mtxX)` either returns a matrix :math:`\mtxX`
/// or its transpose, and both :math:`\mtxA` and :math:`\mtxS` are sparse. Same restrictions as
/// lsksps: A must be passed in full, and op(submat(S)) is folded into a transpose view when needed.
/// @endverbatim
template <typename T, SparseMatrix SpMat, typename RNG, SignedInteger sint_t>
void rsksps(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opS,
    int64_t m, // B is m-by-d
    int64_t d, // op(submat(S)) is n-by-d
    int64_t n, // op(A) is m-by-n
    T alpha,
    const SpMat &A,
    int64_t ro_a,
    int64_t co_a,
    const SparseSkOp<T,RNG,sint_t> &S,
    int64_t ro_s,
    int64_t co_s,
    T beta,
    T *B,
    int64_t ldb
) {
    randblas_require(ro_a == 0 && co_a == 0);
    randblas_require(A.index_base == IndexBase::Zero);
    auto [rows_submat_S, cols_submat_S] = dims_before_op(n, d, opS);
    validate_submat_dims(S.n_rows, S.n_cols, rows_submat_S, cols_submat_S, ro_s, co_s);
    randblas_require(A.n_rows == m && A.n_cols == n);
    if (layout == blas::Layout::ColMajor) {
        randblas_require(ldb >= m);
    } else {
        randblas_require(ldb >= d);
    }

    if (m == 0 || d == 0)
        return;
    if (alpha == T(0) || A.nnz == 0) {
        RandBLAS::util::lascl(layout, m, d, beta, B, ldb);
        return;
    }

    bool full_operator = (S.n_rows == rows_submat_S && S.n_cols == cols_submat_S && ro_s == 0 && co_s == 0);
    COOMatrix<T, sint_t> Scoo = full_operator
        ? RandBLAS::sparse::coo_view_of_skop(S)
        : RandBLAS::sparse::submatrix_as_coo(S, rows_submat_S, cols_submat_S, ro_s, co_s);
    if (Scoo.nnz == 0) {
        RandBLAS::util::lascl(layout, m, d, beta, B, ldb);
        return;
    }

    if (opS == blas::Op::NoTrans) {
        spgemm(layout, opA, alpha, A, Scoo, beta, B, ldb);
    } else {
        auto St = Scoo.transpose();
        spgemm(layout, opA, alpha, A, St, beta, B, ldb);
    }
    return;
}

}  // end namespace RandBLAS::sparse_data


namespace RandBLAS {

using namespace RandBLAS::dense;
using namespace RandBLAS::sparse_data;

// MARK: SKSP overloads, full

// =============================================================================
/// \fn sketch_sparse(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d,  int64_t n, int64_t m,
///     T alpha, const DenseSkOp &S, int64_t ro_s, int64_t co_s, const SpMat &A, T beta, T *B, int64_t ldb
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Sketch from the left in an SpMM-like operation
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\submat(\mtxS))}_{d \times m} \cdot \underbrace{\op(\mtxA)}_{m \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(\mtxX)` either returns a matrix :math:`\mtxX`
/// or its transpose, :math:`\mtxA` is a sparse matrix, and :math:`\mtxS` is a dense sketching operator.
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Layout::ColMajor or Layout::RowMajor.
///       * Matrix storage for :math:`\mat(B)`.
///
///      opS - [in]
///       * If :math:`\opS` = NoTrans, then :math:`\op(\submat(\mtxS)) = \submat(\mtxS)`.
///       * If :math:`\opS` = Trans, then :math:`\op(\submat(\mtxS)) = \submat(\mtxS)^T`.
///
///      opA - [in]
///       * If :math:`\opA` = NoTrans, then :math:`\op(\mtxA) = \mtxA`.
///       * If :math:`\opA` = Trans, then :math:`\op(\mtxA) = \mtxA^T`.
///
///      d - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(B)`.
///       * The number of rows in :math:`\op(\submat(\mtxS))`.
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(B)`.
///       * The number of columns in :math:`\op(\mtxA)`.
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\op(\submat(\mtxS))`
///       * The number of rows in :math:`\op(\mtxA)`.
///
///      alpha - [in]
///       * A real scalar.
///       * If zero, then :math:`A` is not accessed.
///
///      S - [in]
///       * A DenseSkOp object.
///       * Defines :math:`\submat(\mtxS)`.
///
///      ro_s - [in]
///       * A nonnegative integer.
///       * The rows of :math:`\submat(\mtxS)` are a contiguous subset of rows of :math:`\mtxS`.
///       * The rows of :math:`\submat(\mtxS)` start at :math:`\mtxS[\texttt{ro_s}, :]`.
///
///      co_s - [in]
///       * A nonnegative integer.
///       * The columns of :math:`\submat(\mtxS)` are a contiguous subset of columns of :math:`\mtxS`.
///       * The columns :math:`\submat(\mtxS)` start at :math:`\mtxS[:,\texttt{co_s}]`. 
///
///      A - [in]
///       * A RandBLAS sparse matrix object.
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`B` need not be set on input.
///
///      B - [in, out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(B)`
///         on the RIGHT-hand side of :math:`(\star)`.
///       * On exit, defines :math:`\mat(B)`
///         on the LEFT-hand side of :math:`(\star)`.
///
///      ldb - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(B)` when reading from :math:`B`.
///
/// @endverbatim
template <SparseMatrix SpMat, typename DenseSkOp, typename T = DenseSkOp::scalar_t>
inline void sketch_sparse(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // op(submat(\mtxA)) is m-by-n
    int64_t m, // op(submat(\mtxS)) is d-by-m
    T alpha,
    const DenseSkOp &S,
    int64_t ro_s,
    int64_t co_s,
    const SpMat &A,
    T beta,
    T *B,
    int64_t ldb
) {
    sparse_data::lsksp3(layout, opS, opA, d, n, m, alpha, S, ro_s, co_s, A, 0, 0, beta, B, ldb);
    return;
}


// =============================================================================
/// \fn sketch_sparse(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d,
///     int64_t n, int64_t m, T alpha, const SpMat &A, const DenseSkOp &S, int64_t ro_s, int64_t co_s, T beta, T *B, int64_t ldb
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Sketch from the right in an SpMM-like operation
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\mtxA)}_{m \times n} \cdot \underbrace{\op(\submat(\mtxS))}_{n \times d} + \beta \cdot \underbrace{\mat(B)}_{m \times d},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(\mtxX)` either returns a matrix :math:`\mtxX`
/// or its transpose, :math:`\mtxA` is a sparse matrix, and :math:`\mtxS` is a dense sketching operator.
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Layout::ColMajor or Layout::RowMajor.
///       * Matrix storage for :math:`\mat(B)`.
///
///      opA - [in]
///       * If :math:`\opA` == NoTrans, then :math:`\op(\mtxA) = \mtxA`.
///       * If :math:`\opA` == Trans, then :math:`\op(\mtxA) = \mtxA^T`.
///
///      opS - [in]
///       * If :math:`\opS` = NoTrans, then :math:`\op(\submat(\mtxS)) = \submat(\mtxS)`.
///       * If :math:`\opS` = Trans, then :math:`\op(\submat(\mtxS)) = \submat(\mtxS)^T`.
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(B)`.
///       * The number of rows in :math:`\op(\mtxA)`.
///
///      d - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(B)`
///       * The number of columns in :math:`\op(\submat(\mtxS))`.
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\op(\mtxA)`
///       * The number of rows in :math:`\op(\submat(\mtxS))`.
///
///      alpha - [in]
///       * A real scalar.
///       * If zero, then :math:`A` is not accessed.
///
///      S - [in]
///       * A DenseSkOp object.
///       * Defines :math:`\submat(\mtxS)`.
///
///      ro_s - [in]
///       * A nonnegative integer.
///       * The rows of :math:`\submat(\mtxS)` are a contiguous subset of rows of :math:`\mtxS`.
///       * The rows of :math:`\submat(\mtxS)` start at :math:`\mtxS[\texttt{ro_s}, :]`.
///
///      co_s - [in]
///       * A nonnegative integer.
///       * The columns of :math:`\submat(\mtxS)` are a contiguous subset of columns of :math:`\mtxS`.
///       * The columns :math:`\submat(\mtxS)` start at :math:`\mtxS[:,\texttt{co_s}]`. 
///
///      A - [in]
///       * A RandBLAS sparse matrix object.
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`B` need not be set on input.
///
///      B - [in, out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(B)`
///         on the RIGHT-hand side of :math:`(\star)`.
///       * On exit, defines :math:`\mat(B)`
///         on the LEFT-hand side of :math:`(\star)`.
///
///      ldb - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(B)` when reading from :math:`B`.
///
/// @endverbatim
template <SparseMatrix SpMat, typename DenseSkOp, typename T = DenseSkOp::scalar_t>
inline void sketch_sparse(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opS,
    int64_t m, // B is m-by-d
    int64_t d, // op(submat(\mtxA)) is m-by-n
    int64_t n, // op(submat(\mtxS)) is n-by-d
    T alpha,
    const SpMat &A,
    const DenseSkOp &S,
    int64_t ro_s,
    int64_t co_s,
    T beta,
    T *B,
    int64_t ldb
) {
    sparse_data::rsksp3(layout, opA, opS, m, d, n, alpha, A, 0, 0, S, ro_s, co_s, beta, B, ldb);
    return;
}


// MARK: SKSP overloads, sparse sketching operator

// =============================================================================
/// \fn sketch_sparse(blas::Layout layout, blas::Op opS, blas::Op opA, int64_t d, int64_t n, int64_t m,
///     T alpha, const SparseSkOp<T,RNG,sint_t> &S, int64_t ro_s, int64_t co_s, const SpMat &A, T beta, T *B, int64_t ldb
/// )
/// @verbatim embed:rst:leading-slashes
/// Sketch a sparse matrix from the left with a sparse sketching operator.
/// Same equation as the DenseSkOp overload above. Requires Intel MKL (the
/// sparse-times-sparse backend has no non-MKL fallback) and :math:`\mtxA` in
/// full: a submatrix of :math:`\mtxA` is not supported through this
/// overload. :math:`\submat(\mtxS)` is fully supported.
/// @endverbatim
template <SparseMatrix SpMat, typename RNG, SignedInteger sint_t, typename T = SpMat::scalar_t>
inline void sketch_sparse(
    blas::Layout layout,
    blas::Op opS,
    blas::Op opA,
    int64_t d, // B is d-by-n
    int64_t n, // A is m-by-n
    int64_t m, // op(submat(\mtxS)) is d-by-m
    T alpha,
    const SparseSkOp<T,RNG,sint_t> &S,
    int64_t ro_s,
    int64_t co_s,
    const SpMat &A,
    T beta,
    T *B,
    int64_t ldb
) {
    sparse_data::lsksps(layout, opS, opA, d, n, m, alpha, S, ro_s, co_s, A, 0, 0, beta, B, ldb);
    return;
}

// =============================================================================
/// \fn sketch_sparse(blas::Layout layout, blas::Op opA, blas::Op opS, int64_t m, int64_t d, int64_t n,
///     T alpha, const SpMat &A, const SparseSkOp<T,RNG,sint_t> &S, int64_t ro_s, int64_t co_s, T beta, T *B, int64_t ldb
/// )
/// @verbatim embed:rst:leading-slashes
/// Sketch a sparse matrix from the right with a sparse sketching operator.
/// Same equation as the DenseSkOp overload above, and the same restrictions
/// as the left-sketching overload just above: requires Intel MKL, and
/// :math:`\mtxA` must be passed in full.
/// @endverbatim
template <SparseMatrix SpMat, typename RNG, SignedInteger sint_t, typename T = SpMat::scalar_t>
inline void sketch_sparse(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opS,
    int64_t m, // B is m-by-d
    int64_t d, // op(submat(\mtxA)) is m-by-n
    int64_t n, // op(submat(\mtxS)) is n-by-d
    T alpha,
    const SpMat &A,
    const SparseSkOp<T,RNG,sint_t> &S,
    int64_t ro_s,
    int64_t co_s,
    T beta,
    T *B,
    int64_t ldb
) {
    sparse_data::rsksps(layout, opA, opS, m, d, n, alpha, A, 0, 0, S, ro_s, co_s, beta, B, ldb);
    return;
}

}  // end namespace RandBLAS
