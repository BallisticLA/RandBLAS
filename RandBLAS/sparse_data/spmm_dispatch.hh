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
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include "RandBLAS/sparse_data/conversions.hh"
#include "RandBLAS/sparse_data/csc_spmm_impl.hh"
#include "RandBLAS/sparse_data/csr_spmm_impl.hh"
#include "RandBLAS/sparse_data/coo_spmm_impl.hh"
#include <vector>
#include <algorithm>


namespace RandBLAS::sparse_data {

template <SparseMatrix SpMat, typename T = SpMat::scalar_t>
void left_spmm(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opB,
    int64_t d, // C is d-by-n
    int64_t n, // \op(B) is m-by-n
    int64_t m, // \op(A) is d-by-m
    T alpha,
    SpMat &A,
    int64_t ro_a,
    int64_t co_a,
    const T *B,
    int64_t ldb,
    T beta,
    T *C,
    int64_t ldc
) {
    using blas::Layout;
    using blas::Op;
    // handle applying a transposed sparse matrix.
    if (opA == Op::Trans) {
        using sint_t = typename SpMat::index_t;
        constexpr bool is_coo = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
        constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
        constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
        if constexpr (is_coo) {
            auto At = RandBLAS::sparse_data::coo::transpose(A);
            left_spmm(layout, Op::NoTrans, opB, d, n, m, alpha, At, co_a, ro_a, B, ldb, beta, C, ldc);
        } else if constexpr (is_csc) {
            auto At = RandBLAS::sparse_data::conversions::transpose_as_csr(A);
            left_spmm(layout, Op::NoTrans, opB, d, n, m, alpha, At, co_a, ro_a, B, ldb, beta, C, ldc);
        } else if constexpr (is_csr) {
            auto At = RandBLAS::sparse_data::conversions::transpose_as_csc(A);
            left_spmm(layout, Op::NoTrans, opB, d, n, m, alpha, At, co_a, ro_a, B, ldb, beta, C, ldc);
        } else {
            randblas_require(false);
        }
        return; 
    }
    // Below this point, we can assume A is not transposed.
    randblas_require( A.index_base == IndexBase::Zero );
    using sint_t = typename SpMat::index_t;
    constexpr bool is_coo = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
    constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
    constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
    randblas_require(is_coo || is_csr || is_csc);

    if constexpr (is_coo) {
        randblas_require(A.n_rows >= d);
        randblas_require(A.n_cols >= m);
    } else {
        randblas_require(A.n_rows == d);
        randblas_require(A.n_cols == m);
        randblas_require(ro_a == 0);
        randblas_require(co_a == 0);
    }
    
    // Dimensions of B, rather than \op(B)
    Layout layout_C = layout;
    Layout layout_opB;
    int64_t rows_B, cols_B;
    if (opB == Op::NoTrans) {
        rows_B = m;
        cols_B = n;
        layout_opB = layout;
    } else {
        rows_B = n;
        cols_B = m;
        layout_opB = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
    }

    // Check dimensions and compute C = beta * C.
    //      Note: both B and C are checked based on "layout"; B is *not* checked on layout_opB.
    if (layout == Layout::ColMajor) {
        randblas_require(ldb >= rows_B);
        randblas_require(ldc >= d);
        for (int64_t i = 0; i < n; ++i)
            RandBLAS::util::safe_scal(d, beta, &C[i*ldc], 1);
    } else {
        randblas_require(ldc >= n);
        randblas_require(ldb >= cols_B);
        for (int64_t i = 0; i < d; ++i)
            RandBLAS::util::safe_scal(n, beta, &C[i*ldc], 1);
    }

    if (alpha == (T) 0)
        return;
    
    // compute the matrix-matrix product
    if constexpr (is_coo) {
        using RandBLAS::sparse_data::coo::apply_coo_left_jki_p11;
        apply_coo_left_jki_p11(alpha, layout_opB, layout_C, d, n, m, A, ro_a, co_a, B, ldb, C, ldc);
    } else if constexpr (is_csc) {
        if (layout_opB == Layout::RowMajor && layout_C == Layout::RowMajor) {
            using RandBLAS::sparse_data::csc::apply_csc_left_kib_rowmajor_1p1;
            apply_csc_left_kib_rowmajor_1p1(alpha, d, n, m, A, B, ldb, C, ldc);
        } else {
            using RandBLAS::sparse_data::csc::apply_csc_left_jki_p11;
            apply_csc_left_jki_p11(alpha, layout_opB, layout_C, d, n, m, A, B, ldb, C, ldc);
        }
    } else {
        if  (layout_opB == Layout::RowMajor && layout_C == Layout::RowMajor) {
             using RandBLAS::sparse_data::csr::apply_csr_left_ikb_rowmajor;
             apply_csr_left_ikb_rowmajor(alpha, d, n, m, A, B, ldb, C, ldc);
        } else {
            using RandBLAS::sparse_data::csr::apply_csr_left_jik_p11;
            apply_csr_left_jik_p11(alpha, layout_opB, layout_C, d, n, m, A, B, ldb, C, ldc);
        }
        
    }
    return;
}

template <SparseMatrix SpMat, typename T = SpMat::scalar_t>
inline void right_spmm(
    blas::Layout layout,
    blas::Op opA,
    blas::Op opB,
    int64_t m, // C is m-by-d
    int64_t d, // op(A) is n-by-d
    int64_t n, // op(B) is m-by-n
    T alpha,
    const T *A,
    int64_t lda,
    SpMat &B,
    int64_t i_off,
    int64_t j_off,
    T beta,
    T *C,
    int64_t ldc
) { 
    //
    // Compute C = op(mat(A)) @ op(submat(\mtxB)) by reduction to left_spmm. We start with
    //
    //      C^T = op(submat(\mtxB))^T @ op(mat(A))^T.
    //
    // Then we interchange the operator "op(*)" in op(submat(\mtxA)) and (*)^T:
    //
    //      C^T = op(submat(\mtxB))^T @ op(mat(A)^T).
    //
    // We tell left_spmm to process (C^T) and (B^T) in the opposite memory layout
    // compared to the layout for (B, C).
    // 
    using blas::Layout;
    using blas::Op;
    auto trans_opB = (opB == Op::NoTrans) ? Op::Trans : Op::NoTrans;
    auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
    left_spmm(
        trans_layout, trans_opB, opA,
        d, m, n, alpha, B, i_off, j_off, A, lda, beta, C, ldc
    );
}

} // end namespace RandBLAS::sparse_data

namespace RandBLAS {

// =============================================================================
/// \fn spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m,
///     int64_t n, int64_t k, T alpha, SpMat &A, const T *B, int64_t ldb, T beta, T *C, int64_t ldc
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Multiply a dense matrix on the left with a sparse matrix:
///
/// .. math::
///     \mat(C) = \alpha \cdot \underbrace{\op(\mtxA)}_{m \times k} \cdot \underbrace{\op(\mat(B))}_{k \times n} + \beta \cdot \underbrace{\mat(C)}_{m \times n},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(\mtxX)` either returns a matrix :math:`\mtxX`
/// or its transpose, and :math:`\mtxA` is sparse.
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Layout::ColMajor or Layout::RowMajor.
///       * Matrix storage for :math:`\mat(B)` and :math:`\mat(C)`.
///
///      opA - [in]
///       * If :math:`\opA` == NoTrans, then :math:`\op(\mtxA) = \mtxA`.
///       * If :math:`\opA` == Trans, then :math:`\op(\mtxA) = \mtxA^T`.
///
///      opB - [in]
///       * If :math:`\opB` = NoTrans, then :math:`\op(\mat(B)) = \mat(B)`.
///       * If :math:`\opB` = Trans, then :math:`\op(\mat(B)) = \mat(B)^T`.
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(C)`.
///       * The number of rows in :math:`\op(\mtxA)`.
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(C)`
///       * The number of columns in :math:`\op(\mat(B))`.
///
///      k - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\op(\mtxA)`
///       * The number of rows in :math:`\op(\mat(B))`.
///
///      alpha - [in]
///       * A real scalar.
///
///      A - [in]
///       * A RandBLAS sparse matrix object.
///       * Defines :math:`\mtxA`.
///
///      B - [in]
///       * Pointer to 1D array of real scalars that define :math:`\mat(B)`.
///
///      ldb - [in]
///       * A nonnegative integer.
///       * The leading dimension of :math:`\mat(B)` when reading from :math:`B.`
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`C` need not be set on input.
///
///      C - [in, out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(C)`
///         on the RIGHT-hand side of :math:`(\star)`.
///       * On exit, defines :math:`\mat(C)`
///         on the LEFT-hand side of :math:`(\star)`.
///
///      ldc - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(C)` when reading from :math:`C`.
///
/// @endverbatim
template <SparseMatrix SpMat, typename T = SpMat::scalar_t>
inline void spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, SpMat &A, const T *B, int64_t ldb, T beta, T *C, int64_t ldc) {
    RandBLAS::sparse_data::left_spmm(layout, opA, opB, m, n, k, alpha, A, 0, 0, B, ldb, beta, C, ldc);
    return;
};

// =============================================================================
/// \fn spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m,
///     int64_t n, int64_t k, T alpha, const T* A, int64_t lda, SpMat &B, T beta, T *C, int64_t ldc
/// ) 
/// @verbatim embed:rst:leading-slashes
/// Multiply a dense matrix on the right with a sparse matrix:
///
/// .. math::
///     \mat(C) = \alpha \cdot \underbrace{\op(\mat(A))}_{m \times k} \cdot \underbrace{\op(\mtxB)}_{k \times n} + \beta \cdot \underbrace{\mat(C)}_{m \times n},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars, :math:`\op(\mtxX)` either returns a matrix :math:`\mtxX`
/// or its transpose, and :math:`\mtxB` is sparse.
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Layout::ColMajor or Layout::RowMajor.
///       * Matrix storage for :math:`\mat(A)` and :math:`\mat(C)`.
///
///      opA - [in]
///       * If :math:`\opA` = NoTrans, then :math:`\op(\mat(A)) = \mat(A)`.
///       * If :math:`\opA` = Trans, then :math:`\op(\mat(A)) = \mat(A)^T`.
///
///      opB - [in]
///       * If :math:`\opB` = NoTrans, then :math:`\op(\mtxB) = \mtxB`.
///       * If :math:`\opB` = Trans, then :math:`\op(\mtxB) = \mtxB^T`.
///
///      m - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(C)`.
///       * The number of rows in :math:`\op(\mat(A))`.
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(C)`.
///       * The number of columns in :math:`\op(\mtxB)`.
///
///      k - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\op(\mat(A))`
///       * The number of rows in :math:`\op(\mtxB)`.
///
///      alpha - [in]
///       * A real scalar.
///
///      A - [in]
///       * Pointer to a 1D array of real scalars.
///
///      lda - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(A)` when reading from :math:`A`. 
///
///      B - [in]
///       * A RandBLAS sparse matrix object.
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`C` need not be set on input.
///
///      C - [in, out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(C)`
///         on the RIGHT-hand side of :math:`(\star)`.
///       * On exit, defines :math:`\mat(C)`
///         on the LEFT-hand side of :math:`(\star)`.
///
///      ldc - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(C)` when reading from :math:`C`.
///
/// @endverbatim
template <SparseMatrix SpMat, typename T = SpMat::scalar_t>
inline void spmm(blas::Layout layout, blas::Op opA, blas::Op opB, int64_t m, int64_t n, int64_t k, T alpha, const T *A, int64_t lda, SpMat &B, T beta, T *C, int64_t ldc) {
    RandBLAS::sparse_data::right_spmm(layout, opA, opB, m, n, k, alpha, A, lda, B, 0, 0, B, beta, C, ldc);
    return;
}

}
