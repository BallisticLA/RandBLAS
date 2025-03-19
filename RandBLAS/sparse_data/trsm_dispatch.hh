// Copyright, 2025. See LICENSE for copyright holder information.
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
#include "RandBLAS/sparse_data/csc_trsm_impl.hh"
#include "RandBLAS/sparse_data/csr_trsm_impl.hh"
#include <vector>
#include <algorithm>


namespace RandBLAS::sparse_data {


// We call these functions "TRSM" instead of "SPTRSM" because they're in the sparse_data namespace.


template <SparseMatrix SpMat>
inline void trsm_matrix_validation( const SpMat &A, blas::Uplo uplo, blas::Diag diag, int mode ) {
    using T = typename SpMat::scalar_t;
    using sint_t = typename SpMat::index_t;
    constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
    int64_t p, ell;
    
    const T* vals = A.vals;
    int64_t m = A.n_rows;
    int64_t flag = -1;
    const bool unitdiag = diag == blas::Diag::Unit;
    if constexpr (is_csr) {
        const sint_t* ptrs = A.rowptr;
        const sint_t* idxs = A.colidxs;
        if (mode >= 0) {
            bool ordered_indices = compressed_indices_are_increasing(m, ptrs, idxs, &flag);
            if (!ordered_indices) {
                std::stringstream ss;
                ss << "Ill-formed CSR matrix; indices in row " << flag << " are not sorted.";
                throw RandBLAS::Error(ss.str());
            }
        }
        for (ell = 0; ell < m; ++ell) {
            p = (uplo == blas::Uplo::Lower) ? ptrs[ell+1] - 1 : ptrs[ell];
            randblas_require(idxs[p] == ell);
            randblas_require(unitdiag || vals[p] != 0.0);
        }
    } else {
        const sint_t* ptrs = A.colptr;
        const sint_t* idxs = A.rowidxs;
        if (mode >= 0) {
            bool ordered_indices = compressed_indices_are_increasing(m, ptrs, idxs, &flag);
            if (!ordered_indices) {
                std::stringstream ss;
                ss << "Ill-formed CSC matrix; indices in column " << flag << " are not sorted.";
                throw RandBLAS::Error(ss.str());
            }
        }
        for (ell = 0; ell < m; ++ell) {
            p = (uplo == blas::Uplo::Lower) ? ptrs[ell] : ptrs[ell+1] - 1;
            randblas_require(idxs[p] == ell);
            randblas_require(unitdiag || vals[p] != 0.0);
        }
    }
    return;
}


// =============================================================================
/// \fn trsm(
///     blas::Layout layout, blas::Op opA, T alpha, const SpMat &A, blas::Uplo uplo, blas::Diag diag, int64_t n, T *B, int64_t ldb, int validation_mode = 1
/// )
/// @verbatim embed:rst:leading-slashes
/// Overwrite :math:`n` columns of a matrix :math:`\mat(B)` with the results of a scaled triangular solve
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\op(\mtxA)^{-1}}_{m \times m} \cdot \underbrace{\mat(B)}_{m \times n},   \tag{$\star$}
///
/// where :math:`\mtxA` is either the sparse matrix :math:`A` or a view of :math:`A` that replaces
/// its diagonal with the vector of all ones, and :math:`\op(\mtxA)` returns either :math:`\mtxA` or :math:`\mtxA^T.`
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Layout::ColMajor or Layout::RowMajor.
///       * Matrix storage for :math:`\mat(B).`
///
///      opA - [in]
///       * If :math:`\opA` = NoTrans, then :math:`\op(\mtxA) = \mtxA.`
///       * If :math:`\opA` = Trans, then :math:`\op(\mtxA) = \mtxA^T.`
///
///      alpha - [in]
///       * A real scalar.
///
///      A - [in]
///       * A RandBLAS CSCMatrix or CSRMatrix.
///       * Considered along with :math:`\texttt{diag}` to define :math:`\mtxA.`
///
///      uplo - [in]
///       * Promises that :math:`A` is structurally upper triangular (if :math:`\texttt{uplo = Uplo::Upper}`) or
///         structurally lower triangular (if :math:`\texttt{uplo = Uplo::Lower}`). Violating this promise will 
///         result in an error if :math:`\texttt{validation_mode} \geq 1` or undefined behavior if 
///         :math:`\texttt{validation_mode} \leq 0.`
///       * Future RandBLAS versions may change behavior so that :math:`\mtxA` is defined as a view of the
///         :math:`\texttt{uplo}` part of :math:`A,` with its diagonal possibly redefined as the vector of 
///         all ones according to :math:`\texttt{diag}.`
///
///      diag - [in]
///       * Diag::Unit or Diag::NonUnit.
///       * If NonUnit, then :math:`\mtxA = A` and the triangular solve is performed as usual.
///       * If Unit, then :math:`\mtxA` is an implicit copy of :math:`A` with its diagonal overwritten to contain all ones.
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(B).`
///
///      B - [in, out]
///       * Pointer to 1D array of real scalars that define :math:`\mat(B).`
///
///      ldb - [in]
///       * A nonnegative integer.
///       * The leading dimension of :math:`\mat(B)` when reading from :math:`B.`
///
///      validation_mode - [in]
///       * A flag used to indicate what checks should be made for validity of :math:`(A,\texttt{diag},\texttt{uplo}).`
///       * If positive, then all checks will be made to ensure that we can correctly apply :math:`\op(\mtxA)^{-1}` to vectors;
///         these checks take :math:`O(A\texttt{.nnz})` time.
///       * If zero, then only checks of cost :math:`O(A\texttt{.n_rows})` are performed.
///       * If negative, then only checks of cost :math:`O(1)` are performed.
///       * The specific default value of this optional argument may change in future releases of RandBLAS.
///
/// @endverbatim
template <SparseMatrix SpMat, typename T = SpMat::scalar_t>
void trsm(
    blas::Layout layout, blas::Op opA, T alpha, const SpMat &A, blas::Uplo uplo, blas::Diag diag, int64_t n, T *B, int64_t ldb,
    int validation_mode = 1
) {
    using blas::Op;
    using blas::Uplo;

    if (opA == Op::Trans) {
        using sint_t = typename SpMat::index_t;
        constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
        constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
        auto trans_uplo = (uplo == Uplo::Lower) ? Uplo::Upper : Uplo::Lower;
        if constexpr (is_csc) {
            auto At = RandBLAS::sparse_data::conversions::transpose_as_csr(A);
            trsm(layout, Op::NoTrans, alpha, At, trans_uplo, diag, n, B, ldb);
        } else if constexpr (is_csr) {
            auto At = RandBLAS::sparse_data::conversions::transpose_as_csc(A);
            trsm(layout, Op::NoTrans, alpha, At, trans_uplo, diag, n, B, ldb);
        } else {
            randblas_require(false);
        }
        return; 
    }

    randblas_require( A.n_rows == A.n_cols );
    randblas_require( A.index_base == IndexBase::Zero );

    using sint_t = typename SpMat::index_t;
    constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
    constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
    randblas_require(is_csr || is_csc);

    if (validation_mode >= 0) trsm_matrix_validation( A, uplo, diag, validation_mode - 1 );

    int64_t m = A.n_rows;
    if (layout == blas::Layout::ColMajor) {
        randblas_require(ldb >= m);
        for (int64_t i = 0; i < n; ++i)
            RandBLAS::util::safe_scal(m, alpha, &B[i*ldb], 1);
    } else {
        randblas_require(ldb >= n);
        for (int64_t i = 0; i < m; ++i)
            RandBLAS::util::safe_scal(n, alpha, &B[i*ldb], 1);
    }

    if (alpha == static_cast<T>(0))
        return;

    if constexpr (is_csr) {
        sparse_data::csr::trsm_jki_p11(layout, uplo, diag, n, A, B, ldb);
    } else {
        sparse_data::csc::trsm_jki_p11(layout, uplo, diag, n, A, B, ldb);
    }
    return;
}

} // end namespace RandBLAS::sparse_data
