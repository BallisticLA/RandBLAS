// Copyright, 2026. See LICENSE for copyright holder information.
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
#include "RandBLAS/util.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include "RandBLAS/sparse_data/coo_spsymm_impl.hh"
#include "RandBLAS/sparse_data/csr_spsymm_impl.hh"
#include "RandBLAS/sparse_data/csc_spsymm_impl.hh"
#include "RandBLAS/sparse_data/symmetric.hh"
// Pulls in the two general RandBLAS::spmm overloads so any TU that sees the
// Symmetric<SpMat> overload below sees the full RandBLAS::spmm overload set,
// not just the one declared in this file.
#include "RandBLAS/sparse_data/spmm_dispatch.hh"
#include "RandBLAS/config.h"

#if defined(RandBLAS_HAS_MKL)
#include "RandBLAS/sparse_data/mkl_spsymm_impl.hh"
#endif

#include <type_traits>
#include <vector>


namespace RandBLAS::sparse_data {

// =============================================================================
/// Dispatched symmetric sparse-times-dense multiplication.
///
/// @verbatim embed:rst:leading-slashes
/// Computes
///
/// .. math::
///     \mat(C) = \alpha \cdot \op_{\ttt{side}}(\mat(A), \mat(B)) + \beta \cdot \mat(C),
///
/// where:
///
///   - :math:`\mat(A)` is a sparse symmetric matrix stored in COO, CSR, or
///     CSC format, with zero-based indices. Only the triangle named by
///     :math:`\ttt{uplo}` is read. The opposite triangle is implied by
///     symmetry. The matrix is required to be square.
///   - :math:`\mat(B)` and :math:`\mat(C)` are dense, both m-by-n.
///   - For :math:`\ttt{side} = \ttt{Left}`, :math:`\mat(A)` is m-by-m and the
///     operation is :math:`\mat(C) = \alpha A B + \beta C`.
///   - For :math:`\ttt{side} = \ttt{Right}`, :math:`\mat(A)` is n-by-n and
///     the operation is :math:`\mat(C) = \alpha B A + \beta C`.
///
/// Arguments are validated at entry. Empty products (a zero dimension, a
/// structurally empty :math:`\mat(A)`, or :math:`\alpha = 0`) leave
/// :math:`\beta \cdot \mat(C)`.
/// @endverbatim
//
// Dispatch mechanics (see also sparse_data/DevNotes.md): side=Right is
// normalized to side=Left at entry (A == A^T, so C = B*A is C^T = A*B^T with
// the B/C buffers reinterpreted in the flipped layout). On MKL builds with
// the index width matching MKL_INT, mkl_spsymm runs with the caller's beta
// (SPARSE_MATRIX_TYPE_SYMMETRIC descriptor; CSC consumed as a
// CSR-of-transpose view with uplo flipped). The per-format hand kernels are
// pure accumulators reached on non-MKL builds, index-width mismatch, or a
// runtime NOT_SUPPORTED from MKL; beta is applied by lascl just before them,
// mirroring left_spmm.
template <SparseMatrix SpMat, typename T = typename SpMat::scalar_t>
void spsymm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    T alpha,
    const SpMat& A,
    const T* B, int64_t ldb,
    T beta,
    T* C, int64_t ldc
) {
    using blas::Layout;
    if (side == blas::Side::Right) {
        // A == A^T, so C = alpha B A + beta C is C^T = alpha A B^T + beta C^T
        // with B^T and C^T read from the same buffers in the flipped layout.
        // The dimensions of C^T are n-by-m; uplo is unchanged (the equation
        // transpose does not change which physical entries of A are stored).
        spsymm(flipped_layout(layout), blas::Side::Left, uplo, n, m, alpha, A, B, ldb, beta, C, ldc);
        return;
    }

    using sint_t = typename SpMat::index_t;
    constexpr bool is_coo = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
    constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
    constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
    static_assert(is_coo || is_csr || is_csc,
                  "RandBLAS::sparse_data::spsymm requires COO, CSR, or CSC.");

    // side is Left from here on: A is m-by-m, B and C are m-by-n.
    randblas_require(A.index_base == IndexBase::Zero);
    randblas_require(A.n_rows == A.n_cols);
    randblas_require(A.n_rows == m);
    if (layout == Layout::ColMajor) {
        randblas_require(ldb >= m);
        randblas_require(ldc >= m);
    } else {
        randblas_require(ldb >= n);
        randblas_require(ldc >= n);
    }

    // Empty products: no output elements, or nothing to accumulate beyond
    // beta * C. Handled here because MKL rejects some valid empty sparse
    // matrices at handle creation (same contract as left_spmm).
    if (m == 0 || n == 0)
        return;
    if (alpha == T(0) || A.nnz == 0) {
        RandBLAS::util::lascl(layout, m, n, beta, C, ldc);
        return;
    }

#if defined(RandBLAS_HAS_MKL)
    if constexpr (sizeof(sint_t) == sizeof(MKL_INT)) {
        // MKL applies beta itself.
        bool handled = mkl::mkl_spsymm(
            layout, uplo, m, n, alpha, A, B, ldb, beta, C, ldc
        );
        if (handled) return;
    }
#endif

    // Fallback path: apply beta here, then run a pure-accumulator hand
    // kernel. Safe after an MKL NOT_SUPPORTED, which is a parameter
    // validation result: C is untouched when it fires.
    RandBLAS::util::lascl(layout, m, n, beta, C, ldc);
    if constexpr (is_csr) {
        csr_spsymm(layout, uplo, m, n, alpha, A, B, ldb, C, ldc);
    } else if constexpr (is_csc) {
        csc_spsymm(layout, uplo, m, n, alpha, A, B, ldb, C, ldc);
    } else if constexpr (is_coo) {
        coo_spsymm(layout, uplo, m, n, alpha, A, B, ldb, C, ldc);
    }
}

// =============================================================================
/// Case D: sparse-symmetric A times sparse B, dense output.
///
/// @verbatim embed:rst:leading-slashes
/// Computes the same operation as the Case-C overload, with :math:`\mat(B)`
/// sparse (COO, CSR, or CSC; any index type). side=Right is normalized to
/// side=Left at entry via the same layout-flip identity, with
/// :math:`B^T` obtained as the lightweight ``B.transpose()`` view.
///
/// Primary path (MKL builds with index widths matching MKL_INT): expand
/// :math:`A`'s stored triangle into a general sparse matrix in
/// :math:`O(\ttt{nnz})` memory (``expand_symmetric_to_general``), then call
/// the existing sparse-times-sparse dense-output routine
/// (``mkl_spgemm_to_dense``) with a GENERAL descriptor. The expansion has to
/// happen on the RandBLAS side either way: MKL's ``mkl_sparse_sp2m`` returns
/// ``SPARSE_STATUS_NOT_SUPPORTED`` when either operand's ``matrix_descr`` is
/// ``SPARSE_MATRIX_TYPE_SYMMETRIC``, and ``mkl_sparse_?_spmmd`` accepts no
/// descriptor at all. Keeping :math:`B` sparse keeps the cost proportional
/// to the actual nonzero structure.
///
/// Fallback path (non-MKL builds, or index width mismatched with MKL_INT):
/// densify :math:`B` into a temporary dense buffer in the caller's layout
/// (cost: an :math:`O(m \cdot n)` temporary, a fallback-only property) and
/// compose through the Case-C overload above.
/// @endverbatim
template <SparseMatrix SpMatA, SparseMatrix SpMatB,
          typename T = typename SpMatA::scalar_t>
void spsymm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    T alpha,
    const SpMatA& A,
    const SpMatB& B,
    T beta,
    T* C, int64_t ldc
) {
    using blas::Layout;
    static_assert(std::is_same_v<T, typename SpMatB::scalar_t>,
                  "Case D: A and B must share scalar_t.");

    if (side == blas::Side::Right) {
        // Same identity as Case C; B^T is a lightweight transpose view.
        auto Bt = B.transpose();
        spsymm(flipped_layout(layout), blas::Side::Left, uplo, n, m, alpha, A, Bt, beta, C, ldc);
        return;
    }

    // side is Left from here on: A is m-by-m, B and C are m-by-n.
    randblas_require(A.index_base == IndexBase::Zero);
    randblas_require(B.index_base == IndexBase::Zero);
    randblas_require(A.n_rows == A.n_cols);
    randblas_require(A.n_rows == m);
    randblas_require(B.n_rows == m);
    randblas_require(B.n_cols == n);
    if (layout == Layout::ColMajor) {
        randblas_require(ldc >= m);
    } else {
        randblas_require(ldc >= n);
    }

    // Empty products: no output elements, or nothing to accumulate beyond
    // beta * C. Handled here because MKL rejects some valid empty sparse
    // matrices at handle creation (same contract as left_spmm).
    if (m == 0 || n == 0)
        return;
    if (alpha == T(0) || A.nnz == 0 || B.nnz == 0) {
        RandBLAS::util::lascl(layout, m, n, beta, C, ldc);
        return;
    }

    using sint_A = typename SpMatA::index_t;
    using sint_B = typename SpMatB::index_t;

#if defined(RandBLAS_HAS_MKL)
    if constexpr (sizeof(sint_A) == sizeof(MKL_INT) && sizeof(sint_B) == sizeof(MKL_INT)) {
        // mkl_spgemm_to_dense applies beta itself.
        auto A_general = expand_symmetric_to_general(A, uplo);
        mkl::mkl_spgemm_to_dense(
            layout, blas::Op::NoTrans, alpha, A_general, B, beta, C, ldc
        );
        return;
    }
#endif

    // Fallback: densify B into a tight buffer in the caller's layout and
    // compose through Case C, which handles beta itself.
    int64_t ldb_dense = (layout == Layout::ColMajor) ? m : n;
    std::vector<T> B_dense(static_cast<size_t>(m) * static_cast<size_t>(n), T(0));

    if constexpr (std::is_same_v<SpMatB, COOMatrix<T, sint_B>>) {
        coo::coo_to_dense(B, layout, B_dense.data());
    } else if constexpr (std::is_same_v<SpMatB, CSRMatrix<T, sint_B>>) {
        csr::csr_to_dense(B, layout, B_dense.data());
    } else if constexpr (std::is_same_v<SpMatB, CSCMatrix<T, sint_B>>) {
        csc::csc_to_dense(B, layout, B_dense.data());
    } else {
        static_assert(sizeof(SpMatB) == 0,
                      "RandBLAS::sparse_data::spsymm: SpMatB must be COO, CSR, or CSC.");
    }

    spsymm(layout, blas::Side::Left, uplo, m, n,
           alpha, A, B_dense.data(), ldb_dense, beta, C, ldc);
}

} // end namespace RandBLAS::sparse_data


namespace RandBLAS {

// =============================================================================
/// Convenience wrapper for symmetric sparse-times-dense matmul.
///
/// Computes \math{C = \alpha A B + \beta C}, where \math{A} is the symmetric
/// sparse matrix and only the triangle named by uplo is read. \math{A} must
/// be square; its order is taken from the matrix itself, so the only
/// dimension argument is \math{n}, the number of columns in \math{B} and
/// \math{C}.
template <SparseMatrix SpMat, typename T = typename SpMat::scalar_t>
inline void spsymm(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    T alpha,
    const SpMat& A,
    const T* B, int64_t ldb,
    T beta,
    T* C, int64_t ldc
) {
    RandBLAS::sparse_data::spsymm(
        layout, blas::Side::Left, uplo, A.n_rows, n,
        alpha, A, B, ldb, beta, C, ldc
    );
}

// =============================================================================
/// Overload of spmm for a sparse matrix tagged as symmetric.
///
/// Computes \math{C = \alpha A B + \beta C}, where the Symmetric wrapper
/// supplies the matrix and the triangle to read; the opposite triangle is
/// implied by symmetry. Same dimension convention as the spsymm overload
/// above: the only dimension argument is \math{n}, the number of columns in
/// \math{B} and \math{C}.
template <SparseMatrix SpMat, typename T = typename SpMat::scalar_t>
inline void spmm(
    blas::Layout layout,
    int64_t n,
    T alpha,
    const Symmetric<SpMat>& A_sym,
    const T* B, int64_t ldb,
    T beta,
    T* C, int64_t ldc
) {
    RandBLAS::spsymm(layout, A_sym.uplo, n, alpha, A_sym.A, B, ldb, beta, C, ldc);
}

} // end namespace RandBLAS
