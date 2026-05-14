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
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include "RandBLAS/sparse_data/coo_spsymm_impl.hh"
#include "RandBLAS/sparse_data/csr_spsymm_impl.hh"
#include "RandBLAS/sparse_data/csc_spsymm_impl.hh"
#include "RandBLAS/sparse_data/symmetric.hh"
#include "RandBLAS/config.h"

#if defined(RandBLAS_HAS_MKL)
#include "RandBLAS/sparse_data/mkl_spsymm_impl.hh"
#endif

#include <type_traits>


namespace RandBLAS::sparse_data {

// =============================================================================
/// Dispatched symmetric sparse-times-dense kernel (Case C in
/// project-plans/randblas-symm-plan.md).
///
/// @verbatim embed:rst:leading-slashes
/// Computes
///
/// .. math::
///     \mat(Y) = \alpha \cdot \op_{\ttt{side}}(\mat(A), \mat(B)) + \beta \cdot \mat(Y),
///
/// where:
///
///   - :math:`\mat(A)` is a sparse symmetric matrix stored in COO, CSR, or
///     CSC format. Only the triangle named by :math:`\ttt{uplo}` is read.
///     The opposite triangle is implied by symmetry. The matrix is required
///     to be square.
///   - :math:`\mat(B)` and :math:`\mat(Y)` are dense, both m-by-n.
///   - For :math:`\ttt{side} = \ttt{Left}`, :math:`\mat(A)` is m-by-m and the
///     operation is :math:`\mat(Y) = \alpha A B + \beta Y`.
///   - For :math:`\ttt{side} = \ttt{Right}`, :math:`\mat(A)` is n-by-n and
///     the operation is :math:`\mat(Y) = \alpha B A + \beta Y`.
///
/// Dispatch:
///   - If RandBLAS was built with MKL and the index type matches MKL_INT,
///     try the MKL fast path (mkl_sparse_d_mm with
///     ``SPARSE_MATRIX_TYPE_SYMMETRIC`` descriptor). The MKL path covers
///     side=Left for CSR and COO; it returns false (fall back) for CSC and
///     for side=Right.
///   - Otherwise (or on MKL fall back), call the format-specific fallback
///     kernel (``csr_spsymm`` / ``csc_spsymm`` / ``coo_spsymm``).
/// @endverbatim
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
    T* Y, int64_t ldy
) {
    using sint_t = typename SpMat::index_t;
    constexpr bool is_coo = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
    constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
    constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
    static_assert(is_coo || is_csr || is_csc,
                  "RandBLAS::sparse_data::spsymm requires COO, CSR, or CSC.");

    randblas_require(A.n_rows == A.n_cols);
    int64_t k = (side == blas::Side::Left) ? m : n;
    randblas_require(A.n_rows == k);

#if defined(RandBLAS_HAS_MKL)
    if constexpr (sizeof(sint_t) == sizeof(MKL_INT)) {
        bool handled = mkl::mkl_spsymm(
            layout, side, uplo, m, n,
            alpha, A, B, ldb, beta, Y, ldy
        );
        if (handled) return;
    }
#endif

    // Fallback path
    if constexpr (is_csr) {
        csr_spsymm(layout, side, uplo, m, n, alpha, A, B, ldb, beta, Y, ldy);
    } else if constexpr (is_csc) {
        csc_spsymm(layout, side, uplo, m, n, alpha, A, B, ldb, beta, Y, ldy);
    } else if constexpr (is_coo) {
        coo_spsymm(layout, side, uplo, m, n, alpha, A, B, ldb, beta, Y, ldy);
    }
}

// =============================================================================
/// Case D: sparse-symmetric A times sparse B, dense output.
///
/// @verbatim embed:rst:leading-slashes
/// Implemented as a composition: densify ``B`` into a temporary dense buffer
/// matching the caller's ``layout``, then call the Case-C ``spsymm`` overload
/// (sparse-symm × dense) on the result. This works in any build (the MKL
/// fast path or the per-format hand kernel both apply, depending on the
/// build), and across all 3 × 3 = 9 sparse-format pairings for ``(A, B)``,
/// since the densification picks the right format-specific helper.
///
/// Why this composition rather than a single MKL ``sp2m`` call: MKL's
/// ``mkl_sparse_sp2m`` returns ``SPARSE_STATUS_NOT_SUPPORTED`` when the
/// ``matrix_descr`` on either operand is ``SPARSE_MATRIX_TYPE_SYMMETRIC``;
/// only ``GENERAL`` is supported there. ``mkl_sparse_d_spmmd`` (which writes
/// directly to dense ``C``) accepts no descriptor at all. So the symmetric
/// expansion has to happen on the RandBLAS side. Composing through Case C
/// gets it for free at the cost of a temporary dense buffer for ``B``
/// (cost: ``O(m * n)``), and for the typical RandNLA workload where ``B``
/// is a sketching operator with ``nnz(B) << m*n`` the cost is small.
///
/// The dimensions of the densified ``B`` are the same as the user's ``Y``:
/// ``m``-by-``n``, with the user's ``layout`` and a tight leading dim. The
/// densified buffer is freed when the temporary ``std::vector<T>`` goes out
/// of scope. ``Y`` itself is never touched until the Case-C call.
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
    T* Y, int64_t ldy
) {
    static_assert(std::is_same_v<T, typename SpMatB::scalar_t>,
                  "Case D: A and B must share scalar_t.");

    randblas_require(A.n_rows == A.n_cols);
    int64_t k = (side == blas::Side::Left) ? m : n;
    randblas_require(A.n_rows == k);
    randblas_require(B.n_rows == m);
    randblas_require(B.n_cols == n);

    // Densify B into a tight buffer in the caller's layout.
    int64_t ldb_dense = (layout == blas::Layout::ColMajor) ? m : n;
    std::vector<T> B_dense(static_cast<size_t>(m) * static_cast<size_t>(n), T(0));

    using sint_B = typename SpMatB::index_t;
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

    // Compose into Case C with the densified B.
    spsymm(layout, side, uplo, m, n,
           alpha, A, B_dense.data(), ldb_dense, beta, Y, ldy);
}

} // end namespace RandBLAS::sparse_data


namespace RandBLAS {

// =============================================================================
/// Convenience wrapper for symmetric sparse-times-dense matmul.
///
/// Computes :math:`\ttt{Y} = \alpha A B + \beta Y` (side=Left default), where
/// :math:`A` is the symmetric sparse matrix and only the triangle named by
/// :math:`\ttt{uplo}` is read.
template <SparseMatrix SpMat, typename T = typename SpMat::scalar_t>
inline void spsymm(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    T alpha,
    const SpMat& A,
    const T* B, int64_t ldb,
    T beta,
    T* Y, int64_t ldy
) {
    RandBLAS::sparse_data::spsymm(
        layout, blas::Side::Left, uplo, m, n,
        alpha, A, B, ldb, beta, Y, ldy
    );
}

// =============================================================================
/// Convenience overload taking a Symmetric<SpMat> wrapper. The uplo is read
/// from the wrapper. Defaults to side=Left.
template <SparseMatrix SpMat, typename T = typename SpMat::scalar_t>
inline void spsymm(
    blas::Layout layout,
    int64_t m, int64_t n,
    T alpha,
    const Symmetric<SpMat>& A_sym,
    const T* B, int64_t ldb,
    T beta,
    T* Y, int64_t ldy
) {
    RandBLAS::sparse_data::spsymm(
        layout, blas::Side::Left, A_sym.uplo, m, n,
        alpha, A_sym.A, B, ldb, beta, Y, ldy
    );
}

} // end namespace RandBLAS
