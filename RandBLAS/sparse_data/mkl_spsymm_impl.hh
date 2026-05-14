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

#include "RandBLAS/config.h"

#if defined(RandBLAS_HAS_MKL)

#if defined(BLAS_ILP64) && !defined(MKL_ILP64)
#define MKL_ILP64
#endif

#include <mkl_spblas.h>
#include <type_traits>

#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include "RandBLAS/sparse_data/mkl_spmm_impl.hh" // reuse make_mkl_handle, to_mkl_layout, check_mkl_status

namespace RandBLAS::sparse_data::mkl {

// ============================================================================
// MKL-accelerated symmetric SpMM: Y = alpha * A * B + beta * Y (side=Left only).
//   A is symmetric sparse (one triangle stored, named by uplo); B and Y dense.
//
// Returns true if MKL handled the call, false to signal fallback to the
// hand-rolled per-format kernel. MKL applies alpha and beta internally; the
// caller therefore passes them through unchanged.
//
// Known limitation that triggers a fallback:
//   - Index type mismatched with MKL_INT.
//
// Both Side values are handled:
//   - side=Left (Y = alpha*A*B + beta*Y): direct call to mkl_sparse_d_mm
//     with the symmetric descriptor.
//   - side=Right (Y = alpha*B*A + beta*Y): A is symmetric so A == A^T,
//     and (B*A)^T = A^T * B^T = A * B^T. We tell MKL to compute A*B^T
//     into Y^T by flipping the MKL layout flag. Concretely, the same
//     user buffers for B and Y are reinterpreted in the opposite layout
//     (ColMajor <-> RowMajor); ldb and ldy carry through unchanged
//     because the reinterpretation has the same leading-dim semantics
//     (rows-of-RowMajor have the same stride as cols-of-ColMajor).
//
// CSC handling: MKL's mkl_sparse_d_mm returns NOT_SUPPORTED for CSC even
// with a symmetric descriptor. We work around this by taking the
// CSC.transpose() view (a lightweight CSR view over the same buffers,
// since A is symmetric so A == A^T) and recursing. The triangle the user
// named in the CSC is in the *opposite* triangle of the CSR view (a CSC
// Upper entry at (i, j) with i <= j becomes a CSR-view entry at (j, i)
// with j >= i, i.e., Lower in the CSR view), so the recursive call flips
// uplo.
// ============================================================================
template <SparseMatrix SpMat, typename T = typename SpMat::scalar_t>
bool mkl_spsymm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    T alpha,
    const SpMat &A,
    const T *B,
    int64_t ldb,
    T beta,
    T *Y,
    int64_t ldy
) {
    using sint_t = typename SpMat::index_t;
    constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;

    if constexpr (is_csc) {
        // Symmetric A: A == A^T. The CSC->CSR view is lightweight (same
        // buffers, reinterpreted) and gives us a CSR matrix MKL accepts;
        // the structurally stored triangle moves from {Upper,Lower} to
        // the opposite side in the CSR view.
        auto At = A.transpose();
        blas::Uplo uplo_flipped = (uplo == blas::Uplo::Upper)
            ? blas::Uplo::Lower
            : blas::Uplo::Upper;
        return mkl_spsymm(layout, side, uplo_flipped, m, n,
                          alpha, At, B, ldb, beta, Y, ldy);
    }

    auto h = make_mkl_handle(A);

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descr.mode = (uplo == blas::Uplo::Upper)
        ? SPARSE_FILL_MODE_UPPER
        : SPARSE_FILL_MODE_LOWER;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    // For side=Right, reinterpret B and Y in the opposite layout to
    // present them to MKL as B^T and Y^T; MKL then computes Y^T = A*B^T
    // which is the transpose of Y = B*A. The number of MKL-side
    // right-hand-side columns is m (the user's row count) rather than n
    // (the user's col count, = A's order).
    blas::Layout mkl_layout = (side == blas::Side::Left)
        ? layout
        : (layout == blas::Layout::ColMajor ? blas::Layout::RowMajor
                                            : blas::Layout::ColMajor);
    int64_t n_rhs = (side == blas::Side::Left) ? n : m;

    sparse_status_t status = mkl_sparse_mm_call(
        SPARSE_OPERATION_NON_TRANSPOSE, alpha, h.handle, descr,
        to_mkl_layout(mkl_layout),
        B, n_rhs, ldb, beta, Y, ldy
    );

    // Some MKL versions return NOT_SUPPORTED for combinations we couldn't
    // predict. Don't throw -- signal fallback.
    if (status == SPARSE_STATUS_NOT_SUPPORTED)
        return false;

    check_mkl_status(status, "mkl_sparse_mm (symmetric)");
    return true;
}

} // namespace RandBLAS::sparse_data::mkl

#endif // RandBLAS_HAS_MKL
