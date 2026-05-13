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
// Known limitations that trigger a fallback:
//   - side == Right: MKL's mkl_sparse_d_mm has no side parameter; the
//     symmetric matrix is always on the left of the dense block. The
//     transpose-trick to express side=Right via layout flips depends on
//     leading-dim assumptions that don't always hold; safer to fall back.
//   - CSC format: MKL's mkl_sparse_d_mm returns NOT_SUPPORTED for CSC even
//     with a symmetric descriptor (mirrors the behavior in mkl_left_spmm).
//   - Index type mismatched with MKL_INT.
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

    if (side != blas::Side::Left)
        return false;
    if constexpr (is_csc)
        return false;

    auto h = make_mkl_handle(A);

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descr.mode = (uplo == blas::Uplo::Upper)
        ? SPARSE_FILL_MODE_UPPER
        : SPARSE_FILL_MODE_LOWER;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    sparse_status_t status;
    if constexpr (std::is_same_v<T, double>) {
        status = mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE, alpha, h.handle, descr,
            to_mkl_layout(layout),
            B, (MKL_INT)n, (MKL_INT)ldb,
            beta, Y, (MKL_INT)ldy
        );
    } else if constexpr (std::is_same_v<T, float>) {
        status = mkl_sparse_s_mm(
            SPARSE_OPERATION_NON_TRANSPOSE, alpha, h.handle, descr,
            to_mkl_layout(layout),
            B, (MKL_INT)n, (MKL_INT)ldb,
            beta, Y, (MKL_INT)ldy
        );
    } else {
        static_assert(sizeof(T) == 0, "MKL sparse BLAS only supports float and double.");
    }

    // Some MKL versions return NOT_SUPPORTED for combinations we couldn't
    // predict. Don't throw -- signal fallback.
    if (status == SPARSE_STATUS_NOT_SUPPORTED)
        return false;

    check_mkl_status(status, "mkl_sparse_mm (symmetric)");
    (void) m;  // m is implied by A.n_rows; kept for signature symmetry
    return true;
}

} // namespace RandBLAS::sparse_data::mkl

#endif // RandBLAS_HAS_MKL
