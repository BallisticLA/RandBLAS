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

#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_spsymm_impl.hh"  // for internal::apply_beta_scale
#include <blas.hh>

namespace RandBLAS::sparse_data {

// =============================================================================
/// COO fallback for symmetric sparse-times-dense.
///
/// Iterates over the (row, col, val) triples directly. For uplo=Upper,
/// structurally populated entries satisfy row <= col; for uplo=Lower,
/// row >= col. Entries outside the named triangle are silently skipped.
/// No assumption on the order of the COO entries.
template <typename T, typename sint_t>
void coo_spsymm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    T alpha,
    const COOMatrix<T, sint_t>& A,
    const T* B, int64_t ldb,
    T beta,
    T* Y, int64_t ldy
) {
    randblas_require(A.n_rows == A.n_cols);
    int64_t k = (side == blas::Side::Left) ? m : n;
    randblas_require(A.n_rows == k);

    internal::apply_beta_scale(layout, m, n, beta, Y, ldy);
    if (alpha == T(0)) return;

    const bool col_major = (layout == blas::Layout::ColMajor);

    for (int64_t p = 0; p < A.nnz; ++p) {
        sint_t i = A.rows[p];
        sint_t j = A.cols[p];
        if (uplo == blas::Uplo::Upper && j < i) continue;
        if (uplo == blas::Uplo::Lower && j > i) continue;
        T av = alpha * A.vals[p];

        if (side == blas::Side::Left) {
            if (col_major) {
                blas::axpy(n, av, &B[j], ldb, &Y[i], ldy);
                if (j != i)
                    blas::axpy(n, av, &B[i], ldb, &Y[j], ldy);
            } else {
                blas::axpy(n, av, &B[j*ldb], 1, &Y[i*ldy], 1);
                if (j != i)
                    blas::axpy(n, av, &B[i*ldb], 1, &Y[j*ldy], 1);
            }
        } else {
            if (col_major) {
                blas::axpy(m, av, &B[i*ldb], 1, &Y[j*ldy], 1);
                if (j != i)
                    blas::axpy(m, av, &B[j*ldb], 1, &Y[i*ldy], 1);
            } else {
                blas::axpy(m, av, &B[i], ldb, &Y[j], ldy);
                if (j != i)
                    blas::axpy(m, av, &B[j], ldb, &Y[i], ldy);
            }
        }
    }
}

} // namespace RandBLAS::sparse_data
