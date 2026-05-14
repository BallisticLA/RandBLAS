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
#include "RandBLAS/util.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include "RandBLAS/sparse_data/spsymm_internal.hh"
#include <blas.hh>

namespace RandBLAS::sparse_data {

// =============================================================================
/// CSC fallback for symmetric sparse-times-dense.
///
/// Same semantics as the CSR variant; iteration is column-driven via
/// colptr/rowidxs/vals. For CSC stored with uplo=Upper, structurally
/// populated entries satisfy row <= col; for uplo=Lower, row >= col.
/// Entries outside the named triangle are silently skipped.
template <typename T, typename sint_t>
void csc_spsymm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    T alpha,
    const CSCMatrix<T, sint_t>& A,
    const T* B, int64_t ldb,
    T beta,
    T* Y, int64_t ldy
) {
    randblas_require(A.n_rows == A.n_cols);
    int64_t k = (side == blas::Side::Left) ? m : n;
    randblas_require(A.n_rows == k);

    RandBLAS::util::lascl(layout, m, n, beta, Y, ldy);
    if (alpha == T(0)) return;

    if (side == blas::Side::Left) {
        // Y = alpha * A * B + ...   (A is m-by-m)
        for (int64_t j = 0; j < m; ++j) {
            for (int64_t p = A.colptr[j]; p < A.colptr[j+1]; ++p) {
                sint_t i = A.rowidxs[p];
                if (uplo == blas::Uplo::Upper && i > j) continue;
                if (uplo == blas::Uplo::Lower && i < j) continue;
                T av = alpha * A.vals[p];
                internal::spsymm_scatter_left(layout, n, av, i, j, B, ldb, Y, ldy);
            }
        }
    } else {
        // Y = alpha * B * A + ...   (A is n-by-n)
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t p = A.colptr[j]; p < A.colptr[j+1]; ++p) {
                sint_t i = A.rowidxs[p];
                if (uplo == blas::Uplo::Upper && i > j) continue;
                if (uplo == blas::Uplo::Lower && i < j) continue;
                T av = alpha * A.vals[p];
                internal::spsymm_scatter_right(layout, m, av, i, j, B, ldb, Y, ldy);
            }
        }
    }
}

} // namespace RandBLAS::sparse_data
