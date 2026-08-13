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
#include "RandBLAS/util.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"

#include <blas.hh>


namespace RandBLAS::sparse_data {

// =============================================================================
// COO sparse-from-left times dense symmetric A into dense B:
//     B = alpha * submat(Scoo) * sym(A, uplo) + B
//
//   - sym(A, uplo) is n-by-n dense symmetric, only the `uplo` triangle stored.
//   - submat(Scoo) is the d-by-n window of Scoo at (ro_s, co_s).
//   - B is d-by-n dense, layout-matched.
//
// The loop is column-driven: each column c of B accumulates, over the window
// nonzeros (row_S, col_S, v) of Scoo, the contribution
//     B(row_S - ro_s, c) += alpha * v * sym(A, uplo)(col_S - co_s, c),
// where the symmetric element read resolves to the stored triangle by
// swapping the index pair when it falls outside it. Columns are independent,
// so the outer loop parallelizes without races, and the accesses to A and B
// walk down single columns in ColMajor.
//
// Beta-scaling of B and the alpha==0 short-circuit are the caller's
// responsibility (so that this kernel can be composed with other accumulators
// without redundant scaling).
// =============================================================================
template <typename T, SignedInteger sint_t>
void coo_lsksys(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t d,
    int64_t n,
    T alpha,
    const COOMatrix<T, sint_t> &Scoo,
    int64_t ro_s,
    int64_t co_s,
    const T *A,
    int64_t lda,
    T *B,
    int64_t ldb
) {
    if (alpha == T(0)) return;

    // Plain locals rather than structured bindings: clang does not (yet)
    // support referencing structured bindings inside OpenMP regions.
    stride_64t sa = layout_to_strides(layout, lda);
    stride_64t sb = layout_to_strides(layout, ldb);
    int64_t irs_a = sa.inter_row_stride, ics_a = sa.inter_col_stride;
    int64_t irs_b = sb.inter_row_stride, ics_b = sb.inter_col_stride;
    bool upper = (uplo == blas::Uplo::Upper);

    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < n; ++c) {
        T* B_c = &B[c * ics_b];
        for (int64_t p = 0; p < Scoo.nnz; ++p) {
            int64_t row_S = (int64_t) Scoo.rows[p];
            int64_t col_S = (int64_t) Scoo.cols[p];
            if (row_S < ro_s || row_S >= ro_s + d) continue;
            if (col_S < co_s || col_S >= co_s + n) continue;
            int64_t i = row_S - ro_s;  // B row
            int64_t r = col_S - co_s;  // row of sym(A) contributing to column c
            // sym(A, uplo)(r, c): read (r, c) if it lies in the stored
            // triangle, else the mirrored (c, r).
            bool mirrored = upper ? (r > c) : (r < c);
            T a_rc = mirrored ? A[c * irs_a + r * ics_a]
                              : A[r * irs_a + c * ics_a];
            B_c[i * irs_b] += alpha * Scoo.vals[p] * a_rc;
        }
    }
}


// =============================================================================
// COO sparse-from-right times dense symmetric A into dense B:
//     B = alpha * sym(A, uplo) * submat(Scoo) + B
//
//   - sym(A, uplo) is n-by-n; submat(Scoo) is the n-by-d window at
//     (ro_s, co_s); B is n-by-d.
//
// Reduction to coo_lsksys via the transpose identity: B = A * S implies
// B^T = S^T * A^T = S^T * A (A symmetric). Reading the A and B buffers in
// the flipped layout presents them as A^T (= A, with the stored triangle
// name flipped) and B^T; S^T is the lightweight Scoo.transpose() view, with
// the window offsets swapped.
//
// Beta-scaling of B and the alpha==0 short-circuit are the caller's
// responsibility.
// =============================================================================
template <typename T, SignedInteger sint_t>
void coo_rsksys(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    int64_t d,
    T alpha,
    const T *A,
    int64_t lda,
    const COOMatrix<T, sint_t> &Scoo,
    int64_t ro_s,
    int64_t co_s,
    T *B,
    int64_t ldb
) {
    auto flipped_layout = (layout == blas::Layout::ColMajor)
        ? blas::Layout::RowMajor
        : blas::Layout::ColMajor;
    auto flipped_uplo = (uplo == blas::Uplo::Upper)
        ? blas::Uplo::Lower
        : blas::Uplo::Upper;
    auto St = Scoo.transpose();
    coo_lsksys(flipped_layout, flipped_uplo, d, n, alpha,
               St, co_s, ro_s, A, lda, B, ldb);
}

} // end namespace RandBLAS::sparse_data
