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
//     B = alpha * submat(Scoo) * sym(A, uplo) + beta * B
//
//   - sym(A, uplo) is n-by-n dense symmetric, only the `uplo` triangle stored.
//   - submat(Scoo) is the d-by-n window of Scoo at (ro_s, co_s).
//   - B is d-by-n dense, layout-matched.
//
// For each stored nonzero (row_S, col_S, v) of Scoo inside the window, the
// kernel contributes alpha*v * (row col_S of A) to row (row_S - ro_s) of B.
// The row of symmetric A is assembled from the stored triangle as two
// blas::axpy calls -- one along the stored column up to (and including) the
// diagonal, one along the stored row past the diagonal -- chosen by `uplo`
// and the matrix layout.
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

    const bool col_major = (layout == blas::Layout::ColMajor);

    for (int64_t p = 0; p < Scoo.nnz; ++p) {
        sint_t row_S = Scoo.rows[p];
        sint_t col_S = Scoo.cols[p];
        if (row_S < ro_s || row_S >= ro_s + d) continue;
        if (col_S < co_s || col_S >= co_s + n) continue;
        int64_t i = static_cast<int64_t>(row_S) - ro_s;  // B row
        int64_t j = static_cast<int64_t>(col_S) - co_s;  // A row index (sym A read as row j)
        T av = alpha * Scoo.vals[p];

        // B[i, :] += av * row_j(A_sym), split by uplo into two contiguous
        // ranges of A so each becomes a single axpy.
        if (uplo == blas::Uplo::Upper) {
            // row j of A:
            //   c in [0, j]: read A[c, j]      (column j of stored Upper, rows 0..j)
            //   c in (j, n-1]: read A[j, c]    (row j of stored Upper, cols j+1..n-1)
            if (col_major) {
                blas::axpy(j + 1, av, &A[j * lda], 1, &B[i], ldb);
                blas::axpy(n - j - 1, av, &A[j + (j + 1) * lda], lda, &B[i + (j + 1) * ldb], ldb);
            } else {
                blas::axpy(j + 1, av, &A[j], lda, &B[i * ldb], 1);
                blas::axpy(n - j - 1, av, &A[j * lda + j + 1], 1, &B[i * ldb + j + 1], 1);
            }
        } else {
            // Lower-stored:
            //   c in [0, j]: read A[j, c]      (row j of stored Lower, cols 0..j)
            //   c in (j, n-1]: read A[c, j]    (column j of stored Lower, rows j+1..n-1)
            if (col_major) {
                blas::axpy(j + 1, av, &A[j], lda, &B[i], ldb);
                blas::axpy(n - j - 1, av, &A[(j + 1) + j * lda], 1, &B[i + (j + 1) * ldb], ldb);
            } else {
                blas::axpy(j + 1, av, &A[j * lda], 1, &B[i * ldb], 1);
                blas::axpy(n - j - 1, av, &A[(j + 1) * lda + j], lda, &B[i * ldb + j + 1], 1);
            }
        }
    }
}


// =============================================================================
// COO sparse-from-right times dense symmetric A into dense B:
//     B = alpha * sym(A, uplo) * submat(Scoo) + beta * B
//
// Same two-axpy scatter pattern as coo_lsksys, but on columns of A: the
// column of A indexed by row_S of S contributes into the column of B indexed
// by col_S - co_s.
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
    if (alpha == T(0)) return;

    const bool col_major = (layout == blas::Layout::ColMajor);

    for (int64_t p = 0; p < Scoo.nnz; ++p) {
        sint_t row_S = Scoo.rows[p];
        sint_t col_S = Scoo.cols[p];
        if (row_S < ro_s || row_S >= ro_s + n) continue;
        if (col_S < co_s || col_S >= co_s + d) continue;
        int64_t i = static_cast<int64_t>(row_S) - ro_s;  // A column index
        int64_t j = static_cast<int64_t>(col_S) - co_s;  // B column
        T av = alpha * Scoo.vals[p];

        // B[:, j] += av * col_i(A_sym), split by uplo:
        if (uplo == blas::Uplo::Upper) {
            // col i of A:
            //   r in [0, i]: read A[r, i]    (column i of stored Upper, rows 0..i)
            //   r in (i, n-1]: read A[i, r]  (row i of stored Upper, cols i+1..n-1)
            if (col_major) {
                blas::axpy(i + 1, av, &A[i * lda], 1, &B[j * ldb], 1);
                blas::axpy(n - i - 1, av, &A[i + (i + 1) * lda], lda, &B[(i + 1) + j * ldb], 1);
            } else {
                blas::axpy(i + 1, av, &A[i], lda, &B[j], ldb);
                blas::axpy(n - i - 1, av, &A[i * lda + i + 1], 1, &B[(i + 1) * ldb + j], ldb);
            }
        } else {
            // Lower-stored col i:
            //   r in [0, i-1]: read A[i, r]  (row i of stored Lower, cols 0..i-1)
            //   r in [i, n-1]: read A[r, i]  (column i of stored Lower, rows i..n-1)
            if (col_major) {
                blas::axpy(i, av, &A[i], lda, &B[j * ldb], 1);
                blas::axpy(n - i, av, &A[i + i * lda], 1, &B[i + j * ldb], 1);
            } else {
                blas::axpy(i, av, &A[i * lda], 1, &B[j], ldb);
                blas::axpy(n - i, av, &A[i * lda + i], lda, &B[i * ldb + j], ldb);
            }
        }
    }
}

} // end namespace RandBLAS::sparse_data
