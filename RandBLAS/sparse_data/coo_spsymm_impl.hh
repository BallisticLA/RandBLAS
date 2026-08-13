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
#include <blas.hh>

namespace RandBLAS::sparse_data {

// =============================================================================
/// COO fallback for symmetric sparse-times-dense (side=Left).
///
/// Accumulates Y += alpha * A * B over the stored (row, col, val) triples;
/// no assumption on their order. For uplo=Upper, structurally populated
/// entries satisfy row <= col; for uplo=Lower, row >= col. Entries outside
/// the named triangle are silently skipped. The caller (the spsymm
/// dispatcher) has already validated the arguments, applied beta scaling to
/// Y, and normalized side=Right away, so this kernel is a pure accumulator.
///
/// Column-driven for the same reasons as csr_spsymm: the outer loop over
/// dense right-hand-side columns is race-free under OpenMP and the inner
/// updates are unit-stride in ColMajor.
template <typename T, typename sint_t>
void coo_spsymm(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    T alpha,
    const COOMatrix<T, sint_t>& A,
    const T* B, int64_t ldb,
    T* Y, int64_t ldy
) {
    (void) m;
    // Plain locals rather than structured bindings: clang does not (yet)
    // support referencing structured bindings inside OpenMP regions.
    stride_64t sb = layout_to_strides(layout, ldb);
    stride_64t sy = layout_to_strides(layout, ldy);
    int64_t irs_b = sb.inter_row_stride, ics_b = sb.inter_col_stride;
    int64_t irs_y = sy.inter_row_stride, ics_y = sy.inter_col_stride;
    bool upper = (uplo == blas::Uplo::Upper);

    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < n; ++c) {
        const T* B_c = &B[c * ics_b];
        T*       Y_c = &Y[c * ics_y];
        for (int64_t p = 0; p < A.nnz; ++p) {
            int64_t i = (int64_t) A.rows[p];
            int64_t j = (int64_t) A.cols[p];
            if (upper ? (j < i) : (j > i)) continue;
            T av = alpha * A.vals[p];
            Y_c[i * irs_y] += av * B_c[j * irs_b];
            if (i != j)
                Y_c[j * irs_y] += av * B_c[i * irs_b];
        }
    }
}

} // namespace RandBLAS::sparse_data
