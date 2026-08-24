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
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include <blas.hh>

namespace RandBLAS::sparse_data {


// =============================================================================
// CSR fallback for the symmetric sparse-times-dense kernel (side=Left).
//
// Accumulates Y += alpha * A * B, where A is m-by-m symmetric with only the
// triangle named by uplo structurally stored, and B, Y are dense m-by-n.
// The caller (the spsymm dispatcher) has already validated the arguments,
// applied beta scaling to Y, and normalized side=Right away, so this kernel
// is a pure accumulator. Entries outside the named triangle are silently
// skipped, so the kernel is robust against callers who store both triangles
// by mistake (it just behaves like the "correctly stored" case).
//
// The loop is column-driven: each dense right-hand-side column c of B/Y is
// processed independently, so the outer loop parallelizes without races and
// the inner updates are unit-stride in ColMajor. Each stored off-diagonal
// entry A(i, j) = v contributes twice per column (the entry itself and the
// implied symmetric A(j, i) = v); diagonal entries contribute once.
template <typename T, SignedInteger sint_t>
void csr_spsymm(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    T alpha,
    const CSRMatrix<T, sint_t>& A,
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
        for (int64_t i = 0; i < A.n_rows; ++i) {
            for (int64_t p = A.rowptr[i]; p < A.rowptr[i+1]; ++p) {
                int64_t j = (int64_t) A.colidxs[p];
                if (upper ? (j < i) : (j > i)) continue;
                T av = alpha * A.vals[p];
                Y_c[i * irs_y] += av * B_c[j * irs_b];
                if (i != j)
                    Y_c[j * irs_y] += av * B_c[i * irs_b];
            }
        }
    }
}

} // namespace RandBLAS::sparse_data
