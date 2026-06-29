// Copyright, 2024. See LICENSE for copyright holder information.
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
#include "RandBLAS/sparse_data/csc_spmm_impl.hh"
#include "RandBLAS/sparse_data/csr_spmm_impl.hh"
#include <vector>
#include <algorithm>
#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

namespace RandBLAS::sparse_data::coo {

#ifdef __cpp_concepts
using RandBLAS::SignedInteger;
#else
#define SignedInteger typename
#endif


// Format choice for the COO->compressed SpMM dispatch. The per-RHS-column kernels loop
// the operator's OUTER dimension once per output column: CSC jki's outer loop is n_cols
// (= m), CSR jik's is n_rows (= d). In the ColMajor path, routing a WIDE operator (d < m)
// through the CSR jik kernel (shorter outer loop) measured 1.1-2x faster than CSC jki for
// both SASO and LASO, with no regression -- so we prefer CSR there. The RowMajor path keeps
// the bandwidth-saturating CSC kib axpy kernel. See the --csr-probe mode of
// examples/.../sketch_general_performance.cc for measurements.
inline NonzeroSort coo_spmm_target_format(
    blas::Layout layout_opB, blas::Layout layout_C, int64_t d, int64_t m
) {
    bool col_major  = (layout_opB == blas::Layout::ColMajor && layout_C == blas::Layout::ColMajor);
    bool prefer_csr = col_major && (d < m);
    return prefer_csr ? NonzeroSort::CSR : NonzeroSort::CSC;
}

// Restrict A0 to the requested d-by-m window at (ro_a, co_a): copy the in-window nonzeros into
// a fresh, memory-owning COO, shifted to local coordinates. We over-allocate to A0.nnz (the
// worst case) so the compaction is a single pass; the result's logical nnz is the in-window
// count, and its sort label is recomputed so the conversion below can hit a fast path.
template <typename T, SignedInteger sint_t>
static COOMatrix<T, sint_t> restrict_to_window(
    const COOMatrix<T, sint_t> &A0, int64_t d, int64_t m, int64_t ro_a, int64_t co_a
) {
    COOMatrix<T, sint_t> sub(d, m);
    if (A0.nnz == 0)
        return sub;
    sub.reserve(A0.nnz);
    int64_t write = 0;
    for (int64_t i = 0; i < A0.nnz; ++i) {
        auto r = A0.rows[i] - ro_a;
        auto c = A0.cols[i] - co_a;
        if (0 <= r && r < d && 0 <= c && c < m) {
            sub.rows[write] = r;
            sub.cols[write] = c;
            sub.vals[write] = A0.vals[i];
            write += 1;
        }
    }
    sub.nnz  = write;
    sub.sort = coo_arrays_determine_sort(sub.nnz, sub.rows, sub.cols);
    return sub;
}

template <typename T, SignedInteger sint_t>
static void apply_coo_via_csx(
    T alpha,
    blas::Layout layout_B,
    blas::Layout layout_C,
    int64_t d,
    int64_t n,
    int64_t m,
    const COOMatrix<T, sint_t> &A0,
    int64_t ro_a,
    int64_t co_a,
    const T *B,
    int64_t ldb,
    T *C,
    int64_t ldc
) {
    randblas_require(A0.index_base == IndexBase::Zero);

    // The exact d-by-m operator: A0 itself when it already has those dimensions, else an owning
    // copy restricted to the window. Binding "op" by reference lets the common full-operator
    // path use A0 (with its existing sort) at zero cost; "windowed" is then an empty placeholder.
    bool submatrix = (A0.n_rows != d) || (A0.n_cols != m);
    COOMatrix<T, sint_t> windowed = submatrix
        ? restrict_to_window(A0, d, m, ro_a, co_a)
        : COOMatrix<T, sint_t>(0, 0);
    const COOMatrix<T, sint_t> &op = submatrix ? windowed : A0;
    if (op.nnz == 0)
        return;  // structurally-zero operator; C was already scaled by beta upstream.

    // Choose the compressed format by the dispatch heuristic, materialize "op" in that format
    // (a zero-copy view when its sort already matches, else an O(nnz) conversion), and call the
    // matching per-RHS-column kernel -- exactly the kernels left_spmm uses for a native CSR/CSC
    // operand. coo_spmm_target_format only selects CSR for ColMajor/ColMajor, which is why only
    // apply_csr_jik_p11 appears here; CSC is the catch-all and keeps its RowMajor axpy kernel.
    // Keep this in sync with coo_spmm_target_format and with left_spmm's native dispatch.
    NonzeroSort target = coo_spmm_target_format(layout_B, layout_C, d, m);
    std::vector<sint_t> ptr;  // backs a zero-copy view's ptr array; must outlive the kernel call
    if (target == NonzeroSort::CSR) {
        auto M = RandBLAS::sparse_data::coo_to_csr_view_or_copy(op, ptr);
        csr::apply_csr_jik_p11(alpha, layout_B, layout_C, d, n, m, M, B, ldb, C, ldc);
    } else {
        auto M = RandBLAS::sparse_data::coo_to_csc_view_or_copy(op, ptr);
        if (layout_B == layout_C && layout_B == blas::Layout::RowMajor) {
            csc::apply_csc_kib_1p1_rowmajor(alpha, n, M, B, ldb, C, ldc);
        } else {
            csc::apply_csc_jki_p11(alpha, layout_B, layout_C, n, M, B, ldb, C, ldc);
        }
    }
    return;
}

} // end namespace
