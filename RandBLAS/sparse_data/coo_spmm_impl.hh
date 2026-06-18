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


template <typename T, SignedInteger sint_t>
static void apply_coo_via_csc(
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

    // Format choice (PROTOTYPE). The per-RHS-column kernels loop the operator's OUTER
    // dimension once per output column: CSC jki's outer loop is n_cols (= m), CSR jik's
    // is n_rows (= d). In the ColMajor path, routing a WIDE operator (d < m) through the
    // CSR jik kernel (shorter outer loop) measured 1.1-2x faster than CSC jki for both
    // SASO and LASO, with no regression -- so we prefer CSR there. The RowMajor path
    // keeps the bandwidth-saturating CSC kib axpy kernel. The COO must be sorted either
    // way, so the format choice is otherwise free (though a SASO that sampling sorted to
    // CSC will be re-sorted to CSR here; the kernel win exceeds that cost). See the
    // --csr-probe mode of examples/.../sketch_general_performance.cc for measurements.
    bool col_major  = (layout_B == blas::Layout::ColMajor && layout_C == blas::Layout::ColMajor);
    bool prefer_csr = col_major && (d < m);
    NonzeroSort target = prefer_csr ? NonzeroSort::CSR : NonzeroSort::CSC;

    bool submatrix = (A0.n_rows != d) || (A0.n_cols != m);
    if (submatrix || A0.sort != target) {
        auto A1 = A0.deepcopy();
        auto new_nnz = A1.nnz;
        if (submatrix) {
            int64_t write = 0;
            for (int64_t i = 0; i < A1.nnz; ++i) {
                auto r = A1.rows[i] - ro_a;
                auto c = A1.cols[i] - co_a;
                if (0 <= r && r < d && 0 <= c && c < m) {
                    A1.rows [write] = r;
                    A1.cols [write] = c;
                    A1.vals [write] = A1.vals[i];
                    write += 1;
                }
            }
            new_nnz = write;
        }
        COOMatrix<T,sint_t> A2(d, m,   new_nnz, A1.vals, A1.rows, A1.cols, false);
        A2.sort_arrays(target);
        apply_coo_via_csc(alpha, layout_B, layout_C, d, n, m, A2, 0, 0, B, ldb, C, ldc);
        return;
    }

    if (prefer_csr) {
        // ColMajor + wide: CSR jik (outer loop over the d rows).
        auto rowptr = new sint_t[d+1];
        sorted_idxs_to_compressed_ptr( A0.nnz, A0.rows, d, rowptr );
        CSRMatrix<T, sint_t> A_csr( d, m, A0.nnz, A0.vals, rowptr, A0.cols );
        using RandBLAS::sparse_data::csr::apply_csr_jik_p11;
        apply_csr_jik_p11(alpha, layout_B, layout_C, d, n, m, A_csr, B, ldb, C, ldc);
        delete [] rowptr;
        return;
    }

    auto colptr = new sint_t[m+1];
    sorted_idxs_to_compressed_ptr(  A0.nnz, A0.cols, m,       colptr );
    CSCMatrix<T, sint_t> A_csc( d, m, A0.nnz, A0.vals, A0.rows, colptr );
    if (layout_B == layout_C && layout_B == blas::Layout::RowMajor) {
        using RandBLAS::sparse_data::csc::apply_csc_kib_1p1_rowmajor;
        apply_csc_kib_1p1_rowmajor(alpha, n, A_csc, B, ldb, C, ldc);
    } else {
        using RandBLAS::sparse_data::csc::apply_csc_jki_p11;
        apply_csc_jki_p11(alpha, layout_B, layout_C, n, A_csc, B, ldb, C, ldc);
    }
    delete [] colptr;
    return;
}

} // end namespace
