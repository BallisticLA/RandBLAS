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
#include <vector>
#include <algorithm>
#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

namespace RandBLAS::sparse_data::coo {

using RandBLAS::SignedInteger;

template <typename T, SignedInteger sint_t = int64_t>
static int64_t set_filtered_coo(
    // COO-format matrix data
    const T       *vals,
    const sint_t *rowidxs,
    const sint_t *colidxs,
    int64_t nnz,
    // submatrix bounds
    int64_t col_start,
    int64_t col_end,
    int64_t row_start,
    int64_t row_end,
    // COO-format submatrix data
    T       *new_vals,
    sint_t *new_rowidxs,
    sint_t *new_colidxs
) {
    int64_t new_nnz = 0;
    for (int64_t ell = 0; ell < nnz; ++ell) {
        if (
            row_start <= rowidxs[ell] && rowidxs[ell] < row_end &&
            col_start <= colidxs[ell] && colidxs[ell] < col_end
        ) {
            new_vals[new_nnz] = vals[ell];
            new_rowidxs[new_nnz] = rowidxs[ell] - row_start;
            new_colidxs[new_nnz] = colidxs[ell] - col_start;
            new_nnz += 1;
        }
    }
    return new_nnz;
}


template <typename T, SignedInteger sint_t>
static void apply_coo_left_jki_p11(
    T alpha,
    blas::Layout layout_B,
    blas::Layout layout_C,
    int64_t d,
    int64_t n,
    int64_t m,
    COOMatrix<T, sint_t> &A0,
    int64_t ro_a,
    int64_t co_a,
    const T *B,
    int64_t ldb,
    T *C,
    int64_t ldc
) {
    randblas_require(A0.index_base == IndexBase::Zero);

    // Step 1: reduce to the case of CSC sort order.
    if (A0.sort != NonzeroSort::CSC) {
        auto orig_sort = A0.sort;
        sort_coo_data(NonzeroSort::CSC, A0);
        apply_coo_left_jki_p11(alpha, layout_B, layout_C, d, n, m, A0, ro_a, co_a, B, ldb, C, ldc);
        sort_coo_data(orig_sort, A0);
        return;
    }

    // Step 2: make a CSC-sort-order COOMatrix that represents the desired submatrix of A.
    //      While we're at it, reduce to the case when alpha = 1.0 by scaling the values
    //      of the matrix we just created.
    int64_t A_nnz;
    int64_t A0_nnz = A0.nnz;
    std::vector<sint_t> A_rows(A0_nnz, 0);
    std::vector<sint_t> A_colptr(std::max(A0_nnz, m + 1), 0);
    std::vector<T> A_vals(A0_nnz, 0.0);
    A_nnz = set_filtered_coo(
        A0.vals, A0.rows, A0.cols, A0.nnz,
        co_a, co_a + m,
        ro_a, ro_a + d,
        A_vals.data(), A_rows.data(), A_colptr.data()
    );
    sorted_nonzero_locations_to_pointer_array(A_nnz, A_colptr.data(), m);

    CSCMatrix<T, sint_t> A_csc(d, m, A_nnz, A_vals.data(), A_rows.data(), A_colptr.data());

    RandBLAS::sparse_data::csc::apply_csc_left_jki_p11(
        alpha, layout_B, layout_C, n, A_csc, B, ldb, C, ldc
    );
    return;
}


} // end namespace
