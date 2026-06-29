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
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include <vector>
#include <algorithm>
#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

namespace RandBLAS::sparse_data::csc {

#ifdef __cpp_concepts
using RandBLAS::SignedInteger;
#else
#define SignedInteger typename
#endif

template <typename T, SignedInteger sint_t = int64_t>
static void apply_csc_to_vector_ki(
    T alpha,
    // CSC-format data
    const T *vals,
    const sint_t *rowidxs,
    const sint_t *colptr,
    // input-output vector data
    int64_t len_v,
    const T *v,
    int64_t incv,   // stride between elements of v
    T *Av,          // Av += A * v.
    int64_t incAv   // stride between elements of Av
) {
    int64_t i = 0;
    for (int64_t c = 0; c < len_v; ++c) {
        T scale = alpha * v[c * incv];
        while (i < colptr[c+1]) {
            int64_t row = rowidxs[i];
            Av[row * incAv] += (vals[i] * scale);
            i += 1;
        }
    }
}

template <typename T, SignedInteger sint_t>
static void apply_csc_jki_p11(
    T alpha,
    blas::Layout layout_B,
    blas::Layout layout_C,
    int64_t n,
    const CSCMatrix<T, sint_t> &A,
    const T *B,
    int64_t ldb,
    T *C,
    int64_t ldc
) {
    randblas_require(A.index_base == IndexBase::Zero);

    auto m = A.n_cols;

    auto s = layout_to_strides(layout_B, ldb);
    auto B_inter_col_stride = s.inter_col_stride;
    auto B_inter_row_stride = s.inter_row_stride;

    s = layout_to_strides(layout_C, ldc);
    auto C_inter_col_stride = s.inter_col_stride;
    auto C_inter_row_stride = s.inter_row_stride;

    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < n; j++) {
        const T* B_col = &B[B_inter_col_stride * j];
              T* C_col = &C[C_inter_col_stride * j];
        apply_csc_to_vector_ki<T>(
            alpha,
            A.vals, A.rowidxs, A.colptr,
            m, B_col, B_inter_row_stride,
            C_col, C_inter_row_stride
        );
    }
    return;
}

template <typename T, SignedInteger sint_t>
static void apply_csc_kib_1p1_rowmajor(
    T alpha,
    int64_t n,
    const CSCMatrix<T, sint_t> &A,
    const T *B,
    int64_t ldb,
    T *C,
    int64_t ldc
) {
    randblas_require(A.index_base == IndexBase::Zero);

    auto d = A.n_rows;
    auto m = A.n_cols;

    #pragma omp parallel default(shared)
    {
        #if defined(RandBLAS_HAS_OpenMP)
            int t = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
        #else
            int t = 0;
            int num_threads = 1;
        #endif

        int i_lower = (d * t) / num_threads;
        int i_upper = (d * (t + 1)) / num_threads;
        for (int64_t k = 0; k < m; ++k) {
            // Rank-1 update: C[:,:] += A[:,k] @ B[k,:]
            const T* row_B = &B[k*ldb];
            for (int64_t ell = A.colptr[k]; ell < A.colptr[k+1]; ++ell) {
                int64_t i = A.rowidxs[ell];
                if (i_lower <= i && i < i_upper) {
                    T* row_C = &C[i*ldc];
                    T scale = alpha * A.vals[ell];
                    blas::axpy(n, scale, row_B, 1, row_C, 1);
                }
            }
        }
    }
    return;
}

} // end namespace RandBLAS::sparse_data::csc
