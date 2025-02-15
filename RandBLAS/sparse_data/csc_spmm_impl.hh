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

using RandBLAS::SignedInteger;

template <typename T, SignedInteger sint_t = int64_t>
static void apply_csc_to_vector_from_left_ki(
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
        T scale = v[c * incv];
        while (i < colptr[c+1]) {
            int64_t row = rowidxs[i];
            Av[row * incAv] += (vals[i] * scale);
            i += 1;
        }
    }
}

template <typename T, SignedInteger sint_t = int64_t>
static void apply_regular_csc_to_vector_from_left_ki(
    // data for "regular CSC": CSC with fixed nnz per col,
    // which obviates the requirement for colptr.
    const T *vals,
    const sint_t *rowidxs,
    int64_t col_nnz,
    // input-output vector data
    int64_t len_v,
    const T *v,
    int64_t incv,   // stride between elements of v
    T *Av,          // Av += A * v.
    int64_t incAv   // stride between elements of Av
) {
    for (int64_t c = 0; c < len_v; ++c) {
        T scale = v[c * incv];
        for (int64_t j = c * col_nnz; j < (c + 1) * col_nnz; ++j) {
            int64_t row = rowidxs[j];
            Av[row * incAv] += (vals[j] * scale);
        }
    }
}

template <typename T, SignedInteger sint_t>
static void apply_csc_left_jki_p11(
    T alpha,
    blas::Layout layout_B,
    blas::Layout layout_C,
    int64_t d,
    int64_t n,
    int64_t m,
    const CSCMatrix<T, sint_t> &A,
    const T *B,
    int64_t ldb,
    T *C,
    int64_t ldc
) {
    randblas_require(A.index_base == IndexBase::Zero);
    T *vals = A.vals;
    if (alpha != (T) 1.0) {
        vals = new T[A.nnz]{};
        blas::axpy(A.nnz, alpha, A.vals, 1, vals, 1);
    }

    randblas_require(d == A.n_rows);
    randblas_require(m == A.n_cols);

    bool fixed_nnz_per_col = true;
    for (int64_t ell = 2; (ell < m + 1) && fixed_nnz_per_col; ++ell)
        fixed_nnz_per_col = (A.colptr[1] == A.colptr[ell]);

    auto s = layout_to_strides(layout_B, ldb);
    auto B_inter_col_stride = s.inter_col_stride;
    auto B_inter_row_stride = s.inter_row_stride;

    s = layout_to_strides(layout_C, ldc);
    auto C_inter_col_stride = s.inter_col_stride;
    auto C_inter_row_stride = s.inter_row_stride;

    #pragma omp parallel default(shared)
    {
        const T *B_col = nullptr;
        T *C_col = nullptr;
        #pragma omp for schedule(static)
        for (int64_t j = 0; j < n; j++) {
            B_col = &B[B_inter_col_stride * j];
            C_col = &C[C_inter_col_stride * j];
            if (fixed_nnz_per_col) {
                apply_regular_csc_to_vector_from_left_ki<T>(
                    vals, A.rowidxs, A.colptr[1],
                    m, B_col, B_inter_row_stride,
                    C_col, C_inter_row_stride
                );
            } else {
                apply_csc_to_vector_from_left_ki<T>(
                    vals, A.rowidxs, A.colptr,
                    m, B_col, B_inter_row_stride,
                    C_col, C_inter_row_stride
                ); 
            }
        }
    }
    if (alpha != (T) 1.0) {
        delete [] vals;
    }
    return;
}

template <typename T, SignedInteger sint_t>
static void apply_csc_left_kib_rowmajor_1p1(
    T alpha,
    int64_t d,
    int64_t n,
    int64_t m,
    const CSCMatrix<T, sint_t> &A,
    const T *B,
    int64_t ldb,
    T *C,
    int64_t ldc
) {
    randblas_require(A.index_base == IndexBase::Zero);

    randblas_require(d == A.n_rows);
    randblas_require(m == A.n_cols);


    int num_threads = 1;
    #if defined(RandBLAS_HAS_OpenMP)
    #pragma omp parallel 
    {
        num_threads = omp_get_num_threads();
    }
    #endif

    int* block_bounds = new int[num_threads + 1]{};
    int block_size = d / num_threads;
    if (block_size == 0) { block_size = 1;}
    for (int t = 0; t < num_threads; ++t)
        block_bounds[t+1] = block_bounds[t] + block_size;
    block_bounds[num_threads] += d % num_threads;

    #pragma omp parallel default(shared)
    {
        #if defined(RandBLAS_HAS_OpenMP)
        int t = omp_get_thread_num();
        #else
        int t = 0;
        #endif
        int i_lower = block_bounds[t];
        int i_upper = block_bounds[t+1];
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

    delete [] block_bounds;
    return;
}

} // end namespace RandBLAS::sparse_data::csc
