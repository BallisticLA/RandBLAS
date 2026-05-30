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
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include <vector>
#include <algorithm>
#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif


namespace RandBLAS::sparse_data::csr {

#ifdef __cpp_concepts
using RandBLAS::SignedInteger;
#else
#define SignedInteger typename
#endif

// Core CSR SpMV accumulation:  Av[i] += alpha * sum_ell vals[ell] * v[colidxs[ell]].
// Templated on UnitStride so the hot ColMajor case (incv == incAv == 1) drops the
// runtime index multiplies; `if constexpr` keeps a single source for both variants.
// The simd reduction permits FP reassociation, which breaks the serial dependence
// on the accumulator (no effect when compiled without OpenMP -- the pragma is
// ignored and the loop simply runs unvectorized, as with the rest of this file).
template <bool UnitStride, typename T, SignedInteger sint_t>
static void apply_csr_to_vector_ik_impl(
    T alpha,
    const T *vals, const sint_t *rowptr, const sint_t *colidxs,
    const T *v, int64_t incv,
    int64_t len_Av, T *Av, int64_t incAv
) {
    UNUSED(incv); UNUSED(incAv);
    // ^ silence compiler complaints if UnitStride == true.
    for (int64_t i = 0; i < len_Av; ++i) {
        T Av_i_diff = 0.0;
        #pragma omp simd reduction(+:Av_i_diff)
        for (int64_t ell = rowptr[i]; ell < rowptr[i+1]; ++ell) {
            int64_t j = colidxs[ell];
            if constexpr (UnitStride) {
                Av_i_diff += vals[ell] * v[j];
            } else {
                Av_i_diff += vals[ell] * v[j*incv];
            }
        }
        if constexpr (UnitStride) {
            Av[i] += alpha * Av_i_diff;
        } else {
            Av[i*incAv] += alpha * Av_i_diff;
        }
    }
}

template <typename T, SignedInteger sint_t = int64_t>
static void apply_csr_to_vector_ik(
    T alpha,
    // CSR-format data
    const T *vals,
    const sint_t *rowptr,
    const sint_t *colidxs,
    // input-output vector data
    const T *v,
    int64_t incv,   // stride between elements of v
    int64_t len_Av,
    T *Av,          // Av += alpha * A * v.
    int64_t incAv   // stride between elements of Av
) {
    // Unit stride (incv == incAv == 1) is the ColMajor case; specialize on it so
    // the inner reduction carries no index multiplies. Other strides fall through.
    if (incv == 1 && incAv == 1) {
        apply_csr_to_vector_ik_impl<true>(
            alpha, vals, rowptr, colidxs, v, incv, len_Av, Av, incAv);
    } else {
        apply_csr_to_vector_ik_impl<false>(
            alpha, vals, rowptr, colidxs, v, incv, len_Av, Av, incAv);
    }
}

template <typename T, SignedInteger sint_t>
static void apply_csr_jik_p11(
    T alpha,
    blas::Layout layout_B,
    blas::Layout layout_C,
    int64_t d,
    int64_t n,
    int64_t m,
    const CSRMatrix<T, sint_t> &A,
    const T *B,
    int64_t ldb,
    T *C,
    int64_t ldc
) {
    randblas_require(A.index_base == IndexBase::Zero);
    T *vals = A.vals;
    randblas_require(d == A.n_rows);
    randblas_require(m == A.n_cols);

    auto s = layout_to_strides(layout_B, ldb);
    auto B_inter_col_stride = s.inter_col_stride;
    auto B_inter_row_stride = s.inter_row_stride;

    s = layout_to_strides(layout_C, ldc);
    auto C_inter_col_stride = s.inter_col_stride;
    auto C_inter_row_stride = s.inter_row_stride;

    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < n; j++) {
        const T *B_col = &B[B_inter_col_stride * j];
              T *C_col = &C[C_inter_col_stride * j];
        apply_csr_to_vector_ik(alpha,
                vals, A.rowptr, A.colidxs,
                B_col, B_inter_row_stride,
            d, C_col, C_inter_row_stride
        );
    }
    return;
}

template <typename T, SignedInteger sint_t>
static void apply_csr_ikb_p1b_rowmajor(
    T alpha,
    int64_t d,
    int64_t n,
    int64_t m,
    const CSRMatrix<T, sint_t> &A,
    const T *B,
    int64_t ldb,
    T *C,
    int64_t ldc
) {
    randblas_require(A.index_base == IndexBase::Zero);

    randblas_require(d == A.n_rows);
    randblas_require(m == A.n_cols);

    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < d; ++i) {
        // C[i, 0:n] += alpha * A[i, :] @ B[:, 0:n]
        T* row_C = &C[i*ldc];
        for (int64_t ell = A.rowptr[i]; ell < A.rowptr[i+1]; ++ell) {
            // we're working with A[i,k] for k = A.colidxs[ell]
            // compute C[i, 0:n] += alpha * A[i,k] * B[k, 0:n]
            T scale = alpha * A.vals[ell];
            int64_t k = A.colidxs[ell];
            const T* row_B = &B[k*ldb];
            blas::axpy(n, scale, row_B, 1, row_C, 1);
        }
    }
    return;
}

} // end namespace RandBLAS::sparse_data::csr
