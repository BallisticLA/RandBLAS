#pragma once

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
#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

namespace RandBLAS::sparse_data::csr {


template <typename T, SignedInteger sint_t = int64_t>
static void lower_trsv(
    // CSR-format data
    const T      *vals,  // vals
    const sint_t *ptrs,  // rowptr
    const sint_t *inds,  // colidxs
    // input-output vector data
    int64_t lenx,
    T *x,
    int64_t incx
) {
    int64_t j, p;
    if (incx == 1) {
        for (j = 0; j < lenx; ++j) {
            T &xj = x[j];
            for (p = ptrs[j]; p < ptrs[j+1] - 1; ++p)
                xj -= vals[p] * x[inds[p]];
            xj /= vals[ptrs[j+1]-1];
        }
    } else {
        for (j = 0; j < lenx; ++j) {
            T &xj = x[j*incx];
            for (p = ptrs[j]; p < ptrs[j+1] - 1; ++p)
                xj -= vals[p] * x[inds[p]*incx];
            xj /= vals[ptrs[j+1]-1];
        }
    }
}

template <typename T, SignedInteger sint_t = int64_t>
static void upper_trsv(
    // CSR-format data
    const T      *vals,  // vals
    const sint_t *ptrs,  // rowptr
    const sint_t *inds,  // colidxs
    // input-output vector data
    int64_t lenx,
    T *x,
    int64_t incx
) {
    int64_t j, p;
    if (incx == 1) {
        for (j = lenx - 1; j >= 0; --j) {
            T &xj = x[j];
            for (p = ptrs[j]+1; p < ptrs[j+1]; ++p)
                xj -= vals[p] * x[inds[p]];
            xj /= vals[ptrs[j]];
        }
    } else {
        for (j = lenx - 1; j >= 0; --j) {
            T &xj = x[j*incx];
            for (p = ptrs[j]+1; p < ptrs[j+1]; ++p)
                xj -= vals[p] * x[inds[p]*incx];
            xj /= vals[ptrs[j]];
        }
    }
}

template <typename T, SignedInteger sint_t = int64_t>
static void trsm_jki_p11(
    blas::Layout layout_B,
    blas::Uplo uplo,
    int64_t n,
    int64_t k,
    const CSRMatrix<T, sint_t> &A,
    T *B,
    int64_t ldb
){
    randblas_require(n == A.n_rows);
    randblas_require(n == A.n_cols);

    const sint_t* ptrs = A.rowptr;
    const sint_t* idxs = A.colidxs;
    const T*      vals = A.vals;

    int64_t j, p, ell;
    for (ell = 0; ell < n; ++ell) {
        if (uplo == blas::Uplo::Lower) {
            p = ptrs[ell+1] - 1;
        } else {
            p = ptrs[ell];
        }
        randblas_require(idxs[p] == ell);
        randblas_require(vals[p] != 0.0);
    }

    auto s = layout_to_strides(layout_B, ldb);
    auto B_inter_col_stride = s.inter_col_stride;
    auto B_inter_row_stride = s.inter_row_stride;

    if (uplo == blas::Uplo::Lower) {
        #pragma omp parallel default(shared) private(j, p, ell)
        {
            #pragma omp for schedule(static)
            for (ell = 0; ell < k; ell++) {
                T* col_B = &B[ell * B_inter_col_stride];
                for (j = 0; j < n; ++j) {
                    T &Bjl = col_B[j * B_inter_row_stride];
                    for (p = ptrs[j]; p < ptrs[j+1]-1; ++p)
                        Bjl -= vals[p] * col_B[idxs[p] * B_inter_row_stride];
                    Bjl /= vals[ptrs[j+1]-1];
                }
            }
        }
    } else {
        #pragma omp parallel default(shared) private(j, p, ell)
        {
            #pragma omp for schedule(static)
            for (ell = 0; ell < k; ell++) {
                T* col_B = &B[ell * B_inter_col_stride];
                for (j = n - 1; j >= 0; --j) {
                    T &Bjl = col_B[j * B_inter_row_stride];
                    for (p = ptrs[j] + 1; p < ptrs[j+1]; ++p)
                        Bjl -= vals[p] * col_B[idxs[p] * B_inter_row_stride];
                    Bjl /= vals[ptrs[j]];
                }
            }
        }
    }
}

}

