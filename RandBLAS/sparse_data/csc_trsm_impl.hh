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

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#if defined(RandBLAS_HAS_OpenMP)
#include <omp.h>
#endif

namespace RandBLAS::sparse_data::csc {

template <typename T, SignedInteger sint_t = int64_t>
static void lower_trsv(
    const bool nonunit,
    // CSC-format data
    const T      *vals,  // vals
    const sint_t *inds,  // rowidxs
    const sint_t *ptrs,  // colptr
    // input-output vector data
    int64_t lenx,
    T *x,
    int64_t incx
) {
    int64_t j, p;
    if (incx == 1) {
        for (j = 0; j < lenx; ++j) {
            if (nonunit)
                x[j] /= vals[ptrs[j]];
            for (p = ptrs[j] + 1; p < ptrs[j+1]; ++p)
                x[inds[p]] -= vals[p] * x[j];
        }
    } else {
        for (j = 0; j < lenx; ++j) {
            if (nonunit)
                x[j*incx] /= vals[ptrs[j]];
            for (p = ptrs[j]+1; p < ptrs[j+1]; ++p)
                x[inds[p]*incx] -= vals[p] * x[j*incx];
        }
    }
}

template <typename T, SignedInteger sint_t = int64_t>
static void upper_trsv(
    const bool nonunit,
    // CSC-format data
    const T      *vals,  // vals
    const sint_t *inds,  // rowidxs
    const sint_t *ptrs,  // colptr
    // input-output vector data
    int64_t lenx,
    T *x,
    int64_t incx
) {
    int64_t j, p;
    if (incx == 1) {
        for (j = lenx - 1; j >= 0; --j) {
            if (nonunit)
                x[j] /= vals[ptrs[j+1]-1];
            for (p = ptrs[j]; p < ptrs[j+1] - 1; ++p)
                x[inds[p]] -= vals[p] * x[j];
        }
    } else {
        for (j = lenx - 1; j >= 0; --j) {
            if (nonunit)
                x[j*incx] /= vals[ptrs[j+1] - 1];
            for (p = ptrs[j]; p < ptrs[j+1] - 1; ++p)
                x[inds[p]*incx] -= vals[p] * x[j*incx];
        } 
    }
}

template<typename T, SignedInteger sint_t = int64_t>
static void trsm_jki_p11(
    blas::Layout layout_B,
    blas::Uplo uplo,
    blas::Diag diag,
    int64_t n,
    const CSCMatrix<T, sint_t> &A,
    T *B,
    int64_t ldb
){
    int64_t m = A.n_rows;
    const sint_t* ptrs = A.colptr;
    const sint_t* idxs = A.rowidxs;
    const T*      vals = A.vals;

    auto s = layout_to_strides(layout_B, ldb);
    auto B_inter_col_stride = s.inter_col_stride;
    auto B_inter_row_stride = s.inter_row_stride;

    const bool nonunit = diag == blas::Diag::NonUnit;

    int64_t j, p, ell;
    if (uplo == blas::Uplo::Lower) {
        #pragma omp parallel default(shared) private(j, p, ell)
        {
            #pragma omp for schedule(static)
            for (ell = 0; ell < n; ell++) {
                T* col_B = &B[ell * B_inter_col_stride];
                for (j = 0; j < m; ++j) {
                    T &Bjl = col_B[j * B_inter_row_stride];
                    if (nonunit)
                        Bjl /= vals[ptrs[j]];
                    for (p = ptrs[j]+1; p < ptrs[j+1]; ++p)
                        col_B[idxs[p] * B_inter_row_stride] -= vals[p] * Bjl;
                }
            }
        }
    } else {
        #pragma omp parallel default(shared) private(j, p, ell)
        {
            #pragma omp for schedule(static)
            for (ell = 0; ell < n; ell++) {
                T* col_B = &B[ell * B_inter_col_stride];
                for (j = m-1; j >= 0; --j) {
                    T &Bjl = col_B[j * B_inter_row_stride];
                    if (nonunit)
                        Bjl /= vals[ptrs[j+1] - 1];
                    for (p = ptrs[j]; p < ptrs[j+1] - 1; ++p)
                        col_B[idxs[p] * B_inter_row_stride] -= vals[p] * Bjl;
                }
            }
        }
    }
}
    
}
