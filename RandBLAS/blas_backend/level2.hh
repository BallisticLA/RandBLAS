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

#pragma once

/// @file
///
/// Built-in BLAS level-2 routines: gemv, ger.

#include "RandBLAS/blas_backend/enums.hh"
#include "RandBLAS/blas_backend/gemm.hh"
#include "RandBLAS/exceptions.hh"
#include <cstdint>

namespace blas {

// ---------------------------------------------------------------------------
// Level 2: gemv
//
// y = alpha * op(A) * x + beta * y,  A is m×n (before transposition)
// ---------------------------------------------------------------------------

template <typename T>
void gemv(
    Layout layout,
    Op trans,
    int64_t m, int64_t n,
    T alpha, T const* A, int64_t lda,
             T const* x, int64_t incx,
    T beta,  T*       y, int64_t incy)
{
    auto dim_out = (trans == Op::NoTrans) ? m : n;
    auto dim_in  = (trans == Op::NoTrans) ? n : m;

    if (layout == Layout::RowMajor) {
        gemm(Layout::ColMajor, Op::NoTrans, trans, 1, dim_out, dim_in, alpha, x, incx, A, lda, beta, y, incy);
        return;
    }
    // Scale y by beta
    if (beta == T(0)) {
        for (int64_t i = 0; i < dim_out; ++i) y[i * incy] = T(0);
    } else if (beta != T(1)) {
        for (int64_t i = 0; i < dim_out; ++i) y[i * incy] *= beta;
    }

    if (trans == Op::NoTrans) {
        // take a linear combination of columns of A
        for (int64_t j = 0; j < dim_in; ++j) {
            T xval = alpha * x[j * incx];
            T const* Aj = A + j * lda;
            for (int64_t i = 0; i < dim_out; ++i)
                y[i * incy] += xval * Aj[i];
        }
    } else {
        // dot products with rows of A^T = columns of A
        for (int64_t j = 0; j < dim_out; ++j) {
            T const* Aj = A + j * lda;
            T acc = T(0);
            for (int64_t i = 0; i < dim_in; ++i)
                acc += Aj[i] * x[i * incx];
            y[j * incy] += alpha * acc;
        }
    }
}

//
//  The following function is adapted from BLAS++. Copyright assertion below.
//
//      Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
//      SPDX-License-Identifier: BSD-3-Clause
//      This program is free software: you can redistribute it and/or modify it under
//      the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//

template <typename T>
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T *A, int64_t lda
) {
    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const T zero = 0;

    // check arguments
    randblas_require(m >= 0);
    randblas_require(n >= 0);
    randblas_require(incx != 0);
    randblas_require(incy != 0);

    if (layout == Layout::ColMajor)
        randblas_require(lda >= m);
    else
        randblas_require(lda >= n);

    // quick return
    if (m == 0 || n == 0 || alpha == zero)
        return;

    if (layout == Layout::ColMajor) {
        if (incx == 1 && incy == 1) {
            // unit stride
            for (int64_t j = 0; j < n; ++j) {
                // note: NOT skipping if y[j] is zero, for consistent NAN handling
                T tmp = alpha * y[j];
                for (int64_t i = 0; i < m; ++i) {
                    A(i, j) += x[i] * tmp;
                }
            }
        }
        else if (incx == 1) {
            // x unit stride, y non-unit stride
            int64_t jy = (incy > 0 ? 0 : (-n + 1)*incy);
            for (int64_t j = 0; j < n; ++j) {
                T tmp = alpha * y[jy];
                for (int64_t i = 0; i < m; ++i) {
                    A(i, j) += x[i] * tmp;
                }
                jy += incy;
            }
        }
        else {
            // x and y non-unit stride
            int64_t kx = (incx > 0 ? 0 : (-m + 1)*incx);
            int64_t jy = (incy > 0 ? 0 : (-n + 1)*incy);
            for (int64_t j = 0; j < n; ++j) {
                T tmp = alpha * y[jy];
                int64_t ix = kx;
                for (int64_t i = 0; i < m; ++i) {
                    A(i, j) += x[ix] * tmp;
                    ix += incx;
                }
                jy += incy;
            }
        }
    }
    else {
        // RowMajor
        if (incx == 1 && incy == 1) {
            // unit stride
            for (int64_t i = 0; i < m; ++i) {
                // note: NOT skipping if x[i] is zero, for consistent NAN handling
                T tmp = alpha * x[i];
                for (int64_t j = 0; j < n; ++j) {
                    A(j, i) += tmp * y[j];
                }
            }
        }
        else if (incy == 1) {
            // x non-unit stride, y unit stride
            int64_t ix = (incx > 0 ? 0 : (-m + 1)*incx);
            for (int64_t i = 0; i < m; ++i) {
                T tmp = alpha * x[ix];
                for (int64_t j = 0; j < n; ++j) {
                    A(j, i) += tmp * y[j];
                }
                ix += incx;
            }
        }
        else {
            // x and y non-unit stride
            int64_t ky = (incy > 0 ? 0 : (-n + 1)*incy);
            int64_t ix = (incx > 0 ? 0 : (-m + 1)*incx);
            for (int64_t i = 0; i < m; ++i) {
                T tmp = alpha * x[ix];
                int64_t jy = ky;
                for (int64_t j = 0; j < n; ++j) {
                    A(j, i) += tmp * y[jy];
                    jy += incy;
                }
                ix += incx;
            }
        }
    }

    #undef A
}

} // namespace blas
