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
/// Built-in BLAS level-1 routines: scal, copy, axpy, dot, nrm2.

#include "RandBLAS/blas_backend/enums.hh"
#include <cmath>
#include <cstdint>

namespace blas {

// ---------------------------------------------------------------------------
// Level 1: scal, copy, axpy, dot, nrm2
// ---------------------------------------------------------------------------

template <typename T>
void scal(int64_t n, T alpha, T* x, int64_t incx) {
    if (incx == 1) {
        for (int64_t i = 0; i < n; ++i)
            x[i] *= alpha;
    } else {
        for (int64_t i = 0; i < n; ++i)
            x[i * incx] *= alpha;
    }
}

template <typename T>
void copy(int64_t n, T const* x, int64_t incx, T* y, int64_t incy) {
    if (incx == 1 && incy == 1) {
        for (int64_t i = 0; i < n; ++i)
            y[i] = x[i];
    } else {
        for (int64_t i = 0; i < n; ++i)
            y[i * incy] = x[i * incx];
    }
}

template <typename T>
void axpy(int64_t n, T alpha, T const* x, int64_t incx, T* y, int64_t incy) {
    if (incx == 1 && incy == 1) {
        for (int64_t i = 0; i < n; ++i)
            y[i] += alpha * x[i];
    } else {
        for (int64_t i = 0; i < n; ++i)
            y[i * incy] += alpha * x[i * incx];
    }
}

template <typename T>
T dot(int64_t n, T const* x, int64_t incx, T const* y, int64_t incy) {
    T acc = T(0);
    if (incx == 1 && incy == 1) {
        for (int64_t i = 0; i < n; ++i)
            acc += x[i] * y[i];
    } else {
        for (int64_t i = 0; i < n; ++i)
            acc += x[i * incx] * y[i * incy];
    }
    return acc;
}

template <typename T>
T nrm2(int64_t n, T const* x, int64_t incx) {
    T acc = T(0);
    if (incx == 1) {
        for (int64_t i = 0; i < n; ++i)
            acc += x[i] * x[i];
    } else {
        for (int64_t i = 0; i < n; ++i)
            acc += x[i * incx] * x[i * incx];
    }
    return std::sqrt(acc);
}

} // namespace blas
