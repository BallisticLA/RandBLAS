// Copyright, 2026. See LICENSE for copyright holder information.
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

#include <blas.hh>
#include <cstdint>

namespace RandBLAS::sparse_data::internal {

// =============================================================================
/// Add the contribution of one stored A(i, j) = v entry (and the implied
/// symmetric A(j, i) = v when i != j) to a side=Left spsymm output:
///
///   Y[i, :] += av * B[j, :]
///   if (i != j) Y[j, :] += av * B[i, :]
///
/// where av = alpha * v has already been folded by the caller. The inner
/// row updates use blas::axpy with strides determined by `layout`.
///
/// Shared by all three spsymm fallback kernels (CSR / CSC / COO) -- they
/// differ only in their outer iteration order; the per-stored-entry scatter
/// is identical.
template <typename T>
inline void spsymm_scatter_left(blas::Layout layout, int64_t n, T av,
                                int64_t i, int64_t j,
                                const T* B, int64_t ldb,
                                T* Y, int64_t ldy) {
    if (layout == blas::Layout::ColMajor) {
        blas::axpy(n, av, &B[j], ldb, &Y[i], ldy);
        if (i != j)
            blas::axpy(n, av, &B[i], ldb, &Y[j], ldy);
    } else {
        blas::axpy(n, av, &B[j*ldb], 1, &Y[i*ldy], 1);
        if (i != j)
            blas::axpy(n, av, &B[i*ldb], 1, &Y[j*ldy], 1);
    }
}

// =============================================================================
/// Side=Right counterpart of spsymm_scatter_left. For stored A(i, j) = v:
///
///   Y[:, j] += av * B[:, i]
///   if (i != j) Y[:, i] += av * B[:, j]
template <typename T>
inline void spsymm_scatter_right(blas::Layout layout, int64_t m, T av,
                                 int64_t i, int64_t j,
                                 const T* B, int64_t ldb,
                                 T* Y, int64_t ldy) {
    if (layout == blas::Layout::ColMajor) {
        blas::axpy(m, av, &B[i*ldb], 1, &Y[j*ldy], 1);
        if (i != j)
            blas::axpy(m, av, &B[j*ldb], 1, &Y[i*ldy], 1);
    } else {
        blas::axpy(m, av, &B[i], ldb, &Y[j], ldy);
        if (i != j)
            blas::axpy(m, av, &B[j], ldb, &Y[i], ldy);
    }
}

} // namespace RandBLAS::sparse_data::internal
