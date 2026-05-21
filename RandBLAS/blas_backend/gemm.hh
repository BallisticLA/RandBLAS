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
/// Built-in BLAS gemm:  C = alpha * op(A) * op(B) + beta * C.
///
/// Normalizes RowMajor to ColMajor, then dispatches to one of four
/// sub-routines (nn/tn/nt/tt) based on the transposition flags.
/// Each sub-routine uses 2D loop tiling for cache efficiency.

#include "RandBLAS/blas_backend/enums.hh"
#include <algorithm>
#include <cstdint>

namespace blas {

namespace _builtin_detail {

// Cache-line tile size (in elements). Tuned for 64-byte cache lines.
static constexpr int64_t GEMM_BLOCK = 64;

// Scale C by beta, handling the beta==0 case safely (avoids NaN from 0*inf).
template <typename T>
static inline void scale_C(int64_t m, int64_t n, T beta, T* C, int64_t ldc) {
    if (beta == T(0)) {
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < m; ++i)
                C[i + j * ldc] = T(0);
    } else if (beta != T(1)) {
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < m; ++i)
                C[i + j * ldc] *= beta;
    }
}

// ColMajor, transA=NoTrans, transB=NoTrans
//   C[i,j] += alpha * A[i,l] * B[l,j]
//   A[i,l] = A[i + l*lda]  — col l of A (length m) is contiguous
//   B[l,j] = B[l + j*ldb]  — col j of B (length k) is contiguous
template <typename T>
void gemm_nn(
    int64_t m, int64_t n, int64_t k,
    T alpha, T const* A, int64_t lda,
             T const* B, int64_t ldb,
    T beta,  T*       C, int64_t ldc)
{
    scale_C(m, n, beta, C, ldc);
    for (int64_t j = 0; j < n; ++j) {
        T* Cj = C + j * ldc;
        for (int64_t l0 = 0; l0 < k; l0 += GEMM_BLOCK) {
            int64_t l1 = std::min(l0 + GEMM_BLOCK, k);
            for (int64_t i0 = 0; i0 < m; i0 += GEMM_BLOCK) {
                int64_t i1 = std::min(i0 + GEMM_BLOCK, m);
                for (int64_t l = l0; l < l1; ++l) {
                    T bval = alpha * B[l + j * ldb];
                    T const* Al = A + l * lda;
                    for (int64_t i = i0; i < i1; ++i)
                        Cj[i] += bval * Al[i];
                }
            }
        }
    }
}

// ColMajor, transA=Trans, transB=NoTrans
//   C[i,j] += alpha * A^T[i,l] * B[l,j]
//            = alpha * A[l,i]   * B[l,j]
//   A[l,i] = A[l + i*lda]  — col i of A (length k) is contiguous
//   B[l,j] = B[l + j*ldb]  — col j of B (length k) is contiguous
template <typename T>
void gemm_tn(
    int64_t m, int64_t n, int64_t k,
    T alpha, T const* A, int64_t lda,   // A is k×m col-major
             T const* B, int64_t ldb,   // B is k×n col-major
    T beta,  T*       C, int64_t ldc)
{
    scale_C(m, n, beta, C, ldc);
    for (int64_t j = 0; j < n; ++j) {
        T* Cj      = C + j * ldc;
        T const* Bj = B + j * ldb;   // col j of B, length k
        for (int64_t i0 = 0; i0 < m; i0 += GEMM_BLOCK) {
            int64_t i1 = std::min(i0 + GEMM_BLOCK, m);
            for (int64_t l0 = 0; l0 < k; l0 += GEMM_BLOCK) {
                int64_t l1 = std::min(l0 + GEMM_BLOCK, k);
                for (int64_t i = i0; i < i1; ++i) {
                    T const* Ai = A + i * lda;   // col i of A, length k
                    T acc = T(0);
                    for (int64_t l = l0; l < l1; ++l)
                        acc += Ai[l] * Bj[l];
                    Cj[i] += alpha * acc;
                }
            }
        }
    }
}

// ColMajor, transA=NoTrans, transB=Trans
//   C[i,j] += alpha * A[i,l] * B^T[l,j]
//            = alpha * A[i,l] * B[j,l]
//   A[i,l] = A[i + l*lda]  — col l of A (length m) is contiguous
//   B[j,l] = B[j + l*ldb]  — col l of B (length n) is contiguous
//   Accessing B column-by-column (varying l outer) is cache-friendly.
template <typename T>
void gemm_nt(
    int64_t m, int64_t n, int64_t k,
    T alpha, T const* A, int64_t lda,   // A is m×k col-major
             T const* B, int64_t ldb,   // B is n×k col-major (op(B)=B^T is k×n)
    T beta,  T*       C, int64_t ldc)
{
    scale_C(m, n, beta, C, ldc);
    for (int64_t l0 = 0; l0 < k; l0 += GEMM_BLOCK) {
        int64_t l1 = std::min(l0 + GEMM_BLOCK, k);
        for (int64_t j0 = 0; j0 < n; j0 += GEMM_BLOCK) {
            int64_t j1 = std::min(j0 + GEMM_BLOCK, n);
            for (int64_t i0 = 0; i0 < m; i0 += GEMM_BLOCK) {
                int64_t i1 = std::min(i0 + GEMM_BLOCK, m);
                for (int64_t l = l0; l < l1; ++l) {
                    T const* Al = A + l * lda;   // col l of A, length m
                    T const* Bl = B + l * ldb;   // col l of B, length n; Bl[j] = B[j,l]
                    for (int64_t j = j0; j < j1; ++j) {
                        T bval = alpha * Bl[j];
                        T* Cj  = C + j * ldc;
                        for (int64_t i = i0; i < i1; ++i)
                            Cj[i] += bval * Al[i];
                    }
                }
            }
        }
    }
}

// ColMajor, transA=Trans, transB=Trans
//   C[i,j] += alpha * A^T[i,l] * B^T[l,j]
//            = alpha * A[l,i]   * B[j,l]
//   A[l,i] = A[l + i*lda]  — col i of A (length k) is contiguous
//   B[j,l] = B[j + l*ldb]  — col l of B (length n) is contiguous
//   Loop order: l outer (contiguous B col), i middle (contiguous A col), j inner.
template <typename T>
void gemm_tt(
    int64_t m, int64_t n, int64_t k,
    T alpha, T const* A, int64_t lda,   // A is k×m col-major
             T const* B, int64_t ldb,   // B is n×k col-major
    T beta,  T*       C, int64_t ldc)
{
    scale_C(m, n, beta, C, ldc);
    for (int64_t l0 = 0; l0 < k; l0 += GEMM_BLOCK) {
        int64_t l1 = std::min(l0 + GEMM_BLOCK, k);
        for (int64_t i0 = 0; i0 < m; i0 += GEMM_BLOCK) {
            int64_t i1 = std::min(i0 + GEMM_BLOCK, m);
            for (int64_t j0 = 0; j0 < n; j0 += GEMM_BLOCK) {
                int64_t j1 = std::min(j0 + GEMM_BLOCK, n);
                for (int64_t l = l0; l < l1; ++l) {
                    T const* Bl = B + l * ldb;   // col l of B, length n; Bl[j] = B[j,l]
                    for (int64_t i = i0; i < i1; ++i) {
                        T aval = alpha * A[l + i * lda];   // A[l,i] = col i of A at index l
                        T* Ci  = C + i;                    // C[i,j] = Ci[j*ldc]
                        for (int64_t j = j0; j < j1; ++j)
                            Ci[j * ldc] += aval * Bl[j];
                    }
                }
            }
        }
    }
}

} // namespace _builtin_detail

template <typename T>
void gemm(
    Layout layout,
    Op transA, Op transB,
    int64_t m, int64_t n, int64_t k,
    T alpha, T const* A, int64_t lda,
             T const* B, int64_t ldb,
    T beta,  T*       C, int64_t ldc)
{
    if (layout == Layout::RowMajor) {
        gemm(blas::Layout::ColMajor, transB, transA, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc);
        return;
    }

    // ColMajor dispatch
    bool ta = (transA != Op::NoTrans);
    bool tb = (transB != Op::NoTrans);
    if      (!ta && !tb) _builtin_detail::gemm_nn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else if ( ta && !tb) _builtin_detail::gemm_tn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else if (!ta &&  tb) _builtin_detail::gemm_nt(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else                 _builtin_detail::gemm_tt(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
}

} // namespace blas
