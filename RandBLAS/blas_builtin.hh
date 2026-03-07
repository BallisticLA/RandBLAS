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
/// Minimal blas:: namespace for use when BLAS++ is not available.
/// Provides the enumerations and routines used by RandBLAS:
///   Enumerations: Layout, Op, Uplo, Diag
///   Level 1:      scal, copy, axpy
///   Level 3:      gemm
///
/// Enum values match BLAS++ so that code compiled with either backend is
/// binary-compatible if the enums are ever compared by underlying value.
///
/// The gemm implementation uses 2D loop tiling for reasonable cache efficiency.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace blas {

// ---------------------------------------------------------------------------
// Enumerations (values match BLAS++ / standard CBLAS character codes)
// ---------------------------------------------------------------------------

enum class Layout : char { ColMajor = 'C', RowMajor = 'R' };
enum class Op     : char { NoTrans = 'N', Trans = 'T', ConjTrans = 'C' };
enum class Uplo   : char { Upper = 'U', Lower = 'L' };
enum class Diag   : char { NonUnit = 'N', Unit = 'U' };
enum class Side   : char { Left = 'L', Right = 'R' };

// ---------------------------------------------------------------------------
// Level 1: scal, copy, axpy
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

// ---------------------------------------------------------------------------
// Level 3: gemm
//
// C = alpha * op(A) * op(B) + beta * C
//
// The implementation first normalizes RowMajor to ColMajor (standard trick),
// then dispatches to one of four sub-routines based on (transA, transB).
// Each sub-routine uses 2D loop tiling for cache efficiency.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Level 3: syrk
//
// C = alpha * op(A) * op(A)^T + beta * C,  C is n×n symmetric
//
// Signature: syrk(layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
//   trans=NoTrans: op(A) is n×k (A is n×k col-major)
//   trans=Trans:   op(A) is n×k, so A is k×n col-major
// Only the triangle specified by uplo is written.
// ---------------------------------------------------------------------------

template <typename T>
void syrk(
    Layout layout,
    Uplo uplo,
    Op trans,
    int64_t n, int64_t k,
    T alpha, T const* A, int64_t lda,
    T beta,  T*       C, int64_t ldc)
{
    // Normalize RowMajor → ColMajor: flip uplo and trans.
    // Row-major syrk(uplo, trans) = col-major syrk(flip(uplo), flip(trans)).
    if (layout == Layout::RowMajor) {
        uplo  = (uplo  == Uplo::Upper) ? Uplo::Lower  : Uplo::Upper;
        trans = (trans == Op::NoTrans) ? Op::Trans     : Op::NoTrans;
    }

    // Scale the relevant triangle of C by beta.
    for (int64_t j = 0; j < n; ++j) {
        int64_t i_start = (uplo == Uplo::Upper) ? 0 : j;
        int64_t i_end   = (uplo == Uplo::Upper) ? j + 1 : n;
        if (beta == T(0)) {
            for (int64_t i = i_start; i < i_end; ++i) C[i + j * ldc] = T(0);
        } else if (beta != T(1)) {
            for (int64_t i = i_start; i < i_end; ++i) C[i + j * ldc] *= beta;
        }
    }

    if (trans == Op::NoTrans) {
        // C += alpha * A * A^T,  A is n×k col-major: A[i,l] = A[i + l*lda]
        for (int64_t l = 0; l < k; ++l) {
            T const* Al = A + l * lda;   // col l of A, length n
            for (int64_t j = 0; j < n; ++j) {
                T val = alpha * Al[j];
                int64_t i_end = (uplo == Uplo::Upper) ? j + 1 : n;
                int64_t i_start = (uplo == Uplo::Upper) ? 0 : j;
                for (int64_t i = i_start; i < i_end; ++i)
                    C[i + j * ldc] += val * Al[i];
            }
        }
    } else {
        // C += alpha * A^T * A,  A is k×n col-major: A[l,i] = A[l + i*lda]
        // col i of A has length k and is contiguous.
        for (int64_t j = 0; j < n; ++j) {
            T const* Aj = A + j * lda;   // col j of A, length k
            int64_t i_start = (uplo == Uplo::Upper) ? 0 : j;
            int64_t i_end   = (uplo == Uplo::Upper) ? j + 1 : n;
            for (int64_t i = i_start; i < i_end; ++i) {
                T const* Ai = A + i * lda;   // col i of A, length k
                T acc = T(0);
                for (int64_t l = 0; l < k; ++l)
                    acc += Ai[l] * Aj[l];
                C[i + j * ldc] += alpha * acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Level 3: symm
//
// C = alpha * A * B + beta * C  (side=Left)
// C = alpha * B * A + beta * C  (side=Right)
// A is symmetric (m×m for Left, n×n for Right); only the uplo triangle is stored.
//
// Signature: symm(layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
// ---------------------------------------------------------------------------

template <typename T>
void symm(
    Layout layout,
    Side side,
    Uplo uplo,
    int64_t m, int64_t n,
    T alpha, T const* A, int64_t lda,
             T const* B, int64_t ldb,
    T beta,  T*       C, int64_t ldc)
{
    // Normalize RowMajor → ColMajor: swap side, swap m/n, flip uplo.
    if (layout == Layout::RowMajor) {
        std::swap(m, n);
        side = (side == Side::Left) ? Side::Right : Side::Left;
        uplo = (uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;
        // We'll call ourselves recursively with ColMajor.
        symm(Layout::ColMajor, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    // Scale C
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            C[i + j * ldc] = (beta == T(0)) ? T(0) : beta * C[i + j * ldc];

    // Helper: access the symmetric matrix A at (r, c), using only the stored triangle.
    // For Uplo::Upper: A[r,c] = A[r + c*lda] if r<=c, else A[c + r*lda].
    // For Uplo::Lower: A[r,c] = A[r + c*lda] if r>=c, else A[c + r*lda].
    auto A_sym = [&](int64_t r, int64_t c) -> T {
        bool stored = (uplo == Uplo::Upper) ? (r <= c) : (r >= c);
        return stored ? A[r + c * lda] : A[c + r * lda];
    };

    if (side == Side::Left) {
        // C += alpha * A * B,  A is m×m symmetric, B is m×n
        for (int64_t j = 0; j < n; ++j) {
            T const* Bj = B + j * ldb;
            T*       Cj = C + j * ldc;
            for (int64_t i = 0; i < m; ++i) {
                T acc = T(0);
                for (int64_t l = 0; l < m; ++l)
                    acc += A_sym(i, l) * Bj[l];
                Cj[i] += alpha * acc;
            }
        }
    } else {
        // C += alpha * B * A,  A is n×n symmetric, B is m×n
        for (int64_t j = 0; j < n; ++j) {
            T*       Cj = C + j * ldc;
            for (int64_t i = 0; i < m; ++i) {
                T acc = T(0);
                for (int64_t l = 0; l < n; ++l)
                    acc += B[i + l * ldb] * A_sym(l, j);
                Cj[i] += alpha * acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Level 3: trmm
//
// B ← alpha * op(A) * B  (side=Left)   or   B ← alpha * B * op(A)  (side=Right)
// A is triangular (uplo), optionally unit diagonal (diag).
//
// Signature: trmm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
//   B is m×n on input and output (col-major after normalization).
// ---------------------------------------------------------------------------

template <typename T>
void trmm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t m, int64_t n,
    T alpha, T const* A, int64_t lda,
             T*       B, int64_t ldb)
{
    // Normalize RowMajor → ColMajor:
    //   trmm_rowmaj(side, uplo, trans, m, n, A, B)
    //   = trmm_colmaj(flip(side), flip(uplo), flip(trans), n, m, A, B^T)
    // B^T is n×m col-major, but B is m×n row-major, same memory — just swap ldb interpretation.
    if (layout == Layout::RowMajor) {
        side  = (side  == Side::Left)   ? Side::Right  : Side::Left;
        uplo  = (uplo  == Uplo::Upper)  ? Uplo::Lower  : Uplo::Upper;
        trans = (trans == Op::NoTrans)  ? Op::Trans    : Op::NoTrans;
        std::swap(m, n);
        // Now B is treated as n×m col-major (same memory, ldb unchanged).
    }

    bool unit  = (diag  == Diag::Unit);
    bool upper = (uplo  == Uplo::Upper);
    bool left  = (side  == Side::Left);
    bool notrans = (trans == Op::NoTrans);

    // Process column-by-column for the output B (m×n col-major).
    if (left) {
        int64_t sz = m; // A is m×m
        if (notrans) {
            // B[:,j] ← alpha * op(A) * B[:,j], op(A)=A upper/lower triangular
            for (int64_t j = 0; j < n; ++j) {
                T* Bj = B + j * ldb;
                if (upper) {
                    // i from 0 to m-1; A[i,k] only for k >= i
                    for (int64_t i = 0; i < sz; ++i) {
                        T val = unit ? Bj[i] : A[i + i * lda] * Bj[i];
                        for (int64_t k = i + 1; k < sz; ++k)
                            val += A[i + k * lda] * Bj[k];
                        Bj[i] = alpha * val;
                    }
                } else {
                    // Lower triangular: i from m-1 to 0; A[i,k] only for k <= i
                    for (int64_t i = sz - 1; i >= 0; --i) {
                        T val = unit ? Bj[i] : A[i + i * lda] * Bj[i];
                        for (int64_t k = 0; k < i; ++k)
                            val += A[i + k * lda] * Bj[k];
                        Bj[i] = alpha * val;
                    }
                }
            }
        } else {
            // trans: B[:,j] ← alpha * A^T * B[:,j]
            for (int64_t j = 0; j < n; ++j) {
                T* Bj = B + j * ldb;
                if (upper) {
                    // A^T is lower triangular; i from m-1 to 0
                    for (int64_t i = sz - 1; i >= 0; --i) {
                        T val = unit ? Bj[i] : A[i + i * lda] * Bj[i];
                        // A^T[i,k] = A[k,i] = A[k + i*lda] for k < i
                        for (int64_t k = 0; k < i; ++k)
                            val += A[k + i * lda] * Bj[k];
                        Bj[i] = alpha * val;
                    }
                } else {
                    // Lower A, A^T is upper triangular; i from 0 to m-1
                    for (int64_t i = 0; i < sz; ++i) {
                        T val = unit ? Bj[i] : A[i + i * lda] * Bj[i];
                        for (int64_t k = i + 1; k < sz; ++k)
                            val += A[k + i * lda] * Bj[k];
                        Bj[i] = alpha * val;
                    }
                }
            }
        }
    } else {
        // Right: B[i,:] ← alpha * B[i,:] * op(A), A is n×n
        // Process row-by-row.
        int64_t sz = n;
        if (notrans) {
            // B[i,j] ← alpha * sum_k B[i,k] * A[k,j]
            for (int64_t i = 0; i < m; ++i) {
                if (upper) {
                    // j from n-1 to 0; A[k,j] only for k <= j
                    for (int64_t j = sz - 1; j >= 0; --j) {
                        T val = unit ? B[i + j * ldb] : B[i + j * ldb] * A[j + j * lda];
                        for (int64_t k = 0; k < j; ++k)
                            val += B[i + k * ldb] * A[k + j * lda];
                        B[i + j * ldb] = alpha * val;
                    }
                } else {
                    // Lower A: j from 0 to n-1; A[k,j] only for k >= j
                    for (int64_t j = 0; j < sz; ++j) {
                        T val = unit ? B[i + j * ldb] : B[i + j * ldb] * A[j + j * lda];
                        for (int64_t k = j + 1; k < sz; ++k)
                            val += B[i + k * ldb] * A[k + j * lda];
                        B[i + j * ldb] = alpha * val;
                    }
                }
            }
        } else {
            // B[i,j] ← alpha * sum_k B[i,k] * A^T[k,j] = alpha * sum_k B[i,k] * A[j,k]
            for (int64_t i = 0; i < m; ++i) {
                if (upper) {
                    // A^T is lower; j from 0 to n-1
                    for (int64_t j = 0; j < sz; ++j) {
                        T val = unit ? B[i + j * ldb] : B[i + j * ldb] * A[j + j * lda];
                        for (int64_t k = j + 1; k < sz; ++k)
                            val += B[i + k * ldb] * A[j + k * lda];
                        B[i + j * ldb] = alpha * val;
                    }
                } else {
                    // A^T is upper; j from n-1 to 0
                    for (int64_t j = sz - 1; j >= 0; --j) {
                        T val = unit ? B[i + j * ldb] : B[i + j * ldb] * A[j + j * lda];
                        for (int64_t k = 0; k < j; ++k)
                            val += B[i + k * ldb] * A[j + k * lda];
                        B[i + j * ldb] = alpha * val;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Level 3: trsm
//
// op(A) * X = alpha * B  (side=Left)   or   X * op(A) = alpha * B  (side=Right)
// Solves for X, overwrites B in place.
// A is triangular (uplo), optionally unit diagonal (diag).
//
// Signature: trsm(layout, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
//   B is m×n on input (col-major after normalization).
// ---------------------------------------------------------------------------

template <typename T>
void trsm(
    Layout layout,
    Side side,
    Uplo uplo,
    Op trans,
    Diag diag,
    int64_t m, int64_t n,
    T alpha, T const* A, int64_t lda,
             T*       B, int64_t ldb)
{
    // Normalize RowMajor → ColMajor: same transformation as trmm.
    if (layout == Layout::RowMajor) {
        side  = (side  == Side::Left)   ? Side::Right  : Side::Left;
        uplo  = (uplo  == Uplo::Upper)  ? Uplo::Lower  : Uplo::Upper;
        trans = (trans == Op::NoTrans)  ? Op::Trans    : Op::NoTrans;
        std::swap(m, n);
    }

    // Scale B by alpha first (avoids carrying alpha through the solves).
    if (alpha != T(1)) {
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < m; ++i)
                B[i + j * ldb] *= alpha;
    }

    bool unit    = (diag  == Diag::Unit);
    bool upper   = (uplo  == Uplo::Upper);
    bool left    = (side  == Side::Left);
    bool notrans = (trans == Op::NoTrans);

    if (left) {
        // Solve op(A) * X = B column by column.
        int64_t sz = m;
        for (int64_t j = 0; j < n; ++j) {
            T* Bj = B + j * ldb;
            if (notrans) {
                if (upper) {
                    // Backward substitution: i from m-1 to 0
                    for (int64_t i = sz - 1; i >= 0; --i) {
                        for (int64_t k = i + 1; k < sz; ++k)
                            Bj[i] -= A[i + k * lda] * Bj[k];
                        if (!unit) Bj[i] /= A[i + i * lda];
                    }
                } else {
                    // Forward substitution: i from 0 to m-1
                    for (int64_t i = 0; i < sz; ++i) {
                        for (int64_t k = 0; k < i; ++k)
                            Bj[i] -= A[i + k * lda] * Bj[k];
                        if (!unit) Bj[i] /= A[i + i * lda];
                    }
                }
            } else {
                // trans: solve A^T * X = B, i.e. substitute using A^T
                if (upper) {
                    // A^T is lower: forward substitution on A^T
                    for (int64_t i = 0; i < sz; ++i) {
                        // A^T[i,k] = A[k,i] = A[k + i*lda] for k < i
                        for (int64_t k = 0; k < i; ++k)
                            Bj[i] -= A[k + i * lda] * Bj[k];
                        if (!unit) Bj[i] /= A[i + i * lda];
                    }
                } else {
                    // A^T is upper: backward substitution on A^T
                    for (int64_t i = sz - 1; i >= 0; --i) {
                        for (int64_t k = i + 1; k < sz; ++k)
                            Bj[i] -= A[k + i * lda] * Bj[k];
                        if (!unit) Bj[i] /= A[i + i * lda];
                    }
                }
            }
        }
    } else {
        // Solve X * op(A) = B row by row.
        int64_t sz = n;
        for (int64_t i = 0; i < m; ++i) {
            if (notrans) {
                if (upper) {
                    // Forward substitution: j from 0 to n-1
                    // B[i,j] -= sum_{k<j} B[i,k] * A[k,j]
                    for (int64_t j = 0; j < sz; ++j) {
                        for (int64_t k = 0; k < j; ++k)
                            B[i + j * ldb] -= B[i + k * ldb] * A[k + j * lda];
                        if (!unit) B[i + j * ldb] /= A[j + j * lda];
                    }
                } else {
                    // Backward substitution: j from n-1 to 0
                    for (int64_t j = sz - 1; j >= 0; --j) {
                        for (int64_t k = j + 1; k < sz; ++k)
                            B[i + j * ldb] -= B[i + k * ldb] * A[k + j * lda];
                        if (!unit) B[i + j * ldb] /= A[j + j * lda];
                    }
                }
            } else {
                // X * A^T = B: A^T[j,k] = A[k,j]
                if (upper) {
                    // A^T is lower: backward subst over j
                    for (int64_t j = sz - 1; j >= 0; --j) {
                        for (int64_t k = j + 1; k < sz; ++k)
                            B[i + j * ldb] -= B[i + k * ldb] * A[j + k * lda];
                        if (!unit) B[i + j * ldb] /= A[j + j * lda];
                    }
                } else {
                    // A^T is upper: forward subst over j
                    for (int64_t j = 0; j < sz; ++j) {
                        for (int64_t k = 0; k < j; ++k)
                            B[i + j * ldb] -= B[i + k * ldb] * A[j + k * lda];
                        if (!unit) B[i + j * ldb] /= A[j + j * lda];
                    }
                }
            }
        }
    }
}

} // namespace blas
