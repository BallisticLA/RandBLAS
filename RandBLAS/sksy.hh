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

#include "RandBLAS/util.hh"
#include "RandBLAS/base.hh"
#include "RandBLAS/skge.hh"


// =============================================================================
// Symmetric sketching helpers (SYMM-backed). See project-plans/randblas-symm-plan.md
// for the four-case API design and the implementation status of each case.
//   - lsksy3, rsksy3: Case A (dense-symm A x dense Omega), via blas::symm.
//   - SparseSkOp branches of sketch_symmetric: Case B stubs (not implemented).
// =============================================================================

namespace RandBLAS::dense {

// =============================================================================
/// LSKSY3: SYMM-backed left-sketch with a symmetric matrix A.
///
/// Computes B = alpha * submat(S) * mat(A) + beta * B, where:
///   - mat(A) is n-by-n symmetric. Only the triangle named by `uplo` is read.
///   - submat(S) is the d-by-n view of S at (ro_s, co_s).
///   - mat(B) is d-by-n.
///
/// When S has no materialized buffer, the submatrix is realized via
/// `submatrix_as_blackbox` (same pattern as `lskge3`). When the buffered S's
/// storage layout matches the caller's `layout`, the final call is
/// `blas::symm` with `side = Right` (since A is on the right of S in the
/// operation). When layouts mismatch, SYMM cannot transpose S on-the-fly;
/// this first cut falls back to `blas::gemm` with `opS = Trans`, sacrificing
/// the SYMM speedup on that path. Optimization target: transpose-copy of S
/// into matching layout, then SYMM. See project-plans/randblas-symm-plan.md §7.
template <typename T, typename DenseSkOp>
void lsksy3(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t d, // B is d-by-n
    int64_t n, // A is n-by-n, S is d-by-n (after submat view)
    T alpha,
    const DenseSkOp &S,
    int64_t ro_s,
    int64_t co_s,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
){
    constexpr bool maybe_denseskop = !std::is_same_v<std::remove_cv_t<DenseSkOp>, BLASFriendlyOperator<T>>;
    if constexpr (maybe_denseskop) {
        if (!S.buff) {
            auto submat_S = submatrix_as_blackbox<BLASFriendlyOperator<T>>(S, d, n, ro_s, co_s);
            lsksy3(layout, uplo, d, n, alpha, submat_S, 0, 0, A, lda, beta, B, ldb);
            return;
        }
    }
    randblas_require( S.buff != nullptr );
    randblas_require( S.n_rows >= d + ro_s );
    randblas_require( S.n_cols >= n + co_s );
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= n);
        randblas_require(ldb >= d);
    } else {
        randblas_require(lda >= n);
        randblas_require(ldb >= n);
    }

    auto [pos, lds] = offset_and_ldim(S.layout, S.n_rows, S.n_cols, ro_s, co_s);
    T* S_ptr = &S.buff[pos];

    if (S.layout == layout) {
        // Fast path: SYMM directly.
        blas::symm(layout, blas::Side::Right, uplo, d, n,
                   alpha, A, lda, S_ptr, lds, beta, B, ldb);
    } else {
        // Layout mismatch: SYMM has no opS flag. Fall back to GEMM with
        // opS = Trans. Loses the SYMM 1.3-1.8x speedup on this path but is
        // correct. Optimization target tracked in randblas-symm-plan.md §7.
        blas::gemm(layout, blas::Op::Trans, blas::Op::NoTrans, d, n, n,
                   alpha, S_ptr, lds, A, lda, beta, B, ldb);
    }
    return;
}


// =============================================================================
/// RSKSY3: SYMM-backed right-sketch with a symmetric matrix A.
///
/// Computes B = alpha * mat(A) * submat(S) + beta * B, where:
///   - mat(A) is n-by-n symmetric. Only the triangle named by `uplo` is read.
///   - submat(S) is the n-by-d view of S at (ro_s, co_s).
///   - mat(B) is n-by-d.
///
/// Same materialization and layout-mismatch fallback semantics as `lsksy3`.
/// Final call (matching layout): `blas::symm` with `side = Left`.
template <typename T, typename DenseSkOp>
void rsksy3(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n, // A is n-by-n, S is n-by-d (after submat view), B is n-by-d
    int64_t d,
    T alpha,
    const T *A,
    int64_t lda,
    const DenseSkOp &S,
    int64_t ro_s,
    int64_t co_s,
    T beta,
    T *B,
    int64_t ldb
){
    constexpr bool maybe_denseskop = !std::is_same_v<std::remove_cv_t<DenseSkOp>, BLASFriendlyOperator<T>>;
    if constexpr (maybe_denseskop) {
        if (!S.buff) {
            auto submat_S = submatrix_as_blackbox<BLASFriendlyOperator<T>>(S, n, d, ro_s, co_s);
            rsksy3(layout, uplo, n, d, alpha, A, lda, submat_S, 0, 0, beta, B, ldb);
            return;
        }
    }
    randblas_require( S.buff != nullptr );
    randblas_require( S.n_rows >= n + ro_s );
    randblas_require( S.n_cols >= d + co_s );
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= n);
        randblas_require(ldb >= n);
    } else {
        randblas_require(lda >= n);
        randblas_require(ldb >= d);
    }

    auto [pos, lds] = offset_and_ldim(S.layout, S.n_rows, S.n_cols, ro_s, co_s);
    T* S_ptr = &S.buff[pos];

    if (S.layout == layout) {
        blas::symm(layout, blas::Side::Left, uplo, n, d,
                   alpha, A, lda, S_ptr, lds, beta, B, ldb);
    } else {
        // Layout-mismatch fallback (see lsksy3).
        blas::gemm(layout, blas::Op::NoTrans, blas::Op::Trans, n, d, n,
                   alpha, A, lda, S_ptr, lds, beta, B, ldb);
    }
    return;
}

} // end namespace RandBLAS::dense


namespace RandBLAS {

using namespace RandBLAS::dense;
using namespace RandBLAS::sparse;


namespace detail {

// =============================================================================
/// Shared Case-B stub-throw helper for the four SparseSkOp-taking
/// sketch_symmetric specializations. The variadic parameter pack consumes
/// the caller's arguments so the compiler doesn't warn about unused
/// parameters in the (genuinely-unused, by design) stub bodies.
template <typename ...Args>
[[noreturn]] inline void throw_sketch_symmetric_case_b(Args&&...) {
    randblas_require(
        false &&
        "RandBLAS::sketch_symmetric with a sparse sketching operator is the "
        "Case-B kernel from project-plans/randblas-symm-plan.md. Not implemented "
        "in this PR --- the API signature is reserved so future PRs can fill in "
        "the body without breaking source compatibility. Composition fallback: "
        "densify the SkOp then call sketch_symmetric on the dense version."
    );
    __builtin_unreachable();
}

} // namespace detail


// MARK: SUBMAT(S)

// =============================================================================
/// \fn sketch_symmetric(blas::Layout layout, blas::Uplo uplo,
///     int64_t d, int64_t n, T alpha,
///     const SKOP &S, int64_t ro_s, int64_t co_s,
///     const T *A, int64_t lda, T beta, T *B, int64_t ldb
/// )
/// @verbatim embed:rst:leading-slashes
/// Sketch from the left in a SYMM-like operation
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\submat(\mtxS)}_{d \times n} \cdot \underbrace{\mat(A)}_{n \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars and :math:`\mtxS` is a sketching operator.
///
/// .. dropdown:: FAQ
///   :animate: fade-in-slide-down
///
///     **What's** :math:`\mat(A)?`
///
///       It's a symmetric matrix of order :math:`n`, stored with only the triangle named by
///       :math:`\texttt{uplo}` populated. The opposite triangle is not read.
///
///       The :math:`\texttt{layout}` parameter governs the indexing convention into :math:`A`:
///
///             .. math::
///                 \mat(A)_{ij} = A[i + j \cdot \lda] \quad (\text{ColMajor})
///                              = A[i \cdot \lda + j] \quad (\text{RowMajor}).
///
///       Unlike the pre-SYMM API, both triangles need not match: only the stored triangle is
///       read by :math:`\ttt{blas::symm}` underneath.
///
/// .. dropdown:: Full parameter descriptions
///     :animate: fade-in-slide-down
///
///      layout - [in]
///       * Either Layout::ColMajor or Layout::RowMajor
///       * Matrix storage for :math:`\mat(A)` and :math:`\mat(B).`
///
///      uplo - [in]
///       * Either Uplo::Upper or Uplo::Lower
///       * Names the triangle of :math:`\mat(A)` that is stored and read.
///
///      d - [in]
///       * A nonnegative integer.
///       * The number of rows in :math:`\mat(B)` and :math:`\submat(\mtxS).`
///
///      n - [in]
///       * A nonnegative integer.
///       * The number of columns in :math:`\mat(B)` and :math:`\submat(\mtxS).`
///       * The number of rows and columns in :math:`\mat(A).`
///
///      alpha - [in]
///       * A real scalar.
///       * If zero, then :math:`A` is not accessed.
///
///      S - [in]
///       * A SketchingOperator object (DenseSkOp or SparseSkOp).
///       * Defines :math:`\submat(\mtxS).` SparseSkOp is currently not supported and
///         calls with a SparseSkOp will throw a :math:`\ttt{RandBLAS::Error}` ---
///         this is the Case-B kernel from `project-plans/randblas-symm-plan.md`.
///
///      ro_s - [in]
///       * A nonnegative integer.
///       * The rows of :math:`\submat(\mtxS)` start at :math:`\mtxS[\texttt{ro_s}, :].`
///
///      co_s - [in]
///       * A nonnegative integer.
///       * The columns of :math:`\submat(\mtxS)` start at :math:`\mtxS[:, \texttt{co_s}].`
///
///      A - [in]
///       * Pointer to a 1D array of real scalars.
///       * Defines :math:`\mat(A).`
///
///      lda - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(A)` when reading from :math:`A.`
///
///      beta - [in]
///       * A real scalar.
///       * If zero, then :math:`B` need not be set on input.
///
///      B - [in, out]
///       * Pointer to 1D array of real scalars.
///       * On entry, defines :math:`\mat(B)` on the RIGHT-hand side of :math:`(\star).`
///       * On exit, defines :math:`\mat(B)` on the LEFT-hand side of :math:`(\star).`
///
///      ldb - [in]
///       * A nonnegative integer.
///       * Leading dimension of :math:`\mat(B)` when reading from :math:`B.`
///
/// @endverbatim
template <SketchingOperator SKOP, typename T = typename SKOP::scalar_t>
inline void sketch_symmetric(
    blas::Layout layout, blas::Uplo uplo,
    int64_t d, int64_t n,
    T alpha,
    const SKOP &S, int64_t ro_s, int64_t co_s,
    const T* A, int64_t lda,
    T beta,
    T* B, int64_t ldb
);

template <typename T, typename RNG>
inline void sketch_symmetric(
    blas::Layout layout, blas::Uplo uplo,
    int64_t d, int64_t n,
    T alpha,
    const DenseSkOp<T, RNG> &S, int64_t ro_s, int64_t co_s,
    const T* A, int64_t lda,
    T beta,
    T* B, int64_t ldb
) {
    RandBLAS::dense::lsksy3(layout, uplo, d, n, alpha, S, ro_s, co_s, A, lda, beta, B, ldb);
}

template <typename T, typename RNG, RandBLAS::SignedInteger sint_t>
inline void sketch_symmetric(
    blas::Layout layout, blas::Uplo uplo,
    int64_t d, int64_t n,
    T alpha,
    const SparseSkOp<T, RNG, sint_t> &S, int64_t ro_s, int64_t co_s,
    const T* A, int64_t lda,
    T beta,
    T* B, int64_t ldb
) {
    detail::throw_sketch_symmetric_case_b(layout, uplo, d, n, alpha, S, ro_s, co_s, A, lda, beta, B, ldb);
}


// =============================================================================
/// \fn sketch_symmetric(blas::Layout layout, blas::Uplo uplo,
///     int64_t n, int64_t d, T alpha,
///     const T *A, int64_t lda,
///     const SKOP &S, int64_t ro_s, int64_t co_s,
///     T beta, T *B, int64_t ldb
/// )
/// @verbatim embed:rst:leading-slashes
/// Sketch from the right in a SYMM-like operation
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\mat(A)}_{n \times n} \cdot \underbrace{\submat(\mtxS)}_{n \times d} + \beta \cdot \underbrace{\mat(B)}_{n \times d},    \tag{$\star$}
///
/// where :math:`\alpha` and :math:`\beta` are real scalars and :math:`\mtxS` is a sketching operator.
///
/// See the left-side overload above for the meaning of :math:`\mat(A)` and its
/// :math:`\texttt{uplo}` storage convention. The roles of :math:`d` and :math:`n` are mirrored:
/// :math:`d` is the embedding dimension (cols of :math:`\submat(\mtxS)`) and :math:`n` is the
/// order of :math:`\mat(A).`
///
/// @endverbatim
template <SketchingOperator SKOP, typename T = typename SKOP::scalar_t>
inline void sketch_symmetric(
    blas::Layout layout, blas::Uplo uplo,
    int64_t n, int64_t d,
    T alpha,
    const T* A, int64_t lda,
    const SKOP &S, int64_t ro_s, int64_t co_s,
    T beta,
    T* B, int64_t ldb
);

template <typename T, typename RNG>
inline void sketch_symmetric(
    blas::Layout layout, blas::Uplo uplo,
    int64_t n, int64_t d,
    T alpha,
    const T* A, int64_t lda,
    const DenseSkOp<T, RNG> &S, int64_t ro_s, int64_t co_s,
    T beta,
    T* B, int64_t ldb
) {
    RandBLAS::dense::rsksy3(layout, uplo, n, d, alpha, A, lda, S, ro_s, co_s, beta, B, ldb);
}

template <typename T, typename RNG, RandBLAS::SignedInteger sint_t>
inline void sketch_symmetric(
    blas::Layout layout, blas::Uplo uplo,
    int64_t n, int64_t d,
    T alpha,
    const T* A, int64_t lda,
    const SparseSkOp<T, RNG, sint_t> &S, int64_t ro_s, int64_t co_s,
    T beta,
    T* B, int64_t ldb
) {
    detail::throw_sketch_symmetric_case_b(layout, uplo, n, d, alpha, A, lda, S, ro_s, co_s, beta, B, ldb);
}


// MARK: FULL(S)

// =============================================================================
/// \fn sketch_symmetric(blas::Layout layout, blas::Uplo uplo, T alpha,
///     const SKOP &S, const T *A, int64_t lda,
///     T beta, T *B, int64_t ldb
/// )
/// @verbatim embed:rst:leading-slashes
/// Sketch from the left in a SYMM-like operation, with :math:`\mtxS` used in full
/// (no submatrix offsets).
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\mtxS}_{d \times n} \cdot \underbrace{\mat(A)}_{n \times n} + \beta \cdot \underbrace{\mat(B)}_{d \times n},    \tag{$\star$}
///
/// The dimensions :math:`d` and :math:`n` are taken from :math:`\mtxS` directly
/// (:math:`d = \mtxS.\ttt{dist.n\_rows}`, :math:`n = \mtxS.\ttt{dist.n\_cols}`).
///
/// See the submatrix overload above for the meaning of :math:`\mat(A)` and
/// :math:`\texttt{uplo}`.
///
/// @endverbatim
template <SketchingOperator SKOP, typename T = typename SKOP::scalar_t>
inline void sketch_symmetric(
    blas::Layout layout, blas::Uplo uplo,
    T alpha,
    const SKOP &S,
    const T* A, int64_t lda,
    T beta,
    T* B, int64_t ldb
);

template <typename T, typename RNG>
inline void sketch_symmetric(
    blas::Layout layout, blas::Uplo uplo,
    T alpha,
    const DenseSkOp<T, RNG> &S,
    const T* A, int64_t lda,
    T beta,
    T* B, int64_t ldb
) {
    int64_t d = S.dist.n_rows;
    int64_t n = S.dist.n_cols;
    RandBLAS::dense::lsksy3(layout, uplo, d, n, alpha, S, 0, 0, A, lda, beta, B, ldb);
}

template <typename T, typename RNG, RandBLAS::SignedInteger sint_t>
inline void sketch_symmetric(
    blas::Layout layout, blas::Uplo uplo,
    T alpha,
    const SparseSkOp<T, RNG, sint_t> &S,
    const T* A, int64_t lda,
    T beta,
    T* B, int64_t ldb
) {
    detail::throw_sketch_symmetric_case_b(layout, uplo, alpha, S, A, lda, beta, B, ldb);
}


// =============================================================================
/// \fn sketch_symmetric(blas::Layout layout, blas::Uplo uplo, T alpha,
///     const T *A, int64_t lda, const SKOP &S,
///     T beta, T *B, int64_t ldb
/// )
/// @verbatim embed:rst:leading-slashes
/// Sketch from the right in a SYMM-like operation, with :math:`\mtxS` used in full.
///
/// .. math::
///     \mat(B) = \alpha \cdot \underbrace{\mat(A)}_{n \times n} \cdot \underbrace{\mtxS}_{n \times d} + \beta \cdot \underbrace{\mat(B)}_{n \times d},    \tag{$\star$}
///
/// The dimensions :math:`n` and :math:`d` are taken from :math:`\mtxS` directly
/// (:math:`n = \mtxS.\ttt{dist.n\_rows}`, :math:`d = \mtxS.\ttt{dist.n\_cols}`).
///
/// See the submatrix overload above for the meaning of :math:`\mat(A)` and
/// :math:`\texttt{uplo}`.
///
/// @endverbatim
template <SketchingOperator SKOP, typename T = typename SKOP::scalar_t>
inline void sketch_symmetric(
    blas::Layout layout, blas::Uplo uplo,
    T alpha,
    const T* A, int64_t lda,
    const SKOP &S,
    T beta,
    T* B, int64_t ldb
);

template <typename T, typename RNG>
inline void sketch_symmetric(
    blas::Layout layout, blas::Uplo uplo,
    T alpha,
    const T* A, int64_t lda,
    const DenseSkOp<T, RNG> &S,
    T beta,
    T* B, int64_t ldb
) {
    int64_t n = S.dist.n_rows;
    int64_t d = S.dist.n_cols;
    RandBLAS::dense::rsksy3(layout, uplo, n, d, alpha, A, lda, S, 0, 0, beta, B, ldb);
}

template <typename T, typename RNG, RandBLAS::SignedInteger sint_t>
inline void sketch_symmetric(
    blas::Layout layout, blas::Uplo uplo,
    T alpha,
    const T* A, int64_t lda,
    const SparseSkOp<T, RNG, sint_t> &S,
    T beta,
    T* B, int64_t ldb
) {
    detail::throw_sketch_symmetric_case_b(layout, uplo, alpha, S, A, lda, beta, B, ldb);
}

} // end namespace RandBLAS
