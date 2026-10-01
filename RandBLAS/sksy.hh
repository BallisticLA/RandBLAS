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
#include "RandBLAS/sparse_data/coo_sksys_impl.hh"

#include <type_traits>
#include <vector>


// =============================================================================
// Symmetric sketching helpers (SYMM-backed). See sparse_data/DevNotes.md for
// the four-case design (dense/sparse symmetric operand x dense/sparse factor).
//   - lsksy3, rsksy3: Case A (dense-symm A x dense Omega), via blas::symm.
//   - lsksys, rsksys: Case B (dense-symm A x sparse SkOp). Thin wrappers
//     handling validation, beta, and SparseSkOp materialization; the actual
//     column-driven accumulation kernel lives in
//     sparse_data/coo_sksys_impl.hh as coo_lsksys / coo_rsksys.
// =============================================================================

namespace RandBLAS::dense {

// =============================================================================
// Copy the n_rows-by-n_cols matrix at src (read in src_layout with leading
// dimension ld_src) into a tight buffer laid out in target_layout, whose
// leading dimension is written to ld_out. Used by lsksy3 / rsksy3 when the
// SkOp's storage layout mismatches the caller's: blas::symm cannot transpose
// an operand on the fly, so the copy keeps the SYMM speedup at an O(size)
// one-time cost.
// =============================================================================
template <typename T>
inline std::vector<T> transpose_copy_to_layout(
    blas::Layout target_layout, int64_t n_rows, int64_t n_cols,
    const T* src, blas::Layout src_layout, int64_t ld_src,
    int64_t &ld_out
) {
    ld_out = (target_layout == blas::Layout::ColMajor) ? n_rows : n_cols;
    std::vector<T> out(static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols));
    auto [irs_in,  ics_in]  = layout_to_strides(src_layout, ld_src);
    auto [irs_out, ics_out] = layout_to_strides(target_layout, ld_out);
    util::omatcopy(n_rows, n_cols, src, irs_in, ics_in, out.data(), irs_out, ics_out);
    return out;
}

// =============================================================================
// LSKSY3: SYMM-backed left-sketch with a symmetric matrix A.
//
// Computes B = alpha * submat(S) * mat(A) + beta * B, where:
//   - mat(A) is n-by-n symmetric. Only the triangle named by `uplo` is read.
//   - submat(S) is the d-by-n view of S at (ro_s, co_s).
//   - mat(B) is d-by-n.
//
// When S has no materialized buffer, the submatrix is realized via
// `submatrix_as_blackbox` (same pattern as `lskge3`). When the buffered S's
// storage layout matches the caller's `layout`, the final call is
// `blas::symm` with `side = Right` (since A is on the right of S in the
// operation). When layouts mismatch, SYMM cannot transpose S on the fly, so
// we transpose-copy S into a tight buffer matching the caller's layout (cost:
// `O(d * n)` for the copy) and then call SYMM on the copy. This keeps the
// SYMM speedup on the matvec, at the cost of the one-time copy.
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
) {
    constexpr bool maybe_denseskop = !std::is_same_v<std::remove_cv_t<DenseSkOp>, BLASFriendlyOperator<T>>;
    if constexpr (maybe_denseskop) {
        if (!S.buff) {
            auto submat_S = submatrix_as_blackbox<BLASFriendlyOperator<T>>(S, d, n, ro_s, co_s);
            lsksy3(layout, uplo, d, n, alpha, submat_S, 0, 0, A, lda, beta, B, ldb);
            return;
        }
    }
    randblas_require( S.buff != nullptr );
    validate_submat_dims(S.n_rows, S.n_cols, d, n, ro_s, co_s);
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
        int64_t lds_new;
        auto S_copy = transpose_copy_to_layout(layout, d, n, S_ptr, S.layout, lds, lds_new);
        blas::symm(layout, blas::Side::Right, uplo, d, n,
                   alpha, A, lda, S_copy.data(), lds_new, beta, B, ldb);
    }
    return;
}


// =============================================================================
// RSKSY3: SYMM-backed right-sketch with a symmetric matrix A.
//
// Computes B = alpha * mat(A) * submat(S) + beta * B, where:
//   - mat(A) is n-by-n symmetric. Only the triangle named by `uplo` is read.
//   - submat(S) is the n-by-d view of S at (ro_s, co_s).
//   - mat(B) is n-by-d.
//
// Same materialization and layout-mismatch fallback semantics as `lsksy3`.
// Final call (matching layout): `blas::symm` with `side = Left`.
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
) {
    constexpr bool maybe_denseskop = !std::is_same_v<std::remove_cv_t<DenseSkOp>, BLASFriendlyOperator<T>>;
    if constexpr (maybe_denseskop) {
        if (!S.buff) {
            auto submat_S = submatrix_as_blackbox<BLASFriendlyOperator<T>>(S, n, d, ro_s, co_s);
            rsksy3(layout, uplo, n, d, alpha, A, lda, submat_S, 0, 0, beta, B, ldb);
            return;
        }
    }
    randblas_require( S.buff != nullptr );
    validate_submat_dims(S.n_rows, S.n_cols, n, d, ro_s, co_s);
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
        int64_t lds_new;
        auto S_copy = transpose_copy_to_layout(layout, n, d, S_ptr, S.layout, lds, lds_new);
        blas::symm(layout, blas::Side::Left, uplo, n, d,
                   alpha, A, lda, S_copy.data(), lds_new, beta, B, ldb);
    }
    return;
}

} // end namespace RandBLAS::dense


namespace RandBLAS::sparse {

// =============================================================================
// LSKSYS: dense symmetric A on the right of a SparseSkOp.
//
// Computes B = alpha * submat(S) * mat(A) + beta * B, where:
//   - mat(A) is n-by-n dense symmetric, with only the `uplo` triangle stored.
//   - submat(S) is the d-by-n view of S at (ro_s, co_s); S is a SparseSkOp.
//   - mat(B) is d-by-n dense.
//
// Validation mirrors the dense-path lsksy3. When S is unmaterialized, only
// the requested d-by-n window is sampled (submatrix_as_coo, the same pattern
// lskges uses); a materialized S is consumed through a lightweight COO view
// with the window filtered inside the kernel. The kernel itself
// (sparse_data::coo_lsksys) is a column-driven pure accumulator; beta is
// applied here, exactly once.
template <typename T, typename RNG, SignedInteger sint_t>
void lsksys(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t d,
    int64_t n,
    T alpha,
    const SparseSkOp<T, RNG, sint_t> &S,
    int64_t ro_s,
    int64_t co_s,
    const T *A,
    int64_t lda,
    T beta,
    T *B,
    int64_t ldb
) {
    validate_submat_dims(S.n_rows, S.n_cols, d, n, ro_s, co_s);
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= n);
        randblas_require(ldb >= d);
    } else {
        randblas_require(lda >= n);
        randblas_require(ldb >= n);
    }

    util::lascl(layout, d, n, beta, B, ldb);
    if (alpha == T(0)) return;

    if (S.nnz < 0) {
        // Sample only the requested window rather than materializing all of S.
        auto Ssub = submatrix_as_coo(S, d, n, ro_s, co_s);
        RandBLAS::sparse_data::coo_lsksys(
            layout, uplo, d, n, alpha, Ssub, 0, 0, A, lda, B, ldb
        );
        return;
    }
    auto Scoo = coo_view_of_skop(S);
    RandBLAS::sparse_data::coo_lsksys(
        layout, uplo, d, n, alpha, Scoo, ro_s, co_s, A, lda, B, ldb
    );
}


// =============================================================================
// RSKSYS: dense symmetric A on the left of a SparseSkOp.
//
// Computes B = alpha * mat(A) * submat(S) + beta * B, where:
//   - mat(A) is n-by-n dense symmetric, with only the `uplo` triangle stored.
//   - submat(S) is the n-by-d view of S at (ro_s, co_s); S is a SparseSkOp.
//   - mat(B) is n-by-d dense.
//
// Same validation, window-sampling, and beta conventions as lsksys; the
// kernel (sparse_data::coo_rsksys) reduces to coo_lsksys via the transpose
// identity.
template <typename T, typename RNG, SignedInteger sint_t>
void rsksys(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    int64_t d,
    T alpha,
    const T *A,
    int64_t lda,
    const SparseSkOp<T, RNG, sint_t> &S,
    int64_t ro_s,
    int64_t co_s,
    T beta,
    T *B,
    int64_t ldb
) {
    validate_submat_dims(S.n_rows, S.n_cols, n, d, ro_s, co_s);
    if (layout == blas::Layout::ColMajor) {
        randblas_require(lda >= n);
        randblas_require(ldb >= n);
    } else {
        randblas_require(lda >= n);
        randblas_require(ldb >= d);
    }

    util::lascl(layout, n, d, beta, B, ldb);
    if (alpha == T(0)) return;

    if (S.nnz < 0) {
        // Sample only the requested window rather than materializing all of S.
        auto Ssub = submatrix_as_coo(S, n, d, ro_s, co_s);
        RandBLAS::sparse_data::coo_rsksys(
            layout, uplo, n, d, alpha, A, lda, Ssub, 0, 0, B, ldb
        );
        return;
    }
    auto Scoo = coo_view_of_skop(S);
    RandBLAS::sparse_data::coo_rsksys(
        layout, uplo, n, d, alpha, A, lda, Scoo, ro_s, co_s, B, ldb
    );
}

} // end namespace RandBLAS::sparse


namespace RandBLAS {

using namespace RandBLAS::dense;
using namespace RandBLAS::sparse;


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
///       * Defines :math:`\submat(\mtxS).` DenseSkOp dispatches to a SYMM-backed
///         kernel (Case A); SparseSkOp dispatches to a column-driven
///         accumulation kernel that reads only the named triangle of
///         :math:`A`. See ``RandBLAS/sparse_data/DevNotes.md``.
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
) {
    if constexpr (requires { S.buff; S.layout; }) {
        RandBLAS::dense::lsksy3(layout, uplo, d, n, alpha, S, ro_s, co_s, A, lda, beta, B, ldb);
    } else if constexpr (requires { S.nnz; }) {
        RandBLAS::sparse::lsksys(layout, uplo, d, n, alpha, S, ro_s, co_s, A, lda, beta, B, ldb);
    } else {
        static_assert(sizeof(SKOP) == 0,
            "sketch_symmetric supports DenseSkOp and SparseSkOp. For other "
            "SketchingOperator types, apply the operator to a fully-stored A "
            "with sketch_general.");
        // see GitHub PR #155 for why we don't use static_assert(false, ...).
    }
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
) {
    if constexpr (requires { S.buff; S.layout; }) {
        RandBLAS::dense::rsksy3(layout, uplo, n, d, alpha, A, lda, S, ro_s, co_s, beta, B, ldb);
    } else if constexpr (requires { S.nnz; }) {
        RandBLAS::sparse::rsksys(layout, uplo, n, d, alpha, A, lda, S, ro_s, co_s, beta, B, ldb);
    } else {
        static_assert(sizeof(SKOP) == 0,
            "sketch_symmetric supports DenseSkOp and SparseSkOp. For other "
            "SketchingOperator types, apply the operator to a fully-stored A "
            "with sketch_general.");
        // see GitHub PR #155 for why we don't use static_assert(false, ...).
    }
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
) {
    int64_t d = S.dist.n_rows;
    int64_t n = S.dist.n_cols;
    sketch_symmetric(layout, uplo, d, n, alpha, S, 0, 0, A, lda, beta, B, ldb);
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
) {
    int64_t n = S.dist.n_rows;
    int64_t d = S.dist.n_cols;
    sketch_symmetric(layout, uplo, n, d, alpha, A, lda, S, 0, 0, beta, B, ldb);
}


// MARK: LEGACY (sym_check_tol)

// =============================================================================
// The four overloads below reproduce the pre-Uplo sketch_symmetric API and
// semantics exactly: mat(A) must be stored in the format of a general matrix
// with BOTH triangles populated (both are read), a runtime symmetry check
// runs first (skip it, at your own peril, by passing sym_check_tol < 0), and
// the operation forwards to sketch_general. They are retained so that
// existing call sites keep compiling and behaving identically; new code
// should prefer the blas::Uplo overloads above, which read only the named
// triangle and skip the O(n^2) runtime check.
// =============================================================================

// =============================================================================
/// \fn sketch_symmetric(blas::Layout layout, int64_t d, int64_t n, T alpha,
///     const SKOP &S, int64_t ro_s, int64_t co_s, const T *A, int64_t lda,
///     T beta, T *B, int64_t ldb, T sym_check_tol = 0
/// )
/// @verbatim embed:rst:leading-slashes
/// Legacy overload, retained for API compatibility. Check that :math:`\mat(A)`
/// is symmetric up to tolerance :math:`\texttt{sym_check_tol}` (pass a negative
/// tolerance to skip the check), then sketch from the left:
/// :math:`\mat(B) = \alpha \cdot \submat(\mtxS) \cdot \mat(A) + \beta \cdot \mat(B)`.
/// Requires both triangles of :math:`\mat(A)` populated; both are read.
/// Prefer the ``blas::Uplo`` overloads for new code.
/// @endverbatim
template <SketchingOperator SKOP, typename T = typename SKOP::scalar_t>
inline void sketch_symmetric(
    // B = alpha*S*A + beta*B
    blas::Layout layout,
    int64_t d, int64_t n,
    T alpha,
    const SKOP &S, int64_t ro_s, int64_t co_s,
    const T* A, int64_t lda,
    T beta,
    T* B, int64_t ldb,
    T sym_check_tol = 0
) {
    RandBLAS::util::require_symmetric(layout, A, n, lda, sym_check_tol);
    sketch_general(layout, blas::Op::NoTrans, blas::Op::NoTrans, d, n, n, alpha, S, ro_s, co_s, A, lda, beta, B, ldb);
}

// =============================================================================
/// \fn sketch_symmetric(blas::Layout layout, int64_t n, int64_t d, T alpha,
///     const T *A, int64_t lda, const SKOP &S, int64_t ro_s, int64_t co_s,
///     T beta, T *B, int64_t ldb, T sym_check_tol = 0
/// )
/// @verbatim embed:rst:leading-slashes
/// Legacy overload, retained for API compatibility. Same contract as its
/// left-sketch sibling above, sketching from the right:
/// :math:`\mat(B) = \alpha \cdot \mat(A) \cdot \submat(\mtxS) + \beta \cdot \mat(B)`.
/// @endverbatim
template <SketchingOperator SKOP, typename T = typename SKOP::scalar_t>
inline void sketch_symmetric(
    // B = alpha*A*S + beta*B, where A is a symmetric matrix stored in the format of a general matrix.
    blas::Layout layout,
    int64_t n, int64_t d,
    T alpha,
    const T* A, int64_t lda,
    const SKOP &S, int64_t ro_s, int64_t co_s,
    T beta,
    T* B, int64_t ldb,
    T sym_check_tol = 0
) {
    RandBLAS::util::require_symmetric(layout, A, n, lda, sym_check_tol);
    sketch_general(layout, blas::Op::NoTrans, blas::Op::NoTrans, n, d, n, alpha, A, lda, S, ro_s, co_s, beta, B, ldb);
}

// =============================================================================
/// \fn sketch_symmetric(blas::Layout layout, T alpha, const SKOP &S,
///     const T *A, int64_t lda, T beta, T *B, int64_t ldb, T sym_check_tol = 0
/// )
/// @verbatim embed:rst:leading-slashes
/// Legacy overload, retained for API compatibility; :math:`\mtxS` used in
/// full. Same contract as the submatrix legacy overloads.
/// @endverbatim
template <SketchingOperator SKOP, typename T = typename SKOP::scalar_t>
inline void sketch_symmetric(
    // B = alpha*S*A + beta*B
    blas::Layout layout,
    T alpha,
    const SKOP &S,
    const T* A, int64_t lda,
    T beta,
    T* B, int64_t ldb,
    T sym_check_tol = 0
) {
    int64_t d = S.dist.n_rows;
    int64_t n = S.dist.n_cols;
    RandBLAS::util::require_symmetric(layout, A, n, lda, sym_check_tol);
    sketch_general(layout, blas::Op::NoTrans, blas::Op::NoTrans, d, n, n, alpha, S, 0, 0, A, lda, beta, B, ldb);
}

// =============================================================================
/// \fn sketch_symmetric(blas::Layout layout, T alpha, const T *A, int64_t lda,
///     const SKOP &S, T beta, T *B, int64_t ldb, T sym_check_tol = 0
/// )
/// @verbatim embed:rst:leading-slashes
/// Legacy overload, retained for API compatibility; :math:`\mtxS` used in
/// full, sketching from the right. Same contract as the submatrix legacy
/// overloads.
/// @endverbatim
template <SketchingOperator SKOP, typename T = typename SKOP::scalar_t>
inline void sketch_symmetric(
    // B = alpha*A*S + beta*B, where A is a symmetric matrix stored in the format of a general matrix.
    blas::Layout layout,
    T alpha,
    const T* A, int64_t lda,
    const SKOP &S,
    T beta,
    T* B, int64_t ldb,
    T sym_check_tol = 0
) {
    int64_t n = S.dist.n_rows;
    int64_t d = S.dist.n_cols;
    RandBLAS::util::require_symmetric(layout, A, n, lda, sym_check_tol);
    sketch_general(layout, blas::Op::NoTrans, blas::Op::NoTrans, n, d, n, alpha, A, lda, S, 0, 0, beta, B, ldb);
}

} // end namespace RandBLAS
