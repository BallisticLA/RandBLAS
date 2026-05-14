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
//   - lsksys, rsksys: Case B (dense-symm A x sparse SkOp), per-stored-entry
//     two-axpy scatter that reads only the uplo triangle of A.
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
/// operation). When layouts mismatch, SYMM cannot transpose S on the fly, so
/// we transpose-copy S into a tight buffer matching the caller's layout (cost:
/// `O(d * n)` for the copy) and then call SYMM on the copy. This keeps the
/// SYMM speedup on the matvec, at the cost of the one-time copy.
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
        // Layout mismatch: transpose-copy S into a tight buffer in the
        // caller's layout, then SYMM. Costs O(d*n) for the copy, but keeps
        // the SYMM speedup (~1.3-1.8x over GEMM) on the d*n*n_A matvec.
        int64_t lds_new = (layout == blas::Layout::ColMajor) ? d : n;
        std::vector<T> S_copy(static_cast<size_t>(d) * static_cast<size_t>(n));
        auto [irs_in,  ics_in]  = layout_to_strides(S.layout, lds);
        auto [irs_out, ics_out] = layout_to_strides(layout, lds_new);
        util::omatcopy(d, n, S_ptr, irs_in, ics_in,
                       S_copy.data(), irs_out, ics_out);
        blas::symm(layout, blas::Side::Right, uplo, d, n,
                   alpha, A, lda, S_copy.data(), lds_new, beta, B, ldb);
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
        // Layout mismatch: transpose-copy S into a tight matching-layout
        // buffer, then SYMM. See lsksy3 for the trade-off discussion.
        int64_t lds_new = (layout == blas::Layout::ColMajor) ? n : d;
        std::vector<T> S_copy(static_cast<size_t>(n) * static_cast<size_t>(d));
        auto [irs_in,  ics_in]  = layout_to_strides(S.layout, lds);
        auto [irs_out, ics_out] = layout_to_strides(layout, lds_new);
        util::omatcopy(n, d, S_ptr, irs_in, ics_in,
                       S_copy.data(), irs_out, ics_out);
        blas::symm(layout, blas::Side::Left, uplo, n, d,
                   alpha, A, lda, S_copy.data(), lds_new, beta, B, ldb);
    }
    return;
}

} // end namespace RandBLAS::dense


namespace RandBLAS::sparse {

// =============================================================================
/// LSKSYS: dense symmetric A on the right of a SparseSkOp.
///
/// Computes B = alpha * submat(S) * mat(A) + beta * B, where:
///   - mat(A) is n-by-n dense symmetric, with only the `uplo` triangle stored.
///   - submat(S) is the d-by-n view of S at (ro_s, co_s); S is a SparseSkOp.
///   - mat(B) is d-by-n dense.
///
/// Inner loop: for each stored nonzero (row_S, col_S, v) of S inside the
/// submatrix window, contribute alpha*v * (row col_S of A) to row (row_S - ro_s)
/// of B. The row of symmetric A is assembled from the stored triangle as two
/// blas::axpy calls (one along the stored column, one along the stored row past
/// the diagonal).
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
    if (S.nnz < 0) {
        SparseSkOp<T, RNG, sint_t> shallowcopy(S.dist, S.seed_state);
        fill_sparse(shallowcopy);
        lsksys(layout, uplo, d, n, alpha, shallowcopy, ro_s, co_s, A, lda, beta, B, ldb);
        return;
    }

    util::lascl(layout, d, n, beta, B, ldb);
    if (alpha == T(0)) return;

    auto Scoo = coo_view_of_skop(S);
    const bool col_major = (layout == blas::Layout::ColMajor);

    for (int64_t p = 0; p < Scoo.nnz; ++p) {
        sint_t row_S = Scoo.rows[p];
        sint_t col_S = Scoo.cols[p];
        if (row_S < ro_s || row_S >= ro_s + d) continue;
        if (col_S < co_s || col_S >= co_s + n) continue;
        int64_t i = static_cast<int64_t>(row_S) - ro_s;  // B row
        int64_t j = static_cast<int64_t>(col_S) - co_s;  // A row index (sym A read as row j)
        T av = alpha * Scoo.vals[p];

        // B[i, :] += av * row_j(A_sym), split by uplo into two contiguous
        // ranges of A so each becomes a single axpy.
        if (uplo == blas::Uplo::Upper) {
            // row j of A:
            //   c in [0, j]: read A[c, j]      (column j of stored Upper, rows 0..j)
            //   c in (j, n-1]: read A[j, c]    (row j of stored Upper, cols j+1..n-1)
            if (col_major) {
                blas::axpy(j + 1, av, &A[j * lda], 1, &B[i], ldb);
                blas::axpy(n - j - 1, av, &A[j + (j + 1) * lda], lda, &B[i + (j + 1) * ldb], ldb);
            } else {
                blas::axpy(j + 1, av, &A[j], lda, &B[i * ldb], 1);
                blas::axpy(n - j - 1, av, &A[j * lda + j + 1], 1, &B[i * ldb + j + 1], 1);
            }
        } else {
            // Lower-stored:
            //   c in [0, j]: read A[j, c]      (row j of stored Lower, cols 0..j)
            //   c in (j, n-1]: read A[c, j]    (column j of stored Lower, rows j+1..n-1)
            if (col_major) {
                blas::axpy(j + 1, av, &A[j], lda, &B[i], ldb);
                blas::axpy(n - j - 1, av, &A[(j + 1) + j * lda], 1, &B[i + (j + 1) * ldb], ldb);
            } else {
                blas::axpy(j + 1, av, &A[j * lda], 1, &B[i * ldb], 1);
                blas::axpy(n - j - 1, av, &A[(j + 1) * lda + j], lda, &B[i * ldb + j + 1], 1);
            }
        }
    }
}


// =============================================================================
/// RSKSYS: dense symmetric A on the left of a SparseSkOp.
///
/// Computes B = alpha * mat(A) * submat(S) + beta * B, where:
///   - mat(A) is n-by-n dense symmetric, with only the `uplo` triangle stored.
///   - submat(S) is the n-by-d view of S at (ro_s, co_s); S is a SparseSkOp.
///   - mat(B) is n-by-d dense.
///
/// Same two-axpy scatter pattern as lsksys, but on columns of A (the column
/// of A indexed by row_S of S contributes into the column of B indexed by
/// col_S - co_s).
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
    if (S.nnz < 0) {
        SparseSkOp<T, RNG, sint_t> shallowcopy(S.dist, S.seed_state);
        fill_sparse(shallowcopy);
        rsksys(layout, uplo, n, d, alpha, A, lda, shallowcopy, ro_s, co_s, beta, B, ldb);
        return;
    }

    util::lascl(layout, n, d, beta, B, ldb);
    if (alpha == T(0)) return;

    auto Scoo = coo_view_of_skop(S);
    const bool col_major = (layout == blas::Layout::ColMajor);

    for (int64_t p = 0; p < Scoo.nnz; ++p) {
        sint_t row_S = Scoo.rows[p];
        sint_t col_S = Scoo.cols[p];
        if (row_S < ro_s || row_S >= ro_s + n) continue;
        if (col_S < co_s || col_S >= co_s + d) continue;
        int64_t i = static_cast<int64_t>(row_S) - ro_s;  // A column index
        int64_t j = static_cast<int64_t>(col_S) - co_s;  // B column
        T av = alpha * Scoo.vals[p];

        // B[:, j] += av * col_i(A_sym), split by uplo:
        if (uplo == blas::Uplo::Upper) {
            // col i of A:
            //   r in [0, i]: read A[r, i]    (column i of stored Upper, rows 0..i)
            //   r in (i, n-1]: read A[i, r]  (row i of stored Upper, cols i+1..n-1)
            if (col_major) {
                blas::axpy(i + 1, av, &A[i * lda], 1, &B[j * ldb], 1);
                blas::axpy(n - i - 1, av, &A[i + (i + 1) * lda], lda, &B[(i + 1) + j * ldb], 1);
            } else {
                blas::axpy(i + 1, av, &A[i], lda, &B[j], ldb);
                blas::axpy(n - i - 1, av, &A[i * lda + i + 1], 1, &B[(i + 1) * ldb + j], ldb);
            }
        } else {
            // Lower-stored col i:
            //   r in [0, i-1]: read A[i, r]  (row i of stored Lower, cols 0..i-1)
            //   r in [i, n-1]: read A[r, i]  (column i of stored Lower, rows i..n-1)
            if (col_major) {
                blas::axpy(i, av, &A[i], lda, &B[j * ldb], 1);
                blas::axpy(n - i, av, &A[i + i * lda], 1, &B[i + j * ldb], 1);
            } else {
                blas::axpy(i, av, &A[i * lda], 1, &B[j], ldb);
                blas::axpy(n - i, av, &A[i * lda + i], lda, &B[i * ldb + j], ldb);
            }
        }
    }
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
    RandBLAS::sparse::lsksys(layout, uplo, d, n, alpha, S, ro_s, co_s, A, lda, beta, B, ldb);
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
    RandBLAS::sparse::rsksys(layout, uplo, n, d, alpha, A, lda, S, ro_s, co_s, beta, B, ldb);
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
    int64_t d = S.dist.n_rows;
    int64_t n = S.dist.n_cols;
    RandBLAS::sparse::lsksys(layout, uplo, d, n, alpha, S, 0, 0, A, lda, beta, B, ldb);
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
    int64_t n = S.dist.n_rows;
    int64_t d = S.dist.n_cols;
    RandBLAS::sparse::rsksys(layout, uplo, n, d, alpha, A, lda, S, 0, 0, beta, B, ldb);
}

} // end namespace RandBLAS
