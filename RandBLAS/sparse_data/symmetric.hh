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

#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"
#include "RandBLAS/exceptions.hh"

#include <type_traits>


namespace RandBLAS::sparse_data {

// =============================================================================
/// Lightweight non-owning wrapper marking a SparseMatrix as symmetric with a
/// stored triangle.
///
/// @verbatim embed:rst:leading-slashes
/// Holds:
///   - A const reference to the underlying SparseMatrix :math:`A.`
///   - :math:`\ttt{blas::Uplo uplo}`: names the triangle of :math:`A` that is
///     structurally populated. The opposite triangle is implied by symmetry.
///
/// The wrapper performs **no validation** of the matrix contents; it is the
/// caller's responsibility to guarantee that the named triangle is correctly
/// populated and that the opposite triangle is either structurally absent or
/// will be ignored by the SYMM-aware consumer (kernels in this library that
/// accept a ``Symmetric<SpMat>`` agree to consult ``uplo`` and respect this
/// contract). Construction does enforce that :math:`A` is square
/// (``A.n_rows == A.n_cols``) via :math:`\ttt{randblas\_require}`.
///
/// This wrapper is how a caller requests symmetric semantics: passing a
/// ``Symmetric<SpMat>`` to :math:`\ttt{spmm}` dispatches to the
/// symmetry-aware kernels, while passing the bare :math:`\ttt{SparseMatrix}`
/// treats the stored entries as a general matrix. The wrapper type is
/// intentionally separate from the :math:`\ttt{SparseMatrix}` concept so the
/// two meanings cannot be confused by accident.
///
/// The wrapper is non-owning: it holds a reference to :math:`A`, not a copy.
/// Keep the named object alive for the lifetime of the wrapper. Wrapping a
/// temporary (for example, the by-value view returned by ``.transpose()``)
/// is rejected at compile time via deleted rvalue overloads, since the
/// temporary would be destroyed at the end of the statement and leave the
/// wrapper dangling.
/// @endverbatim
template <SparseMatrix SpMat>
struct Symmetric {
    const SpMat& A;
    const blas::Uplo uplo;

    using scalar_t = typename SpMat::scalar_t;
    using index_t = typename SpMat::index_t;

    Symmetric(const SpMat& A_in, blas::Uplo uplo_in) : A(A_in), uplo(uplo_in) {
        randblas_require(A_in.n_rows == A_in.n_cols);
    }

    // Binding a temporary would dangle at the end of the statement. The
    // const&& form catches const and non-const rvalues alike (transpose()
    // views are const prvalues), while lvalues still bind to the const&
    // constructor above.
    Symmetric(const SpMat&&, blas::Uplo) = delete;
};


// =============================================================================
/// Construct a ``Symmetric`` wrapper around ``A`` with the named triangle.
/// Syntactic sugar for the constructor; lets callers write
/// ``as_symmetric(A, blas::Uplo::Upper)`` and rely on template argument
/// deduction.
template <SparseMatrix SpMat>
inline Symmetric<SpMat> as_symmetric(const SpMat& A, blas::Uplo uplo) {
    return Symmetric<SpMat>(A, uplo);
}

// Wrapping a temporary (e.g. as_symmetric(A.transpose(), uplo)) would leave
// the wrapper referencing a destroyed object; reject it at compile time.
// The const&& form catches const and non-const rvalues alike without the
// forwarding-reference trap that a plain && overload would create.
template <SparseMatrix SpMat>
Symmetric<SpMat> as_symmetric(const SpMat&& A, blas::Uplo uplo) = delete;


// =============================================================================
/// Expand the stored triangle of a symmetric sparse matrix into an owning
/// general (both triangles populated) COOMatrix.
///
/// @verbatim embed:rst:leading-slashes
/// Reads only the triangle of :math:`A` named by :math:`\ttt{uplo}`; entries
/// outside it are skipped, matching the semantics of the spsymm kernels.
/// Each stored off-diagonal entry :math:`(i, j, v)` is emitted twice (as
/// :math:`(i, j, v)` and :math:`(j, i, v)`); diagonal entries once. Memory
/// cost is :math:`O(\ttt{nnz})`.
///
/// This is the bridge from one-triangle symmetric storage to consumers that
/// only accept general sparse matrices (notably MKL's sparse-times-sparse
/// routines, which reject ``SPARSE_MATRIX_TYPE_SYMMETRIC`` descriptors).
/// @endverbatim
template <SparseMatrix SpMat, typename T = typename SpMat::scalar_t,
          typename sint_t = typename SpMat::index_t>
COOMatrix<T, sint_t> expand_symmetric_to_general(const SpMat& A, blas::Uplo uplo) {
    randblas_require(A.n_rows == A.n_cols);
    randblas_require(A.index_base == IndexBase::Zero);

    constexpr bool is_coo = std::is_same_v<SpMat, COOMatrix<T, sint_t>>;
    constexpr bool is_csr = std::is_same_v<SpMat, CSRMatrix<T, sint_t>>;
    constexpr bool is_csc = std::is_same_v<SpMat, CSCMatrix<T, sint_t>>;
    static_assert(is_coo || is_csr || is_csc,
                  "expand_symmetric_to_general requires COO, CSR, or CSC.");

    bool upper = (uplo == blas::Uplo::Upper);
    // in_triangle(i, j): the entry belongs to the named triangle.
    auto in_triangle = [upper](int64_t i, int64_t j) {
        return upper ? (j >= i) : (j <= i);
    };

    // Pass 1: count the general-matrix entries.
    int64_t nnz_general = 0;
    auto count_entry = [&](int64_t i, int64_t j) {
        if (!in_triangle(i, j)) return;
        nnz_general += (i == j) ? 1 : 2;
    };
    if constexpr (is_coo) {
        for (int64_t p = 0; p < A.nnz; ++p)
            count_entry((int64_t) A.rows[p], (int64_t) A.cols[p]);
    } else if constexpr (is_csr) {
        for (int64_t i = 0; i < A.n_rows; ++i)
            for (int64_t p = A.rowptr[i]; p < A.rowptr[i+1]; ++p)
                count_entry(i, (int64_t) A.colidxs[p]);
    } else {
        for (int64_t j = 0; j < A.n_cols; ++j)
            for (int64_t p = A.colptr[j]; p < A.colptr[j+1]; ++p)
                count_entry((int64_t) A.rowidxs[p], j);
    }

    // Pass 2: fill.
    COOMatrix<T, sint_t> G(A.n_rows, A.n_cols);
    if (nnz_general == 0) return G;
    reserve_coo(nnz_general, G);
    int64_t q = 0;
    auto emit_entry = [&](int64_t i, int64_t j, T v) {
        if (!in_triangle(i, j)) return;
        G.rows[q] = (sint_t) i; G.cols[q] = (sint_t) j; G.vals[q] = v; ++q;
        if (i != j) {
            G.rows[q] = (sint_t) j; G.cols[q] = (sint_t) i; G.vals[q] = v; ++q;
        }
    };
    if constexpr (is_coo) {
        for (int64_t p = 0; p < A.nnz; ++p)
            emit_entry((int64_t) A.rows[p], (int64_t) A.cols[p], A.vals[p]);
    } else if constexpr (is_csr) {
        for (int64_t i = 0; i < A.n_rows; ++i)
            for (int64_t p = A.rowptr[i]; p < A.rowptr[i+1]; ++p)
                emit_entry(i, (int64_t) A.colidxs[p], A.vals[p]);
    } else {
        for (int64_t j = 0; j < A.n_cols; ++j)
            for (int64_t p = A.colptr[j]; p < A.colptr[j+1]; ++p)
                emit_entry((int64_t) A.rowidxs[p], j, A.vals[p]);
    }
    return G;
}

} // end namespace RandBLAS::sparse_data


namespace RandBLAS {
    using RandBLAS::sparse_data::Symmetric;
    using RandBLAS::sparse_data::as_symmetric;
}
