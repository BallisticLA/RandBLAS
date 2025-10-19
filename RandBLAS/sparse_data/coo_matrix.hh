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

#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/conversions.hh"


#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>


namespace RandBLAS::sparse_data {

#ifdef __cpp_concepts
using RandBLAS::SignedInteger;
#else
#define SignedInteger typename
#endif


template <typename T, SignedInteger sint_t>
COOMatrix<T, sint_t> deepcopy_coo(const COOMatrix<T, sint_t> &A) {
    COOMatrix<T, sint_t> other(A.n_rows, A.n_cols);
    if (A.nnz > 0) {
        other.reserve(A.nnz);
        std::copy(A.rows, A.rows + A.nnz, other.rows);
        std::copy(A.cols, A.cols + A.nnz, other.cols);
        std::copy(A.vals, A.vals + A.nnz, other.vals);
        other.sort = A.sort;
    }
    other.index_base = A.index_base;
    return other;
}


// =============================================================================
/// Let \math{\mtxA} denote a sparse matrix with \math{\ttt{nnz}} structural nonzeros.
/// Its COO representation consists of declared dimensions, \math{\ttt{n_rows}}
/// and \math{\ttt{n_cols}}, as well as a triplet of arrays 
/// \math{(\ttt{vals},\ttt{rows},\ttt{cols})} where
/// @verbatim embed:rst:leading-slashes
///
/// .. math::
///
///         \mtxA_{\ttt{rows}[\ell],\ttt{cols}[\ell]} = \ttt{vals}[\ell] \quad\text{for all}\quad  \ell \in \{0,\ldots,\ttt{nnz}-1\}.
///
/// @endverbatim
///  This type conforms to the SparseMatrix concept.
template <typename T, SignedInteger sint_t = int64_t>
struct COOMatrix {

    // ---------------------------------------------------------------------------
    /// Real scalar type used for structural nonzeros in this matrix.
    using scalar_t = T;

    // ---------------------------------------------------------------------------
    /// Signed integer type used in the rows and cols array members.
    using index_t = sint_t; 

    // ----------------------------------------------------------------------------
    ///  The number of rows in this sparse matrix.
    const int64_t n_rows;

    // ----------------------------------------------------------------------------
    ///  The number of columns in this sparse matrix.
    const int64_t n_cols;

    // ----------------------------------------------------------------------------
    ///  If true, then RandBLAS has permission to allocate and attach memory to the reference
    ///  members of this matrix (vals, rows, and cols). If true *at destruction time*, then delete []
    ///  will be called on each non-null reference member of this matrix.
    ///
    ///  RandBLAS only writes to this member at construction time.
    ///
    bool own_memory;
    
    // ---------------------------------------------------------------------------
    ///  The number of structral nonzeros in this matrix.
    int64_t nnz;
    
    // ---------------------------------------------------------------------------
    ///  A flag to indicate whether (rows, cols) are interpreted
    ///  with zero-based or one-based indexing.
    IndexBase index_base;
    
    // ---------------------------------------------------------------------------
    ///  Reference to an array that holds values for the structural nonzeros of this matrix.
    ///
    ///  If non-null, this must have length at least nnz.
    ///  \internal
    ///  **Memory management note.** Because this length requirement is not a function of
    ///  only const variables, calling A.reserve(nnz) on a COOMatrix "A" will raise an error
    ///  if A.vals is non-null.
    T *vals;
    
    // ---------------------------------------------------------------------------
    ///  Reference to an array that holds row indices for the structural nonzeros of this matrix.
    ///
    ///  If non-null, this must have length at least nnz.
    ///  \internal
    ///  **Memory management note.** Because this length requirement is not a function of
    ///  only const variables, calling A.reserve(nnz) on a COOMatrix "A" will raise an error
    ///  if A.rows is non-null.
    ///  \endinternal
    sint_t *rows;
    
    // ---------------------------------------------------------------------------
    ///  Reference to an array that holds column indicies for the structural nonzeros of this matrix.
    ///
    ///  If non-null, this must have length at least nnz.
    ///  \internal
    ///  **Memory management note.** Because this length requirement is not a function of
    ///  only const variables, calling A.reserve(nnz) on a COOMatrix "A" will raise an error
    ///  if A.cols is non-null.
    ///  \endinternal
    sint_t *cols;

    // ---------------------------------------------------------------------------
    ///  A flag to indicate if the data in (vals, rows, cols) is sorted in a 
    ///  CSC-like order, a CSR-like order, or neither order.
    NonzeroSort sort;

    // ---------------------------------------------------------------------------
    ///  **Standard constructor.** Initializes n_rows and n_cols at the provided values.
    ///  The vals, rows, and cols members are set to null pointers;
    ///  nnz is set to Zero, index_base is set to
    ///  zero, and COOMatrix::own_memory is set to true.
    ///  
    ///  This constructor is intended for use with COOMatrix::reserve(int64_t nnz).
    ///
    COOMatrix(
        int64_t n_rows,
        int64_t n_cols
    ) : n_rows(n_rows), n_cols(n_cols), own_memory(true), nnz(0), index_base(IndexBase::Zero),
        vals(nullptr), rows(nullptr), cols(nullptr), sort(NonzeroSort::None) {};

    // ---------------------------------------------------------------------------
    /// **Expert constructor.** Arguments passed to this function are used to initialize members of the same names;
    /// COOMatrix::own_memory is set to false.
    /// If compute_sort_type is true, then the sort member will be computed by inspecting
    /// the contents of (rows, cols). If compute_sort_type is false, then the sort member is set to None.
    /// 
    COOMatrix(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        T *vals,
        sint_t *rows,
        sint_t *cols,
        bool compute_sort_type = true,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), own_memory(false), nnz(nnz), index_base(index_base),
        vals(vals), rows(rows), cols(cols) {
        if (compute_sort_type) {
            sort = coo_arrays_determine_sort(nnz, rows, cols);
        } else {
            sort = NonzeroSort::None;
        }
    };

    ~COOMatrix() {
        if (own_memory) coo_arrays_free(vals, rows, cols);
    };

    // -----------------------------------------------------
    /// This function requires that own_memory is true, that arg_nnz > 0, 
    //  and that vals, rows, and cols are null. If any of these conditions
    /// are not met then this function will raise an error.
    /// 
    /// If no error is raised then this function redirects
    /// vals, rows, and cols to new arrays of length arg_nnz,
    /// and nnz is set to arg_nnz.
    ///
    void reserve(int64_t arg_nnz) {
        randblas_require(arg_nnz > 0);
        randblas_require(own_memory);
        coo_arrays_allocate(arg_nnz, vals, rows, cols);
        nnz = arg_nnz;
        return;
    };

    // -----------------------------------------------------
    /// Sort the (vals, rows, cols) underlying this COOMatrix for
    /// fast conversion to CSR format (if s == NonzeroSort::CSR)
    /// or CSC format (if s == NonzeroSort::CSC). 
    ///
    /// This function has no effect if `sort == s` or `s == NonzeroSort::None`.
    /// 
    void sort_arrays(NonzeroSort s) {
        coo_arrays_apply_sort(s, nnz, vals, rows, cols);
        if (s != NonzeroSort::None)
            sort = s;
    };

    // ---------------------------------------------------------
    /// This function requires that n_rows == n_cols and index_base == Zero.
    ///
    /// perm is a permutation of {0, 1, ..., n_rows - 1}. It defines a 
    /// permutation matrix P = I(perm,:) of order n_rows.
    ///
    /// Let A denote the abstract mathematical object represented by this
    /// COOMatrix. This function overwrites A by
    ///
    ///     A := P * A * P'.
    ///
    /// In MATLAB notation, this is equivalent to A := A(perm, perm).
    ///
    template <SignedInteger index_t>
    void symperm_inplace(const index_t* perm, bool preserve_sort = true) {
        randblas_require(n_rows == n_cols);
        randblas_require(index_base == IndexBase::Zero);
        apply_index_mapper(n_rows, perm, nnz, rows);
        apply_index_mapper(n_rows, perm, nnz, cols);
        if (preserve_sort) { 
            sort_arrays(sort);
        } else {
            sort = NonzeroSort::None;
        }
        return;
    }

    // ---------------------------------------------------------
    /// Return a memory-owning copy of this COOMatrix.
    ///
    COOMatrix<T, sint_t> deepcopy() const {
        return deepcopy_coo(*this);
    }

    // ---------------------------------------------------------
    /// Return a memory-owning CSRMatrix representation of this COOMatrix.
    ///
    /// If sort != NonzeroSort::CSR, then this function internally creates
    /// a tempoary deep copy of this COOMatrix.
    ///
    CSRMatrix<T, sint_t> as_owning_csr() const {
        CSRMatrix<T, sint_t> csr(n_rows, n_cols);
        coo_to_csr(*this, csr);
        return csr;
    }

    // ---------------------------------------------------------
    /// Return a memory-owning CSCMatrix representation of this COOMatrix.
    ///
    /// If sort != NonzeroSort::CSC, then this function internally creates
    /// a tempoary deep copy of this COOMatrix.
    ///
    CSCMatrix<T, sint_t> as_owning_csc() const {
        CSCMatrix<T, sint_t> csc(n_rows, n_cols);
        coo_to_csc(*this, csc);
        return csc;
    }

    // ---------------------------------------------------------
    /// Return a const view of the transpose of this COOMatrix.
    ///
    const COOMatrix<T, sint_t> transpose() const {
        return transpose_as_coo(*this);
    }

    /////////////////////////////////////////////////////////////////////
    //
    //      Undocumented functions (don't appear in doxygen)
    //
    /////////////////////////////////////////////////////////////////////

    // move constructor
    COOMatrix(COOMatrix<T, sint_t> &&other) 
    : n_rows(other.n_rows), n_cols(other.n_cols), own_memory(other.own_memory), nnz(other.nnz), index_base(other.index_base),
      vals(nullptr), rows(nullptr), cols(nullptr), sort(other.sort) {
        std::swap(rows, other.rows);
        std::swap(cols, other.cols);
        std::swap(vals, other.vals);
        other.nnz = 0;
        other.sort = NonzeroSort::None;
    }
};

#ifdef __cpp_concepts
static_assert(SparseMatrix<COOMatrix<float>>);
static_assert(SparseMatrix<COOMatrix<double>>);
#endif

template <typename COOMatrix>
void print_sparse(COOMatrix const &A) {
    std::cout << "COOMatrix information" << std::endl;
    int64_t nnz = A.nnz;
    std::cout << "\tn_rows = " << A.n_rows << std::endl;
    std::cout << "\tn_cols = " << A.n_cols << std::endl;
    if (A.rows != nullptr) {
        std::cout << "\tvector of row indices\n\t\t";
        for (int64_t i = 0; i < nnz; ++i) {
            std::cout << A.rows[i] << ", ";
        }
    } else {
        std::cout << "\trows is the null pointer.\n\t\t";
    }
    std::cout << std::endl;
    if (A.cols != nullptr) {
        std::cout << "\tvector of column indices\n\t\t";
        for (int64_t i = 0; i < nnz; ++i) {
            std::cout << A.cols[i] << ", ";
        }
    } else {
        std::cout << "\tcols is the null pointer.\n\t\t";
    }
    std::cout << std::endl;
    if (A.vals != nullptr) {
        std::cout << "\tvector of values\n\t\t";
        for (int64_t i = 0; i < nnz; ++i) {
            std::cout << A.vals[i] << ", ";
        }
    } else {
        std::cout << "\tvals is the null pointer.\n\t\t";
    }
    std::cout << std::endl;
    return;
}

/// This function is deprecated as of RandBLAS 1.1; call M.reserve(nnz) instead.
template <typename T, SignedInteger sint_t>
inline void reserve_coo(int64_t nnz, COOMatrix<T,sint_t> &M) {
    M.reserve(nnz);
    return;
}

/// This function is deprecated as of RandBLAS 1.1; call M.sort_arrays(s) instead.
template <typename T>
void sort_coo_data(NonzeroSort s, COOMatrix<T> &spmat) {
    coo_arrays_apply_sort(s, spmat.nnz, spmat.vals, spmat.rows, spmat.cols);
    spmat.sort = s;
    return;
}

} // end namespace RandBLAS::sparse_data


namespace RandBLAS::sparse_data::coo {

using namespace RandBLAS::sparse_data;
using blas::Layout;

template <typename T>
void dense_to_coo(int64_t stride_row, int64_t stride_col, T *mat, T abs_tol, COOMatrix<T> &spmat) {
    int64_t n_rows = spmat.n_rows;
    int64_t n_cols = spmat.n_cols;
    int64_t nnz = nnz_in_dense(n_rows, n_cols, stride_row, stride_col, mat, abs_tol);
    spmat.reserve(nnz);
    nnz = 0;
    #define MAT(_i, _j) mat[(_i) * stride_row + (_j) * stride_col]
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            T val = MAT(i, j);
            if (abs(val) > abs_tol) {
                spmat.vals[nnz] = val;
                spmat.rows[nnz] = i;
                spmat.cols[nnz] = j;
                nnz += 1;
            }
        }
    }
    return;
}

template <typename T>
void dense_to_coo(Layout layout, T* mat, T abs_tol, COOMatrix<T> &spmat) {
    if (layout == Layout::ColMajor) {
        dense_to_coo(1, spmat.n_rows, mat, abs_tol, spmat);
    } else {
        dense_to_coo(spmat.n_cols, 1, mat, abs_tol, spmat);
    }
}

template <typename T>
void coo_to_dense(const COOMatrix<T> &spmat, int64_t stride_row, int64_t stride_col, T *mat) {
    #define MAT(_i, _j) mat[(_i) * stride_row + (_j) * stride_col]
    for (int64_t i = 0; i < spmat.n_rows; ++i) {
        for (int64_t j = 0; j < spmat.n_cols; ++j) {
            MAT(i, j) = 0.0;
        }
    }
    for (int64_t ell = 0; ell < spmat.nnz; ++ell) {
        int64_t i = spmat.rows[ell];
        int64_t j = spmat.cols[ell];
        if (spmat.index_base == IndexBase::One) {
            i -= 1;
            j -= 1;
        }
        MAT(i, j) = spmat.vals[ell];
    }
    return;
}

template <typename T>
void coo_to_dense(const COOMatrix<T> &spmat, Layout layout, T *mat) {
    if (layout == Layout::ColMajor) {
        coo_to_dense(spmat, 1, spmat.n_rows, mat);
    } else {
        coo_to_dense(spmat, spmat.n_cols, 1, mat);
    }
}

} // end namespace RandBLAS::sparse_data::coo

