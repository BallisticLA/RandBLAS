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
#include <algorithm>

namespace RandBLAS::sparse_data {

using RandBLAS::SignedInteger;

// =============================================================================
///
///  Let \math{\mtxA} denote a sparse matrix with \math{\ttt{nnz}} structural nonzeros.
///  Its CSC representation consists of declared dimensions, \math{\ttt{n_rows}}
///  and \math{\ttt{n_cols}}, and a triplet of arrays 
///  \math{(\ttt{vals},\ttt{rowidxs},\ttt{colptr}).}
///
///  The \math{\ttt{j}^{\text{th}}} column of \math{\mtxA} has 
///  \math{\ttt{colptr[j+1] - colptr[j]}} structural nonzeros.
///  The \math{\ttt{k}^{\text{th}}} structural nonzero in column \math{\ttt{j}} appears in
///  row \math{\ttt{rowidxs[colptr[j] + k]}} and is equal to \math{\ttt{vals[colptr[j] + k]}.}
/// 
///  This type conforms to the SparseMatrix concept.
template <typename T, SignedInteger sint_t = int64_t>
struct CSCMatrix {
    // ------------------------------------------------------------------------
    /// Real scalar type used for structural nonzeros in this matrix.
    using scalar_t = T;

    // ------------------------------------------------------------------------
    /// Signed integer type used in the rowidxs and colptr array members.
    using index_t = sint_t; 

    // ------------------------------------------------------------------------
    ///  The number of rows in this sparse matrix.
    const int64_t n_rows;

    // ------------------------------------------------------------------------
    ///  The number of columns in this sparse matrix.
    const int64_t n_cols;

    // ------------------------------------------------------------------------
    ///  If true, then RandBLAS has permission to allocate and attach memory to the reference
    ///  members of this matrix (vals, rowidxs, colptr). If true *at destruction time*, then delete []
    ///  will be called on each non-null reference member of this matrix.
    ///
    ///  RandBLAS only writes to this member at construction time.
    ///
    bool own_memory;
    
    // ------------------------------------------------------------------------
    ///  The number of structral nonzeros in this matrix.
    ///
    int64_t nnz;
    
    // ------------------------------------------------------------------------
    ///  A flag to indicate whether rowidxs is interpreted
    ///  with zero-based or one-based indexing.
    ///
    IndexBase index_base;
    
    // ------------------------------------------------------------------------
    ///  Reference to an array that holds values for the structural nonzeros of this matrix.
    ///
    ///  If non-null, this must have length at least nnz.
    ///  \internal
    ///  **Memory management note.** Because this length requirement is not a function of
    ///  only const variables, calling reserve_csc(nnz, A) on a CSCMatrix "A" will raise an error
    ///  if A.vals is non-null.
    ///  \endinternal
    T *vals;

    // ------------------------------------------------------------------------
    ///  Reference to a row index array in the CSC format, interpreted in \math{\ttt{index_base}}.
    ///
    ///  If non-null, then must have length at least nnz.
    ///  \internal
    ///  **Memory management note.** Because this length requirement is not a function of
    ///  only const variables, calling reserve_cs(nnz, A) on a CSCMatrix "A" will raise an error
    ///  if A.rowidxs is non-null.
    ///  \endinternal
    sint_t *rowidxs;
    
    // ------------------------------------------------------------------------
    /// Reference to a pointer offset array for the CSC format. 
    ///
    ///  If non-null, then must have length at least \math{\ttt{n_cols + 1}}.
    ///
    sint_t *colptr;

    // ---------------------------------------------------------------------------
    ///  **Standard constructor.** Initializes n_rows and n_cols at the provided values.
    ///  The vals, rowidxs, and colptr members are null-initialized;
    ///  \math{\ttt{nnz}} is set to zero, \math{\ttt{index_base}} is set to
    ///  Zero, and CSCMatrix::own_memory is set to true.
    ///  
    ///  This constructor is intended for use with reserve_csc(int64_t nnz, CSCMatrix &A).
    ///
    CSCMatrix( 
        int64_t n_rows,
        int64_t n_cols
    ) : n_rows(n_rows), n_cols(n_cols), own_memory(true), nnz(0), index_base(IndexBase::Zero),
        vals(nullptr), rowidxs(nullptr), colptr(nullptr) { };

    // ------------------------------------------------------------------------
    /// **Expert constructor.** Arguments passed to this function are used to initialize members of the same names;
    /// CSCMatrix::own_memory is set to false.
    ///
    CSCMatrix(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        T *vals,
        sint_t *rowidxs,
        sint_t *colptr,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), own_memory(false), nnz(nnz), index_base(index_base),
        vals(vals), rowidxs(rowidxs), colptr(colptr) { };

    ~CSCMatrix() {
        if (own_memory) {
            if (rowidxs != nullptr) delete [] rowidxs;
            if (colptr  != nullptr) delete [] colptr;
            if (vals    != nullptr) delete [] vals;
        }
    };

    CSCMatrix(CSCMatrix<T, sint_t> &&other) 
    : n_rows(other.n_rows), n_cols(other.n_cols), own_memory(other.own_memory), nnz(other.nnz), index_base(other.index_base),
      vals(nullptr), rowidxs(nullptr), colptr(nullptr) {
        std::swap(rowidxs, other.rowidxs);
        std::swap(colptr , other.colptr );
        std::swap(vals   , other.vals   );
        other.nnz = 0;
    };
};

#ifdef __cpp_concepts
static_assert(SparseMatrix<CSCMatrix<float>>);
static_assert(SparseMatrix<CSCMatrix<double>>);
#endif

// -----------------------------------------------------
///
/// This function requires that M.own_memory is true, that
/// M.rowidxs is null, and that M.vals is null. If any of
/// these conditions are not met then this function will
/// raise an error.
/// 
/// Special logic applies to M.colptr because its documented length
/// requirement is determined by the const variable M.n_cols.
///
/// - If M.colptr is non-null then it is left unchanged,
///   and it is presumed to point to an array of length
///   at least M.n_cols + 1.
///
/// - If M.colptr is null, then it will be redirected to
///   a new array of type sint_t and length (M.n_cols + 1).
///
/// From there, M.nnz is overwritten by nnz, and the reference
/// members M.rowidxs and M.vals are redirected to new
/// arrays of length nnz (of types sint_t and T, respectively).
///
template <typename T, SignedInteger sint_t>
void reserve_csc(int64_t nnz, CSCMatrix<T,sint_t> &M) {
    randblas_require(nnz > 0);
    randblas_require(M.own_memory);
    randblas_require(M.rowidxs == nullptr);
    randblas_require(M.vals    == nullptr);
    if (M.colptr == nullptr)
        M.colptr = new sint_t[M.n_cols + 1]{0};
    M.nnz = nnz;
    M.rowidxs = new sint_t[nnz]{0};
    M.vals    = new T[nnz]{0.0};
    return;
}

} // end namespace RandBLAS::sparse_data

namespace RandBLAS::sparse_data::csc {

using namespace RandBLAS::sparse_data;
using blas::Layout;

template <typename T>
void csc_to_dense(const CSCMatrix<T> &spmat, int64_t stride_row, int64_t stride_col, T *mat) {
    randblas_require(spmat.index_base == IndexBase::Zero);
    #define MAT(_i, _j) mat[(_i) * stride_row + (_j) * stride_col]
    for (int64_t i = 0; i < spmat.n_rows; ++i) {
        for (int64_t j = 0; j < spmat.n_cols; ++j) {
            MAT(i, j) = 0.0;
        }
    }
    for (int64_t j = 0; j < spmat.n_cols; ++j) {
        for (int64_t ell = spmat.colptr[j]; ell < spmat.colptr[j+1]; ++ell) {
            int64_t i = spmat.rowidxs[ell];
            if (spmat.index_base == IndexBase::One)
                i -= 1;
            MAT(i, j) = spmat.vals[ell];
        }
    }
    return;
}

template <typename T>
void csc_to_dense(const CSCMatrix<T> &spmat, Layout layout, T *mat) {
    if (layout == Layout::ColMajor) {
        csc_to_dense(spmat, 1, spmat.n_rows, mat);
    } else {
        csc_to_dense(spmat, spmat.n_cols, 1, mat);
    }
    return;
}

template <typename T>
void dense_to_csc(int64_t stride_row, int64_t stride_col, T *mat, T abs_tol, CSCMatrix<T> &spmat) {
    int64_t n_rows = spmat.n_rows;
    int64_t n_cols = spmat.n_cols;
    #define MAT(_i, _j) mat[(_i) * stride_row + (_j) * stride_col]
    // Step 1: count the number of entries with absolute value at least abstol
    int64_t nnz = nnz_in_dense(n_rows, n_cols, stride_row, stride_col, mat, abs_tol);
    // Step 2: allocate memory needed by the sparse matrix
    reserve_csc(nnz, spmat);
    // Step 3: traverse the dense matrix again, populating the sparse matrix as we go
    nnz = 0;
    spmat.colptr[0] = 0;
    for (int64_t j = 0; j < n_cols; ++j) {
        for (int64_t i = 0; i < n_rows; ++i) {
            T val = MAT(i, j);
            if (abs(val) > abs_tol) {
                spmat.vals[nnz] = val;
                spmat.rowidxs[nnz] = i;
                nnz += 1;
            }
        }
        spmat.colptr[j+1] = nnz;
    }
    return;
}

template <typename T>
void dense_to_csc(Layout layout, T* mat, T abs_tol, CSCMatrix<T> &spmat) {
    if (layout == Layout::ColMajor) {
        dense_to_csc(1, spmat.n_rows, mat, abs_tol, spmat);
    } else {
        dense_to_csc(spmat.n_cols, 1, mat, abs_tol, spmat);
    }
    return;
}

} // end namespace RandBLAS::sparse_data::csc
