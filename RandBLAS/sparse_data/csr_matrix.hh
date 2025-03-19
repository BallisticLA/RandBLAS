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
// ^ only used once, but I don't want the RandBLAS prefix
// in the doxygen.


// =============================================================================
///
///  Let \math{\mtxA} denote a sparse matrix with \math{\ttt{nnz}} structural nonzeros.
///  Its CSR representation consists of declared dimensions, \math{\ttt{n_rows}}
///  and \math{\ttt{n_cols}}, and a triplet of arrays 
///  \math{(\ttt{vals},\ttt{rowptr},\ttt{colidxs}).}
///
///  The \math{\ttt{i}^{\text{th}}} row of \math{\mtxA} has 
///  \math{\ttt{rowptr[i+1] - rowptr[i]}} structural nonzeros.
///  The \math{\ttt{k}^{\text{th}}} structural nonzero in row \math{\ttt{i}} appears in
///  column \math{\ttt{colidxs[rowptr[i] + k]}} and is equal to \math{\ttt{vals[rowptr[i] + k]}.}
/// 
///  This type conforms to the SparseMatrix concept.
template <typename T, SignedInteger sint_t = int64_t>
struct CSRMatrix {

    // ------------------------------------------------------------------------
    /// Real scalar type used for structural nonzeros in this matrix.
    using scalar_t = T;

    // ------------------------------------------------------------------------
    /// Signed integer type used in the rowptr and colidxs array members.
    using index_t = sint_t; 

    // ------------------------------------------------------------------------
    ///  The number of rows in this sparse matrix.
    const int64_t n_rows;

    // ------------------------------------------------------------------------
    ///  The number of columns in this sparse matrix.
    const int64_t n_cols;

    // ------------------------------------------------------------------------
    ///  If true, then RandBLAS has permission to allocate and attach memory to the reference
    ///  members of this matrix (vals, rowptr, colidxs). If true *at destruction time*, then delete []
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
    ///  A flag to indicate whether colidxs is interpreted
    ///  with zero-based or one-based indexing.
    ///
    IndexBase index_base;
    
    // ------------------------------------------------------------------------
    ///  Reference to an array that holds values for the structural nonzeros of this matrix.
    ///
    ///  If non-null, this must have length at least nnz.
    ///  \internal
    ///  **Memory management note.** Because this length requirement is not a function of
    ///  only const variables, calling reserve_csr(nnz, A) on a CSRMatrix "A" will raise an error
    ///  if A.vals is non-null.
    ///  \endinternal
    T *vals;
    
    // ------------------------------------------------------------------------
    /// Reference to a pointer offset array for the CSR format. 
    ///
    ///  If non-null, then must have length at least \math{\ttt{n_rows + 1}}.
    ///
    sint_t *rowptr;
    
    // ------------------------------------------------------------------------
    ///  Reference to a column index array in the CSR format, interpreted in \math{\ttt{index_base}}.
    ///
    ///  If non-null, then must have length at least nnz.
    ///  \internal
    ///  **Memory management note.** Because this length requirement is not a function of
    ///  only const variables, calling reserve_csr(nnz, A) on a CSRMatrix "A" will raise an error
    ///  if A.colidxs is non-null.
    ///  \endinternal
    sint_t *colidxs;

    // ------------------------------------------------------------------------
    ///  **Standard constructor.** Initializes n_rows and n_cols at the provided values.
    ///  The vals, rowptr, and colidxs members are set to null pointers;
    ///  nnz is set to zero, index_base is set to
    ///  Zero, and CSRMatrix::own_memory is set to true.
    ///  
    ///  This constructor is intended for use with reserve_csr(int64_t nnz, CSRMatrix &A).
    ///
    CSRMatrix(
        int64_t n_rows,
        int64_t n_cols
    ) : n_rows(n_rows), n_cols(n_cols), own_memory(true), nnz(0), index_base(IndexBase::Zero),
        vals(nullptr), rowptr(nullptr), colidxs(nullptr) { };

    // ------------------------------------------------------------------------
    /// **Expert constructor.** Arguments passed to this function are used to initialize members of the same names;
    /// CSRMatrix::own_memory is set to false.
    ///
    CSRMatrix(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        T *vals,
        sint_t *rowptr,
        sint_t *colidxs,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), own_memory(false), nnz(nnz), index_base(index_base),
        vals(vals), rowptr(rowptr), colidxs(colidxs) { };

    ~CSRMatrix() {
        if (own_memory) {
            if (rowptr  != nullptr) delete [] rowptr;
            if (colidxs != nullptr) delete [] colidxs;
            if (vals    != nullptr) delete [] vals;
        }
    };

    // move constructor
    CSRMatrix(CSRMatrix<T, sint_t> &&other)
    : n_rows(other.n_rows), n_cols(other.n_cols), own_memory(other.own_memory), nnz(other.nnz), index_base(other.index_base),
      vals(nullptr), rowptr(nullptr), colidxs(nullptr) {
        std::swap(colidxs, other.colidxs);
        std::swap(rowptr , other.rowptr );
        std::swap(vals   , other.vals   );
        other.nnz = 0;
    };

};

#ifdef __cpp_concepts
static_assert(SparseMatrix<CSRMatrix<float>>);
static_assert(SparseMatrix<CSRMatrix<double>>);
#endif

// -----------------------------------------------------
///
/// This function requires that M.own_memory is true, that
/// M.colidxs is null, and that M.vals is null. If any of
/// these conditions are not met then this function will
/// raise an error.
/// 
/// Special logic applies to M.rowptr because its documented length
/// requirement is determined by the const variable M.n_rows.
///
/// - If M.rowptr is non-null then it is left unchanged,
///   and it is presumed to point to an array of length
///   at least M.n_rows + 1.
///
/// - If M.rowptr is null, then it will be redirected to
///   a new array of type sint_t and length (M.n_rows + 1).
///
/// From there, M.nnz is overwritten by nnz, and the reference
/// members M.colidxs and M.vals are redirected to new
/// arrays of length nnz (of types sint_t and T, respectively).
///
template <typename T, SignedInteger sint_t>
void reserve_csr(int64_t nnz, CSRMatrix<T, sint_t> &M) {
    randblas_require(nnz > 0);
    randblas_require(M.own_memory);
    randblas_require(M.colidxs == nullptr);
    randblas_require(M.vals    == nullptr);
    if (M.rowptr == nullptr) 
        M.rowptr = new sint_t[M.n_rows + 1]{0};
    M.nnz = nnz;
    M.colidxs = new sint_t[nnz]{0};
    M.vals    = new T[nnz]{0.0};
   return;
}

} // end namespace RandBLAS::sparse_data

namespace RandBLAS::sparse_data::csr {

using namespace RandBLAS::sparse_data;
using blas::Layout;

template <typename T>
void csr_to_dense(const CSRMatrix<T> &spmat, int64_t stride_row, int64_t stride_col, T *mat) {
    randblas_require(spmat.index_base == IndexBase::Zero);
    auto rowptr  = spmat.rowptr;
    auto colidxs = spmat.colidxs;
    auto vals = spmat.vals;
    #define MAT(_i, _j) mat[(_i) * stride_row + (_j) * stride_col]
    for (int64_t i = 0; i < spmat.n_rows; ++i) {
        for (int64_t j = 0; j < spmat.n_cols; ++j) {
            MAT(i, j) = 0.0;
        }
    }
    for (int64_t i = 0; i < spmat.n_rows; ++i) {
        for (int64_t ell = rowptr[i]; ell < rowptr[i+1]; ++ell) {
            int j = colidxs[ell];
            if (spmat.index_base == IndexBase::One)
                j -= 1;
            MAT(i, j) = vals[ell];
        }
    }
    return;
}

template <typename T>
void csr_to_dense(const CSRMatrix<T> &spmat, Layout layout, T *mat) {
    if (layout == Layout::ColMajor) {
        csr_to_dense(spmat, 1, spmat.n_rows, mat);
    } else {
        csr_to_dense(spmat, spmat.n_cols, 1, mat);
    }
    return;
}

template <typename T>
void dense_to_csr(int64_t stride_row, int64_t stride_col, T *mat, T abs_tol, CSRMatrix<T> &spmat) {
    int64_t n_rows = spmat.n_rows;
    int64_t n_cols = spmat.n_cols;
    #define MAT(_i, _j) mat[(_i) * stride_row + (_j) * stride_col]
    // Step 1: count the number of entries with absolute value at least abstol
    int64_t nnz = nnz_in_dense(n_rows, n_cols, stride_row, stride_col, mat, abs_tol);
    // Step 2: allocate memory needed by the sparse matrix
    reserve_csr(nnz, spmat);
    // Step 3: traverse the dense matrix again, populating the sparse matrix as we go
    nnz = 0;
    spmat.rowptr[0] = 0;
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            T val = MAT(i, j);
            if (abs(val) > abs_tol) {
                spmat.vals[nnz] = val;
                spmat.colidxs[nnz] = j;
                nnz += 1;
            }
        }
        spmat.rowptr[i+1] = nnz;
    }
    return;
}

template <typename T>
void dense_to_csr(Layout layout, T* mat, T abs_tol, CSRMatrix<T> &spmat) {
    if (layout == Layout::ColMajor) {
        dense_to_csr(1, spmat.n_rows, mat, abs_tol, spmat);
    } else {
        dense_to_csr(spmat.n_cols, 1, mat, abs_tol, spmat);
    }
    return;
}


} // end namespace RandBLAS::sparse_data::csr
