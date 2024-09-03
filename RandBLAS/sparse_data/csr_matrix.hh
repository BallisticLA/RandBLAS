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
/// A CSR-format sparse matrix that complies with the SparseMatrix concept.
///
template <typename T, SignedInteger sint_t = int64_t>
struct CSRMatrix {
    using scalar_t = T;
    using index_t = sint_t; 
    const int64_t n_rows;
    const int64_t n_cols;
    bool own_memory;
    int64_t nnz = 0;
    IndexBase index_base;
    T *vals;
    
    // ---------------------------------------------------------------------------
    ///  Pointer offset array for the CSR format. The number of nonzeros in row i
    ///  is given by rowptr[i+1] - rowptr[i]. The column index of the k-th nonzero
    ///  in row i is colidxs[rowptr[i] + k].
    ///  
    sint_t *rowptr;
    
    // ---------------------------------------------------------------------------
    ///  Column index array in the CSR format. 
    ///  
    sint_t *colidxs;

    // Constructs an empty sparse matrix of given dimensions.
    // Data can't stored in this object until a subsequent call to reserve(int64_t nnz).
    // This constructor initializes \math{\ttt{own_memory(true)},} and so
    // all data stored in this object is deleted once its destructor is invoked.
    //
    CSRMatrix(
        int64_t n_rows,
        int64_t n_cols
    ) : n_rows(n_rows), n_cols(n_cols), own_memory(true), nnz(0), index_base(IndexBase::Zero),
        vals(nullptr), rowptr(nullptr), colidxs(nullptr) { };

    // ---------------------------------------------------------------------------
    /// @verbatim embed:rst:leading-slashes
    /// Constructs a sparse matrix based on declared dimensions and the data in three buffers
    /// (vals, rowptr, colidxs). 
    /// This constructor initializes :math:`\ttt{own_memory(false)}`, and
    /// so the provided buffers are unaffected when this object's destructor
    /// is invoked.
    ///
    /// .. dropdown:: Full parameter descriptions
    ///     :animate: fade-in-slide-down
    ///
    ///      n_rows - [in]
    ///       * The number of rows in this sparse matrix.
    ///
    ///      n_cols - [in]
    ///       * The number of columns in this sparse matrix.
    ///
    ///      nnz - [in]
    ///       * The number of structural nonzeros in the matrix.
    ///
    ///      vals - [in]
    ///       * Pointer to array of real numerical type T, of length at least nnz.
    ///       * Stores values of structural nonzeros as part of the CSR format.
    ///
    ///      rowptr - [in]
    ///       * Pointer to array of sint_t, of length at least n_rows + 1.
    ///
    ///      colidxs - [in]
    ///       * Pointer to array of sint_t, of length at least nnz.
    ///
    ///      index_base - [in]
    ///       * IndexBase::Zero or IndexBase::One
    ///
    /// @endverbatim
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


template <typename T, SignedInteger sint_t>
void reserve(int64_t nnz, CSRMatrix<T,sint_t> &M) {
    randblas_require(M.own_memory);
    if (M.rowptr == nullptr) 
        M.rowptr = new sint_t[M.n_rows + 1]{0};
    M.nnz = nnz;
    if (nnz > 0) {
        randblas_require(M.colidxs == nullptr);
        randblas_require(M.vals    == nullptr);
        M.colidxs = new sint_t[nnz]{0};
        M.vals    = new T[nnz]{0.0};
    }
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
    reserve(nnz, spmat);
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
