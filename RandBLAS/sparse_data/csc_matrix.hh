#ifndef randblas_sparse_data_csc
#define randblas_sparse_data_csc
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include <algorithm>

namespace RandBLAS::sparse_data {

// =============================================================================
/// A CSC-format sparse matrix that complies with the SparseMatrix concept.
///
template <typename T, RandBLAS::SignedInteger sint_t = int64_t>
struct CSCMatrix {
    using scalar_t = T;
    using index_t = sint_t; 
    const int64_t n_rows;
    const int64_t n_cols;
    const bool own_memory;
    int64_t nnz = 0;
    IndexBase index_base;
    T *vals = nullptr;

    // ---------------------------------------------------------------------------
    ///  Row index array in the CSC format. 
    ///  
    sint_t *rowidxs = nullptr;
    
    // ---------------------------------------------------------------------------
    ///  Pointer offset array for the CSC format. The number of nonzeros in column j
    ///  is given by colptr[j+1] - colptr[j]. The row index of the k-th nonzero
    ///  in column j is rowidxs[colptr[j] + k].
    ///  
    sint_t *colptr = nullptr;
    bool _can_reserve = true;

    CSCMatrix(
        int64_t n_rows,
        int64_t n_cols,
        IndexBase index_base
    ) : n_rows(n_rows), n_cols(n_cols), own_memory(true), index_base(index_base) { };

    CSCMatrix(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        T *vals,
        sint_t *rowidxs,
        sint_t *colptr,
        IndexBase index_base
    ) : n_rows(n_rows), n_cols(n_cols), own_memory(false), index_base(index_base) {
        this->nnz = nnz;
        this->vals = vals;
        this->rowidxs = rowidxs;
        this->colptr = colptr;
    };

    // Constructs an empty sparse matrix of given dimensions.
    // Data can't stored in this object until a subsequent call to reserve(int64_t nnz).
    // This constructor initializes \math{\ttt{own_memory(true)},} and so
    // all data stored in this object is deleted once its destructor is invoked.
    //
    CSCMatrix(
        int64_t n_rows,
        int64_t n_cols
    ) : CSCMatrix(n_rows, n_cols, IndexBase::Zero) { };

    // ---------------------------------------------------------------------------
    /// @verbatim embed:rst:leading-slashes
    /// Constructs a sparse matrix based on declared dimensions and the data in three buffers
    /// (vals, rowidxs, colptr). 
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
    ///       * Stores values of structural nonzeros as part of the CSC format.
    ///
    ///      rowidxs - [in]
    ///       * Pointer to array of sint_t, of length at least nnz.
    ///
    ///      colptr - [in]
    ///       * Pointer to array of sint_t, of length at least n_cols + 1.
    ///
    /// @endverbatim
    CSCMatrix(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        T *vals,
        sint_t *rowidxs,
        sint_t *colptr
    ) : CSCMatrix(n_rows, n_cols, nnz, vals, rowidxs, colptr, IndexBase::Zero) {};

    ~CSCMatrix() {
        if (this->own_memory) {
            delete [] this->rowidxs;
            delete [] this->colptr;
            delete [] this->vals;
        }
    };


    // @verbatim embed:rst:leading-slashes
    // Attach three buffers to this CSCMatrix, (vals, rowidxs, colptr), of sufficient
    // size for this matrix to hold nnz structural nonzeros.
    // This function can only be called if :math:`\ttt{own_memory == true},`` and
    // it can only be called once.
    //
    // @endverbatim
    void reserve(int64_t nnz) {
        randblas_require(this->_can_reserve);
        randblas_require(this->own_memory);
        this->colptr = new sint_t[this->n_cols + 1]{0};
        this->nnz = nnz;
        if (this->nnz > 0) {
            this->rowidxs = new sint_t[nnz]{0};
            this->vals = new T[nnz]{0.0};
        }
        this->_can_reserve = false;
    };

    CSCMatrix(CSCMatrix<T, sint_t> &&other) 
    : n_rows(other.n_rows), n_cols(other.n_cols), own_memory(other.own_memory), index_base(other.index_base) {
        this->nnz = other.nnz;
        std::swap(this->rowidxs, other.rowidxs);
        std::swap(this->colptr , other.colptr );
        std::swap(this->vals   , other.vals   );
        this->_can_reserve = other._can_reserve;
        other.nnz = 0;
    };
};

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
    spmat.reserve(nnz);
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


}

#endif