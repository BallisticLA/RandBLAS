#ifndef randblas_sparse_data_csc
#define randblas_sparse_data_csc
#include "RandBLAS/base.hh"
#include "RandBLAS/sparse_data/base.hh"


namespace RandBLAS::sparse_data {

template <typename T, RandBLAS::SignedInteger sint_t = int64_t>
struct CSCMatrix {
    const int64_t n_rows;
    const int64_t n_cols;
    const IndexBase index_base;
    const bool own_memory;
    int64_t nnz;
    T *vals;
    sint_t *rowidxs;
    sint_t *colptr;
    bool _can_reserve = true;

    CSCMatrix(
        int64_t n_rows,
        int64_t n_cols,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), index_base(index_base), own_memory(true) {
        this->nnz = 0;
        this->vals = nullptr;
        this->rowidxs = nullptr;
        this->colptr = nullptr;
    };

    CSCMatrix(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        T *vals,
        sint_t *rowidxs,
        sint_t *colptr,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), index_base(index_base), own_memory(false) {
        this->nnz = nnz;
        this->vals = vals;
        this->rowptr = rowidxs;
        this->colidxs = colptr;
    };

    ~CSCMatrix() {
        if (this->own_memory) {
            delete [] this->rowidxs;
            delete [] this->colptr;
            delete [] this->vals;
        }
    };

    void reserve(int64_t nnz) {
        randblas_require(this->_can_reserve);
        randblas_require(this->own_memory);
        this->nnz = nnz;
        this->rowidxs = new sint_t[nnz]{0};
        this->colptr = new sint_t[this->n_cols + 1]{0};
        this->vals = new T[nnz]{0.0};
        this->_can_reserve = false;
    };

};

} // end namespace RandBLAS::sparse_data

namespace RandBLAS::sparse_data::csc {

using namespace RandBLAS::sparse_data;
using blas::Layout;

template <typename T>
void csc_to_dense(const CSCMatrix<T> &spmat, int64_t stride_row, int64_t stride_col, T *mat) {
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