#ifndef randblas_sparse_data_csr
#define randblas_sparse_data_csr
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"

namespace RandBLAS::sparse_data {

template <typename T>
struct CSRMatrix {
    const int64_t n_rows;
    const int64_t n_cols;
    const IndexBase index_base;
    const bool own_memory;
    int64_t nnz;
    T *vals;
    int64_t *rowptr;
    int64_t *colidxs;
    bool _can_reserve = true;

    CSRMatrix(
        int64_t n_rows,
        int64_t n_cols,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), index_base(index_base), own_memory(true) {
        this->nnz = 0;
        this->vals = nullptr;
        this->rowptr = nullptr;
        this->colidxs = nullptr;
    };

    CSRMatrix(
        int64_t n_rows,
        int64_t n_cols,
        int64_t nnz,
        T *vals,
        int64_t *rowptr,
        int64_t *colidxs,
        IndexBase index_base = IndexBase::Zero
    ) : n_rows(n_rows), n_cols(n_cols), index_base(index_base), own_memory(false) {
        this->nnz = nnz;
        this->vals = vals;
        this->rowptr = rowptr;
        this->colidxs = colidxs;
    };

    ~CSRMatrix() {
        if (this->own_memory) {
            delete [] this->rowptr;
            delete [] this->colidxs;
            delete [] this->vals;
        }
    };

    void reserve(int64_t nnz) {
        randblas_require(this->_can_reserve);
        randblas_require(this->own_memory);
        this->nnz = nnz;
        this->rowptr = new int64_t[this->n_rows + 1]{0};
        this->colidxs = new int64_t[nnz]{0};
        this->vals = new T[nnz]{0.0};
        this->_can_reserve = false;
    };

};

} // end namespace RandBLAS::sparse_data

namespace RandBLAS::sparse_data::csr {

using namespace RandBLAS::sparse_data;
using blas::Layout;

template <typename T>
void csr_to_dense(const CSRMatrix<T> &spmat, int64_t stride_row, int64_t stride_col, T *mat) {
    auto rowptr = spmat.rowptr;
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
    spmat.reserve(nnz);
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
#endif
