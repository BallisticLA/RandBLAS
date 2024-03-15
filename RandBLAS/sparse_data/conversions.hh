#ifndef randblas_sparse_data_conversions
#define randblas_sparse_data_conversions
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo_matrix.hh"
#include "RandBLAS/sparse_data/csr_matrix.hh"
#include "RandBLAS/sparse_data/csc_matrix.hh"

namespace RandBLAS::sparse_data::conversions {

using namespace RandBLAS::sparse_data;
using RandBLAS::SignedInteger;

template <typename T, SignedInteger sint_t1 = int64_t, SignedInteger sint_t2 = int64_t>
void coo_to_csc(COOMatrix<T, sint_t1> &coo, CSCMatrix<T, sint_t2> &csc) {
    randblas_require(csc.n_rows == coo.n_rows);
    randblas_require(csc.n_cols == csc.n_cols);
    randblas_require(csc.index_base == IndexBase::Zero);
    randblas_require(coo.index_base == IndexBase::Zero);
    sort_coo_data(NonzeroSort::CSC, coo);
    csc.reserve(coo.nnz);
    csc.colptr[0] = 0;
    int64_t ell = 0;
    for (int64_t j = 0; j < coo.n_cols; ++j) {
        while (ell < coo.nnz && coo.cols[ell] == j) {
            csc.rowidxs[ell] = (sint_t2) coo.rows[ell];
            csc.vals[ell] = coo.vals[ell];
            ++ell;
        }
        csc.colptr[j+1] = (sint_t2) ell;
    }
    return;
}

template <typename T, SignedInteger sint_t1 = int64_t, SignedInteger sint_t2 = int64_t>
void csc_to_coo(CSCMatrix<T, sint_t1> &csc, COOMatrix<T, sint_t2> &coo) {
    randblas_require(csc.n_rows == coo.n_rows);
    randblas_require(csc.n_cols == coo.n_cols);
    randblas_require(csc.index_base == IndexBase::Zero);
    randblas_require(coo.index_base == IndexBase::Zero);
    coo.reserve(csc.nnz);
    int64_t ell = 0;
    for (int64_t j = 0; j < csc.n_cols; ++j) {
        for (int64_t i = csc.colptr[j]; i < csc.colptr[j+1]; ++i) {
            coo.vals[ell] = csc.vals[ell];
            coo.rows[ell] = (sint_t2) i;
            coo.cols[ell] = (sint_t2) j;
            ++ell;
        }
    }
    coo.sort = NonzeroSort::CSC;
    return;
}

template <typename T, SignedInteger sint_t1 = int64_t, SignedInteger sint_t2 = int64_t>
void coo_to_csr(COOMatrix<T, sint_t1> &coo, CSRMatrix<T, sint_t2> &csr) {
    randblas_require(csr.n_rows == coo.n_rows);
    randblas_require(csr.n_cols == coo.n_cols);
    randblas_require(csr.index_base == IndexBase::Zero);
    randblas_require(coo.index_base == IndexBase::Zero);
    sort_coo_data(NonzeroSort::CSR, coo);
    csr.reserve(coo.nnz);
    csr.rowptr[0] = (sint_t2) 0;
    int64_t ell = 0;
    for (int64_t i = 0; i < coo.n_rows; ++i) {
        while (ell < coo.nnz && coo.rows[ell] == i) {
            csr.colidxs[ell] = (sint_t2) coo.cols[ell];
            csr.vals[ell] = coo.vals[ell];
            ++ell;
        }
        csr.rowptr[i+1] = (sint_t2) ell;
    }
    return;
}

template <typename T, SignedInteger sint_t1 = int64_t, SignedInteger sint_t2 = int64_t>
void csr_to_coo(CSRMatrix<T, sint_t1> &csr, COOMatrix<T, sint_t2> &coo) {
    randblas_require(csr.n_rows == coo.n_rows);
    randblas_require(csr.n_cols == coo.n_cols);
    randblas_require(csr.index_base == IndexBase::Zero);
    randblas_require(coo.index_base == IndexBase::Zero);
    coo.reserve(csr.nnz);
    int64_t ell = 0;
    for (int64_t i = 0; i < csr.n_rows; ++i) {
        for (int64_t j = csr.rowptr[i]; j < csr.rowptr[i+1]; ++j) {
            coo.vals[ell] = csr.vals[ell];
            coo.rows[ell] = (sint_t2) i;
            coo.cols[ell] = (sint_t2) j;
            ++ell;
        }
    }
    coo.sort = NonzeroSort::CSR;
    return;
}

template <typename T>
CSRMatrix<T> transpose_as_csr(CSCMatrix<T> &A, bool share_memory = true) {
    if (share_memory) {
        CSRMatrix<T> At(A.n_cols, A.n_rows, A.nnz, A.vals, A.colptr, A.rowidxs, A.index_base);
        return At;
    } else {
        CSRMatrix<T> At(A.n_cols, A.n_rows, A.index_base);
        At.reserve(A.nnz);
        for (int64_t i = 0; i < A.nnz; ++i) {
            At.colidxs[i] = A.rowidxs[i];
            At.vals[i] = A.vals[i];
        }
        for (int64_t i = 0; i < A.n_cols + 1; ++i)
            At.rowptr[i] = A.colptr[i];
        return At;
    }
}

template <typename T>
CSCMatrix<T> transpose_as_csc(CSRMatrix<T> &A, bool share_memory = true) {
    if (share_memory) {
        CSCMatrix<T> At(A.n_cols, A.n_rows, A.nnz, A.vals, A.colidxs, A.rowptr, A.index_base);
        return At;
    } else {
        CSCMatrix<T> At(A.n_cols, A.n_rows, A.index_base);
        At.reserve(A.nnz);
        for (int64_t i = 0; i < A.nnz; ++i) {
            At.rowidxs[i] = A.colidxs[i];
            At.vals[i] = A.vals[i];
        }
        for (int64_t i = 0; i < A.n_rows + 1; ++i)
            At.colptr[i] = A.rowptr[i];
        return At;
    }
}

template <typename T>
void reindex_inplace(CSCMatrix<T> &A, IndexBase desired) {
    if (A.index_base == desired)
        return;
    if (A.index_base == IndexBase::One) {
        for (int64_t ell = 0; ell < A.nnz; ++ell)
            A.rowidxs[ell] -= 1;
    } else {
        for (int64_t ell = 0; ell < A.nnz; ++ell)
            A.rowidxs[ell] += 1;
    }
    A.index_base = desired;
    return;
}

template <typename T>
void reindex_inplace(CSRMatrix<T> &A, IndexBase desired) {
    if (A.index_base == desired)
        return;
    if (A.index_base == IndexBase::One) {
        for (int64_t ell = 0; ell < A.nnz; ++ell)
            A.colidxs[ell] -= 1;
    } else {
        for (int64_t ell = 0; ell < A.nnz; ++ell)
            A.colidxs[ell] += 1;  
    }
    A.index_base = desired;
    return;
}

template <typename T>
void reindex_inplace(COOMatrix<T> &A, IndexBase desired) {
    if (A.index_base == desired)
        return;
    if (A.index_base == IndexBase::One) {
        for (int64_t ell = 0; ell < A.nnz; ++ell) {
            A.rows[ell] -= 1;
            A.cols[ell] -= 1;
        }
    } else {
        for (int64_t ell = 0; ell < A.nnz; ++ell) {
            A.rows[ell] += 1;
            A.cols[ell] += 1;
        }
    }
    A.index_base = desired;
    return;
}

} // end namespace RandBLAS::sparse_data::conversions

#endif
