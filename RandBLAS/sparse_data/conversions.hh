#ifndef randblas_sparse_data_conversions
#define randblas_sparse_data_conversions
#include "RandBLAS/base.hh"
#include "RandBLAS/exceptions.hh"
#include "RandBLAS/sparse_data/base.hh"
#include "RandBLAS/sparse_data/coo.hh"
#include "RandBLAS/sparse_data/csr.hh"
#include "RandBLAS/sparse_data/csc.hh"

namespace RandBLAS::sparse_data::conversions {

using namespace RandBLAS::sparse_data;

template <typename T>
void coo_to_csc(COOMatrix<T> &coo, CSCMatrix<T> &csc) {
    randblas_require(csc.n_rows == coo.n_rows);
    randblas_require(csc.n_cols == csc.n_cols);
    sort_coo_data(NonzeroSort::CSC, coo);
    csc.reserve(coo.nnz);
    csc.colptr[0] = 0;
    int64_t ell = 0;
    for (int64_t j = 0; j < coo.n_cols; ++j) {
        while (ell < coo.nnz && coo.cols[ell] == j) {
            csc.rowidxs[ell] = coo.rows[ell];
            csc.vals[ell] = coo.vals[ell];
            ++ell;
        }
        csc.colptr[j+1] = ell;
    }
    return;
}

template <typename T>
void csc_to_coo(CSCMatrix<T> &csc, COOMatrix<T> &coo) {
    randblas_require(csc.n_rows == coo.n_rows);
    randblas_require(csc.n_cols == coo.n_cols);
    coo.reserve(csc.nnz);
    int64_t ell = 0;
    for (int64_t j = 0; j < csc.n_cols; ++j) {
        for (int64_t i = csc.colptr[j]; i < csc.colptr[j+1]; ++i) {
            coo.vals[ell] = csc.vals[ell];
            coo.rows[ell] = i;
            coo.cols[ell] = j;
            ++ell;
        }
    }
    coo.sort = NonzeroSort::CSR;
    return;
}

template <typename T>
void coo_to_csr(COOMatrix<T> &coo, CSRMatrix<T> &csr) {
    sort_coo_data(NonzeroSort::CSR, coo);
    csr.reserve(coo.nnz);
    csr.rowptr[0] = 0;
    int64_t ell = 0;
    for (int64_t i = 0; i < coo.n_rows; ++i) {
        while (ell < coo.nnz && coo.rows[ell] == i) {
            csr.colidxs[ell] = coo.cols[ell];
            csr.vals[ell] = coo.vals[ell];
            ++ell;
        }
        csr.rowptr[i+1] = ell;
    }
    return;
}

template <typename T>
void csr_to_coo(CSRMatrix<T> &csr, COOMatrix<T> &coo) {
    randblas_require(csr.n_rows == coo.n_rows);
    randblas_require(csr.n_cols == coo.n_cols);
    coo.reserve(csr.nnz);
    int64_t ell = 0;
    for (int64_t i = 0; i < csr.n_rows; ++i) {
        for (int64_t j = csr.rowptr[i]; j < csr.rowptr[i+1]; ++j) {
            coo.vals[ell] = csr.vals[ell];
            coo.rows[ell] = i;
            coo.cols[ell] = j;
            ++ell;
        }
    }
    coo.sort = NonzeroSort::CSR;
    return;
}


} // end namespace 

#endif
