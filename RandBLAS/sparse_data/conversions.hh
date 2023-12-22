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


} // end namespace 

#endif
